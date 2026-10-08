"""Phase 8 — the typed client. Pydantic in, calibrated decisions out.

The integer encoding, the pre-mask renormalization and the temperature table are
all internal: callers see enum values and probabilities.

    class Decision(BaseModel):
        route: Literal["billing", "technical", "account", "sales", "abuse"]
        severity: Literal["P0", "P1", "P2", "P3"]
        needs_human: bool

    jev = Jev()
    r = jev.decide(Decision, "card declined on renewal, third time")
    r.value        -> Decision(route="billing", severity="P2", needs_human=True)
    r.confidence   -> {"route": 0.94, "severity": 0.71, "needs_human": 0.88}
    r.latency_ms   -> 31.4

Field names are mapped to tasks by matching their declared `Literal` / `bool`
options against `spec/schemas/*.json`, so a Pydantic model is a *view* over the
schemas rather than a second source of truth that can drift from them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Type, get_args, get_origin

from jev.ollama import Decision as RawDecision
from jev.ollama import JevOllama
from jev.spec import TaskSpec, load_specs


@dataclass
class Result:
    value: Any
    """The populated Pydantic model. `None` for any field that abstained."""
    confidence: dict[str, float]
    distribution: dict[str, dict[str, float]]
    latency_ms: float
    trustworthy: bool
    """False if any field's legal_mass < 0.95 or its decision token was not at 0."""
    raw: dict[str, RawDecision] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    """Fields whose confidence fell below the threshold."""

    @property
    def min_confidence(self) -> float:
        return min(self.confidence.values()) if self.confidence else 0.0


class SchemaMismatch(ValueError):
    pass


def _field_options(annotation) -> tuple[str, ...] | None:
    """Extract the closed option set from a type annotation."""
    if annotation is bool:
        return ("false", "true")
    origin = get_origin(annotation)
    if origin is Literal:
        args = get_args(annotation)
        if all(isinstance(a, (str, int, bool)) for a in args):
            return tuple(str(a).lower() if isinstance(a, bool) else str(a)
                         for a in args)
    # Optional[X] / X | None -> unwrap, abstain is modelled separately
    if origin is not None:
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            return _field_options(inner[0])
    return None


def map_model_to_tasks(model_cls: Type) -> dict[str, TaskSpec]:
    """Match each Pydantic field to the task whose enum it declares.

    Raises rather than guessing: a field that matches no schema is almost always a
    typo or a stale copy of the enum, and silently dropping it would produce a
    model populated with defaults that look like real decisions.
    """
    specs = load_specs()
    by_options: dict[tuple[str, ...], list[TaskSpec]] = {}
    for s in specs.values():
        by_options.setdefault(tuple(sorted(v.lower() for v in s.values)), []).append(s)

    out: dict[str, TaskSpec] = {}
    fields = getattr(model_cls, "model_fields", None)
    if fields is None:
        raise SchemaMismatch(f"{model_cls!r} is not a Pydantic v2 model")
    for name, info in fields.items():
        # 1. exact task id
        if name in specs:
            out[name] = specs[name]
            continue
        # 2. option-set match
        opts = _field_options(info.annotation)
        if opts is not None:
            # `by_options` is keyed case-insensitively, so normalize the lookup too.
            cands = by_options.get(tuple(sorted(o.lower() for o in opts)))
            if cands and len(cands) == 1:
                out[name] = cands[0]
                continue
            if cands:
                raise SchemaMismatch(
                    f"field {name!r} options {opts} match several tasks "
                    f"({[c.id for c in cands]}); rename the field to the task id")
        raise SchemaMismatch(
            f"field {name!r} (annotation {info.annotation!r}) matches no task in "
            f"spec/schemas. Known tasks: {sorted(specs)}. Either name the field "
            f"after a task id or declare its exact Literal option set.")
    if not out:
        raise SchemaMismatch(f"{model_cls.__name__} declares no mappable fields")
    return out


def _coerce(spec: TaskSpec, value: str | None, annotation) -> Any:
    if value is None:
        return None
    if annotation is bool or _field_options(annotation) == ("false", "true"):
        return value == "true"
    return value


class Jev:
    def __init__(self, model: str = "jev", host: str = "http://localhost:11434",
                 keep_alive: str = "30m", top_logprobs: int = 20):
        self._cli = JevOllama(model=model, host=host, keep_alive=keep_alive,
                              top_logprobs=top_logprobs)

    def close(self):
        self._cli.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def health(self) -> dict:
        return self._cli.health()

    def decide(self, model_cls: Type, text: str, *,
               threshold: float | None = None) -> Result:
        """One decision per declared field.

        Issued as k single-field requests sharing a cached prompt prefix. Spike 0.1
        measured the one-shot alternative at 9x the decode cost with corrupted
        probabilities, so this is faster *and* more trustworthy than asking for a
        JSON object.
        """
        mapping = map_model_to_tasks(model_cls)
        t0 = time.perf_counter()
        raw, conf, dist, warns, deferred, values = {}, {}, {}, [], [], {}
        trustworthy = True
        annotations = {n: i.annotation
                       for n, i in model_cls.model_fields.items()}

        for name, spec in mapping.items():
            d = self._cli.decide(spec.id, text)
            raw[name] = d
            conf[name] = d.confidence
            dist[name] = d.distribution
            warns.extend(f"{name}: {w}" for w in d.warnings)
            trustworthy &= d.trustworthy

            val = d.value
            if threshold is not None and d.confidence < threshold:
                deferred.append(name)
                if spec.abstain_enabled:
                    val = None
                else:
                    # Safety gates must fail CLOSED, never defer to None.
                    # gate_human_v1: unsure means escalate to a human.
                    val = spec.values[-1]
                    warns.append(
                        f"{name}: confidence {d.confidence:.3f} < {threshold} and "
                        f"abstain is disabled for {spec.id}; failed closed to "
                        f"{val!r} rather than deferring")
            values[name] = _coerce(spec, val, annotations[name])

        try:
            obj = model_cls(**values)
        except Exception:
            # A field that abstained may violate a non-Optional annotation. Report
            # the dict rather than crashing, and say so.
            obj = values
            warns.append(
                "could not construct the Pydantic model (a field abstained or was "
                "deferred); `value` is a plain dict. Declare those fields Optional "
                "to get a typed object back.")

        return Result(value=obj, confidence=conf, distribution=dist,
                      latency_ms=(time.perf_counter() - t0) * 1000,
                      trustworthy=trustworthy, raw=raw, warnings=warns,
                      deferred=deferred)

    def decide_or_defer(self, model_cls: Type, text: str,
                        threshold: float = 0.9) -> Result:
        """The primary intended usage pattern.

        A calibrated model's value is knowing when not to be trusted: act
        automatically above the threshold, route the rest to a human or to a larger
        model. Fields with abstain disabled fail closed instead of deferring.
        """
        return self.decide(model_cls, text, threshold=threshold)

    def decide_one(self, task_id: str, text: str) -> RawDecision:
        """Single task by id, with the full raw diagnostics."""
        return self._cli.decide(task_id, text)

    def decide_batch(self, model_cls: Type, texts: list[str], *,
                     threshold: float | None = None) -> list[Result]:
        """Sequential over inputs, reusing one HTTP connection.

        Not parallel on purpose: this GPU holds one model and Ollama serializes
        requests anyway, so concurrency would add contention, not throughput.
        """
        return [self.decide(model_cls, t, threshold=threshold) for t in texts]

    def unload(self):
        """Free the GPU — required before training (HARDWARE.md revision 4)."""
        self._cli.unload()
