"""Load the task specs and render prompts. The single source of truth at runtime.

`spec/schemas/*.json` are authored by hand; everything else in the project derives
from them through this module — training targets, the Ollama `format` payload, the
eval contract and the SDK's typed surface. `spec/label_vocab.json` is a *generated*
artifact (see `build_label_vocab`), versioned so a shipped GGUF can be paired with
the exact mapping it was trained against, and checked against the schemas by
`tests/test_spec.py`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SPEC_DIR = Path(__file__).resolve().parent.parent / "spec"
SCHEMA_DIR = SPEC_DIR / "schemas"

#: Reserved single-digit code meaning "I decline to decide". Never a real label.
ABSTAIN_CODE = 9


@dataclass(frozen=True)
class TaskSpec:
    id: str
    kind: str
    question: str
    description: str
    values: tuple[str, ...]
    ordinal: bool
    abstain_enabled: bool
    priors: dict[str, float]
    boundaries: dict

    # --- codes ------------------------------------------------------------
    @property
    def codes(self) -> tuple[int, ...]:
        """Label codes, excluding abstain."""
        return tuple(range(len(self.values)))

    @property
    def legal_codes(self) -> tuple[int, ...]:
        """Every code the grammar will permit, including abstain if enabled."""
        return self.codes + ((ABSTAIN_CODE,) if self.abstain_enabled else ())

    @property
    def n_classes(self) -> int:
        return len(self.values)

    def code_to_value(self, code: int) -> str | None:
        """Map a digit back to its enum value. Abstain maps to None."""
        if code == ABSTAIN_CODE:
            if not self.abstain_enabled:
                raise ValueError(f"{self.id}: abstain is disabled but got code 9")
            return None
        if not 0 <= code < len(self.values):
            raise ValueError(f"{self.id}: code {code} outside 0..{len(self.values) - 1}")
        return self.values[code]

    def value_to_code(self, value: str | None) -> int:
        if value is None:
            if not self.abstain_enabled:
                raise ValueError(f"{self.id}: abstain is disabled")
            return ABSTAIN_CODE
        try:
            return self.values.index(value)
        except ValueError:
            raise ValueError(
                f"{self.id}: {value!r} not in {list(self.values)}"
            ) from None

    # --- wire formats -----------------------------------------------------
    def format_payload(self) -> dict:
        """The Ollama `format` value: a bare-integer enum.

        Spike 0.1 is the whole reason this is an integer and not a string enum or
        an object — a bare digit puts the decision at token 0 with nothing forced
        ahead of it, so the returned logprob is the model's real belief. See
        spikes/RESULTS.md.
        """
        return {"type": "integer", "enum": list(self.legal_codes)}

    def public_json_schema(self) -> dict:
        """The typed surface the SDK exposes — string enums, not digits.

        The integer encoding is an internal wire detail; callers never see it.
        """
        schema: dict = {"type": "string", "enum": list(self.values)}
        if self.abstain_enabled:
            schema = {"anyOf": [schema, {"type": "null"}]}
        schema["title"] = self.id
        schema["description"] = self.question
        return schema

    def digit_token_ids(self, tokenizer) -> dict[int, int]:
        """Map each legal code to its single token id for this tokenizer.

        Raises if any code is not exactly one token — that would break the
        one-decode-step-per-decision invariant the whole design rests on.
        """
        out = {}
        for c in self.legal_codes:
            ids = tokenizer.encode(str(c), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(
                    f"{self.id}: code {c} tokenizes to {len(ids)} tokens ({ids}); "
                    "the single-token label invariant is violated for this tokenizer"
                )
            out[c] = ids[0]
        return out


def _load_one(path: Path) -> TaskSpec:
    d = json.loads(path.read_text())
    out = d["output"]
    if d["abstain"]["code"] != ABSTAIN_CODE:
        raise ValueError(f"{path}: abstain code must be {ABSTAIN_CODE}")
    if out["codes"] != list(range(len(out["values"]))):
        raise ValueError(f"{path}: codes must be 0..n-1 in value order")
    if len(out["values"]) > ABSTAIN_CODE:
        raise ValueError(f"{path}: >{ABSTAIN_CODE} classes collides with abstain")
    if len(set(out["values"])) != len(out["values"]):
        raise ValueError(f"{path}: duplicate enum values")
    return TaskSpec(
        id=d["id"],
        kind=d["kind"],
        question=d["question"],
        description=d.get("description", ""),
        values=tuple(out["values"]),
        ordinal=bool(out["ordinal"]),
        abstain_enabled=bool(d["abstain"]["enabled"]),
        priors=d.get("priors", {}),
        boundaries=d.get("boundaries", {}),
    )


@lru_cache(maxsize=1)
def load_specs(schema_dir: str | Path | None = None) -> dict[str, TaskSpec]:
    d = Path(schema_dir) if schema_dir else SCHEMA_DIR
    specs = {}
    for p in sorted(d.glob("*.json")):
        s = _load_one(p)
        if s.id != p.stem:
            raise ValueError(f"{p}: id {s.id!r} does not match filename")
        specs[s.id] = s
    if not specs:
        raise FileNotFoundError(f"no schemas in {d}")
    return specs


def get_spec(task_id: str) -> TaskSpec:
    specs = load_specs()
    if task_id not in specs:
        raise KeyError(f"unknown task {task_id!r}; have {sorted(specs)}")
    return specs[task_id]


# --- prompt rendering ----------------------------------------------------

@lru_cache(maxsize=1)
def load_template() -> dict:
    return json.loads((SPEC_DIR / "template.json").read_text())


_WS = re.compile(r"[ \t]+")


def normalize_input(text: str, max_chars: int | None = None) -> str:
    """Canonicalize input text.

    Deterministic and lossy on purpose: identical normalization at train and serve
    time is what keeps the prompt prefix byte-identical, which is what makes
    Ollama's prefix cache hit across the k requests of a multi-field decision.
    """
    if max_chars is None:
        max_chars = load_template()["max_input_chars"]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(_WS.sub(" ", ln).rstrip() for ln in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def render_prompt(task_id: str, input_text: str) -> str:
    """The ONLY way a prompt should ever be built.

    Input first, task tag last, terminator owning the trailing space — all three
    for measured reasons recorded in spec/tasks.md.
    """
    get_spec(task_id)  # validate
    t = load_template()
    return t["template"].format(
        input=normalize_input(input_text, t["max_input_chars"]), task_id=task_id
    )


def shared_prefix(input_text: str) -> str:
    """The portion of the prompt common to every task for one input.

    The SDK uses this to reason about cache hits when fanning out k fields.
    """
    t = load_template()
    tmpl = t["template"]
    head = tmpl.split("#{task_id}")[0]
    return head.format(input=normalize_input(input_text, t["max_input_chars"]))


# --- generated artifact --------------------------------------------------

def build_label_vocab() -> dict:
    """Build the versioned enum<->digit mapping shipped beside the weights.

    A consumer holding the GGUF but the wrong label_vocab decodes plausible-looking
    nonsense, so this travels with the model and is asserted in tests.
    """
    specs = load_specs()
    t = load_template()
    return {
        "version": 1,
        "abstain_code": ABSTAIN_CODE,
        "template": t["template"],
        "encoding": "bare-integer enum, one decode step per field (spikes/RESULTS.md 0.1)",
        "tasks": {
            s.id: {
                "kind": s.kind,
                "ordinal": s.ordinal,
                "abstain_enabled": s.abstain_enabled,
                "legal_codes": list(s.legal_codes),
                "code_to_value": {
                    **{str(c): s.values[c] for c in s.codes},
                    **({str(ABSTAIN_CODE): None} if s.abstain_enabled else {}),
                },
                "value_to_code": {v: i for i, v in enumerate(s.values)},
            }
            for s in specs.values()
        },
    }


def write_label_vocab(path: str | Path | None = None) -> Path:
    p = Path(path) if path else SPEC_DIR / "label_vocab.json"
    p.write_text(json.dumps(build_label_vocab(), indent=2) + "\n")
    return p


if __name__ == "__main__":
    specs = load_specs()
    print(f"{len(specs)} tasks")
    for s in specs.values():
        print(f"  {s.id:<16} {s.kind:<13} {s.n_classes} classes "
              f"{'ordinal' if s.ordinal else 'nominal':<8} "
              f"abstain={str(s.abstain_enabled):<5} format={s.format_payload()}")
    print(f"\nwrote {write_label_vocab()}")
    print("\nexample prompt:")
    print(repr(render_prompt("route_v1", "I was charged twice for  October\n\n")))
