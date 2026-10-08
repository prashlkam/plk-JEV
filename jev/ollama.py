"""Ollama client that recovers trustworthy probabilities from pre-mask logprobs.

Spike 0.1 found that Ollama reports logprobs *before* the grammar mask is applied.
For the bare-integer encoding this is benign — the model already puts ~99% of its
mass on the legal digits, so the mask is near non-binding — but it is only benign
if you actually check, and renormalize. This module does both:

  * renormalize over the legal code set, so probabilities sum to 1 over the
    decisions that were actually reachable;
  * report `legal_mass`, the pre-mask probability that landed on the legal set,
    which is the gate metric for whether the number means anything at all.

Uses /api/generate with raw=True. That matters: JEV is trained on a bare prompt
template, so any chat template Ollama would otherwise inject would put the model
off-distribution and move the decision away from token 0.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from .spec import ABSTAIN_CODE, TaskSpec, get_spec, load_specs, render_prompt

DEFAULT_HOST = "http://localhost:11434"
#: Ollama REJECTS top_logprobs > 20 with a 400 ("must be between 0 and 20"), so
#: 20 is both the default and the ceiling. This matters: blocker_v1 and intent_v1
#: have 7 legal codes, and there is no way to buy more headroom. Whether 20 was
#: enough is therefore a measurement, not an assumption — `Decision.legal_coverage`
#: reports it per request and the eval harness aggregates it.
MAX_TOP_LOGPROBS = 20
DEFAULT_TOP_LOGPROBS = MAX_TOP_LOGPROBS

#: ASCII digits only. `str.isdigit()` is a trap here: it returns True for
#: superscripts like "²" (which int() then REFUSES, crashing) and for Arabic-Indic
#: digits like "٣" (which int() happily parses as 3, silently matching the wrong
#: token to a label code). Both are present in real tokenizer vocabularies and both
#: showed up in top_logprobs during the first live calibration run.
_ASCII_DIGITS = frozenset("0123456789")


def as_code(token: str) -> int | None:
    """Parse a token into a label code, or None if it is not a bare ASCII digit."""
    t = token.strip()
    if len(t) == 1 and t in _ASCII_DIGITS:
        return int(t)
    return None


@dataclass
class Decision:
    task_id: str
    code: int
    value: str | None
    confidence: float
    """Calibrated probability of `code`, after temperature scaling."""
    distribution: dict[str, float]
    """Calibrated probability per enum value (plus `__abstain__` when enabled)."""
    legal_mass: float
    """Pre-mask probability on the legal set. <0.95 means do not trust `confidence`."""
    legal_coverage: float
    """Fraction of legal codes whose logprob was visible in top_logprobs."""
    raw_confidence: float
    """Confidence before temperature scaling, for diagnostics."""
    temperature: float
    decision_token_index: int
    """Must be 0. Anything else means the template regressed; see spec/tasks.md."""
    latency_ms: float
    eval_count: int
    prompt_eval_count: int
    prompt_eval_ms: float
    """Prompt-evaluation time. Ollama exposes NO cache-hit flag, and
    `prompt_eval_count` always reports the full prompt length even for an
    identical repeated request (measured: 55 tokens on every one of 4 identical
    calls). So prompt-eval *time* is the only observable proxy for prefix reuse —
    it drops ~40% on follow-up requests sharing a prefix. Do not treat it as
    proof of a cache hit."""
    warnings: list[str] = field(default_factory=list)

    @property
    def abstained(self) -> bool:
        return self.code == ABSTAIN_CODE

    @property
    def trustworthy(self) -> bool:
        return self.legal_mass >= 0.95 and self.decision_token_index == 0


def load_calibration(path: str | Path | None = None) -> dict:
    """Per-task temperatures produced by train/calibrate.py.

    Missing file means T=1 everywhere — uncalibrated but functional, and the SDK
    warns so this is never silently the case in production.
    """
    p = Path(path) if path else Path(__file__).resolve().parent.parent / "spec" / "calibration.json"
    if not p.exists():
        return {"version": 0, "fitted": False, "temperatures": {}}
    return json.loads(p.read_text())


class JevOllama:
    def __init__(self, model: str = "jev", host: str = DEFAULT_HOST,
                 calibration: dict | None = None, keep_alive: str = "30m",
                 top_logprobs: int = DEFAULT_TOP_LOGPROBS, timeout: float = 120.0):
        self.model = model
        self.host = host.rstrip("/")
        self.keep_alive = keep_alive
        if not 0 <= top_logprobs <= MAX_TOP_LOGPROBS:
            raise ValueError(
                f"top_logprobs={top_logprobs}: Ollama accepts 0..{MAX_TOP_LOGPROBS} "
                f"and returns HTTP 400 outside that range")
        self.top_logprobs = top_logprobs
        self.cal = calibration if calibration is not None else load_calibration()
        # One client, so connections are reused across the k requests of a fan-out.
        self._http = httpx.Client(base_url=self.host, timeout=timeout)
        self._warned_uncal = False

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- temperature ----------------------------------------------------
    def temperature(self, task_id: str) -> float:
        return float(self.cal.get("temperatures", {}).get(task_id, 1.0))

    # --- single field ---------------------------------------------------
    def decide(self, task_id: str, input_text: str) -> Decision:
        spec = get_spec(task_id)
        prompt = render_prompt(task_id, input_text)
        body = {
            "model": self.model,
            "prompt": prompt,
            "raw": True,              # no chat template: JEV is trained bare
            "stream": False,
            "format": spec.format_payload(),
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,     # argmax IS the answer; uncertainty rides in logprobs
                "top_p": 1.0,
                "top_k": 0,
                "num_predict": 4,     # 1 digit + EOS, with slack
                "seed": 0,
            },
        }
        t0 = time.perf_counter()
        r = self._http.post("/api/generate", json=body)
        r.raise_for_status()
        d = r.json()
        latency_ms = (time.perf_counter() - t0) * 1000
        return self._parse(spec, d, latency_ms)

    def _parse(self, spec: TaskSpec, d: dict, latency_ms: float) -> Decision:
        warnings: list[str] = []
        if not self.cal.get("fitted") and not self._warned_uncal:
            self._warned_uncal = True
            warnings.append(
                "calibration.json absent or unfitted: temperatures default to 1.0, "
                "so confidences are UNCALIBRATED. Run train/calibrate.py."
            )

        text = (d.get("response") or "").strip()
        lp = d.get("logprobs") or []

        # Which generated token carried the decision? Must be index 0.
        idx = next((i for i, t in enumerate(lp)
                    if as_code(t["token"]) is not None), -1)
        if idx != 0:
            warnings.append(
                f"decision_token_index={idx} (expected 0): the prompt template or "
                "grammar changed, and position-0 corruption may be back. "
                "See spikes/RESULTS.md 0.1."
            )

        legal = set(spec.legal_codes)
        emitted = as_code(text)
        if emitted is None or emitted not in legal:
            # The grammar should make this impossible. If it happens, the model or
            # the Modelfile is mismatched with the spec.
            raise ValueError(
                f"{spec.id}: response {text!r} is not a legal code {sorted(legal)}. "
                "The grammar guarantee is broken — check the Modelfile and format payload."
            )

        tok = lp[idx] if 0 <= idx < len(lp) else (lp[0] if lp else None)
        logprobs: dict[int, float] = {}
        if tok is not None:
            for cand in tok.get("top_logprobs") or []:
                c = as_code(cand["token"])
                if c is not None and c in legal:
                    logprobs[c] = max(logprobs.get(c, -math.inf), cand["logprob"])
            # The emitted token's own logprob is reported separately and may be
            # absent from top_logprobs precisely because logprobs are pre-mask.
            if as_code(tok["token"]) == emitted:
                logprobs[emitted] = max(logprobs.get(emitted, -math.inf), tok["logprob"])

        legal_mass = sum(math.exp(v) for v in logprobs.values())
        coverage = len(logprobs) / len(legal)

        if emitted not in logprobs:
            # Cannot see it; impute conservatively at the floor of what we did see
            # so the emitted decision still gets a probability, and say so.
            floor = min(logprobs.values()) if logprobs else -20.0
            logprobs[emitted] = floor
            warnings.append(
                f"emitted code {emitted} absent from top_logprobs; imputed at the "
                "observed floor. Raise top_logprobs or treat this confidence as a lower bound."
            )
        if coverage < 1.0:
            warnings.append(
                f"legal-set coverage {coverage:.2f}: {len(legal) - len(logprobs)} of "
                f"{len(legal)} codes missing from top_logprobs; renormalization is approximate."
            )
        if legal_mass < 0.95:
            warnings.append(
                f"legal_mass={legal_mass:.3f} < 0.95: the grammar mask is doing the "
                "deciding, not the model. Confidence is not trustworthy."
            )

        T = self.temperature(spec.id)
        codes = sorted(logprobs)
        scaled = [logprobs[c] / T for c in codes]
        mx = max(scaled)
        exps = [math.exp(s - mx) for s in scaled]
        Z = sum(exps)
        probs = {c: e / Z for c, e in zip(codes, exps)}

        raw_mx = max(logprobs.values())
        raw_exps = {c: math.exp(v - raw_mx) for c, v in logprobs.items()}
        raw_Z = sum(raw_exps.values())

        dist = {}
        for c, p in probs.items():
            key = "__abstain__" if c == ABSTAIN_CODE else spec.values[c]
            dist[key] = p

        return Decision(
            task_id=spec.id,
            code=emitted,
            value=spec.code_to_value(emitted),
            confidence=probs[emitted],
            distribution=dist,
            legal_mass=legal_mass,
            legal_coverage=coverage,
            raw_confidence=raw_exps[emitted] / raw_Z,
            temperature=T,
            decision_token_index=idx,
            latency_ms=latency_ms,
            eval_count=int(d.get("eval_count") or 0),
            prompt_eval_count=int(d.get("prompt_eval_count") or 0),
            prompt_eval_ms=(d.get("prompt_eval_duration") or 0) / 1e6,
            warnings=warnings,
        )

    # --- fan-out --------------------------------------------------------
    def decide_all(self, input_text: str,
                   task_ids: list[str] | None = None) -> dict[str, Decision]:
        """Every task for one input.

        Issued as k sequential single-field requests on purpose. Spike 0.1 measured
        the alternative — one array/object call — at 9x the decode cost *and* with
        corrupted logprobs. Here requests 2..k hit Ollama's prefix cache because the
        template puts the input before the task tag.
        """
        ids = task_ids if task_ids is not None else list(load_specs())
        return {t: self.decide(t, input_text) for t in ids}

    def decide_or_defer(self, task_id: str, input_text: str,
                        threshold: float = 0.9) -> Decision | None:
        """`None` when the model is not confident enough to act.

        The primary intended usage pattern. Note `gate_human_v1` has abstain
        disabled because it is a safety gate: rather than deferring, sub-threshold
        confidence there is forced to the fail-closed class by the SDK layer.
        """
        d = self.decide(task_id, input_text)
        spec = get_spec(task_id)
        if d.confidence >= threshold and not d.abstained:
            return d
        if not spec.abstain_enabled:
            return d  # caller must fail closed; see jev_sdk for the typed wrapper
        return None

    # --- housekeeping ---------------------------------------------------
    def unload(self):
        """Free the GPU. Required before training: a resident model costs 964 MiB
        of the 3.3 GiB available (HARDWARE.md revision 4)."""
        self._http.post("/api/generate",
                        json={"model": self.model, "keep_alive": 0, "prompt": ""})

    def health(self) -> dict:
        tags = self._http.get("/api/tags").json()
        names = [m["name"] for m in tags.get("models", [])]
        return {"host": self.host, "model": self.model,
                "model_present": any(n.split(":")[0] == self.model.split(":")[0]
                                     for n in names),
                "models": names}
