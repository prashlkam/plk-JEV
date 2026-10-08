"""The pre-mask renormalization path, tested against fabricated Ollama payloads.

This is the subtlest logic in the project. Spike 0.1 established that Ollama
reports logprobs BEFORE the grammar mask, so `JevOllama._parse` has to:

  * take the emitted token as ground truth for the argmax (the grammar picked it,
    and it may be absent from its own top_logprobs);
  * renormalize over the legal code set, since the pre-mask distribution does not
    sum to 1 over legal codes;
  * report `legal_mass` so a caller can tell whether the model or the mask decided;
  * apply the per-task temperature;
  * flag incomplete coverage rather than silently producing a confident number.

Each test pins one of those against a payload shaped exactly like the real API
response recorded in spikes/raw_0_1.json.
"""

from __future__ import annotations

import math

import pytest

from jev.ollama import JevOllama
from jev.spec import get_spec


def payload(emitted, top, *, logprob=None, eval_count=2, prompt_eval_count=30):
    """Build an /api/generate response. `top` maps token string -> logprob."""
    return {
        "response": str(emitted),
        "eval_count": eval_count,
        "prompt_eval_count": prompt_eval_count,
        "logprobs": [{
            "token": str(emitted),
            "logprob": logprob if logprob is not None
            else top.get(str(emitted), -10.0),
            "top_logprobs": [{"token": k, "logprob": v} for k, v in top.items()],
        }],
    }


@pytest.fixture
def cli():
    c = JevOllama(model="test", calibration={"fitted": True, "temperatures": {}})
    yield c
    c.close()


def parse(cli, task_id, d):
    return cli._parse(get_spec(task_id), d, latency_ms=1.0)


# --- renormalization ----------------------------------------------------

def test_renormalizes_over_the_legal_set(cli):
    """Pre-mask mass on legal codes is 0.6; confidence must be relative to that,
    not the raw 0.4."""
    top = {"0": math.log(0.4), "1": math.log(0.2), "zebra": math.log(0.4)}
    d = parse(cli, "gate_human_v1", payload(0, top))
    assert d.code == 0
    assert d.legal_mass == pytest.approx(0.6)
    assert d.confidence == pytest.approx(0.4 / 0.6)
    assert sum(d.distribution.values()) == pytest.approx(1.0)


def test_emitted_token_wins_even_if_absent_from_top_logprobs(cli):
    """The grammar can force a token the model rated below the top-k cutoff.

    Ollama still reports that token's own logprob in the per-token `logprob`
    field — spike 0.1 saw `{"` come back at -14.4 while its top_logprobs listed
    only `Yes`/`No`. So the emitted token's probability is known EXACTLY and is
    never imputed; it just has to be merged in from outside top_logprobs.
    """
    top = {"1": math.log(0.9)}
    d = parse(cli, "gate_human_v1", payload(0, top, logprob=math.log(1e-6)))
    assert d.code == 0, "emitted token must win the argmax"
    assert d.legal_mass == pytest.approx(0.9 + 1e-6)
    assert d.confidence == pytest.approx(1e-6 / (0.9 + 1e-6))
    assert not any("imputed" in w for w in d.warnings), \
        "the top-level logprob field makes imputation unnecessary here"


def test_imputation_only_when_there_are_no_logprobs_at_all(cli):
    """The genuine fallback: logprobs missing entirely (e.g. an older Ollama, or
    a request that somehow lost the flag). Confidence then IS a guess, and the
    caller must be told so."""
    d = parse(cli, "gate_human_v1",
              {"response": "1", "eval_count": 2, "prompt_eval_count": 5,
               "logprobs": []})
    assert d.code == 1
    assert any("imputed" in w for w in d.warnings)
    assert d.trustworthy is False


def test_distribution_keys_are_enum_values_not_digits(cli):
    top = {"0": math.log(0.3), "1": math.log(0.6)}
    d = parse(cli, "gate_human_v1", payload(1, top))
    assert set(d.distribution) == {"false", "true"}
    assert d.value == "true"


def test_abstain_maps_to_none_and_is_flagged(cli):
    top = {"0": math.log(0.1), "9": math.log(0.8)}
    d = parse(cli, "route_v1", payload(9, top))
    assert d.abstained is True
    assert d.value is None
    assert "__abstain__" in d.distribution


def test_abstain_code_rejected_when_disabled(cli):
    """gate_human_v1 has no 9 in its grammar; a 9 means a spec/model mismatch."""
    with pytest.raises(ValueError, match="not a legal code"):
        parse(cli, "gate_human_v1", payload(9, {"9": math.log(0.9)}))


def test_illegal_response_raises(cli):
    """The grammar should make this impossible; if it happens the Modelfile or the
    format payload is wrong, and failing loudly is the only safe option."""
    with pytest.raises(ValueError, match="grammar guarantee is broken"):
        parse(cli, "route_v1", payload(7, {"7": math.log(0.9)}))
    with pytest.raises(ValueError, match="not a legal code"):
        parse(cli, "route_v1", {"response": "billing", "logprobs": []})


# --- the legal_mass gate ------------------------------------------------

def test_high_legal_mass_is_trustworthy(cli):
    top = {"0": math.log(0.70), "1": math.log(0.29), "junk": math.log(0.01)}
    d = parse(cli, "gate_human_v1", payload(0, top))
    assert d.legal_mass == pytest.approx(0.99)
    assert d.trustworthy is True
    assert not any("legal_mass" in w for w in d.warnings)


def test_low_legal_mass_warns_and_is_untrustworthy(cli):
    """The spike 0.1 failure mode: the mask decided, so confidence is an artifact
    of renormalizing noise — even though it will look like a clean 0.8."""
    top = {"0": math.log(0.08), "1": math.log(0.02), "Yes": math.log(0.9)}
    d = parse(cli, "gate_human_v1", payload(0, top))
    assert d.legal_mass == pytest.approx(0.10)
    assert d.confidence == pytest.approx(0.8)      # looks fine...
    assert d.trustworthy is False                   # ...but is not
    assert any("legal_mass" in w for w in d.warnings)


def test_partial_coverage_is_reported(cli):
    """6 classes + abstain = 7 legal codes; only 2 visible."""
    top = {"0": math.log(0.5), "1": math.log(0.4)}
    d = parse(cli, "blocker_v1", payload(0, top))
    assert d.legal_coverage == pytest.approx(2 / 7)
    assert any("coverage" in w for w in d.warnings)


def test_decision_token_index_must_be_zero(cli):
    """Spike 0.3's template gate: scaffolding ahead of the digit means the
    position-0 corruption is back."""
    d = {
        "response": "1",
        "logprobs": [
            {"token": '{"', "logprob": -14.0, "top_logprobs": []},
            {"token": "1", "logprob": math.log(0.9),
             "top_logprobs": [{"token": "1", "logprob": math.log(0.9)},
                              {"token": "0", "logprob": math.log(0.1)}]},
        ],
    }
    out = parse(cli, "gate_human_v1", d)
    assert out.decision_token_index == 1
    assert out.trustworthy is False
    assert any("decision_token_index" in w for w in out.warnings)


# --- temperature --------------------------------------------------------

def test_temperature_flattens_confidence():
    top = {"0": math.log(0.9), "1": math.log(0.1)}
    with JevOllama(model="t", calibration={"fitted": True,
                                           "temperatures": {"gate_human_v1": 3.0}}) as c:
        hot = c._parse(get_spec("gate_human_v1"), payload(0, top), 1.0)
    with JevOllama(model="t", calibration={"fitted": True, "temperatures": {}}) as c:
        cold = c._parse(get_spec("gate_human_v1"), payload(0, top), 1.0)
    assert hot.temperature == 3.0
    assert hot.confidence < cold.confidence, "T>1 must reduce confidence"
    assert hot.raw_confidence == pytest.approx(cold.confidence), \
        "raw_confidence must be pre-temperature"


def test_temperature_cannot_change_the_decision():
    top = {"0": math.log(0.55), "1": math.log(0.44)}
    codes = set()
    for T in (0.2, 1.0, 5.0):
        with JevOllama(model="t", calibration={
                "fitted": True, "temperatures": {"gate_human_v1": T}}) as c:
            codes.add(c._parse(get_spec("gate_human_v1"), payload(0, top), 1.0).code)
    assert codes == {0}


def test_missing_calibration_warns_once():
    with JevOllama(model="t", calibration={"version": 0, "fitted": False,
                                           "temperatures": {}}) as c:
        top = {"0": math.log(0.9), "1": math.log(0.1)}
        first = c._parse(get_spec("gate_human_v1"), payload(0, top), 1.0)
        second = c._parse(get_spec("gate_human_v1"), payload(0, top), 1.0)
    assert any("UNCALIBRATED" in w for w in first.warnings)
    assert not any("UNCALIBRATED" in w for w in second.warnings), \
        "should warn once per client, not per call"


# --- misc ---------------------------------------------------------------

def test_prompt_eval_ms_is_surfaced(cli):
    """There is no cache-hit flag in the Ollama API — `prompt_eval_count` reports
    the full prompt length even on an identical repeated request — so the SDK
    surfaces prompt-eval TIME and nothing stronger."""
    top = {"0": math.log(0.9), "1": math.log(0.1)}
    d = payload(0, top)
    d["prompt_eval_duration"] = 22_000_000      # ns
    out = parse(cli, "gate_human_v1", d)
    assert out.prompt_eval_ms == pytest.approx(22.0)
    assert not hasattr(out, "prompt_cached"), \
        "a boolean cache flag would be unfalsifiable against this API"


def test_duplicate_candidate_tokens_take_the_max(cli):
    """Tokenizers can surface the same digit twice (e.g. with whitespace); taking
    the max rather than summing avoids inflating legal_mass past 1."""
    top_list = [{"token": "0", "logprob": math.log(0.3)},
                {"token": " 0", "logprob": math.log(0.5)},
                {"token": "1", "logprob": math.log(0.2)}]
    d = {"response": "0", "eval_count": 2, "prompt_eval_count": 5,
         "logprobs": [{"token": "0", "logprob": math.log(0.5),
                       "top_logprobs": top_list}]}
    out = parse(cli, "gate_human_v1", d)
    assert out.legal_mass <= 1.0 + 1e-9
    assert out.legal_mass == pytest.approx(0.7)  # max(0.3, 0.5) + 0.2


# --- the unicode-digit trap (found live during the first GGUF calibration) ---

def test_as_code_rejects_non_ascii_digits():
    """`str.isdigit()` is True for "²" (int() then CRASHES) and for "٣" (int()
    parses it as 3, silently matching the wrong token to a label code). Both occur
    in real tokenizer vocabularies and both appeared in top_logprobs on the first
    live run against the quantized GGUF."""
    from jev.ollama import as_code
    assert as_code("3") == 3
    assert as_code(" 3 ") == 3
    for bad in ["²", "٣", "½", "一", "10", "1.0", "", "x", "-1"]:
        assert as_code(bad) is None, bad


def test_superscript_in_top_logprobs_does_not_crash(cli):
    """Regression: this exact payload shape crashed train/calibrate.py."""
    d = {
        "response": "1", "eval_count": 2, "prompt_eval_count": 5,
        "logprobs": [{
            "token": "1", "logprob": math.log(0.8),
            "top_logprobs": [
                {"token": "1", "logprob": math.log(0.8)},
                {"token": "²", "logprob": math.log(0.1)},   # isdigit() -> True
                {"token": "٣", "logprob": math.log(0.05)},  # isdigit() -> True
                {"token": "0", "logprob": math.log(0.05)},
            ],
        }],
    }
    out = parse(cli, "gate_human_v1", d)
    assert out.code == 1
    # Only the two ASCII digits count toward the legal set.
    assert out.legal_mass == pytest.approx(0.85)
