"""Invariants that the whole design rests on. These are not formalities.

Each test here corresponds to something a Phase 0 spike measured or something that,
if it silently broke, would produce plausible-looking but wrong confidences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jev.spec import (ABSTAIN_CODE, build_label_vocab, get_spec, load_specs,
                      normalize_input, render_prompt, shared_prefix)

ROOT = Path(__file__).resolve().parent.parent


def test_specs_load():
    specs = load_specs()
    assert len(specs) == 6
    assert set(specs) == {"route_v1", "triage_v1", "gate_human_v1",
                          "intent_v1", "blocker_v1", "urgency_v1"}


@pytest.mark.parametrize("task_id", sorted(load_specs()))
def test_codes_roundtrip(task_id):
    s = get_spec(task_id)
    for c in s.codes:
        assert s.value_to_code(s.code_to_value(c)) == c
    for v in s.values:
        assert s.code_to_value(s.value_to_code(v)) == v


@pytest.mark.parametrize("task_id", sorted(load_specs()))
def test_abstain_never_collides_with_a_label(task_id):
    s = get_spec(task_id)
    assert s.n_classes <= ABSTAIN_CODE, "a label code would collide with abstain"
    assert ABSTAIN_CODE not in s.codes


def test_gate_has_abstain_disabled():
    """gate_human_v1 is a safety gate: unsure must fail closed, not defer."""
    assert get_spec("gate_human_v1").abstain_enabled is False
    assert ABSTAIN_CODE not in get_spec("gate_human_v1").legal_codes


@pytest.mark.parametrize("task_id", sorted(load_specs()))
def test_format_payload_is_a_bare_integer_enum(task_id):
    """Spike 0.1: a string enum forces an opening quote at p~7.8e-09, which
    corrupts the decision token. Integer enums put the decision at position 0."""
    f = get_spec(task_id).format_payload()
    assert f["type"] == "integer", "string/object encodings corrupt position 0"
    assert f["enum"] == list(get_spec(task_id).legal_codes)
    assert all(0 <= c <= 9 for c in f["enum"]), "codes must be single digits"


def test_abstain_disabled_means_no_9_in_grammar():
    assert 9 not in get_spec("gate_human_v1").format_payload()["enum"]


# --- prompt template ----------------------------------------------------

def test_prompt_terminator_owns_the_trailing_space():
    """Spike 0.3: space-prefixed digits are never single tokens, so a prefix
    ending `answer:` would leave the model wanting ` 1` (two tokens) and the
    grammar would force the bare `1` against its preference."""
    p = render_prompt("route_v1", "hello")
    assert p.endswith("answer: "), repr(p)
    assert not p.endswith("answer:")


def test_input_comes_before_the_task_tag():
    """Multi-field decisions issue k requests for one input; the long shared
    prefix must come first or Ollama's prefix cache cannot help."""
    p = render_prompt("route_v1", "UNIQUEBODY")
    assert p.index("UNIQUEBODY") < p.index("#route_v1")


def test_shared_prefix_is_identical_across_tasks():
    text = "my card was declined"
    pre = shared_prefix(text)
    for t in load_specs():
        assert render_prompt(t, text).startswith(pre)


def test_prompt_rendering_is_deterministic():
    """Prefix-cache hits depend on byte-identical rendering."""
    a = [render_prompt("route_v1", "  weird\r\n\n\n  spacing \t here  ")
         for _ in range(5)]
    assert len(set(a)) == 1


def test_normalize_input_is_idempotent():
    raw = "  line \t one \r\n\r\n\r\n   line   two  "
    once = normalize_input(raw)
    assert normalize_input(once) == once


def test_normalize_input_truncates():
    assert len(normalize_input("x" * 5000, max_chars=100)) <= 100


def test_unknown_task_raises():
    with pytest.raises(KeyError):
        render_prompt("not_a_task", "hi")


def test_bad_code_raises():
    with pytest.raises(ValueError):
        get_spec("route_v1").code_to_value(7)


def test_abstain_on_disabled_task_raises():
    with pytest.raises(ValueError):
        get_spec("gate_human_v1").code_to_value(ABSTAIN_CODE)


# --- the generated artifact ---------------------------------------------

def test_label_vocab_on_disk_matches_the_schemas():
    """spec/label_vocab.json ships beside the GGUF. A consumer holding the weights
    and a stale vocab decodes plausible-looking nonsense, so drift must fail CI."""
    p = ROOT / "spec" / "label_vocab.json"
    assert p.exists(), "run `python -m jev.spec` to generate it"
    on_disk = json.loads(p.read_text())
    assert on_disk == build_label_vocab(), (
        "spec/label_vocab.json is stale; regenerate with `python -m jev.spec`")


def test_label_vocab_records_the_template():
    """The template is part of the contract: a model trained on one prompt and
    served with another silently loses accuracy and calibration."""
    v = build_label_vocab()
    assert v["template"] == render_prompt("route_v1", "{X}").replace(
        "{X}", "{input}").replace("#route_v1", "#{task_id}")


def test_priors_are_a_distribution_where_declared():
    for s in load_specs().values():
        if not s.priors:
            continue
        assert set(s.priors) <= set(s.values), f"{s.id}: prior over unknown value"
        assert abs(sum(s.priors.values()) - 1.0) < 0.02, f"{s.id}: priors must sum to 1"
