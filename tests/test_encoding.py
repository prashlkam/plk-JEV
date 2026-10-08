"""Tokenizer-level invariants and the SDK's schema mapping.

The one-decode-step-per-decision design is only true if every label code is
exactly one token. If a tokenizer change broke that, latency would silently
double and the logprob would stop being the decision's probability.
"""
from __future__ import annotations

from typing import Literal, Optional

import pytest

from jev.spec import load_specs

MODELS = ["HuggingFaceTB/SmolLM2-135M"]


@pytest.fixture(scope="module", params=MODELS)
def tok(request):
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(request.param)


def test_every_legal_code_is_exactly_one_token(tok):
    for spec in load_specs().values():
        ids = spec.digit_token_ids(tok)
        assert set(ids) == set(spec.legal_codes)
        assert all(isinstance(v, int) for v in ids.values())


def test_digit_token_ids_are_distinct_within_a_task(tok):
    for spec in load_specs().values():
        ids = spec.digit_token_ids(tok)
        assert len(set(ids.values())) == len(ids), f"{spec.id}: colliding token ids"


def test_space_prefixed_digits_are_not_single_tokens(tok):
    """Spike 0.3's trap. If this ever starts passing as single-token, the
    `answer: ` trailing space in the template could be revisited."""
    multi = [d for d in "0123456789"
             if len(tok.encode(" " + d, add_special_tokens=False)) != 1]
    assert multi, ("space-prefixed digits became single tokens for this "
                   "tokenizer; re-examine the template terminator")


def test_terminator_does_not_merge_with_the_decision_digit(tok):
    """A BPE merge across the prompt/generation boundary would corrupt position 0."""
    from jev.spec import render_prompt
    base = tok.encode(render_prompt("route_v1", "hello"), add_special_tokens=False)
    for d in "0123456789":
        joint = tok.encode(render_prompt("route_v1", "hello") + d,
                           add_special_tokens=False)
        assert joint[:len(base)] == base, f"digit {d} merged with the terminator"
        assert len(joint) == len(base) + 1


# --- SDK schema mapping -------------------------------------------------

def test_sdk_maps_fields_named_after_tasks():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from serve.jev_sdk import map_model_to_tasks

    class M(BaseModel):
        route_v1: Optional[Literal["billing", "technical", "account",
                                   "sales", "abuse"]]
        gate_human_v1: bool

    m = map_model_to_tasks(M)
    assert m["route_v1"].id == "route_v1"
    assert m["gate_human_v1"].id == "gate_human_v1"


def test_sdk_maps_by_option_set_not_just_name():
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from serve.jev_sdk import map_model_to_tasks

    class M(BaseModel):
        severity: Optional[Literal["P0", "P1", "P2", "P3"]]

    assert map_model_to_tasks(M)["severity"].id == "triage_v1"


def test_sdk_rejects_an_unmappable_field():
    """Silently dropping a field would populate the model with defaults that look
    like real decisions."""
    pytest.importorskip("pydantic")
    from pydantic import BaseModel
    from serve.jev_sdk import SchemaMismatch, map_model_to_tasks

    class M(BaseModel):
        mystery: Optional[Literal["a", "b"]]

    with pytest.raises(SchemaMismatch):
        map_model_to_tasks(M)
