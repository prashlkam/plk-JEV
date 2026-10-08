"""Metric definitions, checked against cases where the right answer is known.

Calibration metrics are easy to get subtly wrong and a wrong ECE would silently
invalidate every result in the project, so each one is pinned to a case computable
by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from jev import metrics as M


def onehot(labels, k):
    p = np.zeros((len(labels), k))
    p[np.arange(len(labels)), labels] = 1.0
    return p


# --- basics -------------------------------------------------------------

def test_accuracy_and_f1_perfect():
    y = np.array([0, 1, 2, 1])
    p = onehot(y, 3)
    assert M.accuracy(p, y) == 1.0
    assert M.macro_f1(p, y) == 1.0
    assert M.brier(p, y) == 0.0
    assert M.ece(p, y) == pytest.approx(0.0)


def test_brier_worst_case_is_two():
    """Sum-form multiclass Brier ranges [0, 2]: confidently wrong costs 2."""
    y = np.array([0])
    p = np.array([[0.0, 1.0]])
    assert M.brier(p, y) == pytest.approx(2.0)


def test_nll_matches_hand_computation():
    y = np.array([0, 1])
    p = np.array([[0.8, 0.2], [0.25, 0.75]])
    expected = -(np.log(0.8) + np.log(0.75)) / 2
    assert M.nll(p, y) == pytest.approx(expected)


def test_normalization_is_enforced():
    with pytest.raises(ValueError):
        M.accuracy(np.array([[0.5, 0.2]]), np.array([0]))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        M.accuracy(np.array([[0.5, 0.5]]), np.array([0, 1]))


def test_empty_is_nan_not_crash():
    p = np.zeros((0, 3))
    y = np.zeros(0, dtype=int)
    assert np.isnan(M.accuracy(p, y))
    assert np.isnan(M.ece(p, y))


# --- ECE ----------------------------------------------------------------

def test_ece_is_zero_for_a_perfectly_calibrated_model():
    """70% confidence, right exactly 70% of the time -> ECE 0."""
    n = 1000
    p = np.tile([0.7, 0.3], (n, 1))
    y = np.array([0] * 700 + [1] * 300)
    assert M.ece(p, y) == pytest.approx(0.0, abs=0.01)


def test_ece_catches_overconfidence():
    """Always 99% confident, right only half the time -> ECE ~0.49."""
    n = 1000
    p = np.tile([0.99, 0.01], (n, 1))
    y = np.array([0] * 500 + [1] * 500)
    assert M.ece(p, y) == pytest.approx(0.49, abs=0.02)


def test_mce_is_at_least_ece():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(600, 4))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = rng.integers(0, 4, 600)
    assert M.mce(p, y) >= M.ece(p, y) - 1e-9


def test_temperature_scaling_reduces_ece_on_an_overconfident_model():
    """The property train/calibrate.py depends on."""
    rng = np.random.default_rng(1)
    n, k = 4000, 4
    y = rng.integers(0, k, n)
    logits = rng.normal(size=(n, k))
    logits[np.arange(n), y] += 1.2
    hot = logits * 4.0                      # deliberately overconfident
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(
        z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    assert M.ece(sm(hot / 4.0), y) < M.ece(sm(hot), y)


def test_temperature_scaling_cannot_change_accuracy():
    """A monotone transform preserves the argmax — the reason temperature scaling
    is safe to apply unconditionally."""
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(500, 5))
    y = rng.integers(0, 5, 500)
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(
        z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    for T in (0.1, 0.5, 2.0, 11.0):
        assert M.accuracy(sm(logits / T), y) == M.accuracy(sm(logits), y)


# --- selective prediction ----------------------------------------------

def test_selective_accuracy_improves_with_lower_coverage():
    """Confidence that ranks correctness is the whole point of decide_or_defer."""
    n = 1000
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.5, 1.0, n)
    correct = rng.uniform(size=n) < conf       # correctness tracks confidence
    p = np.zeros((n, 2))
    p[:, 0] = np.where(correct, conf, 1 - conf)
    p[:, 1] = 1 - p[:, 0]
    y = np.zeros(n, dtype=int)
    s = M.selective_accuracy(p, y)
    assert s["0.5"]["accuracy"] > s["1.0"]["accuracy"]


def test_aurc_lower_when_confidence_ranks_correctness():
    n = 800
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, n)
    good = np.zeros((n, 2))
    good[np.arange(n), y] = 0.9
    good[np.arange(n), 1 - y] = 0.1
    shuf = good[rng.permutation(n)]
    assert M.aurc(good, y) < M.aurc(shuf, y)


# --- ordinal ------------------------------------------------------------

def test_ordinal_mae_penalizes_distance():
    """P0-as-P3 is worse than P0-as-P1; plain accuracy cannot see that."""
    y = np.array([0, 0])
    near = onehot(np.array([1, 1]), 4)
    far = onehot(np.array([3, 3]), 4)
    assert M.accuracy(near, y) == M.accuracy(far, y) == 0.0
    assert M.ordinal_mae(near, y) < M.ordinal_mae(far, y)


def test_qwk_perfect_and_random():
    y = np.array([0, 1, 2, 3] * 25)
    assert M.quadratic_weighted_kappa(onehot(y, 4), y) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    rand = onehot(rng.integers(0, 4, len(y)), 4)
    assert M.quadratic_weighted_kappa(rand, y) < 0.3


# --- the spike 0.1 gate -------------------------------------------------

def test_legal_mass_stats_flags_a_mask_dependent_model():
    """Low legal mass means the grammar chose, not the model, so the confidence is
    an artifact of renormalizing noise. The harness must be able to see it."""
    s = M.legal_mass_stats([0.99, 0.98, 0.3, 0.2])
    assert s["frac_below_0.95"] == pytest.approx(0.5)
    assert s["min"] == pytest.approx(0.2)
    assert M.legal_mass_stats([])["n"] == 0
    assert M.legal_mass_stats([None, 0.99])["n"] == 1


def test_report_includes_every_planned_metric():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 4, 300)
    logits = rng.normal(size=(300, 4))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    r = M.report(p, y, ordinal=True, masses=[0.99] * 300)
    for key in ("accuracy", "macro_f1", "ece", "ece_adaptive", "mce", "brier",
                "nll", "aurc", "selective", "ordinal_mae", "qwk", "legal_mass",
                "mean_confidence"):
        assert key in r, key
