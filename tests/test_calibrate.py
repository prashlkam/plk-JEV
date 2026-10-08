"""Temperature fitting. CPU-only, so it is cheap to pin properly.

`fit_temperature` is a golden-section search rather than a gradient fit, so the
things worth checking are that it finds the known optimum, that it respects the
two guarantees temperature scaling is chosen for, and that it degrades sanely.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from calibrate import fit_temperature, fit_vector_scaling, softmax  # noqa: E402

from jev import metrics as M  # noqa: E402


def synth(n=4000, k=4, true_T=1.0, seed=0, margin=1.5):
    """Logits whose correctly-calibrated temperature is `true_T` by construction."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    z = rng.normal(size=(n, k))
    z[np.arange(n), y] += margin
    p = softmax(z)                       # the calibrated distribution
    # sample labels FROM it so T=1 is genuinely optimal for z
    y = np.array([rng.choice(k, p=row) for row in p])
    return z * true_T, y                 # scaling z by T makes T the optimum


def test_recovers_a_known_temperature():
    for true_T in (0.5, 1.0, 2.0, 4.0):
        z, y = synth(true_T=true_T, seed=1)
        T, _ = fit_temperature(z, y)
        assert T == pytest.approx(true_T, rel=0.15), f"true {true_T}, fitted {T}"


def test_fitting_reduces_nll_versus_identity():
    z, y = synth(true_T=3.0, seed=2)
    T, nll_T = fit_temperature(z, y)
    nll_1 = M.nll(softmax(z), y)
    assert nll_T <= nll_1 + 1e-9


def test_fitting_reduces_ece_on_an_overconfident_model():
    """The property the whole phase exists for."""
    z, y = synth(true_T=5.0, seed=3)
    T, _ = fit_temperature(z, y)
    assert M.ece(softmax(z / T), y) < M.ece(softmax(z), y)


def test_fitting_never_changes_accuracy():
    z, y = synth(true_T=4.0, seed=4)
    T, _ = fit_temperature(z, y)
    assert M.accuracy(softmax(z / T), y) == M.accuracy(softmax(z), y)


def test_already_calibrated_input_yields_T_near_one():
    z, y = synth(true_T=1.0, seed=5)
    T, _ = fit_temperature(z, y)
    assert 0.85 < T < 1.18, T


def test_search_stays_inside_its_bounds():
    z, y = synth(true_T=1.0, seed=6)
    T, _ = fit_temperature(z, y, lo=0.5, hi=1.5)
    assert 0.5 <= T <= 1.5


def test_handles_a_single_class_present():
    """Degenerate but real: a rare task where `cal` saw one label."""
    z = np.tile([3.0, -1.0, -1.0], (200, 1))
    y = np.zeros(200, dtype=int)
    T, _ = fit_temperature(z, y)
    assert np.isfinite(T) and T > 0


def test_vector_scaling_fits_at_least_as_well_in_sample():
    """More capacity than one T, so in-sample NLL must not be worse. The point of
    the held-out comparison in calibrate.py is that OUT of sample it often is."""
    z, y = synth(true_T=3.0, seed=7)
    T, nll_T = fit_temperature(z, y)
    _, _, nll_v = fit_vector_scaling(z, y)
    assert nll_v <= nll_T + 0.02


def test_softmax_rows_normalize():
    z = np.random.default_rng(8).normal(size=(50, 6)) * 20
    p = softmax(z)
    assert np.allclose(p.sum(1), 1.0)
    assert (p >= 0).all()
