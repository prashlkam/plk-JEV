"""Calibration and decision-quality metrics.

These are easy to get subtly wrong, so every definition used here is stated
explicitly in its docstring. All functions take `probs` as an (N, K) array over the
K *label* classes (abstain excluded — abstain is a decision, not a class) and
`labels` as (N,) integer indices into that same axis.

Nothing here imports torch; the eval harness scores an Ollama HTTP endpoint, and the
training loop reuses the same code on numpy arrays so that dev-set ECE computed
during training is the identical quantity reported at the end.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, K), got {probs.shape}")
    if len(probs) != len(labels):
        raise ValueError(f"{len(probs)} probs vs {len(labels)} labels")
    if len(probs) and (labels.min() < 0 or labels.max() >= probs.shape[1]):
        raise ValueError("labels outside the class axis")
    s = probs.sum(axis=1)
    if len(probs) and not np.allclose(s, 1.0, atol=1e-3):
        raise ValueError(f"probs must be normalized; sums span {s.min()}..{s.max()}")
    return probs, labels


def accuracy(probs, labels) -> float:
    probs, labels = _check(probs, labels)
    return float((probs.argmax(1) == labels).mean()) if len(labels) else float("nan")


def macro_f1(probs, labels) -> float:
    """Unweighted mean per-class F1. Rare classes count as much as common ones."""
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    pred = probs.argmax(1)
    f1s = []
    for k in range(probs.shape[1]):
        tp = np.sum((pred == k) & (labels == k))
        fp = np.sum((pred == k) & (labels != k))
        fn = np.sum((pred != k) & (labels == k))
        if tp + fp + fn == 0:
            continue  # class absent from both -> undefined, omit
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def nll(probs, labels) -> float:
    """Mean negative log likelihood of the true class."""
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    return float(-np.log(probs[np.arange(len(labels)), labels] + EPS).mean())


def brier(probs, labels) -> float:
    """Multiclass Brier score, sum form: mean_i sum_k (p_ik - y_ik)^2.

    Range [0, 2]. Lower is better. Decomposes into calibration + refinement, so a
    single number that punishes both overconfidence and poor discrimination. Note
    some papers divide by K or by 2 — this does neither, so compare only against
    numbers computed the same way.
    """
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ece(probs, labels, n_bins: int = 15) -> float:
    """Expected Calibration Error, top-label, equal-width bins on (1/K, 1].

    ECE = sum_b (n_b / N) * |acc(b) - conf(b)|, where conf is the mean predicted
    top-class probability in bin b and acc is the empirical frequency of that
    prediction being right.

    Equal-width binning is the standard (Guo et al. 2017) and is what the 15-bin
    figure in PLAN.md 7 refers to. It is biased downward when confidences cluster;
    `ece_adaptive` is reported alongside it for exactly that reason.
    """
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    lo = 1.0 / probs.shape[1]
    edges = np.linspace(lo, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if i == 0:
            m |= conf <= edges[0]
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ece_adaptive(probs, labels, n_bins: int = 15) -> float:
    """ECE with equal-*mass* bins (quantile edges).

    Reported next to `ece` because equal-width ECE can look deceptively small when
    a model's confidences are bunched into one or two bins — which is exactly the
    failure mode of the stock 0.6B baseline measured in spikes/RESULTS.md, whose
    confidences all landed in 0.57-0.71.
    """
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    conf = probs.max(1)
    correct = probs.argmax(1) == labels
    qs = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    total = 0.0
    for i in range(n_bins):
        m = (conf > qs[i]) & (conf <= qs[i + 1])
        if not m.any():
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def mce(probs, labels, n_bins: int = 15) -> float:
    """Maximum Calibration Error — the worst single bin. Worst-case guarantee."""
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    conf, correct = probs.max(1), probs.argmax(1) == labels
    lo = 1.0 / probs.shape[1]
    edges = np.linspace(lo, 1.0, n_bins + 1)
    worst = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() < 10:  # ignore bins too small to estimate
            continue
        worst = max(worst, abs(correct[m].mean() - conf[m].mean()))
    return float(worst)


def reliability(probs, labels, n_bins: int = 15) -> dict:
    """Per-bin data for a reliability diagram."""
    probs, labels = _check(probs, labels)
    conf, correct = probs.max(1), probs.argmax(1) == labels
    lo = 1.0 / probs.shape[1]
    edges = np.linspace(lo, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if i == 0:
            m |= conf <= edges[0]
        rows.append({
            "bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]),
            "n": int(m.sum()),
            "mean_conf": float(conf[m].mean()) if m.any() else None,
            "accuracy": float(correct[m].mean()) if m.any() else None,
        })
    return {"bins": rows, "n_bins": n_bins, "lo": float(lo)}


def selective_accuracy(probs, labels, coverages=(0.5, 0.8, 0.95, 1.0)) -> dict:
    """Accuracy when you auto-act on the most-confident fraction and defer the rest.

    The metric that actually matters in production: a decision model earns its keep
    by knowing when not to be trusted. Ties at the threshold are included, so
    realized coverage can exceed the requested value.
    """
    probs, labels = _check(probs, labels)
    if not len(labels):
        return {str(c): float("nan") for c in coverages}
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(np.float64)
    order = np.argsort(-conf, kind="stable")
    out = {}
    for c in coverages:
        k = max(1, int(round(c * len(labels))))
        idx = order[:k]
        out[str(c)] = {
            "coverage": float(k / len(labels)),
            "accuracy": float(correct[idx].mean()),
            "threshold": float(conf[idx][-1]),
        }
    return out


def aurc(probs, labels) -> float:
    """Area Under the Risk-Coverage curve. Lower is better.

    Threshold-free summary of how well confidence ranks correctness — insensitive
    to calibration, sensitive to ordering. Pairs with ECE: a model can have great
    AURC and terrible ECE (right ranking, wrong numbers) and temperature scaling
    fixes the second without touching the first.
    """
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    conf = probs.max(1)
    err = (probs.argmax(1) != labels).astype(np.float64)
    order = np.argsort(-conf, kind="stable")
    risks = np.cumsum(err[order]) / np.arange(1, len(labels) + 1)
    return float(risks.mean())


# --- ordinal tasks -------------------------------------------------------

def ordinal_mae(probs, labels) -> float:
    """Mean |predicted code - true code|. For triage_v1 / urgency_v1.

    Plain accuracy treats P0-as-P3 and P0-as-P1 as equally wrong; they are not.
    """
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    return float(np.abs(probs.argmax(1) - labels).mean())


def quadratic_weighted_kappa(probs, labels) -> float:
    """QWK: agreement above chance, penalizing errors by squared code distance."""
    probs, labels = _check(probs, labels)
    if not len(labels):
        return float("nan")
    k = probs.shape[1]
    pred = probs.argmax(1)
    O = np.zeros((k, k))
    for a, b in zip(labels, pred):
        O[a, b] += 1
    w = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / (k - 1) ** 2
    hist_t = np.bincount(labels, minlength=k)
    hist_p = np.bincount(pred, minlength=k)
    E = np.outer(hist_t, hist_p) / len(labels)
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > EPS else float("nan")


# --- the grammar-mask gate ----------------------------------------------

def legal_mass_stats(masses) -> dict:
    """Summarize the pre-mask probability that landed on the legal label set.

    The gate metric from spike 0.1. Logprobs from Ollama are PRE-grammar-mask, so
    confidence is only trustworthy when the model was already going to emit a legal
    token. Low legal mass means the grammar, not the model, chose the answer —
    and the reported confidence is then an artifact of renormalizing noise.

    A model can post a beautiful ECE while being untrustworthy if this is low, so
    the harness treats <0.95 mean as a hard failure regardless of other metrics.
    """
    m = np.asarray([x for x in masses if x is not None], dtype=np.float64)
    if not len(m):
        return {"n": 0}
    return {
        "n": int(len(m)),
        "mean": float(m.mean()),
        "p05": float(np.quantile(m, 0.05)),
        "min": float(m.min()),
        "frac_below_0.95": float((m < 0.95).mean()),
        "frac_below_0.50": float((m < 0.50).mean()),
    }


def report(probs, labels, *, ordinal: bool = False, n_bins: int = 15,
           masses=None) -> dict:
    """Everything in PLAN.md 7's table for one task."""
    out = {
        "n": int(len(labels)),
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "ece": ece(probs, labels, n_bins),
        "ece_adaptive": ece_adaptive(probs, labels, n_bins),
        "mce": mce(probs, labels, n_bins),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "aurc": aurc(probs, labels),
        "selective": selective_accuracy(probs, labels),
        "mean_confidence": float(np.asarray(probs).max(1).mean()) if len(labels) else float("nan"),
    }
    if ordinal:
        out["ordinal_mae"] = ordinal_mae(probs, labels)
        out["qwk"] = quadratic_weighted_kappa(probs, labels)
    if masses is not None:
        out["legal_mass"] = legal_mass_stats(masses)
    return out
