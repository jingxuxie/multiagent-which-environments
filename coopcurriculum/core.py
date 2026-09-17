"""Finite-class cooperative curriculum certificates. All returns are in [0, 1]."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def matrix(a: ArrayLike) -> FloatArray:
    a = np.asarray(a, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("Expected a nonempty finite matrix")
    if np.any(a < -1e-10) or np.any(a > 1 + 1e-10):
        raise ValueError("Returns must be in [0, 1]")
    return np.clip(a, 0, 1)


def simplex(mu: ArrayLike, m: int) -> FloatArray:
    mu = np.asarray(mu, dtype=float)
    if mu.shape != (m,) or not np.isfinite(mu).all():
        raise ValueError("Invalid curriculum shape or entries")
    if np.min(mu) < -1e-8 or abs(mu.sum() - 1) > 1e-7:
        raise ValueError("Curriculum must belong to the simplex")
    mu = np.maximum(mu, 0)
    return mu / mu.sum()


def edges(bad: ArrayLike, k: int) -> NDArray[np.int64]:
    b = np.asarray(bad)
    if b.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if b.ndim != 2 or b.shape[1] != 2 or not np.isfinite(b).all():
        raise ValueError("Bad pairs must have shape (n, 2)")
    if not np.equal(b, np.floor(b)).all() or b.min() < 0 or b.max() >= k:
        raise ValueError("Invalid policy index")
    return np.unique(np.sort(b.astype(np.int64), axis=1), axis=0)


def bad_pairs(target: ArrayLike, threshold: float) -> NDArray[np.int64]:
    t = matrix(target)
    if t.shape[0] != t.shape[1] or not 0 <= threshold <= 1:
        raise ValueError("Square target matrix and threshold in [0,1] required")
    # One bad ordering suffices: the universal deployment guarantee includes both.
    return edges(np.argwhere(t < threshold), t.shape[0])


def gaps(a: ArrayLike, mu: ArrayLike) -> FloatArray:
    a = matrix(a)
    values = simplex(mu, len(a)) @ a
    return values.max() - values


def margin(a: ArrayLike, mu: ArrayLike, bad: ArrayLike) -> float:
    a = matrix(a)
    b = edges(bad, a.shape[1])
    if not len(b):
        return float("inf")
    g = gaps(a, mu)
    return float(np.max(g[b], axis=1).min())


def near_optimal(a: ArrayLike, mu: ArrayLike, eta: float) -> NDArray[np.int64]:
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("eta must be finite and nonnegative")
    return np.flatnonzero(gaps(a, mu) <= eta)


def worst_crossplay(a: ArrayLike, mu: ArrayLike, target: ArrayLike, eta: float) -> float:
    a, target = matrix(a), matrix(target)
    if target.shape != (a.shape[1], a.shape[1]):
        raise ValueError("Target and policy class dimensions differ")
    active = near_optimal(a, mu, eta)
    return float(target[np.ix_(active, active)].min())


def interval_margin(lower: ArrayLike, upper: ArrayLike, mu: ArrayLike,
                    bad: ArrayLike) -> float:
    """Exact infimum of the margin over an independent cell-wise return box."""
    lo, hi = matrix(lower), matrix(upper)
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError("Invalid interval matrix")
    mu = simplex(mu, len(lo))
    b = edges(bad, lo.shape[1])
    if not len(b):
        return float("inf")
    lower_values, upper_values = mu @ lo, mu @ hi
    pair_ceiling = np.min(upper_values[b], axis=1).max()
    return float(max(0., lower_values.max() - pair_ceiling))


def interval_witness(lower: ArrayLike, upper: ArrayLike, mu: ArrayLike,
                     bad: ArrayLike) -> FloatArray:
    """Construct a matrix attaining interval_margin, including loops and zero width."""
    lo, hi = matrix(lower), matrix(upper)
    if lo.shape != hi.shape or np.any(lo > hi):
        raise ValueError("Invalid interval matrix")
    mu = simplex(mu, len(lo))
    b = edges(bad, lo.shape[1])
    if not len(b):
        return lo.copy()
    lv, uv = mu @ lo, mu @ hi
    i, j = b[np.argmax(np.min(uv[b], axis=1))]
    level = min(uv[i], uv[j])
    values = lv.copy()
    values[i], values[j] = max(lv[i], level), max(lv[j], level)
    fraction = np.divide(values - lv, uv - lv, out=np.zeros_like(lv), where=uv > lv)
    return lo + (hi - lo) * np.clip(fraction, 0, 1)


def hoeffding_box(means: ArrayLike, n: ArrayLike, delta: float) -> tuple[FloatArray, FloatArray]:
    """Fixed entry-wise sample counts; no independence across cells is required."""
    means = matrix(means)
    counts = np.broadcast_to(np.asarray(n, dtype=float), means.shape)
    if not 0 < delta < 1 or np.any(counts <= 0) or not np.isfinite(counts).all():
        raise ValueError("Positive counts and delta in (0,1) required")
    radius = np.sqrt(np.log(2 * means.size / delta) / (2 * counts))
    return np.maximum(0, means - radius), np.minimum(1, means + radius)
