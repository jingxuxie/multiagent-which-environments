"""Observable context, two roles, two stations, and fixed completion-cost reward.

A policy is a binary convention for each public context. Role 0 takes station b;
role 1 takes station 1-b. A mismatched pair collides exactly where their bits differ.
Training interventions alter unobserved station/role processing costs, not the
reward formula or observable source label. Target station costs are balanced.
"""
from __future__ import annotations
from functools import lru_cache
import numpy as np
from numpy.typing import ArrayLike
from .core import matrix, simplex


@lru_cache(maxsize=20)
def bit_vectors(d: int) -> np.ndarray:
    if not isinstance(d, int) or not 1 <= d <= 16:
        raise ValueError("Use between 1 and 16 explicitly enumerated bits")
    out = ((np.arange(2**d)[:, None] >> np.arange(d)) & 1).astype(float)
    out.setflags(write=False)
    return out


def task_tables(diagnostics: ArrayLike, q: ArrayLike,
                theta: ArrayLike | None = None) -> tuple[np.ndarray, np.ndarray]:
    dmat = matrix(diagnostics)
    _, d = dmat.shape
    if np.any(dmat.sum(axis=1) > 1 + 1e-10):
        raise ValueError("Total completion penalty cannot exceed 1")
    q = simplex(q, d)
    bits = bit_vectors(d)
    theta = np.zeros(d) if theta is None else np.asarray(theta)
    if theta.shape != (d,) or not np.isin(theta, [0, 1]).all():
        raise ValueError("theta must be a binary vector")
    wrong = np.abs(bits - theta)
    a = 1 - dmat @ wrong.T
    t = 1 - np.einsum('ijk,k->ij', np.abs(bits[:, None, :] - bits[None, :, :]), q)
    return a, t


def balanced_margin(exposure: ArrayLike, q: ArrayLike, epsilon: float) -> float:
    """Exact min over harmful coordinate sets and their two-way partitions.

    Uses O(3**d) arithmetic; this is a small-instance oracle, not a scalable claim.
    """
    a = np.asarray(exposure, dtype=float)
    d = len(a)
    q = simplex(q, d)
    if a.shape != (d,) or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError("Nonnegative finite exposure is required")
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0,1)")
    bits = bit_vectors(d)
    costs, mass = bits @ a, bits @ q
    result = float('inf')
    for u in np.flatnonzero(mass > epsilon + 1e-12):
        v = int(u)
        while True:
            result = min(result, max(costs[v], costs[int(u) ^ v]))
            if v == 0:
                break
            v = (v - 1) & int(u)
    return float(result)


def subset_coverage(exposure: ArrayLike, q: ArrayLike, epsilon: float) -> float:
    a = np.asarray(exposure, dtype=float)
    b = bit_vectors(len(a))
    q = simplex(q, len(a))
    return float(np.min((b @ a)[b @ q > epsilon + 1e-12]))


def station_return(context: int, bit0: int, bit1: int, theta: np.ndarray,
                   penalty: float, scale: float = 1.) -> float:
    """Physical completion cost: total baseline cost 2s, extra 2s*penalty if wrong.
    Shared reward is 2 - total_cost/(2s); a collision has reward zero.
    Changing scale changes physical task features but leaves reward differences fixed.
    """
    if bit0 not in (0, 1) or bit1 not in (0, 1) or not 0 <= penalty <= 1 or scale <= 0:
        raise ValueError("Invalid station task")
    station0, station1 = bit0, 1 - bit1
    if station0 == station1:
        return 0.
    total_cost = 2 * scale * (1 + penalty * (bit0 != theta[context]))
    return float(2 - total_cost / (2 * scale))
