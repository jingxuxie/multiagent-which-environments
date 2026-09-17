"""LP approximation and exact small-instance MILP for diagnostic curriculum design."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import warnings
from numpy.typing import ArrayLike
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from .core import matrix, simplex, edges, margin


@dataclass
class Design:
    mu: np.ndarray
    margin: float
    surrogate: float
    reference: int | None


def pair_sum_design(a: ArrayLike, bad: ArrayLike) -> Design:
    """Enumerate optimal references; maximize the smallest sum of bad-pair gaps.

    Returns Gamma(mu) >= Gamma* / 2, for a nonempty fixed bad-pair set.
    Runtime is polynomial in the explicitly enumerated policy class, not in a
    compact representation of exponentially many policies.
    """
    a = matrix(a)
    m, k = a.shape
    b = edges(bad, k)
    if not len(b):
        return Design(np.full(m, 1 / m), float('inf'), float('inf'), None)
    best = None
    for ref in range(k):
        d = a[:, ref, None] - a
        ub = [np.r_[-d[:, i], 0.] for i in range(k)]
        ub += [np.r_[-d[:, i] - d[:, j], 1.] for i, j in b]
        res = linprog(np.r_[np.zeros(m), -1.], A_ub=np.array(ub),
                      b_ub=np.zeros(len(ub)), A_eq=[np.r_[np.ones(m), 0.]],
                      b_eq=[1.], bounds=[(0, 1)] * m + [(0, 2)], method='highs')
        if not res.success:
            if res.status == 2:  # this policy cannot be an optimal reference
                continue
            raise RuntimeError(f"Reference LP failed: {res.message}")
        mu = simplex(res.x[:m], m)
        candidate = Design(mu, margin(a, mu, b), float(res.x[-1]), ref)
        if best is None or candidate.surrogate > best.surrogate + 1e-9:
            best = candidate
    if best is None:
        raise RuntimeError("No optimal reference was found")
    return best


def exact_design(a: ArrayLike, bad: ArrayLike) -> Design:
    """Exact bounded MILP oracle: anchor binaries enforce the maximum score;
    exclusion binaries form a vertex cover of the bad-pair graph.
    """
    a = matrix(a)
    m, k = a.shape
    b = edges(bad, k)
    if not len(b):
        return Design(np.full(m, 1 / m), float('inf'), float('inf'), None)
    # x = [mu_m, maximum_score, margin, excluded_k, anchor_k]
    size, sidx, tidx, yi, zi = m + 2 + 2 * k, m, m + 1, m + 2, m + 2 + k
    rows, lower, upper = [], [], []
    def add(coeff, low=-np.inf, high=np.inf):
        rows.append(coeff); lower.append(low); upper.append(high)
    row = np.zeros(size); row[:m] = 1; add(row, 1, 1)
    row = np.zeros(size); row[zi:] = 1; add(row, 1, 1)
    for i in range(k):
        row = np.zeros(size); row[sidx] = 1; row[:m] = -a[:, i]
        add(row, 0, np.inf)  # maximum >= score_i
        row = row.copy(); row[zi+i] = 1
        add(row, -np.inf, 1)  # maximum <= score_i if anchor_i=1
        row = np.zeros(size); row[tidx] = 1; row[sidx] = -1
        row[:m] = a[:, i]; row[yi+i] = 1
        add(row, -np.inf, 1)  # margin <= gap_i if excluded_i=1
    for i, j in b:
        row = np.zeros(size); row[yi+i] += 1; row[yi+j] += 1
        add(row, 1, np.inf)
    objective = np.zeros(size); objective[tidx] = -1
    integrality = np.zeros(size); integrality[yi:] = 1
    with warnings.catch_warnings():
        # SciPy forwards this documented HiGHS option to its backend.
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=RuntimeWarning)
        res = milp(objective, integrality=integrality, bounds=Bounds(0, 1),
                   constraints=LinearConstraint(np.array(rows), lower, upper),
                   options={'mip_rel_gap': 1e-10, 'mip_feasibility_tolerance': 1e-9,
                            'time_limit': 60.})
    if not res.success:
        raise RuntimeError(f"Exact MILP did not certify optimality: {res.message}")
    mu = simplex(res.x[:m], m)
    actual = margin(a, mu, b)
    if abs(actual - res.x[tidx]) > 1e-6:
        raise RuntimeError("MILP objective disagrees with direct certificate")
    return Design(mu, actual, actual, int(np.argmax(res.x[zi:])))


def coverage_design(diagnostics: ArrayLike, target_weights: ArrayLike,
                    epsilon: float) -> Design:
    """Structured pair-sum LP, using bad coordinate sets rather than policy pairs."""
    from .assignment import bit_vectors, balanced_margin
    dmat = matrix(diagnostics)
    m, d = dmat.shape
    q = simplex(target_weights, d)
    if not 0 <= epsilon < 1:
        raise ValueError("epsilon must be in [0,1)")
    sets = bit_vectors(d)
    badsets = sets[sets @ q > epsilon + 1e-12]
    coeff = dmat @ badsets.T
    ub = np.column_stack([-coeff.T, np.ones(len(badsets))])
    res = linprog(np.r_[np.zeros(m), -1.], A_ub=ub,
                  b_ub=np.zeros(len(ub)), A_eq=[np.r_[np.ones(m), 0.]],
                  b_eq=[1.], bounds=[(0, 1)] * m + [(0, 1)], method='highs')
    if not res.success:
        raise RuntimeError(res.message)
    mu = simplex(res.x[:m], m)
    g = balanced_margin(mu @ dmat, q, epsilon)
    return Design(mu, g, float(res.x[-1]), 0)
