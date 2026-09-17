import itertools
import numpy as np
import pytest
from coopcurriculum.core import (margin, gaps, near_optimal, worst_crossplay,
    bad_pairs, interval_margin, interval_witness, hoeffding_box, simplex, edges)
from coopcurriculum.design import pair_sum_design, exact_design, coverage_design
from coopcurriculum.assignment import (bit_vectors, task_tables, balanced_margin,
    subset_coverage, station_return)


@pytest.mark.parametrize('seed', range(12))
def test_margin_characterization(seed):
    rng = np.random.default_rng(seed)
    a, target = rng.random((4, 6)), rng.random((6, 6))
    np.fill_diagonal(target, 1)
    mu, threshold = rng.dirichlet(np.ones(4)), .6
    b = bad_pairs(target, threshold)
    g = margin(a, mu, b)
    for eta in [0., g, max(0, g-1e-8), g+1e-8, .8]:
        assert (worst_crossplay(a, mu, target, eta) >= threshold) == (g > eta)


@pytest.mark.parametrize('seed', range(10))
def test_lp_factor_and_milp(seed):
    rng = np.random.default_rng(200+seed)
    a = rng.random((3, 5))
    b = edges([(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], 5)
    lp, opt = pair_sum_design(a, b), exact_design(a, b)
    assert lp.margin + 1e-7 >= .5 * opt.margin
    assert lp.margin <= opt.margin + 1e-7
    # Independent coarse enumeration lower bound on the MILP optimum.
    for i in range(6):
        for j in range(6-i):
            mu = np.array([i, j, 5-i-j]) / 5
            assert margin(a, mu, b) <= opt.margin + 1e-7


@pytest.mark.parametrize('small', [.1, .01, .001])
def test_half_factor_is_tight(small):
    a = np.array([[1, 0, 1], [1, .5-small, .5-small]])
    b = [(1, 2)]
    lp, opt = pair_sum_design(a, b), exact_design(a, b)
    assert opt.margin == pytest.approx(1)
    assert lp.margin == pytest.approx(.5+small)
    assert len(near_optimal(a, opt.mu, .1)) == 2  # compatible, nonunique solutions


@pytest.mark.parametrize('seed', range(16))
def test_rectangular_certificate_is_sharp(seed):
    rng = np.random.default_rng(500+seed)
    lower = rng.uniform(0, .7, (4, 5))
    upper = lower + rng.random((4, 5)) * (1-lower)
    mu = rng.dirichlet(np.ones(4))
    b = [(0, 1), (2, 3), (1, 4)] + ([(4, 4)] if seed % 2 else [])
    val = interval_margin(lower, upper, mu, b)
    witness = interval_witness(lower, upper, mu, b)
    assert np.all(witness >= lower-1e-10) and np.all(witness <= upper+1e-10)
    assert margin(witness, mu, b) == pytest.approx(val, abs=1e-10)
    for _ in range(10):
        a = lower + rng.random(lower.shape) * (upper-lower)
        assert margin(a, mu, b) >= val-1e-10


@pytest.mark.parametrize('seed', range(10))
def test_balanced_coverage_against_all_policy_pairs(seed):
    rng = np.random.default_rng(800+seed)
    d = 4 + seed % 2
    diagnostics = rng.dirichlet(np.ones(d), size=5) * rng.uniform(.1, .6, (5, 1))
    q, mu, eps = rng.dirichlet(np.ones(d)), rng.dirichlet(np.ones(5)), .217
    a, t = task_tables(diagnostics, q)
    b = bad_pairs(t, 1-eps)
    val = balanced_margin(mu @ diagnostics, q, eps)
    assert margin(a, mu, b) == pytest.approx(val, abs=1e-10)
    cover = subset_coverage(mu @ diagnostics, q, eps)
    assert cover / 2 - 1e-10 <= val <= cover + 1e-10
    structured = coverage_design(diagnostics, q, eps)
    full = pair_sum_design(a, b)
    assert structured.surrogate == pytest.approx(full.surrogate, abs=1e-7)


def test_dilution_and_impossibility():
    diagnostics = np.array([[.2, 0], [0, .2]])
    a, t = task_tables(diagnostics, [.5, .5])
    g = margin(a, [.5, .5], bad_pairs(t, .9))
    for neutral in [1, 8, 18]:
        expanded = np.vstack([a, np.ones((neutral, 4))])
        new = margin(expanded, np.ones(neutral+2)/(neutral+2), bad_pairs(t, .9))
        assert new == pytest.approx(g*2/(neutral+2))
    a, t = task_tables([[.3, 0], [.2, 0]], [.5, .5])
    assert pair_sum_design(a, bad_pairs(t, .9)).margin == pytest.approx(0)


def test_station_implementation_matches_tables():
    theta = np.array([0, 1, 0])
    q, delta = np.array([.2, .3, .5]), np.array([.1, .2, .4])
    a, t = task_tables(np.diag(delta), q, theta)
    bits = bit_vectors(3).astype(int)
    for k, b in enumerate(bits):
        for j in range(3):
            assert station_return(j, b[j], b[j], theta, delta[j], 1.7) == pytest.approx(a[j, k])
        for ell, c in enumerate(bits):
            value = sum(q[j]*station_return(j, b[j], c[j], theta, 0, .6) for j in range(3))
            assert value == pytest.approx(t[k, ell])


def test_edge_cases():
    a = np.array([[.5, .5], [.8, .8]])
    assert margin(a, [.5, .5], []) == float('inf')
    assert pair_sum_design(a, []).margin == float('inf')
    assert exact_design(a, [(0, 0), (1, 1)]).margin == pytest.approx(0)
    assert interval_margin(a, a, [0, 1], [(0, 1)]) == pytest.approx(0)
    assert np.array_equal(interval_witness(a, a, [0, 1], []), a)
    assert edges([(1, 0), (0, 1)], 2).shape == (1, 2)
    lo, hi = hoeffding_box(a, 10, .05)
    assert np.all(lo <= a) and np.all(hi >= a)
    with pytest.raises(ValueError): simplex([.1, .1], 2)
    with pytest.raises(ValueError): edges([(.5, 1)], 2)
    with pytest.raises(ValueError): margin([[1.5]], [1], [(0, 0)])
    with pytest.raises(ValueError): interval_margin([[.8]], [[.2]], [1], [(0, 0)])

@pytest.mark.parametrize('seed', range(10))
def test_milp_against_independent_vertex_cover_enumeration(seed):
    from scipy.optimize import linprog
    rng=np.random.default_rng(900+seed)
    m,k=3,5
    a=rng.random((m,k))
    bad=[(0,1),(1,2),(2,3),(3,4),(4,0)]
    value=0.
    for mask in range(2**k):
        covered=[bool(mask>>i&1) for i in range(k)]
        if not all(covered[i] or covered[j] for i,j in bad):continue
        for ref in range(k):
            if covered[ref]:continue
            d=a[:,ref,None]-a
            rows=[np.r_[-d[:,i],0] for i in range(k)]
            rows += [np.r_[-d[:,i],1] for i in range(k) if covered[i]]
            r=linprog([0,0,0,-1],A_ub=rows,b_ub=np.zeros(len(rows)),
                      A_eq=[[1,1,1,0]],b_eq=[1],bounds=[(0,1)]*4,method='highs')
            if r.success:value=max(value,r.x[-1])
    assert exact_design(a,bad).margin==pytest.approx(value,abs=1e-8)


def test_finite_class_expansion_breaks_certificate():
    a=np.array([[1,.8],[1,.6]])
    old=margin(a,[.5,.5],[(0,1)])
    expanded=np.column_stack([a,a[:,0]])
    assert old>0 and margin(expanded,[.5,.5],[(0,1),(0,2)])==0


@pytest.mark.parametrize('seed',range(4))
def test_learner_seed_batching(seed):
    from coopcurriculum.learning import learn_batch
    mu=np.array([.4,.6]);delta=np.array([.1,.4]);theta=np.array([0,1])
    for kind in ['search','reinforce']:
        single,g=learn_batch(mu,delta,theta,0,64,[seed],kind)
        batch,gg=learn_batch(mu,delta,theta,0,64,[10,seed,11],kind)
        assert np.array_equal(single[0],batch[1])
        assert g[0]==pytest.approx(gg[1])


@pytest.mark.parametrize('h',[.001,.01,.1,.25])
def test_bernoulli_lower_bound_constant(h):
    p,q=.5-h,.5+h
    kl=p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))
    assert kl<=16*h*h+1e-12


def test_two_local_optima_are_not_optimization_success():
    # Both deterministic conventions are strict local maxima even when one is worse.
    for delta in [.08,.2,.5]:
        r0,r1=.8,.8-delta
        critical=r0/(r0+r1)
        f=lambda p:r0*(1-p)**2+r1*p*p
        assert .5<critical<1
        assert f(0)>f(.001) and f(1)>f(.999)
        assert f(0)>f(1)
