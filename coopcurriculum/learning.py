"""Small self-play learners receiving only sampled shared reward, never theta labels."""
from __future__ import annotations
import numpy as np
from scipy.special import expit


def learn_batch(mu, delta, theta, neutral_count, budget, seeds, kind='search'):
    """Train independent seeds. Modes 0..d-1 diagnose one context; later modes
    sample a context uniformly but have no allocation penalty. Source identity
    is not observed by a learner. 'search' explores deterministic policy bits
    jointly during training; 'reinforce' samples agents' actions independently.
    Both deploy deterministic role-conditioned conventions without shared coins.
    """
    delta, theta, mu = map(np.asarray, (delta, theta, mu))
    d, n = len(delta), len(seeds)
    if len(mu) != d + neutral_count or kind not in ('search', 'reinforce'):
        raise ValueError('Invalid library or learner')
    # Every seed has its own generator, so batching does not alter a seed's law.
    rngs = [np.random.default_rng(int(seed)) for seed in seeds]
    # Pre-generate exogenous randomness; identical arrays are reused across curricula.
    init = np.stack([r.normal(0, 1.2, d) for r in rngs])
    randoms = np.stack([r.random((budget, 6)) for r in rngs], axis=1)
    qvalues = np.full((n, d, 2), .5)
    counts = np.ones((n, d, 2))
    logits = init.copy()
    baseline = np.zeros((n, d))
    ids = np.arange(n)
    cdf = np.cumsum(mu); cdf[-1] = 1
    for step, u in enumerate(randoms):
        modes = np.searchsorted(cdf, u[:, 0], side='right')
        context = np.where(modes < d, modes, (u[:, 1] * d).astype(int))
        penalty = np.where(modes < d, delta[context], 0.)
        if kind == 'search':
            values = qvalues[ids, context]
            greedy = (values[:, 1] > values[:, 0]).astype(int)
            ties = values[:, 1] == values[:, 0]
            greedy[ties] = (u[ties, 3] < .5)
            action = np.where(u[:, 2] < .2, (u[:, 3] < .5).astype(int), greedy)
            reward = (u[:, 4] < .8 - penalty * (action != theta[context])).astype(float)
            counts[ids, context, action] += 1
            qvalues[ids, context, action] += (reward - qvalues[ids, context, action]) / counts[ids, context, action]
        else:
            prob = expit(logits[ids, context])
            a0, a1 = (u[:, 2] < prob).astype(int), (u[:, 3] < prob).astype(int)
            reward = ((a0 == a1) & (u[:, 4] < .8 - penalty * (a0 != theta[context]))).astype(float)
            advantage = reward - baseline[ids, context]
            logits[ids, context] += .12 * advantage * (a0 + a1 - 2 * prob)
            logits[ids, context] = np.clip(logits[ids, context], -10, 10)
            baseline[ids, context] += .05 * advantage
    if kind == 'search':
        bits = (qvalues[:, :, 1] > qvalues[:, :, 0]).astype(int)
        # A seeded private default resolves ties; a common external convention would
        # make cross-play trivial but does not guarantee good training reward.
        tied = qvalues[:, :, 1] == qvalues[:, :, 0]
        bits[tied] = (init[tied] > 0)
    else:
        bits = (logits > 0).astype(int)
    exposure = mu[:d] * delta
    gaps = np.abs(bits - theta) @ exposure
    return bits, gaps
