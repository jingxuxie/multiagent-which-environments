# Research status and claim boundaries

## Completed

A full 19-page first research draft, including proofs, related work, raw synthetic data, code, figures, and negative results. Local validation: 74 passing tests, complete fixed-seed experiment run, numerical table generation, PDF compilation and rendered-page inspection. Eleven scientific CSVs reproduce byte-for-byte locally; machine timings are separate. These checks are self-review, not independent peer review or formal proof verification.

The central contribution candidate is the exact two-learner balanced-coverage characterization in a restricted assignment game, combined with a general curriculum optimizer and a sharp rectangular-uncertainty certificate. The basic graph/margin reformulation, covering perspective, and the use of Hoeffding intervals are explicitly not claimed as independent inventions.

## What is not established

- Independent confirmation of novelty against every machine-teaching and diameter-control formulation.
- Generalization outside the specified policy class or to arbitrary target-only observations.
- Convergence of local self-play optimization. The experiment and proposition show an obstruction instead.
- Optimality of the LP for every finite table; its guarantee is a half approximation.
- A half-approximation hardness result, optimal sample complexity in the general model, or optimal sparse-curriculum support.
- Dominance over simpler curricula on mean cross-play. Inverse-gap weighting wins the strongest search endpoint.
- Competitive performance against full implementations of Other-Play, CEC, COLE, or multi-agent UED.
- Physical robotics, human teamwork, LLM-agent performance, long-horizon or deep-MARL evidence.
- Final ICLR-format compliance, authorship decisions, submission, or acceptance.

## Next scientific priorities

1. Independent theorem and novelty review, especially the relation between balanced target diameter and machine teaching. Look for either a stronger structural separation or a simpler pre-existing theorem that subsumes the result.
2. A second genuinely different sequential cooperative environment. Use a shared finite-state policy class with exact evaluation before adding learning. Do not present nuisance rescaling of the current task as independent generalization evidence.
3. Curriculum design with uncertainty in the diagnostic structure, rather than a known D matrix; distinguish teacher sample cost from learner interaction budget.
4. Test whether an optimization-aware objective can improve the margin/rate mismatch without losing transparent guarantees. The REINFORCE failure should remain a counterexample, not be silently removed.

## Quantitative evidence, not just positive examples

The LP matches the exact oracle in 96/120 random geometries (tolerance 1e-7), has mean ratio 0.967, and has a separately constructed ratio approaching 0.5. The interval rule gives 167/1,200 claims with no observed false claims, versus 952 claims and 211 false claims for plug-in selection. Two thresholds reuse each of 600 datasets, so 1,200 is not an independent-trial count.

For 24-neutral-mode assignment training at 4,096 episodes, coverage-search mean cross-play is 0.97554 with seed-jackknife SE 0.00275 and worst return 0.95. Inverse-gap search gives 1.0. Coverage REINFORCE yields 0.57652, SE 0.04004, worst 0, and only 2/24 theorem-eligible policies. All target values are exact expectations for deployed deterministic conventions, not held-out episode estimates.
