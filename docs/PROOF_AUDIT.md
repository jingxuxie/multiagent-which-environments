# Proof and implementation audit

All proofs are in `paper/main.tex`; no theorem is presented as externally verified. Numerical tests can find contradictions but cannot prove universal statements.

| Claim | Key reasoning | Tests / restrictions |
|---|---|---|
| Pair exclusion | A bad pair survives iff both gaps are at most eta | Equality belongs to the near-optimal set; certification needs strict >. Loops and asymmetric target orderings are covered. |
| Balanced coverage | Delete common error coordinates, then partition the disagreement set | 300 full-pair vs partition identities; nonnegative shared diagnostic penalties; finite convention class. |
| Observability | Every harmful subset must contain a positive-exposure coordinate | Unseen mass <= epsilon iff some positive-margin curriculum exists; no uniform lower bound on tiny penalties. |
| Dilution | A common baseline cancels; all old gaps scale by retained mass | Exact diagnostic example. Enlarging the feasible curriculum library never lowers its optimum. |
| Pair-sum approximation | Nonnegative max <= sum <= 2 max; optimal-reference polytopes cover simplex | 120 oracle comparisons, analytic near-tight family. Factor tight for the surrogate only. |
| Exact MILP | One maximizing anchor; excluded vertices cover bad edges | Big-M=1 from bounded returns; independent vertex-cover enumeration LP tests. HiGHS feasibility tolerance explicitly tightened. |
| Rectangular envelope | Compete the largest pessimistic score against the smaller optimistic score in the worst bad pair | 400 attaining witnesses plus interior checks; sharp only over independent cell intervals, not all additional structural constraints. |
| Adaptive certificates | One simultaneous source/target coverage event controls all selected mixtures | Fixed class and fixed-count cell sampling. Not optional-stopping or growing-class validity. |
| Two-world lower bound | Binary decision data processing and bounded Bernoulli KL | Fixed sample, validity AND completeness. Not a lower bound on agreement via a pre-agreed convention. |
| Local optimization obstruction | Independent-action self-play has a convex quadratic between two boundary local maxima | Does not assert convergence failure for every global or exploratory optimizer. |

## Additional checks

Input validation covers return ranges, simplex weights, square target matrices, malformed bounds, empty bad sets, diagonal bad edges, and zero-width boxes. A batched learner has the same seed-specific law as separate execution. Budget runs share initializations and prefixes. Adding an observationally unconstrained policy invalidates formerly positive finite-class certificates in 100 explicit cases.

## Statistical accounting

Seed jackknifing is used because target cross-play pairs share learned policies. Calibration uses 600 independently seeded datasets and two thresholds per dataset. Binomial sufficient-statistic simulations are not counted as executed training rollouts. Learner episodes are 3,612,672; learned seed/budget/method rows are 2,016, some with matched random streams. No success probability is inferred from treating all such rows as independent.

## Review gaps

No proof assistant, independent researcher, or anonymous reviewer has checked the arguments. Tests do not cover arbitrary real-valued boundary geometries: floating solvers have numerical tolerances, while formal statements use exact inequalities. The nontrivial novelty claim should be reviewed against general machine-teaching and disagreement-diameter results.
