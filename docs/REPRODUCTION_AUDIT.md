# Cross-runner reproduction audit

Research commit: `89e4ff95a8bc05305a8a891815eb073fa4303132`.

The GitHub Actions run [35189419485](https://github.com/jingxuxie/multiagent-which-environments/actions/runs/35189419485) completed successfully. It installed the pinned scientific dependencies on a fresh Ubuntu runner, passed all 74 tests, regenerated all experiments and summary tables, verified every reference endpoint, compiled the 19-page manuscript, uploaded the artifact, and published the ordinary source files and generated results to main.

## Independent execution environment, not independent scientific review

Local reference Python: 3.13.5. Runner Python: 3.13.15. Both use NumPy 2.3.5 and SciPy 1.17.0. The runner's scientific experiment loop took 13.888 seconds, excluding dependency installation, plotting, tests, and LaTeX. This is an execution measurement, not a claim about every laptop.

The downloaded artifact SHA-256 matched the GitHub artifact digest:

`22fe5405d0e01489b79a01112121d8893244b446fe59e8245856ebf1d2da044f`

Comparing all 11 scientific CSV files against the original local run checked 38,620 data cells. Nine CSVs are byte-identical. Differences in the other two files are floating-point-scale:

- `interval_checks.csv`: 139 differing text cells; maximum absolute numeric difference approximately 1e-13.
- `structured_checks.csv`: 158 differing text cells; maximum absolute numeric difference approximately 2.36e-16.

All cells passed the 1e-8 absolute/relative comparison. The two local reference executions reproduced all 11 CSVs byte-for-byte. Timing metadata are intentionally not treated as scientific data.

The local and runner manuscript TeX sources are byte-identical. Both PDFs contain 19 pages with identical extracted text page by page. The PDFs themselves need not be byte-identical because of build metadata. The manuscript was rendered and visually inspected locally; representative runner pages were checked again.

## Scope

These are reproducibility and self-consistency checks, not independent validation of novelty, peer review, or formal proof verification. The negative policy-gradient result and simpler-baseline wins remain in the paper. The draft still needs independent scientific review and final venue formatting before submission.
