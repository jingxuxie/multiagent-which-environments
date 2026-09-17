# Which Environments Teach Cooperation?
## Diagnostic Exposure for Compatible Self-Play

A theory-first, synthetic, CPU-only research project. No LLM API, GPU, external dataset, human data, or robot is required.

**Status:** complete first research draft with self-checked proofs and reproducible experiments, not independently reviewed or yet recommended for submission unchanged. The manuscript is in a readable research format, not an asserted final ICLR template.

[Read the manuscript](paper/main.pdf) · [LaTeX source](paper/main.tex) · [Research status](docs/RESEARCH_STATUS.md) · [Proof audit](docs/PROOF_AUDIT.md)

### Question

Which distributions over training environments make **all approximately optimal self-play policies mutually compatible** on a specified target task family? Environment count is insufficient: informative modes must receive enough exposure to separate harmful conventions at the learner's optimization accuracy.

### Results established in this draft

1. An exact **balanced diagnostic-coverage formula** for a restricted two-role assignment game. A harmful disagreement can be split between two learners' self-play error budgets. It implies an observability obstruction, useful compatible ambiguity, and an exact dilution law.
2. A finite-class **pair-sum LP optimizer** attaining at least half the optimal compatibility margin, with a sharp example for this surrogate and an independent exact MILP oracle. This is not an inapproximability result for every algorithm.
3. A **sharp rectangular-interval certificate** with an explicit attaining return table. A simultaneous fixed-data confidence event permits adaptive curriculum selection without a union bound over curricula. A restricted two-world calibration lower bound complements it.
4. A local-optimization obstruction: curriculum information and a good global self-play objective do not guarantee that independent-action policy gradient reaches compatible solutions.

The basic margin characterization is elementary. The paper explicitly acknowledges connections to zero-shot coordination, machine teaching, version-space diameter, and environment-curriculum methods.

### Completed validation

- **74 passing tests**, including an independently implemented vertex-cover oracle.
- 120 random finite geometries; 300 structured identities; 400 constructive interval witnesses; 100 policy-class expansion failures.
- 600 calibration datasets and **1,200 certificate decisions** (two dependent thresholds per dataset). Plug-in design issued 952 claims, 211 false; the interval rule issued 167 claims, none false in these trials. The zero count is not the proof of validity.
- **2,016 learned-policy runs**, 24 independent seeds per setting, **3,612,672 executed training episodes**. Pairwise uncertainty uses a seed jackknife rather than treating cross-play pairs as independent.
- All **11 scientific CSV files** reproduced byte-for-byte in two local runs. Runtime and platform metadata are intentionally separate.

### Important mixed results

At 4,096 training episodes, with 24 neutral modes, deterministic convention search obtains mean cross-play **0.835** under uniform training and **0.976** under coverage weighting. However, the simpler inverse-gap baseline reaches **1.000**. Coverage weighting maximizes a robust tolerance objective, not mean agreement or finite-time learning speed. It deliberately permits two compatible conventions; all 24 search seeds qualify for the theorem and their worst pair returns 0.95.

Independent-action REINFORCE under the same coverage curriculum reaches mean cross-play only **0.577**, with worst pair 0 and just **2/24** policies satisfying the chosen true-gap threshold. This negative result is retained in the paper.

### Reproduce

Python 3.11+; reference Python 3.13.5. Scientific dependencies are pinned in `pyproject.toml`.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/run_all.py
python experiments/report.py
python experiments/verify_reference.py
bash scripts/build_paper.sh
```

The last step requires a standard LaTeX installation with `pdflatex` and `bibtex` (or `bibtex.original`). Ubuntu packages: `texlive-latex-base texlive-latex-extra texlive-fonts-recommended lmodern`.

The numerical experiment suite took about ten seconds in the reference container, excluding installation, tests, plotting, and LaTeX. Runtime on other CPUs is not guaranteed. Raw scientific data are in `results/*.csv`; `results/manifest.json` records their hashes; `results/summary.json` and the manuscript tables are generated from those files. `scripts/compare_results.py` checks complete numerical equivalence across runs.

### Scope and limitations

The principal guarantees assume a specified finite policy class, a stationary curriculum, a given or bounded target compatibility relation, and globally near-optimal final self-play policies. The structured theory additionally assumes shared convention features and unobserved source-mode identity. The learned experiment is a one-step contextual assignment game, not long-horizon deep MARL, physical simulation, or a large-agent benchmark. The teacher knows the diagnostic structure; teacher computation is not counted as learner interactions. Baselines are controlled mechanisms, not reproductions of CEC, Other-Play, COLE, or UED.

A second meaningfully different cooperative task, policy-class generalization, and independent proof/novelty review are the highest-priority remaining work. See the status document for the precise claim boundaries.

### Repository layout

`coopcurriculum/` contains validated objectives, optimizers, the assignment game, and learners. `experiments/` regenerates raw results, figures, tables, and summary checks. `tests/` contains theory and implementation checks. `paper/` contains the manuscript and bibliography; `docs/` contains audits. GitHub Actions reproduces the package and builds the PDF.

No software or manuscript license has been selected on the owner's behalf.
