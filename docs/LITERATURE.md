# Primary-literature positioning

Sources below were checked through their primary paper or official proceedings pages during development. Dates and titles refer to the versions cited in the manuscript, not claims about their latest acceptance status.

- Hu et al. (2020), **Other-Play for Zero-Shot Coordination**. https://proceedings.mlr.press/v119/hu20a.html. Known environmental symmetries and specialized conventions. We do not claim to introduce zero-shot coordination.
- Treutlein et al. (2021), **A New Formalism, Method and Open Issues for Zero-Shot Coordination**. https://arxiv.org/abs/2106.06613. Incompatible maximizers and label-free coordination. Important overlap with our motivation.
- Jha et al. (2025), **Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination**. https://arxiv.org/abs/2504.12714. Environment-distribution training. Our candidate distinction is an exact near-optimal-set certificate, not environment diversity itself.
- You et al. (2025), **Automatic Curriculum Design for Zero-Shot Human-AI Coordination**. https://arxiv.org/abs/2503.07275. Environment and co-player curriculum selection. This rules out broad claims that selecting tasks for coordination is new.
- Li et al. (2023), **Tackling Cooperative Incompatibility for Zero-Shot Human-AI Coordination**. https://arxiv.org/abs/2306.03034. COLE and incompatibility graph structure. We do not claim incompatibility graphs as novel.
- Brown and Niekum (2019), **Machine Teaching for Inverse Reinforcement Learning: Algorithms and Applications**. https://arxiv.org/abs/1805.07687. Informative teaching, reward-equivalence classes, covering. Direct novelty risk for our curriculum optimization formulation.
- Tosh and Dasgupta (2017), **Diameter-Based Active Learning**. https://proceedings.mlr.press/v70/tosh17a.html. Version-space disagreement diameter. Our use of pairwise compatibility is closely related rather than entirely separate.
- Hoeffding (1963), **Probability Inequalities for Sums of Bounded Random Variables**. DOI 10.1080/01621459.1963.10500830. Standard fixed-sample concentration; not a contribution here.

The proposed contribution is the specific balanced self-play error-budget formula in a restricted assignment family, its curriculum implications, the sharp factor example, and the closed-form rectangular certificate. Search and self-review do not establish novelty conclusively. No full baseline reproduction of the named deep-MARL methods has been performed.
