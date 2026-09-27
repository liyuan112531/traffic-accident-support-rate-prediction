"""The 11 heterogeneous expert models.

The pool deliberately spans different inductive biases rather than varying a
single model family: a constant baseline, a linear model, three tree ensembles
built on different algorithms, a convex combination of the base models, an
extremely randomised forest, a histogram gradient booster, a kernel machine, and
a prior-data fitted network. Diversity is the point: the meta-learner consumes
the experts' predictions and the disagreement between them, so a pool of
near-identical models would carry almost no information in blocks A and B.

All hyperparameters are the pre-registered fixed values used in the study. They
were fixed before any result was inspected and no hyperparameter search is
performed, which structurally removes tuning leakage. Changing them invalidates
comparability with the manuscript.

Optional dependencies
---------------------
``tabpfn`` requires model weights that are downloaded at first use and are not
distributed with this repository. When the package or its weights are
unavailable the expert is reported as unavailable rather than silently replaced,
and the reported meta width shrinks accordingly.
"""

from .registry import (
    availability,
    C3_BASE_POOL,
    EXPERT_NAMES,
    FROZEN_POOL,
    ExpertSpec,
    build_expert,
    describe_experts,
    enforce_convex_base,
    is_available,
    resolve_pool,
)

__all__ = [
    "availability",
    "C3_BASE_POOL",
    "EXPERT_NAMES",
    "FROZEN_POOL",
    "ExpertSpec",
    "build_expert",
    "describe_experts",
    "enforce_convex_base",
    "is_available",
    "resolve_pool",
]
