"""
Config space: map between an 18-D vector and TrustScoreConfig for alignment optimization.

Vector layout (18 dimensions):
  [0]  w_trustworthiness          [0, 1]
  [1]  w_explainability           [0, 1]
  [2]  sigmoid_steepness          [0.1, 2.0]
  [3]  sigmoid_shift              [-1, 1]
  [4..17] error-subtype weights   [0, 2] each  (14 weights; see SUBTYPE_PARAMS)

w_bias = 1 - w_trustworthiness - w_explainability, clamped.
Backward compatible: a legacy 4-D vector is accepted and defaults are used for dims 4-17.
"""

from typing import Any, Dict, List, Tuple
import copy

from config.settings import (
    TrustScoreConfig,
    AggregationWeights,
    AggregationStrategyConfig,
    DEFAULT_CONFIG,
)

# ---------------------------------------------------------------------------
# Ordered list of (category, subtype) for the 14 error-subtype weight dims.
# The order here defines the vector indices 4..17.
# ---------------------------------------------------------------------------
SUBTYPE_PARAMS: List[Tuple[str, str]] = [
    ("T", "spelling"),
    ("T", "factual_error"),
    ("T", "hallucination"),
    ("T", "inconsistency"),
    ("B", "demographic_bias"),
    ("B", "cultural_bias"),
    ("B", "gender_bias"),
    ("B", "political_bias"),
    ("B", "sycophancy_bias"),
    ("B", "confirmation_bias"),
    ("E", "unclear_explanation"),
    ("E", "missing_context"),
    ("E", "overly_complex"),
    ("E", "assumption_not_stated"),
]

# Default subtype weights (from DEFAULT_CONFIG.error_subtypes)
_DEFAULT_SUBTYPE_WEIGHTS: List[float] = [
    DEFAULT_CONFIG.get_error_subtype_weight(cat, sub)
    for cat, sub in SUBTYPE_PARAMS
]

# ---------------------------------------------------------------------------
# Full 18-D config vector names and bounds
# ---------------------------------------------------------------------------
_BASE_NAMES = ["w_trustworthiness", "w_explainability", "sigmoid_steepness", "sigmoid_shift"]
_BASE_BOUNDS: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.1, 2.0),
    (-1.0, 1.0),
]

_SUBTYPE_NAMES = [f"{cat}_{sub}" for cat, sub in SUBTYPE_PARAMS]
_SUBTYPE_BOUNDS: List[Tuple[float, float]] = [(0.0, 2.0)] * len(SUBTYPE_PARAMS)

CONFIG_VECTOR_NAMES: List[str] = _BASE_NAMES + _SUBTYPE_NAMES
CONFIG_VECTOR_BOUNDS: List[Tuple[float, float]] = _BASE_BOUNDS + _SUBTYPE_BOUNDS

DIM = len(CONFIG_VECTOR_NAMES)        # 18
DIM_BASE = len(_BASE_NAMES)           # 4 (legacy vector length)
DIM_SUBTYPE = len(SUBTYPE_PARAMS)     # 14


def _clamp_weights(w_t: float, w_e: float) -> Tuple[float, float, float]:
    """Ensure weights are in [0,1] and sum to 1."""
    w_t = max(0.0, min(1.0, w_t))
    w_e = max(0.0, min(1.0, w_e))
    w_b = 1.0 - w_t - w_e
    w_b = max(0.0, min(1.0, w_b))
    total = w_t + w_e + w_b
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (w_t / total, w_e / total, w_b / total)


def vector_to_config(vec: List[float]) -> TrustScoreConfig:
    """
    Build TrustScoreConfig from an 18-D (or legacy 4-D) vector.

    Accepts vectors of length >= 4.  If only 4 elements are provided the
    default subtype weights are used for the remaining 14 dimensions.
    """
    if len(vec) < DIM_BASE:
        raise ValueError(f"Expected at least {DIM_BASE} dimensions, got {len(vec)}")

    w_t, w_e, k, shift = vec[0], vec[1], vec[2], vec[3]
    w_t, w_e, w_b = _clamp_weights(w_t, w_e)
    k = max(0.1, min(2.0, k))
    shift = max(-1.0, min(1.0, shift))

    # Subtype weights: use provided values or fall back to defaults
    subtype_weights: List[float] = []
    for i, (lo, hi) in enumerate(_SUBTYPE_BOUNDS):
        idx = DIM_BASE + i
        if idx < len(vec):
            subtype_weights.append(max(lo, min(hi, vec[idx])))
        else:
            subtype_weights.append(_DEFAULT_SUBTYPE_WEIGHTS[i])

    base = copy.deepcopy(DEFAULT_CONFIG)
    base.aggregation_weights = AggregationWeights(
        trustworthiness=w_t,
        explainability=w_e,
        bias=w_b,
    )
    base.aggregation_strategy = AggregationStrategyConfig(
        aggregation_method="weighted_mean",
        use_quality_scores=True,
        sigmoid_steepness=k,
        sigmoid_shift=shift,
    )

    # Apply subtype weights onto config.error_subtypes
    for (cat, sub), weight in zip(SUBTYPE_PARAMS, subtype_weights):
        if cat in base.error_subtypes and sub in base.error_subtypes[cat]:
            base.error_subtypes[cat][sub]["weight"] = weight

    return base


def config_to_vector(config: TrustScoreConfig) -> List[float]:
    """Extract the full 18-D alignment vector from a TrustScoreConfig."""
    w = config.aggregation_weights
    s = config.aggregation_strategy
    base = [
        w.trustworthiness,
        w.explainability,
        s.sigmoid_steepness,
        s.sigmoid_shift,
    ]
    subtype = [
        config.get_error_subtype_weight(cat, sub)
        for cat, sub in SUBTYPE_PARAMS
    ]
    return base + subtype


def config_to_dict(config: TrustScoreConfig) -> Dict[str, Any]:
    """Human-readable dict for persistence (all tunable params)."""
    w = config.aggregation_weights
    s = config.aggregation_strategy
    subtype_weights = {
        f"{cat}_{sub}": config.get_error_subtype_weight(cat, sub)
        for cat, sub in SUBTYPE_PARAMS
    }
    return {
        "aggregation_weights": {
            "trustworthiness": w.trustworthiness,
            "explainability": w.explainability,
            "bias": w.bias,
        },
        "sigmoid_steepness": s.sigmoid_steepness,
        "sigmoid_shift": s.sigmoid_shift,
        "aggregation_method": s.aggregation_method,
        "use_quality_scores": s.use_quality_scores,
        "error_subtype_weights": subtype_weights,
    }


def sample_random_config(random_state: Any = None) -> List[float]:
    """Sample one random config vector within bounds (for Option A / B)."""
    import random
    if random_state is not None:
        random.seed(random_state)
    vec = []
    for lo, hi in CONFIG_VECTOR_BOUNDS:
        vec.append(random.uniform(lo, hi))
    # Normalize first two so they don't exceed 1 when summed
    w_t, w_e = vec[0], vec[1]
    w_t, w_e, _ = _clamp_weights(w_t, w_e)
    vec[0], vec[1] = w_t, w_e
    return vec
