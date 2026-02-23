"""
Config space: map between a 4-D vector and TrustScoreConfig for alignment optimization.
Vector: [w_trustworthiness, w_explainability, sigmoid_steepness, sigmoid_shift]
(w_bias = 1 - w_trustworthiness - w_explainability, clamped).
"""

from typing import Any, Dict, List, Tuple
import copy

from config.settings import (
    TrustScoreConfig,
    AggregationWeights,
    AggregationStrategyConfig,
    DEFAULT_CONFIG,
)


# Bounds for the 5-D vector (used by optimizers)
# Index 0: w_trustworthiness [0, 1]
# Index 1: w_explainability [0, 1]
# Index 2: sigmoid_steepness [0.1, 2.0]
# Index 3: sigmoid_shift [-1, 1]
# (w_bias derived; no index 4)
CONFIG_VECTOR_NAMES = ["w_trustworthiness", "w_explainability", "sigmoid_steepness", "sigmoid_shift"]
CONFIG_VECTOR_BOUNDS: List[Tuple[float, float]] = [
    (0.0, 1.0),
    (0.0, 1.0),
    (0.1, 2.0),
    (-1.0, 1.0),
]
DIM = 4  # 4 free params; w_bias derived


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
    Build TrustScoreConfig from a 4-D vector.
    vec: [w_trustworthiness, w_explainability, sigmoid_steepness, sigmoid_shift]
    """
    if len(vec) < DIM:
        raise ValueError(f"Expected at least {DIM} dimensions, got {len(vec)}")
    w_t, w_e, k, shift = vec[0], vec[1], vec[2], vec[3]
    w_t, w_e, w_b = _clamp_weights(w_t, w_e)
    k = max(0.1, min(2.0, k))
    shift = max(-1.0, min(1.0, shift))

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
    return base


def config_to_vector(config: TrustScoreConfig) -> List[float]:
    """Extract the 4-D alignment vector from a TrustScoreConfig."""
    w = config.aggregation_weights
    s = config.aggregation_strategy
    return [
        w.trustworthiness,
        w.explainability,
        s.sigmoid_steepness,
        s.sigmoid_shift,
    ]


def config_to_dict(config: TrustScoreConfig) -> Dict[str, Any]:
    """Human-readable dict for persistence (aggregation_weights + aggregation_strategy params)."""
    w = config.aggregation_weights
    s = config.aggregation_strategy
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
    }


def sample_random_config(random_state: Any = None) -> List[float]:
    """Sample one random config vector within bounds (for Option A / B)."""
    import random
    if random_state is not None:
        random.seed(random_state)
    vec = []
    for (lo, hi) in CONFIG_VECTOR_BOUNDS:
        vec.append(random.uniform(lo, hi))
    # Normalize first two so they don't exceed 1 when summed
    w_t, w_e = vec[0], vec[1]
    w_t, w_e, _ = _clamp_weights(w_t, w_e)
    vec[0], vec[1] = w_t, w_e
    return vec
