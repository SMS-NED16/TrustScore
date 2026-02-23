"""
Option A: Direct random search over config space.
"""

from typing import List, Tuple, Any, Optional, Callable, Dict
from alignment.config_space import (
    vector_to_config,
    config_to_dict,
    sample_random_config,
)
from config.settings import TrustScoreConfig


def run_option_a(
    evaluate_fn: Callable[[TrustScoreConfig], Tuple[float, float]],
    max_evals: int = 50,
    random_seed: Optional[int] = None,
    on_eval: Optional[Callable[[List[float], Dict, float, float], None]] = None,
) -> Tuple[List[float], Dict[str, Any], float, float, List[Tuple[List[float], float]]]:
    """
    Random search over config vector. evaluate_fn(config) -> (pearson, spearman).
    We maximize Spearman. on_eval(vec, config_dict, pearson, spearman) called after each eval for logging.
    Returns (best_vector, best_config_dict, best_pearson, best_spearman, all_evals).
    """
    best_spearman = float("-inf")
    best_vec = None
    best_dict = None
    best_pearson = float("-inf")
    all_evals: List[Tuple[List[float], float]] = []

    for i in range(max_evals):
        vec = sample_random_config(random_seed if random_seed is not None else i)
        config = vector_to_config(vec)
        pearson, spearman = evaluate_fn(config)
        spearman_val = spearman if spearman == spearman else float("-inf")  # NaN -> -inf
        all_evals.append((list(vec), spearman_val))
        cdict = config_to_dict(config)
        if on_eval:
            on_eval(vec, cdict, pearson, spearman)
        if spearman_val > best_spearman:
            best_spearman = spearman_val
            best_pearson = pearson
            best_vec = list(vec)
            best_dict = cdict

    if best_vec is None:
        best_vec = sample_random_config(0)
        best_dict = config_to_dict(vector_to_config(best_vec))
    return (best_vec, best_dict, best_pearson, best_spearman, all_evals)
