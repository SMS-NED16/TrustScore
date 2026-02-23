"""
Option B: LightGBM meta-model predicts correlation from config vector;
use it to pick best among candidates.
"""

from typing import List, Tuple, Any, Optional, Callable, Dict
from alignment.config_space import (
    vector_to_config,
    config_to_dict,
    sample_random_config,
)
from config.settings import TrustScoreConfig


def run_option_b(
    evaluate_fn: Callable[[TrustScoreConfig], Tuple[float, float]],
    max_evals: int = 50,
    random_seed: Optional[int] = None,
    on_eval: Optional[Callable[[List[float], Dict, float, float], None]] = None,
) -> Tuple[List[float], Dict[str, Any], float, float, Any]:
    """
    Generate configs, evaluate each to get train correlation, train LightGBM to predict
    correlation from config vector; then sample M new configs, predict, pick argmax, evaluate.
    Returns (best_vector, best_config_dict, best_pearson, best_spearman, lgb_model).
    """
    try:
        import lightgbm as lgb
        import numpy as np
    except ImportError:
        raise ImportError("Option B requires lightgbm. pip install lightgbm")

    # Phase 1: random configs and true correlations
    X_list = []
    y_list = []
    best_spearman = float("-inf")
    best_vec = None
    best_dict = None
    best_pearson = float("-inf")

    for i in range(max_evals):
        vec = sample_random_config(random_seed if random_seed is not None else i)
        config = vector_to_config(vec)
        pearson, spearman = evaluate_fn(config)
        spearman_val = spearman if spearman == spearman else float("-inf")
        X_list.append(vec)
        y_list.append(spearman_val)
        cdict = config_to_dict(config)
        if on_eval:
            on_eval(vec, cdict, pearson, spearman)
        if spearman_val > best_spearman:
            best_spearman = spearman_val
            best_pearson = pearson
            best_vec = list(vec)
            best_dict = cdict

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    # Train meta-model
    model = lgb.LGBMRegressor(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=random_seed or 42,
        verbosity=-1,
    )
    model.fit(X, y)

    # Phase 2: sample more configs, predict, take best by prediction, then evaluate
    n_candidates = 100
    candidates = [sample_random_config((random_seed or 42) + 1000 + j) for j in range(n_candidates)]
    X_cand = np.array(candidates, dtype=np.float64)
    pred = model.predict(X_cand)
    best_idx = int(np.argmax(pred))
    vec_cand = list(candidates[best_idx])
    config_cand = vector_to_config(vec_cand)
    pearson_c, spearman_c = evaluate_fn(config_cand)
    if on_eval:
        on_eval(vec_cand, config_to_dict(config_cand), pearson_c, spearman_c)
    spearman_c_val = spearman_c if spearman_c == spearman_c else float("-inf")
    if spearman_c_val > best_spearman:
        best_spearman = spearman_c_val
        best_pearson = pearson_c
        best_vec = vec_cand
        best_dict = config_to_dict(config_cand)

    if best_vec is None:
        best_vec = sample_random_config(0)
        best_dict = config_to_dict(vector_to_config(best_vec))
    return (best_vec, best_dict, best_pearson, best_spearman, model)
