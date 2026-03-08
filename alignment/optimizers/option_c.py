"""
Option C: Continuous black-box optimization using Optuna over the 4-D config vector.
"""

from typing import List, Tuple, Any, Optional, Callable, Dict
from alignment.config_space import (
    vector_to_config,
    config_to_dict,
    CONFIG_VECTOR_NAMES,
    CONFIG_VECTOR_BOUNDS,
)
from config.settings import TrustScoreConfig


def run_option_c(
    evaluate_fn: Callable[[TrustScoreConfig], Tuple[float, float]],
    max_evals: int = 50,
    random_seed: Optional[int] = None,
    on_eval: Optional[Callable[[List[float], Dict, float, float], None]] = None,
) -> Tuple[List[float], Dict[str, Any], float, float, Any]:
    """
    Maximize Spearman over the config space using Optuna.
    evaluate_fn(config) -> (pearson, spearman).
    Returns (best_vector, best_config_dict, best_pearson, best_spearman, study).
    """
    import optuna

    best_spearman = float("-inf")
    best_pearson = float("-inf")
    best_vec = None
    best_dict = None

    def objective(trial: optuna.Trial) -> float:
        vec = [
            trial.suggest_float(name, low=lo, high=hi)
            for name, (lo, hi) in zip(CONFIG_VECTOR_NAMES, CONFIG_VECTOR_BOUNDS)
        ]
        config = vector_to_config(vec)
        pearson, spearman = evaluate_fn(config)
        spearman_val = spearman if spearman == spearman else -1.0
        cdict = config_to_dict(config)
        if on_eval:
            on_eval(vec, cdict, pearson, spearman)
        nonlocal best_spearman, best_pearson, best_vec, best_dict
        if spearman_val > best_spearman:
            best_spearman = spearman_val
            best_pearson = pearson
            best_vec = list(vec)
            best_dict = cdict
        return spearman_val

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=min(10, max(1, max_evals // 5)),
        seed=random_seed,
    )
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=max_evals, show_progress_bar=False)

    # Use Optuna's best trial (single-objective: best_trial, not best_trials)
    best_trial = getattr(study, "best_trial", None)
    if best_trial is not None:
        try:
            best_vec = [study.best_params[name] for name in CONFIG_VECTOR_NAMES]
            config = vector_to_config(best_vec)
            best_pearson, best_spearman = evaluate_fn(config)
            best_dict = config_to_dict(config)
        except Exception:
            # Re-evaluation failed (e.g. cache issue); keep in-objective best if set
            pass
    if best_vec is None:
        best_vec = [0.5, 0.3, 0.5, 0.0]
        best_dict = config_to_dict(vector_to_config(best_vec))
    return (best_vec, best_dict, best_pearson, best_spearman, study)
