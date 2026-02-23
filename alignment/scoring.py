"""
Cache I/O for (LLMRecord, GradedSpans), re-aggregation with a given config,
and correlation computation.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple

from config.settings import TrustScoreConfig
from modules.aggregator import Aggregator
from models.llm_record import LLMRecord, GradedSpans
from datetime import datetime


def _model_dump(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json") if hasattr(obj.model_dump, "__call__") else obj.model_dump()
    return obj.dict()


def _model_validate(cls: type, data: Dict[str, Any]) -> Any:
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)


def _serialize_datetime(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def cache_path(cache_dir: str, task_name: str, sample_id: str) -> str:
    """Path to one sample's cache file (safe filename from sample_id)."""
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sample_id))
    return os.path.join(cache_dir, task_name, f"{safe_id}.json")


def save_to_cache(
    cache_dir: str,
    task_name: str,
    sample_id: str,
    llm_record: LLMRecord,
    graded_spans: GradedSpans,
) -> str:
    """Save (LLMRecord, GradedSpans) to cache. Returns path."""
    path = cache_path(cache_dir, task_name, sample_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec_dict = _model_dump(llm_record)
    spans_dict = _model_dump(graded_spans)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"llm_record": rec_dict, "graded_spans": spans_dict}, f, default=_serialize_datetime)
    return path


def load_from_cache(
    cache_dir: str,
    task_name: str,
    sample_id: str,
) -> Tuple[LLMRecord, GradedSpans]:
    """Load (LLMRecord, GradedSpans) from cache. Raises FileNotFoundError if missing."""
    path = cache_path(cache_dir, task_name, sample_id)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rec = _model_validate(LLMRecord, data["llm_record"])
    spans = _model_validate(GradedSpans, data["graded_spans"])
    return rec, spans


def reaggregate(
    config: TrustScoreConfig,
    llm_record: LLMRecord,
    graded_spans: GradedSpans,
    use_quality: bool = True,
) -> float:
    """
    Re-aggregate with the given config; return trust_quality (or trust_score if not use_quality).
    """
    agg = Aggregator(config)
    out = agg.aggregate(llm_record, graded_spans)
    return out.summary.trust_quality if use_quality else out.summary.trust_score


def score_samples(
    config: TrustScoreConfig,
    sample_ids: List[str],
    cache_dir: str,
    task_name: str,
    use_quality: bool = True,
) -> List[Optional[float]]:
    """
    For each sample_id, load from cache and re-aggregate with config.
    Returns list of trust scores (same order as sample_ids).
    Missing cache entries get None; caller can filter.
    """
    scores: List[Optional[float]] = []
    for sid in sample_ids:
        try:
            rec, spans = load_from_cache(cache_dir, task_name, sid)
            s = reaggregate(config, rec, spans, use_quality=use_quality)
            scores.append(s)
        except FileNotFoundError:
            scores.append(None)
    return scores


def correlation(
    trust_scores: List[float],
    human_scores: List[float],
) -> Tuple[float, float]:
    """
    Pearson and Spearman correlation. Filters out None in trust_scores (and corresponding human).
    Returns (pearson, spearman).
    """
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    pairs = [(t, h) for t, h in zip(trust_scores, human_scores) if t is not None and h is not None]
    if len(pairs) < 2:
        return (float("nan"), float("nan"))
    t, h = zip(*pairs)
    t_arr = np.array(t, dtype=float)
    h_arr = np.array(h, dtype=float)
    try:
        pearson = pearsonr(t_arr, h_arr)[0]
    except Exception:
        pearson = float("nan")
    try:
        spearman = spearmanr(t_arr, h_arr)[0]
    except Exception:
        spearman = float("nan")
    return (float(pearson), float(spearman))


def compute_correlation_for_samples(
    config: TrustScoreConfig,
    samples: List[Dict[str, Any]],
    task: Any,
    cache_dir: str,
    task_name: str,
    use_quality: bool = True,
) -> Tuple[float, float, List[float], List[float]]:
    """
    Get trust scores for samples (by unique_dataset_id), human scores from task,
    then Pearson and Spearman. Returns (pearson, spearman, trust_scores, human_scores).
    """
    ids = [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(samples)]
    human = [task.get_human_score(s) for s in samples]
    trust = score_samples(config, ids, cache_dir, task_name, use_quality=use_quality)
    # Filter to only cached
    valid_trust = []
    valid_human = []
    for t, h in zip(trust, human):
        if t is not None:
            valid_trust.append(t)
            valid_human.append(h)
    if len(valid_trust) < 2:
        return (float("nan"), float("nan"), trust, human)
    p, s = correlation(valid_trust, valid_human)
    return (p, s, trust, human)
