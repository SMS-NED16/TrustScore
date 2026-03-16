"""
Evaluate best configs on train, val, test and produce correlation table and report.
"""

import json
import os
from typing import List, Dict, Any, Optional

import numpy as np

from alignment.config_space import vector_to_config, config_to_dict, SUBTYPE_PARAMS
from alignment.scoring import compute_correlation_for_samples, score_samples_full
from config.settings import TrustScoreConfig


def evaluate_config(
    config: TrustScoreConfig,
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    task: Any,
    cache_dir: str,
    task_name: str,
) -> Dict[str, Any]:
    """
    Compute Pearson, Spearman, and Kendall for this config on train, val, test.
    """
    results = {}
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        if not samples:
            results[f"{split_name}_pearson"] = None
            results[f"{split_name}_spearman"] = None
            results[f"{split_name}_kendall"] = None
            continue
        p, s, k, _, _ = compute_correlation_for_samples(
            config, samples, task, cache_dir, task_name, use_quality=True
        )
        results[f"{split_name}_pearson"] = p
        results[f"{split_name}_spearman"] = s
        results[f"{split_name}_kendall"] = k
    return results


def compute_category_statistics(
    config: TrustScoreConfig,
    samples: List[Dict[str, Any]],
    cache_dir: str,
    task_name: str,
) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Compute mean, median, and std for T/E/B/aggregated scores across samples.
    Returns {"T": {"mean": ..., "median": ..., "std": ...}, "E": ..., "B": ..., "aggregated": ...}.
    """
    ids = [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(samples)]
    full_scores = score_samples_full(config, ids, cache_dir, task_name)

    valid = [fs for fs in full_scores if fs is not None]
    if not valid:
        empty = {"mean": None, "median": None, "std": None}
        return {"T": dict(empty), "E": dict(empty), "B": dict(empty), "aggregated": dict(empty)}

    stats = {}
    for key in ["T", "E", "B", "aggregated"]:
        values = np.array([v[key] for v in valid], dtype=float)
        stats[key] = {
            "mean": round(float(np.mean(values)), 4),
            "median": round(float(np.median(values)), 4),
            "std": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 4),
        }
    return stats


def compute_per_sample_scores(
    config: TrustScoreConfig,
    samples: List[Dict[str, Any]],
    task: Any,
    cache_dir: str,
    task_name: str,
) -> List[Dict[str, Any]]:
    """
    Return per-sample score dicts with sample_id, T, E, B, aggregated, and human_score.
    """
    ids = [s.get("unique_dataset_id") or s.get("sample_id", str(i)) for i, s in enumerate(samples)]
    full_scores = score_samples_full(config, ids, cache_dir, task_name)

    results = []
    for sample, sid, fs in zip(samples, ids, full_scores):
        entry = {"sample_id": sid}
        if fs is not None:
            entry.update(fs)
        else:
            entry.update({"T": None, "E": None, "B": None, "aggregated": None})
        try:
            entry["human_score"] = task.get_human_score(sample)
        except Exception:
            entry["human_score"] = None
        results.append(entry)
    return results


def evaluate_best_configs(
    best_configs: Dict[str, Dict[str, Any]],
    train_samples: List[Dict[str, Any]],
    val_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    task: Any,
    cache_dir: str,
    task_name: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate each method's best config on train/val/test splits.
    Returns correlations (Pearson, Spearman, Kendall) and category statistics per method.
    """
    all_samples = train_samples + val_samples + test_samples
    alignment_results = {}
    for method, data in best_configs.items():
        vec = data.get("config_vector")
        if not vec:
            continue
        config = vector_to_config(vec)
        result = evaluate_config(
            config, train_samples, val_samples, test_samples, task, cache_dir, task_name
        )
        result["category_stats"] = compute_category_statistics(
            config, all_samples, cache_dir, task_name
        )
        alignment_results[method] = result
    return alignment_results


def _fmt(x: Any) -> str:
    return f"{x:.4f}" if x is not None and x == x else "\u2014"


def _fmt_stat(stats: Optional[Dict[str, Optional[float]]]) -> str:
    if stats is None or stats.get("mean") is None:
        return "\u2014"
    mean = stats["mean"]
    std = stats.get("std", 0)
    return f"{mean:.1f} \u00b1 {std:.1f}"


def write_report(
    results_dir: str,
    run_id: str,
    task_name: str,
    n_train: int,
    n_val: int,
    n_test: int,
    best_configs: Dict[str, Dict[str, Any]],
    alignment_results: Dict[str, Dict[str, Any]],
) -> str:
    """Write alignment_report.md; return path."""
    path = os.path.join(results_dir, run_id, "alignment_report.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [
        "# Alignment Report",
        "",
        f"- **Task:** {task_name}",
        f"- **Run ID:** {run_id}",
        f"- **Splits:** train={n_train}, val={n_val}, test={n_test}",
        f"- **Config dimensions:** {4 + len(SUBTYPE_PARAMS)} (4 base + {len(SUBTYPE_PARAMS)} subtype weights)",
        "",
        "## Best configs per method",
        "",
    ]
    for method, data in best_configs.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data.get("config_dict", data), indent=2))
        lines.append("```")
        lines.append("")

    # --- Correlation table (Pearson / Spearman / Kendall) ---
    lines.append("## Correlation (Pearson / Spearman / Kendall)")
    lines.append("")
    lines.append(
        "| Method | Train Pearson | Train Spearman | Train Kendall "
        "| Val Pearson | Val Spearman | Val Kendall "
        "| Test Pearson | Test Spearman | Test Kendall |"
    )
    lines.append(
        "|--------|---------------|----------------|---------------"
        "|-------------|--------------|-------------"
        "|--------------|---------------|--------------|"
    )
    for method, res in alignment_results.items():
        tp = res.get("train_pearson")
        ts = res.get("train_spearman")
        tk = res.get("train_kendall")
        vp = res.get("val_pearson")
        vs = res.get("val_spearman")
        vk = res.get("val_kendall")
        ep = res.get("test_pearson")
        es = res.get("test_spearman")
        ek = res.get("test_kendall")
        lines.append(
            f"| {method} | {_fmt(tp)} | {_fmt(ts)} | {_fmt(tk)} "
            f"| {_fmt(vp)} | {_fmt(vs)} | {_fmt(vk)} "
            f"| {_fmt(ep)} | {_fmt(es)} | {_fmt(ek)} |"
        )
    lines.append("")

    # --- Category score statistics (mean +/- std) ---
    lines.append("## Category Score Statistics (mean \u00b1 std, across all samples)")
    lines.append("")
    lines.append("| Method | T Score | E Score | B Score | Aggregated TEBScore |")
    lines.append("|--------|---------|---------|---------|---------------------|")
    for method, res in alignment_results.items():
        cs = res.get("category_stats", {})
        lines.append(
            f"| {method} "
            f"| {_fmt_stat(cs.get('T'))} "
            f"| {_fmt_stat(cs.get('E'))} "
            f"| {_fmt_stat(cs.get('B'))} "
            f"| {_fmt_stat(cs.get('aggregated'))} |"
        )
    lines.append("")

    # --- Error subtype weight comparison table ---
    methods = list(best_configs.keys())
    subtype_keys = [f"{cat}_{sub}" for cat, sub in SUBTYPE_PARAMS]
    lines.append("## Error Subtype Weights Comparison")
    lines.append("")
    header = "| Subtype | " + " | ".join(methods) + " |"
    sep = "|---------|" + "|".join(["--------"] * len(methods)) + "|"
    lines.append(header)
    lines.append(sep)
    for key in subtype_keys:
        row = f"| {key} |"
        for m in methods:
            cdict = best_configs[m].get("config_dict", {})
            sw = cdict.get("error_subtype_weights", {})
            val = sw.get(key)
            row += f" {_fmt(val)} |"
        lines.append(row)
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path
