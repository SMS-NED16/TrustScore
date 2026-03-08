"""
Evaluate best configs on train, val, test and produce correlation table and report.
"""

import json
import os
from typing import List, Dict, Any, Optional
from alignment.config_space import vector_to_config, config_to_dict, SUBTYPE_PARAMS
from alignment.scoring import compute_correlation_for_samples
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
    Compute Pearson and Spearman for this config on train, val, test.
    Returns dict with train_pearson, train_spearman, val_pearson, val_spearman, test_pearson, test_spearman.
    """
    results = {}
    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        if not samples:
            results[f"{split_name}_pearson"] = None
            results[f"{split_name}_spearman"] = None
            continue
        p, s, _, _ = compute_correlation_for_samples(
            config, samples, task, cache_dir, task_name, use_quality=True
        )
        results[f"{split_name}_pearson"] = p
        results[f"{split_name}_spearman"] = s
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
    best_configs: { "a": { "config_vector": [...], "config_dict": {...} }, "b": ..., "c": ... }
    Returns { "a": { train_pearson, train_spearman, ... }, "b": ..., "c": ... }.
    """
    alignment_results = {}
    for method, data in best_configs.items():
        vec = data.get("config_vector")
        if not vec:
            continue
        config = vector_to_config(vec)
        alignment_results[method] = evaluate_config(
            config, train_samples, val_samples, test_samples, task, cache_dir, task_name
        )
    return alignment_results


def _fmt(x: Any) -> str:
    return f"{x:.4f}" if x is not None and x == x else "—"


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

    # --- Correlation table ---
    lines.append("## Correlation (Pearson / Spearman)")
    lines.append("")
    lines.append("| Method | Train Pearson | Train Spearman | Val Pearson | Val Spearman | Test Pearson | Test Spearman |")
    lines.append("|--------|----------------|-----------------|-------------|---------------|----------------|----------------|")
    for method, res in alignment_results.items():
        tp = res.get("train_pearson")
        ts = res.get("train_spearman")
        vp = res.get("val_pearson")
        vs = res.get("val_spearman")
        ep = res.get("test_pearson")
        es = res.get("test_spearman")
        lines.append(f"| {method} | {_fmt(tp)} | {_fmt(ts)} | {_fmt(vp)} | {_fmt(vs)} | {_fmt(ep)} | {_fmt(es)} |")
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
