"""
Collect results from multiple alignment runs and produce a comparison table.

Usage:
    python -m alignment.collect_results --runs alignment/results/run1 alignment/results/run2
    python -m alignment.collect_results --results-dir alignment/results
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _stat(cs: Dict, cat: str, key: str) -> Optional[float]:
    v = cs.get(cat, {})
    return v.get(key) if isinstance(v, dict) else None


def load_run(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load all artifacts from a single run directory."""
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    results_path  = os.path.join(run_dir, "alignment_results.json")
    configs_path  = os.path.join(run_dir, "best_configs.json")
    splits_path   = os.path.join(run_dir, "splits.json")

    if not os.path.exists(manifest_path) or not os.path.exists(results_path):
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(results_path, "r", encoding="utf-8") as f:
        alignment_results = json.load(f)

    best_configs = {}
    if os.path.exists(configs_path):
        with open(configs_path, "r", encoding="utf-8") as f:
            best_configs = json.load(f)

    splits = {}
    if os.path.exists(splits_path):
        with open(splits_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

    return {
        "run_dir":           run_dir,
        "manifest":          manifest,
        "splits":            splits,
        "alignment_results": alignment_results,
        "best_configs":      best_configs,
    }


def _make_row(
    run_data:        Dict[str, Any],
    method_key:      str,
    optimizer_label: str,   # "default" | "lightgbm" | "optuna"
    results:         Dict[str, Dict],
    configs:         Dict[str, Dict],
) -> Optional[Dict[str, Any]]:
    """Build one CSV row matching the canonical all_results schema."""
    if method_key not in results:
        return None

    manifest = run_data["manifest"]
    splits   = run_data.get("splits", {})
    res      = results[method_key]
    cs       = res.get("category_stats", {})

    n_train = splits.get("n_train")
    n_val   = splits.get("n_val")
    n_test  = splits.get("n_test")
    n_obs   = (n_train or 0) + (n_val or 0) + (n_test or 0) or manifest.get("max_samples")

    # Config weights — present only for tuned methods
    cfg = configs.get(method_key, {})
    cd  = cfg.get("config_dict", {}) if cfg else {}
    aw  = cd.get("aggregation_weights", {})
    sw  = cd.get("error_subtype_weights", {})

    return {
        # Identity
        "run_id":    os.path.basename(run_data["run_dir"]),
        "dataset":   manifest.get("task"),
        "model":     manifest.get("judge_model"),
        "tuned":     method_key != "default",
        "optimizer": optimizer_label,
        # Manifest metadata
        "max_evals":    manifest.get("max_evals"),
        "random_seed":  manifest.get("random_seed"),
        "timestamp":    manifest.get("timestamp"),
        "git_commit":   manifest.get("git_commit"),
        # Split sizes
        "n_train": n_train,
        "n_val":   n_val,
        "n_test":  n_test,
        "n_obs":   n_obs,
        # Category weights
        "weight_T":          aw.get("trustworthiness"),
        "weight_E":          aw.get("explainability"),
        "weight_B":          aw.get("bias"),
        "sigmoid_steepness": cd.get("sigmoid_steepness"),
        "sigmoid_shift":     cd.get("sigmoid_shift"),
        # T error-subtype weights
        "w_T_spelling":      sw.get("T_spelling"),
        "w_T_factual_error": sw.get("T_factual_error"),
        "w_T_hallucination": sw.get("T_hallucination"),
        "w_T_inconsistency": sw.get("T_inconsistency"),
        # B error-subtype weights
        "w_B_demographic_bias":  sw.get("B_demographic_bias"),
        "w_B_cultural_bias":     sw.get("B_cultural_bias"),
        "w_B_gender_bias":       sw.get("B_gender_bias"),
        "w_B_political_bias":    sw.get("B_political_bias"),
        "w_B_sycophancy_bias":   sw.get("B_sycophancy_bias"),
        "w_B_confirmation_bias": sw.get("B_confirmation_bias"),
        # E error-subtype weights
        "w_E_unclear_explanation":   sw.get("E_unclear_explanation"),
        "w_E_missing_context":       sw.get("E_missing_context"),
        "w_E_overly_complex":        sw.get("E_overly_complex"),
        "w_E_assumption_not_stated": sw.get("E_assumption_not_stated"),
        # Category score statistics
        "T_mean":   _stat(cs, "T", "mean"),
        "T_median": _stat(cs, "T", "median"),
        "T_std":    _stat(cs, "T", "std"),
        "E_mean":   _stat(cs, "E", "mean"),
        "E_median": _stat(cs, "E", "median"),
        "E_std":    _stat(cs, "E", "std"),
        "B_mean":   _stat(cs, "B", "mean"),
        "B_median": _stat(cs, "B", "median"),
        "B_std":    _stat(cs, "B", "std"),
        "agg_mean":   _stat(cs, "aggregated", "mean"),
        "agg_median": _stat(cs, "aggregated", "median"),
        "agg_std":    _stat(cs, "aggregated", "std"),
        # Correlations — all splits, all metrics
        "train_pearson":  res.get("train_pearson"),
        "train_spearman": res.get("train_spearman"),
        "train_kendall":  res.get("train_kendall"),
        "val_pearson":    res.get("val_pearson"),
        "val_spearman":   res.get("val_spearman"),
        "val_kendall":    res.get("val_kendall"),
        "test_pearson":   res.get("test_pearson"),
        "test_spearman":  res.get("test_spearman"),
        "test_kendall":   res.get("test_kendall"),
    }


def build_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build up to three rows per run: default, lightgbm, optuna."""
    rows = []
    for run_data in runs:
        results = run_data["alignment_results"]
        configs = run_data["best_configs"]
        for method_key, optimizer_label in [
            ("default", "default"),
            ("b",       "lightgbm"),
            ("c",       "optuna"),
        ]:
            row = _make_row(run_data, method_key, optimizer_label, results, configs)
            if row is not None:
                rows.append(row)
    return rows


_FIELDNAMES = [
    "run_id", "dataset", "model", "tuned", "optimizer",
    "max_evals", "random_seed", "timestamp", "git_commit",
    "n_train", "n_val", "n_test", "n_obs",
    "weight_T", "weight_E", "weight_B", "sigmoid_steepness", "sigmoid_shift",
    "w_T_spelling", "w_T_factual_error", "w_T_hallucination", "w_T_inconsistency",
    "w_B_demographic_bias", "w_B_cultural_bias", "w_B_gender_bias",
    "w_B_political_bias", "w_B_sycophancy_bias", "w_B_confirmation_bias",
    "w_E_unclear_explanation", "w_E_missing_context",
    "w_E_overly_complex", "w_E_assumption_not_stated",
    "T_mean", "T_median", "T_std",
    "E_mean", "E_median", "E_std",
    "B_mean", "B_median", "B_std",
    "agg_mean", "agg_median", "agg_std",
    "train_pearson", "train_spearman", "train_kendall",
    "val_pearson", "val_spearman", "val_kendall",
    "test_pearson", "test_spearman", "test_kendall",
]


def write_csv_table(rows: List[Dict[str, Any]], output_path: str) -> str:
    path = output_path if output_path.endswith(".csv") else output_path + ".csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _FIELDNAMES})
    return path


def write_markdown_table(rows: List[Dict[str, Any]], output_path: str) -> str:
    """Write a compact human-readable summary (not the full schema)."""
    path = output_path if output_path.endswith(".md") else output_path + ".md"
    lines = [
        "# Alignment Results Summary",
        "",
        "| Run ID | Dataset | Model | Optimizer | Train ρ | Val ρ | Test ρ | Test r | Test τ |",
        "|--------|---------|-------|-----------|---------|-------|--------|--------|--------|",
    ]

    def _f(x):
        return f"{x:.4f}" if x is not None and x == x else "—"

    for r in rows:
        lines.append(
            f"| {r['run_id']} "
            f"| {r['dataset']} "
            f"| {r['model']} "
            f"| {r['optimizer']} "
            f"| {_f(r['train_spearman'])} "
            f"| {_f(r['val_spearman'])} "
            f"| {_f(r['test_spearman'])} "
            f"| {_f(r['test_pearson'])} "
            f"| {_f(r['test_kendall'])} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Collect alignment results from multiple runs into a comparison table."
    )
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Paths to individual run directories")
    parser.add_argument("--results-dir", default=None,
                        help="Parent directory; auto-discovers run subdirectories")
    parser.add_argument("--output", default="comparison_table",
                        help="Output file base name (writes .md and .csv)")
    args = parser.parse_args()

    run_dirs = []
    if args.runs:
        run_dirs = args.runs
    elif args.results_dir:
        for entry in sorted(os.listdir(args.results_dir)):
            candidate = os.path.join(args.results_dir, entry)
            if os.path.isdir(candidate) and os.path.exists(
                    os.path.join(candidate, "run_manifest.json")):
                run_dirs.append(candidate)
    else:
        parser.error("Provide --runs or --results-dir")

    if not run_dirs:
        print("No valid run directories found.")
        sys.exit(1)

    runs = []
    for rd in run_dirs:
        data = load_run(rd)
        if data:
            runs.append(data)
            print(f"  Loaded: {rd} (model={data['manifest'].get('judge_model', '?')})")
        else:
            print(f"  Skipped (missing artifacts): {rd}")

    if not runs:
        print("No runs with valid artifacts found.")
        sys.exit(1)

    rows = build_rows(runs)
    before = len(rows)

    # Normalise model name for deduplication:
    # strip HuggingFace org prefix and collapse known aliases
    _MODEL_ALIASES = {"gemma-2-9b-it": "gemma-2-9b"}

    def _norm(name: str) -> str:
        name = name.split("/")[-1] if "/" in name else name
        return _MODEL_ALIASES.get(name, name)

    seen: Dict[tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (row["dataset"], _norm(row["model"] or ""), row["optimizer"])
        existing = seen.get(key)
        if existing is None:
            seen[key] = row
        else:
            ts_new = row["test_spearman"] if row["test_spearman"] is not None else float("-inf")
            ts_old = existing["test_spearman"] if existing["test_spearman"] is not None else float("-inf")
            if ts_new > ts_old:
                seen[key] = row
    rows = list(seen.values())

    csv_path = write_csv_table(rows, args.output)
    md_path  = write_markdown_table(rows, args.output)

    print(f"\nComparison table written:")
    print(f"  CSV:      {csv_path}")
    print(f"  Markdown: {md_path}")
    print(f"  Rows:     {len(rows)} (deduplicated from {before}; up to 3 per combination: default / lightgbm / optuna)")


if __name__ == "__main__":
    main()
