"""
Experiment History Viewer
=========================
Scans all alignment results folders and produces a summary table of every run,
tagged by date, model, dataset size, splits manifest, and alignment metrics.

Usage (Jupyter / script):
    from analysis.experiment_history import build_experiment_table
    df = build_experiment_table("/workspace/alignment/results")
    df  # renders as table in Jupyter

    # or export:
    df.to_csv("experiment_history.csv", index=False)

CLI:
    python analysis/experiment_history.py --results_dir /workspace/alignment/results
    python analysis/experiment_history.py --results_dir /workspace/alignment/results --csv out.csv
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _parse_run_datetime(run_id: str) -> Optional[datetime]:
    """Parse datetime from folder name: YYYYMMDD_HHMMSS_<uuid>"""
    try:
        return datetime.strptime("_".join(run_id.split("_")[:2]), "%Y%m%d_%H%M%S")
    except Exception:
        return None


def _manifest_sample_count(manifest_path: Optional[str]) -> Optional[int]:
    """
    If splits_manifest_path points to another run's splits.json,
    read it and return the total sample count recorded there.
    """
    if not manifest_path or not os.path.isfile(manifest_path):
        return None
    data = _read_json(Path(manifest_path))
    if not data:
        return None
    n = (len(data.get("train_ids", [])) +
         len(data.get("val_ids", [])) +
         len(data.get("test_ids", [])))
    return n or None


def _extract_results_metrics(
    alignment_results: Dict[str, Any],
    methods: list[str],
) -> Dict[str, Any]:
    """
    Flatten alignment_results.json into per-method columns.
    Produces columns like: default_val_pearson, a_test_spearman, etc.
    """
    row: Dict[str, Any] = {}
    for method in methods:
        m = alignment_results.get(method, {})
        for split in ("train", "val", "test"):
            for metric in ("pearson", "spearman", "kendall"):
                key = f"{split}_{metric}"
                row[f"{method}_{key}"] = m.get(key)
        # aggregated category mean on test split (most useful summary)
        cat_stats = m.get("category_stats", {})
        for cat in ("T", "E", "B", "aggregated"):
            row[f"{method}_{cat}_mean"] = cat_stats.get(cat, {}).get("mean")
    return row


# ── main builder ─────────────────────────────────────────────────────────────

def build_experiment_table(results_dir: str) -> pd.DataFrame:
    """
    Walk every run folder under results_dir and return a DataFrame with one
    row per run.

    Columns
    -------
    Identity:
        run_id, datetime, date, time

    Config  (from run_manifest.json):
        task, method, judge_model, use_llama,
        max_samples, max_evals, num_judges_per_category,
        temperature, max_tokens, random_seed,
        splits_manifest_path

    Dataset (from splits.json):
        n_train, n_val, n_test, n_total,
        manifest_n_samples  (sample count from the linked splits manifest)

    Results (from alignment_results.json) — repeated per method:
        {method}_train_pearson / spearman / kendall
        {method}_val_pearson   / spearman / kendall
        {method}_test_pearson  / spearman / kendall
        {method}_T_mean, {method}_E_mean, {method}_B_mean, {method}_aggregated_mean
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows = []
    run_folders = sorted(
        [d for d in results_path.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    for folder in run_folders:
        run_id = folder.name
        row: Dict[str, Any] = {"run_id": run_id}

        # ── datetime from folder name ──────────────────────────────────────
        dt = _parse_run_datetime(run_id)
        row["datetime"] = dt
        row["date"] = dt.date() if dt else None
        row["time"] = dt.strftime("%H:%M:%S") if dt else None

        # ── run_manifest.json ──────────────────────────────────────────────
        manifest = _read_json(folder / "run_manifest.json") or {}
        row["task"]                    = manifest.get("task")
        row["method"]                  = manifest.get("method")
        row["judge_model"]             = manifest.get("judge_model") or manifest.get("llama_model")
        row["use_llama"]               = manifest.get("use_llama")
        row["max_samples"]             = manifest.get("max_samples")
        row["max_evals"]               = manifest.get("max_evals")
        row["num_judges_per_category"] = manifest.get("num_judges_per_category")
        row["temperature"]             = manifest.get("temperature")
        row["max_tokens"]              = manifest.get("max_tokens")
        row["random_seed"]             = manifest.get("random_seed")
        splits_manifest_path           = manifest.get("splits_manifest_path")
        row["splits_manifest_path"]    = splits_manifest_path

        # ── splits.json ────────────────────────────────────────────────────
        splits = _read_json(folder / "splits.json") or {}
        n_train = splits.get("n_train", len(splits.get("train_ids", [])))
        n_val   = splits.get("n_val",   len(splits.get("val_ids",   [])))
        n_test  = splits.get("n_test",  len(splits.get("test_ids",  [])))
        row["n_train"] = n_train or None
        row["n_val"]   = n_val   or None
        row["n_test"]  = n_test  or None
        row["n_total"] = (n_train + n_val + n_test) or None
        row["manifest_n_samples"] = _manifest_sample_count(splits_manifest_path)

        # ── alignment_results.json ─────────────────────────────────────────
        alignment_results = _read_json(folder / "alignment_results.json") or {}
        methods_present = [m for m in ("default", "a", "b", "c") if m in alignment_results]
        row["methods_run"] = ", ".join(methods_present) if methods_present else None
        row.update(_extract_results_metrics(alignment_results, ["default", "a", "b", "c"]))

        rows.append(row)

    df = pd.DataFrame(rows)

    # ── column ordering ────────────────────────────────────────────────────
    identity_cols = ["run_id", "datetime", "date", "time"]
    config_cols   = [
        "task", "method", "judge_model", "use_llama",
        "max_samples", "max_evals", "num_judges_per_category",
        "temperature", "max_tokens", "random_seed",
        "splits_manifest_path",
    ]
    dataset_cols  = ["n_train", "n_val", "n_test", "n_total", "manifest_n_samples"]
    meta_cols     = ["methods_run"]
    result_cols   = [c for c in df.columns if c not in
                     identity_cols + config_cols + dataset_cols + meta_cols]

    ordered = identity_cols + config_cols + dataset_cols + meta_cols + sorted(result_cols)
    df = df[[c for c in ordered if c in df.columns]]

    return df


# ── pretty-print helpers ──────────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a condensed view: one row per run with only the most useful columns.
    Best method is selected automatically as the one with the highest test_pearson.
    """
    keep = [
        "run_id", "date", "time", "task", "judge_model",
        "n_total", "manifest_n_samples", "max_samples",
        "num_judges_per_category", "methods_run",
    ]
    # pick best test_pearson across all methods for headline metric
    pearson_cols = [c for c in df.columns if c.endswith("_test_pearson")]
    spearman_cols = [c for c in df.columns if c.endswith("_test_spearman")]

    out = df[[c for c in keep if c in df.columns]].copy()

    if pearson_cols:
        out["best_test_pearson"]  = df[pearson_cols].max(axis=1).round(4)
        out["best_method"]        = df[pearson_cols].idxmax(axis=1).str.replace("_test_pearson", "")
    if spearman_cols:
        out["best_test_spearman"] = df[spearman_cols].max(axis=1).round(4)

    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alignment experiment history viewer")
    parser.add_argument(
        "--results_dir",
        default="/workspace/alignment/results",
        help="Path to the alignment results directory",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to save the full table as CSV",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print condensed summary table instead of full table",
    )
    args = parser.parse_args()

    df = build_experiment_table(args.results_dir)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"Saved full table to {args.csv}")

    display_df = summary_table(df) if args.summary else df
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(display_df.to_string(index=False))
