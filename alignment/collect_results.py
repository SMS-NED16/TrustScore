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


def _fmt(x: Optional[float]) -> str:
    if x is None or x != x:
        return "\u2014"
    return f"{x:.4f}"


def _fmt_stat(stats: Optional[Dict[str, Optional[float]]]) -> str:
    if stats is None or stats.get("mean") is None:
        return "\u2014"
    mean = stats["mean"]
    std = stats.get("std", 0)
    median = stats.get("median")
    return f"{mean:.1f} \u00b1 {std:.1f} (med {median:.1f})"


def _fmt_stat_csv(stats: Optional[Dict[str, Optional[float]]]) -> str:
    if stats is None or stats.get("mean") is None:
        return ""
    return f"{stats['mean']:.1f} +/- {stats.get('std', 0):.1f} (med {stats.get('median', 0):.1f})"


def load_run(run_dir: str) -> Optional[Dict[str, Any]]:
    """Load all artifacts from a single run directory."""
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    results_path = os.path.join(run_dir, "alignment_results.json")
    configs_path = os.path.join(run_dir, "best_configs.json")

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

    return {
        "run_dir": run_dir,
        "manifest": manifest,
        "alignment_results": alignment_results,
        "best_configs": best_configs,
    }


def find_best_tuned_method(alignment_results: Dict[str, Dict]) -> Optional[str]:
    """Pick the tuned method with the highest test Spearman (excluding 'default')."""
    best_method = None
    best_val = float("-inf")
    for method, res in alignment_results.items():
        if method == "default":
            continue
        spearman = res.get("test_spearman")
        if spearman is not None and spearman == spearman and spearman > best_val:
            best_val = spearman
            best_method = method
    return best_method


def build_rows(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build table rows: two rows per run (untuned + best tuned)."""
    rows = []
    for run_data in runs:
        manifest = run_data["manifest"]
        results = run_data["alignment_results"]
        configs = run_data["best_configs"]

        task = manifest.get("task", "unknown")
        model = manifest.get("judge_model", "unknown")
        n_obs = (
            manifest.get("n_train", 0)
            + manifest.get("n_val", 0)
            + manifest.get("n_test", 0)
        ) or None

        # Row 1: default (untuned)
        if "default" in results:
            res = results["default"]
            cs = res.get("category_stats", {})
            config_summary = ""
            if "default" in configs:
                cd = configs["default"].get("config_dict", {})
                aw = cd.get("aggregation_weights", {})
                config_summary = f"T={aw.get('trustworthiness', '')}, E={aw.get('explainability', '')}, B={aw.get('bias', '')}"
            rows.append({
                "dataset": task,
                "model": model,
                "n_obs": n_obs,
                "t_score": cs.get("T"),
                "e_score": cs.get("E"),
                "b_score": cs.get("B"),
                "aggregated": cs.get("aggregated"),
                "tuned": "No",
                "configs": config_summary,
                "test_pearson": res.get("test_pearson"),
                "test_spearman": res.get("test_spearman"),
                "test_kendall": res.get("test_kendall"),
                "run_dir": run_data["run_dir"],
            })

        # Row 2: best tuned method
        best_method = find_best_tuned_method(results)
        if best_method and best_method in results:
            res = results[best_method]
            cs = res.get("category_stats", {})
            config_summary = ""
            if best_method in configs:
                cd = configs[best_method].get("config_dict", {})
                aw = cd.get("aggregation_weights", {})
                config_summary = f"T={aw.get('trustworthiness', '')}, E={aw.get('explainability', '')}, B={aw.get('bias', '')}"
            rows.append({
                "dataset": task,
                "model": model,
                "n_obs": n_obs,
                "t_score": cs.get("T"),
                "e_score": cs.get("E"),
                "b_score": cs.get("B"),
                "aggregated": cs.get("aggregated"),
                "tuned": f"Yes ({best_method})",
                "configs": config_summary,
                "test_pearson": res.get("test_pearson"),
                "test_spearman": res.get("test_spearman"),
                "test_kendall": res.get("test_kendall"),
                "run_dir": run_data["run_dir"],
            })

    return rows


def write_markdown_table(rows: List[Dict[str, Any]], output_path: str) -> str:
    path = output_path if output_path.endswith(".md") else output_path + ".md"

    lines = [
        "# Multi-Model Comparison Table",
        "",
        "| Run Dir | Dataset | Model | N | T Score | E Score | B Score | Aggregated TEBScore | Tuned | Configs | Pearson | Spearman | Kendall |",
        "|---------|---------|-------|---|---------|---------|---------|---------------------|-------|---------|---------|----------|---------|",
    ]
    for r in rows:
        n_str = str(r["n_obs"]) if r.get("n_obs") is not None else "—"
        lines.append(
            f"| {os.path.basename(r['run_dir'])} "
            f"| {r['dataset']} "
            f"| {r['model']} "
            f"| {n_str} "
            f"| {_fmt_stat(r['t_score'])} "
            f"| {_fmt_stat(r['e_score'])} "
            f"| {_fmt_stat(r['b_score'])} "
            f"| {_fmt_stat(r['aggregated'])} "
            f"| {r['tuned']} "
            f"| {r['configs']} "
            f"| {_fmt(r['test_pearson'])} "
            f"| {_fmt(r['test_spearman'])} "
            f"| {_fmt(r['test_kendall'])} |"
        )
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def write_csv_table(rows: List[Dict[str, Any]], output_path: str) -> str:
    path = output_path if output_path.endswith(".csv") else output_path + ".csv"

    fieldnames = [
        "Run Dir", "Dataset", "Model", "N", "T Score", "E Score", "B Score",
        "Aggregated TEBScore", "Tuned", "Configs",
        "Pearson", "Spearman", "Kendall",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "Run Dir": os.path.basename(r["run_dir"]),
                "Dataset": r["dataset"],
                "Model": r["model"],
                "N": r["n_obs"] if r.get("n_obs") is not None else "",
                "T Score": _fmt_stat_csv(r["t_score"]),
                "E Score": _fmt_stat_csv(r["e_score"]),
                "B Score": _fmt_stat_csv(r["b_score"]),
                "Aggregated TEBScore": _fmt_stat_csv(r["aggregated"]),
                "Tuned": r["tuned"],
                "Configs": r["configs"],
                "Pearson": _fmt(r["test_pearson"]),
                "Spearman": _fmt(r["test_spearman"]),
                "Kendall": _fmt(r["test_kendall"]),
            })
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Collect alignment results from multiple runs into a comparison table."
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Paths to individual run directories",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Parent results directory; auto-discovers all run subdirectories",
    )
    parser.add_argument(
        "--output", default="comparison_table",
        help="Output file base name (writes .md and .csv)",
    )
    args = parser.parse_args()

    run_dirs = []
    if args.runs:
        run_dirs = args.runs
    elif args.results_dir:
        for entry in sorted(os.listdir(args.results_dir)):
            candidate = os.path.join(args.results_dir, entry)
            if os.path.isdir(candidate) and os.path.exists(os.path.join(candidate, "run_manifest.json")):
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
    md_path = write_markdown_table(rows, args.output)
    csv_path = write_csv_table(rows, args.output)

    print(f"\nComparison table written:")
    print(f"  Markdown: {md_path}")
    print(f"  CSV:      {csv_path}")
    print(f"  Rows:     {len(rows)} ({len(runs)} runs x 2 rows each)")


if __name__ == "__main__":
    main()
