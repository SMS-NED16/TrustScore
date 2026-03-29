"""
Generate alignment result figures for the paper.

Usage (from project root):
    python alignment/paper/plot_alignment_results.py --csv alignment/paper/all_results.csv

Outputs (written to --out-dir, default: same directory as CSV):
    alignment_bar_chart.png   -- grouped bar chart: Default / LightGBM / Optuna per dataset
    alignment_heatmap.png     -- side-by-side heatmap: Default / Tuned Spearman ρ
    alignment_delta_heatmap.png -- delta-rho heatmaps (LightGBM vs Default, Optuna vs Default)
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ORDER = ["GPT-4o-mini", "Mistral 7B", "Gemma 2 9B", "Llama 3.1 8B", "Qwen 2.5 7B", "Phi-3 Mini"]
DATASET_ORDER = ["SummEval", "SimpEval", "FeedbackQA", "TopicalChat"]

_MODEL_NAME_MAP = {
    "gpt-4o-mini":               "GPT-4o-mini",
    "Llama-3.1-8B-Instruct":     "Llama 3.1 8B",
    "Meta-Llama-3.1-8B-Instruct":"Llama 3.1 8B",
    "Mistral-7B-Instruct-v0.3":  "Mistral 7B",
    "gemma-2-9b":                "Gemma 2 9B",
    "gemma-2-9b-it":             "Gemma 2 9B",
    "Qwen2.5-7B-Instruct":       "Qwen 2.5 7B",
    "Phi-3-mini-4k-instruct":    "Phi-3 Mini",
}
_DATASET_NAME_MAP = {
    "feedbackqa":     "FeedbackQA",
    "simpeval":       "SimpEval",
    "summeval":       "SummEval",
    "topicalchat_usr":"TopicalChat",
}

COLORS  = {"Default": "#6baed6", "LightGBM": "#ee854a", "Optuna": "#2171b5"}
HATCHES = {"Default": "",        "LightGBM": "//",       "Optuna": ""}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    # Normalise column names from multiple CSV formats:
    #   - notebook format:        dataset, model, optimizer, test_spearman  (already correct)
    #   - old collect_results:    Dataset, Model, Optimizer, Spearman
    #   - new collect_results:    Dataset, Model, Optimizer, Test Spearman
    rename = {}
    for old, new in [
        ("Dataset", "dataset"), ("Model", "model"),
        ("Optimizer", "optimizer"), ("Tuned", "tuned"),
        ("Spearman", "test_spearman"),       # old collect_results
        ("Test Spearman", "test_spearman"),  # new collect_results
    ]:
        if old in df.columns:
            rename[old] = new
    df.rename(columns=rename, inplace=True)

    # Normalise optimizer label capitalisation (notebook writes lowercase)
    if "optimizer" in df.columns:
        df["optimizer"] = df["optimizer"].map(
            lambda v: {"default": "Default", "lightgbm": "LightGBM", "optuna": "Optuna"}.get(
                str(v).strip().lower(), str(v).strip()
            )
        )

    # If old format (tuned bool), derive optimizer label
    if "optimizer" not in df.columns and "tuned" in df.columns:
        df["optimizer"] = df["tuned"].apply(
            lambda v: "Default" if str(v).strip().lower() in ("false","no","default") else "Optuna"
        )

    df["model"]   = df["model"].apply(lambda m: m.split("/")[-1] if "/" in m else m)
    df["model"]   = df["model"].map(_MODEL_NAME_MAP).fillna(df["model"])
    df["dataset"] = df["dataset"].map(_DATASET_NAME_MAP).fillna(df["dataset"])
    df["test_spearman"] = pd.to_numeric(df["test_spearman"], errors="coerce")

    # Deduplicate: keep best test_spearman per (dataset, model, optimizer)
    df = (df.sort_values("test_spearman", ascending=False)
            .drop_duplicates(subset=["dataset","model","optimizer"], keep="first")
            .reset_index(drop=True))
    return df


def pivot_for(df: pd.DataFrame, optimizer: str) -> pd.DataFrame:
    sub = df[df["optimizer"] == optimizer]
    m = sub.pivot(index="model", columns="dataset", values="test_spearman")
    return m.reindex(index=MODEL_ORDER, columns=DATASET_ORDER)


# ---------------------------------------------------------------------------
# Figure 1: Grouped bar chart (Default / LightGBM / Optuna)
# ---------------------------------------------------------------------------

def plot_bar_chart(df: pd.DataFrame, output_path: str) -> None:
    optimizers = [o for o in ["Default","LightGBM","Optuna"] if o in df["optimizer"].values]
    n_opt = len(optimizers)
    width = 0.75 / n_opt
    x = np.arange(len(MODEL_ORDER))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for ax, dataset in zip(axes, DATASET_ORDER):
        for i, opt in enumerate(optimizers):
            sub = df[(df["dataset"] == dataset) & (df["optimizer"] == opt)].set_index("model")
            vals = [sub.loc[m, "test_spearman"] if m in sub.index else np.nan for m in MODEL_ORDER]
            offset = (i - n_opt / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=opt, color=COLORS[opt], hatch=HATCHES[opt],
                   alpha=0.88, edgecolor="white", linewidth=0.5)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(dataset, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Spearman Correlation")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=COLORS[o], hatch=HATCHES[o],
                               edgecolor="grey", alpha=0.88, label=o)
               for o in optimizers]
    fig.legend(handles=handles, loc="upper right", fontsize=11)
    fig.suptitle(
        "TEBScore Alignment with Human Judgments\nDefault vs LightGBM vs Optuna",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 2: Side-by-side Spearman heatmap (Default | Optuna)
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, output_path: str) -> None:
    optimizers = [o for o in ["Default","LightGBM","Optuna"] if o in df["optimizer"].values]
    n = len(optimizers)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, opt in zip(axes, optimizers):
        matrix = pivot_for(df, opt)
        im = ax.imshow(matrix.values, cmap="RdYlGn", aspect="auto", vmin=-0.4, vmax=0.4)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(DATASET_ORDER, fontsize=11)
        ax.set_yticklabels(MODEL_ORDER, fontsize=11)
        ax.set_title(f"{opt} Config", fontsize=13, fontweight="bold")
        for i in range(len(MODEL_ORDER)):
            for j in range(len(DATASET_ORDER)):
                val = matrix.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=10, color="black" if abs(val) < 0.25 else "white")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")

    fig.suptitle("TEBScore Spearman Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Figure 3: Delta-rho heatmaps (LightGBM vs Default, Optuna vs Default)
# ---------------------------------------------------------------------------

def plot_delta_heatmap(df: pd.DataFrame, output_path: str) -> None:
    tuned_opts = [o for o in ["LightGBM","Optuna"] if o in df["optimizer"].values]
    if not tuned_opts:
        print("  No tuned optimizers found — skipping delta heatmap.")
        return

    default_matrix = pivot_for(df, "Default")
    n = len(tuned_opts)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, opt in zip(axes, tuned_opts):
        tuned_matrix = pivot_for(df, opt)
        delta = tuned_matrix.values - default_matrix.values
        vmax = max(0.5, np.nanmax(np.abs(delta)))

        im = ax.imshow(delta, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(DATASET_ORDER, fontsize=11)
        ax.set_yticklabels(MODEL_ORDER, fontsize=11)
        ax.set_title(f"Δρ  ({opt} − Default)", fontsize=13, fontweight="bold")

        for i in range(len(MODEL_ORDER)):
            for j in range(len(DATASET_ORDER)):
                val = delta[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                            fontsize=10, color="black" if abs(val) < 0.3 * vmax else "white")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Δρ")

    fig.suptitle("Tuning Gain (Δρ) vs. Default Configuration",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate alignment paper figures.")
    parser.add_argument("--csv", required=True, help="Path to collected results CSV")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: same directory as CSV)")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.csv} ...")
    df = load_and_prepare(args.csv)
    print(f"  {len(df)} rows after deduplication")
    print(f"  Optimizers present: {sorted(df['optimizer'].unique())}")

    plot_bar_chart(df,       os.path.join(out_dir, "alignment_bar_chart.png"))
    plot_heatmap(df,         os.path.join(out_dir, "alignment_heatmap.png"))
    plot_delta_heatmap(df,   os.path.join(out_dir, "alignment_delta_heatmap.png"))
    print("Done.")


if __name__ == "__main__":
    main()
