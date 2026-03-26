"""
Gartner-style quadrant plot: evaluation metrics and frameworks.
Labels are anchored with leader lines and placed to avoid overlap.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1) Data: edit x/y freely
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "name": [
        "BLEU / ROUGE",
        "BERTScore",
        "BARTScore",
        "TrustScore (2024)",
        "QuestEval",
        "GPT-4-Eval",
        "TIGERScore",
        "Themis",
        "OpeNLGauge",
        "HELM",
        "OpenAI Evals",
        "TruLens",
        "Giskard",
        "ARES",
        "RAGChecker",
        "TrustScore (Proposed)",
    ],
    "x": [
        -0.95,  # BLEU/ROUGE
        -0.85,  # BERTScore
        -0.80,  # BARTScore
        -0.20,  # TrustScore (2024)
        -0.55,  # QuestEval
        -0.50,  # GPT-4-Eval
        -0.45,  # TIGERScore
        -0.40,  # Themis
        -0.45,  # OpeNLGauge
         0.70,  # HELM
         0.58,  # OpenAI Evals (moved toward quadrant center, farther from TrustScore)
         0.55,  # TruLens
         0.52,  # Giskard
         0.65,  # ARES
         0.60,  # RAGChecker
         0.95,  # TrustScore (Proposed)
    ],
    "y": [
        -0.95,  # BLEU/ROUGE
        -0.85,  # BERTScore
        -0.80,  # BARTScore
         0.85,  # TrustScore (2024)
         0.75,  # QuestEval
         0.85,  # GPT-4-Eval
         0.80,  # TIGERScore
         0.80,  # Themis
         0.80,  # OpeNLGauge
        -0.60,  # HELM
         0.52,  # OpenAI Evals (closer to quadrant center)
         0.48,  # TruLens
         0.44,  # Giskard
         0.10,  # ARES (hybrid)
         0.10,  # RAGChecker (hybrid)
         0.85,  # TrustScore (Proposed)
    ],
    "group": [
        "Ref-Based Metric", "Ref-Based Metric", "Ref-Based Metric",
        "Ref-Free Metric", "Ref-Free Metric", "Ref-Free Metric",
        "Ref-Free Metric", "Ref-Free Metric", "Ref-Free Metric",
        "Ref-Based Framework",
        "Ref-Free Framework", "Ref-Free Framework", "Ref-Free Framework",
        "Ref-Free Framework", "Ref-Free Framework",
        "Proposed",
    ],
})

df["is_proposed"] = df["group"] == "Proposed"

# ---------------------------------------------------------------------------
# 2) Explicit label positions OUTSIDE the quadrant [-1,1] x [-1,1].
#    Left-side points: labels at label_x = -1.08 (left of quadrant), right-aligned.
#    Right-side points: labels at label_x = 1.08 (right of quadrant), left-aligned.
#    Leader lines connect (x, y) -> (label_x, label_y).
# ---------------------------------------------------------------------------
LABEL_LEFT_X = -1.08   # Left of quadrant; labels sit here, right-aligned
LABEL_RIGHT_X = 1.08   # Right of quadrant; labels sit here, left-aligned

df["label_x"] = df["x"] + 0.08
df["label_y"] = df["y"]
df["label_ha"] = "left"   # default; set to "right" for left-side labels

# Left column: 9 labels with ~0.30 vertical spacing (no overlap; clear of y-axis text)
# Top-left: Ref-free metrics (top to bottom)
df.loc[df["name"] == "TrustScore (2024)", ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, 0.92, "right"]
df.loc[df["name"] == "GPT-4-Eval",        ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, 0.60, "right"]
df.loc[df["name"] == "OpeNLGauge",        ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, 0.28, "right"]
df.loc[df["name"] == "QuestEval",         ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, -0.04, "right"]
df.loc[df["name"] == "TIGERScore",        ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, -0.36, "right"]
df.loc[df["name"] == "Themis",            ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, -0.68, "right"]
# Bottom-left: Ref-based metrics
df.loc[df["name"] == "BLEU / ROUGE", ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, -1.00, "right"]
df.loc[df["name"] == "BERTScore",    ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, -1.28, "right"]
df.loc[df["name"] == "BARTScore",    ["label_x", "label_y", "label_ha"]] = [LABEL_LEFT_X, -1.48, "right"]

# Right column: 7 labels with ~0.30 vertical spacing (top to bottom)
df.loc[df["name"] == "TrustScore (Proposed)", ["label_x", "label_y"]] = [LABEL_RIGHT_X, 0.92]
df.loc[df["name"] == "OpenAI Evals",          ["label_x", "label_y"]] = [LABEL_RIGHT_X, 0.60]
df.loc[df["name"] == "TruLens",               ["label_x", "label_y"]] = [LABEL_RIGHT_X, 0.28]
df.loc[df["name"] == "Giskard",               ["label_x", "label_y"]] = [LABEL_RIGHT_X, -0.04]
df.loc[df["name"] == "RAGChecker", ["label_x", "label_y"]] = [LABEL_RIGHT_X, -0.36]
df.loc[df["name"] == "ARES",       ["label_x", "label_y"]] = [LABEL_RIGHT_X, -0.68]
df.loc[df["name"] == "HELM", ["label_x", "label_y"]] = [LABEL_RIGHT_X, -0.96]

# ---------------------------------------------------------------------------
# 3) Plot
# ---------------------------------------------------------------------------
BLUE_ACCENT = "#1f77b4"
LEADER_LINE_KW = dict(color="gray", linewidth=0.6, zorder=2, clip_on=False)

fig, ax = plt.subplots(figsize=(7.16, 7.16), facecolor="white")
ax.set_facecolor("white")

# Limits: quadrant [-1,1] x [-1,1] with margin so labels sit outside
ax.set_aspect("equal")
ax.set_xlim(-1.32, 1.32)
ax.set_ylim(-1.58, 1.12)

# Quadrant lines
ax.axvline(0, color="gray", linestyle="--", linewidth=0.4)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.4)

# Grid
ax.grid(True, which="major", linestyle="-", linewidth=0.25, color="gray", alpha=0.7)
ax.grid(True, which="minor", linestyle="-", linewidth=0.1, color="gray", alpha=0.4)
ax.set_axisbelow(True)

# Points
other = df[~df["is_proposed"]]
prop = df[df["is_proposed"]]

ax.scatter(other["x"], other["y"], s=28, c="black", zorder=3)
ax.scatter(prop["x"], prop["y"], s=80, c=BLUE_ACCENT, zorder=4)

# Leader lines: point -> label anchor (left edge of label)
for _, row in df.iterrows():
    ax.plot(
        [row["x"], row["label_x"]],
        [row["y"], row["label_y"]],
        **LEADER_LINE_KW,
    )

# Labels at (label_x, label_y); left-side use ha="right", right-side use ha="left"
for _, row in df.iterrows():
    bbox = dict(
        boxstyle="round,pad=0.32,rounding_size=0.15",
        facecolor="white",
        edgecolor="black",
        linewidth=0.35 if row["is_proposed"] else 0.2,
    )
    fs = 11.0 if row["is_proposed"] else 10.5
    ha = row.get("label_ha", "left")
    ax.text(
        row["label_x"], row["label_y"], row["name"],
        fontsize=fs,
        weight="bold" if row["is_proposed"] else "normal",
        verticalalignment="center",
        horizontalalignment=ha,
        bbox=bbox,
        zorder=5,
        clip_on=False,
    )

# Axis semantic labels OUTSIDE the quadrant (in axes coordinates)
# Bottom: Metric and Framework below the plot
ax.text(0.25, -0.06, "Metric", fontsize=10, ha="center", va="top",
        transform=ax.transAxes, clip_on=False)
ax.text(0.75, -0.06, "Framework", fontsize=10, ha="center", va="top",
        transform=ax.transAxes, clip_on=False)
# Left: Reference-Based and Reference-Free / Hybrid well to the left (no overlap with point labels)
ax.text(-0.14, 0.25, "Reference-Based", fontsize=9.5, rotation=90, ha="center", va="center",
        transform=ax.transAxes, clip_on=False)
ax.text(-0.14, 0.75, "Reference-Free / Hybrid", fontsize=9.5, rotation=90, ha="center", va="center",
        transform=ax.transAxes, clip_on=False)

# Hide numeric ticks but keep grid
ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.tick_params(axis="both", which="both", labelbottom=False, labelleft=False, length=0)

ax.set_xlabel(None)
ax.set_ylabel(None)

plt.subplots_adjust(left=0.18, right=0.92, top=0.96, bottom=0.06)

for spine in ax.spines.values():
    spine.set_color("grey")
    spine.set_linewidth(0.7)

plt.tight_layout()

# ---------------------------------------------------------------------------
# 4) Export (square, two-column friendly)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(_script_dir, "gartner_quadrant.pdf")
out_png = os.path.join(_script_dir, "gartner_quadrant.png")
fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2, dpi=150)
fig.savefig(out_png, bbox_inches="tight", pad_inches=0.2, dpi=150)
print(f"Saved {out_pdf} and {out_png}")

# Uncomment to display in interactive session:
# plt.show()
plt.close(fig)
