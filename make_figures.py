"""Generate must-have figures for the GRPO length-regularization report.

Produces:
  figures/pareto.{pdf,png}            Accuracy vs avg CoT tokens, one panel per benchmark.
  figures/accuracy_bars.{pdf,png}     Accuracy by lambda, MATH-trained vs GSM8K-trained.
  figures/length_violins.{pdf,png}    Per-problem token-count distributions on math500.

Summary metrics are baked in from run.log / run_math.log so figs 1+2 work without
filesystem access. Violins read predictions.jsonl from RESULTS_ROOT.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Data (extracted from run.log and run_math.log summary blocks)
# -----------------------------------------------------------------------------
# Schema: METRICS[training_set][benchmark][lambda] = (accuracy, avg_cot_tokens)
LAMBDAS = [0.0, 0.1, 0.3, 0.5]
BENCHMARKS = ["gsm8k", "math500", "svamp"]

METRICS = {
    "GSM8K-trained": {
        "gsm8k":   {0.0: (0.85064, 312.31), 0.1: (0.84913, 203.91),
                    0.3: (0.84003, 196.05), 0.5: (0.84685, 194.39)},
        "math500": {0.0: (0.614,   583.68), 0.1: (0.588,   455.68),
                    0.3: (0.582,   442.29), 0.5: (0.596,   439.95)},
        "svamp":   {0.0: (0.89333, 219.96), 0.1: (0.91000, 125.29),
                    0.3: (0.90000, 119.07), 0.5: (0.89333, 121.82)},
    },
    "MATH-trained": {
        "gsm8k":   {0.0: (0.84761, 314.20), 0.1: (0.84913, 247.18),
                    0.3: (0.85368, 245.70), 0.5: (0.85368, 243.45)},
        "math500": {0.0: (0.612,   587.50), 0.1: (0.622,   448.26),
                    0.3: (0.612,   446.04), 0.5: (0.618,   442.68)},
        "svamp":   {0.0: (0.91000, 221.42), 0.1: (0.90667, 150.12),
                    0.3: (0.89667, 148.78), 0.5: (0.91000, 144.88)},
    },
}

RESULTS_ROOT = Path("/data/locus/project_data/project_data1/projects/justinl5/anlp/results")
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

GEN_CAP = 1024  # generation token cap at eval time

LAMBDA_COLORS = {0.0: "#4C72B0", 0.1: "#DD8452", 0.3: "#55A868", 0.5: "#C44E52"}
TRAIN_COLORS  = {"GSM8K-trained": "#4C72B0", "MATH-trained": "#DD8452"}
TRAIN_MARKERS = {"GSM8K-trained": "o",       "MATH-trained": "s"}


def save(fig, stem: str) -> None:
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"  wrote {out}")


# -----------------------------------------------------------------------------
# Figure 1: Pareto (accuracy vs avg CoT tokens), one panel per benchmark
# -----------------------------------------------------------------------------
def fig_pareto() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    for ax, bench in zip(axes, BENCHMARKS):
        for train in METRICS:
            xs, ys, lams = [], [], []
            for lam in LAMBDAS:
                acc, tok = METRICS[train][bench][lam]
                xs.append(tok); ys.append(acc * 100); lams.append(lam)
            # connecting line (Pareto trace)
            ax.plot(xs, ys, "-", color=TRAIN_COLORS[train], alpha=0.45,
                    linewidth=1.4, zorder=1)
            # points colored by lambda
            for x, y, lam in zip(xs, ys, lams):
                ax.scatter(x, y,
                           s=110,
                           c=LAMBDA_COLORS[lam],
                           marker=TRAIN_MARKERS[train],
                           edgecolor="black", linewidth=0.6,
                           zorder=3)
            # label the lam=0 point so the reader sees the anchor
            x0, y0 = xs[0], ys[0]
            ax.annotate("λ=0", xy=(x0, y0), xytext=(6, -10),
                        textcoords="offset points", fontsize=8, color="0.25")
        ax.set_title(bench, fontsize=12)
        ax.set_xlabel("avg CoT tokens")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("accuracy (%)")

    # legends — one for λ (color), one for training set (marker)
    lam_handles = [plt.Line2D([0], [0], marker="o", linestyle="",
                              markerfacecolor=LAMBDA_COLORS[l],
                              markeredgecolor="black", markersize=9,
                              label=f"λ={l}") for l in LAMBDAS]
    train_handles = [plt.Line2D([0], [0], marker=TRAIN_MARKERS[t], linestyle="",
                                markerfacecolor="white", markeredgecolor="black",
                                markersize=9, label=t) for t in METRICS]
    leg1 = axes[-1].legend(handles=lam_handles, title="lambda", loc="lower left",
                           fontsize=8, title_fontsize=9, framealpha=0.9)
    axes[-1].add_artist(leg1)
    axes[-1].legend(handles=train_handles, title="training set", loc="lower right",
                    fontsize=8, title_fontsize=9, framealpha=0.9)

    fig.suptitle("Accuracy vs CoT length (Pareto view)", y=1.02, fontsize=13)
    save(fig, "pareto")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: grouped bars, accuracy by lambda, two training sets per benchmark
# -----------------------------------------------------------------------------
def fig_accuracy_bars() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)
    x = np.arange(len(LAMBDAS))
    width = 0.38
    for ax, bench in zip(axes, BENCHMARKS):
        for i, train in enumerate(METRICS):
            accs = [METRICS[train][bench][l][0] * 100 for l in LAMBDAS]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, accs, width, label=train,
                          color=TRAIN_COLORS[train], edgecolor="black", linewidth=0.6)
            for b, a in zip(bars, accs):
                ax.text(b.get_x() + b.get_width() / 2, a + 0.3, f"{a:.1f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"λ={l}" for l in LAMBDAS])
        ax.set_title(bench, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        # zoom y-axis around the data so small differences read
        all_accs = [METRICS[t][bench][l][0] * 100 for t in METRICS for l in LAMBDAS]
        lo, hi = min(all_accs), max(all_accs)
        pad = max(2.0, (hi - lo) * 0.4)
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.suptitle("Accuracy by λ across benchmarks and training sets", y=1.02, fontsize=13)
    save(fig, "accuracy_bars")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: violins of per-problem CoT length on math500 (MATH-trained run)
# -----------------------------------------------------------------------------
def load_cot_tokens(run_prefix: str, bench: str, lam: float) -> list[int] | None:
    """Read predictions.jsonl and return list of cot_tokens. Returns None if missing."""
    path = RESULTS_ROOT / f"{run_prefix}_{bench}_lam{lam}" / "predictions.jsonl"
    if not path.exists():
        return None
    toks: list[int] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("cot_tokens")
            if t is not None:
                toks.append(int(t))
    return toks


def fig_length_violins() -> None:
    """One panel per benchmark, violin per λ, MATH-trained model."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=False)
    any_data = False
    for ax, bench in zip(axes, BENCHMARKS):
        per_lambda: dict[float, list[int]] = {}
        for lam in LAMBDAS:
            toks = load_cot_tokens("grpo_mathtrained", bench, lam)
            if toks:
                per_lambda[lam] = toks
        if not per_lambda:
            ax.text(0.5, 0.5, f"predictions.jsonl missing\nfor {bench}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="0.4")
            ax.set_title(bench, fontsize=12)
            ax.set_xticks([])
            continue
        any_data = True
        positions = list(range(1, len(per_lambda) + 1))
        data = [per_lambda[l] for l in sorted(per_lambda)]
        labels = [f"λ={l}" for l in sorted(per_lambda)]
        parts = ax.violinplot(data, positions=positions, showmeans=False,
                              showmedians=True, widths=0.85)
        for pc, lam in zip(parts["bodies"], sorted(per_lambda)):
            pc.set_facecolor(LAMBDA_COLORS[lam])
            pc.set_edgecolor("black")
            pc.set_alpha(0.75)
        for key in ("cbars", "cmins", "cmaxes", "cmedians"):
            if key in parts:
                parts[key].set_color("black")
                parts[key].set_linewidth(1.0)

        # overlay mean as a white dot
        means = [np.mean(d) for d in data]
        ax.scatter(positions, means, marker="D", color="white",
                   edgecolor="black", s=30, zorder=4, label="mean")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_title(bench, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
        # generation cap line — most relevant on math500
        if bench == "math500":
            ax.axhline(GEN_CAP, color="red", linestyle="--", linewidth=1.2,
                       label=f"gen cap ({GEN_CAP})")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("CoT tokens (per problem)")
    fig.suptitle("Per-problem CoT length distributions (MATH-trained)",
                 y=1.02, fontsize=13)
    if not any_data:
        print("  WARNING: no predictions.jsonl found under "
              f"{RESULTS_ROOT} — violin figure will be blank.")
    save(fig, "length_violins")
    plt.close(fig)


def main() -> None:
    print(f"Output dir: {OUT_DIR}")
    print("Figure 1: Pareto")
    fig_pareto()
    print("Figure 2: accuracy bars")
    fig_accuracy_bars()
    print("Figure 3: length violins")
    fig_length_violins()
    print("done.")


if __name__ == "__main__":
    main()
