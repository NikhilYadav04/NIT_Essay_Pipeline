"""
analyze.py — Metric Analysis and Plots for Batch Results
=========================================================

Reads results/batch_results.csv and computes:
  - QWK (Quadratic Weighted Kappa)      — agreement with human scores
  - Pearson r                            — linear correlation
  - MAE (Mean Absolute Error)            — average score gap
  - Exact match rate                     — % scores that match exactly
  - Adjacent match rate                  — % scores within ±1 of human

Generates 4 plots saved to results/plots/:
  1. Scatter: MAGIC vs Human
  2. Scatter: AutoSCORE vs Human
  3. Bar: QWK comparison
  4. Heatmap: confusion matrix for each pipeline

Usage:
    python batch_eval/analyze.py
    python batch_eval/analyze.py --pipeline magic   # only MAGIC metrics
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to files
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix
from scipy.stats import pearsonr

# ── Paths ──────────────────────────────────────────────────────────────────────
BATCH_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(BATCH_DIR, "results", "batch_results.csv")
PLOTS_DIR    = os.path.join(BATCH_DIR, "results", "plots")


# ── Metrics helper ────────────────────────────────────────────────────────────

def compute_metrics(human: pd.Series, pred: pd.Series, label: str) -> dict:
    """Return a dict of all metrics for a (human, pred) pair."""
    h = human.values.astype(int)
    p = pred.values.astype(int)

    # Clip predictions to [0, 6] to avoid kappa range errors
    p = np.clip(p, 0, 6)
    h = np.clip(h, 0, 6)

    qwk  = cohen_kappa_score(h, p, weights="quadratic", labels=list(range(7)))
    r, _ = pearsonr(h, p)
    mae  = mean_absolute_error(h, p)
    exact = np.mean(h == p)
    adj   = np.mean(np.abs(h - p) <= 1)

    return {
        "pipeline": label,
        "n":        len(h),
        "qwk":      round(qwk,  3),
        "pearson_r": round(r,   3),
        "mae":      round(mae,  3),
        "exact_pct": round(exact * 100, 1),
        "adj_pct":   round(adj  * 100, 1),
        "mean_pred": round(p.mean(), 2),
        "std_pred":  round(p.std(),  2),
        "mean_human": round(h.mean(), 2),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

BLUE  = "#4C9BE8"
GREEN = "#56C78A"
RED   = "#E8644C"

def plot_scatter(ax, human, pred, label, color, metrics: dict):
    """Scatter plot with jitter to show density."""
    jitter = 0.15
    hj = human + np.random.uniform(-jitter, jitter, len(human))
    pj = pred  + np.random.uniform(-jitter, jitter, len(pred))

    ax.scatter(hj, pj, alpha=0.55, s=40, color=color, edgecolors="white", linewidths=0.4)

    # Perfect agreement line
    lo, hi = 0, 6
    ax.plot([lo, hi], [lo, hi], "--", color="white", alpha=0.4, linewidth=1, label="Perfect agreement")

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_xlabel("Human Score (0–6)", fontsize=11, color="white")
    ax.set_ylabel(f"{label} Score (0–6)", fontsize=11, color="white")
    ax.set_title(f"{label} vs Human Scores", fontsize=13, fontweight="bold", color="white", pad=12)

    # Metrics annotation
    info = (
        f"QWK = {metrics['qwk']:.3f}\n"
        f"r   = {metrics['pearson_r']:.3f}\n"
        f"MAE = {metrics['mae']:.3f}\n"
        f"Exact = {metrics['exact_pct']:.1f}%\n"
        f"±1    = {metrics['adj_pct']:.1f}%"
    )
    ax.text(0.04, 0.96, info, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", color="white",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e2e", alpha=0.8, edgecolor=color))

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


def plot_confusion_hm(ax, human, pred, label, color):
    """Heatmap of confusion matrix, 0–6 scale."""
    labels = list(range(7))
    cm = confusion_matrix(
        np.clip(human.values, 0, 6),
        np.clip(pred.values,  0, 6),
        labels=labels
    )

    cmap = LinearSegmentedColormap.from_list("custom", ["#0d0d1a", color], N=256)
    im = ax.imshow(cm, cmap=cmap, aspect="auto")

    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(labels, color="white")
    ax.set_yticklabels(labels, color="white")
    ax.set_xlabel("Predicted Score", fontsize=10, color="white")
    ax.set_ylabel("Human Score",     fontsize=10, color="white")
    ax.set_title(f"{label} — Confusion Matrix", fontsize=12, fontweight="bold", color="white", pad=10)

    # Add count annotations
    for i in range(7):
        for j in range(7):
            val = cm[i, j]
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val < cm.max() * 0.6 else "#0d0d1a",
                        fontsize=8, fontweight="bold")

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


def plot_metric_bars(ax, metrics_dict: dict):
    """Side-by-side bar chart for QWK, Pearson, Exact%, and ±1%."""
    metrics  = ["QWK", "Pearson r", "Exact %", "±1 %"]
    x      = np.arange(len(metrics))

    pipelines = list(metrics_dict.keys())
    num_pipelines = len(pipelines)

    # We dynamically calculate width and positions
    width = 0.8 / num_pipelines

    colors = {
        "MAGIC": BLUE,
        "AutoSCORE": GREEN,
        "Hybrid": RED
    }

    for idx, name in enumerate(pipelines):
        m = metrics_dict[name]
        v = [m["qwk"], m["pearson_r"], m["exact_pct"] / 100, m["adj_pct"] / 100]
        pos = x - 0.4 + (idx * width) + (width / 2)
        color = colors.get(name, "#cccccc")
        bars = ax.bar(pos, v, width, label=name, color=color, alpha=0.88, edgecolor="white", linewidth=0.5)

        # Add labels
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score / Rate", fontsize=11, color="white")
    ax.set_title("Pipeline Comparison — Key Metrics", fontsize=13, fontweight="bold", color="white", pad=12)
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1e1e2e", edgecolor="#333355")
    ax.tick_params(colors="white")
    ax.axhline(0.7, color="white", linestyle=":", alpha=0.3, linewidth=1)

    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


def plot_score_distribution(ax, df: pd.DataFrame):
    """Overlapping histograms of human, MAGIC, AutoSCORE, Hybrid score distributions."""
    bins = np.arange(-0.5, 7.5, 1)

    human_scores = np.clip(df["human_score_norm"].values, 0, 6)
    ax.hist(human_scores,   bins=bins, alpha=0.55, color="white",  label="Human",     edgecolor="#333355")

    if "magic_score" in df.columns:
        valid_magic = df[df["magic_score"] >= 0]
        if len(valid_magic) > 0:
            magic_scores = np.clip(valid_magic["magic_score"].values, 0, 6)
            ax.hist(magic_scores, bins=bins, alpha=0.45, color=BLUE,   label="MAGIC",     edgecolor="#333355")

    if "autoscore_holistic" in df.columns:
        valid_auto = df[df["autoscore_holistic"] >= 0]
        if len(valid_auto) > 0:
            auto_scores = np.clip(valid_auto["autoscore_holistic"].values, 0, 6)
            ax.hist(auto_scores,  bins=bins, alpha=0.45, color=GREEN,  label="AutoSCORE", edgecolor="#333355")

    if "hybrid_score" in df.columns:
        valid_hybrid = df[df["hybrid_score"] >= 0]
        # Filter out failed SRCE essays
        if "hybrid_srce_fields" in df.columns:
            valid_hybrid = valid_hybrid[valid_hybrid["hybrid_srce_fields"] > 0]
        else:
            known_failed = {
                "AAAVUP14319000159461", "AAAVUP14319000159396", "AAAVUP14319000159373", 
                "AAAVUP14319000159293", "AAAVUP14319000157959", "AAAVUP14319000157815", 
                "AAAVUP14319000157200", "AAAVUP14319000156063"
            }
            valid_hybrid = valid_hybrid[~valid_hybrid["essay_id"].isin(known_failed)]
            
        if len(valid_hybrid) > 0:
            hybrid_scores = np.clip(valid_hybrid["hybrid_score"].values, 0, 6)
            ax.hist(hybrid_scores, bins=bins, alpha=0.45, color=RED,    label="Hybrid",    edgecolor="#333355")

    ax.set_xlabel("Score (0–6)", fontsize=11, color="white")
    ax.set_ylabel("Count", fontsize=11, color="white")
    ax.set_title("Score Distributions — Human vs Pipelines", fontsize=13, fontweight="bold", color="white", pad=12)
    ax.legend(framealpha=0.3, labelcolor="white", facecolor="#1e1e2e", edgecolor="#333355")
    ax.set_xticks(range(7))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze batch_results.csv")
    parser.add_argument("--pipeline", choices=["both", "magic", "autoscore", "hybrid", "all"], default="both")
    parser.add_argument("--tag", default="", help="Label for this run e.g. mistral7b, llama31_100")
    args = parser.parse_args()

    tag = ("_" + args.tag.replace(":", "-").replace(" ", "_")) if args.tag else ""

    # ── Load results ───────────────────────────────────────────────────────────
    if not os.path.exists(RESULTS_FILE):
        print(f"[ERROR] Results file not found: {RESULTS_FILE}")
        print("  Run batch_runner.py first.")
        sys.exit(1)

    df = pd.read_csv(RESULTS_FILE)
    print(f"\nLoaded {len(df)} rows from batch_results.csv")

    # ── Filter to valid rows ───────────────────────────────────────────────────
    valid_magic = df[df["magic_score"] >= 0].copy() if "magic_score" in df.columns else pd.DataFrame()
    valid_auto  = df[df["autoscore_holistic"] >= 0].copy() if "autoscore_holistic" in df.columns else pd.DataFrame()
    
    valid_hybrid = pd.DataFrame()
    failed_hybrid_count = 0
    if "hybrid_score" in df.columns:
        valid_hybrid = df[df["hybrid_score"] >= 0].copy()
        
        # Check if we have hybrid_srce_fields
        if "hybrid_srce_fields" in df.columns:
            failed_mask = df["hybrid_srce_fields"] == 0
            failed_hybrid_count = failed_mask.sum()
            valid_hybrid = valid_hybrid[~valid_hybrid["essay_id"].isin(df[failed_mask]["essay_id"])]
        else:
            # Fallback to the 8 known failed essay IDs from the Ollama run
            known_failed = {
                "AAAVUP14319000159461", "AAAVUP14319000159396", "AAAVUP14319000159373", 
                "AAAVUP14319000159293", "AAAVUP14319000157959", "AAAVUP14319000157815", 
                "AAAVUP14319000157200", "AAAVUP14319000156063"
            }
            # Only count/filter those actually present in the dataset
            actual_failed = known_failed.intersection(set(valid_hybrid["essay_id"]))
            failed_hybrid_count = len(actual_failed)
            valid_hybrid = valid_hybrid[~valid_hybrid["essay_id"].isin(known_failed)]

    if len(valid_magic) > 0:
        print(f"  MAGIC valid rows    : {len(valid_magic)} / {len(df)}")
    if len(valid_auto) > 0:
        print(f"  AutoSCORE valid rows: {len(valid_auto)} / {len(df)}")
    if len(valid_hybrid) > 0:
        print(f"  Hybrid valid rows   : {len(valid_hybrid)} / {len(df)}  (Filtered out {failed_hybrid_count} failed SRCE essays)")

    # ── Compute metrics ────────────────────────────────────────────────────────
    magic_metrics = None
    auto_metrics  = None
    hybrid_metrics = None

    print("\n" + "=" * 65)
    print("  METRIC RESULTS")
    print("=" * 65)

    header = f"  {'Pipeline':<14} {'N':>5}  {'QWK':>7}  {'r':>7}  {'MAE':>6}  {'Exact%':>8}  {'±1%':>8}  {'MeanPred':>9}"
    print(header)
    print("  " + "-" * 63)

    if args.pipeline in ("both", "all", "magic") and len(valid_magic) > 0:
        magic_metrics = compute_metrics(valid_magic["human_score_norm"], valid_magic["magic_score"], "MAGIC")
        m = magic_metrics
        print(f"  {'MAGIC':<14} {m['n']:>5}  {m['qwk']:>7.3f}  {m['pearson_r']:>7.3f}  {m['mae']:>6.3f}  {m['exact_pct']:>8.1f}  {m['adj_pct']:>8.1f}  {m['mean_pred']:>9.2f}")

    if args.pipeline in ("both", "all", "autoscore") and len(valid_auto) > 0:
        auto_metrics = compute_metrics(valid_auto["human_score_norm"], valid_auto["autoscore_holistic"], "AutoSCORE")
        m = auto_metrics
        print(f"  {'AutoSCORE':<14} {m['n']:>5}  {m['qwk']:>7.3f}  {m['pearson_r']:>7.3f}  {m['mae']:>6.3f}  {m['exact_pct']:>8.1f}  {m['adj_pct']:>8.1f}  {m['mean_pred']:>9.2f}")

    if args.pipeline in ("all", "hybrid") and len(valid_hybrid) > 0:
        hybrid_metrics = compute_metrics(valid_hybrid["human_score_norm"], valid_hybrid["hybrid_score"], "Hybrid")
        m = hybrid_metrics
        print(f"  {'Hybrid':<14} {m['n']:>5}  {m['qwk']:>7.3f}  {m['pearson_r']:>7.3f}  {m['mae']:>6.3f}  {m['exact_pct']:>8.1f}  {m['adj_pct']:>8.1f}  {m['mean_pred']:>9.2f}")

    print(f"\n  Human mean score: {df['human_score_norm'].mean():.2f}  (std: {df['human_score_norm'].std():.2f})")
    print("=" * 65)

    # ── QWK interpretation ─────────────────────────────────────────────────────
    def interpret_qwk(q):
        if   q >= 0.80: return "Excellent"
        elif q >= 0.60: return "Substantial"
        elif q >= 0.40: return "Moderate"
        elif q >= 0.20: return "Fair"
        else:           return "Poor"

    print("\n  QWK Interpretation:")
    if magic_metrics:
        print(f"  MAGIC    : {magic_metrics['qwk']:.3f} → {interpret_qwk(magic_metrics['qwk'])}")
    if auto_metrics:
        print(f"  AutoSCORE: {auto_metrics['qwk']:.3f} → {interpret_qwk(auto_metrics['qwk'])}")
    if hybrid_metrics:
        print(f"  Hybrid   : {hybrid_metrics['qwk']:.3f} → {interpret_qwk(hybrid_metrics['qwk'])}")

    # Winner
    active_metrics = [m for m in [magic_metrics, auto_metrics, hybrid_metrics] if m is not None]
    if len(active_metrics) > 1:
        sorted_m = sorted(active_metrics, key=lambda x: x["qwk"], reverse=True)
        winner = sorted_m[0]["pipeline"]
        second = sorted_m[1]["pipeline"]
        delta = sorted_m[0]["qwk"] - sorted_m[1]["qwk"]
        print(f"\n  🏆 Higher QWK: {winner} (beats {second} by {delta:.3f})")

    # ── Save metrics CSV ───────────────────────────────────────────────────────
    metrics_rows = [m for m in [magic_metrics, auto_metrics, hybrid_metrics] if m is not None]
    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = os.path.join(BATCH_DIR, "results", f"metrics_summary{tag}.csv")
        metrics_df.to_csv(metrics_path, index=False)
        print(f"\n  Metrics saved → {metrics_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    BG = "#0d0d1a"

    # Identify active pipelines
    active_pipelines = []
    if magic_metrics:
        active_pipelines.append(("MAGIC", valid_magic, "magic_score", BLUE, magic_metrics))
    if auto_metrics:
        active_pipelines.append(("AutoSCORE", valid_auto, "autoscore_holistic", GREEN, auto_metrics))
    if hybrid_metrics:
        active_pipelines.append(("Hybrid", valid_hybrid, "hybrid_score", RED, hybrid_metrics))

    # --- Plot 1: Grid (scatter + confusion for each active pipeline) ---
    if len(active_pipelines) >= 2:
        cols = len(active_pipelines)
        fig = plt.figure(figsize=(8 * cols, 14), facecolor=BG)
        gs  = gridspec.GridSpec(2, cols, hspace=0.4, wspace=0.35)

        for idx, (name, df_val, col_name, color, m) in enumerate(active_pipelines):
            ax_scat = fig.add_subplot(gs[0, idx])
            ax_scat.set_facecolor(BG)
            ax_conf = fig.add_subplot(gs[1, idx])
            ax_conf.set_facecolor(BG)

            plot_scatter(ax_scat, df_val["human_score_norm"], df_val[col_name], name, color, m)
            plot_confusion_hm(ax_conf, df_val["human_score_norm"], df_val[col_name], name, color)

        title_names = " vs ".join([name for name, _, _, _, _ in active_pipelines])
        fig.suptitle(f"{title_names} — ASAP-100 Batch Evaluation",
                     fontsize=16, fontweight="bold", color="white", y=0.98)

        out1 = os.path.join(PLOTS_DIR, f"scatter_confusion{tag}.png")
        fig.savefig(out1, bbox_inches="tight", facecolor=BG, dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved → {out1}")

        # --- Plot 2: Bar chart + distribution ---
        fig2, (ax_bar, ax_dist) = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
        ax_bar.set_facecolor(BG)
        ax_dist.set_facecolor(BG)

        metrics_dict = {name: m for name, _, _, _, m in active_pipelines}
        plot_metric_bars(ax_bar, metrics_dict)
        plot_score_distribution(ax_dist, df)

        fig2.suptitle(f"{title_names} — Metric Summary & Distributions",
                      fontsize=14, fontweight="bold", color="white", y=1.02)

        out2 = os.path.join(PLOTS_DIR, f"metrics_and_distributions{tag}.png")
        fig2.savefig(out2, bbox_inches="tight", facecolor=BG, dpi=150)
        plt.close(fig2)
        print(f"  Plot saved → {out2}")

    elif len(active_pipelines) == 1:
        name, df_val, col_name, color, m = active_pipelines[0]
        # single pipeline plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
        ax1.set_facecolor(BG); ax2.set_facecolor(BG)
        plot_scatter(ax1, df_val["human_score_norm"], df_val[col_name], name, color, m)
        plot_confusion_hm(ax2, df_val["human_score_norm"], df_val[col_name], name, color)
        out = os.path.join(PLOTS_DIR, f"{name.lower()}_analysis{tag}.png")
        fig.savefig(out, bbox_inches="tight", facecolor=BG, dpi=150)
        plt.close(fig)
        print(f"\n  Plot saved → {out}")

    print("\n✅ Analysis complete.\n")


if __name__ == "__main__":
    main()
