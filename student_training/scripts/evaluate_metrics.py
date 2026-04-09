"""
evaluate_metrics.py
===================
Compute and visualise performance metrics from a zero-shot or trained-model
results JSONL file (produced by zero_shot_eval.py or future train_eval.py).

METRICS:
  - Confusion Matrix  (TP/FP/TN/FN at threshold=0.5)
  - Average Precision (AP)  — primary thesis metric
  - AUC-ROC
  - F1, Precision, Recall   (at optimal F1 threshold)
  - Mean score: positive clips vs negative clips

GRAPHS SAVED:
  1. confusion_matrix.png       — heatmap with counts and %
  2. roc_curve.png              — ROC with AUC annotation
  3. pr_curve.png               — Precision-Recall with AP annotation
  4. score_distribution.png     — histogram: positive vs negative scores
  5. group_ap_bar.png           — AP per group (0.5s / 1.0s / 1.5s)
  6. examples_table.txt         — top 5 correct + top 5 incorrect predictions

Usage:
  python student_training/scripts/evaluate_metrics.py \
    --results  outputs/zero_shot/zero_shot_teacher_100.jsonl \
    --out_dir  outputs/metrics/zero_shot_teacher_100 \
    [--tag     "Zero-Shot Baseline"]      # label used in graph titles
    [--threshold 0.5]                     # decision threshold for binary metrics
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works on RunPod without display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# =============================================================================
# Load results
# =============================================================================

def load_results(jsonl_path: str) -> pd.DataFrame:
    """Load results JSONL into a DataFrame, validate required columns."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Validate required columns
    for col in ["video_id", "ground_truth", "score"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in results file")

    # Ensure numeric types
    df["ground_truth"] = pd.to_numeric(df["ground_truth"], errors="coerce")
    df["score"]        = pd.to_numeric(df["score"], errors="coerce")

    # Drop rows with missing ground truth or score
    before = len(df)
    df = df.dropna(subset=["ground_truth", "score"])
    if len(df) < before:
        print(f"  WARNING: Dropped {before - len(df)} rows with missing values")

    return df


# =============================================================================
# Metrics computation
# =============================================================================

def compute_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Compute all scalar metrics."""
    y_true  = df["ground_truth"].astype(int).values
    y_score = df["score"].astype(float).values
    y_pred  = (y_score >= threshold).astype(int)

    # Guard against single-class data (cannot compute AUC/AP)
    n_pos = y_true.sum()
    n_neg = (1 - y_true).sum()
    if n_pos == 0 or n_neg == 0:
        print(f"  WARNING: Only one class present (pos={n_pos}, neg={n_neg}). "
              f"AUC/AP will be 0.")
        auc  = 0.0
        ap   = 0.0
    else:
        auc = roc_auc_score(y_true, y_score)
        ap  = average_precision_score(y_true, y_score)

    # F1 at given threshold
    f1   = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Optimal F1 threshold (sweep)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_score)
    f1_scores = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls),
        0
    )
    best_idx = np.argmax(f1_scores[:-1]) if len(f1_scores) > 1 else 0
    optimal_threshold = float(thresholds_pr[best_idx]) if len(thresholds_pr) > 0 else threshold
    optimal_f1 = float(f1_scores[best_idx])

    return {
        "n_total":           len(df),
        "n_positive":        int(n_pos),
        "n_negative":        int(n_neg),
        "threshold":         threshold,
        "ap":                round(float(ap), 4),
        "auc_roc":           round(float(auc), 4),
        "f1":                round(float(f1), 4),
        "precision":         round(float(prec), 4),
        "recall":            round(float(rec), 4),
        "tp":                int(tp),
        "fp":                int(fp),
        "tn":                int(tn),
        "fn":                int(fn),
        "accuracy":          round((tp + tn) / max(len(df), 1), 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "optimal_f1":        round(optimal_f1, 4),
        "mean_score_pos":    round(float(df[df["ground_truth"] == 1]["score"].mean()), 4),
        "mean_score_neg":    round(float(df[df["ground_truth"] == 0]["score"].mean()), 4),
        # Raw arrays for plotting
        "_y_true":           y_true,
        "_y_score":          y_score,
        "_y_pred":           y_pred,
        "_pr_precisions":    precisions,
        "_pr_recalls":       recalls,
        "_pr_thresholds":    thresholds_pr,
    }


def compute_group_metrics(df: pd.DataFrame, threshold: float) -> dict:
    """Compute AP per group (0.5s / 1.0s / 1.5s before event), if group column exists."""
    if "group" not in df.columns:
        return {}

    group_map = {0: "0.5s before event", 1: "1.0s before event", 2: "1.5s before event"}
    results = {}
    for g, label in group_map.items():
        sub = df[df["group"] == g]
        if len(sub) == 0:
            continue
        y_true  = sub["ground_truth"].astype(int).values
        y_score = sub["score"].astype(float).values
        if y_true.sum() == 0 or (1 - y_true).sum() == 0:
            results[label] = {"ap": float("nan"), "n": len(sub)}
        else:
            results[label] = {
                "ap": round(float(average_precision_score(y_true, y_score)), 4),
                "n":  len(sub),
            }
    return results


# =============================================================================
# Plotting
# =============================================================================

PALETTE = {"positive": "#e74c3c", "negative": "#2ecc71", "line": "#2c3e50", "highlight": "#3498db"}


def plot_confusion_matrix(metrics: dict, out_path: str, tag: str):
    cm = np.array([[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]])
    total = cm.sum()
    cm_pct = cm / max(total, 1) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=["Pred: No", "Pred: Yes"],
        yticklabels=["True: No", "True: Yes"],
        ax=ax, linewidths=0.5, linecolor="gray",
    )
    # Annotate with count and %
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.5,
                    f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > total * 0.3 else "black")

    ax.set_title(f"Confusion Matrix — {tag}\n"
                 f"Acc={metrics['accuracy']:.3f}  F1={metrics['f1']:.3f}  "
                 f"Thr={metrics['threshold']}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_roc_curve(metrics: dict, out_path: str, tag: str):
    y_true, y_score = metrics["_y_true"], metrics["_y_score"]
    if y_true.sum() == 0 or (1 - y_true).sum() == 0:
        print("  SKIP ROC: single class")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = metrics["auc_roc"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color=PALETTE["line"], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["line"])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {tag}", fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_pr_curve(metrics: dict, out_path: str, tag: str):
    y_true, y_score = metrics["_y_true"], metrics["_y_score"]
    if y_true.sum() == 0:
        print("  SKIP PR: no positive class")
        return

    precisions = metrics["_pr_precisions"]
    recalls    = metrics["_pr_recalls"]
    ap         = metrics["ap"]
    baseline   = y_true.sum() / len(y_true)  # random classifier line

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recalls, precisions, color=PALETTE["highlight"], lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(y=baseline, color="gray", linestyle="--", lw=1,
               label=f"Random = {baseline:.3f}")
    ax.fill_between(recalls, precisions, alpha=0.1, color=PALETTE["highlight"])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(f"Precision-Recall Curve — {tag}", fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_score_distribution(df: pd.DataFrame, out_path: str, tag: str):
    pos_scores = df[df["ground_truth"] == 1]["score"].values
    neg_scores = df[df["ground_truth"] == 0]["score"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(pos_scores, bins=bins, alpha=0.65, color=PALETTE["positive"],
            label=f"Collision (n={len(pos_scores)})", density=True)
    ax.hist(neg_scores, bins=bins, alpha=0.65, color=PALETTE["negative"],
            label=f"No Collision (n={len(neg_scores)})", density=True)
    ax.axvline(x=0.5, color="black", linestyle="--", lw=1, label="threshold=0.5")
    ax.set_xlabel("Predicted P(collision)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Score Distribution — {tag}", fontsize=11)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_group_ap_bar(group_metrics: dict, out_path: str, tag: str):
    if not group_metrics:
        return

    labels = list(group_metrics.keys())
    ap_vals = [group_metrics[l]["ap"] for l in labels]
    n_vals  = [group_metrics[l]["n"] for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, ap_vals, color=PALETTE["highlight"], edgecolor="black", width=0.5)
    for bar, n, ap in zip(bars, n_vals, ap_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"AP={ap:.3f}\nn={n}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Average Precision (AP)", fontsize=11)
    ax.set_title(f"AP per Time-to-Event Group — {tag}", fontsize=11)
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, label="Random")
    plt.xticks(rotation=10, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def save_examples_table(df: pd.DataFrame, out_path: str, tag: str, n: int = 5):
    """Save a text table of best and worst predictions."""
    y_pred = (df["score"] >= 0.5).astype(int)
    df = df.copy()
    df["y_pred"]  = y_pred
    df["correct"] = (df["y_pred"] == df["ground_truth"]).astype(int)
    df["confidence_in_prediction"] = df["score"].apply(
        lambda s: abs(s - 0.5)
    )

    correct   = df[df["correct"] == 1].nlargest(n, "confidence_in_prediction")
    incorrect = df[df["correct"] == 0].nlargest(n, "confidence_in_prediction")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"  Prediction Examples -- {tag}\n")
        f.write(f"{'='*70}\n\n")

        f.write(f"{'-'*35}\n")
        f.write(f"  TOP {n} CORRECT PREDICTIONS\n")
        f.write(f"{'-'*35}\n")
        for _, row in correct.iterrows():
            gt_label  = "COLLISION" if row["ground_truth"] == 1 else "SAFE"
            verdict   = row.get("collision_verdict", "N/A")
            f.write(f"  [{row['video_id']}]  GT={gt_label}  "
                    f"Verdict={verdict}  Score={row['score']:.3f}\n")
            if row.get("verdict_reasoning"):
                f.write(f"    Reasoning: {str(row['verdict_reasoning'])[:120]}\n")

        f.write(f"\n{'-'*35}\n")
        f.write(f"  TOP {n} WRONG PREDICTIONS\n")
        f.write(f"{'-'*35}\n")
        for _, row in incorrect.iterrows():
            gt_label = "COLLISION" if row["ground_truth"] == 1 else "SAFE"
            verdict  = row.get("collision_verdict", "N/A")
            f.write(f"  [{row['video_id']}]  GT={gt_label}  "
                    f"Verdict={verdict}  Score={row['score']:.3f}\n")
            if row.get("verdict_reasoning"):
                f.write(f"    Reasoning: {str(row['verdict_reasoning'])[:120]}\n")

    print(f"  Saved: {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute metrics and generate graphs from evaluation results"
    )
    parser.add_argument("--results",   required=True,
                        help="Results JSONL (from zero_shot_eval.py)")
    parser.add_argument("--out_dir",   required=True,
                        help="Directory to save metrics JSON + graphs")
    parser.add_argument("--tag",       default="Evaluation",
                        help="Label used in graph titles (e.g. 'Zero-Shot Baseline')")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for binary metrics (default: 0.5)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading results: {args.results}")
    df = load_results(args.results)
    print(f"  Records loaded : {len(df)}")
    print(f"  Positives      : {int(df['ground_truth'].sum())}")
    print(f"  Negatives      : {int((df['ground_truth'] == 0).sum())}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics       = compute_metrics(df, args.threshold)
    group_metrics = compute_group_metrics(df, args.threshold)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Results: {args.tag}")
    print(f"{'='*55}")
    print(f"  Total clips       : {metrics['n_total']}")
    print(f"  Positives / Negatives : {metrics['n_positive']} / {metrics['n_negative']}")
    print(f"  -----------------------------------------")
    print(f"  Average Precision : {metrics['ap']:.4f}  <- primary metric")
    print(f"  AUC-ROC           : {metrics['auc_roc']:.4f}")
    print(f"  F1  (thr={args.threshold})  : {metrics['f1']:.4f}")
    print(f"  Precision         : {metrics['precision']:.4f}")
    print(f"  Recall            : {metrics['recall']:.4f}")
    print(f"  Accuracy          : {metrics['accuracy']:.4f}")
    print(f"  -----------------------------------------")
    print(f"  Confusion Matrix  (threshold={args.threshold}):")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TN={metrics['tn']}")
    print(f"  -----------------------------------------")
    print(f"  Mean score (pos)  : {metrics['mean_score_pos']:.4f}")
    print(f"  Mean score (neg)  : {metrics['mean_score_neg']:.4f}")
    print(f"  Optimal thr/F1    : {metrics['optimal_threshold']:.3f} / {metrics['optimal_f1']:.4f}")

    if group_metrics:
        print(f"\n  AP by group (time before event):")
        for label, gm in group_metrics.items():
            print(f"    {label}: AP={gm['ap']:.4f}  (n={gm['n']})")
    print(f"{'='*55}\n")

    # ── Save metrics JSON ─────────────────────────────────────────────────────
    metrics_to_save = {k: v for k, v in metrics.items() if not k.startswith("_")}
    metrics_to_save["group_metrics"] = group_metrics
    metrics_to_save["tag"]           = args.tag

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"  Metrics saved  : {metrics_path}")

    # ── Generate graphs ───────────────────────────────────────────────────────
    print("\nGenerating graphs...")
    plot_confusion_matrix(
        metrics,
        os.path.join(args.out_dir, "confusion_matrix.png"),
        args.tag
    )
    plot_roc_curve(
        metrics,
        os.path.join(args.out_dir, "roc_curve.png"),
        args.tag
    )
    plot_pr_curve(
        metrics,
        os.path.join(args.out_dir, "pr_curve.png"),
        args.tag
    )
    plot_score_distribution(
        df,
        os.path.join(args.out_dir, "score_distribution.png"),
        args.tag
    )
    plot_group_ap_bar(
        group_metrics,
        os.path.join(args.out_dir, "group_ap_bar.png"),
        args.tag
    )
    save_examples_table(
        df,
        os.path.join(args.out_dir, "examples_table.txt"),
        args.tag
    )

    print(f"\nAll outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
