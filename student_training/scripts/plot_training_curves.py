"""
plot_training_curves.py
=======================
Read epoch_metrics.jsonl (written by train_lora.py once per epoch) and produce
diagnostic curves over the whole training run.

Why this script exists
----------------------
`evaluate_metrics.py` only reports metrics for ONE checkpoint at a time
(the JSONL it's given is the result of running `trained_eval.py` on a single
`step_XXXXXX/` folder). For thesis figures showing "how the model trained
over time" we need the per-epoch numbers from `epoch_metrics.jsonl`.

Note on what these numbers mean
--------------------------------
`epoch_metrics.jsonl` is computed on the 80/20 split of the 100-clip TRAINING
set (train_f1, val_f1, val_loss). It is NOT the held-out 677-clip private
test set. Therefore the curves below show optimisation behaviour and
checkpoint selection — not generalisation. The single test-set AP/F1 from
`evaluate_metrics.py` on `step_000270_test.jsonl` is the generalisation
number.

Outputs (all PNGs):
  1. f1_curves.png           — train_f1 vs val_f1 across epochs
  2. ap_curves.png           — train_ap vs val_ap across epochs
  3. val_loss_curve.png      — val_loss across epochs
  4. precision_curves.png    — train_precision vs val_precision (skipped if missing)
  5. recall_curves.png       — train_recall vs val_recall       (skipped if missing)
  6. combined.png            — 2x2 panel: F1, AP, val_loss, gap (always)
  7. combined_pr.png         — 2x3 panel including precision + recall (if present)
  8. summary.json            — best epoch by val_f1, chosen epoch info, gap

Note: precision and recall are only present in epoch_metrics.jsonl from runs
trained with train_lora.py revision >= "fix(train): log precision+recall".
Older runs (e.g. e2_lora_100clips) only have F1/AP — the precision and recall
plots will be skipped automatically with a printed warning.

Usage:
  python student_training/scripts/plot_training_curves.py \
      --metrics outputs/checkpoints/checkpoints/e2_lora_100clips/epoch_metrics.jsonl \
      --out_dir outputs/metrics/e2_lora_100clips_training \
      --tag     "E2 LoRA 100 clips" \
      --chosen_epoch 27
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Loading
# ────────────────────────────────────────────────────────────────────────────

def load_epoch_metrics(path: str) -> list:
    """
    Load epoch_metrics.jsonl, sorted by step. Handles the case where training
    was resumed and the file contains multiple runs concatenated (the step
    counter restarts), by keeping only the last occurrence of each step.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # If training was resumed mid-run, the same step may appear twice.
    # Keep the LATEST entry per step (later in the file = newer run).
    by_step = {}
    for r in rows:
        by_step[int(r["step"])] = r
    rows = [by_step[s] for s in sorted(by_step.keys())]

    # Ensure the epoch counter is monotonic too (older logs sometimes reset).
    # We re-derive a "global epoch" 1..N for plotting.
    for i, r in enumerate(rows, start=1):
        r["global_epoch"] = i
    return rows


# ────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "train":  "#3498db",
    "val":    "#e74c3c",
    "loss":   "#2c3e50",
    "marker": "#f1c40f",
}


def _annotate_chosen(ax, epochs, values, chosen_epoch, label, color):
    """Draw a vertical line and a dot on `chosen_epoch` showing its value."""
    if chosen_epoch is None:
        return
    if chosen_epoch < 1 or chosen_epoch > len(epochs):
        return
    x = epochs[chosen_epoch - 1]
    y = values[chosen_epoch - 1]
    ax.axvline(x=x, color=PALETTE["marker"], linestyle="--", lw=1, alpha=0.6)
    ax.plot([x], [y], "o", color=PALETTE["marker"], markersize=8,
            markeredgecolor="black", zorder=5,
            label=f"Chosen: epoch {chosen_epoch} ({label}={y:.3f})")


def plot_f1(rows, out_path, tag, chosen_epoch):
    epochs    = [r["global_epoch"] for r in rows]
    train_f1  = [r.get("train_f1") for r in rows]
    val_f1    = [r.get("val_f1")   for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_f1, color=PALETTE["train"], lw=2, label="Train F1")
    ax.plot(epochs, val_f1,   color=PALETTE["val"],   lw=2, label="Val F1")
    _annotate_chosen(ax, epochs, val_f1, chosen_epoch, "val_f1", PALETTE["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_title(f"F1 across epochs — {tag}")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ap(rows, out_path, tag, chosen_epoch):
    epochs    = [r["global_epoch"] for r in rows]
    train_ap  = [r.get("train_ap") for r in rows]
    val_ap    = [r.get("val_ap")   for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, train_ap, color=PALETTE["train"], lw=2, label="Train AP")
    ax.plot(epochs, val_ap,   color=PALETTE["val"],   lw=2, label="Val AP")
    _annotate_chosen(ax, epochs, val_ap, chosen_epoch, "val_ap", PALETTE["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title(f"AP across epochs — {tag}")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def plot_val_loss(rows, out_path, tag, chosen_epoch):
    epochs   = [r["global_epoch"] for r in rows]
    val_loss = [r.get("val_loss") for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs, val_loss, color=PALETTE["loss"], lw=2, label="Val loss")
    _annotate_chosen(ax, epochs, val_loss, chosen_epoch, "val_loss", PALETTE["loss"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"Val loss across epochs — {tag}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


def _has_field(rows, field) -> bool:
    """True if at least one row has a non-None value for `field`."""
    return any(r.get(field) is not None for r in rows)


def plot_precision(rows, out_path, tag, chosen_epoch):
    if not (_has_field(rows, "train_precision") or _has_field(rows, "val_precision")):
        print(f"  SKIP {out_path}: precision not in epoch_metrics.jsonl "
              f"(this run was trained before precision/recall logging was added).")
        return False

    epochs    = [r["global_epoch"] for r in rows]
    train_p   = [r.get("train_precision") for r in rows]
    val_p     = [r.get("val_precision")   for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if _has_field(rows, "train_precision"):
        ax.plot(epochs, train_p, color=PALETTE["train"], lw=2, label="Train Precision")
    if _has_field(rows, "val_precision"):
        ax.plot(epochs, val_p,   color=PALETTE["val"],   lw=2, label="Val Precision")
        _annotate_chosen(ax, epochs, val_p, chosen_epoch, "val_precision", PALETTE["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision across epochs — {tag}")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return True


def plot_recall(rows, out_path, tag, chosen_epoch):
    if not (_has_field(rows, "train_recall") or _has_field(rows, "val_recall")):
        print(f"  SKIP {out_path}: recall not in epoch_metrics.jsonl "
              f"(this run was trained before precision/recall logging was added).")
        return False

    epochs    = [r["global_epoch"] for r in rows]
    train_r   = [r.get("train_recall") for r in rows]
    val_r     = [r.get("val_recall")   for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if _has_field(rows, "train_recall"):
        ax.plot(epochs, train_r, color=PALETTE["train"], lw=2, label="Train Recall")
    if _has_field(rows, "val_recall"):
        ax.plot(epochs, val_r,   color=PALETTE["val"],   lw=2, label="Val Recall")
        _annotate_chosen(ax, epochs, val_r, chosen_epoch, "val_recall", PALETTE["val"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title(f"Recall across epochs — {tag}")
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")
    return True


def plot_combined(rows, out_path, tag, chosen_epoch):
    epochs    = [r["global_epoch"] for r in rows]
    train_f1  = [r.get("train_f1") for r in rows]
    val_f1    = [r.get("val_f1")   for r in rows]
    train_ap  = [r.get("train_ap") for r in rows]
    val_ap    = [r.get("val_ap")   for r in rows]
    val_loss  = [r.get("val_loss") for r in rows]
    f1_gap    = [r.get("f1_gap")   for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # F1
    ax = axes[0, 0]
    ax.plot(epochs, train_f1, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_f1,   color=PALETTE["val"],   lw=2, label="Val")
    _annotate_chosen(ax, epochs, val_f1, chosen_epoch, "val_f1", PALETTE["val"])
    ax.set_title("F1"); ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.3); ax.legend()

    # AP
    ax = axes[0, 1]
    ax.plot(epochs, train_ap, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_ap,   color=PALETTE["val"],   lw=2, label="Val")
    _annotate_chosen(ax, epochs, val_ap, chosen_epoch, "val_ap", PALETTE["val"])
    ax.set_title("Average Precision"); ax.set_xlabel("Epoch"); ax.set_ylabel("AP")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.3); ax.legend()

    # Val loss
    ax = axes[1, 0]
    ax.plot(epochs, val_loss, color=PALETTE["loss"], lw=2)
    _annotate_chosen(ax, epochs, val_loss, chosen_epoch, "val_loss", PALETTE["loss"])
    ax.set_title("Validation loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss")
    ax.grid(alpha=0.3)

    # F1 gap (overfitting indicator)
    ax = axes[1, 1]
    ax.plot(epochs, f1_gap, color="#9b59b6", lw=2, label="train_f1 - val_f1")
    ax.axhline(y=0, color="gray", linestyle="--", lw=1)
    _annotate_chosen(ax, epochs, f1_gap, chosen_epoch, "gap", "#9b59b6")
    ax.set_title("Train–Val F1 gap (overfitting indicator)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 gap")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(f"Training diagnostics — {tag}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_combined_pr(rows, out_path, tag, chosen_epoch):
    """2x3 panel: F1, AP, val_loss / Precision, Recall, train-val F1 gap."""
    epochs    = [r["global_epoch"] for r in rows]
    train_f1  = [r.get("train_f1")        for r in rows]
    val_f1    = [r.get("val_f1")          for r in rows]
    train_ap  = [r.get("train_ap")        for r in rows]
    val_ap    = [r.get("val_ap")          for r in rows]
    train_p   = [r.get("train_precision") for r in rows]
    val_p     = [r.get("val_precision")   for r in rows]
    train_r   = [r.get("train_recall")    for r in rows]
    val_r     = [r.get("val_recall")      for r in rows]
    val_loss  = [r.get("val_loss")        for r in rows]
    f1_gap    = [r.get("f1_gap")          for r in rows]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    # Row 1: F1, Precision, Recall
    ax = axes[0, 0]
    ax.plot(epochs, train_f1, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_f1,   color=PALETTE["val"],   lw=2, label="Val")
    _annotate_chosen(ax, epochs, val_f1, chosen_epoch, "val_f1", PALETTE["val"])
    ax.set_title("F1"); ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.3); ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, train_p, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_p,   color=PALETTE["val"],   lw=2, label="Val")
    _annotate_chosen(ax, epochs, val_p, chosen_epoch, "val_precision", PALETTE["val"])
    ax.set_title("Precision"); ax.set_xlabel("Epoch"); ax.set_ylabel("Precision")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.3); ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, train_r, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_r,   color=PALETTE["val"],   lw=2, label="Val")
    _annotate_chosen(ax, epochs, val_r, chosen_epoch, "val_recall", PALETTE["val"])
    ax.set_title("Recall"); ax.set_xlabel("Epoch"); ax.set_ylabel("Recall")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.3); ax.legend()

    # Row 2: AP, val_loss, F1 gap
    ax = axes[1, 0]
    ax.plot(epochs, train_ap, color=PALETTE["train"], lw=2, label="Train")
    ax.plot(epochs, val_ap,   color=PALETTE["val"],   lw=2, label="Val")
    _annotate_chosen(ax, epochs, val_ap, chosen_epoch, "val_ap", PALETTE["val"])
    ax.set_title("Average Precision"); ax.set_xlabel("Epoch"); ax.set_ylabel("AP")
    ax.set_ylim([0, 1.05]); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, val_loss, color=PALETTE["loss"], lw=2)
    _annotate_chosen(ax, epochs, val_loss, chosen_epoch, "val_loss", PALETTE["loss"])
    ax.set_title("Validation loss"); ax.set_xlabel("Epoch"); ax.set_ylabel("Val loss")
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(epochs, f1_gap, color="#9b59b6", lw=2, label="train_f1 - val_f1")
    ax.axhline(y=0, color="gray", linestyle="--", lw=1)
    _annotate_chosen(ax, epochs, f1_gap, chosen_epoch, "gap", "#9b59b6")
    ax.set_title("Train–Val F1 gap (overfit indicator)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1 gap")
    ax.grid(alpha=0.3); ax.legend()

    fig.suptitle(f"Training diagnostics (with P/R) — {tag}", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────

def best_epoch_summary(rows: list, chosen_epoch: int = None) -> dict:
    """Pick the row that maximises val_f1 (with ties broken by smaller f1_gap)."""
    def _key(r):
        # Maximise val_f1, then prefer smaller positive gap (less overfit),
        # then smaller val_loss as a final tiebreaker.
        return (r.get("val_f1", 0.0),
                -abs(r.get("f1_gap", 0.0)),
                -r.get("val_loss", 1e9))

    best = max(rows, key=_key)

    def _row_summary(r):
        return {
            "global_epoch":    r["global_epoch"],
            "step":            r.get("step"),
            "ckpt_dir":        r.get("ckpt_dir"),
            "train_f1":        r.get("train_f1"),
            "train_precision": r.get("train_precision"),
            "train_recall":    r.get("train_recall"),
            "val_f1":          r.get("val_f1"),
            "val_precision":   r.get("val_precision"),
            "val_recall":      r.get("val_recall"),
            "train_ap":        r.get("train_ap"),
            "val_ap":          r.get("val_ap"),
            "val_loss":        r.get("val_loss"),
            "f1_gap":          r.get("f1_gap"),
        }

    out = {
        "n_epochs":         len(rows),
        "best_by_val_f1":   _row_summary(best),
    }
    if chosen_epoch is not None and 1 <= chosen_epoch <= len(rows):
        out["chosen_epoch"] = _row_summary(rows[chosen_epoch - 1])
    return out


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-epoch training curves from epoch_metrics.jsonl"
    )
    parser.add_argument("--metrics", required=True,
                        help="Path to epoch_metrics.jsonl produced by train_lora.py")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to save PNGs and summary.json")
    parser.add_argument("--tag", default="LoRA training",
                        help="Label used in graph titles")
    parser.add_argument("--chosen_epoch", type=int, default=None,
                        help="Mark this global epoch on the curves (1-indexed). "
                             "E.g. 27 for step_000270 in the e2_lora_100clips run.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading: {args.metrics}")
    rows = load_epoch_metrics(args.metrics)
    print(f"  Epochs found: {len(rows)}")

    plot_f1      (rows, os.path.join(args.out_dir, "f1_curves.png"),        args.tag, args.chosen_epoch)
    plot_ap      (rows, os.path.join(args.out_dir, "ap_curves.png"),        args.tag, args.chosen_epoch)
    plot_val_loss(rows, os.path.join(args.out_dir, "val_loss_curve.png"),   args.tag, args.chosen_epoch)
    has_p = plot_precision(rows, os.path.join(args.out_dir, "precision_curves.png"), args.tag, args.chosen_epoch)
    has_r = plot_recall   (rows, os.path.join(args.out_dir, "recall_curves.png"),    args.tag, args.chosen_epoch)
    plot_combined(rows, os.path.join(args.out_dir, "combined.png"),         args.tag, args.chosen_epoch)
    if has_p and has_r:
        plot_combined_pr(rows, os.path.join(args.out_dir, "combined_pr.png"), args.tag, args.chosen_epoch)

    summary = best_epoch_summary(rows, args.chosen_epoch)
    summary["tag"] = args.tag
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Best by val_f1: epoch {summary['best_by_val_f1']['global_epoch']} "
          f"(step {summary['best_by_val_f1']['step']}) "
          f"val_f1={summary['best_by_val_f1']['val_f1']}  "
          f"val_ap={summary['best_by_val_f1']['val_ap']}")
    if "chosen_epoch" in summary:
        c = summary["chosen_epoch"]
        print(f"  Chosen epoch  : {c['global_epoch']} (step {c['step']}) "
              f"val_f1={c['val_f1']}  val_ap={c['val_ap']}")
    print(f"  Saved: {summary_path}")
    print(f"\nAll outputs → {args.out_dir}")


if __name__ == "__main__":
    main()






