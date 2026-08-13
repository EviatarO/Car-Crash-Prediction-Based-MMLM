"""Loss/AP/LR-vs-epoch curves for semsup_train.py's epoch_metrics.jsonl.

plot_training_curves.py exists but targets a DIFFERENT schema (train_lora.py's
train_f1/val_f1/val_loss fields, from the superseded InternVL3.5 pipeline) - none of
those field names exist in semsup_train.py's output, so it silently produces nothing
useful here. This script reads the actual fields the current trainer writes:
crash_loss, sem_loss, val_crash_loss, val_sem_loss, train_total_loss, val_total_loss,
train_val_gap, val_ap, lr, epoch_s.

    python student_training/scripts/plot_semsup_curves.py \
        --metrics /path/to/epoch_metrics.jsonl \
        --train-metrics /path/to/train_metrics.json \
        --out-dir /path/to/figures \
        --tag "A1-v2 (4,446 pool)"
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_epochs(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _mark_selected(ax, epochs, values, selected_epoch, label):
    if selected_epoch is None or selected_epoch not in epochs:
        return
    i = epochs.index(selected_epoch)
    if values[i] is None:
        return
    ax.scatter([selected_epoch], [values[i]], s=90, zorder=5, color="black",
               marker="*", label=f"selected (ep{selected_epoch})")


def plot_loss(rows, out_path, tag, selected_epoch):
    epochs = [r["epoch"] for r in rows]
    tr = [r.get("crash_loss") for r in rows]
    va = [r.get("val_crash_loss") for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, tr, "o-", label="train crash_loss", color="#2E5090")
    ax.plot(epochs, va, "o-", label="val crash_loss", color="#CC0000")
    _mark_selected(ax, epochs, va, selected_epoch, "selected")
    ax.set_xlabel("epoch"); ax.set_ylabel("crash loss (CE)")
    ax.set_title(f"Loss vs epoch — {tag}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_val_ap(rows, out_path, tag, selected_epoch):
    epochs = [r["epoch"] for r in rows]
    ap = [r.get("val_ap") for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, ap, "o-", color="#00B050")
    _mark_selected(ax, epochs, ap, selected_epoch, "selected")
    ax.set_xlabel("epoch"); ax.set_ylabel("val AP (per-clip)")
    ax.set_title(f"Val AP vs epoch — {tag}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_lr(rows, out_path, tag):
    epochs = [r["epoch"] for r in rows]
    lr = [r.get("lr") for r in rows]
    if all(v is None for v in lr):
        print(f"  SKIP {out_path}: no 'lr' field (older run, predates --lr-schedule)")
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, lr, "o-", color="#FFC000")
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate (end of epoch)")
    ax.set_title(f"LR schedule — {tag}")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def plot_gap(rows, out_path, tag, selected_epoch):
    epochs = [r["epoch"] for r in rows]
    gap = [r.get("train_val_gap") for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, gap, "o-", color="#7030A0")
    ax.axhline(0, color="grey", lw=1, ls="--")
    _mark_selected(ax, epochs, gap, selected_epoch, "selected")
    ax.set_xlabel("epoch"); ax.set_ylabel("val_total_loss − train_total_loss")
    ax.set_title(f"Overfit gap vs epoch — {tag}\n(>0 and growing = overfitting)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="path to epoch_metrics.jsonl")
    ap.add_argument("--train-metrics", default=None,
                     help="path to train_metrics.json, to mark the selected (best_epoch) point")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    rows = load_epochs(args.metrics)
    if not rows:
        raise SystemExit(f"No rows in {args.metrics}")

    selected_epoch = None
    if args.train_metrics:
        tm = json.load(open(args.train_metrics, encoding="utf-8"))
        selected_epoch = tm.get("best_epoch")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_loss(rows, out_dir / "loss_curve.png", args.tag, selected_epoch)
    plot_val_ap(rows, out_dir / "val_ap_curve.png", args.tag, selected_epoch)
    plot_lr(rows, out_dir / "lr_curve.png", args.tag)
    plot_gap(rows, out_dir / "train_val_gap.png", args.tag, selected_epoch)

    print(f"[figures] {out_dir}  ({len(rows)} epochs, selected={selected_epoch})")


if __name__ == "__main__":
    main()
