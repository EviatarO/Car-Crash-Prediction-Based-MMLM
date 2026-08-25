"""Train-vs-val loss-per-epoch figure, one per semantic-supervision arm, in the same
visual style as the E3a InternVL3.5 reference figure (blue circle train / red square val /
orange dashed best-checkpoint line).

Reads each arm's epoch_metrics.jsonl (semsup_train.py's schema) directly — no aggregation,
no recomputation, just the total optimized loss (train_total_loss/val_total_loss, which
equals crash_loss alone for the crash-only arms since semantic_weight=0 there).

    python student_training/scripts/plot_loss_vs_epoch_all_arms.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "figures" / "loss_curves"

# (arm_dir, display title, best_epoch or None, note)
ARMS = [
    ("a1_1761", "A1_1761 — crash-only LoRA (champion, test AP 0.900)", 4, None),
    ("b_1761_par", "B_1761 parallel — joint InfoNCE, V10 (leaky) captions", 4, None),
    ("b_v2_1761", "B-v2 — joint InfoNCE, V12 (clean) captions", 2, None),
    ("b_v3_1761", "B-v3 — joint InfoNCE, warm-started + per-group clip", 10,
     "local file only has epochs 9-12 (epochs 1-8 overwritten locally by a later\n"
     "12-epoch extension pull; correct 1-8 file still exists on the pod, not yet re-pulled)"),
    ("p1_stageB", "P1 Stage B — crash-only, LoRA warm-started from Stage A", 2, None),
]


def load_epochs(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["epoch"])
    return rows


def plot_arm(arm_dir, title, best_epoch, note):
    metrics_path = ROOT / "outputs" / "e4_vjepa_reason" / arm_dir / "epoch_metrics.jsonl"
    if not metrics_path.exists():
        print(f"  SKIP {arm_dir}: no epoch_metrics.jsonl")
        return None

    rows = load_epochs(metrics_path)
    epochs = [r["epoch"] for r in rows]
    train_loss = [r.get("train_total_loss") for r in rows]
    val_loss = [r.get("val_total_loss") for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(epochs, train_loss, "o-", color="#2E5090", label="Train loss (avg/epoch)",
             markersize=6, linewidth=1.8)
    ax.plot(epochs, val_loss, "s-", color="#CC0000", label="Val loss",
             markersize=6, linewidth=1.8)

    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(best_epoch, color="#FF8C00", linestyle="--", linewidth=1.6,
                    label=f"Best ckpt (ep{best_epoch})")

    n_ep = len(epochs)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss vs. Validation Loss\n{title} ({n_ep} epochs)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    if note:
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=7.5,
                 color="dimgray", va="bottom", ha="left", style="italic")

    fig.tight_layout()
    out_path = OUT_DIR / f"loss_vs_epoch_{arm_dir}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# 2x2 grid for the status deck. B-v3 is deliberately excluded: its local
# epoch_metrics.jsonl holds only epochs 9-12 (1-8 were overwritten by a later 12-epoch
# extension pull; the correct file is still on the pod). A partial curve reads as broken.
GRID_ARMS = [
    ("a1_1761", "A1_1761  -  crash-only  (champion, test AP 0.900)", 4),
    ("b_1761_par", "B_1761 parallel  -  joint, V10 captions  (0.890)", 4),
    ("b_v2_1761", "B-v2  -  joint, V12 clean captions  (0.880)", 2),
    ("p1_stageB", "P1 Stage B  -  two-stage, warm-started  (0.827)", 2),
]


# dark theme, matching the 2026-08 status deck background (E3a-status palette)
BG = "#1C2340"
FG = "#FFFFFF"
MUTED_C = "#A0B4CC"


def plot_grid():
    plt.rcParams.update({
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED_C,
        "ytick.color": MUTED_C, "axes.edgecolor": MUTED_C,
    })
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0))
    fig.patch.set_facecolor(BG)
    for ax, (arm_dir, title, best_epoch) in zip(axes.ravel(), GRID_ARMS):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(MUTED_C)
        rows = load_epochs(ROOT / "outputs" / "e4_vjepa_reason" / arm_dir / "epoch_metrics.jsonl")
        epochs = [r["epoch"] for r in rows]
        ax.plot(epochs, [r.get("train_total_loss") for r in rows], "o-",
                color="#00BFFF", label="Train loss", markersize=4.5, linewidth=1.7)
        ax.plot(epochs, [r.get("val_total_loss") for r in rows], "s-",
                color="#FF6B6B", label="Val loss", markersize=4.5, linewidth=1.7)
        if best_epoch in epochs:
            ax.axvline(best_epoch, color="#FFA726", linestyle="--", linewidth=1.5,
                       label=f"selected (ep{best_epoch})")
        ax.set_title(title, fontsize=10.5, fontweight="bold", color="#00BFFF")
        ax.set_xlabel("Epoch", fontsize=10.5)
        ax.set_ylabel("Loss", fontsize=10.5)
        ax.grid(alpha=0.18, color=MUTED_C)
        ax.legend(fontsize=8.5, facecolor="#243060", edgecolor=MUTED_C,
                  labelcolor=FG)
        ax.tick_params(labelsize=9.5)
    fig.suptitle("Training vs Validation Loss  -  every arm overfits after epoch 2-3",
                 fontsize=13, fontweight="bold", color="#00BFFF", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out = OUT_DIR / "loss_grid_4arms_2026-08-22.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for arm_dir, title, best_epoch, note in ARMS:
        out = plot_arm(arm_dir, title, best_epoch, note)
        if out:
            print(f"  wrote {out}")
            written.append(out)
    print(f"\n{len(written)}/{len(ARMS)} per-arm figures written to {OUT_DIR}")
    print(f"  wrote {plot_grid()}")


if __name__ == "__main__":
    main()
