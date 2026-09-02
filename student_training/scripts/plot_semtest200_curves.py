"""
plot_semtest200_curves.py
===========================
Loss-vs-epoch and val_ap-vs-epoch figures for the SemTest-200 arms (vision / v10 /
v12 / v12shuf), dark-theme styling matching plot_loss_vs_epoch_all_arms.py's grid.

Checkpoint selection shown as a dashed line = argmax val_ap (ties -> later epoch),
the trainer's own --select-by val_ap rule (semsup_train.py's `ranked` sort). At small
val n this ranking is noisy - see the annotated margin between the top-2 epochs in
each panel's title.

SHARED Y-AXIS (2026-08-29 fix): the 2x2 loss grid previously let each panel
autoscale its own crash_loss (and, for the 3 semantic arms, its own twin-axis
sem_loss) independently, and the vision panel had no twin axis at all - so the four
panels were not visually comparable (a curve that LOOKED lower in one panel could be
numerically higher than another), and the panels had different plot geometry. Fixed:
one global crash_loss y-limit and one global sem_loss y-limit across all panels
(computed from the actual data, not hardcoded), and an empty twin axis is drawn on
the vision panel so all four panels share identical geometry and tick positions.

Usage:
  python plot_semtest200_curves.py
  python plot_semtest200_curves.py --results-dir ../../outputs/semtest200_v2/results \
      --out-dir ../../outputs/semtest200_v2/figures --val-n 300
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

# (arm_dir, display title, selected_epoch, has_semantic_loss)
DEFAULT_ARMS = [
    ("vision", "vision-only (crash-loss control)", 8, False),
    ("v10", "v10 - InfoNCE, V10 captions (leaky)", 8, True),
    ("v12", "v12 - InfoNCE, V12 captions (clean)", 10, True),
    ("v12shuf", "v12shuf - InfoNCE, V12 shuffled (content-destroyed)", 10, True),
]

BG = "#1C2340"
FG = "#FFFFFF"
MUTED_C = "#A0B4CC"
TRAIN_C = "#00BFFF"
VAL_C = "#FF6B6B"
SEM_TRAIN_C = "#7CFC98"
SEM_VAL_C = "#FFD166"
SEL_C = "#FFA726"


def load_epochs(results_dir, arm_dir):
    path = results_dir / arm_dir / "epoch_metrics.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _padded_range(values, pad_frac=0.05):
    lo, hi = min(values), max(values)
    span = hi - lo
    if span == 0:
        span = abs(hi) if hi else 1.0
    return lo - pad_frac * span, hi + pad_frac * span


def plot_loss_grid(results_dir, out_dir, arms):
    plt.rcParams.update({
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED_C,
        "ytick.color": MUTED_C, "axes.edgecolor": MUTED_C,
    })
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.0))
    fig.patch.set_facecolor(BG)

    all_rows = {arm_dir: load_epochs(results_dir, arm_dir) for arm_dir, *_ in arms}

    # ---- one global y-range for crash_loss, one for sem_loss, shared by every panel ----
    crash_vals = [v for rows in all_rows.values() for r in rows
                  for v in (r["crash_loss"], r["val_crash_loss"])]
    crash_ylim = _padded_range(crash_vals)
    sem_vals = [v for (arm_dir, _, _, has_sem) in arms if has_sem
                for r in all_rows[arm_dir] for v in (r["sem_loss"], r["val_sem_loss"])]
    sem_ylim = _padded_range(sem_vals) if sem_vals else (0.0, 1.0)

    for ax, (arm_dir, title, sel_ep, has_sem) in zip(axes.ravel(), arms):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(MUTED_C)
        rows = all_rows[arm_dir]
        epochs = [r["epoch"] for r in rows]

        ax.plot(epochs, [r["crash_loss"] for r in rows], "o-", color=TRAIN_C,
                 label="Train crash_loss", markersize=4.5, linewidth=1.7)
        ax.plot(epochs, [r["val_crash_loss"] for r in rows], "s-", color=VAL_C,
                 label="Val crash_loss", markersize=4.5, linewidth=1.7)
        ax.set_ylim(*crash_ylim)

        # Every panel gets a twin axis, even vision (which plots nothing on it) -
        # this keeps plot geometry and tick layout IDENTICAL across all 4 panels,
        # which independent-axis panels previously did not (vision had no twin at all).
        ax2 = ax.twinx()
        ax2.set_facecolor(BG)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.set_ylim(*sem_ylim)
        if has_sem:
            ax2.plot(epochs, [r["sem_loss"] for r in rows], "^--", color=SEM_TRAIN_C,
                      label="Train sem_loss", markersize=4, linewidth=1.3, alpha=0.85)
            ax2.plot(epochs, [r["val_sem_loss"] for r in rows], "v--", color=SEM_VAL_C,
                      label="Val sem_loss", markersize=4, linewidth=1.3, alpha=0.85)
        ax2.set_ylabel("InfoNCE sem_loss", fontsize=9, color=MUTED_C)
        ax2.tick_params(labelsize=8.5, colors=MUTED_C)

        if sel_ep in epochs:
            ax.axvline(sel_ep, color=SEL_C, linestyle="--", linewidth=1.5,
                        label=f"selected (ep{sel_ep})")

        # margin annotation: gap between top-2 val_ap epochs (noise indicator at small n)
        ranked = sorted(rows, key=lambda r: (-r["val_ap"], r["epoch"]))
        margin = ranked[0]["val_ap"] - ranked[1]["val_ap"] if len(ranked) > 1 else float("nan")
        ax.set_title(f"{title}\nval_ap margin vs runner-up: {margin:+.4f}",
                      fontsize=9.5, fontweight="bold", color=TRAIN_C)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("crash_loss (cross-entropy)", fontsize=10)
        ax.grid(alpha=0.18, color=MUTED_C)

        h1, l1 = ax.get_legend_handles_labels()
        if has_sem:
            h2, l2 = ax2.get_legend_handles_labels()
            h1, l1 = h1 + h2, l1 + l2
        ax.legend(h1, l1, fontsize=7.5, facecolor="#243060", edgecolor=MUTED_C,
                   labelcolor=FG, loc="upper right")
        ax.tick_params(labelsize=9)

    fig.suptitle("SemTest-200: Train vs Val crash_loss per arm  -  dashed line = selected "
                 "checkpoint  -  y-axes shared across all 4 panels",
                 fontsize=13, fontweight="bold", color=TRAIN_C, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = out_dir / "loss_curves_2x2.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def plot_val_ap_overlay(results_dir, out_dir, arms, val_n):
    plt.rcParams.update({
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED_C,
        "ytick.color": MUTED_C, "axes.edgecolor": MUTED_C,
    })
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for sp in ax.spines.values():
        sp.set_color(MUTED_C)

    colors = {"vision": "#00BFFF", "v10": "#FF6B6B", "v12": "#7CFC98", "v12shuf": "#FFD166"}
    for arm_dir, title, sel_ep, has_sem in arms:
        rows = load_epochs(results_dir, arm_dir)
        epochs = [r["epoch"] for r in rows]
        val_ap = [r["val_ap"] for r in rows]
        ax.plot(epochs, val_ap, "o-", color=colors.get(arm_dir, FG), label=title,
                 markersize=5, linewidth=1.8)
        sel_val_ap = next(r["val_ap"] for r in rows if r["epoch"] == sel_ep)
        ax.scatter([sel_ep], [sel_val_ap], color=colors.get(arm_dir, FG), s=140,
                    marker="*", zorder=5, edgecolors=FG, linewidths=0.8)

    ax.axhline(0.5, color=MUTED_C, linestyle=":", linewidth=1.2, alpha=0.7)
    ax.text(0.3, 0.505, "chance (balanced val)", fontsize=8, color=MUTED_C, style="italic")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(f"val_ap (n={val_n} clips)", fontsize=11)
    ax.set_title(f"SemTest-200: val AP vs epoch, all {len(arms)} arms  (* = selected checkpoint)",
                  fontsize=12.5, fontweight="bold", color="#00BFFF")
    ax.grid(alpha=0.18, color=MUTED_C)
    ax.legend(fontsize=9, facecolor="#243060", edgecolor=MUTED_C, labelcolor=FG,
               loc="lower right")
    ax.tick_params(labelsize=9.5)

    fig.tight_layout()
    out = out_dir / "val_ap_vs_epoch.png"
    fig.savefig(out, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=str(ROOT / "outputs" / "semtest200" / "results"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "semtest200" / "figures"))
    ap.add_argument("--val-n", type=int, default=40,
                     help="val pool size shown in the val_ap axis label (40 for the "
                          "original SemTest-200 single split, or the fold size / 300 "
                          "for a v2 CV run)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  wrote {plot_loss_grid(results_dir, out_dir, DEFAULT_ARMS)}")
    print(f"  wrote {plot_val_ap_overlay(results_dir, out_dir, DEFAULT_ARMS, args.val_n)}")


if __name__ == "__main__":
    main()
