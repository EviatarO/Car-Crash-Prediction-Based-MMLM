"""The semantic positive: caption retrieval far above chance, and still scaling with data.

Two panels:
  (a) retrieval@1 by condition, against the chance line and the collapse control
  (b) retrieval@1 vs number of training captions, log-log, showing no saturation

Numbers read from outputs/e4_vjepa_reason/b1_taps/*.json so the figure can never drift
from the measured values.

    python student_training/scripts/make_semantic_positive_figure.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TAPS = ROOT / "outputs" / "e4_vjepa_reason" / "b1_taps"
OUT = ROOT / "reports" / "figures" / "semantic_retrieval_scaling_2026-08-22.png"

# E3a-status dark palette - the figure sits directly on the slide background
BG = "#1C2340"
C_BAR = "#00E676"
C_POOLED = "#00BFFF"
C_CTRL = "#5A6785"
C_CHANCE = "#FF6B6B"
FG = "#FFFFFF"
MUTED = "#A0B4CC"


def load(name):
    return json.load(open(TAPS / name, encoding="utf-8"))


def main():
    s25, s50, s100 = load("b1_metrics_scaling_25pct.json"), \
                     load("b1_metrics_scaling_50pct.json"), \
                     load("b1_metrics_scaling_100pct.json")
    pooled = load("b1_metrics_pooled.json")
    chance = s100["control_mean_embedding"]["chance_retrieval_clip"]
    ctrl = s100["control_mean_embedding"]["retrieval_top1_acc_clip"]

    plt.rcParams.update({
        "text.color": FG, "axes.labelcolor": FG, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.edgecolor": MUTED, "font.size": 11,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0))
    fig.patch.set_facecolor(BG)
    for a in (ax1, ax2):
        a.set_facecolor(BG)
        for sp in a.spines.values():
            sp.set_color(MUTED)

    # ---------------- panel (a): retrieval by condition
    labels = ["Collapse control\n(ignores the video)", "At the classifier's\nown bottleneck",
              "Full patch grid\n(all captions)"]
    vals = [ctrl * 100, pooled["held_out_retrieval_top1_acc_clip"] * 100,
            s100["held_out_retrieval_top1_acc_clip"] * 100]
    cols = [C_CTRL, C_POOLED, C_BAR]
    bars = ax1.bar(labels, vals, color=cols, edgecolor="none", width=0.6)

    ax1.axhline(chance * 100, color=C_CHANCE, linestyle="--", linewidth=1.8,
                label=f"chance = {chance*100:.2f}%  (1 in 221)")
    for b, v in zip(bars, vals):
        mult = v / (chance * 100)
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.45,
                 f"{v:.2f}%\n{mult:.0f}x chance", ha="center", va="bottom",
                 fontsize=11, fontweight="bold", color=FG)
    ax1.set_ylabel("clip-level retrieval@1  (%)", fontsize=12)
    ax1.set_title("(a)  From FROZEN features, can a probe pick\nthe right caption "
                  "out of 221 unseen clips?",
                  fontsize=12, fontweight="bold", color="#00BFFF")
    ax1.set_ylim(0, max(vals) * 1.42)
    ax1.legend(fontsize=10, loc="upper left", facecolor="#243060",
               edgecolor=MUTED, labelcolor=FG)
    ax1.grid(axis="y", alpha=0.18, color=MUTED)
    ax1.tick_params(labelsize=10.5)

    # ---------------- panel (b): scaling
    ns = np.array([s25["n_train"], s50["n_train"], s100["n_train"]], dtype=float)
    accs = np.array([s25["held_out_retrieval_top1_acc_clip"],
                     s50["held_out_retrieval_top1_acc_clip"],
                     s100["held_out_retrieval_top1_acc_clip"]]) * 100
    mults = accs / (chance * 100)

    ax2.plot(ns, accs, "o-", color=C_BAR, linewidth=2.2, markersize=9, zorder=3)
    for n, a, m in zip(ns, accs, mults):
        ax2.annotate(f"{m:.0f}x", (n, a), textcoords="offset points", xytext=(8, -14),
                     fontsize=12, fontweight="bold", color=C_BAR)
    ax2.axhline(chance * 100, color=C_CHANCE, linestyle="--", linewidth=1.8,
                label=f"chance = {chance*100:.2f}%")

    # power-law fit, reported as the slope only
    slope = np.polyfit(np.log(ns), np.log(accs), 1)[0]
    xs = np.linspace(ns[0] * 0.85, ns[-1] * 2.9, 50)
    ax2.plot(xs, np.exp(np.polyval(np.polyfit(np.log(ns), np.log(accs), 1), np.log(xs))),
             ":", color=MUTED, linewidth=1.6,
             label=f"power-law fit  (slope {slope:.2f})", zorder=1)
    ax2.axvline(4446, color="#FFA726", linestyle="-.", linewidth=1.6, alpha=0.9)
    # sit BELOW the fitted line and left of the marker: above it the text crosses the fit,
    # to the right it runs into the legend and the axis edge
    ax2.text(4200, accs[-1] * 0.40, "full pool\navailable\n(4,446)",
             fontsize=9.5, color="#FFA726", va="center", ha="right", fontweight="bold")

    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("training rows (captions used to fit the probe)", fontsize=11)
    ax2.set_ylabel("clip-level retrieval@1  (%)", fontsize=12)
    ax2.set_title("(b)  More captions keep helping\n(no sign of levelling off)",
                  fontsize=12.5, fontweight="bold", color="#00BFFF")
    ax2.legend(fontsize=9.5, loc="lower right", facecolor="#243060",
               edgecolor=MUTED, labelcolor=FG)
    ax2.grid(alpha=0.18, which="both", color=MUTED)
    ax2.tick_params(labelsize=10.5)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {OUT}")
    print(f"  chance={chance*100:.3f}%  control={ctrl*100:.3f}%  "
          f"pooled={pooled['held_out_retrieval_top1_acc_clip']*100:.2f}%  "
          f"full={accs[-1]:.2f}%  slope={slope:.3f}")


if __name__ == "__main__":
    main()
