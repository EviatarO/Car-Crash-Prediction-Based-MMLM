"""Dataset + caption-generation pipeline figure for the 2026-08-22 status deck.

Shows how the 1,761-window caption pool was actually assembled: the full Nexar window
pool, the A0 inference pass that mined the failures, the matched TP/TN top-up, and the
teacher captioning run.

Counts are read from the real files so the figure cannot drift from the data.

    python student_training/scripts/make_dataset_figure_2026-08-22.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "outputs" / "semantic_captions"
OUT = ROOT / "reports" / "figures" / "dataset_pipeline_2026-08-22.png"

BG = "#1C2340"
PANEL = "#243060"
PANEL_DK = "#202A50"
CYAN = "#00BFFF"
CYAN_DK = "#0096CC"
WHITE = "#FFFFFF"
MUTED = "#A0B4CC"
GREEN = "#00E676"
ORANGE = "#FFA726"


def counts():
    pool = sum(1 for _ in open(CAP / "Pool_Train4500_Full_4446.jsonl", encoding="utf-8"))
    fails = sum(1 for _ in open(CAP / "Caption_Train4500_Failures_587.jsonl", encoding="utf-8"))
    rows = [json.loads(l) for l in open(CAP / "Caption_V12_Neutral_1761_fortrain.jsonl",
                                        encoding="utf-8")]
    vids = len({json.loads(l)["video_id"]
                for l in open(CAP / "Pool_Train4500_Full_4446.jsonl", encoding="utf-8")})
    return pool, vids, fails, len(rows)


def block(ax, x, y, w, h, title, lines, accent=CYAN):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.02",
                                facecolor=PANEL, edgecolor=accent, linewidth=1.2, zorder=2))
    ax.add_patch(Rectangle((x, y + h - 0.035), w, 0.035, facecolor=accent,
                           edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h - 0.115, title, ha="center", va="center", fontsize=10.5,
            fontweight="bold", color=accent, zorder=4)
    ax.text(x + w / 2, y + h * 0.34, "\n".join(lines), ha="center", va="center",
            fontsize=8.6, color=WHITE, zorder=4, linespacing=1.55)


def arrow(ax, x1, y1, x2, y2, color=CYAN):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 linewidth=1.5, color=color, zorder=5, shrinkA=0, shrinkB=0))


def main():
    pool, vids, fails, total = counts()
    matched = total - fails

    fig, ax = plt.subplots(figsize=(11.6, 3.4))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y, h = 0.34, 0.60
    cy = y + h / 2
    w = 0.163
    xs = [0.010, 0.208, 0.406, 0.604, 0.822]

    block(ax, xs[0], y, w, h, "Nexar windows",
          [f"{vids:,} clips x 3 windows", f"= {pool:,} total", "balanced YES / NO"], MUTED)
    block(ax, xs[1], y, w, h, "A0 inference",
          ["score all windows with", "the frozen baseline", "(no training)"], CYAN_DK)
    block(ax, xs[2], y, w, h, "mine failures",
          [f"{fails} windows where", "the baseline was wrong", "(the hard cases)"], ORANGE)
    block(ax, xs[3], y, w, h, "top up",
          [f"+ {matched:,} windows the", "baseline got right",
           f"= {total:,} to caption"], GREEN)
    block(ax, xs[4], y, 0.168, h, "teacher captions",
          ["gemini-3.6-flash", "via OpenRouter", f"{total:,} captions - $32.82"], CYAN)

    for i in range(4):
        x1 = xs[i] + (w if i < 4 else 0)
        arrow(ax, x1, cy, xs[i + 1], cy)

    ax.text(0.906, 0.265, "captions .jsonl", ha="center", fontsize=8.5, color=CYAN,
            style="italic")

    ax.text(0.5, 0.115,
            "The pool is deliberately enriched with hard cases  -  2 parts correct : 1 part "
            "mined failure.",
            ha="center", fontsize=9.2, color=WHITE, fontweight="bold")
    ax.text(0.5, 0.030,
            "TTE buckets are only approximately balanced: positives 253 / 268 / 335 at 0.5 / 1.0 / "
            "1.5 s;  negatives have no event to count down to, so they use midpoint offsets "
            "(287 / 298 / 320).",
            ha="center", fontsize=7.8, color=MUTED, style="italic")

    fig.savefig(OUT, dpi=220, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"  wrote {OUT}")
    print(f"  pool={pool} vids={vids} fails={fails} matched={matched} total={total}")


if __name__ == "__main__":
    main()
