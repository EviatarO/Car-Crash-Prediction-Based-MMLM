"""Architecture figures for the 2026-08-22 status deck (dark theme, E3a-status palette).

Three levels:
  L1  the idea       - conceptual, no tensor shapes
  L2  inference      - full vision path, every block annotated, shapes on the arrows
  L3  training       - both branches, loss equations, shapes on BOTH paths

Design constraint that drove the rewrite of the earlier version: each block carries a
NAME + a one-line "what it is", nothing more. Parameter counts and layer internals live
in the slide notes. Tensor shapes stay on the arrows.

Rendered on the deck background (#1C2340) so the figures sit flush on the slide.

    python student_training/scripts/make_arch_figures_2026-08-22.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "figures"

# ---- E3a-status palette
BG = "#1C2340"
PANEL = "#243060"
PANEL_DK = "#202A50"
CYAN = "#00BFFF"
CYAN_DK = "#0096CC"
WHITE = "#FFFFFF"
MUTED = "#A0B4CC"
GREEN = "#00E676"
ORANGE = "#FFA726"

FS_NAME = 10.5
FS_SUB = 8.0
FS_SHAPE = 7.8

Z_BOX, Z_ARROW, Z_TEXT = 2, 5, 6


def _fit(text, w, fs_max, per_char=0.00895):
    """Shrink a label so it stays inside its box. per_char is the measured fraction of
    axes-width one character occupies at fs_max on this canvas."""
    need = len(text) * per_char
    if need <= 0:
        return fs_max
    return min(fs_max, fs_max * (w * 0.92) / need)


def block(ax, x, y, w, h, name, sub=None, accent=CYAN, fc=PANEL):
    """Card with a coloured strip on top - the E3a-status card idiom."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=fc, edgecolor=accent, linewidth=1.1, zorder=Z_BOX))
    ax.add_patch(Rectangle((x, y + h - 0.018), w, 0.018, facecolor=accent,
                           edgecolor="none", zorder=Z_BOX + 1))
    ty = y + h * 0.60 if sub else y + h * 0.45
    ax.text(x + w / 2, ty, name, ha="center", va="center", fontsize=_fit(name, w, FS_NAME),
            fontweight="bold", color=accent, zorder=Z_TEXT)
    if sub:
        widest = max(sub.split("\n"), key=len)
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=_fit(widest, w, FS_SUB, per_char=0.0060),
                color=WHITE, zorder=Z_TEXT, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, label=None, lab_dy=0.030, color=CYAN):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.3, color=color, zorder=Z_ARROW,
                                 shrinkA=0, shrinkB=0))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + lab_dy, label, ha="center", va="bottom",
                fontsize=FS_SHAPE, color=MUTED, style="italic", zorder=Z_TEXT)


def elbow(ax, x1, y1, x2, y2, color=CYAN):
    ax.plot([x1, x2], [y1, y1], color=color, linewidth=1.3, zorder=Z_ARROW,
            solid_capstyle="round")
    ax.add_patch(FancyArrowPatch((x2, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.3, color=color, zorder=Z_ARROW,
                                 shrinkA=0, shrinkB=0))


def new_ax(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    return fig, ax


def save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=220, facecolor=BG, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def legend(ax, items, y=0.02):
    total = 0.16 * len(items)
    x0 = 0.5 - total / 2
    for i, (col, lab) in enumerate(items):
        x = x0 + i * 0.16
        ax.add_patch(Rectangle((x, y), 0.016, 0.026, facecolor=col, edgecolor="none",
                               zorder=Z_BOX))
        ax.text(x + 0.022, y + 0.013, lab, ha="left", va="center", fontsize=8, color=MUTED,
                zorder=Z_TEXT)


# ------------------------------------------------------------------------------- L1
def fig_L1():
    fig, ax = new_ax(11.5, 4.5)

    ax.text(0.015, 0.895, "WHILE TRAINING", fontsize=11, fontweight="bold", color=CYAN)
    block(ax, 0.015, 0.605, 0.150, 0.215, "dashcam video", "16 frames", accent=MUTED)
    block(ax, 0.245, 0.605, 0.185, 0.215, "vision model", "LoRA-tuned", accent=CYAN)
    block(ax, 0.545, 0.700, 0.175, 0.160, "will it crash?", "GT label", accent=ORANGE)
    block(ax, 0.545, 0.455, 0.175, 0.160, "describe the scene", "teacher caption", accent=ORANGE)
    block(ax, 0.790, 0.455, 0.180, 0.160, "teacher's sentence", "written by a large VLM",
          accent=MUTED)

    arrow(ax, 0.165, 0.7125, 0.245, 0.7125)
    arrow(ax, 0.430, 0.730, 0.545, 0.780)
    arrow(ax, 0.430, 0.695, 0.545, 0.535)
    arrow(ax, 0.790, 0.535, 0.720, 0.535)

    ax.add_patch(FancyBboxPatch((0.528, 0.425), 0.458, 0.455,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor="none", edgecolor=ORANGE, linewidth=1.3,
                                linestyle="--", zorder=0))
    ax.text(0.757, 0.388, "extra teaching signal  -  present only while training",
            ha="center", fontsize=8.5, color=ORANGE, style="italic")

    ax.text(0.015, 0.300, "AFTER TRAINING   (what ships)", fontsize=11, fontweight="bold",
            color=GREEN)
    block(ax, 0.015, 0.075, 0.150, 0.180, "dashcam video", accent=MUTED)
    block(ax, 0.245, 0.075, 0.185, 0.180, "vision model", "trained", accent=GREEN)
    block(ax, 0.545, 0.075, 0.175, 0.180, "P(collision)", accent=GREEN)
    arrow(ax, 0.165, 0.165, 0.245, 0.165, color=GREEN)
    arrow(ax, 0.430, 0.165, 0.545, 0.165, color=GREEN)
    ax.text(0.855, 0.165, "no text, no language model,\nno extra cost at run time",
            ha="center", va="center", fontsize=9, color=GREEN, style="italic")

    return save(fig, "arch_L1_overview_2026-08-22.png")


# ------------------------------------------------------------------------------- L2
def fig_L2():
    fig, ax = new_ax(11.5, 3.9)
    y, h = 0.400, 0.360
    cy = y + h / 2

    block(ax, 0.008, y, 0.148, h, "16 frames",
          "1280x720 dashcam\nsquash-resize + norm", accent=MUTED)
    block(ax, 0.206, y, 0.182, h, "V-JEPA2 ViT-L",
          "24-layer video encoder\n+ LoRA merged in", accent=GREEN)
    block(ax, 0.438, y, 0.182, h, "temporal processor",
          "attentive probe: 1 learned\nquery over 2,560 tokens", accent=CYAN_DK)
    block(ax, 0.670, y, 0.180, h, "classifier",
          "MLPHead, 3 layers\nLinear-GELU-LN x2 + Linear", accent=CYAN_DK)
    block(ax, 0.882, y, 0.110, h, "P(collision)", "softmax(z)", accent=CYAN)

    # shape labels go ABOVE the row - the inter-box gaps are far narrower than the text
    for x1, x2, lab in [(0.156, 0.206, "(1, 16, 3, 256, 320)"),
                        (0.388, 0.438, "(1, 2560, 1024)"),
                        (0.620, 0.670, "(1, 1024)"),
                        (0.850, 0.882, "(1, 2) logits")]:
        arrow(ax, x1, cy, x2, cy)
        ax.text((x1 + x2) / 2, y + h + 0.030, lab, ha="center", va="bottom",
                fontsize=FS_SHAPE, color=MUTED, style="italic", zorder=Z_TEXT)

    ax.text(0.5, 0.235,
            "Architecturally identical to stock BADAS-Open  -  LoRA folds into the base weights,\n"
            "so parameter count, latency and dependencies are unchanged. Only the weight values are ours.",
            ha="center", va="center", fontsize=9.5, color=WHITE, fontweight="bold")
    ax.text(0.5, 0.115, "Predictor and SigLIP text encoder are not loaded at inference.",
            ha="center", va="center", fontsize=9, color=ORANGE, style="italic")

    legend(ax, [(MUTED, "input"), (GREEN, "fine-tuned (LoRA)"), (CYAN_DK, "frozen"),
                (CYAN, "output")], y=0.015)
    return save(fig, "arch_L2_inference_2026-08-22.png")


# ------------------------------------------------------------------------------- L3
def fig_L3():
    fig, ax = new_ax(12.2, 5.4)

    # train-only region behind everything
    ax.add_patch(FancyBboxPatch((0.398, 0.105), 0.596, 0.455,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=PANEL_DK, edgecolor=ORANGE, linewidth=1.3,
                                linestyle="--", zorder=0))
    ax.text(0.986, 0.122, "train-only  -  discarded at inference", ha="right", fontsize=8,
            color=ORANGE, style="italic", zorder=Z_TEXT)

    # ---- shared trunk
    block(ax, 0.010, 0.545, 0.118, 0.175, "16 frames", "1280x720", accent=MUTED)
    block(ax, 0.192, 0.545, 0.160, 0.175, "V-JEPA2 ViT-L", "frozen trunk", accent=CYAN_DK)
    block(ax, 0.205, 0.400, 0.134, 0.100, "LoRA  r=16", "TRAINABLE", accent=GREEN)
    arrow(ax, 0.272, 0.500, 0.272, 0.545, color=GREEN)
    arrow(ax, 0.128, 0.6325, 0.192, 0.6325, "(1, 16, 3, 256, 320)", lab_dy=0.098)

    # ---- crash path: labels sit ABOVE the row so they cannot land on the box titles
    yc, hc = 0.735, 0.155
    cyc = yc + hc / 2
    lab_y = yc + hc + 0.018
    block(ax, 0.420, yc, 0.148, hc, "temporal processor", "frozen", accent=CYAN_DK)
    block(ax, 0.632, yc, 0.118, hc, "classifier", "frozen", accent=CYAN_DK)
    block(ax, 0.800, yc, 0.108, hc, "P(collision)", "softmax(z)", accent=CYAN)
    block(ax, 0.936, yc, 0.058, hc, "L_crash", "CE(z, y)", accent=ORANGE)
    elbow(ax, 0.352, 0.678, 0.494, yc)
    ax.text(0.424, 0.686, "(1, 2560, 1024)", ha="center", va="bottom", fontsize=FS_SHAPE,
            color=MUTED, style="italic", zorder=Z_TEXT)
    arrow(ax, 0.568, cyc, 0.632, cyc)
    ax.text(0.600, lab_y, "(1, 1024)", ha="center", fontsize=FS_SHAPE, color=MUTED,
            style="italic", zorder=Z_TEXT)
    arrow(ax, 0.750, cyc, 0.800, cyc)
    ax.text(0.775, lab_y, "(1, 2) logits", ha="center", fontsize=FS_SHAPE, color=MUTED,
            style="italic", zorder=Z_TEXT)
    arrow(ax, 0.908, cyc, 0.936, cyc)

    # ---- semantic path
    ys, hs = 0.170, 0.155
    cys = ys + hs / 2
    block(ax, 0.420, ys, 0.148, hs, "Predictor", "8 learned queries\n-> mean-pool", accent=GREEN)
    block(ax, 0.632, ys, 0.128, hs, "L_sem", "InfoNCE", accent=ORANGE)
    block(ax, 0.822, ys, 0.172, hs, "SigLIP text encoder", "frozen", accent=CYAN_DK)
    block(ax, 0.822, 0.435, 0.172, 0.110, "teacher caption", "one sentence / window",
          accent=MUTED)

    elbow(ax, 0.352, 0.590, 0.494, ys + hs, color=GREEN)
    ax.text(0.504, 0.470, "(1, 2560, 1024)", ha="left", va="center", fontsize=FS_SHAPE,
            color=MUTED, style="italic", zorder=Z_TEXT)
    arrow(ax, 0.568, cys, 0.632, cys)
    ax.text(0.600, ys + hs + 0.018, "(1, 768)", ha="center", fontsize=FS_SHAPE, color=MUTED,
            style="italic", zorder=Z_TEXT)
    arrow(ax, 0.822, cys, 0.760, cys)
    ax.text(0.791, ys + hs + 0.018, "(1, 768)", ha="center", fontsize=FS_SHAPE, color=MUTED,
            style="italic", zorder=Z_TEXT)
    arrow(ax, 0.908, 0.435, 0.908, ys + hs, color=MUTED)
    ax.text(0.494, ys - 0.030, "(1, 8, 768)  ->  mean over queries", ha="center",
            fontsize=7.4, color=MUTED, style="italic", zorder=Z_TEXT)

    # ---- loss equation, left column, clear of the dashed region (starts x=0.398)
    ax.add_patch(FancyBboxPatch((0.010, 0.165), 0.352, 0.150,
                                boxstyle="round,pad=0.008,rounding_size=0.012",
                                facecolor=PANEL_DK, edgecolor=ORANGE, linewidth=1.5,
                                zorder=Z_BOX))
    ax.text(0.186, 0.257, "L  =  L_crash  +  0.05 . L_sem", ha="center", va="center",
            fontsize=11, fontweight="bold", color=ORANGE, family="monospace", zorder=Z_TEXT)
    ax.text(0.186, 0.203, "the control arm sets the second term to zero", ha="center",
            va="center", fontsize=8, color=MUTED, style="italic", zorder=Z_TEXT)

    legend(ax, [(MUTED, "input"), (CYAN_DK, "frozen"), (GREEN, "trainable"),
                (ORANGE, "loss term")], y=0.012)
    return save(fig, "arch_L3_training_2026-08-22.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn in (fig_L1, fig_L2, fig_L3):
        print(f"  wrote {fn()}")


if __name__ == "__main__":
    main()
