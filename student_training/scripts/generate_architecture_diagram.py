"""
generate_architecture_diagram.py
================================
Render two presentation-ready PNG block diagrams of the Teacher → Student
collision-anticipation pipeline. Each diagram is its own file with a side
legend, so they can be embedded independently in slides / the report.

Outputs (PNG, white background, 200 DPI):
  outputs/reports/figures/architecture_training.png
  outputs/reports/figures/architecture_inference.png

Sources of truth used to draw the diagram:
  - teacher_distillation/scripts/Teacher_dataset_distill_v11.py
        DEFAULT_MODEL = "google/gemini-3.1-pro-preview"
        Pass-1 (PROMPT_G)  → Pass-2 (PROMPT_DEBATE_G, only on mismatches)
  - student_training/configs/train_lora.yaml
        InternVL3.5-4B-Flash, LoRA r=16/α=32 on q,k,v,o,gate,up,down
        loss_alpha=0.5, lr=2e-4, num_epochs=50, grad_accum=8
  - student_training/models/internvl_lora.py
        ScoreHead = Linear(hidden,1) → raw logit (σ applied outside)
        Trainable: LoRA + mlp1 projector + ScoreHead
        Frozen:    InternViT-300M vision encoder + ViR pixel-shuffle
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR      = PROJECT_ROOT / "outputs" / "reports" / "figures"
OUT_TRAIN    = OUT_DIR / "architecture_training.png"
OUT_INFER    = OUT_DIR / "architecture_inference.png"


# ── Style palette ────────────────────────────────────────────────────────────
COLORS = {
    "data":      "#dfe9f3",
    "teacher":   "#fde2c4",
    "frozen":    "#e9ecef",
    "trainable": "#c8e6c9",
    "loss":      "#f8d7da",
    "output":    "#fff3cd",
    "title":     "#1f2d3d",
    "edge":      "#34495e",
}
BORDER = "#5d6d7e"

LEGEND = [
    ("Data / dataset",                        COLORS["data"]),
    ("Teacher  (Gemini 3.1 Pro Preview)",     COLORS["teacher"]),
    ("Frozen module",                         COLORS["frozen"]),
    ("Trainable module",                      COLORS["trainable"]),
    ("Loss / supervision",                    COLORS["loss"]),
    ("Output / artefact",                     COLORS["output"]),
]


# ── Drawing primitives ───────────────────────────────────────────────────────

def box(ax, x, y, w, h, text, color, fontsize=9, bold=False, italic=False,
        edge=BORDER, lw=1.2):
    """Rounded rectangle with centred text that is clipped to the box."""
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=lw, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    style  = "italic" if italic else "normal"
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, fontweight=weight, fontstyle=style,
        color=COLORS["title"], wrap=True, clip_on=True,
    )


def arrow(ax, xy_from, xy_to, label=None, color=None, lw=1.4,
          style="-|>", curved=False, label_dy=0.14, label_fs=8):
    color = color or COLORS["edge"]
    cs = "arc3,rad=0.0" if not curved else "arc3,rad=0.18"
    ax.add_patch(FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle=style, mutation_scale=14,
        linewidth=lw, color=color, connectionstyle=cs,
    ))
    if label:
        mx = (xy_from[0] + xy_to[0]) / 2
        my = (xy_from[1] + xy_to[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom",
                fontsize=label_fs, color=color, style="italic")


def add_legend(ax):
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.text(0.1, 9.6, "Legend", fontsize=11, fontweight="bold",
            color=COLORS["title"])
    y = 8.8
    for label, color in LEGEND:
        ax.add_patch(FancyBboxPatch(
            (0.1, y - 0.25), 0.55, 0.45,
            boxstyle="round,pad=0.01,rounding_size=0.05",
            linewidth=1.0, edgecolor=BORDER, facecolor=color,
        ))
        ax.text(0.85, y, label, fontsize=9, va="center",
                color=COLORS["title"])
        y -= 0.85


# ── Diagram 1: TRAINING ──────────────────────────────────────────────────────

def draw_training(ax):
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 13)
    ax.axis("off")

    ax.text(0.4, 12.4,
            "Offline training pipeline  —  Teacher → Student knowledge distillation",
            fontsize=14, fontweight="bold", color=COLORS["title"])
    ax.text(0.4, 11.85,
            "100 Nexar dash-cam clips · Gemini 3.1 Pro Preview labels them · "
            "InternVL3.5-4B-Flash is LoRA-tuned to imitate verdict + reasoning",
            fontsize=10, style="italic", color="#5d6d7e")

    # Row A — data → teacher → JSONL → split
    yA = 9.0
    box(ax, 0.4, yA, 3.4, 1.7,
        "Nexar dash-cam clips\n100 train  /  677 test\n16 frames, stride 4, 448×448",
        COLORS["data"], fontsize=9)
    arrow(ax, (3.8, yA + 0.85), (4.6, yA + 0.85))

    box(ax, 4.6, yA, 4.2, 1.7,
        "Pass-1  ·  PROMPT_G\nGemini 3.1 Pro Preview\n(via OpenRouter)\n→ structured JSON verdict",
        COLORS["teacher"], fontsize=9, bold=True)
    arrow(ax, (8.8, yA + 0.85), (9.6, yA + 0.85), label="if mismatch GT")

    box(ax, 9.6, yA, 4.2, 1.7,
        "Pass-2  ·  PROMPT_DEBATE_G\nGemini re-decides only on\nclips that disagree with GT\n→ p2_* fields",
        COLORS["teacher"], fontsize=9, bold=True)
    arrow(ax, (13.8, yA + 0.85), (14.6, yA + 0.85))

    box(ax, 14.6, yA, 3.6, 1.7,
        "teacher_dataset_v11.jsonl\nverdict + score + reasoning",
        COLORS["output"], fontsize=9, bold=True)
    arrow(ax, (18.2, yA + 0.85), (19.0, yA + 0.85))

    box(ax, 19.0, yA, 2.6, 1.7,
        "Stratified\n80 / 20\ntrain / val split",
        COLORS["data"], fontsize=9)

    # Down arrow into student container
    arrow(ax, (20.3, yA), (20.3, 7.6), label_dy=0)

    # Student container
    box(ax, 0.4, 2.8, 21.2, 4.6, "", "#ffffff", edge="#a0aab2")
    ax.text(0.7, 7.05,
            "Student model  —  InternVL3.5-4B-Flash  (4.94 B params total)",
            fontsize=11, fontweight="bold", color=COLORS["title"])

    yS = 4.6
    # Frozen
    box(ax, 0.7, yS, 3.6, 1.6,
        "InternViT-300M\nvision encoder",
        COLORS["frozen"], fontsize=9)
    ax.text(2.5, yS - 0.32, "FROZEN", ha="center", fontsize=8,
            color="#7f8c8d", style="italic")
    arrow(ax, (4.3, yS + 0.8), (4.7, yS + 0.8), lw=1.1)

    box(ax, 4.7, yS, 2.6, 1.6,
        "ViR pixel-shuffle\ncompression",
        COLORS["frozen"], fontsize=9)
    ax.text(6.0, yS - 0.32, "FROZEN", ha="center", fontsize=8,
            color="#7f8c8d", style="italic")
    arrow(ax, (7.3, yS + 0.8), (7.7, yS + 0.8), lw=1.1)

    # Trainable
    box(ax, 7.7, yS, 3.0, 1.6,
        "MLP projector  (mlp1)\nvision → LLM tokens",
        COLORS["trainable"], fontsize=9, bold=True)
    arrow(ax, (10.7, yS + 0.8), (11.1, yS + 0.8), lw=1.1)

    box(ax, 11.1, yS, 4.2, 1.6,
        "Qwen3-4B LLM  +  LoRA\nq,k,v,o,gate,up,down\n(r = 16, α = 32)",
        COLORS["trainable"], fontsize=9, bold=True)

    arrow(ax, (15.3, yS + 1.05), (15.7, yS + 1.05), lw=1.1)
    arrow(ax, (15.3, yS + 0.55), (15.7, yS + 0.55), lw=1.1)

    box(ax, 15.7, yS + 0.85, 5.7, 0.75,
        "ScoreHead  ·  Linear(h→1) → logit  →  σ  →  P(collision) ∈ [0,1]",
        COLORS["trainable"], fontsize=8.5, bold=True)
    box(ax, 15.7, yS, 5.7, 0.75,
        "LM head  →  generates JSON reasoning",
        COLORS["trainable"], fontsize=9, bold=True)

    # Loss row
    yL = 3.05
    box(ax, 15.7, yL, 2.7, 0.7,
        "BCE-with-logits\n(score, target)",
        COLORS["loss"], fontsize=8.5)
    box(ax, 18.7, yL, 2.7, 0.7,
        "Cross-entropy\n(reasoning tokens)",
        COLORS["loss"], fontsize=8.5)

    arrow(ax, (17.05, yS), (17.05, yL + 0.7), color="#c0392b", lw=1.1)
    arrow(ax, (20.05, yS), (20.05, yL + 0.7), color="#c0392b", lw=1.1)

    ax.text(0.7, 3.45,
            "Combined loss:   L  =  α · BCE  +  (1 − α) · CE      (α = 0.5)",
            fontsize=10, color="#c0392b", fontweight="bold")
    ax.text(0.7, 3.0,
            "AdamW · cosine LR · lr = 2e-4 · grad-accum 8 · bf16 + grad-checkpointing",
            fontsize=9, color="#5d6d7e", style="italic")

    # Footer
    ax.text(0.4, 1.55,
            "Trainable parameters:  ≈ 50 M  (≈ 1 % of model)  =  LoRA  +  MLP projector  +  ScoreHead",
            fontsize=10, color="#27ae60", fontweight="bold")
    ax.text(0.4, 1.05,
            "Schedule:  50 epochs (intentional overfit run)  ·  stratified 80/20 train/val from the same 100 clips",
            fontsize=9, color="#5d6d7e", style="italic")
    ax.text(0.4, 0.6,
            "Checkpoint selection:  best by val F1 on held-out 20 % slice  →  epoch 27 / step 270",
            fontsize=9, color="#5d6d7e", style="italic")


# ── Diagram 2: INFERENCE ─────────────────────────────────────────────────────

def draw_inference(ax):
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(0.4, 8.4,
            "Online inference pipeline  —  16-frame dash-cam window in  →  collision probability + reasoning out",
            fontsize=14, fontweight="bold", color=COLORS["title"])
    ax.text(0.4, 7.85,
            "Selected checkpoint:  epoch 27 / step 270  ·  evaluated on 677 held-out Nexar clips  ·  test AP = 0.636",
            fontsize=10, style="italic", color="#5d6d7e")

    yM = 4.5
    # Input
    box(ax, 0.4, yM, 3.4, 1.7,
        "Dash-cam clip\n16 frames @ 448×448\nImageNet mean / std",
        COLORS["data"], fontsize=9)
    arrow(ax, (3.8, yM + 0.85), (4.4, yM + 0.85))

    # Vision tower (frozen)
    box(ax, 4.4, yM, 3.2, 1.7,
        "InternViT-300M\nvision encoder",
        COLORS["frozen"], fontsize=9)
    ax.text(6.0, yM - 0.32, "FROZEN", ha="center", fontsize=8,
            color="#7f8c8d", style="italic")
    arrow(ax, (7.6, yM + 0.85), (8.1, yM + 0.85), lw=1.1)

    box(ax, 8.1, yM, 2.7, 1.7,
        "ViR + MLP\nprojector (mlp1)",
        COLORS["frozen"], fontsize=9)
    ax.text(9.45, yM - 0.32, "(projector trained)", ha="center", fontsize=8,
            color="#7f8c8d", style="italic")
    arrow(ax, (10.8, yM + 0.85), (11.3, yM + 0.85),
          label="visual tokens", lw=1.1)

    # LLM
    box(ax, 11.3, yM, 4.0, 1.7,
        "Qwen3-4B  +  LoRA\nLM forward pass\n(visual + text tokens)",
        COLORS["trainable"], fontsize=9, bold=True)

    # Branch into two outputs
    arrow(ax, (15.3, yM + 1.25), (16.4, yM + 1.95),
          curved=True, label="hidden state", label_dy=0.05)
    arrow(ax, (15.3, yM + 0.45), (16.4, yM - 0.25),
          curved=True, label="generate", label_dy=-0.45)

    # Output 1 — score
    box(ax, 16.4, yM + 1.4, 2.6, 1.4,
        "ScoreHead\nLinear → σ\n→ P(collision)",
        COLORS["trainable"], fontsize=9, bold=True)
    arrow(ax, (19.0, yM + 2.1), (19.4, yM + 2.1), lw=1.1)
    box(ax, 19.4, yM + 1.4, 2.2, 1.4,
        "AP / F1\nPR-curve\nROC",
        COLORS["output"], fontsize=9, bold=True)

    # Output 2 — reasoning JSON
    box(ax, 16.4, yM - 1.7, 2.6, 1.4,
        "LM head\n+ tokenizer\n→ JSON reasoning",
        COLORS["trainable"], fontsize=9, bold=True)
    arrow(ax, (19.0, yM - 1.0), (19.4, yM - 1.0), lw=1.1)
    box(ax, 19.4, yM - 1.7, 2.2, 1.4,
        "Interpretability\nscene · verdict\n· why",
        COLORS["output"], fontsize=9, bold=True)

    # Footer notes
    ax.text(0.4, 1.6,
            "Same vision preprocessing as training (build_transform, ImageNet mean/std, 448×448)  —  no train/test mismatch.",
            fontsize=9.5, color="#5d6d7e", style="italic")
    ax.text(0.4, 1.05,
            "Two outputs from one forward pass:  (i) calibrated score for AP/F1 ranking,  (ii) JSON for human-readable reasoning.",
            fontsize=9.5, color="#5d6d7e", style="italic")
    ax.text(0.4, 0.5,
            "Decision threshold:  default 0.5 is conservative on this distribution;  optimal F1 is reached at threshold ≈ 0.195.",
            fontsize=9.5, color="#5d6d7e", style="italic")


# ── Figure builders ──────────────────────────────────────────────────────────

def render_training():
    fig = plt.figure(figsize=(18, 9), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[14, 2.4], wspace=0.02)
    ax_main = fig.add_subplot(gs[0])
    ax_leg  = fig.add_subplot(gs[1])
    draw_training(ax_main)
    add_legend(ax_leg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_TRAIN, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_TRAIN}  ({OUT_TRAIN.stat().st_size/1024:.1f} KB)")


def render_inference():
    fig = plt.figure(figsize=(18, 7), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[14, 2.4], wspace=0.02)
    ax_main = fig.add_subplot(gs[0])
    ax_leg  = fig.add_subplot(gs[1])
    draw_inference(ax_main)
    add_legend(ax_leg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_INFER, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUT_INFER}  ({OUT_INFER.stat().st_size/1024:.1f} KB)")


def main():
    render_training()
    render_inference()


if __name__ == "__main__":
    main()

