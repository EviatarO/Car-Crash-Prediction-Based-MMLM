"""
plot_grad_trace.py
===================
Companion figure for semsup_train.py --aux-mode {occ,rel}'s grad_trace.jsonl (child plan
2026-09-19-AA-token-relevance-aux, Phase 4). One line per epoch: per-encoder-layer (0-23)
crash-grad norm / aux-grad norm / cosine, the vanishing-gradient ratio, which layers (if any)
leaked aux gradient past --aux-layer, and the per-layer LoRA weight-change norm.

Renders, per run:
  1. crash-grad norm vs aux-grad norm, one line per layer-bucket across epochs (log y)
  2. cos(crash, aux) per layer-bucket across epochs
  3. LoRA weight-change norm per layer-bucket across epochs (the NEXT_LORA_PLACEMENT.md
     cross-reference: a converged layer can have a small gradient and still be the one
     that moved)
  4. vanishing_ratio across epochs, with the 1e-3 flag threshold marked

Usage:
  python plot_grad_trace.py /path/to/out_dir  [--out /path/to/figure.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_trace(out_dir: Path) -> list:
    path = out_dir / "grad_trace.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} - was this run started with --aux-mode set?")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    rows = load_trace(out_dir)
    epochs = [r["epoch"] for r in rows]
    aux_layer = rows[0]["aux_layer"]
    layers = sorted((int(k) for k in rows[0]["per_layer"] if k.isdigit()))
    has_predictor_stack = any("predictor_stack" in r["per_layer"] for r in rows)
    cmap = plt.get_cmap("viridis", max(len(layers), 2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax_norm, ax_cos, ax_wchange, ax_vanish = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for i, layer in enumerate(layers):
        aux_norms = [r["per_layer"][str(layer)]["aux_grad_norm"] for r in rows]
        wchange = [r.get("lora_weight_change_norm", {}).get(str(layer)) for r in rows]
        cos = [r["per_layer"][str(layer)]["cos"] for r in rows]
        color = cmap(i)
        ax_norm.plot(epochs, [max(v, 1e-8) for v in aux_norms], color=color, lw=1.3,
                    label=f"L{layer}" if layer in (0, aux_layer, layers[-1]) else None)
        ax_cos.plot(epochs, cos, color=color, lw=1.3)
        we = [(e, v) for e, v in zip(epochs, wchange) if v is not None]
        if we:
            ax_wchange.plot([e for e, _ in we], [v for _, v in we], color=color, lw=1.3,
                            label=f"L{layer}" if layer in (0, aux_layer, layers[-1]) else None)
    if has_predictor_stack:
        ps_norms = [r["per_layer"].get("predictor_stack", {}).get("aux_grad_norm", 0.0) for r in rows]
        ax_norm.plot(epochs, [max(v, 1e-8) for v in ps_norms], "k--", lw=1.5, label="predictor_stack")

    ax_norm.set_yscale("log")
    ax_norm.set_title("|g_aux| per encoder layer (log scale)")
    ax_norm.set_xlabel("epoch"); ax_norm.legend(fontsize=8, ncol=3)
    ax_norm.axhline(1e-6, color="red", ls=":", lw=1, label="~noise floor")

    ax_cos.axhline(0, color="grey", lw=0.8)
    ax_cos.set_title("cos(g_crash, g_aux) per encoder layer")
    ax_cos.set_xlabel("epoch"); ax_cos.set_ylim(-1.05, 1.05)

    ax_wchange.set_title("per-epoch LoRA weight-change norm ||W_e - W_{e-1}||")
    ax_wchange.set_xlabel("epoch"); ax_wchange.legend(fontsize=8, ncol=3)

    vanish = [r["vanishing_ratio"] for r in rows]
    vanish_flag = [r["vanishing_flag"] for r in rows]
    ax_vanish.plot(epochs, [v if v is not None else np.nan for v in vanish], "o-", color="tab:blue")
    ax_vanish.axhline(1e-3, color="red", ls="--", lw=1, label="flag threshold (1e-3)")
    ax_vanish.set_yscale("log")
    ax_vanish.set_title(f"vanishing ratio: |g_aux|(layer0) / |g_aux|(layer{aux_layer})")
    ax_vanish.set_xlabel("epoch"); ax_vanish.legend(fontsize=8)
    for e, v, flag in zip(epochs, vanish, vanish_flag):
        if flag:
            ax_vanish.plot(e, v, "rx", ms=10)

    leaked_epochs = [r["epoch"] for r in rows if r["leaked_into_later_layers"]]
    title_note = f"  ⚠ aux gradient leaked past layer {aux_layer} at epoch(s) {leaked_epochs}" \
        if leaked_epochs else "  (no leakage past aux_layer in any epoch)"
    fig.suptitle(f"{out_dir.name}  —  grad_trace, aux_layer={aux_layer}{title_note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = Path(args.out) if args.out else out_dir / "grad_trace.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
