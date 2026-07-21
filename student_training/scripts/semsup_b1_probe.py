"""
semsup_b1_probe.py
===================
Stage B1: Predictor-only probe. BADAS ViT-L trunk FROZEN, SigLIP FROZEN.
Trains ONLY the Predictor (a ResamplerProjector with num_queries=1) to map the
BADAS patch grid -> a single vector matching the frozen SigLIP caption embedding.

Diagnostic question: do frozen BADAS features already encode the caption
semantics? Also produces a warm-start checkpoint for Stage B's predictor.

No crash training here - AP is unaffected (nothing about the trunk/head changes).

Usage (RunPod):
  python semsup_b1_probe.py --config ../configs/e4_stageA.yaml \
      --epochs 15 --out-dir /root/semsup/b1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import (  # noqa: E402
    TrainableBadasWrapper, load_siglip, siglip_text_embed,
    load_training_examples, clip_level_split,
)
from vjepa_reason import ResamplerProjector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e4_stageA.yaml (BADAS hf_repo etc.)")
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N examples")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        stagea_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] BADAS (frozen)")
    badas = TrainableBadasWrapper(stagea_cfg, lora_target_modules=None)  # frozen
    print(f"[load] SigLIP: {args.siglip_model}")
    siglip_model, siglip_tok = load_siglip(args.siglip_model, device)
    dt = siglip_model.config.text_config.hidden_size if hasattr(siglip_model.config, "text_config") \
        else siglip_model.config.hidden_size

    predictor = ResamplerProjector(in_dim=1024, out_dim=dt, num_queries=1,
                                    hidden_dim=512, n_heads=8).to(device)
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.lr)

    examples = load_training_examples(limit=args.limit)
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac)
    print(f"[data] train={len(train_ex)}  val={len(val_ex)} (clip-level split, Dt={dt})")

    def get_patches(ex):
        with torch.no_grad():
            _, patches = badas.forward(ex["frame_paths"])
        # BADAS may run in fp16; the Predictor is fp32 - cast at this boundary.
        return patches.unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, P, D)

    def predict(ex):
        patches = get_patches(ex)
        pred = predictor(patches).squeeze(1)               # (1, Dt)
        return F.normalize(pred, dim=-1)

    def target(ex):
        return siglip_text_embed([ex["caption"]], siglip_model, siglip_tok, device)  # (1, Dt)

    print("\n[train] Predictor only (BADAS + SigLIP frozen)")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        predictor.train()
        total_loss = 0.0
        for ex in train_ex:
            opt.zero_grad()
            pred = predict(ex)
            tgt = target(ex)
            loss = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(train_ex))
        print(f"  epoch {epoch}/{args.epochs}  train_loss={avg:.4f}  ({time.time()-t0:.1f}s)")

    # Held-out eval: cosine similarity + retrieval accuracy (nearest caption among val set)
    predictor.eval()
    val_preds, val_tgts = [], []
    with torch.no_grad():
        for ex in val_ex:
            val_preds.append(predict(ex))
            val_tgts.append(target(ex))
    if val_preds:
        P = torch.cat(val_preds, dim=0)   # (Nval, Dt)
        T = torch.cat(val_tgts, dim=0)    # (Nval, Dt)
        diag_cos = F.cosine_similarity(P, T, dim=-1)
        sim_matrix = P @ T.T              # (Nval, Nval)
        top1 = sim_matrix.argmax(dim=1)
        retrieval_acc = (top1 == torch.arange(len(val_ex), device=device)).float().mean().item()
        mean_cos = diag_cos.mean().item()
        print(f"\n[eval] held-out (n={len(val_ex)}): mean_cosine={mean_cos:.4f}  "
              f"retrieval_top1_acc={retrieval_acc:.4f}")
    else:
        mean_cos = retrieval_acc = float("nan")

    torch.save(predictor.state_dict(), out_dir / "predictor_b1.pt")
    with open(out_dir / "b1_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_train": len(train_ex), "n_val": len(val_ex),
            "held_out_mean_cosine": mean_cos, "held_out_retrieval_top1_acc": retrieval_acc,
            "epochs": args.epochs, "siglip_model": args.siglip_model,
        }, f, indent=2)
    print(f"[save] {out_dir / 'predictor_b1.pt'}  {out_dir / 'b1_metrics.json'}")


if __name__ == "__main__":
    main()
