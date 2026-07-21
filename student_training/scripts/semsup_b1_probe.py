"""
semsup_b1_probe.py
===================
Stage B1: Predictor-only probe. BADAS ViT-L trunk FROZEN, SigLIP FROZEN.
Trains ONLY the Predictor (a ResamplerProjector with num_queries=1) to map the
BADAS patch grid -> a single vector matching the frozen SigLIP caption embedding.

Diagnostic question: do frozen BADAS features already encode the caption
semantics? Also produces a warm-start checkpoint for Stage B's predictor.

No crash training here - AP is unaffected (nothing about the trunk/head changes).

Frozen features (BADAS patches + SigLIP targets) are cached once up front, so
epochs cost milliseconds and we can afford many of them + early stopping.

Reported against a CONSTANT mean-embedding control: because SigLIP embeddings of
near-synonymous crash captions are anisotropic, a predictor that ignores the video
entirely still scores a high mean_cosine. retrieval_top1_acc vs. that baseline is
the honest signal.

Usage (RunPod):
  python semsup_b1_probe.py --config ../configs/e4_stageA.yaml \
      --epochs 100 --out-dir /workspace/semsup/b1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

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
    ap.add_argument("--epochs", type=int, default=100,
                    help="max epochs; features are cached so epochs are cheap")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15,
                    help="early-stop after this many epochs with no val_loss improvement")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N examples")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

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

    # -------------------------------------------------------------------------
    # Cache the frozen features ONCE. BADAS and SigLIP never update, so patches
    # and targets are identical every epoch - recomputing them per epoch was
    # ~15x wasted ViT-L forward passes. Cached on CPU, moved per batch.
    # -------------------------------------------------------------------------
    def build_cache(exs, tag):
        patches, targets = [], []
        for ex in tqdm(exs, desc=f"[cache] {tag}", leave=False):
            with torch.no_grad():
                _, p = badas.forward(ex["frame_paths"])
                t = siglip_text_embed([ex["caption"]], siglip_model, siglip_tok, device)
            # BADAS may run in fp16; the Predictor is fp32 - cast at this boundary.
            patches.append(p.to(dtype=torch.float32).cpu())
            targets.append(t.squeeze(0).cpu())
        return torch.stack(patches), torch.stack(targets)  # (N,P,D), (N,Dt)

    print("\n[cache] precomputing frozen BADAS patches + SigLIP targets")
    tc = time.time()
    Xtr, Ytr = build_cache(train_ex, "train")
    Xva, Yva = build_cache(val_ex, "val")
    print(f"[cache] done in {time.time()-tc:.1f}s  train={tuple(Xtr.shape)}  val={tuple(Xva.shape)}")

    # BADAS is no longer needed - free ~4GB of GPU before training.
    del badas
    if device == "cuda":
        torch.cuda.empty_cache()

    def evaluate(X, Y):
        """Returns (loss, mean_cosine, retrieval_top1_acc) on a cached split."""
        predictor.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                xb = X[i:i + args.batch_size].to(device)
                preds.append(F.normalize(predictor(xb).squeeze(1), dim=-1))
        P = torch.cat(preds, dim=0)
        T = Y.to(device)
        diag = F.cosine_similarity(P, T, dim=-1)
        loss = (1 - diag).mean().item()
        top1 = (P @ T.T).argmax(dim=1)
        acc = (top1 == torch.arange(len(X), device=device)).float().mean().item()
        return loss, diag.mean().item(), acc

    # -------------------------------------------------------------------------
    # Collapse control. SigLIP embeddings of 267 near-synonymous crash captions
    # are highly anisotropic: a predictor that IGNORES the video and always emits
    # the mean caption embedding still scores a high mean_cosine. Any real result
    # must beat this baseline - retrieval_top1_acc (chance = 1/n_val) is the
    # honest metric, mean_cosine alone is gameable.
    # -------------------------------------------------------------------------
    mean_emb = F.normalize(Ytr.mean(dim=0, keepdim=True), dim=-1).to(device)
    Tva = Yva.to(device)
    base_cos = F.cosine_similarity(mean_emb.expand_as(Tva), Tva, dim=-1).mean().item()
    base_top1 = (mean_emb.expand_as(Tva) @ Tva.T).argmax(dim=1)
    base_acc = (base_top1 == torch.arange(len(Xva), device=device)).float().mean().item()
    print(f"[control] constant mean-embedding baseline: mean_cosine={base_cos:.4f}  "
          f"retrieval_top1_acc={base_acc:.4f}  (chance={1/max(1,len(Xva)):.4f})")

    print("\n[train] Predictor only (BADAS + SigLIP frozen)")
    t0 = time.time()
    history, best = [], []          # best = [(val_loss, epoch, path)], keep 3
    best_loss, since_improved = float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        perm = torch.randperm(len(Xtr))
        total_loss, nb = 0.0, 0
        pbar = tqdm(range(0, len(Xtr), args.batch_size),
                    desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for i in pbar:
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            opt.zero_grad()
            pred = F.normalize(predictor(xb).squeeze(1), dim=-1)
            loss = (1 - F.cosine_similarity(pred, yb, dim=-1)).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
            pbar.set_postfix(loss=f"{total_loss/nb:.4f}")

        tr_loss = total_loss / max(1, nb)
        va_loss, va_cos, va_acc = evaluate(Xva, Yva)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_mean_cosine": va_cos, "val_retrieval_top1_acc": va_acc})
        print(f"  epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  "
              f"val_loss={va_loss:.4f}  val_cos={va_cos:.4f}  val_ret@1={va_acc:.4f}  "
              f"({time.time()-t0:.1f}s)")

        # Keep the 3 lowest-val_loss checkpoints.
        ckpt = out_dir / f"predictor_b1_ep{epoch:03d}.pt"
        torch.save(predictor.state_dict(), ckpt)
        best.append((va_loss, epoch, ckpt))
        best.sort(key=lambda r: r[0])
        for _, _, stale in best[3:]:
            stale.unlink(missing_ok=True)
        best = best[:3]

        if va_loss < best_loss - 1e-5:
            best_loss, since_improved = va_loss, 0
        else:
            since_improved += 1
            if since_improved >= args.patience:
                print(f"[early-stop] no val_loss improvement for {args.patience} epochs "
                      f"(best={best_loss:.4f} @ epoch {best[0][1]})")
                break

    # predictor_b1.pt = the BEST checkpoint (Stage B warm-starts from this path).
    best_loss, best_epoch, best_path = best[0]
    predictor.load_state_dict(torch.load(best_path, map_location=device))
    torch.save(predictor.state_dict(), out_dir / "predictor_b1.pt")
    final_loss, mean_cos, retrieval_acc = evaluate(Xva, Yva)

    print(f"\n[eval] BEST checkpoint (epoch {best_epoch}, n_val={len(val_ex)}): "
          f"mean_cosine={mean_cos:.4f}  retrieval_top1_acc={retrieval_acc:.4f}")
    print(f"[eval] vs. collapse control:  mean_cosine={base_cos:.4f}  "
          f"retrieval_top1_acc={base_acc:.4f}")
    verdict = ("LEARNED something video-specific" if retrieval_acc > base_acc
               else "NO evidence beyond the constant-embedding baseline")
    print(f"[verdict] {verdict}")

    with open(out_dir / "b1_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_train": len(train_ex), "n_val": len(val_ex),
            "best_epoch": best_epoch, "best_val_loss": best_loss,
            "held_out_mean_cosine": mean_cos, "held_out_retrieval_top1_acc": retrieval_acc,
            "control_mean_embedding": {"mean_cosine": base_cos, "retrieval_top1_acc": base_acc,
                                        "chance_retrieval": 1 / max(1, len(val_ex))},
            "top3_checkpoints": [{"epoch": e, "val_loss": l, "path": str(p)}
                                  for l, e, p in best],
            "epochs_run": len(history), "epochs_max": args.epochs,
            "lr": args.lr, "batch_size": args.batch_size, "patience": args.patience,
            "seed": args.seed, "siglip_model": args.siglip_model,
            "history": history,
        }, f, indent=2)
    print(f"[save] {out_dir / 'predictor_b1.pt'} (best)  {out_dir / 'b1_metrics.json'}")
    print(f"[save] top-3: {[str(p.name) for _, _, p in best]}")


if __name__ == "__main__":
    main()
