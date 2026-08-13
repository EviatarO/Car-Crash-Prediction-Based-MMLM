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

--loss cosine (default): 1 - cos(pred, target). PROVEN DEGENERATE (2026-07-25
project review, verified against the recorded metrics): the analytic minimizer
for a video-blind predictor is target_mean/||target_mean||, worth 1-||target_mean||.
On the 267-caption set that floor is ~0.1352; the real (trained) run reached
0.1345 - beating the floor by 0.53% of the available range, with retrieval@1
exactly at chance. Scaling data will not fix an objective sitting at its own
degenerate optimum.

--loss infonce: batched in-batch-negatives contrastive loss (CLIP/SigLIP-style).
The shared target-mean direction contributes equally to every column of the
softmax and CANCELS, so the collapse solution scores at chance instead of at
0.865 - this is what actually distinguishes "the objective was wrong" from
"the data is too small". Sibling-TTE rows of the same video_id are masked out
of the negative set (their captions are near-duplicates - see
docs_agents/EXPERIMENTS.md's val-split diagnostic), since treating them as
false negatives would penalize a correct near-match.

Usage (RunPod):
  python semsup_b1_probe.py --config ../configs/e4_stageA.yaml \
      --loss infonce --epochs 100 --out-dir /workspace/semsup/b1_infonce
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
    load_training_examples, clip_level_split, CAPTIONS_JSONL,
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
    ap.add_argument("--loss", choices=["cosine", "infonce"], default="cosine",
                    help="cosine = proven-degenerate regression (see module docstring); "
                         "infonce = in-batch contrastive, sibling-TTE-masked")
    ap.add_argument("--infonce-tau-init", type=float, default=0.07,
                    help="initial temperature (learnable), CLIP/SigLIP-standard init")
    ap.add_argument("--captions", default=None,
                    help="override caption JSONL (default: the 267-row "
                         "Caption_Train_All_Clips.jsonl). Use for the prompt-bakeoff "
                         "arm_{a,b,c}.jsonl files from semsup_caption_qa.py, or the V12 "
                         "1,761-window pool.")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="subsample this fraction of TRAIN video_ids (by clip, seeded on "
                         "--seed) AFTER the train/val split, so val stays IDENTICAL across "
                         "different fractions - for a clean scaling curve (retrieval vs n) "
                         "without val-set drift confounding the comparison. 1.0 = no subsample.")
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

    # num_queries=8 (not 1), hidden_dim=256 (not 512): the old config was
    # ~5.13M params - 1.8x the trunk's ~2.8M LoRA trainable count it's meant to
    # gently steer - AND its self-attention block was mathematically a no-op
    # (softmax over 1 key), so ~1M of those params were dead weight (2026-07-25
    # review, A-2). This config is ~1.25M params. Multi-token output is
    # mean-pooled to one Dt vector before comparison to the SigLIP target.
    predictor = ResamplerProjector(in_dim=1024, out_dim=dt, num_queries=8,
                                    hidden_dim=256, n_heads=8, ffn_mult=2).to(device)
    trainable = list(predictor.parameters())
    log_tau = None
    if args.loss == "infonce":
        # Learnable temperature (CLIP/SigLIP convention). log-parameterized so it
        # can't go negative; clamped at use-time to a sane range.
        log_tau = torch.nn.Parameter(torch.log(torch.tensor(args.infonce_tau_init)).to(device))
        trainable = trainable + [log_tau]
    opt = torch.optim.AdamW(trainable, lr=args.lr)

    examples = load_training_examples(limit=args.limit, captions_path=args.captions)
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac)
    print(f"[data] train={len(train_ex)}  val={len(val_ex)} (clip-level split, Dt={dt})")

    if args.train_frac < 1.0:
        # Subsample TRAIN video_ids only, val_ex is untouched above this point - so
        # every --train-frac value in a scaling-curve sweep scores against the exact
        # same held-out clips, and the curve isn't confounded by val-set drift.
        import random as _random
        train_vids = sorted({e["video_id"] for e in train_ex})
        _random.Random(args.seed).shuffle(train_vids)
        n_keep = max(1, int(len(train_vids) * args.train_frac))
        keep_vids = set(train_vids[:n_keep])
        train_ex = [e for e in train_ex if e["video_id"] in keep_vids]
        print(f"[data] --train-frac={args.train_frac}: subsampled to "
              f"train={len(train_ex)} rows / {n_keep} clips (val unchanged at {len(val_ex)})")

    # -------------------------------------------------------------------------
    # Cache the frozen features ONCE. BADAS and SigLIP never update, so patches
    # and targets are identical every epoch - recomputing them per epoch was
    # ~15x wasted ViT-L forward passes. Cached on CPU, moved per batch.
    # -------------------------------------------------------------------------
    def build_cache(exs, tag):
        patches, targets, vids = [], [], []
        for ex in tqdm(exs, desc=f"[cache] {tag}", leave=False):
            with torch.no_grad():
                _, p = badas.forward(ex["frame_paths"])
                t = siglip_text_embed([ex["caption"]], siglip_model, siglip_tok, device)
            # BADAS may run in fp16; the Predictor is fp32 - cast at this boundary.
            patches.append(p.to(dtype=torch.float32).cpu())
            targets.append(t.squeeze(0).cpu())
            vids.append(ex["video_id"])  # needed to mask sibling-TTE false negatives
        return torch.stack(patches), torch.stack(targets), vids  # (N,P,D), (N,Dt), list[N]

    print("\n[cache] precomputing frozen BADAS patches + SigLIP targets")
    tc = time.time()
    Xtr, Ytr, vids_tr = build_cache(train_ex, "train")
    Xva, Yva, vids_va = build_cache(val_ex, "val")
    print(f"[cache] done in {time.time()-tc:.1f}s  train={tuple(Xtr.shape)}  val={tuple(Xva.shape)}")

    # BADAS is no longer needed - free ~4GB of GPU before training.
    del badas
    if device == "cuda":
        torch.cuda.empty_cache()

    def evaluate(X, Y, vids):
        """Returns (loss, mean_cosine, retrieval_top1_acc, retrieval_top1_acc_sibling_ok,
        retrieval_top1_acc_clip) on a cached split. `loss`/`mean_cosine` are row-level
        (row independence doesn't matter for a plain average). The three retrieval
        numbers differ in what counts as a "hit":
          - retrieval_top1_acc: exact-row match (the original, strict metric).
          - _sibling_ok: also counts a hit against a different TTE window of the
            SAME clip (near-duplicate caption) - see EXPERIMENTS.md T-8.
          - _clip (T-3, PRIMARY): rows are first pooled per clip (mean, renorm) so
            retrieval happens among ~17 real clips, not 51 correlated TTE-window
            rows. This is the metric that should actually be trusted - row-level
            retrieval silently inflates/deflates because most "candidates" are
            near-duplicates of each other (see EXPERIMENTS.md's val-split
            diagnostic: row-level val_ap ranked checkpoints in the OPPOSITE order
            from test_AP for Stage B)."""
        predictor.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X), args.batch_size):
                xb = X[i:i + args.batch_size].to(device)
                preds.append(F.normalize(predictor(xb).mean(dim=1), dim=-1))
        P = torch.cat(preds, dim=0)
        T = Y.to(device)
        diag = F.cosine_similarity(P, T, dim=-1)
        loss = (1 - diag).mean().item()
        top1 = (P @ T.T).argmax(dim=1)
        idx_arange = torch.arange(len(X), device=device)
        acc = (top1 == idx_arange).float().mean().item()
        vids_arr = list(vids)
        sib_hit = torch.tensor(
            [vids_arr[int(top1[i])] == vids_arr[i] for i in range(len(X))],
            device=device, dtype=torch.float32,
        )
        acc_sibling_ok = sib_hit.mean().item()
        acc_clip = clip_level_retrieval_acc(P, T, vids_arr)
        return loss, diag.mean().item(), acc, acc_sibling_ok, acc_clip

    def clip_level_retrieval_detail(P, T, vids_list):
        """Pool rows sharing a video_id (mean, renormalize) before retrieval, so
        the candidate pool is the real independent sample size (~17 clips), not
        51 correlated TTE-window rows. See evaluate()'s docstring. Returns
        (clip_ids SORTED for a canonical cross-run order, per-clip hit 0/1 list)
        so a paired arm-vs-arm comparison (semsup_promptbakeoff_report.py) can
        resample clips together across two separately-trained arms - the
        aggregate accuracy alone can't support that."""
        from collections import defaultdict
        by_p, by_t = defaultdict(list), defaultdict(list)
        for i, v in enumerate(vids_list):
            by_p[v].append(P[i])
            by_t[v].append(T[i])
        clip_ids = sorted(by_p.keys())
        if len(clip_ids) < 2:
            return [], []
        Pc = torch.stack([F.normalize(torch.stack(by_p[v]).mean(0), dim=-1) for v in clip_ids])
        Tc = torch.stack([F.normalize(torch.stack(by_t[v]).mean(0), dim=-1) for v in clip_ids])
        top1c = (Pc @ Tc.T).argmax(dim=1)
        idxc = torch.arange(len(clip_ids), device=P.device)
        hits = (top1c == idxc).int().tolist()
        return clip_ids, hits

    def clip_level_retrieval_acc(P, T, vids_list):
        clip_ids, hits = clip_level_retrieval_detail(P, T, vids_list)
        if not clip_ids:
            return float("nan")
        return sum(hits) / len(hits)

    def infonce_loss(pred, tgt, vids_batch, log_tau):
        """In-batch contrastive loss. Same-video (sibling-TTE) rows are masked out
        of the negative set - their captions are near-duplicates, so treating them
        as false negatives would penalize a correct near-match instead of a wrong
        one. Unlike cosine regression, the shared target-mean direction cancels in
        the softmax, so the collapse solution scores at chance, not at ||E[t]||."""
        tau = log_tau.exp().clamp(min=1e-2, max=1.0)
        logits = (pred @ tgt.T) / tau
        vb = list(vids_batch)
        same_vid = torch.tensor([[a == b for b in vb] for a in vb], device=pred.device)
        mask = same_vid & ~torch.eye(len(vb), dtype=torch.bool, device=pred.device)
        logits = logits.masked_fill(mask, float("-inf"))
        labels = torch.arange(len(vb), device=pred.device)
        return F.cross_entropy(logits, labels)

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
    base_sib_hit = torch.tensor(
        [vids_va[int(base_top1[i])] == vids_va[i] for i in range(len(Xva))],
        device=device, dtype=torch.float32,
    )
    base_acc_sibling_ok = base_sib_hit.mean().item()
    base_acc_clip = clip_level_retrieval_acc(mean_emb.expand_as(Tva), Tva, vids_va)
    print(f"[control] constant mean-embedding baseline: mean_cosine={base_cos:.4f}  "
          f"retrieval_top1_acc={base_acc:.4f}  (sibling_ok={base_acc_sibling_ok:.4f}, "
          f"clip={base_acc_clip:.4f})  (chance={1/max(1,len(Xva)):.4f})")

    print(f"\n[train] Predictor only (BADAS + SigLIP frozen)  loss={args.loss}")
    t0 = time.time()
    # Checkpoint/early-stop criterion (2026-08-12 fix - was val_loss unconditionally,
    # even under --loss infonce, where val_loss is a temperature-scaled softmax CE
    # that does NOT rank checkpoints the same way as retrieval accuracy - the actual
    # thing predictor_b1.pt is warm-started for downstream. /project-review found this
    # exact mismatch on the 267-row run: epoch 28 (lowest val_loss, selected) scored
    # val_retrieval_top1_acc_clip=0.1086 while epoch 43 (never selected) scored 0.1267.
    # cosine keeps the original val_loss criterion (its own degenerate-optimum analysis
    # was done in terms of that loss, so changing it there would break comparability).
    sel_key = "retrieval_clip" if args.loss == "infonce" else "loss"
    def sel_value(hist_row):
        return hist_row["val_retrieval_top1_acc_clip"] if sel_key == "retrieval_clip" \
            else hist_row["val_loss"]
    sel_better = (lambda new, old: new > old) if sel_key == "retrieval_clip" \
        else (lambda new, old: new < old)
    print(f"[select] checkpoint/early-stop criterion: "
          f"{'val_retrieval_top1_acc_clip (higher better)' if sel_key == 'retrieval_clip' else 'val_loss (lower better)'}")
    history, best = [], []          # best = [(sel_value, epoch, path)], keep 3, sorted BEST-first
    best_sel, since_improved = (float("-inf") if sel_key == "retrieval_clip" else float("inf")), 0

    for epoch in range(1, args.epochs + 1):
        predictor.train()
        perm = torch.randperm(len(Xtr))
        total_loss, nb = 0.0, 0
        pbar = tqdm(range(0, len(Xtr), args.batch_size),
                    desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for i in pbar:
            idx = perm[i:i + args.batch_size]
            xb, yb = Xtr[idx].to(device), Ytr[idx].to(device)
            vb = [vids_tr[j] for j in idx.tolist()]
            opt.zero_grad()
            pred = F.normalize(predictor(xb).mean(dim=1), dim=-1)
            if args.loss == "infonce":
                loss = infonce_loss(pred, yb, vb, log_tau)
            else:
                loss = (1 - F.cosine_similarity(pred, yb, dim=-1)).mean()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
            pbar.set_postfix(loss=f"{total_loss/nb:.4f}")

        tr_loss = total_loss / max(1, nb)
        va_loss, va_cos, va_acc, va_acc_sib, va_acc_clip = evaluate(Xva, Yva, vids_va)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
                        "val_mean_cosine": va_cos, "val_retrieval_top1_acc": va_acc,
                        "val_retrieval_top1_acc_sibling_ok": va_acc_sib,
                        "val_retrieval_top1_acc_clip": va_acc_clip})
        print(f"  epoch {epoch}/{args.epochs}  train_loss={tr_loss:.4f}  "
              f"val_loss={va_loss:.4f}  val_cos={va_cos:.4f}  val_ret@1={va_acc:.4f}  "
              f"val_ret@1_sib={va_acc_sib:.4f}  val_ret@1_clip={va_acc_clip:.4f}  "
              f"({time.time()-t0:.1f}s)")

        # Keep the 3 best-by-sel_key checkpoints (val_loss for cosine,
        # val_retrieval_top1_acc_clip for infonce - see sel_key comment above).
        ckpt = out_dir / f"predictor_b1_ep{epoch:03d}.pt"
        torch.save(predictor.state_dict(), ckpt)
        cur_sel = sel_value(history[-1])
        best.append((cur_sel, epoch, ckpt))
        # sort so index 0 is always the BEST regardless of sel_key's direction
        best.sort(key=lambda r: r[0], reverse=(sel_key == "retrieval_clip"))
        for _, _, stale in best[3:]:
            stale.unlink(missing_ok=True)
        best = best[:3]

        if sel_better(cur_sel, best_sel):
            best_sel, since_improved = cur_sel, 0
        else:
            since_improved += 1
            if since_improved >= args.patience:
                print(f"[early-stop] no {('retrieval@1_clip' if sel_key=='retrieval_clip' else 'val_loss')} "
                      f"improvement for {args.patience} epochs "
                      f"(best={best_sel:.4f} @ epoch {best[0][1]})")
                break

    # predictor_b1.pt = the BEST checkpoint (Stage B warm-starts from this path).
    # best_sel_final is in sel_key's units (val_loss for cosine, retrieval_top1_acc_clip
    # for infonce) - NOT always "loss", despite the historical variable name elsewhere.
    best_sel_final, best_epoch, best_path = best[0]
    predictor.load_state_dict(torch.load(best_path, map_location=device))
    torch.save(predictor.state_dict(), out_dir / "predictor_b1.pt")
    final_loss, mean_cos, retrieval_acc, retrieval_acc_sib, retrieval_acc_clip = evaluate(Xva, Yva, vids_va)

    # Per-clip hit/miss at the best checkpoint, for a paired cross-arm comparison
    # (semsup_promptbakeoff_report.py) - the aggregate number above can't support
    # resampling clips together across two separately-trained arms.
    predictor.eval()
    with torch.no_grad():
        preds_final = []
        for i in range(0, len(Xva), args.batch_size):
            xb = Xva[i:i + args.batch_size].to(device)
            preds_final.append(F.normalize(predictor(xb).mean(dim=1), dim=-1))
        P_final = torch.cat(preds_final, dim=0)
    val_clip_ids, val_clip_hits = clip_level_retrieval_detail(P_final, Yva.to(device), vids_va)

    print(f"\n[eval] BEST checkpoint (epoch {best_epoch}, n_val={len(val_ex)}): "
          f"mean_cosine={mean_cos:.4f}  retrieval_top1_acc={retrieval_acc:.4f}  "
          f"sibling_ok={retrieval_acc_sib:.4f}  clip={retrieval_acc_clip:.4f}")
    print(f"[eval] vs. collapse control:  mean_cosine={base_cos:.4f}  "
          f"retrieval_top1_acc={base_acc:.4f}  sibling_ok={base_acc_sibling_ok:.4f}  "
          f"clip={base_acc_clip:.4f}")
    # Verdict uses the PER-CLIP metric (T-3: the statistically sound one - see
    # evaluate()'s docstring for why row-level retrieval is unreliable at this
    # scale) rather than the exact-row metric used by earlier runs.
    verdict = ("LEARNED something video-specific" if retrieval_acc_clip > base_acc_clip
               else "NO evidence beyond the constant-embedding baseline")
    print(f"[verdict] {verdict}  (decided on clip-level retrieval, n_clips="
          f"{len(set(vids_va))})")

    with open(out_dir / "b1_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "loss": args.loss,
            "infonce_tau_init": args.infonce_tau_init if args.loss == "infonce" else None,
            "infonce_tau_final": (float(log_tau.exp().clamp(min=1e-2, max=1.0).item())
                                  if log_tau is not None else None),
            "n_train": len(train_ex), "n_val": len(val_ex),
            "n_val_clips": len(set(vids_va)),
            "selection_criterion": sel_key,   # "loss" (cosine) or "retrieval_clip" (infonce)
            "best_epoch": best_epoch,
            "best_selection_value": best_sel_final,
            "train_frac": args.train_frac,
            "held_out_mean_cosine": mean_cos,
            "held_out_retrieval_top1_acc": retrieval_acc,
            "held_out_retrieval_top1_acc_sibling_ok": retrieval_acc_sib,
            "held_out_retrieval_top1_acc_clip": retrieval_acc_clip,
            "val_clip_ids": val_clip_ids, "val_clip_hits": val_clip_hits,
            "control_mean_embedding": {"mean_cosine": base_cos, "retrieval_top1_acc": base_acc,
                                        "retrieval_top1_acc_sibling_ok": base_acc_sibling_ok,
                                        "retrieval_top1_acc_clip": base_acc_clip,
                                        "chance_retrieval": 1 / max(1, len(val_ex)),
                                        "chance_retrieval_clip": 1 / max(1, len(set(vids_va)))},
            "top3_checkpoints": [{"epoch": e, "val_loss": l, "path": str(p)}
                                  for l, e, p in best],
            "epochs_run": len(history), "epochs_max": args.epochs,
            "lr": args.lr, "batch_size": args.batch_size, "patience": args.patience,
            "seed": args.seed, "siglip_model": args.siglip_model,
            "captions_path": str(args.captions or CAPTIONS_JSONL),
            "history": history,
        }, f, indent=2)
    print(f"[save] {out_dir / 'predictor_b1.pt'} (best)  {out_dir / 'b1_metrics.json'}")
    print(f"[save] top-3: {[str(p.name) for _, _, p in best]}")


if __name__ == "__main__":
    main()
