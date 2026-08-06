"""
semsup_train.py
================
Unified A1 / B trainer. LoRA-unfreezes the BADAS ViT-L trunk; always trains
with the crash loss (CE on BADAS's own 2-logit head vs label).

  --semantic-weight 0    -> Stage A1 (crash-only control)
  --semantic-weight >0   -> Stage B   (crash + semantic-aux; needs --predictor-init
                             from semsup_b1_probe.py for warm-start, or trains the
                             Predictor from scratch alongside if omitted)

Selects the best epoch by held-out (clip-level, from the 267-caption set) crash
AP, then scores the REAL 677-clip Private test set with the selected checkpoint
and writes a results JSONL compatible with evaluate_metrics.py's schema
({video_id, ground_truth, score, group}).

IMPORTANT: run --dry-run-modules in semsup_common.py FIRST on the pod to confirm
real LoRA target_modules names before running this for real (BADAS internals are
only knowable at runtime - see plan risk note).

Usage (RunPod):
  python semsup_train.py --config ../configs/e4_stageA.yaml \
      --lora-target-modules qkv,proj,fc1,fc2 \
      --semantic-weight 0.0 --epochs 8 --out-dir /root/semsup/a1 \
      --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
      --test-frames-root ../../dataset/test

  # Stage B (add semantic loss, warm-start predictor from B1):
  python semsup_train.py --config ../configs/e4_stageA.yaml \
      --lora-target-modules qkv,proj,fc1,fc2 \
      --semantic-weight 0.3 --predictor-init /root/semsup/b1/predictor_b1.pt \
      --epochs 8 --out-dir /root/semsup/b \
      --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
      --test-frames-root ../../dataset/test
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import (  # noqa: E402
    TrainableBadasWrapper, load_siglip, siglip_text_embed,
    load_training_examples, clip_level_split,
)
from vjepa_reason import ResamplerProjector  # noqa: E402
from e4_stageA_badas_open_eval import load_manifest, frame_paths_for  # noqa: E402
from metrics_core import metrics_from_arrays  # noqa: E402


def evaluate_crash_ap(badas, examples, device):
    """AP computed per CLIP, not per row. `examples` are TTE windows of the
    same clip and share one label (verified: every clip in the caption set is
    single-label across its TTE variants), so scoring per row and computing AP
    over 51 correlated rows silently inflates AP - the real independent
    sample size is ~17 clips, not 51 rows (see EXPERIMENTS.md's val-split
    diagnostic: val_ap saturated at 0.96-0.98 and ranked checkpoints in the
    OPPOSITE order from test_AP). Aggregate (mean) a clip's row scores into
    one point before computing AP. Applies identically to A1 and B, since
    both call this same function - no arm-specific tuning."""
    badas.nn_model.eval()
    from collections import defaultdict
    by_clip = defaultdict(list)
    with torch.no_grad():
        for ex in examples:
            logits, _ = badas.forward(ex["frame_paths"])
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
            by_clip[ex["video_id"]].append((score, ex["label"]))
    ys, yt = [], []
    for pairs in by_clip.values():
        scores = [s for s, _ in pairs]
        labels = {l for _, l in pairs}
        # Verified empirically every clip is single-label across its TTE
        # windows; majority-vote as a defensive fallback rather than crash if
        # that ever stops holding (e.g. a future mixed-label caption source).
        label = next(iter(labels)) if len(labels) == 1 else round(sum(l for _, l in pairs) / len(pairs))
        ys.append(sum(scores) / len(scores))
        yt.append(label)
    if len(set(yt)) < 2:
        return float("nan")
    return average_precision_score(yt, ys)


def build_caption_bank(examples, siglip_model, siglip_tok, device, batch=64):
    """Precompute the frozen SigLIP embedding for every example's caption, once.

    WHY A BANK (this is what makes InfoNCE possible here at all): InfoNCE needs
    NEGATIVES - other captions to contrast against - and the obvious objection is
    that TrainableBadasWrapper is batch-size-1, so there is no batch to draw them
    from. But the negatives live entirely on the TARGET side, and SigLIP is frozen:
    t_j never carries a gradient. So they can be precomputed once and reused for
    every anchor, giving hundreds of negatives at ~4MB (1761 x 768 x 4 bytes) and
    zero extra autograd graphs. The alternative - holding 8 full ViT-L forward
    graphs alive across the grad-accum window - risks OOM for no benefit.

    Returns (bank (N, Dt) L2-normalized on `device`, vids list parallel to it).
    """
    texts = [ex["caption"] for ex in examples]
    vids = [ex["video_id"] for ex in examples]
    chunks = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            emb = siglip_text_embed(texts[i:i + batch], siglip_model, siglip_tok, device)
            chunks.append(emb.detach())
    bank = torch.cat(chunks, dim=0)
    bank = F.normalize(bank, dim=-1)
    return bank, vids


def infonce_from_bank(pred, anchor_idx, bank, vids, log_tau):
    """InfoNCE for ONE anchor against the whole frozen caption bank.

    Ported from semsup_b1_probe.py's in-batch infonce_loss(), with the in-batch
    target matrix swapped for the precomputed bank (see build_caption_bank).
    Sibling-TTE masking is preserved and matters: the same video at a different
    TTE has a near-duplicate caption, so scoring it as a negative would punish a
    correct near-match instead of a wrong one.

    Unlike cosine regression, the shared target-mean direction cancels in the
    softmax, so a predictor that ignores the video and emits a constant scores at
    chance (1/N) rather than at ||E[t]|| - which is exactly the degenerate optimum
    that made the cosine objective flat (B1: 0.53% of available range, retrieval
    at chance). See outputs/semantic_captions/b1_metrics.json.
    """
    tau = log_tau.exp().clamp(min=1e-2, max=1.0)
    logits = (pred @ bank.T).squeeze(0) / tau          # (N,)
    anchor_vid = vids[anchor_idx]
    same_vid = torch.tensor([v == anchor_vid for v in vids], device=pred.device)
    same_vid[anchor_idx] = False                        # keep the positive itself
    logits = logits.masked_fill(same_vid, float("-inf"))
    label = torch.tensor([anchor_idx], device=pred.device)
    return F.cross_entropy(logits.unsqueeze(0), label)


def evaluate_val_loss(badas, examples, device, predictor, siglip_model, siglip_tok,
                       semantic_weight, semantic_loss="cosine", val_bank=None,
                       val_vids=None, log_tau=None):
    """Mirrors the training step's loss computation (crash CE + semantic_weight *
    semantic loss), no gradient, averaged per-window over `examples`.
    Added so train_loss vs val_loss can be compared per epoch (see epoch_metrics.jsonl)
    to spot overfitting directly, rather than relying on val_ap alone - val_ap only
    tells you ranking quality, not whether the model has started memorizing train
    windows while val loss climbs. Same aggregation level (per-row, not per-clip) as
    the training loop's own loss accounting, so train_loss and val_loss are
    comparable on a like-for-like basis.

    For semantic_loss='infonce' the val bank is the VAL set's own captions, so val
    InfoNCE is contrasted against val-internal negatives - the analogue of what the
    train loss does, and it keeps the two numbers on the same scale (both are
    -log(1/N)-bounded, though note N differs between train and val, so compare the
    TREND across epochs rather than the absolute train-vs-val difference for
    infonce)."""
    badas.nn_model.eval()
    if predictor is not None:
        predictor.eval()
    total_crash, total_sem, n = 0.0, 0.0, 0
    with torch.no_grad():
        for i, ex in enumerate(examples):
            try:
                logits, patches = badas.forward(ex["frame_paths"])
            except (OSError, RuntimeError):
                continue
            label = torch.tensor([ex["label"]], device=device)
            crash_loss = F.cross_entropy(logits, label)
            sem_loss = torch.tensor(0.0, device=device)
            if predictor is not None:
                patches32 = patches.unsqueeze(0).to(dtype=torch.float32)
                pred = predictor(patches32).mean(dim=1)
                pred = F.normalize(pred, dim=-1)
                if semantic_loss == "infonce":
                    # use the stamped bank index, not the loop counter - val_ex
                    # happens not to be shuffled, but relying on that is exactly
                    # the assumption that would break silently if it ever changes.
                    sem_loss = infonce_from_bank(pred, ex.get("_bank_idx", i),
                                                  val_bank, val_vids, log_tau)
                else:
                    tgt = siglip_text_embed([ex["caption"]], siglip_model, siglip_tok, device)
                    sem_loss = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
            total_crash += crash_loss.item()
            total_sem += sem_loss.item()
            n += 1
    if n == 0:
        return float("nan"), float("nan")
    return total_crash / n, total_sem / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora-target-modules", required=True,
                     help="comma-separated substrings, e.g. 'qkv,proj,fc1,fc2'")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--semantic-weight", type=float, default=0.0)
    ap.add_argument("--semantic-loss", default="cosine", choices=["cosine", "infonce"],
                     help="'cosine' (default, preserves the original Stage-B behavior) is "
                          "1-cos(pred, SigLIP(caption)). It has a DEGENERATE optimum: a "
                          "predictor that ignores the video and emits the mean caption "
                          "embedding scores 1-||E[t]||, and on this caption set that mean has "
                          "norm 0.865 - B1's real trained run beat that baseline by only 0.53% "
                          "of the available range, with retrieval at exactly chance. "
                          "'infonce' contrasts each anchor against a bank of frozen SigLIP "
                          "caption embeddings (see build_caption_bank/infonce_from_bank); the "
                          "shared mean direction cancels in the softmax so the collapse "
                          "solution scores at chance instead of winning. B1 measured 4x chance "
                          "retrieval under infonce vs exactly chance under cosine.")
    ap.add_argument("--infonce-tau-init", type=float, default=0.07,
                     help="initial temperature for --semantic-loss infonce (learnable, "
                          "matches semsup_b1_probe.py's default)")
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    ap.add_argument("--predictor-init", default=None, help="warm-start from B1 checkpoint")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--captions-path", default=None,
                     help="override the caption JSONL (default: Caption_Train_All_Clips.jsonl, "
                          "the 267-row pool). The training pool IS whichever file this points "
                          "at - e.g. for a matched A1_587-vs-B_587 comparison, both runs must "
                          "pass the SAME --captions-path so the only difference is "
                          "--semantic-weight.")
    ap.add_argument("--keep-top-k", type=int, default=3,
                     help="how many epoch checkpoints to keep, ranked by val_ap (default 3, "
                          "matching the original A1/B behavior). Set >= --epochs to keep every "
                          "epoch - useful when you want to pick a checkpoint by hand later using "
                          "the train/val loss gap in epoch_metrics.jsonl rather than trusting "
                          "val_ap alone (val_ap is noisy at this data scale and only measures "
                          "ranking quality, not overfitting).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-manifest", default=None, help="e.g. test_manifest_hires.jsonl (677 Private)")
    ap.add_argument("--test-frames-root", default=None, help="e.g. dataset/test")
    ap.add_argument("--test-limit", type=int, default=0, help="debug: score only first N test clips")
    ap.add_argument("--seed", type=int, default=0,
                     help="seeds random/torch RNG (LoRA init, example shuffle) so A1 "
                          "and B are comparable runs, not confounded by different init")
    ap.add_argument("--min-examples", type=int, default=1,
                     help="fail fast if fewer than this many training examples load "
                          "(catches a partially-synced/missing frames volume early)")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    import yaml
    with open(args.config, encoding="utf-8") as f:
        stagea_cfg = yaml.safe_load(f)
    for section, key in (("data", "frame_filename_pattern"), ("data", "gt_field")):
        if section not in stagea_cfg or key not in stagea_cfg[section]:
            raise KeyError(f"{args.config} missing required key: {section}.{key}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = "B (crash+semantic)" if args.semantic_weight > 0 else "A1 (crash-only)"
    sem_note = f"  semantic_loss={args.semantic_loss}" if args.semantic_weight > 0 else ""
    print(f"[cfg] stage={stage}  semantic_weight={args.semantic_weight}{sem_note}  "
          f"lora_target_modules={args.lora_target_modules}  seed={args.seed}")

    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    badas = TrainableBadasWrapper(
        stagea_cfg, lora_target_modules=target_modules,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    )
    # peft's save_pretrained() auto-generates a model card BEFORE writing any
    # adapter weights, and assumes base_model.config supports `in` (a HF
    # PretrainedConfig). BADAS's V-JEPA2 uses a plain ModelArgs dataclass
    # instead, so that step crashes save_pretrained() every time, before any
    # checkpoint is written. We don't need the model card - skip it.
    badas.nn_model.create_or_update_model_card = lambda *a, **k: None
    trainable = [p for p in badas.nn_model.parameters() if p.requires_grad]

    predictor = None
    siglip_model = siglip_tok = None
    if args.semantic_weight > 0:
        print(f"[load] SigLIP: {args.siglip_model}")
        siglip_model, siglip_tok = load_siglip(args.siglip_model, device)
        dt = siglip_model.config.text_config.hidden_size if hasattr(siglip_model.config, "text_config") \
            else siglip_model.config.hidden_size
        # num_queries=8 (not 1), hidden_dim=256 (not 512): the old num_queries=1
        # config was ~5.13M params - 1.8x the LoRA trainable count it was meant
        # to gently steer - AND its self-attention block was mathematically a
        # no-op (softmax over 1 key), so ~1M of those params were dead weight
        # (2026-07-25 review, A-2). This config is ~1.25M params, genuinely
        # "small/weak" relative to the ~2.8M LoRA trunk. Multi-token output is
        # mean-pooled to a single Dt vector before comparison to the SigLIP
        # target (see predictor(...).mean(dim=1) below).
        predictor = ResamplerProjector(in_dim=1024, out_dim=dt, num_queries=8,
                                        hidden_dim=256, n_heads=8, ffn_mult=2).to(device)
        if args.predictor_init:
            predictor.load_state_dict(torch.load(args.predictor_init, map_location=device))
            print(f"[load] warm-started predictor from {args.predictor_init}")
        trainable += list(predictor.parameters())

    # Learnable InfoNCE temperature (same contract as semsup_b1_probe.py). Must be
    # in the optimizer's param list or it silently stays at its init value.
    log_tau = None
    if args.semantic_weight > 0 and args.semantic_loss == "infonce":
        log_tau = torch.nn.Parameter(
            torch.log(torch.tensor(args.infonce_tau_init, device=device)))
        trainable = trainable + [log_tau]

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    examples = load_training_examples(limit=args.limit, captions_path=args.captions_path)
    if len(examples) < args.min_examples:
        raise RuntimeError(
            f"Only {len(examples)} training examples loaded (< --min-examples "
            f"{args.min_examples}). This usually means dataset/train is missing or "
            f"partially synced - check the symlink/volume before training on a "
            f"silently-shrunk dataset."
        )
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[data] train={len(train_ex)}  val={len(val_ex)} (clip-level split)")

    # Frozen caption banks for InfoNCE negatives - built ONCE (SigLIP never trains,
    # so these are constants). Train anchors contrast against the train bank, val
    # against the val bank; mixing them would leak val captions into the train
    # objective's negative set.
    train_bank = val_bank = None
    train_bank_vids = val_bank_vids = None
    if args.semantic_weight > 0 and args.semantic_loss == "infonce":
        train_bank, train_bank_vids = build_caption_bank(train_ex, siglip_model, siglip_tok, device)
        val_bank, val_bank_vids = build_caption_bank(val_ex, siglip_model, siglip_tok, device)
        # CRITICAL: the training loop does random.shuffle(train_ex) in place every
        # epoch, so a row's position in the shuffled list no longer matches its row
        # in the bank. Stamp each example with its permanent bank index now, and use
        # THAT as the InfoNCE anchor index - otherwise every anchor would be
        # contrasted against the wrong "positive" caption and the loss would be
        # silently training on mislabeled pairs.
        for i, ex in enumerate(train_ex):
            ex["_bank_idx"] = i
        for i, ex in enumerate(val_ex):
            ex["_bank_idx"] = i
        print(f"[data] InfoNCE caption banks: train={tuple(train_bank.shape)} "
              f"val={tuple(val_bank.shape)}  (chance retrieval: train=1/{len(train_ex)}, "
              f"val=1/{len(val_ex)})")

    saved = []          # [(val_ap, epoch)] for every epoch, ranked at the end
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        badas.nn_model.train()
        if predictor is not None:
            predictor.train()
        random.shuffle(train_ex)
        opt.zero_grad()
        total_crash, total_sem, n, n_failed = 0.0, 0.0, 0, 0
        for step, ex in enumerate(train_ex):
            try:
                logits, patches = badas.forward(ex["frame_paths"])
            except (OSError, RuntimeError) as e:
                # A truncated/missing frame mid-run must not kill an 8-epoch GPU
                # job outright - skip the example, keep going, but surface it loudly.
                n_failed += 1
                print(f"  [warn] skipping {ex['video_id']} (tte={ex['tte']}): {e}")
                continue
            label = torch.tensor([ex["label"]], device=device)
            crash_loss = F.cross_entropy(logits, label)

            sem_loss = torch.tensor(0.0, device=device)
            if predictor is not None:
                # BADAS may run in fp16; the Predictor is fp32. .to(dtype=) is a
                # differentiable cast (autograd supports it) so the semantic-loss
                # gradient still flows back into the LoRA-unfrozen trunk.
                patches32 = patches.unsqueeze(0).to(dtype=torch.float32)
                # mean over the num_queries=8 tokens -> one Dt vector, comparable
                # to the single SigLIP caption embedding (was .squeeze(1) when
                # num_queries was 1 - see the predictor construction comment).
                pred = predictor(patches32).mean(dim=1)
                pred = F.normalize(pred, dim=-1)
                if args.semantic_loss == "infonce":
                    # ex["_bank_idx"], NOT the loop index - train_ex is reshuffled
                    # every epoch (see the bank-construction comment above).
                    sem_loss = infonce_from_bank(pred, ex["_bank_idx"], train_bank,
                                                  train_bank_vids, log_tau)
                else:
                    tgt = siglip_text_embed([ex["caption"]], siglip_model, siglip_tok, device)
                    sem_loss = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()

            loss = (crash_loss + args.semantic_weight * sem_loss) / args.grad_accum
            loss.backward()
            total_crash += crash_loss.item()
            total_sem += sem_loss.item()
            n += 1
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad()
        if n % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad()
        if n_failed:
            print(f"  [warn] {n_failed}/{len(train_ex)} examples failed to load this epoch")

        val_ap = evaluate_crash_ap(badas, val_ex, device)
        val_crash_loss, val_sem_loss = evaluate_val_loss(
            badas, val_ex, device, predictor, siglip_model, siglip_tok, args.semantic_weight,
            semantic_loss=args.semantic_loss, val_bank=val_bank, val_vids=val_bank_vids,
            log_tau=log_tau)
        elapsed = time.time() - t0
        avg_crash = total_crash / n if n else float("nan")
        avg_sem = total_sem / n if n else float("nan")
        # combined train/val loss, same weighting as the actual optimized objective -
        # this (not crash_loss alone) is what "train vs val gap" should compare, since
        # for B the model is optimizing crash+semantic jointly.
        train_total_loss = avg_crash + args.semantic_weight * avg_sem
        val_total_loss = val_crash_loss + args.semantic_weight * val_sem_loss
        train_val_gap = val_total_loss - train_total_loss  # >0 and growing = overfitting
        print(f"  epoch {epoch}/{args.epochs}  crash_loss={avg_crash:.4f}  "
              f"sem_loss={avg_sem:.4f}  val_crash_loss={val_crash_loss:.4f}  "
              f"val_sem_loss={val_sem_loss:.4f}  val_ap={val_ap:.4f}  "
              f"train_val_gap={train_val_gap:.4f}  ({elapsed:.1f}s)")

        with open(out_dir / "epoch_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "crash_loss": avg_crash, "sem_loss": avg_sem,
                "val_crash_loss": None if val_crash_loss != val_crash_loss else val_crash_loss,
                "val_sem_loss": None if val_sem_loss != val_sem_loss else val_sem_loss,
                "train_total_loss": train_total_loss, "val_total_loss": val_total_loss,
                "train_val_gap": None if train_val_gap != train_val_gap else train_val_gap,
                "val_ap": None if val_ap != val_ap else val_ap,
                "n_failed": n_failed, "elapsed_s": round(elapsed, 1),
            }) + "\n")

        ep_dir = out_dir / f"epoch_{epoch:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        badas.nn_model.save_pretrained(str(ep_dir / "lora_adapter"))
        if predictor is not None:
            torch.save(predictor.state_dict(), ep_dir / "predictor.pt")
        saved.append((val_ap, epoch))

    # Rank epochs: highest val_ap first; NaN -> -inf so a degenerate run (single-
    # class val split) falls back to the LAST epochs by number. Ties -> later
    # epoch wins. Keep the top --keep-top-k checkpoints (default 3, mirrors B1);
    # pass --keep-top-k >= --epochs to keep every epoch, e.g. to pick a checkpoint
    # by hand afterward using epoch_metrics.jsonl's train_val_gap column instead of
    # trusting val_ap alone.
    ranked = sorted(saved, key=lambda r: (r[0] if r[0] == r[0] else float("-inf"), r[1]),
                    reverse=True)
    topk = ranked[:args.keep_top_k]
    keep = {e for _, e in topk}
    for _, e in saved:
        if e not in keep:
            shutil.rmtree(out_dir / f"epoch_{e:02d}", ignore_errors=True)
    best_ap, best_epoch = topk[0]
    print(f"\n[done] top-{args.keep_top_k} by val_ap: " +
          ", ".join(f"ep{e} (val_ap={va:.4f})" for va, e in topk))
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"stage": stage,
                    "best_val_ap": (None if best_ap != best_ap else round(best_ap, 4)),
                    "best_epoch": best_epoch,
                    "keep_top_k": args.keep_top_k,
                    "top_checkpoints": [{"epoch": e, "val_ap": (None if va != va else round(va, 4))}
                                         for va, e in topk],
                    "n_train": len(train_ex), "n_val": len(val_ex),
                    "semantic_weight": args.semantic_weight,
                    "semantic_loss": args.semantic_loss if args.semantic_weight > 0 else None,
                    "infonce_tau_final": (float(log_tau.exp().item())
                                           if log_tau is not None else None),
                    "captions_path": args.captions_path or "outputs/semantic_captions/Caption_Train_All_Clips.jsonl",
                    # Full run config, so "was B run identical to A1 except for
                    # semantic_weight" is a fact checkable from disk, not an
                    # assertion in a markdown file.
                    "args": vars(args)}, f, indent=2)

    if not (args.test_manifest and args.test_frames_root):
        return

    # ---- Test-score EACH kept checkpoint (--keep-top-k of them) on the real test set. ----
    # Which epoch "wins" on val is noisy at this data scale, so scoring all of them
    # shows the spread rather than betting the headline on one checkpoint - and with
    # --keep-top-k set high, this is also how a per-epoch overfitting curve gets built
    # (read alongside epoch_metrics.jsonl's train_val_gap).
    from safetensors.torch import load_file
    from peft.utils import set_peft_model_state_dict

    records = load_manifest(args.test_manifest)
    if args.test_limit:
        records = records[: args.test_limit]
    pattern = stagea_cfg["data"]["frame_filename_pattern"]
    gt_field = stagea_cfg["data"]["gt_field"]

    def score_checkpoint(epoch, res_path):
        adapter_sd = load_file(str(out_dir / f"epoch_{epoch:02d}" / "lora_adapter"
                                   / "adapter_model.safetensors"))
        set_peft_model_state_dict(badas.nn_model, adapter_sd)
        badas.nn_model.eval()
        yt, ys, grp = [], [], []
        n_failed = 0
        # Stream + flush per clip: a failure at clip 500/677 must not discard the
        # 500 already-scored clips (this scores the top-3 checkpoints back-to-back,
        # so a late failure previously meant re-running everything before it too).
        with open(res_path, "w", encoding="utf-8") as f:
            for r in records:
                paths = frame_paths_for(r, args.test_frames_root, pattern)
                try:
                    with torch.no_grad():
                        logits, _ = badas.forward(paths)
                        s = float(torch.softmax(logits, dim=1)[0, 1].item())
                except (OSError, RuntimeError) as e:
                    n_failed += 1
                    print(f"  [warn] skipping test clip {r.get('video_id')}: {e}")
                    continue
                gt, g = int(r[gt_field]), r.get("group")
                f.write(json.dumps({"video_id": r["video_id"], "ground_truth": gt,
                                     "group": g, "score": round(s, 4)}) + "\n")
                f.flush()
                yt.append(gt); ys.append(s); grp.append(g)
        if n_failed:
            print(f"  [warn] {n_failed}/{len(records)} test clips failed to score")
        return yt, ys, grp

    summary = []
    for rank, (va, epoch) in enumerate(topk, 1):
        print(f"\n[test] scoring top-{rank} checkpoint (epoch {epoch}, "
              f"val_ap={va:.4f}) on {len(records)} clips ...")
        res_path = out_dir / f"test_results_ep{epoch:02d}.jsonl"
        yt, ys, grp = score_checkpoint(epoch, res_path)
        m = metrics_from_arrays(yt, ys, groups=grp, threshold=0.5)
        with open(out_dir / f"metrics_ep{epoch:02d}.json", "w", encoding="utf-8") as f:
            json.dump({"stage": stage, "epoch": epoch, "rank": rank,
                       "val_ap": (None if va != va else round(va, 4)), **m}, f, indent=2)
        per = m.get("per_tte_ap", {})
        print(f"       test_AP={m['ap']}  AUC={m['auc_roc']}  F1={m['f1']} "
              f"(F1*={m['f1_optimal']}@{m['optimal_threshold']})  "
              f"recall={m['recall_sensitivity_tpr']}  spec={m['specificity_tnr']}  "
              f"acc={m['accuracy']}  Brier={m['brier']}  ECE={m['ece']}")
        print(f"       per-TTE AP: " +
              "  ".join(f"{k}={v['ap']}(n={v['n']})" for k, v in per.items()))
        summary.append({"rank": rank, "epoch": epoch,
                        "val_ap": (None if va != va else round(va, 4)),
                        "test_ap": m["ap"], "auc_roc": m["auc_roc"], "f1": m["f1"],
                        "f1_optimal": m["f1_optimal"], "recall": m["recall_sensitivity_tpr"],
                        "specificity": m["specificity_tnr"], "brier": m["brier"],
                        "ece": m["ece"], "per_tte_ap": per})

    with open(out_dir / "test_summary.json", "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "semantic_weight": args.semantic_weight,
                   "n_test": len(records), "checkpoints": summary}, f, indent=2)
    print(f"\n[test] wrote per-checkpoint metrics + {out_dir / 'test_summary.json'} "
          f"(best = top-1, epoch {best_epoch})")


if __name__ == "__main__":
    main()
