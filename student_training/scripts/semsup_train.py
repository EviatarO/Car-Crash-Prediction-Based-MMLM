"""
semsup_train.py
================
Unified A1 / B / P1-two-stage trainer. LoRA-unfreezes the BADAS ViT-L trunk.
Optimized loss = crash_weight * crash_CE + semantic_weight * semantic_loss.

  --crash-weight 1 (default) --semantic-weight 0   -> Stage A1 (crash-only control)
  --crash-weight 1           --semantic-weight >0   -> Stage B   (crash + semantic-aux;
                              needs --predictor-init from semsup_b1_probe.py for
                              warm-start, or trains the Predictor from scratch alongside)
  --crash-weight 0           --semantic-weight >0   -> Stage A of the P1 two-stage
                              design (2026-08 plan): semantic objective ONLY, no crash
                              gradient reaches the trunk. Requires --select-by retrieval
                              (val_ap is uninformative when nothing optimizes it).

Selects the best epoch by --select-by (default val_ap, clip-level; 'retrieval' = clip-
level caption retrieval@1, required for Stage A), then scores the REAL 677-clip Private
test set with the selected checkpoint(s) and writes a results JSONL compatible with
evaluate_metrics.py's schema ({video_id, ground_truth, score, group}).

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

  # P1 Stage A (semantic-only, select on retrieval - no test scoring, no crash label
  # ever touches the trunk):
  python semsup_train.py --config ../configs/e4_stageA.yaml \
      --lora-target-modules query,key,value \
      --crash-weight 0.0 --semantic-weight 1.0 --semantic-loss infonce \
      --select-by retrieval --epochs 12 --clip-grad-per-group \
      --out-dir /workspace/semsup/p1_stageA
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
# Lifted to module level in semsup_b1_probe.py 2026-08-17 (P1 plan, change #2)
# specifically so this import is possible - was nested inside that file's main().
from semsup_b1_probe import clip_level_retrieval_acc  # noqa: E402


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


def _clip_grads(args, lora_params, aux_params, trainable):
    """Gradient clipping, matching A1's budget for the LoRA trunk when requested.

    Default (--clip-grad-per-group not set): ONE clip_grad_norm_(trainable, 1.0) over
    every trainable param combined - the original behavior, kept as default so existing
    runs (A1, B_1761-parallel, B-v2) remain reproducible byte-for-byte.

    --clip-grad-per-group: clip lora_params and aux_params (Predictor+log_tau) against
    SEPARATE budgets of 1.0 each. See the CLI help for --clip-grad-per-group for why this
    matters: without it, a large early Predictor gradient can inflate the shared global
    norm and shrink the LoRA trunk's effective update below what A1 (LoRA-only, nothing
    else in its `trainable` to share the budget with) receives for an identical crash loss.
    """
    if args.clip_grad_per_group and aux_params:
        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        torch.nn.utils.clip_grad_norm_(aux_params, 1.0)
    else:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)


def evaluate_val(badas, examples, device, predictor=None, siglip_model=None,
                 siglip_tok=None, semantic_loss="cosine", val_bank=None,
                 val_vids=None, log_tau=None, full_bank=None,
                 retrieval_tolerance=0.92):
    """ONE pass over the val set ->
    (val_ap, val_crash_loss, val_sem_loss, n_failed, retrieval_stats).

    Replaces the old evaluate_crash_ap + evaluate_val_loss pair. Each of those ran a
    FULL ViT-L forward over every val window, and the second returned a strict
    superset of the first's work - ~7% of every epoch spent recomputing something
    already computed 30 seconds earlier.

    Merging also closes a real failure mode: evaluate_crash_ap had NO per-clip error
    handling while the training loop and evaluate_val_loss both did. A single
    truncated JPEG on the val volume therefore killed the process *after* a full
    epoch of training and *before* the checkpoint was written - losing the epoch
    outright. Both concerns now share one guarded loop and one failure counter.

    AP is per CLIP, not per row: `examples` are TTE windows of the same clip sharing
    one label, so computing AP over correlated rows silently inflates it (see
    EXPERIMENTS.md's val-split diagnostic - row-level val_ap saturated at 0.96-0.98
    and ranked checkpoints in the OPPOSITE order from test_AP). Losses stay per-row,
    matching the training loop's own accounting so train/val are like-for-like.

    For semantic_loss='infonce' the val bank is the VAL set's own captions, keeping
    val InfoNCE the analogue of the train term. Note N differs between train and val,
    so compare the TREND across epochs, not the absolute train-vs-val difference.

    retrieval_stats (2026-08-17, P1 plan changes #3/#5): a dict, EMPTY unless both
    `predictor` and `val_bank` are given (i.e. semantic_loss='infonce' - cosine has
    no bank to retrieve against). Every anchor's own true target is looked up via
    its `_bank_idx` rather than assumed positionally aligned with `val_bank`, so
    this is correct even when some val windows fail to load this epoch (n_failed>0
    would otherwise desync a positional lookup). Keys:
      retrieval_clip            - strict clip-level retrieval@1 among the val set's
                                   own ~221 clips (chance = 1/n_clips). PRIMARY metric.
      collapse_control_clip     - same task, but every prediction replaced by this
                                   epoch's constant mean target embedding. If the
                                   real model doesn't clear this, it learned nothing
                                   beyond "always guess the average caption" - see
                                   the B1 probe's identical control and EXPERIMENTS.md.
      retrieval_clip_full1761   - retrieval among the val clips' own targets PLUS
                                   every train-set caption as extra distractors
                                   (only computed when `full_bank` is given). Chance
                                   drops from ~1/221 to ~1/(221+len(full_bank)).
                                   Distractors are train ROWS not train CLIPS (a
                                   train clip with multiple TTE windows contributes
                                   multiple near-duplicate distractor rows) - they
                                   can never be a val clip's correct answer (train
                                   and val share no video_id), so this only makes
                                   the task marginally harder, not incorrect.
      retrieval_clip_tolerant   - counts the top-1 retrieval a HIT if it is within
                                   `retrieval_tolerance` cosine of the true target,
                                   even when it is not the exact match - catches the
                                   case where the model retrieves a DIFFERENT clip's
                                   caption that still correctly describes the scene,
                                   which strict retrieval@1 scores as a plain miss.
      embed_margin_mean         - mean(s_true - max(s_other)) across anchors, on the
                                   SAME same-video-masked similarity row InfoNCE
                                   trains against. Shrinking/negative = the true
                                   caption is losing its lead over the field.
      embed_max_q_mean          - mean softmax-max-probability across anchors, on
                                   that same masked row. ->1.0 = SATURATION (the
                                   failure mode that actually matters here - see
                                   docs_agents/ARCHITECTURE_BLOCKS.md's embedding-
                                   health note: temperature AMPLIFIES this band's
                                   gradient ~18x, so saturation kills it, not size).
      embed_std_s_mean          - mean std of the masked similarity row. ->0 =
                                   every caption looks equally (dis)similar to this
                                   anchor - a different collapse signature.
      embed_std_p               - std of predicted embeddings across ALL processed
                                   val rows (mean over the Dt feature dims). ->0 =
                                   the Predictor is emitting a near-constant vector
                                   regardless of input video - the exact degenerate
                                   solution that made the original cosine loss null
                                   (see B1's collapse control in EXPERIMENTS.md).
      n_retrieval_clips         - denominator for retrieval_clip (~221 on val).
    """
    badas.nn_model.eval()
    if predictor is not None:
        predictor.eval()
    from collections import defaultdict
    by_clip = defaultdict(list)
    total_crash, total_sem, n, n_failed = 0.0, 0.0, 0, 0
    # Only populated when predictor+val_bank both exist - see retrieval_stats above.
    pred_list, tgt_list, vid_list, bank_idx_list = [], [], [], []
    with torch.no_grad():
        # Same prefetch pipeline as the training loop - see
        # TrainableBadasWrapper.prefetch_clips()'s docstring. `i` is the
        # position in `examples` (identical to enumerate(examples)'s i, since
        # val_ex is never shuffled), so the InfoNCE _bank_idx fallback below
        # is unaffected by switching from enumerate() to this generator.
        for i, ex, clip, err in badas.prefetch_clips(examples, num_workers=8, prefetch=16):
            if err is not None:
                n_failed += 1
                print(f"  [warn] val: skipping {ex.get('video_id')} "
                      f"(tte={ex.get('tte')}): {err}")
                continue
            logits, patches = badas.forward_clip(clip.to(device))

            # --- ranking signal (per clip) ---
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
            by_clip[ex["video_id"]].append((score, ex["label"]))

            # --- loss signal (per row) ---
            label = torch.tensor([ex["label"]], device=device)
            crash_loss = F.cross_entropy(logits, label)
            sem_loss = torch.tensor(0.0, device=device)
            if predictor is not None:
                patches32 = patches.unsqueeze(0).to(dtype=torch.float32)
                pred = predictor(patches32).mean(dim=1)
                pred = F.normalize(pred, dim=-1)
                if semantic_loss == "infonce":
                    # stamped bank index, not the loop counter - val_ex happens not
                    # to be shuffled, but relying on that is exactly the assumption
                    # that would break silently if it ever changes.
                    bank_idx = ex.get("_bank_idx", i)
                    sem_loss = infonce_from_bank(pred, bank_idx, val_bank, val_vids, log_tau)
                    if val_bank is not None:
                        pred_list.append(pred.squeeze(0).detach())
                        tgt_list.append(val_bank[bank_idx])
                        vid_list.append(ex["video_id"])
                        bank_idx_list.append(bank_idx)
                else:
                    tgt = siglip_text_embed([ex["caption"]], siglip_model, siglip_tok, device)
                    sem_loss = (1 - F.cosine_similarity(pred, tgt, dim=-1)).mean()
            total_crash += crash_loss.item()
            total_sem += sem_loss.item()
            n += 1

    if n_failed:
        print(f"  [warn] val: {n_failed}/{len(examples)} windows failed to load")
    if n == 0:
        return float("nan"), float("nan"), float("nan"), n_failed, {}

    ys, yt = [], []
    for pairs in by_clip.values():
        labels = {l for _, l in pairs}
        # Verified empirically that every clip is single-label across its TTE
        # windows; majority-vote is a defensive fallback rather than a crash if that
        # ever stops holding (e.g. a future mixed-label caption source).
        label = next(iter(labels)) if len(labels) == 1 else \
            round(sum(l for _, l in pairs) / len(pairs))
        ys.append(sum(s for s, _ in pairs) / len(pairs))
        yt.append(label)
    val_ap = average_precision_score(yt, ys) if len(set(yt)) >= 2 else float("nan")

    retrieval_stats = {}
    if pred_list and val_bank is not None:
        P = torch.stack(pred_list)              # (n_rows, Dt)
        T = torch.stack(tgt_list)                # (n_rows, Dt) - each row's OWN true target

        # --- primary + collapse-control retrieval (reuses the lifted, tested helper) ---
        retrieval_stats["retrieval_clip"] = clip_level_retrieval_acc(P, T, vid_list)
        mean_emb = F.normalize(T.mean(dim=0, keepdim=True), dim=-1).expand_as(T)
        retrieval_stats["collapse_control_clip"] = clip_level_retrieval_acc(mean_emb, T, vid_list)

        # --- pool predictions and their own targets per clip once, reused below ---
        by_p, by_t = defaultdict(list), defaultdict(list)
        for idx, v in enumerate(vid_list):
            by_p[v].append(P[idx])
            by_t[v].append(T[idx])
        clip_ids = sorted(by_p.keys())
        n_clips = len(clip_ids)
        retrieval_stats["n_retrieval_clips"] = n_clips
        if n_clips >= 2:
            Pc = torch.stack([F.normalize(torch.stack(by_p[v]).mean(0), dim=-1) for v in clip_ids])
            Tc = torch.stack([F.normalize(torch.stack(by_t[v]).mean(0), dim=-1) for v in clip_ids])

            # --- retrieval vs the full corpus (val's own clips + all train rows as
            #     extra distractors). Correct index for clip k is k, since Tc is
            #     placed first in the candidate matrix. ---
            if full_bank is not None and full_bank.numel() > 0:
                candidates = torch.cat([Tc, full_bank.to(Tc.device)], dim=0)
                sims_full = Pc @ candidates.T
                top1_full = sims_full.argmax(dim=1)
                correct_idx = torch.arange(n_clips, device=Pc.device)
                retrieval_stats["retrieval_clip_full1761"] = \
                    (top1_full == correct_idx).float().mean().item()

            # --- similarity-tolerant retrieval: credit a near-miss whose retrieved
            #     caption still genuinely resembles the true one. ---
            sims_val = Pc @ Tc.T
            top1_val = sims_val.argmax(dim=1)
            retrieved = Tc[top1_val]
            sim_to_true = F.cosine_similarity(retrieved, Tc, dim=-1)
            retrieval_stats["retrieval_clip_tolerant"] = \
                (sim_to_true >= retrieval_tolerance).float().mean().item()

        # --- embedding-health stats, on the SAME same-video-masked row InfoNCE
        #     actually trains against (see the docstring above for what each catches) ---
        if log_tau is not None and val_vids is not None:
            tau_val = log_tau.exp().clamp(min=1e-2, max=1.0)
            sims_bank = P @ val_bank.T                     # (n_rows, N_val_bank)
            margins, max_qs, stds = [], [], []
            for row_i, bidx in enumerate(bank_idx_list):
                anchor_vid = val_vids[bidx]
                same_vid = torch.tensor([v == anchor_vid for v in val_vids], device=device)
                same_vid[bidx] = False
                s_row = sims_bank[row_i].masked_fill(same_vid, float("-inf"))
                s_true = s_row[bidx].item()
                s_wo_true = s_row.clone()
                s_wo_true[bidx] = float("-inf")
                margins.append(s_true - s_wo_true.max().item())
                q = torch.softmax(s_row / tau_val, dim=0)
                max_qs.append(q.max().item())
                finite = s_row[s_row > -1e30]
                if finite.numel() > 1:
                    stds.append(finite.std().item())
            if margins:
                retrieval_stats["embed_margin_mean"] = sum(margins) / len(margins)
                retrieval_stats["embed_max_q_mean"] = sum(max_qs) / len(max_qs)
            if stds:
                retrieval_stats["embed_std_s_mean"] = sum(stds) / len(stds)

        retrieval_stats["embed_std_p"] = P.std(dim=0).mean().item()

    return val_ap, total_crash / n, total_sem / n, n_failed, retrieval_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora-target-modules", required=True,
                     help="Comma-separated module-name SUBSTRINGS (e.g. 'query,key,value'), "
                          "or a single REGEX if the value starts with 're:'. Prefer the regex "
                          "form: bare 'query,key,value' matches 108 Linears on BADAS-Open - the "
                          "72 encoder ones you want PLUS 36 on the V-JEPA2 predictor "
                          "(latent-forecast) stack, which is not on the classification path. "
                          r"Encoder-only: --lora-target-modules "
                          r"'re:backbone\.encoder\.layer\.\d+\.attention\.(query|key|value)'")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--crash-weight", type=float, default=1.0,
                     help="weight on the crash CE term in the optimized loss (default 1.0, "
                          "matching every prior run). 0.0 = Stage A of the P1 two-stage design "
                          "(2026-08 plan): train the semantic branch ALONE, with no crash "
                          "gradient reaching the trunk at all. The raw (unweighted) crash loss "
                          "is still computed and logged every epoch regardless of this weight - "
                          "it is a free diagnostic of whether the frozen crash head still fits "
                          "the drifting representation, it just doesn't drive the LoRA update "
                          "when --crash-weight 0.")
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
    ap.add_argument("--clip-grad-per-group", action="store_true",
                     help="clip the LoRA trunk and the semantic branch (Predictor+log_tau) "
                          "on SEPARATE gradient-norm budgets (1.0 each), instead of one "
                          "shared clip_grad_norm_(trainable, 1.0) over all params combined. "
                          "Without this, A1 (LoRA-only) and B (LoRA+Predictor) do not use "
                          "the same effective clip budget for the LoRA trunk - large "
                          "Predictor gradients can inflate the shared global norm and "
                          "silently shrink B's crash-loss updates relative to A1's, for "
                          "reasons unrelated to semantic_weight. No-op when semantic_weight=0 "
                          "(A1: aux_params is empty either way).")
    ap.add_argument("--grad-cosine-every", type=int, default=8,
                     help="measure the angle between the crash and semantic gradients on the "
                          "SHARED LoRA params every N windows (0=off). Reported per epoch as "
                          "grad_cos_mean / grad_cos_frac_neg. cos<0 means the two objectives "
                          "pull the trunk in opposing directions (destructive interference); "
                          "cos>0 means they agree. Costs 2 extra partial backward passes on "
                          "the sampled steps (~15% epoch time at N=8) and does NOT touch the "
                          "optimizer - autograd.grad() returns gradients without accumulating "
                          "into .grad, so training is bit-identical with this on or off.")
    ap.add_argument("--lora-init", default=None,
                     help="path to an existing lora_adapter DIRECTORY (e.g. "
                          "/workspace/semsup/a1_1761/epoch_04/lora_adapter) to START training "
                          "from, instead of a fresh random LoRA init. Two uses: (1) SEQUENTIAL "
                          "training - continue a converged crash-only model with the semantic "
                          "loss added; (2) RESUMING an interrupted run - point at the last "
                          "completed epoch's adapter. Note the trunk's frozen base weights are "
                          "unchanged either way; only the LoRA delta is loaded.")
    ap.add_argument("--optimizer-init", default=None,
                     help="path to an optimizer.pt saved by a previous run's epoch dir. Restores "
                          "Adam moment estimates so a resumed run continues the SAME optimization "
                          "trajectory rather than restarting momentum from zero. Optional - "
                          "resuming without it works but is a slightly different trajectory.")
    ap.add_argument("--start-epoch", type=int, default=1,
                     help="epoch number to start counting from when resuming, so epoch dirs and "
                          "epoch_metrics.jsonl continue the original numbering instead of "
                          "overwriting epoch_01. --epochs is still the LAST epoch number to run, "
                          "not a count: --start-epoch 6 --epochs 10 runs epochs 6..10.")
    ap.add_argument("--early-stop-patience", type=int, default=0,
                     help="stop if val_ap hasn't improved for this many consecutive epochs "
                          "(0 = disabled, the default/original behavior). Checkpoints for every "
                          "epoch run are still written, so an early stop loses nothing.")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                     help="'constant' (default, original behavior) or 'cosine': linear "
                          "warmup for --warmup-frac of total optimizer steps, then cosine "
                          "decay to 0 over the rest. A1_1761 trained at a constant 2e-4 and "
                          "overfits early - val_crash_loss bottoms at epoch 2 then climbs to "
                          "0.81 by epoch 6 while train loss keeps falling - a schedule targets "
                          "exactly that failure mode.")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--prefetch-workers", type=int, default=8,
                     help="threads concurrently reading+decoding frames ahead of the GPU. "
                          "Measured 2026-08-11: preprocessing (file read+decode/resize) is "
                          "~100% of per-window wall time on a slow network volume, GPU "
                          "compute is unmeasurable by comparison - so this is the actual "
                          "speed knob, not batch size or LR. 0 disables prefetch (old "
                          "serial behavior, for debugging).")
    ap.add_argument("--prefetch-depth", type=int, default=16,
                     help="how many windows ahead the prefetch pipeline stays filled.")
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
    ap.add_argument("--select-by", default="val_ap", choices=["val_ap", "retrieval"],
                     help="checkpoint-ranking/early-stop metric (default val_ap, unchanged "
                          "behavior). 'retrieval' = clip-level caption retrieval@1 on the val "
                          "set - REQUIRED for Stage A of the P1 two-stage design, where "
                          "crash_weight=0 makes val_ap uninformative about what Stage A is "
                          "actually optimizing. This is the same bug class fixed in "
                          "semsup_b1_probe.py on 2026-08-12 (that probe was selecting InfoNCE "
                          "checkpoints on val_loss, which ranked differently from retrieval and "
                          "picked a measurably worse predictor: 0.1086 vs 0.1267 available). "
                          "Requires --semantic-weight > 0 (a predictor + caption bank must "
                          "exist to compute retrieval at all).")
    ap.add_argument("--retrieval-tolerance", type=float, default=0.92,
                     help="cosine threshold for the similarity-tolerant retrieval@1 variant: "
                          "count a hit if the TOP-1 retrieved caption is within this cosine of "
                          "the true one, even if it is not the exact match. Addresses a real "
                          "blind spot in strict retrieval@1 - if the model retrieves a DIFFERENT "
                          "clip's caption that happens to correctly describe the scene, strict "
                          "retrieval scores it as a miss. Default 0.92 sits above the measured "
                          "p99 cross-video caption similarity (0.870, see "
                          "docs_agents/ARCHITECTURE_BLOCKS.md), so it should not credit "
                          "genuinely-different captions as tolerant hits.")
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

    if args.crash_weight == 0 and args.semantic_weight == 0:
        raise ValueError("--crash-weight and --semantic-weight are both 0 - nothing would be "
                          "optimized. Pick at least one nonzero weight.")
    if args.select_by == "retrieval" and args.semantic_weight <= 0:
        raise ValueError("--select-by retrieval requires --semantic-weight > 0 (a Predictor + "
                          "caption bank must exist to compute retrieval@1 at all).")

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
    # Three-way, not two: crash_weight=0 is Stage A of the P1 two-stage design
    # (semantic-only, no crash gradient reaches the trunk at all) - distinct from
    # both the crash-only control (A1) and the joint crash+semantic arm (B).
    if args.crash_weight > 0 and args.semantic_weight > 0:
        stage = "B (crash+semantic)"
    elif args.crash_weight > 0:
        stage = "A1 (crash-only)"
    else:
        stage = "A (semantic-only, P1 Stage A)"
    sem_note = f"  semantic_loss={args.semantic_loss}" if args.semantic_weight > 0 else ""
    print(f"[cfg] stage={stage}  crash_weight={args.crash_weight}  "
          f"semantic_weight={args.semantic_weight}{sem_note}  select_by={args.select_by}  "
          f"lora_target_modules={args.lora_target_modules}  seed={args.seed}")

    # 're:' prefix -> pass the regex through to peft untouched (peft accepts a single
    # regex string as target_modules). Otherwise keep the legacy comma-substring form
    # so existing runbooks and the recorded A1_1761/B_1761 commands still reproduce.
    if args.lora_target_modules.startswith("re:"):
        target_modules = args.lora_target_modules[3:]
    else:
        target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    badas = TrainableBadasWrapper(
        stagea_cfg, lora_target_modules=target_modules,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    )
    # peft's save_pretrained() auto-generates a model card BEFORE writing any
    # adapter weights, and assumes base_model.config supports `in` (a HF
    # PretrainedConfig). BADAS's V-JEPA2 uses a plain ModelArgs dataclass
    # instead, so that step crashes save_pretrained() every time, before any
    # checkpoint is written. We don't need the model card - skip it.
    badas.nn_model.create_or_update_model_card = lambda *a, **k: None

    # Load an existing LoRA delta BEFORE building the optimizer, so the optimizer
    # is constructed over the same parameter objects that were just overwritten.
    if args.lora_init:
        from safetensors.torch import load_file as _load_sft
        from peft.utils import set_peft_model_state_dict as _set_peft_sd
        adapter_path = Path(args.lora_init)
        sft = adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
        if not sft.exists():
            raise FileNotFoundError(
                f"--lora-init {args.lora_init} does not contain adapter_model.safetensors "
                f"(looked at {sft}). Point it at an epoch's lora_adapter/ directory."
            )
        _set_peft_sd(badas.nn_model, _load_sft(str(sft)))
        print(f"[load] initialized LoRA from {sft}")

    trainable = [p for p in badas.nn_model.parameters() if p.requires_grad]
    # Captured BEFORE the Predictor is appended below: the LoRA trunk params only.
    # The crash-vs-semantic gradient angle is only meaningful on the parameters the
    # two objectives actually SHARE - the Predictor is semantic-only (its crash
    # gradient is identically zero, which would drag any cosine toward 0).
    lora_params = list(trainable)

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

    # Params trained ONLY for the semantic branch (Predictor + log_tau) - everything
    # in `trainable` after `lora_params` was captured above, in construction order.
    aux_params = trainable[len(lora_params):]

    opt = torch.optim.AdamW(trainable, lr=args.lr)
    if args.optimizer_init:
        opt.load_state_dict(torch.load(args.optimizer_init, map_location=device))
        print(f"[load] restored optimizer state from {args.optimizer_init}")

    examples = load_training_examples(limit=args.limit, captions_path=args.captions_path)
    if len(examples) < args.min_examples:
        raise RuntimeError(
            f"Only {len(examples)} training examples loaded (< --min-examples "
            f"{args.min_examples}). This usually means dataset/train is missing or "
            f"partially synced - check the symlink/volume before training on a "
            f"silently-shrunk dataset."
        )
    # A placeholder pool (build_pool_from_manifest.py) carries one identical caption
    # on every row. That is harmless for A1 (no Predictor is built at all) but would
    # give Stage B the exact degenerate target the semantic branch was redesigned to
    # avoid: every clip aligned to the same embedding. Fail loudly rather than train
    # a meaningless B arm for hours.
    if args.semantic_weight > 0:
        n_uniq = len({ex["caption"] for ex in examples})
        if any(ex["caption"].startswith("PLACEHOLDER-NOT-A-CAPTION") for ex in examples):
            raise RuntimeError(
                f"--captions-path points at a PLACEHOLDER pool "
                f"({args.captions_path}) but --semantic-weight={args.semantic_weight} > 0. "
                f"That pool is crash-only (built by build_pool_from_manifest.py); its "
                f"captions are a tripwire, not text. Use a real caption file for Stage B."
            )
        if n_uniq < 0.5 * len(examples):
            print(f"[warn] only {n_uniq} unique captions across {len(examples)} rows "
                  f"({100*n_uniq/len(examples):.0f}%) - duplicate captions weaken InfoNCE "
                  f"negatives (near-duplicates get punished as if they were wrong).")

    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[data] train={len(train_ex)}  val={len(val_ex)} (clip-level split)")

    # Built here, not at optimizer construction, because total_steps needs
    # len(train_ex) - only known after the pool is loaded and split. If resuming
    # (--start-epoch > 1), remaining_epochs covers only what's left to run, so the
    # schedule still decays to 0 exactly at --epochs regardless of where it starts.
    scheduler = None
    if args.lr_schedule == "cosine":
        remaining_epochs = args.epochs - args.start_epoch + 1
        steps_per_epoch = max(1, -(-len(train_ex) // args.grad_accum))  # ceil div
        total_steps = max(1, remaining_epochs * steps_per_epoch)
        warmup_steps = max(1, int(args.warmup_frac * total_steps))
        print(f"[sched] cosine: total_steps={total_steps}  warmup_steps={warmup_steps} "
              f"({steps_per_epoch} steps/epoch x {remaining_epochs} epochs remaining)")

        def _lr_lambda(step):
            import math
            if step < warmup_steps:
                return step / warmup_steps
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

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
    best_sel_seen, epochs_since_improve = float("-inf"), 0
    for epoch in range(args.start_epoch, args.epochs + 1):
        badas.nn_model.train()
        if predictor is not None:
            predictor.train()
        random.shuffle(train_ex)
        opt.zero_grad()
        epoch_t0 = time.time()
        total_crash, total_sem, n, n_failed = 0.0, 0.0, 0, 0
        # Crash-vs-semantic gradient-angle accumulators (diagnostic; see --grad-cosine-every).
        cos_sum, cos_n, cos_neg = 0.0, 0, 0
        gnorm_crash, gnorm_sem = 0.0, 0.0
        gc_probe_failed = False
        # `pending` counts SUCCESSFUL backward() calls since the last opt.step().
        # Driving the accumulation boundary off the enumerate() index instead would
        # desync the moment any example is skipped: some steps would average fewer
        # than grad_accum examples while still dividing by grad_accum, and the
        # post-loop flush could evaluate False while real gradients are still
        # pending - silently discarded by the next epoch's zero_grad().
        pending = 0
        # Prefetch pipeline, not a plain for-loop over badas.forward(): direct
        # profiling (2026-08-11) found preprocessing (file read + decode/resize)
        # is ~100% of per-window wall time on a slow network volume and GPU
        # compute is unmeasurable by comparison - so multiple windows' I/O must
        # happen CONCURRENTLY, not just be overlapped with GPU compute. See
        # TrainableBadasWrapper.prefetch_clips()'s docstring for the full
        # measurement. Error handling is unchanged: a failed window still just
        # increments n_failed and continues, same contract as the old inline
        # try/except.
        for _, ex, clip, err in badas.prefetch_clips(
                train_ex, num_workers=args.prefetch_workers, prefetch=args.prefetch_depth):
            if err is not None:
                # A truncated/missing frame mid-run must not kill an 8-epoch GPU
                # job outright - skip the example, keep going, but surface it loudly.
                n_failed += 1
                print(f"  [warn] skipping {ex['video_id']} (tte={ex['tte']}): {err}")
                continue
            logits, patches = badas.forward_clip(clip.to(device))
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

            # --- crash-vs-semantic gradient angle (diagnostic only, never optimized) ---
            # Measured on lora_params (the SHARED trunk) before the combined backward.
            # retain_graph=True is required because loss.backward() below reuses the graph.
            if (predictor is not None and args.grad_cosine_every
                    and not gc_probe_failed
                    and n % args.grad_cosine_every == 0):
                try:
                    g_c = torch.autograd.grad(crash_loss, lora_params,
                                              retain_graph=True, allow_unused=True)
                    g_s = torch.autograd.grad(sem_loss, lora_params,
                                              retain_graph=True, allow_unused=True)
                    fc = torch.cat([g.flatten() for g in g_c if g is not None])
                    fs = torch.cat([g.flatten() for g in g_s if g is not None])
                    if fc.numel() and fs.numel() and fc.numel() == fs.numel():
                        c = F.cosine_similarity(fc.unsqueeze(0), fs.unsqueeze(0)).item()
                        if c == c:                      # NaN guard (zero-norm gradient)
                            cos_sum += c
                            cos_n += 1
                            cos_neg += int(c < 0)
                            gnorm_crash += fc.norm().item()
                            gnorm_sem += fs.norm().item()
                except RuntimeError as exc:
                    # A freed graph or unused-input edge must not kill an 8-epoch run
                    # over a diagnostic. Report once, then stop trying this epoch.
                    if cos_n == 0:
                        print(f"  [warn] grad-cosine probe disabled this epoch: {exc}")
                    gc_probe_failed = True

            # crash_loss/sem_loss are ALWAYS both computed and logged raw (unweighted) -
            # crash_weight only controls what reaches the backward pass. At
            # --crash-weight 0 (Stage A), crash_loss is still a free diagnostic of
            # whether the frozen head still fits the drifting representation; it
            # just contributes zero gradient.
            loss = (args.crash_weight * crash_loss
                    + args.semantic_weight * sem_loss) / args.grad_accum
            loss.backward()
            total_crash += crash_loss.item()
            total_sem += sem_loss.item()
            n += 1
            pending += 1
            if pending == args.grad_accum:
                _clip_grads(args, lora_params, aux_params, trainable)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad()
                pending = 0
        if pending:                     # tail window - same counter, so never dropped
            _clip_grads(args, lora_params, aux_params, trainable)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad()
            pending = 0
        if n_failed:
            print(f"  [warn] {n_failed}/{len(train_ex)} examples failed to load this epoch")

        # ONE val pass for both the ranking metric and the two loss terms (was two
        # full ViT-L sweeps over the same windows). full_bank=train_bank enables the
        # vs-1761 retrieval variant (None under cosine loss, where no bank exists).
        val_ap, val_crash_loss, val_sem_loss, n_val_failed, retrieval_stats = evaluate_val(
            badas, val_ex, device, predictor, siglip_model, siglip_tok,
            semantic_loss=args.semantic_loss, val_bank=val_bank,
            val_vids=val_bank_vids, log_tau=log_tau, full_bank=train_bank,
            retrieval_tolerance=args.retrieval_tolerance)
        now = time.time()
        epoch_s = now - epoch_t0
        elapsed = now - t0
        avg_crash = total_crash / n if n else float("nan")
        avg_sem = total_sem / n if n else float("nan")
        # combined train/val loss, same weighting as the actual optimized objective -
        # this (not crash_loss alone) is what "train vs val gap" should compare, since
        # for B the model is optimizing crash+semantic jointly.
        train_total_loss = args.crash_weight * avg_crash + args.semantic_weight * avg_sem
        val_total_loss = args.crash_weight * val_crash_loss + args.semantic_weight * val_sem_loss
        train_val_gap = val_total_loss - train_total_loss  # >0 and growing = overfitting
        cur_lr = opt.param_groups[0]["lr"]
        # Checkpoint-ranking/early-stop metric (P1 plan, change #4). Both val_ap and
        # retrieval_clip are "higher is better", so no direction flip is needed
        # between the two --select-by modes (unlike semsup_b1_probe.py's val_loss
        # vs retrieval_clip, which point opposite ways).
        sel_value = (retrieval_stats.get("retrieval_clip", float("nan"))
                     if args.select_by == "retrieval" else val_ap)
        # Gradient-angle summary for this epoch. cos<0 on a step means the crash and
        # semantic objectives asked the shared trunk to move in opposing directions.
        grad_cos_mean = cos_sum / cos_n if cos_n else float("nan")
        grad_cos_frac_neg = cos_neg / cos_n if cos_n else float("nan")
        grad_norm_crash = gnorm_crash / cos_n if cos_n else float("nan")
        grad_norm_sem = gnorm_sem / cos_n if cos_n else float("nan")
        print(f"  epoch {epoch}/{args.epochs}  crash_loss={avg_crash:.4f}  "
              f"sem_loss={avg_sem:.4f}  val_crash_loss={val_crash_loss:.4f}  "
              f"val_sem_loss={val_sem_loss:.4f}  val_ap={val_ap:.4f}  "
              f"train_val_gap={train_val_gap:.4f}  lr={cur_lr:.2e}  ({elapsed:.1f}s)")
        if retrieval_stats:
            print(f"      [retrieval] clip={retrieval_stats.get('retrieval_clip', float('nan')):.4f}  "
                  f"vs_control={retrieval_stats.get('collapse_control_clip', float('nan')):.4f}  "
                  f"tolerant={retrieval_stats.get('retrieval_clip_tolerant', float('nan')):.4f}  "
                  f"vs_full1761={retrieval_stats.get('retrieval_clip_full1761', float('nan')):.4f}  "
                  f"n_clips={retrieval_stats.get('n_retrieval_clips', 0)}  "
                  f"(select_by={args.select_by}, sel_value={sel_value:.4f})")
            print(f"      [embed-health] margin={retrieval_stats.get('embed_margin_mean', float('nan')):.4f}  "
                  f"max_q={retrieval_stats.get('embed_max_q_mean', float('nan')):.4f}  "
                  f"std_s={retrieval_stats.get('embed_std_s_mean', float('nan')):.4f}  "
                  f"std_p={retrieval_stats.get('embed_std_p', float('nan')):.4f}")
        if cos_n:
            # Relative pull = how big the semantic update is vs the crash update AFTER
            # lambda is applied - the number that says whether the aux term is even
            # loud enough to matter, independent of whether it agrees in direction.
            rel = (args.semantic_weight * grad_norm_sem / grad_norm_crash
                   if grad_norm_crash else float("nan"))
            print(f"      [grad] cos(crash,sem)={grad_cos_mean:+.4f}  "
                  f"conflicting={100*grad_cos_frac_neg:.1f}% of {cos_n} sampled steps  "
                  f"|g_crash|={grad_norm_crash:.4f}  |g_sem|={grad_norm_sem:.4f}  "
                  f"lambda*|g_sem|/|g_crash|={rel:.3f}")

        # `_j` centralises the NaN->null guard. json.dumps defaults to allow_nan=True
        # and emits a bare `NaN` token, which is INVALID json: python's own loads()
        # accepts it so it survives local inspection, but jq/JS/Go and most
        # dashboards reject the whole line. n==0 (an unmounted frames volume) is
        # exactly when every one of these goes NaN - i.e. the moment you most need
        # to read the log. Previously only 4 of the 8 float fields were guarded.
        def _j(x):
            return None if isinstance(x, float) and x != x else x

        with open(out_dir / "epoch_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch,
                "crash_loss": _j(avg_crash), "sem_loss": _j(avg_sem),
                "val_crash_loss": _j(val_crash_loss), "val_sem_loss": _j(val_sem_loss),
                "train_total_loss": _j(train_total_loss),
                "val_total_loss": _j(val_total_loss),
                "train_val_gap": _j(train_val_gap),
                "val_ap": _j(val_ap),
                "select_by": args.select_by, "sel_value": _j(sel_value),
                # Retrieval + embedding-health stats (empty dict under cosine loss,
                # or before a Predictor exists at all - see evaluate_val()'s
                # retrieval_stats docstring for what each key means/catches).
                "retrieval_clip": _j(retrieval_stats.get("retrieval_clip", float("nan"))),
                "collapse_control_clip": _j(retrieval_stats.get("collapse_control_clip", float("nan"))),
                "retrieval_clip_full1761": _j(retrieval_stats.get("retrieval_clip_full1761", float("nan"))),
                "retrieval_clip_tolerant": _j(retrieval_stats.get("retrieval_clip_tolerant", float("nan"))),
                "n_retrieval_clips": retrieval_stats.get("n_retrieval_clips", 0),
                "embed_margin_mean": _j(retrieval_stats.get("embed_margin_mean", float("nan"))),
                "embed_max_q_mean": _j(retrieval_stats.get("embed_max_q_mean", float("nan"))),
                "embed_std_s_mean": _j(retrieval_stats.get("embed_std_s_mean", float("nan"))),
                "embed_std_p": _j(retrieval_stats.get("embed_std_p", float("nan"))),
                # Crash-vs-semantic gradient angle on the shared LoRA params.
                # grad_cos_mean < 0 or grad_cos_frac_neg > ~0.5 = destructive
                # interference: the aux objective is fighting crash prediction.
                "grad_cos_mean": _j(grad_cos_mean),
                "grad_cos_frac_neg": _j(grad_cos_frac_neg),
                "grad_norm_crash": _j(grad_norm_crash),
                "grad_norm_sem": _j(grad_norm_sem),
                "grad_cos_n_sampled": cos_n,
                "n_failed": n_failed, "n_val_failed": n_val_failed,
                # epoch_s = THIS epoch; elapsed_s = cumulative since run start.
                # Only the cumulative one existed before, logged under a name that
                # reads as per-epoch - so pod-budget estimates off this file were
                # wrong by a growing factor.
                "epoch_s": round(epoch_s, 1), "elapsed_s": round(elapsed, 1),
                "lr": cur_lr,
            }) + "\n")

        ep_dir = out_dir / f"epoch_{epoch:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        badas.nn_model.save_pretrained(str(ep_dir / "lora_adapter"))
        if predictor is not None:
            torch.save(predictor.state_dict(), ep_dir / "predictor.pt")
        # Optimizer state per epoch, so an interrupted run can resume on the SAME
        # trajectory (--optimizer-init) rather than restarting Adam momentum.
        torch.save(opt.state_dict(), ep_dir / "optimizer.pt")
        # (sel_value, epoch) - val_ap under the default --select-by, retrieval_clip
        # under Stage A. Both are "higher is better" so downstream ranking is unchanged.
        saved.append((sel_value, epoch))

        # Early stopping on sel_value (opt-in). Every epoch's checkpoint is already
        # written above, so stopping early discards nothing.
        if args.early_stop_patience > 0:
            if sel_value == sel_value and sel_value > best_sel_seen:   # NaN-safe
                best_sel_seen, epochs_since_improve = sel_value, 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= args.early_stop_patience:
                    print(f"  [early-stop] {args.select_by} has not improved for "
                          f"{epochs_since_improve} epochs (best={best_sel_seen:.4f}); "
                          f"stopping at epoch {epoch}")
                    break

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
    # best_sel is in --select-by's units (val_ap by default, retrieval_clip under
    # Stage A) - NOT always literally val_ap, despite the historical variable name
    # this replaces. Mislabeling a retrieval score as "val_ap" downstream is exactly
    # the class of bug already fixed once in semsup_b1_probe.py - avoided here by
    # naming the JSON field after args.select_by instead of hardcoding "val_ap".
    best_sel, best_epoch = topk[0]
    print(f"\n[done] top-{args.keep_top_k} by {args.select_by}: " +
          ", ".join(f"ep{e} ({args.select_by}={sv:.4f})" for sv, e in topk))
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"stage": stage,
                    "selection_metric": args.select_by,
                    "best_selection_value": (None if best_sel != best_sel else round(best_sel, 4)),
                    # Kept for backward compatibility with every prior run's schema -
                    # only meaningful when select_by == 'val_ap' (the default).
                    "best_val_ap": (None if (args.select_by != "val_ap" or best_sel != best_sel)
                                     else round(best_sel, 4)),
                    "best_epoch": best_epoch,
                    "keep_top_k": args.keep_top_k,
                    "top_checkpoints": [{"epoch": e, args.select_by: (None if sv != sv else round(sv, 4))}
                                         for sv, e in topk],
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
    # Precomputed once, reused across every checkpoint scored below (records
    # themselves don't change between checkpoints) - also lets prefetch_clips'
    # default key="frame_paths" work unchanged.
    records_wp = [{**r, "frame_paths": frame_paths_for(r, args.test_frames_root, pattern)}
                   for r in records]

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
        # Prefetch pipeline (see TrainableBadasWrapper.prefetch_clips()) - the
        # same I/O-bound bottleneck applies here: ~5.7 min/checkpoint over 677
        # clips was measured almost entirely on file read + decode, not GPU.
        with open(res_path, "w", encoding="utf-8") as f, torch.no_grad():
            for _, ex, clip, err in badas.prefetch_clips(records_wp, num_workers=8, prefetch=16):
                if err is not None:
                    n_failed += 1
                    print(f"  [warn] skipping test clip {ex.get('video_id')}: {err}")
                    continue
                logits, _ = badas.forward_clip(clip.to(device))
                s = float(torch.softmax(logits, dim=1)[0, 1].item())
                gt, g = int(ex[gt_field]), ex.get("group")
                f.write(json.dumps({"video_id": ex["video_id"], "ground_truth": gt,
                                     "group": g, "score": round(s, 4)}) + "\n")
                f.flush()
                yt.append(gt); ys.append(s); grp.append(g)
        if n_failed:
            print(f"  [warn] {n_failed}/{len(records)} test clips failed to score")
        return yt, ys, grp

    summary = []
    for rank, (sv, epoch) in enumerate(topk, 1):
        print(f"\n[test] scoring top-{rank} checkpoint (epoch {epoch}, "
              f"{args.select_by}={sv:.4f}) on {len(records)} clips ...")
        res_path = out_dir / f"test_results_ep{epoch:02d}.jsonl"
        yt, ys, grp = score_checkpoint(epoch, res_path)
        m = metrics_from_arrays(yt, ys, groups=grp, threshold=0.5)
        with open(out_dir / f"metrics_ep{epoch:02d}.json", "w", encoding="utf-8") as f:
            json.dump({"stage": stage, "epoch": epoch, "rank": rank,
                       "selection_metric": args.select_by,
                       "selection_value": (None if sv != sv else round(sv, 4)), **m}, f, indent=2)
        per = m.get("per_tte_ap", {})
        print(f"       test_AP={m['ap']}  AUC={m['auc_roc']}  F1={m['f1']} "
              f"(F1*={m['f1_optimal']}@{m['optimal_threshold']})  "
              f"recall={m['recall_sensitivity_tpr']}  spec={m['specificity_tnr']}  "
              f"acc={m['accuracy']}  Brier={m['brier']}  ECE={m['ece']}")
        print(f"       per-TTE AP: " +
              "  ".join(f"{k}={v['ap']}(n={v['n']})" for k, v in per.items()))
        summary.append({"rank": rank, "epoch": epoch,
                        "selection_metric": args.select_by,
                        "selection_value": (None if sv != sv else round(sv, 4)),
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
