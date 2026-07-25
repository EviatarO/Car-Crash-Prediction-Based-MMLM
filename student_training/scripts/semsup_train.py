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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora-target-modules", required=True,
                     help="comma-separated substrings, e.g. 'qkv,proj,fc1,fc2'")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--semantic-weight", type=float, default=0.0)
    ap.add_argument("--siglip-model", default="google/siglip-base-patch16-224")
    ap.add_argument("--predictor-init", default=None, help="warm-start from B1 checkpoint")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
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
    print(f"[cfg] stage={stage}  semantic_weight={args.semantic_weight}  "
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

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    examples = load_training_examples(limit=args.limit)
    if len(examples) < args.min_examples:
        raise RuntimeError(
            f"Only {len(examples)} training examples loaded (< --min-examples "
            f"{args.min_examples}). This usually means dataset/train is missing or "
            f"partially synced - check the symlink/volume before training on a "
            f"silently-shrunk dataset."
        )
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac, seed=args.seed)
    print(f"[data] train={len(train_ex)}  val={len(val_ex)} (clip-level split)")

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
        elapsed = time.time() - t0
        avg_crash = total_crash / n if n else float("nan")
        avg_sem = total_sem / n if n else float("nan")
        print(f"  epoch {epoch}/{args.epochs}  crash_loss={avg_crash:.4f}  "
              f"sem_loss={avg_sem:.4f}  val_ap={val_ap:.4f}  ({elapsed:.1f}s)")

        with open(out_dir / "epoch_metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch, "crash_loss": avg_crash, "sem_loss": avg_sem,
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
    # epoch wins. Keep the top-3 checkpoints, prune the rest (mirrors B1).
    ranked = sorted(saved, key=lambda r: (r[0] if r[0] == r[0] else float("-inf"), r[1]),
                    reverse=True)
    top3 = ranked[:3]
    keep = {e for _, e in top3}
    for _, e in saved:
        if e not in keep:
            shutil.rmtree(out_dir / f"epoch_{e:02d}", ignore_errors=True)
    best_ap, best_epoch = top3[0]
    print(f"\n[done] top-3 by val_ap: " +
          ", ".join(f"ep{e} (val_ap={va:.4f})" for va, e in top3))
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"stage": stage,
                    "best_val_ap": (None if best_ap != best_ap else round(best_ap, 4)),
                    "best_epoch": best_epoch,
                    "top3": [{"epoch": e, "val_ap": (None if va != va else round(va, 4))}
                             for va, e in top3],
                    "n_train": len(train_ex), "n_val": len(val_ex),
                    "semantic_weight": args.semantic_weight,
                    # Full run config, so "was B run identical to A1 except for
                    # semantic_weight" is a fact checkable from disk, not an
                    # assertion in a markdown file.
                    "args": vars(args)}, f, indent=2)

    if not (args.test_manifest and args.test_frames_root):
        return

    # ---- Test-score EACH of the top-3 checkpoints on the real test set. ----
    # At n~267, which epoch "wins" on val is noisy, so scoring all 3 shows the
    # spread rather than betting the headline on one checkpoint.
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
    for rank, (va, epoch) in enumerate(top3, 1):
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
