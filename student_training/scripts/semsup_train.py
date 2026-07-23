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


def evaluate_crash_ap(badas, examples, device):
    badas.nn_model.eval()
    ys, yt = [], []
    with torch.no_grad():
        for ex in examples:
            logits, _ = badas.forward(ex["frame_paths"])
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
            ys.append(score)
            yt.append(ex["label"])
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
    args = ap.parse_args()

    import yaml
    with open(args.config, encoding="utf-8") as f:
        stagea_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = "B (crash+semantic)" if args.semantic_weight > 0 else "A1 (crash-only)"
    print(f"[cfg] stage={stage}  semantic_weight={args.semantic_weight}  "
          f"lora_target_modules={args.lora_target_modules}")

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
        predictor = ResamplerProjector(in_dim=1024, out_dim=dt, num_queries=1,
                                        hidden_dim=512, n_heads=8).to(device)
        if args.predictor_init:
            predictor.load_state_dict(torch.load(args.predictor_init, map_location=device))
            print(f"[load] warm-started predictor from {args.predictor_init}")
        trainable += list(predictor.parameters())

    opt = torch.optim.AdamW(trainable, lr=args.lr)

    examples = load_training_examples(limit=args.limit)
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac)
    print(f"[data] train={len(train_ex)}  val={len(val_ex)} (clip-level split)")

    best_ap, best_epoch = -1.0, -1
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        badas.nn_model.train()
        if predictor is not None:
            predictor.train()
        random.shuffle(train_ex)
        opt.zero_grad()
        total_crash, total_sem, n = 0.0, 0.0, 0
        for step, ex in enumerate(train_ex):
            logits, patches = badas.forward(ex["frame_paths"])
            label = torch.tensor([ex["label"]], device=device)
            crash_loss = F.cross_entropy(logits, label)

            sem_loss = torch.tensor(0.0, device=device)
            if predictor is not None:
                # BADAS may run in fp16; the Predictor is fp32. .to(dtype=) is a
                # differentiable cast (autograd supports it) so the semantic-loss
                # gradient still flows back into the LoRA-unfrozen trunk.
                patches32 = patches.unsqueeze(0).to(dtype=torch.float32)
                pred = predictor(patches32).squeeze(1)
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
            opt.step()
            opt.zero_grad()

        val_ap = evaluate_crash_ap(badas, val_ex, device)
        print(f"  epoch {epoch}/{args.epochs}  crash_loss={total_crash/n:.4f}  "
              f"sem_loss={total_sem/n:.4f}  val_ap={val_ap:.4f}  ({time.time()-t0:.1f}s)")

        ep_dir = out_dir / f"epoch_{epoch:02d}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        badas.nn_model.save_pretrained(str(ep_dir / "lora_adapter"))
        if predictor is not None:
            torch.save(predictor.state_dict(), ep_dir / "predictor.pt")
        if val_ap == val_ap and val_ap > best_ap:  # NaN-safe
            best_ap, best_epoch = val_ap, epoch

    print(f"\n[done] best held-out val_ap={best_ap:.4f} @ epoch {best_epoch}")
    with open(out_dir / "train_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"stage": stage, "best_val_ap": best_ap, "best_epoch": best_epoch,
                    "n_train": len(train_ex), "n_val": len(val_ex),
                    "semantic_weight": args.semantic_weight}, f, indent=2)

    # Restore the BEST epoch's weights before test scoring. Without this,
    # badas.nn_model still holds whatever the LAST epoch left it at, which is
    # not necessarily the best one (val_ap can peak mid-training - this is
    # exactly what happened in B1's real run, best at epoch 8 of 23).
    if best_epoch == -1:
        print("[warn] no epoch had a valid val_ap (val split may be single-class) "
              "- test-scoring with the LAST epoch's weights, not a selected best")
    else:
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        best_dir = out_dir / f"epoch_{best_epoch:02d}"
        adapter_sd = load_file(str(best_dir / "lora_adapter" / "adapter_model.safetensors"))
        set_peft_model_state_dict(badas.nn_model, adapter_sd)
        if predictor is not None:
            predictor.load_state_dict(torch.load(best_dir / "predictor.pt", map_location=device))
        print(f"[reload] restored epoch {best_epoch} (val_ap={best_ap:.4f}) before test scoring")

    if args.test_manifest and args.test_frames_root:
        print(f"\n[test] scoring 677-Private test set with epoch {best_epoch} checkpoint ...")
        badas.nn_model.eval()
        records = load_manifest(args.test_manifest)
        if args.test_limit:
            records = records[: args.test_limit]
        pattern = stagea_cfg["data"]["frame_filename_pattern"]
        gt_field = stagea_cfg["data"]["gt_field"]
        out_test = out_dir / "test_results.jsonl"
        ys, yt = [], []
        with open(out_test, "w", encoding="utf-8") as f:
            for r in records:
                paths = frame_paths_for(r, args.test_frames_root, pattern)
                with torch.no_grad():
                    logits, _ = badas.forward(paths)
                    score = float(torch.softmax(logits, dim=1)[0, 1].item())
                f.write(json.dumps({
                    "video_id": r["video_id"], "ground_truth": int(r[gt_field]),
                    "group": r.get("group"), "score": round(score, 4),
                }) + "\n")
                ys.append(score)
                yt.append(int(r[gt_field]))
        test_ap = average_precision_score(yt, ys)
        print(f"[test] n={len(records)}  test_AP={test_ap:.4f}  -> {out_test}")
        print("       (run evaluate_metrics.py on this file for the full report)")


if __name__ == "__main__":
    main()
