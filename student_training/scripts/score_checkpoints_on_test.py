"""
score_checkpoints_on_test.py
==============================
Score one or more LoRA checkpoints on the 677-clip TEST manifest (Stage 2 of the
A1-failure-recovery run).

WHY A SEPARATE SCRIPT: semsup_train.py test-scores its kept checkpoints only when
--test-manifest is passed AT TRAINING TIME. The a1fail321 runs deliberately omitted it
(the STOP gate was "review loss curves first"), so the checkpoints exist but were never
test-scored. This scores them after the fact without retraining.

SOFTMAX CONVENTION (matters, and is easy to get wrong here): this uses
softmax(logits)[0,1] with NO /2.0 divisor - identical to semsup_train.py and
score_arms_on_pool1761.py. e4_stageA_badas_open_eval.py DOES divide by 2.0 (the
published-scorer convention). For AP/AUC the difference is irrelevant - dividing
logits by a constant is a monotone transform and rank metrics are invariant to it -
but it DOES move acc@0.5 and any other thresholded metric, so those are only
comparable against numbers produced by this same scorer.

Loads BADAS ONCE and swaps adapters between checkpoints (~3 min/checkpoint of actual
scoring vs ~2 min of model load), so scoring N checkpoints costs far less than N runs.

Usage (on the pod):
  python3 score_checkpoints_on_test.py --config ../configs/e4_stageA.yaml \
      --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
      --test-frames-root /workspace/data/test_HiRes \
      --adapters A1=/workspace/semsup/a1_1761/epoch_04/lora_adapter \
                 v12=/workspace/MMLM_AI/outputs/a1fail321/results/v12/fold_01/epoch_10/lora_adapter \
      --out-dir /workspace/MMLM_AI/outputs/a1fail321/test_scores
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import TrainableBadasWrapper  # noqa: E402


def frame_paths_for(record, frames_root, pattern):
    """Absolute paths of the clip's frames. frames_dir falls back to video_id.
    Same logic as e4_stageA_badas_open_eval.py / e4_stageB_cache_features.py."""
    frames_dir = record.get("frames_dir") or record["video_id"]
    return [os.path.join(frames_root, frames_dir, pattern.format(i))
            for i in record["frame_indices"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-frames-root", required=True)
    ap.add_argument("--adapters", nargs="+", required=True,
                     help="one or more NAME=/path/to/lora_adapter. Use NAME=NONE to score "
                          "the frozen no-adapter baseline.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    records = [json.loads(l) for l in open(args.test_manifest, encoding="utf-8") if l.strip()]
    print(f"[load] test manifest: {len(records)} records  gt_field={gt_field!r}")
    pos = sum(1 for r in records if int(r[gt_field]) == 1)
    print(f"[load] class balance: {pos} positive / {len(records) - pos} negative")

    # frame_paths precomputed once and reused across every checkpoint (records do not
    # change between checkpoints); prefetch_clips reads key="frame_paths" by default.
    records_wp = [{**r, "frame_paths": frame_paths_for(r, args.test_frames_root, pattern)}
                  for r in records]
    missing = [r for r in records_wp if not all(os.path.exists(p) for p in r["frame_paths"])]
    if missing:
        raise SystemExit(f"{len(missing)} records have missing frames on disk, "
                          f"e.g. {missing[0]['video_id']} -> {missing[0]['frame_paths'][0]}")
    print(f"[verify] all {len(records_wp)} records have every frame present on disk")

    print("[setup] LoRA topology: query,key,value (r=16, alpha=32, dropout=0.05)")
    badas = TrainableBadasWrapper(cfg, lora_target_modules=["query", "key", "value"],
                                   lora_r=16, lora_alpha=32, lora_dropout=0.05)
    badas.nn_model.eval()

    from safetensors.torch import load_file
    from peft.utils import set_peft_model_state_dict

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.adapters:
        if "=" not in spec:
            raise SystemExit(f"--adapters entry {spec!r} must be NAME=/path (or NAME=NONE)")
        name, path = spec.split("=", 1)

        if path.upper() == "NONE":
            print(f"\n[score] {name}: frozen baseline, no adapter attached")
        else:
            adapter_path = Path(path)
            sft = (adapter_path / "adapter_model.safetensors") if adapter_path.is_dir() else adapter_path
            if not sft.exists():
                raise SystemExit(f"{name}: adapter not found at {sft}")
            # Every adapter here shares the same topology, so this fully overwrites the
            # previous checkpoint's lora_A/lora_B - no residue carries between arms.
            set_peft_model_state_dict(badas.nn_model, load_file(str(sft)))
            print(f"\n[score] {name}: loaded {sft}")

        n_failed, rows = 0, []
        with torch.no_grad():
            for _, rec, clip, err in badas.prefetch_clips(records_wp,
                                                           num_workers=args.num_workers,
                                                           prefetch=16):
                if err is not None:
                    n_failed += 1
                    print(f"  [warn] skipping {rec.get('video_id')}: {err}")
                    continue
                logits, _ = badas.forward_clip(clip.to(device))
                # NO /2.0 - see the convention note in this module's docstring.
                score = float(torch.softmax(logits, dim=1)[0, 1].item())
                rows.append({
                    "arm": name,
                    "video_id": rec["video_id"],
                    "frames_dir": rec.get("frames_dir"),
                    "gt_verdict": "YES" if int(rec[gt_field]) == 1 else "NO",
                    "score": score,
                })
        if n_failed:
            print(f"  [warn] {n_failed}/{len(records_wp)} clips failed to score")

        out_path = out_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        y = [1 if r["gt_verdict"] == "YES" else 0 for r in rows]
        s = [r["score"] for r in rows]
        from sklearn.metrics import average_precision_score, roc_auc_score
        acc = sum(1 for yy, ss in zip(y, s) if (ss >= 0.5) == bool(yy)) / len(y)
        print(f"  {name}: n={len(rows)}  AP={average_precision_score(y, s):.4f}  "
              f"AUC={roc_auc_score(y, s):.4f}  acc@0.5={acc:.4f}")
        print(f"  [wrote] {out_path}")


if __name__ == "__main__":
    main()
