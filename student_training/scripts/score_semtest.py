"""
score_semtest.py
=================
Score one SemTest-200 checkpoint (or the frozen A0 baseline) on the SemTest-200 pool.

Adapted from score_arms_on_pool1761.py for the 2026-08-26 SemTest-200 plan: same
LoRA-load pattern, but (1) optionally loads an --unfreeze-head checkpoint's
head_state.pt (temporal_processor + classifier weights - unusable without it, since
semsup_train.py --unfreeze-head trains those OUTSIDE the LoRA adapter peft saves),
and (2) does not assume/warn about a fixed pool size (score_arms_on_pool1761.py hard-
codes 1,761; SemTest-200 is 200).

Usage:
  # A0 - frozen baseline, no adapter, no head state
  python3 score_semtest.py --config ../configs/e4_stageA.yaml \
      --captions-path ../../outputs/semtest200/selection_with_captions.jsonl \
      --arm-name A0 --out /workspace/semtest200/scores/A0.jsonl

  # A trained arm (vision-only or semantic), --unfreeze-head was used at train time
  python3 score_semtest.py --config ../configs/e4_stageA.yaml \
      --captions-path ../../outputs/semtest200/selection_with_captions.jsonl \
      --lora-adapter /workspace/semtest200/vision/epoch_05/lora_adapter \
      --head-state /workspace/semtest200/vision/epoch_05/head_state.pt \
      --arm-name sem200_vision --out /workspace/semtest200/scores/sem200_vision.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import TrainableBadasWrapper, load_training_examples  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora-adapter", default=None,
                     help="epoch's lora_adapter/ dir. Omit to score the frozen A0 baseline.")
    ap.add_argument("--head-state", default=None,
                     help="epoch's head_state.pt (only present when the run used "
                          "semsup_train.py --unfreeze-head). A checkpoint trained with "
                          "--unfreeze-head is NOT reproducible without this - peft's "
                          "save_pretrained() persists only the LoRA delta.")
    ap.add_argument("--captions-path", required=True,
                     help="SemTest-200 caption file with explicit frames_dir per row "
                          "(load_training_examples uses it as-is).")
    ap.add_argument("--arm-name", required=True, help="written into every output row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] LoRA topology: query,key,value")
    badas = TrainableBadasWrapper(cfg, lora_target_modules=["query", "key", "value"],
                                   lora_r=16, lora_alpha=32, lora_dropout=0.05,
                                   unfreeze_module_substrings=(
                                       ["temporal_processor", "classifier"]
                                       if args.head_state else None))

    if args.lora_adapter:
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        adapter_path = Path(args.lora_adapter)
        sft = adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
        set_peft_model_state_dict(badas.nn_model, load_file(str(sft)))
        print(f"[load] {args.arm_name}: LoRA weights from {sft}")
    else:
        print(f"[load] {args.arm_name}: frozen baseline, no LoRA adapter attached")

    if args.head_state:
        badas.load_head_state(args.head_state)

    badas.nn_model.eval()

    examples = load_training_examples(captions_path=args.captions_path)
    print(f"[data] scoring {len(examples)} windows")

    n_failed = 0
    rows = []
    with torch.no_grad():
        for _, ex, clip, err in badas.prefetch_clips(examples, num_workers=8, prefetch=16):
            if err is not None:
                n_failed += 1
                print(f"  [warn] skipping {ex.get('video_id')}/{ex.get('frames_dir')}: {err}")
                continue
            logits, _ = badas.forward_clip(clip.to(device))
            score = float(torch.softmax(logits, dim=1)[0, 1].item())
            rows.append({
                "arm": args.arm_name,
                "video_id": ex["video_id"],
                "frames_dir": ex["frames_dir"],
                "requested_time_to_event": ex["tte"],
                "gt_verdict": "YES" if ex["label"] == 1 else "NO",
                "score": score,
            })
    if n_failed:
        print(f"  [warn] {n_failed}/{len(examples)} windows failed to score")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[wrote] {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
