"""Score one checkpoint (or the frozen A0 baseline) on all 1,761 training-pool windows.

Per-window scores were never saved for the semantic arms - every one of them was only ever
run through score_checkpoint() against the 677-clip TEST manifest, never against its own
training pool. This fills that gap so a per-clip A0-vs-arm comparison can be built (see
build_pool1761_comparison.py).

Modeled on p1_stageA_gate.py (same LoRA-load pattern, same prefetch/scoring loop), but scores
the TRAIN POOL via load_training_examples() instead of the test manifest, and --lora-adapter
is optional - omit it to score the frozen A0 baseline with no adapter attached at all.

Usage (on the pod, once per arm):
  # A0 - frozen baseline, no adapter
  python3 score_arms_on_pool1761.py --config ../configs/e4_stageA.yaml \
      --captions-path ../../outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl \
      --arm-name A0 --out /workspace/semsup/pool1761_scores/A0.jsonl

  # A1 / B-v1 / B-v2 / B-v3 / P1 - with the arm's own checkpoint
  python3 score_arms_on_pool1761.py --config ../configs/e4_stageA.yaml \
      --captions-path ../../outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl \
      --lora-adapter /workspace/semsup/a1_1761/epoch_04/lora_adapter \
      --arm-name A1 --out /workspace/semsup/pool1761_scores/A1.jsonl
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
    ap.add_argument("--captions-path", required=True,
                     help="Caption_V12_Neutral_1761_fortrain.jsonl - fixes the exact 1,761"
                          " window set scored (frames_dir resolution comes from this file)")
    ap.add_argument("--arm-name", required=True, help="written into every output row")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] LoRA topology: query,key,value")
    badas = TrainableBadasWrapper(cfg, lora_target_modules=["query", "key", "value"],
                                   lora_r=16, lora_alpha=32, lora_dropout=0.05)

    if args.lora_adapter:
        from safetensors.torch import load_file
        from peft.utils import set_peft_model_state_dict
        adapter_path = Path(args.lora_adapter)
        sft = adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
        set_peft_model_state_dict(badas.nn_model, load_file(str(sft)))
        print(f"[load] {args.arm_name}: LoRA weights from {sft}")
    else:
        print(f"[load] {args.arm_name}: frozen A0 baseline, no adapter attached")
    badas.nn_model.eval()

    examples = load_training_examples(captions_path=args.captions_path)
    if len(examples) != 1761:
        print(f"[WARN] expected 1761 windows, got {len(examples)} - "
              f"check --captions-path resolves the full pool")

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
