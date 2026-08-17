"""P1 Stage A gate: score a Stage-A (semantic-only) checkpoint's encoder against the
UNCHANGED frozen crash head, on the real 677-clip test set - no training.

Stage A never sees the crash loss, so its encoder may have drifted away from what the
frozen head (fitted to BADAS-Open's ORIGINAL features) expects. This is the cheap check
before spending a full Stage-B run: if crash performance already collapsed here, Stage B
is starting from a damaged representation and that is worth knowing before, not after,
~8 more epochs of GPU time.

Mirrors semsup_train.py's score_checkpoint() closely (same LoRA-load pattern, same
prefetch/scoring loop), but standalone - semsup_train.py's version only runs at the end
of a live training loop, tied to checkpoints THAT run just produced.

Usage (on the pod):
  HF_HOME=/root/.cache/huggingface python3 p1_stageA_gate.py \
      --config ../configs/e4_stageA.yaml \
      --lora-adapter /workspace/semsup/p1_stageA/epoch_10/lora_adapter \
      --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
      --test-frames-root ../../dataset/test \
      --out /workspace/semsup/p1_stageA_gate_ep10.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import TrainableBadasWrapper  # noqa: E402
from e4_stageA_badas_open_eval import load_manifest, frame_paths_for  # noqa: E402
from metrics_core import metrics_from_arrays  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora-adapter", required=True,
                     help="path to a Stage-A epoch's lora_adapter/ directory")
    ap.add_argument("--test-manifest", required=True)
    ap.add_argument("--test-frames-root", required=True)
    ap.add_argument("--test-limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[setup] LoRA topology: query,key,value (matches Stage A's recipe)")
    badas = TrainableBadasWrapper(cfg, lora_target_modules=["query", "key", "value"],
                                   lora_r=16, lora_alpha=32, lora_dropout=0.05)

    from safetensors.torch import load_file
    from peft.utils import set_peft_model_state_dict
    adapter_path = Path(args.lora_adapter)
    sft = adapter_path / "adapter_model.safetensors" if adapter_path.is_dir() else adapter_path
    set_peft_model_state_dict(badas.nn_model, load_file(str(sft)))
    print(f"[load] Stage-A LoRA weights from {sft}")
    badas.nn_model.eval()

    records = load_manifest(args.test_manifest)
    if args.test_limit:
        records = records[: args.test_limit]
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]
    records_wp = [{**r, "frame_paths": frame_paths_for(r, args.test_frames_root, pattern)}
                  for r in records]

    yt, ys, grp = [], [], []
    n_failed = 0
    with torch.no_grad():
        for _, ex, clip, err in badas.prefetch_clips(records_wp, num_workers=8, prefetch=16):
            if err is not None:
                n_failed += 1
                print(f"  [warn] skipping test clip {ex.get('video_id')}: {err}")
                continue
            logits, _ = badas.forward_clip(clip.to(device))
            s = float(torch.softmax(logits, dim=1)[0, 1].item())
            yt.append(int(ex[gt_field])); ys.append(s); grp.append(ex.get("group"))
    if n_failed:
        print(f"  [warn] {n_failed}/{len(records)} test clips failed to score")

    m = metrics_from_arrays(yt, ys, groups=grp, threshold=0.5)
    per = m.get("per_tte_ap", {})
    print(f"\n[gate] Stage-A encoder + FROZEN crash head, n_test={len(yt)}")
    print(f"       test_AP={m['ap']}  AUC={m['auc_roc']}  F1={m['f1']} "
          f"(F1*={m['f1_optimal']}@{m['optimal_threshold']})")
    print(f"       per-TTE AP: " + "  ".join(f"{k}={v['ap']}(n={v['n']})" for k, v in per.items()))
    print(f"\n[reference] A0 frozen baseline: test_AP=0.853  AUC=0.864")
    print(f"[reference] A1_1761 control:     test_AP=0.900  AUC=0.904")

    out = {"lora_adapter": str(args.lora_adapter), "n_test": len(yt),
           "n_failed": n_failed, **m}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[wrote] {args.out}")


if __name__ == "__main__":
    main()
