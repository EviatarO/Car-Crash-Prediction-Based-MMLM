"""
e4_stageA_badas_open_eval.py
============================
Stage A of `e4_vjepa_reason`: reproduce BADAS-Open's collision-prediction AP on
our Nexar test halves (Private 677 + Public 667), to anchor the frozen scorer
used by later stages.

WHAT IT DOES (per split)
  1. Loads nexar-ai/BADAS-Open via its OFFICIAL loader (badas_loader.py from HF).
     The loader returns a VJEPAModel wrapper (NOT an nn.Module). The actual
     nn.Module is vjepa.model; the HF AutoVideoProcessor is vjepa.processor.
  2. For each clip: loads the 16 last-window frames as numpy RGB arrays,
     preprocesses via vjepa.processor (224x224, ImageNet norm), runs the
     nn.Module -> logits -> temperature scaling T=2.0 -> softmax -> P(collision).
  3. Writes a per-clip JSONL with the Stage-A schema (see OUTPUT SCHEMA).

CONFIRMED from badas_loader.py + badas/core/preprocessing.py (2026-06-24):
  - img_size = 224  (badas_loader.py: VJEPAModel(img_size=224))
  - resize  = squash (cv2.resize to (size,size), no crop)
  - norm    = ImageNet mean/std
  - temperature = 2.0 before softmax (apply_temperature_scaling in vjepa.py)
  - score   = softmax(logits/2.0)[1]  (positive class probability)

WINDOWING: manifests encode the last-16-frame (~2 s) window at stride 4 (7.5fps).
BADAS targets 8fps; 7.5fps is a benign second-order difference.

OUTPUT SCHEMA (one JSON object per clip)
  {video_id, ground_truth, group, time_before_s, score, collision_verdict, split}
  - ground_truth   = manifest `event_occurs` (1 collision / 0 none)
  - score          = BADAS P(collision) in [0,1]
  - collision_verdict = "YES" if score >= threshold else "NO"   (threshold cfg, default 0.5)
  - split          = "Private" | "Public"  (from --split)

RUN TARGET: RunPod GPU (needs credit). DO NOT run the scoring step locally.
  Local validation without a GPU:
    python student_training/scripts/e4_stageA_badas_open_eval.py \
        --config student_training/configs/e4_stageA.yaml \
        --manifest dataset/manifests/test_manifest_hires.jsonl \
        --frames_root dataset/test --split Private --dry_run
  Full run on RunPod: drop --dry_run (see RUNPOD_E4_STAGEA.txt).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Config + manifest
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_manifest(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def frame_paths_for(record: dict, frames_root: str, pattern: str) -> list:
    """Absolute paths of the clip's frames. `frames_dir` falls back to video_id."""
    frames_dir = record.get("frames_dir") or record["video_id"]
    return [os.path.join(frames_root, frames_dir, pattern.format(i))
            for i in record["frame_indices"]]


# =============================================================================
# Dry-run validation (no GPU, no model)
# =============================================================================

def dry_run(cfg, records, frames_root, split, n_check=5):
    print(f"\n=== DRY RUN ({split}) — no model, no scoring ===")
    pp = cfg["preprocess"]
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]

    gts = [int(r[gt_field]) for r in records]
    from collections import Counter
    print(f"  records          : {len(records)}")
    print(f"  positives/neg    : {sum(gts)} / {len(gts) - sum(gts)}")
    print(f"  group dist       : {dict(Counter(r.get('group') for r in records))}")

    missing, sizes = 0, []
    for r in records[:n_check]:
        paths = frame_paths_for(r, frames_root, pattern)
        if len(paths) != pp["num_frames"]:
            print(f"  WARN {r['video_id']}: {len(paths)} frames != {pp['num_frames']}")
        for p in paths:
            if not os.path.exists(p):
                missing += 1
            else:
                sizes.append(Image.open(p).size)
    print(f"  checked {n_check} clips: missing frames = {missing}")
    print(f"  sample frame sizes: {set(sizes)}  (expect 1280x720 hi-res)")
    if any(s == (256, 256) for s in sizes):
        print("  NOTE: 256x256 source detected -> pre-squashed frames; prefer *_hires "
              "so BADAS owns the resize (plan §3.2).")
    print(f"  preprocess.source = {pp['source']} "
          "(verify img_size/normalization against the badas package on RunPod)")
    print("=== dry run OK (no scores computed) ===\n")


# =============================================================================
# Scoring (RunPod GPU only)
# =============================================================================

def load_badas(cfg):
    """Load BADAS-Open. Returns (vjepa_wrapper, nn_module, device).

    VJEPAModel (wrapper) is NOT an nn.Module — do not call .eval()/.to() on it.
    vjepa.model IS the nn.Module; vjepa.processor is the HF AutoVideoProcessor.
    """
    import torch
    from huggingface_hub import hf_hub_download

    repo = cfg["model"]["hf_repo"]
    print(f"Loading {repo} via official loader ...")
    loader_path = hf_hub_download(repo_id=repo, filename="badas_loader.py")
    sys.path.insert(0, os.path.dirname(loader_path))
    from badas_loader import load_badas_model  # noqa: E402
    vjepa = load_badas_model()          # VJEPAModel wrapper
    nn_model = vjepa.model              # actual nn.Module (ViT-L + pooler + classifier)
    nn_model.eval()
    device = str(vjepa.device)          # already on GPU from load_badas_model
    print(f"  device: {device}  processor: {type(vjepa.processor).__name__}")
    return vjepa, nn_model, device


def preprocess_clip(vjepa, paths):
    """Preprocess 16 JPEG frames -> (1, T, C, H, W) tensor using BADAS's processor."""
    import numpy as np
    import torch

    frames_np = [np.array(Image.open(p).convert("RGB")) for p in paths]

    if vjepa.processor is not None:
        # Official HF AutoVideoProcessor (handles resize + norm correctly)
        inputs = vjepa.processor(videos=frames_np, return_tensors="pt")
        key = "pixel_values_videos" if "pixel_values_videos" in inputs else next(iter(inputs))
        clip = inputs[key]
        if clip.dim() == 4:             # (T, C, H, W) -> (1, T, C, H, W)
            clip = clip.unsqueeze(0)
    else:
        # Fallback: albumentations transform from vjepa.transform
        frames_t = torch.stack([vjepa.transform(image=f)["image"] for f in frames_np])
        clip = frames_t.unsqueeze(0)    # (1, T, C, H, W)

    return clip


def score_split(cfg, records, frames_root, split, out_path, limit=0):
    import torch
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]
    threshold = cfg["data"].get("verdict_threshold", 0.5)

    vjepa, nn_model, device = load_badas(cfg)
    if limit:
        records = records[:limit]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"Scoring {len(records)} {split} clips -> {out_path}")
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for k, r in enumerate(records):
            paths = frame_paths_for(r, frames_root, pattern)
            clip = preprocess_clip(vjepa, paths).to(device)  # (1, T, C, H, W)
            with torch.no_grad():
                logits = nn_model(clip)             # (1, 2)
                logits_t = logits / 2.0             # temperature scaling T=2.0 (BADAS default)
                score = float(torch.softmax(logits_t, dim=1)[0, 1].item())
            fh.write(json.dumps({
                "video_id":          r["video_id"],
                "ground_truth":      int(r[gt_field]),
                "group":             r.get("group"),
                "time_before_s":     r.get("time_before_event_s"),
                "score":             round(score, 4),
                "collision_verdict": "YES" if score >= threshold else "NO",
                "split":             split,
            }) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(records)}")

    # Quick AP/AUC sanity (authoritative metrics come from evaluate_metrics.py).
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = [json.loads(l) for l in open(out_path)]
    yt = np.array([r["ground_truth"] for r in rows])
    ys = np.array([r["score"] for r in rows])
    ap, auc = average_precision_score(yt, ys), roc_auc_score(yt, ys)
    exp, tol = cfg["acceptance"]["expected_ap"], cfg["acceptance"]["ap_tolerance"]
    print(f"\n  {split}: n={len(rows)}  AP={ap:.4f}  AUC={auc:.4f}  "
          f"(expected ~{exp}; {'PASS' if abs(ap - exp) <= tol else 'CHECK PREPROCESSING'})")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--split", required=True, choices=["Private", "Public"])
    ap.add_argument("--output", help="Output JSONL path (required unless --dry_run)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate manifest/frames only. No GPU, no scoring.")
    ap.add_argument("--limit", type=int, default=0, help="Score only first N clips (debug).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest_path = args.manifest if os.path.isabs(args.manifest) \
        else os.path.join(PROJECT_ROOT, args.manifest)
    frames_root = args.frames_root if os.path.isabs(args.frames_root) \
        else os.path.join(PROJECT_ROOT, args.frames_root)
    records = load_manifest(manifest_path)

    if args.dry_run:
        dry_run(cfg, records, frames_root, args.split)
        return

    if not args.output:
        ap.error("--output is required unless --dry_run")
    out_path = args.output if os.path.isabs(args.output) \
        else os.path.join(PROJECT_ROOT, args.output)
    score_split(cfg, records, frames_root, args.split, out_path, args.limit)


if __name__ == "__main__":
    main()
