"""
e4_stageB_cache_features.py
===========================
Stage B, step 1: cache the FROZEN V-JEPA2 patch grid (+ P(collision)) for every
window once, so projector training reads tensors off disk (the ViT-L never runs
during training). See plan §2 step 1 + §2 step 4.

Per window it writes `<cache_dir>/<frames_dir>.pt` (patch grid, fp16) and appends
one line to `<cache_dir>/cache_manifest_<split>.jsonl`:
  {key, video_id, frames_dir, horizon_label, target, score, split, assistant_target}

`score` is the frozen BADAS P(collision) — the free byproduct that feeds the
§5(5) per-TTE characterization.

Run once per source (like Stage A's --split):
  TRAIN: --manifest dataset/teacher_labels/teacher_dataset_e3b.jsonl --split train
  VAL  : --manifest dataset/manifests/val_e3a.jsonl                  --split val
both with --frames_root /workspace/data/train_HiRes  (RunPod GPU).

Local validation (no GPU/model):
  ... --split train --dry_run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def frame_paths_for(record, frames_root, pattern):
    frames_dir = record.get("frames_dir") or record["video_id"]
    return [os.path.join(frames_root, frames_dir, pattern.format(i))
            for i in record["frame_indices"]]


def horizon_for(record):
    """Clean bucket label. e3b carries horizon_label; val carries TTE float."""
    if record.get("horizon_label"):
        return record["horizon_label"]
    tte = record.get("requested_time_to_event")
    return f"VAL_{tte}" if tte is not None else "VAL"


def dry_run(records, frames_root, pattern, split, n_check=6):
    print(f"\n=== DRY RUN ({split}) — no model, no features ===")
    from collections import Counter
    print(f"  records         : {len(records)}")
    print(f"  target dist     : {dict(Counter(int(r.get('target', 0)) for r in records))}")
    print(f"  horizon dist    : {dict(Counter(horizon_for(r) for r in records))}")
    missing, sizes = 0, set()
    from PIL import Image
    for r in records[:n_check]:
        for p in frame_paths_for(r, frames_root, pattern):
            if not os.path.exists(p):
                missing += 1
            else:
                sizes.add(Image.open(p).size)
    print(f"  checked {n_check} clips: missing frames = {missing}")
    print(f"  sample frame sizes: {sizes}")
    print("=== dry run OK ===\n")


def run(args, cfg):
    records = load_jsonl(args.manifest if os.path.isabs(args.manifest)
                         else PROJECT_ROOT / args.manifest)
    frames_root = args.frames_root
    pattern = cfg["data"]["frame_filename_pattern"]

    if args.limit:
        records = records[:args.limit]

    if args.dry_run:
        dry_run(records, frames_root, pattern, args.split)
        return

    import torch  # noqa
    from student_training.models.vjepa_reason import VJEPA2FeatureExtractor

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    man_path = cache_dir / f"cache_manifest_{args.split}.jsonl"

    extractor = VJEPA2FeatureExtractor(cfg, temperature=cfg["preprocess"].get("temperature", 2.0))

    print(f"Caching {len(records)} {args.split} windows -> {cache_dir}")
    shapes = set()
    with open(man_path, "w", encoding="utf-8") as mf:
        for k, r in enumerate(records, 1):
            key = r.get("frames_dir") or r["video_id"]
            paths = frame_paths_for(r, frames_root, pattern)
            feats, score = extractor.extract(paths)            # (P, D) fp16, float
            if torch.isnan(feats).any():
                raise RuntimeError(f"NaN features for {key}")
            torch.save(feats, cache_dir / f"{key}.pt")
            shapes.add(tuple(feats.shape))
            mf.write(json.dumps({
                "key":              key,
                "video_id":         r["video_id"],
                "frames_dir":       key,
                "horizon_label":    horizon_for(r),
                "target":           int(r.get("target", 0)),
                "score":            round(score, 6),
                "split":            args.split,
                "assistant_target": r.get("assistant_target", ""),
            }) + "\n")
            mf.flush()
            if k % 25 == 0 or k == len(records):
                print(f"  {k}/{len(records)}  last_score={score:.4f}  shapes={shapes}")
    print(f"Done. Patch-grid shapes seen: {shapes}")
    print(f"Manifest: {man_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="e4_stageB.yaml (carries model+preprocess)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--cache_dir", default=os.environ.get(
        "E4_CACHE_DIR", "outputs/e4_vjepa_reason/e4_StageB_bridge/cached_features"),
        help="Feature cache dir. On RunPod set E4_CACHE_DIR=/root/... (off the 20GB /workspace volume).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(args.config if os.path.isabs(args.config) else PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    run(args, cfg)


if __name__ == "__main__":
    main()
