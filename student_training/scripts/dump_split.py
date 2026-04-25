"""
dump_split.py
=============
Reproduce the deterministic 80/20 stratified train/val split used by
train_lora.py and write the resulting video_ids to two text files.

This does NOT load the model — only reads the JSONL and replays the
random.Random(seed) shuffle from stratified_split().

Usage:
  python student_training/scripts/dump_split.py \
    --jsonl outputs/teacher_dataset_v11.jsonl \
    --config student_training/configs/train_lora.yaml \
    --out_dir outputs/checkpoints/e2_lora_100clips
"""
import argparse
import json
import random
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",   required=True, help="teacher_dataset_v11.jsonl")
    ap.add_argument("--config",  default="student_training/configs/train_lora.yaml")
    ap.add_argument("--out_dir", required=True, help="Where to write train_ids.txt / val_ids.txt")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed         = cfg.get("seed", 42)
    val_fraction = cfg.get("val_split", 0.2)

    # Read records in file order (CollisionDataset preserves this)
    records = []
    with open(args.jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Mirror stratified_split() exactly
    rng = random.Random(seed)
    pos_idx = [i for i, r in enumerate(records) if r.get("target") == 1]
    neg_idx = [i for i, r in enumerate(records) if r.get("target") == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, round(len(pos_idx) * val_fraction))
    n_val_neg = max(1, round(len(neg_idx) * val_fraction))

    val_idx   = pos_idx[:n_val_pos]   + neg_idx[:n_val_neg]
    train_idx = pos_idx[n_val_pos:]   + neg_idx[n_val_neg:]
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def vid(rec):
        return rec.get("video_id") or rec.get("clip_id") or rec.get("id") or "<unknown>"

    train_path = out_dir / "train_ids.txt"
    val_path   = out_dir / "val_ids.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        for i in train_idx:
            r = records[i]
            f.write(f"{vid(r)}\t{r.get('target')}\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for i in val_idx:
            r = records[i]
            f.write(f"{vid(r)}\t{r.get('target')}\n")

    n_pos_t = sum(1 for i in train_idx if records[i].get("target") == 1)
    n_pos_v = sum(1 for i in val_idx   if records[i].get("target") == 1)
    print(f"seed={seed}  val_split={val_fraction}")
    print(f"train: {len(train_idx)} clips ({n_pos_t} pos / {len(train_idx)-n_pos_t} neg)  → {train_path}")
    print(f"val  : {len(val_idx)} clips ({n_pos_v} pos / {len(val_idx)-n_pos_v} neg)  → {val_path}")


if __name__ == "__main__":
    main()

