"""
aa1_run_set.py
==============
Run the AA.1 v2b detection pipeline (Stage 0 YOLOPv2 cache -> Stage 1 tracks -> Stage 3
selection, on top of Stage 2's load_frame_masks) on a clip set other than the 18 val_e3a clips
it was developed on.

`--set gen18` is the generalization check: 9 positive + 9 negative clips drawn at random
(fixed seed) from the training pool - dataset/train.csv minus the 18 val_e3a development clips
and minus every test-manifest clip. Nothing in the pipeline was tuned on these clips. The
sampled list is written once to <out_dir>/clip_list.json and reused on later runs, so the set
never changes silently.

2026-09-19: gen18's output folder was merged into outputs/aa1_v2_18clips (both sets now go
through the identical pipeline and are reviewed together; the per-clip filenames don't collide
between the two sets' video ids). clip_list.json still lives there, keyed by that folder.

Usage:
  python aa1_run_set.py --set gen18 --stages 0,1      # detection, tracks (no stage "2" driver
                                                       # any more - see aa1_stage2.py)
  python aa1_run_set.py --set gen18 --stages 3        # selection score (after GT labelling)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aa1_detect_track_rank import RAW_VIDEO_ROOT, TRAIN_CSV  # noqa: E402
from aa1_track_stage1 import VAL_E3A_IDS  # noqa: E402

SETS = {"gen18": dict(out_dir=REPO / "outputs" / "aa1_v2_18clips", n_pos=9, n_neg=9, seed=20260919)}
TEST_MANIFESTS = sorted((REPO / "dataset" / "manifests").glob("test_*.jsonl"))


def _test_ids() -> set:
    ids = set()
    for path in TEST_MANIFESTS:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ids.add(str(json.loads(line)["video_id"]).zfill(5))
    return ids


def sample_clips(n_pos: int, n_neg: int, seed: int) -> dict:
    excluded = set(VAL_E3A_IDS) | _test_ids()
    pos, neg = [], []
    with open(TRAIN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = str(row["id"]).zfill(5)
            if vid in excluded or not (RAW_VIDEO_ROOT / f"{vid}.mp4").exists():
                continue
            if int(row["target"]) == 1 and row["time_of_event"]:
                pos.append(vid)
            elif int(row["target"]) == 0:
                neg.append(vid)
    rng = random.Random(seed)
    return dict(seed=seed, n_pool_pos=len(pos), n_pool_neg=len(neg), n_excluded=len(excluded),
                positives=sorted(rng.sample(sorted(pos), n_pos)),
                negatives=sorted(rng.sample(sorted(neg), n_neg)))


def load_or_create_list(cfg: dict) -> list:
    out_dir = cfg["out_dir"]
    path = out_dir / "clip_list.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = sample_clips(cfg["n_pos"], cfg["n_neg"], cfg["seed"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    print(f"[set] {len(rec['positives'])} pos {rec['positives']}")
    print(f"[set] {len(rec['negatives'])} neg {rec['negatives']}")
    return rec["positives"] + rec["negatives"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--stages", default="0,1,2,3")
    args = ap.parse_args()
    cfg = SETS[args.set]
    out_dir = cfg["out_dir"]
    clips = load_or_create_list(cfg)
    stages = {int(s) for s in args.stages.split(",")}

    if 0 in stages:
        from aa1_scene import YOLOPv2
        import aa1_yolop_cache
        yp = YOLOPv2()
        for vid in clips:
            aa1_yolop_cache.run_clip(yp, vid, out_dir)
    if 1 in stages:
        import aa1_track_stage1
        for vid in clips:
            aa1_track_stage1.run_clip(vid, out_dir)
    if 2 in stages or 3 in stages:
        # aa1_stage2 no longer has its own driver (2026-09-19: it was just load_frame_masks +
        # a now-removed geometry_v2.json/overlay that nothing read) - Stage 3 calls
        # load_frame_masks directly, so "stage 2" only needs the module's OUT_DIR pointed here
        import aa1_stage2
        aa1_stage2.OUT_DIR = out_dir  # its loaders read the module-level folder
    if 3 in stages:
        import aa1_stage3
        for vid in clips:
            aa1_stage3.run_clip(vid, out_dir)


if __name__ == "__main__":
    main()
