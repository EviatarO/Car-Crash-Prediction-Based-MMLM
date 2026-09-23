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

2026-09-22: added `--set pool1761`, the AA.2 detection pass for the AA-rel/AA-occ token
supervision plan (~/.claude/plans/CCP based BADAS/2026-09-19_Child-Plan-AA-token-relevance-aux.md).
Its video list is every unique video_id in dataset/manifests/recap_v12_1761.jsonl - the exact
1,761-window pool A1-compress256 trained on - not a fresh random sample, so it's a different
"kind" of set (`kind="pool"`): `load_or_create_list` reads the manifest instead of sampling,
and Stage 0 gets a 1s preroll (child plan Phase 1 step 1: compute_track_scores' alpha/EMA need
~1s of causal history, which the earliest window's earliest tubelet didn't have before - see
aa1_detect_track_rank.decode_span's `preroll_s`). Stage 3 (full overlay video + target-curve
plot per clip) is for the ~36-clip review sets only - at ~1,100 videos it would be pure waste;
a pool-kind set's labels come from aa4_token_labels.py instead, which reuses Stage 0/1's cached
output directly (see that script's docstring).

Usage:
  python aa1_run_set.py --set gen18 --stages 0,1      # detection, tracks (no stage "2" driver
                                                       # any more - see aa1_stage2.py)
  python aa1_run_set.py --set gen18 --stages 3        # selection score (after GT labelling)
  python aa1_run_set.py --set pool1761 --stages 0,1   # AA.2 detection + tracking, full pool
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

POOL1761_MANIFEST = REPO / "dataset" / "manifests" / "recap_v12_1761.jsonl"

SETS = {
    "gen18": dict(kind="sample", out_dir=REPO / "outputs" / "aa1_v2_18clips",
                  n_pos=9, n_neg=9, seed=20260919),
    "pool1761": dict(kind="pool", out_dir=REPO / "outputs" / "aa1_pool1761",
                     manifest=POOL1761_MANIFEST, preroll_s=1.0),
}
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


def pool_video_ids(manifest_path: Path) -> dict:
    """Every unique video_id in a recap-style window manifest (video_id/frames_dir/
    event_occurs/requested_time_to_event/t_seconds - see recap_v12_1761.jsonl), split by class
    for the same printed summary as sample_clips, and filtered to videos whose raw mp4 actually
    exists locally (mirrors sample_clips' own guard)."""
    pos, neg, n_rows = set(), set(), 0
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_rows += 1
            vid = str(row["video_id"]).zfill(5)
            if not (RAW_VIDEO_ROOT / f"{vid}.mp4").exists():
                continue
            (pos if row["event_occurs"] else neg).add(vid)
    return dict(manifest=str(manifest_path), n_windows=n_rows,
                positives=sorted(pos), negatives=sorted(neg))


def load_or_create_list(cfg: dict) -> list:
    out_dir = cfg["out_dir"]
    path = out_dir / "clip_list.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get("kind") == "pool":
            rec = pool_video_ids(cfg["manifest"])
        else:
            rec = sample_clips(cfg["n_pos"], cfg["n_neg"], cfg["seed"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
    print(f"[set] {len(rec['positives'])} pos, {len(rec['negatives'])} neg "
         f"({len(rec['positives']) + len(rec['negatives'])} videos)")
    return rec["positives"] + rec["negatives"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=sorted(SETS))
    ap.add_argument("--stages", default="0,1,2,3")
    args = ap.parse_args()
    cfg = SETS[args.set]
    out_dir = cfg["out_dir"]
    preroll_s = cfg.get("preroll_s", 0.0)
    clips = load_or_create_list(cfg)
    stages = {int(s) for s in args.stages.split(",")}

    if 0 in stages:
        from aa1_scene import YOLOPv2
        import aa1_yolop_cache
        yp = YOLOPv2()
        for vid in clips:
            aa1_yolop_cache.run_clip(yp, vid, out_dir, preroll_s=preroll_s)
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
        if cfg.get("kind") == "pool":
            print(f"[set] '{args.set}' is a pool-kind set ({len(clips)} videos) - Stage 3's "
                 f"full overlay+curves render is for the ~36-clip review sets only. Use "
                 f"aa4_token_labels.py for this set's AA-rel/AA-occ labels instead.")
        else:
            import aa1_stage3
            for vid in clips:
                aa1_stage3.run_clip(vid, out_dir)


if __name__ == "__main__":
    main()
