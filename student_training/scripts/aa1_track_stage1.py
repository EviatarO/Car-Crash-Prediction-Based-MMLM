"""
aa1_track_stage1.py
====================
Stage 1 driver (child plan 2026-09-15-AA1-v2b): run tracking + stitching + hood filter (see
aa1_tracks.py) on all 18 val_e3a clips from the cached YOLOPv2 detections
(outputs/aa1_v2_18clips/<vid>_yolop.json).

- <vid>_tracks_v2.json        raw + stitched tracks, merge log, hood-filtered ids

2026-09-19: dropped this stage's own overlay video/grid/timeline plot (_overlay_v2.mp4,
_grid16_v2.jpg, _timeline_v2.png). Stage 3's own overlay (aa1_stage3.render_overlay) already
shows every tracked box with its id on top of the lane/path geometry, which is what these were
for; keeping a second, tracking-only rendering of the same boxes was redundant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aa1_detect_track_rank import decode_frames, decode_span  # noqa: E402
from aa1_tracks import load_yolop_cache, track_from_yolop, stitch_fragments, is_ego_hood, extend_edge_tracks  # noqa: E402

VAL_E3A_IDS = ["00319", "00077", "00687", "00283", "00147", "00529", "00493", "00474",
               "00372", "01153", "01504", "01643", "01281", "01550", "01737", "02104",
               "02117", "01552"]

# Categorical palette, dataviz skill's validated 8-hue order (pre-validated for line/box
# adjacent-pair contrast) - Stage 3's overlay cycles through it for however many objects a
# window has (aa1_stage3.py imports PALETTE_HEX from here).
PALETTE_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def run_clip(video_id, out_dir: Path):
    cache = load_yolop_cache(out_dir, video_id)
    t_start, t_end = cache["span"]
    frames, timestamps, fps = decode_frames(video_id, t_start, t_end)

    raw_tracks = track_from_yolop(cache, frames)
    stitched, merge_log = stitch_fragments(raw_tracks, frames, timestamps, fps)
    dropped_hood_ids = {tid for tid, pts in stitched.items() if is_ego_hood(pts)}
    _, edge_log = extend_edge_tracks({tid: pts for tid, pts in stitched.items()
                                      if tid not in dropped_hood_ids}, cache)

    record = dict(video_id=video_id, n_frames=len(frames), n_raw_tracks=len(raw_tracks),
                 n_stitched_tracks=len(stitched), n_merges=len(merge_log),
                 n_hood_dropped=len(dropped_hood_ids), hood_track_ids=sorted(dropped_hood_ids),
                 edge_extensions=[dict(track_id=tid, frames_added=n, side=s) for tid, n, s in edge_log],
                 merge_log=[dict(kept=a, absorbed=b, gap_s=g, center_dist_frac=d, hist_sim=s)
                            for a, b, g, d, s in merge_log],
                 tracks={str(tid): [[t, box, conf] for t, box, conf in pts]
                         for tid, pts in stitched.items() if tid not in dropped_hood_ids})
    with open(out_dir / f"{video_id}_tracks_v2.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[{video_id}] raw={len(raw_tracks)} -> stitched={len(stitched)}  "
         f"merges={len(merge_log)}  hood_dropped={len(dropped_hood_ids)}  "
         f"edge_extended={[(tid, n) for tid, n, _ in edge_log]}")
    return record


def main():
    out_dir = REPO / "outputs" / "aa1_v2_18clips"
    results = [run_clip(vid, out_dir) for vid in VAL_E3A_IDS]
    totals = dict(
        raw=sum(r["n_raw_tracks"] for r in results),
        stitched=sum(r["n_stitched_tracks"] for r in results),
        merges=sum(r["n_merges"] for r in results),
        hood=sum(r["n_hood_dropped"] for r in results))
    print(f"\n[summary] total raw tracks={totals['raw']}  stitched={totals['stitched']}  "
         f"merges={totals['merges']}  hood_dropped={totals['hood']}")
    with open(out_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
