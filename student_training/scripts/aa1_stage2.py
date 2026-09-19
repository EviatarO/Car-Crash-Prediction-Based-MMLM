"""
aa1_stage2.py
=============
Shared loaders used by Stage 3 (aa1_stage3.py): decode the cached YOLOPv2 frames into
drivable/lane masks + boxes, trace the drivable-area path, estimate the ego's own lane-line
path (aa1_lanes.estimate_ego_path), and load Stage 1's stitched tracks.

2026-09-19: this used to be its own driver stage with a run_clip/main that wrote
<vid>_geometry_v2.json (per-window object geometry) and rendered <vid>_overlay_lanes.mp4 +
<vid>_grid16_lanes.jpg. Removed - geometry_v2.json was never read by anything downstream (Stage
3 computes its own scores directly from load_frame_masks + the tracks, not from that file), and
the overlay/grid were redundant with Stage 3's own overlay (aa1_stage3.render_overlay), which
already shows the same masks/path together with the tracked boxes and their selection scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
from aa1_lanes import rle_decode, trace_path, estimate_ego_path  # noqa: E402

OUT_DIR = REPO / "outputs" / "aa1_v2_18clips"


def load_frame_masks(video_id: str) -> dict:
    """t -> dict(drivable, lane, boxes, path, ego_path). Decodes every cached YOLOPv2 frame once;
    reused across every object/window for this clip - decoding + tracing is not free
    (~5-10ms/frame), so this must not be repeated per object. The ego path is estimated from the
    lane and drivable masks of the whole clip (aa1_lanes.estimate_ego_path)."""
    with open(OUT_DIR / f"{video_id}_yolop.json", encoding="utf-8") as f:
        cache = json.load(f)
    out = {}
    for fr in cache["frames"]:
        drivable = rle_decode(fr["drivable_rle"])
        lane = rle_decode(fr["lane_rle"])
        boxes = np.array(fr["boxes"], dtype=np.float32).reshape(-1, 4)
        path = trace_path(drivable, lane, vehicle_boxes=boxes)
        out[round(fr["t"], 4)] = dict(drivable=drivable, lane=lane, boxes=boxes, path=path)
    summary = estimate_ego_path(out)
    print(f"[{video_id}] ego path: anchor x={summary['anchor']:.0f} ({summary['n_pair_frames']} "
          f"lane-pair frames), lane-width model={'yes' if summary['lane_width_model'] else 'no'}, "
          f"sources={summary['sources']}")
    return out, cache["fps"], cache["span"]


def load_stitched_tracks(video_id: str) -> dict:
    with open(OUT_DIR / f"{video_id}_tracks_v2.json", encoding="utf-8") as f:
        rec = json.load(f)
    return {int(tid): [(t, box, conf) for t, box, conf in pts] for tid, pts in rec["tracks"].items()}
