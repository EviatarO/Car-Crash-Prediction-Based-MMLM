"""
aa1_tracks.py
=============
Stage 1 (child plan 2026-09-15-AA1-v2b): tracking on top of the cached YOLOPv2 detections
(`outputs/aa1_v2_18clips/<vid>_yolop.json`, written by aa1_yolop_cache.py). Three fixes over
v1's tracking, targeting the user's issues 3 and 5 from the 18-clip review:

1. `lost_track_buffer` raised 30 -> 90 frames (3s @ 30fps; BoT-SORT scales it by frame_rate
   internally). v1 used the package default of 30 (1s), too short for the ~1-2s occlusions in
   00319 (crash car hidden behind another car at the intersection).
2. Offline fragment stitching: BoT-SORT's own re-association can still fail even with a longer
   buffer, because after 1+ seconds hidden the Kalman prediction has drifted and there's no
   appearance model in this tracker (confirmed from its docstring: "IoU association + optional
   CMC", no ReID). This pass merges track B into track A after the fact when A's
   constant-velocity extrapolation lands near B's actual start AND their colour histograms
   match - see stitch_fragments().
3. Ego-hood filter: drop a track that sits in the bottom band of the frame and barely moves in
   position or size - the camera-static signature of a mis-detected hood/dashboard, as opposed
   to a real (moving) vehicle. Kept as a safety net; YOLOPv2 was not observed to box a hood on
   any of the 18 clips (checked directly, see aa1_track_stage1.py's report).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FRAME_W, FRAME_H = 1280, 720
CONF_THRES = 0.30              # matches aa1_scene.CONF_THRES - YOLOPv2 boxes are pre-thresholded
LOST_TRACK_BUFFER = 90         # frames @ frame_rate=30 reference -> 3s (was 30 -> 1s in v1)

# Stitching gates (child plan Stage 1). All PROVISIONAL, set from 00319/00687 only - re-check
# once the 18-clip pass runs.
STITCH_MAX_GAP_S = 3.0
STITCH_MIN_TAIL_SAMPLES = 2     # need >=2 points on A's tail to extrapolate a velocity
STITCH_MAX_CENTER_DIST_FRAC = 1.5   # predicted-vs-actual center distance / predicted box diag
STITCH_SIZE_RATIO_RANGE = (0.4, 2.5)
STITCH_HIST_SIM_MIN = 0.45
STITCH_TAIL_SPAN_S = 0.5       # how much of A's/B's history to use for the velocity fit

# Ego-hood filter (safety net - see module docstring point 3).
HOOD_Y_FRAC = 0.85             # box bottom must be below this fraction of frame height
HOOD_MAX_MOVE_FRAC = 0.03      # center/size relative change, frame-to-frame, to count as "static"
HOOD_MIN_SAMPLES = 10
HOOD_MIN_STATIC_FRAC = 0.8     # fraction of frames that must be "static" to flag as hood


def load_yolop_cache(out_dir: Path, video_id: str) -> dict:
    with open(out_dir / f"{video_id}_yolop.json", encoding="utf-8") as f:
        return json.load(f)


def track_from_yolop(cache: dict, frames: list, conf_thres: float = CONF_THRES,
                     lost_track_buffer: int = LOST_TRACK_BUFFER):
    """Runs BoT-SORT over the cached per-frame YOLOPv2 boxes (already NMS'd by aa1_scene.py).
    `frames` (decoded BGR, same order as cache["frames"]) is required for camera-motion
    compensation (enable_cmc=True) - without it CMC silently no-ops.
    Returns dict[track_id] -> list of (t, [x1,y1,x2,y2], conf), temporal order."""
    from trackers import BoTSORTTracker
    import supervision as sv

    tracker = BoTSORTTracker(frame_rate=cache["fps"], minimum_consecutive_frames=1,
                             enable_cmc=True, track_activation_threshold=conf_thres,
                             high_conf_det_threshold=conf_thres,
                             lost_track_buffer=lost_track_buffer)
    tracks: dict[int, list] = {}
    for fr, frame in zip(cache["frames"], frames):
        boxes = np.array(fr["boxes"], dtype=np.float32).reshape(-1, 4)
        scores = np.array(fr["scores"], dtype=np.float32)
        dets = sv.Detections(xyxy=boxes, confidence=scores) if len(boxes) else sv.Detections.empty()
        tracked = tracker.update(dets, frame=frame)
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            if tid < 0:
                continue
            box = tracked.xyxy[i].tolist()
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else None
            tracks.setdefault(tid, []).append((fr["t"], box, conf))
    return tracks


def _lstsq_slope(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def _extrapolate(tail, t_target):
    """tail: [(t, box)]. Constant-velocity extrapolation of the box CENTER; size held at the
    last observed value (short gaps only, so size shouldn't move much)."""
    ts = [p[0] for p in tail]
    boxes = np.array([p[1] for p in tail], dtype=np.float64)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w, h = boxes[-1, 2] - boxes[-1, 0], boxes[-1, 3] - boxes[-1, 1]
    if len(ts) >= 2 and len(set(ts)) >= 2:
        vcx, vcy = _lstsq_slope(ts, cx), _lstsq_slope(ts, cy)
    else:
        vcx = vcy = 0.0
    dt = t_target - ts[-1]
    pcx, pcy = cx[-1] + vcx * dt, cy[-1] + vcy * dt
    return [pcx - w / 2, pcy - h / 2, pcx + w / 2, pcy + h / 2]


def _color_hist(frame, box, bins=(16, 16)):
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(bins), [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def stitch_fragments(tracks: dict, frames: list, timestamps: list, fps: float,
                     max_gap_s: float = STITCH_MAX_GAP_S):
    """Offline greedy stitching (module docstring point 2). Returns (new_tracks, merge_log)
    where merge_log is [(kept_id, absorbed_id, gap_s, center_dist_frac, hist_sim), ...]."""
    t_to_fi = {round(t, 4): i for i, t in enumerate(timestamps)}

    def frame_at(t):
        fi = t_to_fi.get(round(t, 4))
        return frames[fi] if fi is not None else None

    live = {tid: list(pts) for tid, pts in tracks.items()}
    merge_log = []
    changed = True
    while changed:
        changed = False
        ids = sorted(live.keys(), key=lambda tid: live[tid][-1][0])  # by end time
        used_as_b = set()
        for a_id in ids:
            a_pts = live[a_id]
            a_end_t = a_pts[-1][0]
            best = None  # (cost, b_id, gap, dist_frac, hist_sim)
            for b_id, b_pts in live.items():
                if b_id == a_id or b_id in used_as_b:
                    continue
                b_start_t = b_pts[0][0]
                gap = b_start_t - a_end_t
                if not (0.0 <= gap <= max_gap_s):
                    continue
                tail = [(t, box) for t, box, _ in a_pts if t >= a_end_t - STITCH_TAIL_SPAN_S]
                if len(tail) < STITCH_MIN_TAIL_SAMPLES:
                    tail = a_pts[-STITCH_MIN_TAIL_SAMPLES:]
                pred_box = _extrapolate(tail, b_start_t)
                actual_box = b_pts[0][1]
                pred_cx = (pred_box[0] + pred_box[2]) / 2
                pred_cy = (pred_box[1] + pred_box[3]) / 2
                act_cx = (actual_box[0] + actual_box[2]) / 2
                act_cy = (actual_box[1] + actual_box[3]) / 2
                pred_diag = max(((pred_box[2] - pred_box[0]) ** 2 + (pred_box[3] - pred_box[1]) ** 2) ** 0.5, 1.0)
                dist_frac = ((pred_cx - act_cx) ** 2 + (pred_cy - act_cy) ** 2) ** 0.5 / pred_diag
                if dist_frac > STITCH_MAX_CENTER_DIST_FRAC:
                    continue
                pred_area = max((pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]), 1.0)
                act_area = max((actual_box[2] - actual_box[0]) * (actual_box[3] - actual_box[1]), 1.0)
                size_ratio = act_area / pred_area
                if not (STITCH_SIZE_RATIO_RANGE[0] <= size_ratio <= STITCH_SIZE_RATIO_RANGE[1]):
                    continue
                fa, fb = frame_at(a_end_t), frame_at(b_start_t)
                hist_sim = 0.0
                if fa is not None and fb is not None:
                    ha, hb = _color_hist(fa, a_pts[-1][1]), _color_hist(fb, actual_box)
                    if ha is not None and hb is not None:
                        hist_sim = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
                if hist_sim < STITCH_HIST_SIM_MIN:
                    continue
                cost = dist_frac + (1 - hist_sim)
                if best is None or cost < best[0]:
                    best = (cost, b_id, gap, dist_frac, hist_sim)
            if best is not None:
                _, b_id, gap, dist_frac, hist_sim = best
                live[a_id] = a_pts + live.pop(b_id)
                merge_log.append((a_id, b_id, round(gap, 3), round(dist_frac, 3), round(hist_sim, 3)))
                used_as_b.add(b_id)
                changed = True
                break  # restart the scan - `live` changed
    return live, merge_log


EDGE_PX = 3                 # box touches the frame edge
EDGE_MAX_GAP_S = 0.3        # frames without a usable edge box before the extension stops
EDGE_MIN_Y_OVERLAP = 0.3    # vertical overlap with the previous box (share of the shorter box)
EDGE_MAX_TAKEN_IOU = 0.5    # skip boxes that already belong to another track


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1e-6)


def extend_edge_tracks(tracks: dict, cache: dict, frame_w: int = FRAME_W):
    """Continue a track that ends touching the left/right frame edge with the cached boxes that
    follow at the same edge - including low-confidence ones (>= 0.1) the tracker could not attach.
    BoT-SORT's second-stage match needs IoU >= 0.5 with its constant-velocity prediction, which
    slides off-frame for a partly visible car; on 00319 the crash car (id3) was then dropped at
    18.87s while the detector still boxed its headlight at the right edge (conf 0.12-0.28) until
    the end of the clip. Extends in place; returns (tracks, [(track_id, n_added, side), ...])."""
    frames = cache["frames"]
    index = {round(fr["t"], 4): k for k, fr in enumerate(frames)}
    used: dict[float, list] = {}
    for pts in tracks.values():
        for t, box, _ in pts:
            used.setdefault(round(t, 4), []).append(box)
    log = []
    for tid, pts in tracks.items():
        t_last, cur, _ = pts[-1]
        if cur[2] >= frame_w - EDGE_PX:
            side = "right"
        elif cur[0] <= EDGE_PX:
            side = "left"
        else:
            continue
        added = 0
        for fr in frames[index[round(t_last, 4)] + 1:]:
            t = round(fr["t"], 4)
            if t - t_last > EDGE_MAX_GAP_S:
                break
            best = None
            for b, s in zip(fr["boxes"], fr["scores"]):
                if (side == "right" and b[2] < frame_w - EDGE_PX) or (side == "left" and b[0] > EDGE_PX):
                    continue
                y_ov = (max(0.0, min(b[3], cur[3]) - max(b[1], cur[1]))
                        / max(min(b[3] - b[1], cur[3] - cur[1]), 1.0))
                if y_ov < EDGE_MIN_Y_OVERLAP:
                    continue
                if any(_iou(b, u) > EDGE_MAX_TAKEN_IOU for u in used.get(t, [])):
                    continue
                if best is None or s > best[1]:
                    best = (b, s)
            if best is not None:
                pts.append((fr["t"], list(best[0]), float(best[1])))
                used.setdefault(t, []).append(best[0])
                cur, t_last, added = best[0], t, added + 1
        if added:
            log.append((tid, added, side))
    return tracks, log


def is_ego_hood(pts, frame_h: int = FRAME_H) -> bool:
    """Module docstring point 3. pts: [(t, box, conf)]."""
    if len(pts) < HOOD_MIN_SAMPLES:
        return False
    boxes = np.array([p[1] for p in pts], dtype=np.float64)
    if not np.all(boxes[:, 3] > HOOD_Y_FRAC * frame_h):
        return False
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    size = np.sqrt(np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1e-6))
    dcx = np.abs(np.diff(cx)) / np.maximum(size[:-1], 1.0)
    dcy = np.abs(np.diff(cy)) / np.maximum(size[:-1], 1.0)
    dsize = np.abs(np.diff(size)) / np.maximum(size[:-1], 1.0)
    static = (dcx < HOOD_MAX_MOVE_FRAC) & (dcy < HOOD_MAX_MOVE_FRAC) & (dsize < HOOD_MAX_MOVE_FRAC)
    return bool(static.mean() >= HOOD_MIN_STATIC_FRAC)
