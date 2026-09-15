"""
aa1_detect_track_rank.py
=========================
Stage AA.1 (father plan, detection-guided auxiliary supervision) — offline detection
pipeline SMOKE TEST on the 18 held-out `val_e3a` clips (9 pos / 9 neg; excluded from the
4,446-window training pool, so this is leak-free). Runs the exact chain AA.2-AA.4 will run
at scale: decode raw video at FULL frame rate -> Grounding DINO detect -> BoT-SORT track ->
per-object kinematics -> threat ranking, done SEPARATELY for each of the clip's 3 student
windows (TTE 0.5/1.0/1.5 for positives, MID-4/-8/-10 for negatives) so no window's ranking
ever uses information from after that window's own end. Nothing here touches the student model.

ACCEPTANCE TEST (the reason this script exists): on positive clips, does the true collision
partner rank #1 by threat, in every window it is visible in? Clip 00687 (a gray SUV merging
into the ego lane) is the known cut-in case that a naive "biggest/closest object" heuristic
gets wrong; after the 2026-09-14/15 geometry fixes it ranks #1 in all 3 of its windows.

SCOPING DECISION FOR THIS PASS (stated, not hidden): real lane detection is not wired up.
CLRerNet needs mmdetection/mmcv with custom CUDA ops (heavy install on Windows). UFLD v1/v2
were investigated 2026-09-15: both projects' own pretrained checkpoints are Google-Drive-only
(not scriptable here); the only alternative ONNX exports are 2.9-4.8GB "whole model zoo"
archives for one ~100MB file; the one direct HuggingFace re-upload found (culane_res18.pth,
Aniket200325) is 825MB against an ~100MB expected size from a near-empty, unverified account -
not run, given torch.load's pickle code-execution risk on an unverified checkpoint. Deferred;
revisit with a verified weight source. The ego path is a per-clip CALIBRATED virtual path
instead (child plan 2026-09-14-AA1-threat-ranking-ego-reference): horizon fitted from
detections (refit per window, causally - only that window's own detections), width from
lane/camera-height physics, ego rotation removed via optical flow, bottom occlusion read from
each box's own top-vs-bottom looming. Straight path only - curvature/turns are not modelled;
this is the known remaining source of lane-state error on turning clips (see child plan §1,
problem 3). Tracking is FORWARD-ONLY for this pass (no backward merge / head-padding yet).

DETECTOR: HuggingFace `transformers` port of Grounding DINO (IDEA-Research/grounding-dino-
tiny), not the original repo - avoids the original's custom CUDA-op compilation (Multi-Scale
Deformable Attention), which is painful on Windows. Measured 2026-09-14 on this machine's
local GPU (RTX 1000 Ada, 6GB): ~0.6-0.9s/frame warm - see the module's speed report for what
this implies about the FULL 200k-frame run (D3 in the father plan).

TRACKER: `trackers.BoTSORTTracker` (Roboflow). `minimum_consecutive_frames=1` (not the
default 2/3) because the default silently drops exactly the late-appearing cut-ins this
thread cares about - see father plan I.5/I.6. `high_conf_det_threshold` dropped to match our
own BOX_THRESHOLD (found 2026-09-14: the package default 0.6 blocked a correctly-detected,
0.39-0.65-confidence cut-in vehicle from ever starting a track).

KNOWN DEFERRED ITEM (not done in this pass, cost-only not correctness): negative clips decode
one continuous ~8s block covering all 3 window spans, including the ~2s gap between the
MID-8 and MID-4 windows that no window actually uses (~25% wasted decode+detect). Harmless for
this smoke test's correctness (every window's ranking is still computed causally from only its
own past); worth fixing before AA.2, where it saves real GPU time at 200k-frame scale.

Usage:
  python aa1_detect_track_rank.py --clip 00687 --out-dir ../../outputs/aa1_smoke_18clips
  python aa1_detect_track_rank.py --all --out-dir ../../outputs/aa1_smoke_18clips
  python aa1_detect_track_rank.py --all --out-dir ../../outputs/aa1_smoke_18clips --from-cache
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
RAW_VIDEO_ROOT = Path(
    r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis"
    r"\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\train"
)
TRAIN_CSV = REPO / "dataset" / "train.csv"
OUT_DIR_DEFAULT = REPO / "outputs" / "aa1_smoke_18clips"

FRAME_W, FRAME_H = 1280, 720
PROMPT = "car. truck. bus. motorcycle. bicycle. person."
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25
MIN_BOX_SIDE_PX = 15          # cheap pre-filter: distant specks can't be a partner within 2s
NMS_IOU_THRESHOLD = 0.5        # class-agnostic NMS before tracking - see track_clip()
T_FLOOR = 2.0                  # matches build_train4500_manifest.py's short-clip fallback

# Positive window union span: covers TTE_0.5/1.0/1.5 with margin (father plan AA.2).
POS_SPAN_START_BEFORE_EVENT = 3.5
POS_SPAN_END_BEFORE_EVENT = 0.5
POS_TTE_ENDS = (1.5, 1.0, 0.5)     # student windows, most-future first
# Negative buckets: MID-10/-8/-4 (build_train4500_manifest.py convention). Each window is
# ~1.96s (16 frames @ stride 4 @ ~30fps); see module docstring re: the un-optimized decode span.
NEG_OFFSETS = (10.0, 8.0, 4.0)     # student windows, most-future first

# Ego-path geometry, calibrated per clip (child plan 2026-09-14 §1c). Replaces the fixed
# trapezoid (apex 0.42·H, half-width 0.16·W) that on 00687 put the horizon 140 px too high and
# the path ~2.7x too narrow. Flat-ground pinhole: a ground width L at depth Z spans f·L/Z px and
# its ground row sits f·H_cam/Z below the horizon, so width_px(y) = (L/H_cam)·(y - y_h).
LANE_W_OVER_CAM_H = 2.7        # 3.5 m lane / ~1.3 m dashcam height - physical, not fitted
CAR_W_OVER_LANE_W = 0.5        # 1.8 m car / 3.5 m lane - "one car-width into the path"
HORIZON_FALLBACK_FRAC = 0.60   # only if a window's own causal horizon fit fails
# Bottom-occlusion test (hood, Nexar's blurred bottom band, or frame edge): for a rigid object
# the looming rate read from its top edge equals the one read from its bottom edge. A bottom
# edge that stays pinned while the top edge looms is an occluded bottom. PROVISIONAL constants,
# set looking at 00687 only - re-examine once the 18-clip run gives more cases.
OCCL_MIN_TOP_ALPHA = 0.15
OCCL_MAX_BOTTOM_RATIO = 0.4
OCCL_MIN_LEVER_PX = 40         # min px from horizon before an edge's own looming is trusted
LATERAL_MIN_LEVER_PX = 40      # 2026-09-15 fix (problem 4): same margin for the lateral signal -
                                # near the horizon, path width -> 0 and s/s_rate blow up from
                                # small y_h errors; below this margin lane_state="INVALID"
K_OBJ_OVER_CAM_DEFAULT = 1.3   # object/camera height if a window's horizon fit fails
TTC_INV_CAP = 10.0
TOP_K = 5

# 2026-09-15 fix (problem: <8-sample slopes are noisy, esp. for a just-spawned track): the
# father plan's own rule (AA.4) is >=8 samples in the fit window, else invalid. Implemented as
# a dynamic extension (grow the fit window up to MAX_FIT_SPAN before giving up), not a hard
# drop, so an object doesn't vanish from the ranking the instant it's first detected.
FIT_SPAN = 1.0
MIN_FIT_SAMPLES = 8
MAX_FIT_SPAN = 2.5

# 2026-09-15 fix: the old recency rule ("last seen within 0.5s of window end") is looser than
# the plan's own wording ("present in >=half of the last ~0.5s, to tolerate dropout"; father
# plan AA.3.1). A single detection right at the window boundary used to be enough to qualify.
RECENCY_WINDOW = 0.5
RECENCY_MIN_FRAC = 0.5


def estimate_horizon(tracks, frame_w: int = FRAME_W, frame_h: int = FRAME_H):
    """Horizon row from the detections themselves (Hoiem et al. 2006): on flat ground a box's
    height is h = k·(y2 - y_h), k = object/camera height, so regressing y2 on h gives y_h as
    the intercept. Frame-truncated boxes excluded; bottom-occluded boxes are rejected as
    outliers by the robust refit. Returns (y_h, source, slope, n_used)."""
    ys, hs = [], []
    for pts in tracks.values():
        for _, (x1, y1, x2, y2), _ in pts:
            if x1 <= 2 or y1 <= 2 or x2 >= frame_w - 2 or y2 >= frame_h - 2:
                continue
            ys.append(y2)
            hs.append(y2 - y1)
    if len(ys) >= 30:
        ys, hs = np.asarray(ys), np.asarray(hs)
        keep = np.ones(len(ys), bool)
        for _ in range(3):
            if keep.sum() < 10:
                break
            slope, y_h = np.polyfit(hs[keep], ys[keep], 1)
            r = ys - (y_h + slope * hs)
            mad = np.median(np.abs(r[keep] - np.median(r[keep]))) + 1e-6
            keep = np.abs(r) < 3 * 1.4826 * mad
        # 1/k in [0.4, 1.5] <=> objects 0.67-2.5x the camera height: physically sane
        if keep.sum() >= 10 and 0.4 <= slope <= 1.5 and 0.3 * frame_h < y_h < 0.85 * frame_h:
            return float(y_h), "fit", float(slope), int(keep.sum())
    return HORIZON_FALLBACK_FRAC * frame_h, "fallback", None, len(ys)


def estimate_yaw_shift(frames, y_h: float):
    """Cumulative horizontal image shift u (px, one per frame) caused by ego rotation: median
    LK optical flow in a band just above the horizon, where points are far so translation
    parallax is negligible and flow ≈ rotation. + = scene moves right = ego turning left.
    Computed once per clip from the whole-span horizon (not refit per window - the horizon's
    effect on the flow-sampling band is second-order; see child plan §1 known-leakage note)."""
    shifts = [0.0]
    y0, y1 = max(0, int(y_h) - 150), int(y_h) + 10
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(prev)
    mask[y0:y1, :] = 255
    for f in frames[1:]:
        cur = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        dx = 0.0
        p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8, mask=mask)
        if p0 is not None and len(p0) > 10:
            p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
            ok = st.ravel() == 1
            if ok.sum() > 10:
                dx = float(np.median(p1[ok, 0, 0] - p0[ok, 0, 0]))
        shifts.append(dx)
        prev = cur
    return np.cumsum(shifts)


def path_bounds(y: float, y_h: float, frame_w: int = FRAME_W):
    """Ego-lane path [x_L, x_R] at image row y, straight ahead of the camera."""
    half = 0.5 * LANE_W_OVER_CAM_H * max(y - y_h, 0.0)
    return frame_w / 2 - half, frame_w / 2 + half


def calibrate_clip(frames, timestamps, tracks):
    """Whole-span calibration: used only as the source for decode_frames-derived optical flow
    (yaw_u) and as the fallback horizon. Object ranking uses a per-window REFIT of the horizon
    from only that window's causal tracks - see rank_window()."""
    y_h, horizon_source, horizon_slope, horizon_n = estimate_horizon(tracks)
    n = min(len(frames), len(timestamps))
    yaw_u = estimate_yaw_shift(frames[:n], y_h)
    return dict(y_h=y_h, horizon_source=horizon_source, horizon_slope=horizon_slope,
                horizon_n=horizon_n, yaw_u=yaw_u, timestamps=np.asarray(timestamps[:n]))


# =============================================================================
# Windowing — which span of raw video to decode, and the 3 student windows, per clip
# =============================================================================

def load_event_row(video_id: str):
    with open(TRAIN_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row["id"]).zfill(5) == video_id:
                return row
    raise KeyError(f"{video_id} not found in train.csv")


def video_fps_duration(video_id: str):
    cap = cv2.VideoCapture(str(RAW_VIDEO_ROOT / f"{video_id}.mp4"))
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps, n / fps


def decode_span(video_id: str):
    """Returns (t_start, t_end, is_positive). Raw-pixel seconds into the clip. See module
    docstring's KNOWN DEFERRED ITEM re: negatives decoding one block, not two."""
    row = load_event_row(video_id)
    is_pos = int(row["target"]) == 1
    if is_pos:
        t_event = float(row["time_of_event"])
        t_start = max(T_FLOOR, t_event - POS_SPAN_START_BEFORE_EVENT)
        t_end = max(T_FLOOR, t_event - POS_SPAN_END_BEFORE_EVENT)
        return t_start, t_end, True
    else:
        _, duration = video_fps_duration(video_id)
        mid = duration / 2
        ends = [max(T_FLOOR, mid - off) for off in NEG_OFFSETS]
        t_start = max(T_FLOOR, min(ends) - 2.0)
        t_end = max(ends)
        return t_start, t_end, False


def window_ends(video_id: str):
    """The clip's 3 student-window end times, most-future first, as (label, t_end)."""
    row = load_event_row(video_id)
    is_pos = int(row["target"]) == 1
    if is_pos:
        t_event = float(row["time_of_event"])
        return is_pos, [(f"TTE{tte:g}", max(T_FLOOR, t_event - tte)) for tte in POS_TTE_ENDS]
    else:
        _, duration = video_fps_duration(video_id)
        mid = duration / 2
        return is_pos, [(f"MID-{off:g}", max(T_FLOOR, mid - off)) for off in NEG_OFFSETS]


def decode_frames(video_id: str, t_start: float, t_end: float):
    """Decode every raw frame in [t_start, t_end] at the video's native fps.
    Returns (frames_bgr_list, timestamps_sec_list, fps)."""
    path = RAW_VIDEO_ROOT / f"{video_id}.mp4"
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    i_start = int(round(t_start * fps))
    i_end = int(round(t_end * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i_start)
    frames, timestamps = [], []
    for i in range(i_start, i_end + 1):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        timestamps.append(i / fps)
    cap.release()
    return frames, timestamps, fps


# =============================================================================
# Detection — Grounding DINO (HF transformers port)
# =============================================================================

class Detector:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray):
        """Returns (xyxy (N,4) float32, scores (N,) float32) in ORIGINAL frame pixels."""
        from PIL import Image
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        inputs = self.processor(images=img, text=PROMPT, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[img.size[::-1]])[0]
        boxes = results["boxes"].cpu().numpy().astype(np.float32)
        scores = results["scores"].cpu().numpy().astype(np.float32)
        if len(boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        # cheap pre-filter: drop specks too small to be a collision partner within ~2s
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w >= MIN_BOX_SIDE_PX) & (h >= MIN_BOX_SIDE_PX)
        return boxes[keep], scores[keep]


# =============================================================================
# Tracking — BoT-SORT (forward-only for this smoke pass, see module docstring)
# =============================================================================

def track_clip(detector: Detector, frames, timestamps, fps):
    """Returns dict[track_id] -> list of (t_sec, xyxy, score), in temporal order."""
    from trackers import BoTSORTTracker
    import supervision as sv

    tracker = BoTSORTTracker(frame_rate=fps, minimum_consecutive_frames=1,
                             enable_cmc=True, track_activation_threshold=BOX_THRESHOLD,
                             high_conf_det_threshold=BOX_THRESHOLD)
    tracks: dict[int, list] = {}
    per_frame_det_counts = []
    for frame, t in zip(frames, timestamps):
        xyxy, scores = detector.detect(frame)
        # G-DINO with an open, multi-phrase prompt returns MULTIPLE overlapping candidate
        # boxes per real object with no built-in NMS. Class-agnostic NMS here is required
        # before tracking (found 2026-09-14 on 00687: an untracked object's detections kept
        # splitting association across near-duplicate boxes).
        dets = sv.Detections(xyxy=xyxy, confidence=scores) if len(xyxy) else sv.Detections.empty()
        if len(dets) > 1:
            dets = dets.with_nms(threshold=NMS_IOU_THRESHOLD, class_agnostic=True)
        per_frame_det_counts.append(len(dets))
        tracked = tracker.update(dets, frame=frame)
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i])
            if tid < 0:
                continue
            box = tracked.xyxy[i].tolist()
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else None
            tracks.setdefault(tid, []).append((t, box, conf))
    return tracks, per_frame_det_counts


# =============================================================================
# Geometry — looming, virtual-corridor overlap, threat (father plan AA.4)
# =============================================================================

def _lstsq_slope(t, y):
    """Least-squares slope of y vs t. len(t) must be >= 2."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def track_geometry(track, t_window_end, calib, frame_w=FRAME_W, frame_h=FRAME_H,
                   fit_span=FIT_SPAN):
    """One track -> path-gated ranking signals + AA.4 target dims (child plan 2026-09-14/15).
    `track` = [(t, [x1,y1,x2,y2], conf)], already causal (no point after t_window_end).
    Fit window: the last `fit_span` s, dynamically grown to MAX_FIT_SPAN if that yields fewer
    than MIN_FIT_SAMPLES points (2026-09-15 fix - see module header). Frame- or hood-truncated
    box edges are not object edges and are never used as one."""
    span = fit_span
    pts = [p for p in track if p[0] >= t_window_end - span]
    while len(pts) < MIN_FIT_SAMPLES and span < MAX_FIT_SPAN:
        span = min(span + 0.5, MAX_FIT_SPAN)
        pts = [p for p in track if p[0] >= t_window_end - span]
    low_samples = len(pts) < MIN_FIT_SAMPLES
    if len(pts) < 2:
        return None
    y_h = calib["y_h"]
    ts = np.array([p[0] for p in pts])
    x1, y1, x2, y2 = np.array([p[1] for p in pts], dtype=np.float64).T

    # Re-express every box under the final frame's heading, so ego rotation is not read as
    # object motion (00687: ego turning left moved every static roadside object ~120 px/s).
    idx = np.clip(np.searchsorted(calib["timestamps"], ts - 1e-6), 0, len(calib["yaw_u"]) - 1)
    shift = calib["yaw_u"][idx] - calib["yaw_u"][-1]
    x1c, x2c = x1 - shift, x2 - shift

    cut_side = (x1 <= 2) | (x2 >= frame_w - 2) | (y1 <= 2)
    single_t = len(set(ts)) < 2

    # Bottom occlusion: top-edge looming (y_h - y1 ∝ 1/Z) vs bottom-edge looming (y2 - y_h ∝ 1/Z).
    top_ok = bool(np.all(y1 < y_h - OCCL_MIN_LEVER_PX))
    bottom_ok = bool(np.all(y2 > y_h + OCCL_MIN_LEVER_PX))
    a_top = _lstsq_slope(ts, np.log(y_h - y1)) if top_ok and not single_t else None
    a_bot = _lstsq_slope(ts, np.log(y2 - y_h)) if bottom_ok and not single_t else None
    cut_bottom = bool(np.any(y2 >= frame_h - 2)) or (
        a_top is not None and a_bot is not None
        and a_top > OCCL_MIN_TOP_ALPHA and a_bot < OCCL_MAX_BOTTOM_RATIO * a_top)

    # Looming: slope of log(size ∝ 1/Z). 2026-09-15 fix (problem 2): prefer the top-edge
    # measure WHENEVER the top edge is usable, not only when cut_bottom is also true - the two
    # size definitions estimate the same physical quantity (same d/dt[ln size] under rigid-body
    # motion, see child plan derivation), so switching between them only when occlusion status
    # flips created a spurious jump at that frame, not a real kinematic change.
    if a_top is not None:
        alpha, alpha_mode = a_top, "top"
    elif cut_bottom or cut_side.any():
        alpha, alpha_mode = (0.0 if single_t else _lstsq_slope(ts, np.log(np.maximum(y2 - y1, 1e-3)))), "height"
    else:
        size = np.sqrt(np.maximum((x2 - x1) * (y2 - y1), 1e-6))
        alpha, alpha_mode = (0.0 if single_t else _lstsq_slope(ts, np.log(size))), "area"

    # Ground row. If the bottom is occluded, recover it from the top edge: flat ground gives
    # y_h - y1 = (k-1)·(y2 - y_h), k = object/camera height (from the horizon fit's slope 1/k).
    k = 1.0 / calib["horizon_slope"] if calib["horizon_slope"] else K_OBJ_OVER_CAM_DEFAULT
    if cut_bottom and a_top is not None and k > 1.05:
        y_ground = y_h + (y_h - y1) / (k - 1.0)
    else:
        y_ground = y2
    y_ground = np.minimum(y_ground, frame_h)

    # Proximity rho = Z_bottom / Z (flat ground: Z ∝ 1/(y_ground - y_h)), referenced to the frame
    # bottom, the closest visible ground row. 1 = at or below it.
    rho = float(np.clip((y_ground[-1] - y_h) / max(frame_h - y_h, 1.0), 0, 1))

    # Signed lateral offset s in ego-path widths at the object's ground row: >0 = overlapping
    # the path (fraction of path width covered), <0 = gap. Normalized by PATH width, not box
    # width, so a frame-truncated box is measured correctly. 2026-09-15 fix (problem 4): the
    # validity margin was y_h+5px, so near-horizon objects (where path width -> 0) produced
    # huge, unstable s from tiny y_h errors - raised to LATERAL_MIN_LEVER_PX (40px), matching
    # the occlusion-test's own margin.
    y_ref = y_ground
    lateral_valid = bool(np.all(y_ref > y_h + LATERAL_MIN_LEVER_PX))
    if lateral_valid:
        s = np.empty(len(ts))
        for i in range(len(ts)):
            xL, xR = path_bounds(y_ref[i], y_h, frame_w)
            s[i] = (min(x2c[i], xR) - max(x1c[i], xL)) / (xR - xL)
        s = np.minimum(s, 1.0)
        s_end = float(s[-1])
        s_rate = 0.0 if single_t else _lstsq_slope(ts, s)      # AA.4 target dim - see docstring
        in_path = float(np.clip(s_end / CAR_W_OVER_LANE_W, 0.0, 1.0))
        # ttc_lat_inv: RANKING signal only (not an AA.4 target - 2026-09-15 fix, problem 1).
        # s_rate/(0.5-s_end) blows up as s_end -> 0.5 (denominator floored at 0.05, so a small
        # s_rate change is amplified ~20x near a full car-width in) - fine for picking the top-K
        # candidates, wrong as a regression target (the student would be trained on a scale that
        # swings wildly for a physically smooth approach). Use s_rate directly as the target.
        ttc_lat_inv = min(max(s_rate, 0.0) / max(CAR_W_OVER_LANE_W - s_end, 0.05), TTC_INV_CAP)
    else:
        s_end = s_rate = None
        in_path = ttc_lat_inv = 0.0

    t_long = max(alpha, 0.0) * in_path     # rear-end / stopped / head-on: looming, only in path
    t_lat = rho * ttc_lat_inv              # cut-in / pull-out / crossing: entry, only when near
    threat = max(t_long, t_lat)            # dominant mechanism, 1/s - RANKING ONLY, not a target

    if lateral_valid:
        xL, xR = path_bounds(y_ref[-1], y_h, frame_w)
        lane_state = "IN" if s_end > 0 else ("LEFT" if x2c[-1] < xL else "RIGHT")
    else:
        lane_state = "INVALID"
    return dict(threat=round(threat, 4), t_long=round(t_long, 4), t_lat=round(t_lat, 4),
                alpha=round(float(alpha), 4), alpha_mode=alpha_mode, in_path=round(in_path, 4),
                rho=round(rho, 4),
                s_end=None if s_end is None else round(s_end, 4),
                s_rate=None if s_rate is None else round(float(s_rate), 4),
                ttc_lat_inv=round(float(ttc_lat_inv), 4), lane_state=lane_state,
                y_ground_end=round(float(y_ground[-1]), 1),
                alpha_top=None if a_top is None else round(a_top, 4),
                alpha_bottom=None if a_bot is None else round(a_bot, 4),
                cut_side=bool(cut_side.any()), cut_bottom=cut_bottom,
                n_samples=len(pts), fit_span_used=round(span, 2), low_samples=low_samples,
                last_box=[float(v) for v in pts[-1][1]], last_t=float(ts[-1]))


def recency_ok(pts, t_end: float, fps: float) -> bool:
    """Father plan AA.3.1: keep an object only if present in >=half of the window's last
    RECENCY_WINDOW seconds (tolerates brief dropout, rejects an object only glimpsed once at
    the boundary). 2026-09-15 fix - the old rule only checked the LAST detection's timestamp."""
    lo = t_end - RECENCY_WINDOW
    n_in = sum(1 for t, _, _ in pts if lo - 1e-6 <= t <= t_end + 1e-6)
    need = max(1, round(RECENCY_WINDOW * fps * RECENCY_MIN_FRAC))
    return n_in >= need


def rank_window(tracks, t_end: float, base_calib, fps: float):
    """Causal ranking for ONE student window: every track truncated to t<=t_end, horizon
    REFIT from only that window's own causal detections (so an early window's ranking cannot
    benefit from a later window's boxes), yaw-shift reused from the whole-span estimate but
    sliced to t<=t_end (see calibrate_clip / module docstring re: this residual approximation).
    Returns (ranked [(tid,geom),...] sorted by threat desc, geoms dict, the window's calib)."""
    causal_tracks = {}
    for tid, pts in tracks.items():
        cp = [p for p in pts if p[0] <= t_end + 1e-6]
        if cp:
            causal_tracks[tid] = cp
    y_h, src, slope, n_h = estimate_horizon(causal_tracks)
    idx = max(1, int(np.searchsorted(base_calib["timestamps"], t_end + 1e-6)))
    cal = dict(y_h=y_h, horizon_source=src, horizon_slope=slope, horizon_n=n_h,
              yaw_u=base_calib["yaw_u"][:idx], timestamps=base_calib["timestamps"][:idx])
    geoms = {}
    for tid, pts in causal_tracks.items():
        if not recency_ok(pts, t_end, fps):
            continue
        g = track_geometry(pts, t_end, cal)
        if g is not None:
            geoms[tid] = g
    ranked = sorted(geoms.items(), key=lambda kv: -kv[1]["threat"])
    return ranked, geoms, cal


# =============================================================================
# Overlay rendering
# =============================================================================

def render_overlay(video_id, frames, timestamps, tracks, geoms, calib, out_dir: Path, fps):
    """Writes <vid>_overlay.mp4 (every frame) and <vid>_grid16.jpg (16 evenly spaced frames
    across the WHOLE decoded span - not one window; see module docstring). Ranking shown is
    the PRIMARY window's (closest to the decision point: TTE0.5 / MID-4). Top-K tracks in rank
    colors with threat; other tracks thin gray. Gray line = horizon, cyan = ego-lane path.
    No ego box (removed 2026-09-15: it was a drawn rectangle at a fixed size, not detected or
    measured from anything - see child plan §2 discussion)."""
    h, w = frames[0].shape[:2]
    ranked = sorted(geoms.items(), key=lambda kv: -kv[1]["threat"])
    rank_of = {tid: i for i, (tid, _) in enumerate(ranked)}
    colors = [(0, 0, 255), (0, 140, 255), (0, 255, 255), (0, 255, 0), (255, 120, 0)]
    y_h = int(calib["y_h"])
    xL0, xR0 = path_bounds(h - 1, y_h, w)
    by_t = {}
    for tid, pts in tracks.items():
        for tt, box, _ in pts:
            by_t.setdefault(tt, []).append((tid, box))
    grid_idx = set(np.linspace(0, len(frames) - 1, 16).round().astype(int).tolist())
    tiles = []
    writer = cv2.VideoWriter(str(out_dir / f"{video_id}_overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fi, (frame, t) in enumerate(zip(frames, timestamps)):
        vis = frame.copy()
        cv2.line(vis, (0, y_h), (w, y_h), (200, 200, 200), 1)
        cv2.line(vis, (w // 2, y_h), (int(xL0), h - 1), (255, 255, 0), 2)
        cv2.line(vis, (w // 2, y_h), (int(xR0), h - 1), (255, 255, 0), 2)
        for tid, box in sorted(by_t.get(t, []), key=lambda tb: -rank_of.get(tb[0], 99)):
            x1, y1, x2, y2 = (int(v) for v in box)
            r = rank_of.get(tid)
            if r is not None and r < TOP_K:
                color, thick = colors[r], 3
                label = f"#{r + 1} id{tid} {geoms[tid]['threat']:.2f}"
            else:
                color, thick, label = (160, 160, 160), 1, f"id{tid}"
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)
            cv2.putText(vis, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(vis, f"{video_id}  t={t:.2f}s", (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3)
        writer.write(vis)
        if fi in grid_idx:
            tiles.append(cv2.resize(vis, (w * 2 // 5, h * 2 // 5)))
    writer.release()
    while len(tiles) < 16:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    cv2.imwrite(str(out_dir / f"{video_id}_grid16.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


# =============================================================================
# Diagnostic curves — the 5 AA.4 target dims (+ derived threat) per frame, causal
# =============================================================================

# Palette: dataviz skill's validated 8-hue categorical order, slots 1-5 (blue/orange/aqua/
# yellow/magenta) - pre-validated for line-chart adjacent-pair CVD/normal-vision contrast in
# both light and dark mode, so safe to use up to 5 series without re-running the validator.
CURVE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
CURVE_INK, CURVE_INK2, CURVE_MUTED = "#0b0b0b", "#52514e", "#898781"
CURVE_GRID, CURVE_AXIS, CURVE_SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"


def render_target_curves(video_id, tracks, base_calib, win_ends, out_dir: Path, top_ids,
                         id_labels: dict):
    """Per-frame curves of the 5 AA.4 target dims + derived threat, for `top_ids` (the primary
    window's top-K), computed causally at every decoded frame (whole-span horizon/yaw reused
    per frame, NOT refit per frame - a diagnostic-only approximation; the actual per-window
    RANKING above does refit causally). Writes a PNG + a CSV with every object x every frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts_all = base_calib["timestamps"]
    n_frames = len(ts_all)
    LANE = {"LEFT": -1.0, "IN": 0.0, "RIGHT": 1.0}
    dims = ["alpha", "s_end", "s_rate", "rho", "lane", "threat"]
    series = {tid: {d: np.full(n_frames, np.nan) for d in dims} for tid in top_ids}
    rows = []
    for f in range(n_frames):
        t = ts_all[f]
        cal_f = {**base_calib, "yaw_u": base_calib["yaw_u"][:f + 1],
                 "timestamps": base_calib["timestamps"][:f + 1]}
        for tid in top_ids:
            pts = [p for p in tracks[tid] if p[0] <= t + 1e-6]
            if not pts or abs(pts[-1][0] - t) > 1e-6:
                continue
            g = track_geometry(pts, t, cal_f)
            if g is None or g["low_samples"]:
                continue
            s = series[tid]
            s["alpha"][f], s["rho"][f], s["threat"][f] = g["alpha"], g["rho"], g["threat"]
            s["s_end"][f] = np.nan if g["s_end"] is None else g["s_end"]
            s["s_rate"][f] = np.nan if g["s_rate"] is None else g["s_rate"]
            s["lane"][f] = LANE.get(g["lane_state"], np.nan)
            rows.append(dict(frame=f, t=round(float(t), 3), track_id=tid, object=id_labels[tid],
                             n_samples=g["n_samples"], fit_span_used=g["fit_span_used"],
                             alpha=g["alpha"], alpha_mode=g["alpha_mode"],
                             lane_overlap_s=g["s_end"], lane_overlap_rate=g["s_rate"],
                             proximity=g["rho"], lane_state=g["lane_state"], threat=g["threat"]))

    if rows:
        with open(out_dir / f"{video_id}_target_curves.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    panels = [
        ("alpha", "1. Looming \u03b1  (1/s)", "growth rate of the object's size; 0.5 \u2248 2 s to contact"),
        ("s_end", "2. Lane overlap s  (lane widths)", "share of my lane covered; <0 = distance outside; 0.5 \u2248 one car-width in"),
        ("s_rate", "3. Lane-overlap rate  (lane widths / s)", "AA.4 target dim (fixed 2026-09-15: was the capped ttc_lat_inv ratio, which blew up near s=0.5)"),
        ("rho", "4. Proximity \u03c1", "0 = at horizon (far), 1 = at image bottom (as close as visible)"),
        ("lane", "5. Lane state", "which lane the object is in"),
        ("threat", "Derived: threat (ranking only, not a training target)", "max(\u03b1 \u00d7 in-path, \u03c1 \u00d7 entry rate)"),
    ]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 15.5), sharex=True, facecolor=CURVE_SURFACE)
    x = ts_all - ts_all[0]
    for ax, (key, title, sub) in zip(axes, panels):
        ax.set_facecolor(CURVE_SURFACE)
        for i, tid in enumerate(top_ids):
            y = series[tid][key]
            color = CURVE_COLORS[i % len(CURVE_COLORS)]
            if key == "lane":
                ax.step(x, y, where="mid", color=color, lw=2, label=id_labels[tid])
            else:
                ax.plot(x, y, color=color, lw=2, label=id_labels[tid])
        for label, te in win_ends:
            ax.axvline(te - ts_all[0], color=CURVE_MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        if key == "s_end":
            ax.axhline(0.5, color=CURVE_AXIS, lw=1, zorder=0)
            ax.axhline(0.0, color=CURVE_AXIS, lw=1, zorder=0)
        if key == "lane":
            ax.set_yticks([-1, 0, 1], ["LEFT", "IN", "RIGHT"])
            ax.set_ylim(-1.5, 1.5)
        if key == "rho":
            ax.set_ylim(0, 1.05)
        ax.set_title(title, loc="left", color=CURVE_INK, fontsize=11, fontweight="semibold", pad=16)
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=CURVE_INK2, fontsize=9, va="bottom")
        ax.grid(axis="y", color=CURVE_GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(CURVE_AXIS)
        ax.tick_params(colors=CURVE_MUTED)
    for label, te in win_ends:
        axes[0].text(te - ts_all[0], axes[0].get_ylim()[1], f" {label}", color=CURVE_MUTED,
                    fontsize=8, va="top", ha="left")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=min(len(top_ids), 5),
                  frameon=False, fontsize=9, labelcolor=CURVE_INK2, handlelength=3)
    axes[-1].set_xlabel(f"seconds into the decoded span  ({n_frames} frames)", color=CURVE_INK2)
    fig.suptitle(f"Clip {video_id}: AA.4 target dims per frame (causal, dashed = window ends)",
                x=0.06, ha="left", color=CURVE_INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"{video_id}_target_curves.png", dpi=110, facecolor=CURVE_SURFACE)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def cache_path(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}_tracks.json"


def detect_and_cache(detector, video_id, out_dir: Path):
    """Expensive half: decode + detect + track, written to <vid>_tracks.json."""
    t0 = time.time()
    t_start, t_end, is_pos = decode_span(video_id)
    frames, timestamps, fps = decode_frames(video_id, t_start, t_end)
    t_decode = time.time()
    print(f"[{video_id}] span=[{t_start:.2f},{t_end:.2f}]s  frames={len(frames)}  fps={fps:.1f}"
         f"  (decode {t_decode - t0:.1f}s)")

    tracks, det_counts = track_clip(detector, frames, timestamps, fps)
    t_track = time.time()
    print(f"[{video_id}] {sum(det_counts)} raw detections over {len(frames)} frames"
         f" ({sum(det_counts)/max(len(frames),1):.1f}/frame)  ->  {len(tracks)} tracks"
         f"  ({t_track - t_decode:.1f}s, {(t_track - t_decode)/max(len(frames),1):.3f}s/frame)")

    out_dir.mkdir(parents=True, exist_ok=True)
    cache = dict(video_id=video_id, is_positive=is_pos, span=[t_start, t_end], fps=fps,
                 frame_w=FRAME_W, frame_h=FRAME_H, n_frames=len(frames),
                 timestamps=timestamps, det_counts=det_counts,
                 tracks={str(tid): [[t, box, conf] for t, box, conf in pts]
                         for tid, pts in tracks.items()},
                 timing_s=dict(decode=round(t_decode - t0, 2), track=round(t_track - t_decode, 2)))
    with open(cache_path(out_dir, video_id), "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return cache


def load_cache(out_dir: Path, video_id: str):
    with open(cache_path(out_dir, video_id), encoding="utf-8") as f:
        return json.load(f)


def run_clip(detector, video_id, out_dir: Path, from_cache: bool = False):
    cache = load_cache(out_dir, video_id) if from_cache else detect_and_cache(detector, video_id, out_dir)
    t_start, t_end = cache["span"]
    is_pos, fps = cache["is_positive"], cache["fps"]
    timestamps = cache["timestamps"]
    det_counts = cache["det_counts"]
    tracks = {int(tid): [(t, box, conf) for t, box, conf in pts]
              for tid, pts in cache["tracks"].items()}
    frames, _, _ = decode_frames(video_id, t_start, t_end)
    base_calib = calibrate_clip(frames, timestamps, tracks)
    print(f"[{video_id}] whole-span horizon y={base_calib['y_h']:.0f} "
          f"({base_calib['horizon_source']}, n={base_calib['horizon_n']})  "
          f"yaw shift {base_calib['yaw_u'][-1]:+.0f}px")

    _, wins = window_ends(video_id)
    wins = [(label, te) for label, te in wins if t_start - 1e-6 <= te <= t_end + 1e-6]
    window_records = []
    primary_ranked, primary_geoms_raw, primary_cal, primary_label = [], {}, base_calib, None
    for label, te in wins:
        ranked, geoms, cal = rank_window(tracks, te, base_calib, fps)
        window_records.append(dict(
            label=label, t_end=round(te, 3),
            calib=dict(y_h=round(cal["y_h"], 1), horizon_source=cal["horizon_source"],
                      horizon_slope=cal["horizon_slope"], horizon_n=cal["horizon_n"]),
            n_kept=len(geoms), top_k=TOP_K,
            ranking=[dict(rank=i + 1, track_id=tid, in_top_k=i < TOP_K, **g)
                     for i, (tid, g) in enumerate(ranked)]))
        top3 = ", ".join(f"id{tid}(thr={g['threat']:.2f} {g['lane_state']}"
                         f"{' LOW_N' if g['low_samples'] else ''})" for tid, g in ranked[:3])
        print(f"[{video_id}] {label} (t={te:.2f}s, {len(geoms)} kept): top-3: {top3 or '(none)'}")
        # Primary = closest to the decision point (last in wins, most-future-first order):
        # TTE0.5 for positives, MID-4 for negatives. Reused below for the overlay's ranking
        # colors and the curve figure's object selection - no need to recompute.
        primary_ranked, primary_geoms_raw, primary_cal, primary_label = ranked, geoms, cal, label

    out_dir.mkdir(parents=True, exist_ok=True)
    record = dict(video_id=video_id, is_positive=is_pos, span=[t_start, t_end], fps=fps,
                 n_frames=len(frames), n_raw_detections=sum(det_counts), n_tracks=len(tracks),
                 base_calib=dict(y_h=round(base_calib["y_h"], 1),
                                 horizon_source=base_calib["horizon_source"],
                                 yaw_shift_total_px=round(float(base_calib["yaw_u"][-1]), 1)),
                 windows=window_records, primary_window=primary_label,
                 all_tracks_debug=[dict(track_id=tid, n_pts=len(pts), first_t=pts[0][0],
                                        last_t=pts[-1][0], first_box=pts[0][1],
                                        last_box=pts[-1][1])
                                   for tid, pts in tracks.items()],
                 timing_s=cache["timing_s"])
    with open(out_dir / f"{video_id}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    render_overlay(video_id, frames, timestamps, tracks, primary_geoms_raw, primary_cal, out_dir, fps)

    primary_ids = [tid for tid, _ in primary_ranked[:TOP_K]]
    if primary_ids:
        rank_of = {tid: i + 1 for i, (tid, _) in enumerate(primary_ranked)}
        id_labels = {tid: f"id{tid} (rank{rank_of[tid]}, thr={primary_geoms_raw[tid]['threat']:.2f})"
                    for tid in primary_ids}
        render_target_curves(video_id, tracks, base_calib, wins, out_dir, primary_ids, id_labels)

    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=None, help="single video_id")
    ap.add_argument("--all", action="store_true", help="run all 18 val_e3a clips")
    ap.add_argument("--out-dir", default=str(OUT_DIR_DEFAULT))
    ap.add_argument("--from-cache", action="store_true",
                    help="skip detection; recompute geometry/ranking from <vid>_tracks.json")
    args = ap.parse_args()

    val_e3a_ids = ["00319", "00077", "00687", "00283", "00147", "00529", "00493", "00474",
                   "00372", "01153", "01504", "01643", "01281", "01550", "01737", "02104",
                   "02117", "01552"]
    clips = [args.clip] if args.clip else (val_e3a_ids if args.all else [])
    if not clips:
        raise SystemExit("pass --clip VIDEO_ID or --all")

    detector = None
    if not args.from_cache:
        print("[setup] loading Grounding DINO (IDEA-Research/grounding-dino-tiny) ...")
        detector = Detector()
        print("[setup] ready")

    out_dir = Path(args.out_dir)
    results = []
    for vid in clips:
        results.append(run_clip(detector, vid, out_dir, from_cache=args.from_cache))

    with open(out_dir / "smoke_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[done] {len(results)} clips -> {out_dir}")


if __name__ == "__main__":
    main()
