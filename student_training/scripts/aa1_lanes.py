"""
aa1_lanes.py
============
Stage 2 (child plan 2026-09-15-AA1-v2b): the ego path, traced per frame directly from
YOLOPv2's drivable-area and lane-line masks - replacing v1's horizon-physics geometry
entirely (no fitted/fallback horizon, no fixed lane-width-from-camera-height formula, no
turn/roll compensation needed for the lateral signal - see below for why).

APPROACH (simpler than the v2/v2b plan's original "vanishing point + lane polylines" design,
adopted after prototyping - see child plan addendum):
Trace the ego path row by row, starting at the frame bottom (where the ego unambiguously is,
by construction) and walking upward through the DRIVABLE-AREA mask, at each row keeping the
contiguous drivable run whose center is closest to the previous row's center (simple
connected-component following). This gives x_left(y), x_right(y) directly from the
segmentation - no vanishing-point fit, no assumed camera height/pitch/roll, and curves/turns
are traced as-is because the mask itself curves with the road.
Where the (sparser, noisier) lane-line mask has pixels near the traced drivable edges at a
row, they SHARPEN the boundary to the actual line; otherwise the drivable-area edge is used
directly (this is what makes an unmarked road or an intersection still work - drivable area
segmentation needs no painted lines).

WHY NO EGO-ROTATION COMPENSATION IS NEEDED HERE (unlike v1's estimate_yaw_shift): each frame's
path is traced fresh from THAT frame's own mask. An object's lateral overlap at each past
frame in the fit window is measured against that SAME frame's own traced path, not
reprojected onto a single assumed-static geometry - so a turning or rolled camera is not a
special case, it just produces a differently-shaped path at each frame, which is exactly what
gets measured. v1 needed yaw compensation only because its lane model was one fixed-in-time
formula being compared against multiple past frames.
"""
from __future__ import annotations

import numpy as np

FRAME_W, FRAME_H = 1280, 720
ROW_STEP = 4                    # trace every 4th row - cheap, plenty dense for a lane boundary
ROW_TOP_MARGIN = 20             # stop tracing this many rows before the mask runs out
MIN_RUN_WIDTH_PX = 20           # a drivable run narrower than this is noise, not a lane
LANE_SNAP_MARGIN_PX = 60        # look for lane-line pixels within this many px of a drivable edge
SEED_MAX_CENTER_DIST_PX = 200   # a seed candidate must be within this of frame-center to count -
                                # otherwise the nearest-to-ego qualifying run can be an
                                # irrelevant stray patch at the frame edge (2026-09-15 bug,
                                # 00147: a 29px sliver at x=0 was accepted as the seed simply
                                # for being the first row scanned with ANY run >= MIN_RUN_WIDTH_PX,
                                # while the real lane - straight ahead, partly bridged by the
                                # vehicle-fill - sat unexamined a few rows further up)
MIN_PATH_WIDTH_PX = 40          # a run narrower than this is too close to the vanishing point to
                                # trust - skipped, not stored, but does NOT stop the trace (see
                                # MAX_CONSECUTIVE_MISS; a 2026-09-15 bug had this as a hard break,
                                # which killed tracing on any clip where the seed row itself -
                                # the first visible road sliver past the hood/an obstruction -
                                # happened to be narrower than 40px, e.g. 00687, 00147, 00319)
MAX_CONSECUTIVE_MISS = 6        # rows (at ROW_STEP=4, ~24px) of no-run-or-too-narrow before
                                # really stopping - tolerates a car briefly covering the road,
                                # or the road narrowing before widening again


def rle_decode(rle: dict) -> np.ndarray:
    """Counterpart to aa1_yolop_cache.rle_encode."""
    shape = tuple(rle["shape"])
    runs = rle["runs"]
    val = rle["start"]
    out = np.empty(int(np.prod(shape)), dtype=bool)
    pos = 0
    for run in runs:
        out[pos:pos + run] = bool(val)
        pos += run
        val = 1 - val
    return out.reshape(shape)


def _runs_in_row(mask_row: np.ndarray):
    """Contiguous True runs in a 1D boolean row. Returns [(x_start, x_end), ...] (x_end exclusive)."""
    if not mask_row.any():
        return []
    d = np.diff(mask_row.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    ends = list(np.flatnonzero(d == -1) + 1)
    if mask_row[0]:
        starts = [0] + starts
    if mask_row[-1]:
        ends = ends + [len(mask_row)]
    return list(zip(starts, ends))


def fill_vehicle_gaps(drivable: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """A vehicle sits ON the drivable surface almost by definition, so a close leading vehicle
    should not read as a gap in the road. Without this, a car directly ahead splits the
    drivable mask into two side patches and the tracer picks one arbitrarily - confirmed
    2026-09-15 on 00147: the traced path jumped entirely to a side patch, missing that the
    lane continues straight through/behind the car. Returns drivable OR'd with box interiors
    (only the lower ~60% of each box - the visible wheels/underside area, not the roof/sky
    above it, which is not part of the road)."""
    if len(boxes) == 0:
        return drivable
    out = drivable.copy()
    h, w = drivable.shape
    for x1, y1, x2, y2 in boxes:
        x1, x2 = int(max(0, x1)), int(min(w, x2))
        y_mid = int(y1 + (y2 - y1) * 0.4)
        y2c = int(min(h, y2))
        if x2 > x1 and y2c > y_mid:
            out[y_mid:y2c, x1:x2] = True
    return out


def trace_path(drivable: np.ndarray, lane: np.ndarray, frame_w: int = FRAME_W, frame_h: int = FRAME_H,
               row_step: int = ROW_STEP, vehicle_boxes: np.ndarray | None = None):
    """Returns dict: row -> dict(x_left, x_right, x_center, source in {'lane','drivable'}).
    Traces from the bottom row upward; stops when the drivable run vanishes or narrows below
    MIN_RUN_WIDTH_PX. Rows are only included while a valid run was found (no extrapolation
    here - callers interpolate/hold the nearest valid row, see path_bounds_at).
    `vehicle_boxes`, if given (Nx4 xyxy, all boxes in this frame INCLUDING the object being
    scored - the ego's own path doesn't care which object it is), are filled into the
    drivable mask before tracing - see fill_vehicle_gaps()."""
    if vehicle_boxes is not None and len(vehicle_boxes):
        drivable = fill_vehicle_gaps(drivable, vehicle_boxes)
    path = {}
    y = frame_h - 1
    # Seed at the bottom: the drivable run whose center is nearest the image center - the ego
    # is, by construction, on the drivable surface directly ahead at the very bottom row.
    seed_row, seed_run = None, None
    for yy in range(frame_h - 1, int(frame_h * 0.3), -row_step):
        runs = [r for r in _runs_in_row(drivable[yy]) if r[1] - r[0] >= MIN_RUN_WIDTH_PX]
        if not runs:
            continue
        best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - frame_w / 2))
        if abs((best[0] + best[1]) / 2 - frame_w / 2) <= SEED_MAX_CENTER_DIST_PX:
            seed_row, seed_run = yy, best
            break
    if seed_row is None:
        return path
    prev_center = (seed_run[0] + seed_run[1]) / 2

    miss_count = 0
    for yy in range(seed_row, ROW_TOP_MARGIN, -row_step):
        runs = [r for r in _runs_in_row(drivable[yy]) if r[1] - r[0] >= MIN_RUN_WIDTH_PX]
        candidate = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - prev_center)) if runs else None
        if candidate is None or candidate[1] - candidate[0] < MIN_PATH_WIDTH_PX:
            miss_count += 1
            if miss_count > MAX_CONSECUTIVE_MISS:
                break
            continue
        miss_count = 0
        x0, x1 = candidate
        prev_center = (x0 + x1) / 2

        lane_row = lane[yy]
        left_src, right_src = "drivable", "drivable"
        lo, hi = max(0, x0 - LANE_SNAP_MARGIN_PX), min(frame_w, x0 + LANE_SNAP_MARGIN_PX)
        left_lane_px = np.flatnonzero(lane_row[lo:hi]) + lo
        if len(left_lane_px):
            x0 = float(left_lane_px[np.argmin(np.abs(left_lane_px - x0))])
            left_src = "lane"
        lo, hi = max(0, x1 - LANE_SNAP_MARGIN_PX), min(frame_w, x1 + LANE_SNAP_MARGIN_PX)
        right_lane_px = np.flatnonzero(lane_row[lo:hi]) + lo
        if len(right_lane_px):
            x1 = float(right_lane_px[np.argmin(np.abs(right_lane_px - x1))])
            right_src = "lane"

        path[yy] = dict(x_left=float(x0), x_right=float(x1), x_center=(x0 + x1) / 2,
                        source=f"{left_src}/{right_src}")
    return path


def path_bounds_at(y: float, path: dict, frame_w: int = FRAME_W):
    """Look up (x_left, x_right) at row y from a traced path dict, holding the nearest traced
    row's value if y falls between sample rows or outside the traced range. Returns
    (None, None) if the path is empty (no drivable area found at all in this frame)."""
    if not path:
        return None, None
    rows = np.array(sorted(path.keys()))
    nearest = rows[np.argmin(np.abs(rows - y))]
    p = path[nearest]
    return p["x_left"], p["x_right"]


# =============================================================================
# Per-object geometry (Stage 2) - alpha kept horizon-free (v1's top-edge-above-horizon trick
# needed a global horizon estimate we no longer compute); lateral position/rate/proximity come
# from the per-frame traced path above instead of v1's physics-derived corridor. THREAT IS NOT
# COMPUTED HERE - that needs the collision check, Stage 3 (see child plan). This intentionally
# leaves alpha/lateral/proximity as the only populated fields.
# =============================================================================

MIN_FIT_SAMPLES = 8
MAX_FIT_SPAN = 2.5
FIT_SPAN = 1.0


def _lstsq_slope(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def track_geometry_v2(track, t_window_end, frame_masks: dict, frame_w: int = FRAME_W,
                      frame_h: int = FRAME_H, fit_span: float = FIT_SPAN):
    """`track`: causal [(t, [x1,y1,x2,y2], conf)]. `frame_masks`: t -> dict(drivable, lane,
    boxes) for every frame in the fit window (decoded once by the caller, reused across
    objects/windows - decoding is not free). Returns None if too few points."""
    span = fit_span
    pts = [p for p in track if p[0] >= t_window_end - span]
    while len(pts) < MIN_FIT_SAMPLES and span < MAX_FIT_SPAN:
        span = min(span + 0.5, MAX_FIT_SPAN)
        pts = [p for p in track if p[0] >= t_window_end - span]
    low_samples = len(pts) < MIN_FIT_SAMPLES
    if len(pts) < 2:
        return None
    ts = np.array([p[0] for p in pts])
    boxes = np.array([p[1] for p in pts], dtype=np.float64)
    x1, y1, x2, y2 = boxes.T

    cut_left, cut_right = x1 <= 2, x2 >= frame_w - 2
    cut_top, cut_bottom = y1 <= 2, y2 >= frame_h - 2
    w, h = x2 - x1, y2 - y1
    if not np.any(cut_left | cut_right) and not np.any(cut_top | cut_bottom):
        size, alpha_mode = np.sqrt(np.maximum(w * h, 1e-6)), "area"
    elif not np.any(cut_left | cut_right):
        size, alpha_mode = np.maximum(w, 1e-3), "width"
    elif not np.any(cut_top | cut_bottom):
        size, alpha_mode = np.maximum(h, 1e-3), "height"
    else:
        size, alpha_mode = np.sqrt(np.maximum(w * h, 1e-6)), "area_lowconf"
    single_t = len(set(ts)) < 2
    alpha = 0.0 if single_t else _lstsq_slope(ts, np.log(size))

    s_vals, rho_vals, lane_states = [], [], []
    for t, box in zip(ts, boxes):
        fm = frame_masks.get(round(float(t), 4))
        if fm is None:
            continue
        path = fm["path"]
        if not path:
            continue
        bx1, by1, bx2, by2 = box
        y_ref = min(by2, frame_h - 1)
        xL, xR = path_bounds_at(y_ref, path)
        if xL is None or xR - xL < 1:
            continue
        s = (min(bx2, xR) - max(bx1, xL)) / (xR - xL)
        s = min(s, 1.0)
        s_vals.append((t, s))
        rows = sorted(path.keys())
        row_lo, row_hi = rows[0], rows[-1]  # row_lo = nearest (bottom), row_hi = farthest traced
        rho = float(np.clip((y_ref - row_hi) / max(row_lo - row_hi, 1.0), 0, 1))
        rho_vals.append(rho)
        cx = (bx1 + bx2) / 2
        lane_states.append("IN" if s > 0 else ("LEFT" if cx < (xL + xR) / 2 else "RIGHT"))

    if not s_vals:
        return dict(alpha=round(float(alpha), 4), alpha_mode=alpha_mode, s_end=None, s_rate=None,
                   rho=None, lane_state="INVALID", n_samples=len(pts), fit_span_used=round(span, 2),
                   low_samples=low_samples, last_box=[float(v) for v in pts[-1][1]], last_t=float(ts[-1]))

    s_ts = np.array([t for t, _ in s_vals])
    s_only = np.array([s for _, s in s_vals])
    s_end = float(s_only[-1])
    s_rate = 0.0 if len(set(s_ts)) < 2 else _lstsq_slope(s_ts, s_only)
    rho_end = float(rho_vals[-1])
    lane_state = lane_states[-1]

    return dict(alpha=round(float(alpha), 4), alpha_mode=alpha_mode,
               s_end=round(s_end, 4), s_rate=round(float(s_rate), 4),
               rho=round(rho_end, 4), lane_state=lane_state,
               n_samples=len(pts), fit_span_used=round(span, 2), low_samples=low_samples,
               last_box=[float(v) for v in pts[-1][1]], last_t=float(ts[-1]))
