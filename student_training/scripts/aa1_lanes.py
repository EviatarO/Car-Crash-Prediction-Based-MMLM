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
FLICKER_SMOOTH_N = 5  # majority-vote window for the IN/LEFT/RIGHT label

# Ego path (estimate_ego_path). Pixel thresholds are for 1280x720 frames.
PATH_ROWS = np.arange(int(0.2 * FRAME_H), FRAME_H, 8, dtype=np.float64)  # rows the path is stored at
PATH_MAX_CURVATURE = 120.0  # px, quadratic term in normalised rows; beyond this fit a straight line
PATH_ANCHOR_WEIGHT = 5.0    # weight of the bottom-row anchor in the centre-path fit
PATH_EMA_TAU_S = 0.25
PATH_GATE_PX = 120          # mean |new - current| over the lower half of the image
PATH_RESYNC_S = 0.4         # accept a far measurement once rejections last this long (a turn)
PATH_RESET_HOLD_S = 0.8     # after this long fully held, retry against the calibrated anchor
                            # instead of only ever searching near the (possibly stuck) reference
PAIR_MIN_FRAMES = 8         # frames with both ego lane lines needed for the anchor + lane width
MEASURE_MIN_INL = 6         # RANSAC inliers needed to accept ONE lane line in a single frame
                            # (lower than the calibration pass - near an intersection the mask
                            # is often sparse; the post-fit width check still guards this)
MEASURE_MIN_SPAN = 40       # px of row range an accepted single-frame line must cover


def _lstsq_slope(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def fit_alpha(track, t_window_end, frame_w: int = FRAME_W, frame_h: int = FRAME_H,
             fit_span: float = FIT_SPAN):
    """Causal log-size growth rate (looming), independent of any lane/path geometry - factored
    out 2026-09-18 so the Stage 3 selection score (aa1_collision.py) can use it without also
    computing the lane-overlap fields below. `track`: [(t, [x1,y1,x2,y2], conf)], any causal
    prefix (points after t_window_end are ignored). Expands the 1s fit window up to
    MAX_FIT_SPAN if fewer than MIN_FIT_SAMPLES points are found. Returns
    (alpha, alpha_mode, pts, low_samples, span_used) or None if fewer than 2 points exist.
    alpha_mode picks the size measure by whichever box edges stay clean of the frame border
    this window: area if none of the 4 edges are clipped, else width (top/bottom clipped only),
    height (left/right clipped only), or area_lowconf (both directions clipped)."""
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
    return float(alpha), alpha_mode, pts, low_samples, round(span, 2)


STATIC_LANE_FRAC = 0.95  # a pixel classified 'lane' in at least this fraction of the WHOLE
                         # clip is a fixed part of the vehicle/mount (a real lane line always
                         # moves at least a little as the ego drives), not road paint


def static_lane_mask(frame_masks: dict, ts: list) -> np.ndarray:
    """Pixels that read as 'lane' in >= STATIC_LANE_FRAC of every frame in the clip - almost
    certainly a fixed object visible in the shot (e.g. a spare tire mounted on the ego's own
    bumper), which YOLOPv2 sometimes misclassifies as lane paint because it never moves the way
    a real line does. Confirmed 2026-09-19 on 00486/00932: a bright/patterned mounted object at
    the bottom of frame was lane-classified in 100% of frames, fooling the line fit into
    tracking it instead of the road. 4 known-good clips checked (01153/01504/02117/01737) have
    ZERO pixels this persistent, so the cutoff has margin to spare."""
    if not ts:
        return np.zeros((FRAME_H, FRAME_W), dtype=bool)
    acc = np.zeros_like(frame_masks[ts[0]]["lane"], dtype=np.int32)
    for t in ts:
        acc += frame_masks[t]["lane"]
    return (acc / len(ts)) >= STATIC_LANE_FRAC


def _exclude_vehicles(lane: np.ndarray, boxes) -> np.ndarray:
    """Zero out lane pixels that fall inside a detected vehicle's box - a real lane line can't
    be painted under a car, so any 'lane' pixels there are a false positive (usually headlight
    glare or a reflection on the vehicle's own body). Confirmed 2026-09-19 on 00932: every
    frame's fitted 'lane line' was tracking the glare on a van's grille/headlights directly
    ahead, not the road, because YOLOPv2's lane mask lit up on the reflections."""
    if len(boxes) == 0:
        return lane
    out = lane.copy()
    h, w = lane.shape
    for x1, y1, x2, y2 in boxes:
        x1, x2 = int(max(0, x1)), int(min(w, x2))
        y1, y2 = int(max(0, y1)), int(min(h, y2))
        if x2 > x1 and y2 > y1:
            out[y1:y2, x1:x2] = False
    return out


def _nearest_lane_points(lane, xref_fn, step=4, max_run=60):
    """Row scan outward from the reference path: nearest lane pixel on each side, per row.
    Rows where a lane run crosses the reference (a stripe) or the run is wider than a lane
    line are skipped."""
    left, right = [], []
    for y in range(FRAME_H - 1, int(0.2 * FRAME_H), -step):
        xr = int(np.clip(xref_fn(y), 0, FRAME_W - 1))
        row = lane[y]
        if row[xr]:
            continue
        xs = np.flatnonzero(row)
        lo, hi = xs[xs < xr], xs[xs > xr]
        if lo.size:
            k = lo.size - 1
            while k > 0 and lo[k - 1] == lo[k] - 1:
                k -= 1
            if lo[-1] - lo[k] + 1 <= max_run:
                left.append((y, float(lo[-1])))
        if hi.size:
            k = 0
            while k < hi.size - 1 and hi[k + 1] == hi[k] + 1:
                k += 1
            if hi[k] - hi[0] + 1 <= max_run:
                right.append((y, float(hi[0])))
    return left, right


def _ransac_line(pts, rng, iters=150, thr=6.0, min_inl=10, min_span=60):
    """x = m*y + c fit to (y, x) points -> (m, c, y_min, y_max), or None if too few inliers or
    too short in y."""
    if len(pts) < min_inl:
        return None
    P = np.array(pts)
    ys, xs = P[:, 0], P[:, 1]
    i, j = rng.integers(0, len(P), iters), rng.integers(0, len(P), iters)
    ok = ys[i] != ys[j]
    i, j = i[ok], j[ok]
    if not len(i):
        return None
    m = (xs[j] - xs[i]) / (ys[j] - ys[i])
    c = xs[i] - m * ys[i]
    inl = np.abs(xs[None] - (m[:, None] * ys[None] + c[:, None])) < thr
    best = inl[int(np.argmax(inl.sum(1)))]
    if best.sum() < min_inl or ys[best].max() - ys[best].min() < min_span:
        return None
    m, c = np.polyfit(ys[best], xs[best], 1)
    return float(m), float(c), float(ys[best].min()), float(ys[best].max())


def _u(y):
    return (np.asarray(y, dtype=np.float64) - FRAME_H / 2) / (FRAME_H / 2)


def _ransac_curve(pts, rng, iters=200, thr=6.0, min_inl=10, min_span=60):
    """x = a*u^2 + b*u + c in normalised row u, fit to (y, x) points - follows a curving lane
    line (01737, 00687 turns). Curvature is capped; falls back to a straight fit on the same
    inliers. Returns (coef, y_min, y_max) or None."""
    if len(pts) < min_inl:
        return None
    P = np.array(pts)
    ys, xs = P[:, 0], P[:, 1]
    u = _u(ys)
    idx = rng.integers(0, len(P), (iters, 3))
    Y = ys[idx]
    ok = ((np.abs(Y[:, 0] - Y[:, 1]) >= 8) & (np.abs(Y[:, 0] - Y[:, 2]) >= 8)
          & (np.abs(Y[:, 1] - Y[:, 2]) >= 8))
    idx = idx[ok]
    if not len(idx):
        return None
    U = u[idx]
    A = np.stack([U ** 2, U, np.ones_like(U)], -1)
    coef = np.linalg.solve(A, xs[idx][..., None])[..., 0]
    pred = coef[:, 0:1] * u[None] ** 2 + coef[:, 1:2] * u[None] + coef[:, 2:3]
    inl = np.abs(xs[None] - pred) < thr
    best = inl[int(np.argmax(inl.sum(1)))]
    if best.sum() < min_inl or ys[best].max() - ys[best].min() < min_span:
        return None
    c = np.polyfit(u[best], xs[best], 2)
    if abs(c[0]) > PATH_MAX_CURVATURE:
        c = np.concatenate([[0.0], np.polyfit(u[best], xs[best], 1)])
    return c, float(ys[best].min()), float(ys[best].max())


def _curve_x(fit, y):
    return np.polyval(fit[0], _u(y))


def _calibrate_lanes(frame_masks: dict, ts: list, rng):
    """Anchor (where the ego lane's centre meets the bottom row - the camera is fixed to the
    car, so one value per clip) and a lane-width model w(y) = p*y + q in pixels, which lets a
    single visible line stand in for the pair.

    Fixed reference at the image centre (a chained/adaptive reference was tried and reverted
    2026-09-18: it fixed the one clip it targeted, an off-centre highway, but broke several
    that already worked well with a fixed reference - the reference's shared random generator
    state was consumed a different number of times depending on how many points were found each
    frame, so a small reference change early on desynchronised RANSAC's later draws from the
    fixed-reference run, with no clear win to justify the risk).

    Curve fits, not straight lines (2026-09-19; straight lines were used until then). A clip
    whose road visibly converges/funnels (00283) has real curvature even close to the ego, and
    a straight fit through it can miss lines a curve fit finds cleanly. The fit is NOT
    extrapolated to the bottom row for the anchor - only evaluated up to its own fitted range
    (`hi`, the nearest fitted row) - because a quadratic extrapolated much beyond its own data
    is unstable (this lane mask rarely reaches within ~100px of the frame bottom on any clip
    checked; evaluating a fit whose own range tops out at row ~600 all the way out to row 719
    gave wildly wrong anchors on 01153). The anchor is therefore "where the ego lane centres at
    the nearest row the data actually reaches", not literally the bottom row - close enough for
    a per-clip constant, and it is what _fit_centreline pulls every frame's path through."""
    anchors, samples = [], []
    for t in ts:
        lane = _exclude_vehicles(frame_masks[t]["lane"], frame_masks[t]["boxes"])
        left, right = _nearest_lane_points(lane, lambda y: FRAME_W / 2)
        fl, fr = _ransac_curve(left, rng), _ransac_curve(right, rng)
        if fl is None or fr is None:
            continue
        lo, hi = max(fl[1], fr[1]), min(fl[2], fr[2])
        if hi - lo < 40:
            continue
        rows = np.arange(lo, hi + 1, 8.0)
        w = _curve_x(fr, rows) - _curve_x(fl, rows)
        if np.any(w <= 20) or w[-1] <= w[0]:
            continue  # lane must widen toward the ego
        anchor = (_curve_x(fl, hi) + _curve_x(fr, hi)) / 2
        if not 0.25 * FRAME_W < anchor < 0.75 * FRAME_W:
            continue
        anchors.append(anchor)
        samples.extend(zip(rows, w))
    width_model = None
    if len(anchors) >= PAIR_MIN_FRAMES:
        S = np.array(samples)
        p = np.polyfit(S[:, 0], S[:, 1], 1)
        keep = np.abs(S[:, 1] - np.polyval(p, S[:, 0])) <= 0.3 * np.maximum(np.polyval(p, S[:, 0]), 1.0)
        p = np.polyfit(S[keep, 0], S[keep, 1], 1)
        if p[0] > 0:
            width_model = (float(p[0]), float(p[1]))
    anchor = float(np.median(anchors)) if len(anchors) >= PAIR_MIN_FRAMES else FRAME_W / 2
    return anchor, width_model, len(anchors)


def _lane_width(model, y):
    return model[0] * np.asarray(y, dtype=np.float64) + model[1]


def _fit_centreline(samples, anchor, deg):
    """Centre path through (row, x) samples plus the anchor at the bottom row (weighted), so the
    hood rows - where no lane pixel is visible - are interpolated toward the ego, not
    extrapolated. Returns (coef, top): coef is [a2, a1, a0] for x = polyval(coef, u(y))
    (a2 = 0 for a straight fit), valid at every y, not just the sampled rows - see path_x_at
    and estimate_ego_path's EMA, which blends coef directly for exactly this reason."""
    S = np.array(samples)
    ys = np.concatenate([S[:, 0], [FRAME_H - 1.0]])
    xs = np.concatenate([S[:, 1], [anchor]])
    wts = np.concatenate([np.ones(len(S)), [PATH_ANCHOR_WEIGHT]])
    if deg == 2 and S[:, 0].max() - S[:, 0].min() < 200:
        deg = 1
    c = np.polyfit(_u(ys), xs, deg, w=wts)
    if deg == 2 and abs(c[0]) > PATH_MAX_CURVATURE:
        c = np.polyfit(_u(ys), xs, 1, w=wts)
    if len(c) == 2:
        c = np.concatenate([[0.0], c])
    top = float(S[:, 0].min())
    return c, top


def _measure_path(fm, xref_fn, anchor, model, rng):
    """(x over PATH_ROWS, top row, source) for one frame, or None. Priority: both ego lane lines
    -> centre between them; one line -> that line shifted by half a lane width. With no lane
    line the frame has no measurement (the path is held). A green-corridor fallback was tried
    and dropped (2026-09-18): with a car directly ahead - the rear-end setup - the free green
    area is the lane beside it, so the path bent away from the car being hit (00493, 00529).

    2026-09-19: dropped the lane-width band that used to pre-filter `left`/`right` to points
    near the current reference. It discarded a real, usable, visible line whenever the
    reference itself had already drifted even a little - exactly backwards from what a
    reference is for. The post-fit width-plausibility check a few lines down (w vs the learned
    lane-width model) already rejects an implausible pair after fitting, which is the right
    place for that check. Also lowered the RANSAC bar for a single line (MEASURE_MIN_INL/SPAN):
    near an intersection the lane mask is often genuinely sparse - a handful of real points is
    still better than nothing, and the width/plausibility checks still guard the result."""
    lane = _exclude_vehicles(fm["lane"], fm["boxes"])
    left, right = _nearest_lane_points(lane, xref_fn)
    fl = _ransac_curve(left, rng, min_inl=MEASURE_MIN_INL, min_span=MEASURE_MIN_SPAN)
    fr = _ransac_curve(right, rng, min_inl=MEASURE_MIN_INL, min_span=MEASURE_MIN_SPAN)
    if fl is not None and fr is not None:
        lo, hi = max(fl[1], fr[1]), min(fl[2], fr[2])
        rows = PATH_ROWS[(PATH_ROWS >= lo) & (PATH_ROWS <= hi)]
        if len(rows) >= 5:
            xl, xr = _curve_x(fl, rows), _curve_x(fr, rows)
            w = xr - xl
            ok = w > 20
            if model is not None:
                wm = _lane_width(model, rows)
                ok &= (w >= 0.6 * wm) & (w <= 1.5 * wm)
            if ok.mean() >= 0.7:
                x, top = _fit_centreline(list(zip(rows[ok], ((xl + xr) / 2)[ok])), anchor, 2)
                return x, top, "lane_pair"
    if model is not None:
        best = None
        for fit, sign in ((fl, 1.0), (fr, -1.0)):
            if fit is None:
                continue
            rows = PATH_ROWS[(PATH_ROWS >= fit[1]) & (PATH_ROWS <= fit[2])]
            if len(rows) >= 5 and (best is None or len(rows) > len(best[0])):
                best = (rows, _curve_x(fit, rows) + sign * _lane_width(model, rows) / 2)
        if best is not None:
            x, top = _fit_centreline(list(zip(*best)), anchor, 2)
            return x, top, "lane_single"
    return None


def path_x_at(ego_path: dict, y) -> float:
    """x of the ego path at image row y; rows above the path's top use the top row's x."""
    coef, top = ego_path["coef"], ego_path["top"]
    y = np.clip(y, top, FRAME_H - 1)
    return float(np.polyval(coef, _u(y)))


def estimate_ego_path(frame_masks: dict, seed: int = 0) -> dict:
    """The ego's own path per frame, from the lane-line and drivable masks. Writes
    frame_masks[t]["ego_path"] = dict(coef=[a2,a1,a0], top=<highest row>, src=...) (evaluate
    with path_x_at) and returns a per-clip summary.

    Per frame (_measure_path) the nearest lane line on each side of last frame's path is found
    row by row and fitted as a curve; the path is the centre between the two lines, or one line
    shifted by half a lane width (lane width learned in this clip). The path always passes
    through the clip's anchor at the bottom row (the camera is fixed to the car).

    Smoothing is EMA (PATH_EMA_TAU_S) on the fitted curve's OWN COEFFICIENTS, not on x values
    at each stored row independently (tried first, 2026-09-18): blending per row left a visible
    kink exactly where a row's status flipped between "seen in both the old and new curve" (EMA
    blend) and "newly in range" (took the new value outright, no blend) - confirmed on 00077, a
    30px jump between two rows 8px apart. Blending the 3 coefficients has no such boundary,
    because both curves are defined at every row already.

    A measurement far from the current path is rejected unless rejections last PATH_RESYNC_S (a
    real turn). With no lane lines at all, the path HOLDS its last known shape rather than
    fading toward a straight-ahead default (tried 2026-09-18, reverted 2026-09-19): the fade
    target had no connection to where the road actually goes, so it read as the line "losing
    direction and pointing at the sky" whenever a hold ran long - actively wrong is worse than
    stale. Before the first measurement, frames get the first measured path (camera calibration,
    not object motion); a clip with no measurement at all gets a vertical path above the anchor.

    2026-09-19: two more fixes, both from clips where the path never recovered for the rest of
    the clip once wrong -
    - **The first-ever accepted measurement is now required to be a full `lane_pair` fit**, not
      any single-line guess. It seeds every later frame's reference (and gets literally
      backfilled onto every frame before it), so a noisy first guess propagated indefinitely:
      confirmed on 00486/00505/01532/01478, where an early single-line fit (grabbed a spare
      tire, glare, or a crosswalk stripe) locked in a wrong reference for the whole clip.
    - **After PATH_RESET_HOLD_S of continuous "held" (no measurement found near the current,
      possibly-stale, reference)**, retry once against the calibrated anchor - a fixed,
      independent reference - instead of only ever searching near the position that got stuck
      in the first place. Only adopts the retry if it comes back a full `lane_pair` (same bar as
      the seed). Confirmed needed on 01075 and 01532: real, clearly-visible lines existed for
      the rest of the clip, but the reference-based search anchored to the wrong spot never
      found its way back to them on its own.

    Replaces (2026-09-18) a single straight line to one vanishing point: it started at a fixed
    default in the sky (00372), froze during turns when lane lines went near-horizontal (01737),
    and needed lines on both sides of the ego (00687 has them on one side only)."""
    rng = np.random.default_rng(seed)
    ts = sorted(frame_masks)
    if not ts:
        return {}
    static = static_lane_mask(frame_masks, ts)
    if static.any():
        for t in ts:
            frame_masks[t]["lane"] = frame_masks[t]["lane"] & ~static
    anchor, model, n_pair_frames = _calibrate_lanes(frame_masks, ts, rng)
    sm_coef, sm_top = None, None
    pending, prev_t, rejected_since, held_since = [], None, None, None
    srcs = []
    for t in ts:
        if sm_coef is None:
            xref_fn = lambda y: anchor
        else:
            ep = dict(coef=sm_coef, top=sm_top)
            xref_fn = lambda y, ep=ep: path_x_at(ep, y)
        meas = _measure_path(frame_masks[t], xref_fn, anchor, model, rng)
        a = 1.0 - np.exp(-(t - prev_t) / PATH_EMA_TAU_S) if prev_t is not None and t > prev_t else 1.0
        prev_t = t
        src = "held"
        if meas is not None and (sm_coef is not None or meas[2] == "lane_pair"):
            mcoef, mtop, msrc = meas
            if sm_coef is None:
                sm_coef, sm_top, src = mcoef.copy(), mtop, msrc
            else:
                ys = PATH_ROWS[PATH_ROWS >= FRAME_H / 2]
                diff = float(np.mean(np.abs(np.polyval(mcoef, _u(ys)) - np.polyval(sm_coef, _u(ys)))))
                if diff <= PATH_GATE_PX or (rejected_since is not None and t - rejected_since >= PATH_RESYNC_S):
                    sm_coef = sm_coef + a * (mcoef - sm_coef)
                    sm_top = sm_top + a * (mtop - sm_top)
                    rejected_since, src = None, msrc
                elif rejected_since is None:
                    rejected_since = t
        if src == "held":
            if held_since is None:
                held_since = t
            elif sm_coef is not None and t - held_since >= PATH_RESET_HOLD_S:
                retry = _measure_path(frame_masks[t], lambda y: anchor, anchor, model, rng)
                if retry is not None and retry[2] == "lane_pair":
                    sm_coef, sm_top, src = retry[0].copy(), retry[1], retry[2]
                    held_since = rejected_since = None
        else:
            held_since = None
        if sm_coef is None:
            pending.append(t)
            continue
        for tp in pending:
            frame_masks[tp]["ego_path"] = dict(coef=sm_coef.copy(), top=float(sm_top), src="backfilled")
            srcs.append("backfilled")
        pending = []
        frame_masks[t]["ego_path"] = dict(coef=sm_coef.copy(), top=float(sm_top), src=src)
        srcs.append(src)
    if pending:
        straight = np.array([0.0, 0.0, anchor])
        for tp in pending:
            frame_masks[tp]["ego_path"] = dict(coef=straight.copy(), top=FRAME_H / 2, src="default")
            srcs.append("default")
    return dict(anchor=anchor, n_pair_frames=n_pair_frames, lane_width_model=model,
                sources={s: srcs.count(s) for s in set(srcs)}, n_frames=len(ts))


def track_geometry_v2(track, t_window_end, frame_masks: dict, frame_w: int = FRAME_W,
                      frame_h: int = FRAME_H, fit_span: float = FIT_SPAN):
    """`track`: causal [(t, [x1,y1,x2,y2], conf)]. `frame_masks`: t -> dict(drivable, lane,
    boxes) for every frame in the fit window (decoded once by the caller, reused across
    objects/windows - decoding is not free). Returns None if too few points.

    NOTE 2026-09-18: the selection score (aa1_collision.py) no longer uses s/s_rate/rho from
    this function - see fit_heading_line/side_gap. This function (and its lane-overlap fields)
    is kept only so the per-clip geometry cache still has them for comparison."""
    fit = fit_alpha(track, t_window_end, frame_w, frame_h, fit_span)
    if fit is None:
        return None
    alpha, alpha_mode, pts, low_samples, span = fit
    ts = np.array([p[0] for p in pts])
    boxes = np.array([p[1] for p in pts], dtype=np.float64)

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
        # row values are pixel rows (y), sorted ascending: rows[0] is the smallest y = topmost
        # = FARTHEST traced row; rows[-1] is the largest y = bottom-most = NEAREST. BUG FIXED
        # 2026-09-17: the old code labelled rows[0] "nearest" and rows[-1] "farthest" (backwards),
        # so row_lo - row_hi was always <= 0 and max(..., 1.0) always picked 1.0 - rho was never
        # a real 0..1 position, only ever exactly 0 or 1 (confirmed: 982/982 object-windows).
        row_far, row_near = rows[0], rows[-1]
        rho = float(np.clip((y_ref - row_far) / max(row_near - row_far, 1.0), 0, 1))
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
    # Majority vote over the last FLICKER_SMOOTH_N samples, not just the literal last one - a
    # single-frame IN/LEFT/RIGHT flip near the lane boundary (path-width noise) shouldn't flip
    # the label; ties break toward the most recent sample. Confirmed 2026-09-17 on 00529: s_end
    # crossing 0 for one sample flipped lane_state and dropped the object's score mid-approach.
    recent = lane_states[-FLICKER_SMOOTH_N:]
    counts = {s: recent.count(s) for s in set(recent)}
    top = max(counts.values())
    tied = {s for s, c in counts.items() if c == top}
    lane_state = next(s for s in reversed(recent) if s in tied)

    return dict(alpha=round(float(alpha), 4), alpha_mode=alpha_mode,
               s_end=round(s_end, 4), s_rate=round(float(s_rate), 4),
               rho=round(rho_end, 4), lane_state=lane_state,
               n_samples=len(pts), fit_span_used=round(span, 2), low_samples=low_samples,
               last_box=[float(v) for v in pts[-1][1]], last_t=float(ts[-1]))
