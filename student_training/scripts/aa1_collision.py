"""
aa1_collision.py
=================
Stage 3 (child plan 2026-09-15-AA1-v2b): the object-selection score used to pick the K=5
objects supervised per window. NOT a learned threat, and NEVER fit to the clip crash label
(father plan AA invariant) - it only has to reliably keep the true crash partner inside the
top-5 candidates; once selected, all K get flat (equal) auxiliary-loss weight.

2026-09-18 rewrite, from a second round of user review of the overlays (00147/00283/00319/
00474/00529/00687/01504/02104), on top of the 2026-09-17 rework (path_relevance + fixed rho).
That version still measured lane overlap `s` at the object's OWN row against the traced path's
WIDTH there, which:
- jitters violently frame to frame when the traced row is narrow (00283 id2: score flipping
  0 <-> 2.6 as `s` crossed the hard lane-edge cut)
- explodes when a close object covers the road, collapsing the path width to a few px
  (01504 id70: s_rate as low as -127)
- reads as "in lane" for any box inside a wide drivable area when no lane lines exist, even two
  lanes over (00529 id9)
- gives almost no credit to raw closeness (00474: a 400-600px van scored ~0 for most of the
  clip because it sat beside, not "in", the traced path)

This version drops path-width entirely. Checked first (2026-09-17) that box height alone
already puts the true partner in the top-5 in 23/23 windows where visible across the 9
positive clips (14/23 at #1) - closeness is most of the signal. Three measurements combine it
with lateral position and closing rate, and the whole thing is smoothed over time:

- **Closeness**: box height / CLOSE_DIV, capped at 1.0.
- **Side**: horizontal gap from the object's bottom edge to the ego path at that row
  (aa1_lanes.estimate_ego_path: centre of the ego's own lane lines), in units of the object's
  OWN box height. gap_px/height_px = gap_real/height_real at any distance (both scale by
  focal_length/distance the same way), so this needs no lane width and no "is this row in the
  lane" test - and it still gives partial credit to an object beside the ego, unlike a hard
  in/out lane cut.
- **Approach**: the existing looming rate (log-size slope, aa1_lanes.fit_alpha), clipped to
  0..1.
- **Shielding**: unchanged - a farther box covered horizontally by a nearer one is discounted,
  on the assumption it is sitting behind it on the same sightline.
- **EMA smoothing** (tau=0.3s) over each track's own causal score sequence removes the
  frame-to-frame flips the previous version showed.
- **Candidate pool**: recent (>=half the last 0.5s) + at least MIN_BOX_H tall + only the
  MAX_CANDIDATES tallest. Small/far objects never reach scoring at all (01504: 55% of the old
  top-5 was boxes <40px tall).

s/s_rate/rho/lane_state (aa1_lanes.track_geometry_v2) are still computed and saved to
<vid>_geometry_v2.json for comparison, but no longer feed this score.
"""
from __future__ import annotations

import numpy as np

from aa1_lanes import path_x_at

FRAME_W, FRAME_H = 1280, 720

MIN_BOX_H = 25          # px; below this, not a real candidate at all - 00283's crash truck is
                        # 30px at TTE1.5, the smallest partner box across the 9 positive clips
                        # checked, so this must stay below that
MAX_CANDIDATES = 8      # only the N tallest recent boxes are even scored, per frame/window

RECENCY_WINDOW_S = 0.5
RECENCY_MIN_FRAC = 0.5

CLOSE_DIV = 360.0       # half the frame height (720); not a max size - box_h/CLOSE_DIV just
                        # reaches 1.0 around a 360px box and keeps growing slower after that
ALPHA_CAP = 1.0         # /s; Approach = clip(alpha, 0, ALPHA_CAP)

SIDE_FULL_GAP = 0.3     # side gap (in box heights) within which Side = 1.0
SIDE_ZERO_GAP = 1.5     # side gap at/beyond which Side floors (~2 m for a car-height ruler)
SIDE_FLOOR = 0.15       # a big/close object beside the ego stays in the ranking, but a car
                        # 1.4 box-heights aside no longer ties one on the path (00283 id1 vs id2)
SIDE_TREND_HORIZON_S = 1.0  # project the side gap this far ahead using its own recent trend

EMA_TAU_S = 0.3         # score smoothing time constant

SHIELD_DISCOUNT = 0.25
SHIELD_X_IOU_MIN = 0.35
SHIELD_Y_MARGIN_PX = 10


def _lstsq_slope(t, y):
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.stack([t, np.ones_like(t)], axis=1)
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def side_gap(box, ego_path) -> float:
    """Horizontal gap from the box's bottom edge to the ego path at that row, in units of the
    box's OWN height. 0 if the path passes through the box's x-range. `ego_path` comes from
    aa1_lanes.estimate_ego_path."""
    x1, y1, x2, y2 = box
    h = max(y2 - y1, 1.0)
    x_line = FRAME_W / 2.0 if ego_path is None else path_x_at(ego_path, y2)
    if x1 <= x_line <= x2:
        gap_px = 0.0
    elif x1 > x_line:
        gap_px = x1 - x_line
    else:
        gap_px = x_line - x2
    return gap_px / h


def lane_side(box, ego_path) -> str:
    """LEFT / EGO / RIGHT of the ego path, from the box's bottom edge at that row - the
    categorical counterpart to side_gap, for the target-curve plots (aa1_stage3.py)."""
    x1, y1, x2, y2 = box
    x_line = FRAME_W / 2.0 if ego_path is None else path_x_at(ego_path, y2)
    if x1 <= x_line <= x2:
        return "EGO"
    return "LEFT" if x2 < x_line else "RIGHT"


def side_score(g_eff: float) -> float:
    """1.0 within SIDE_FULL_GAP box-heights of the ego path, floors at SIDE_FLOOR beyond
    SIDE_ZERO_GAP, linear in between."""
    if g_eff <= SIDE_FULL_GAP:
        return 1.0
    if g_eff >= SIDE_ZERO_GAP:
        return SIDE_FLOOR
    frac = (g_eff - SIDE_FULL_GAP) / (SIDE_ZERO_GAP - SIDE_FULL_GAP)
    return 1.0 - (1.0 - SIDE_FLOOR) * frac


def _shielded(tid, box, boxes: dict) -> bool:
    """True if another tracked box in this frame is both nearer (larger y2 = lower in frame,
    closer to the ego) and overlaps this box's x-range: this box is very likely sitting behind
    that one on the same sight line, not an independent object."""
    bx1, by1, bx2, by2 = box
    for otid, obox in boxes.items():
        if otid == tid:
            continue
        ox1, oy1, ox2, oy2 = obox
        if oy2 <= by2 + SHIELD_Y_MARGIN_PX:
            continue  # not nearer
        ix1, ix2 = max(bx1, ox1), min(bx2, ox2)
        iou_x = max(0.0, ix2 - ix1) / max(min(bx2 - bx1, ox2 - ox1), 1.0)
        if iou_x >= SHIELD_X_IOU_MIN:
            return True
    return False


def recency_ok(pts, t_end: float, fps: float) -> bool:
    """Present in >= half of the last RECENCY_WINDOW_S seconds, causally up to t_end."""
    lo = t_end - RECENCY_WINDOW_S
    n_in = sum(1 for t, _, _ in pts if lo - 1e-6 <= t <= t_end + 1e-6)
    need = max(1, round(RECENCY_WINDOW_S * fps * RECENCY_MIN_FRAC))
    return n_in >= need


def candidate_pool(boxes_now: dict, tracks: dict, t: float, fps: float) -> dict:
    """{tid: box} for tracks that are recent, tall enough, and among the MAX_CANDIDATES
    tallest - everything else never reaches scoring at all."""
    pool = {}
    for tid, box in boxes_now.items():
        if box[3] - box[1] < MIN_BOX_H:
            continue
        causal = [p for p in tracks.get(tid, []) if p[0] <= t + 1e-6]
        if not causal or not recency_ok(causal, t, fps):
            continue
        pool[tid] = box
    if len(pool) > MAX_CANDIDATES:
        pool = dict(sorted(pool.items(), key=lambda kv: -(kv[1][3] - kv[1][1]))[:MAX_CANDIDATES])
    return pool


def compute_track_scores(tracks: dict, frame_masks: dict, fps: float) -> dict:
    """One causal forward pass, in time order, over every (track, detection) event in the
    clip - not per-window, per-object recomputation. Returns
    {track_id: {t: dict(score, raw, closeness, side, approach, alpha, g, g_rate, lane, shielded,
    box_h)}}, where `score` is the EMA-smoothed value and `raw` is the unsmoothed one at that
    instant. `alpha`/`g`/`g_rate`/`lane` double as the Stage 4 target-curve series
    (aa1_stage3.render_target_curves plots them directly, no recomputation). Shielding needs
    every track's box at the SAME frame, so this is driven by a merged, time-sorted event list
    across all tracks rather than processed track-by-track."""
    from aa1_lanes import fit_alpha

    by_t: dict[float, dict] = {}
    for tid, pts in tracks.items():
        for t, box, _ in pts:
            by_t.setdefault(round(t, 4), {})[tid] = box

    g_history: dict[int, list] = {}   # tid -> [(t, g), ...] within the last horizon
    ema: dict[int, tuple] = {}        # tid -> (last_t, smoothed_score)
    out: dict[int, dict] = {tid: {} for tid in tracks}

    for t in sorted(by_t):
        boxes_now = by_t[t]
        fm = frame_masks.get(t)
        ego_path = fm["ego_path"] if fm is not None else None
        for tid, box in boxes_now.items():
            causal = [p for p in tracks[tid] if p[0] <= t + 1e-6]
            fit = fit_alpha(causal, t)
            alpha = fit[0] if fit is not None else 0.0
            approach = min(max(alpha, 0.0), ALPHA_CAP)

            box_h = max(box[3] - box[1], 1.0)
            closeness = min(box_h / CLOSE_DIV, 1.0)

            g = side_gap(box, ego_path)
            hist = [p for p in g_history.get(tid, []) if p[0] >= t - SIDE_TREND_HORIZON_S]
            hist.append((t, g))
            g_history[tid] = hist
            g_rate = 0.0 if len({p[0] for p in hist}) < 2 else _lstsq_slope(
                [p[0] for p in hist], [p[1] for p in hist])
            g_eff = min(g, max(g + g_rate * SIDE_TREND_HORIZON_S, 0.0))
            side = side_score(g_eff)

            raw = closeness * side * (1.0 + approach)
            shielded = _shielded(tid, box, boxes_now)
            if shielded:
                raw *= SHIELD_DISCOUNT

            prev = ema.get(tid)
            if prev is None or t <= prev[0]:
                sm = raw
            else:
                a = 1.0 - np.exp(-(t - prev[0]) / EMA_TAU_S)
                sm = a * raw + (1.0 - a) * prev[1]
            ema[tid] = (t, sm)

            out[tid][t] = dict(score=round(float(sm), 4), raw=round(float(raw), 4),
                               closeness=round(float(closeness), 4), side=round(float(side), 4),
                               approach=round(float(approach), 4), alpha=round(float(alpha), 4),
                               g=round(float(g), 4), g_rate=round(float(g_rate), 4),
                               lane=lane_side(box, ego_path),
                               shielded=bool(shielded), box_h=round(float(box_h), 1))
    return out


def select_top_k(objects_now: dict, scores_now: dict, k: int = 5) -> list[tuple[int, float]]:
    """`objects_now`: {tid: box}, already filtered by candidate_pool. `scores_now`: {tid:
    score dict} from compute_track_scores at the relevant t. Same procedure for positives and
    negatives - no label is read here (father plan AA invariant)."""
    scored = [(tid, scores_now[tid]["score"]) for tid in objects_now if tid in scores_now]
    scored.sort(key=lambda p: -p[1])
    return scored[:k]
