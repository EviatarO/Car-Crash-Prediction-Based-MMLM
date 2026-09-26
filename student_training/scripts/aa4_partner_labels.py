"""
aa4_partner_labels.py
======================
Stage AA-H, Stage 0.3 (child plan 2026-09-24-AA-H-head-attention-supervision): the L3
hindsight crash-partner label - for each POSITIVE video, WHICH tracked object (from
AA.2/AA.4's own stitched tracks) is the one the ego actually hits. Used by
semsup_train.py's --aux-label partner_pos so the aux loss's "relevant" set means THIS car
caused the crash, not merely a nearby/closing one (aa4_token_labels.py's `rel` measures the
latter - see the plan's "Are we measuring the right thing?" section for why that distinction
matters for the gradcam loss's FP risk in particular).

WHY A NEW DETECTION PASS. AA.2's cached span for a positive video stops at
t_event - POS_SPAN_END_BEFORE_EVENT (0.5s, aa1_detect_track_rank.py) - BADAS's own windows
never look closer to the event than that, so there is no cached detection of the vehicle at
the moment it actually becomes the crash partner. This script decodes a SHORT extension
[t_event - POS_SPAN_END_BEFORE_EVENT - EXT_OVERLAP, t_event + POST_EVENT] with the SAME
detector AA.2 used (YOLOPv2, aa1_yolop_cache.py), tracks it with a MINIMAL greedy IoU tracker
(not BoT-SORT - the extension is 15-40 frames and does not need re-identification across
occlusion gaps the way a full clip does), and re-identifies each extension track against
AA.2's EXISTING stitched tracks by IoU at the extension's first frame (which overlaps the
cached span by EXT_OVERLAP seconds). The partner is picked from RE-IDENTIFIED tracks only, so
it always maps back to a tid that already has boxes in the video's own windows
(aa4_token_labels.py's tid_pts_by_t) - a track that only appears during the extension itself
can never be a partner label, by design (Stage 0 scope: see DECISIONS.md if this needs
revisiting).

Partner selection, among re-identified tracks still visible in the extension's last few
frames: largest box height, secondarily closest to the ego path (aa1_lanes.path_x_at,
evaluated with the LAST ego_path estimated from the CACHED span - the path does not move
enough in under ~1.5s to need re-estimating from the extension's own, lane-poor, near-crash
frames). "Largest, lowest, near our own path", per the plan.

Output: a `partner` array (2048,), written into EACH of that video's EXISTING window .npz
files (aa4_token_labels.py's own output - `rel`/`occ`/`box_h`/`path_gap` keys are untouched,
`partner` is added alongside them). Computed with aa4_token_labels.py's OWN box_to_cells /
tid_pts_by_t machinery, restricted to the partner tid, so a window where the partner has no
box (e.g. it only appears very close to the event, after that window's own tubelets) gets an
all-zero partner array - token_sets_from_labels (aa_head_losses.py) already treats an
all-zero rel/partner array as "no P here, skip this window's aux term", which is exactly the
plan's "no visible partner -> that window has no label" behavior, with no extra bookkeeping
needed.

GATE (per the plan, REQUIRED before any --aux-label partner_pos pod run): agrees with the 14
hand-checked dev/gen18 partners in >= 80% of windows, and >= 60% of positive windows in the
pool get a label. Checking against the dev/gen18 clips needs --aa2-out-dir pointed at
outputs/aa1_v2_18clips (a DIFFERENT tid numbering than the pool1761 cache) - see --check's
help.

Usage:
  python aa4_partner_labels.py --check 00687                          # one video, prints the
                                                                       # chosen partner tid + score,
                                                                       # writes no .npz files
  python aa4_partner_labels.py --check 00687 --aa2-out-dir ../../outputs/aa1_v2_18clips
                                                                       # against the dev-clip cache,
                                                                       # for the 14-clip gate check
  python aa4_partner_labels.py --limit 20                             # smoke test, writes .npz files
  python aa4_partner_labels.py                                        # every positive video in pool1761
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aa1_stage2  # noqa: E402 - OUT_DIR is set in main()/check_video() before any loader call
from aa1_detect_track_rank import (RAW_VIDEO_ROOT, decode_frames, load_event_row,  # noqa: E402
                                   POS_SPAN_END_BEFORE_EVENT)
from aa1_scene import YOLOPv2  # noqa: E402
from aa1_stage2 import load_frame_masks, load_stitched_tracks  # noqa: E402
from aa1_lanes import path_x_at  # noqa: E402
from aa1_run_set import SETS  # noqa: E402
from aa4_token_labels import (LABEL_MAP, N_TUBELETS, GRID, N_TOKENS, box_to_cells,  # noqa: E402
                              _window_indices)
from aa1_detect_track_rank import window_ends  # noqa: E402
from semsup_extract_promptbakeoff_frames import get_video_meta  # noqa: E402

POOL_CFG = SETS["pool1761"]
LABELS_DIR = REPO / "dataset" / "aa_token_labels"
LOW_CONF_THRES = 0.10          # matches aa1_yolop_cache.py's caching convention

EXT_OVERLAP = 0.6              # s of overlap with the cached span, for track re-identification
POST_EVENT = 0.3                # s decoded PAST the event, per the plan
REID_IOU_MIN = 0.3              # first-extension-frame match to an existing stitched track
STEP_IOU_MIN = 0.1              # frame-to-frame match within the extension (generous - a
                                 # colliding box can grow/shift fast in the final frames)
END_WINDOW_S = 0.25              # a track must be seen within this of the extension's last
                                 # frame to be a partner CANDIDATE at all
PATH_MARGIN_PX = 500.0           # generous "near our own path" tolerance in compress-space
                                 # pixels-equivalent (1280-wide raw frame)


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


# =============================================================================
# Step 1-3: detect + track the extension, re-identify against the cached stitched tracks
# =============================================================================

def detect_extension(yp: YOLOPv2, video_id: str, t_ext_start: float, t_ext_end: float):
    """Per-frame YOLOPv2 boxes/scores over [t_ext_start, t_ext_end], SAME conf threshold as
    AA.2's own cache (aa1_yolop_cache.py's LOW_CONF_THRES). Returns (frames_boxes, timestamps)
    where frames_boxes[i] = list of (box_xyxy, score) for timestamps[i]."""
    frames, timestamps, fps = decode_frames(video_id, t_ext_start, t_ext_end)
    frames_boxes = []
    for frame in frames:
        out = yp.infer(frame, conf_thres=LOW_CONF_THRES)
        frames_boxes.append(list(zip(out["boxes"].tolist(), out["scores"].tolist())))
    return frames_boxes, timestamps, fps


def track_extension(frames_boxes: list, timestamps: list):
    """Minimal greedy IoU tracker over the extension's own frames only - a fresh, purely
    LOCAL track id space (not the cached tracks' tid space; reidentify_tracks() below maps
    these to old tids). Returns (tracks, first_frame_tids) where tracks =
    {local_tid: [(t, box), ...]} in chronological order, and first_frame_tids = the set of
    local_tids that already have a box in the extension's OWN first frame (frame index 0,
    whatever its real timestamp is after decode_frames' fps rounding) - only THESE are ever
    reidentify_tracks() candidates, since only they can be matched against the cached
    tracks' last-known boxes at the extension boundary. Deliberately index-based, not a
    float-timestamp comparison against the requested t_ext_start - decode_frames snaps to
    the nearest real frame, which can be ~1/fps (up to ~33ms at 30fps) later than requested,
    and an earlier version of this function compared against the REQUESTED t_ext_start with
    a 1e-6 tolerance, which silently excluded every single local track (caught in the
    Stage-0 smoke test - see the child plan's status: n_local_tracks=70, n_reidentified=0)."""
    active = {}     # local_tid -> (last_t, last_box)
    tracks = defaultdict(list)
    next_tid = 0
    first_frame_tids = set()
    for frame_idx, (boxes_scores, t) in enumerate(zip(frames_boxes, timestamps)):
        boxes = [b for b, _ in boxes_scores]
        used = set()
        # match each currently-active track to its best available box this frame
        for tid, (_, last_box) in list(active.items()):
            best_j, best_iou = -1, STEP_IOU_MIN
            for j, box in enumerate(boxes):
                if j in used:
                    continue
                iou = iou_xyxy(last_box, box)
                if iou > best_iou:
                    best_j, best_iou = j, iou
            if best_j >= 0:
                used.add(best_j)
                active[tid] = (t, boxes[best_j])
                tracks[tid].append((t, boxes[best_j]))
            else:
                del active[tid]        # track lost - not deleted from `tracks`, just stops growing
        # any unmatched box starts a new local track
        for j, box in enumerate(boxes):
            if j not in used:
                active[next_tid] = (t, box)
                tracks[next_tid].append((t, box))
                if frame_idx == 0:
                    first_frame_tids.add(next_tid)
                next_tid += 1
    return dict(tracks), first_frame_tids


def reidentify_tracks(local_tracks: dict, first_frame_tids: set, tid_pts_by_t: dict,
                      t_ext_start: float) -> dict:
    """Maps local_tid -> old_tid for every FIRST-FRAME local track (see track_extension's
    docstring) whose box IoU-matches an existing stitched track's closest-in-time known box
    near the extension boundary (within 0.5s either side - the cached span can extend
    slightly PAST t_ext_start too, see the module docstring's EXT_OVERLAP)."""
    old_near_box = {}
    for tid, pts in tid_pts_by_t.items():
        near = [(t, box) for t, box in pts.items() if abs(t - t_ext_start) <= 0.5]
        if not near:
            continue
        _, box_best = min(near, key=lambda tb: abs(tb[0] - t_ext_start))
        old_near_box[tid] = box_best

    mapping = {}
    used_old = set()
    for local_tid in first_frame_tids:
        _, box0 = local_tracks[local_tid][0]
        best_old, best_iou = None, REID_IOU_MIN
        for old_tid, old_box in old_near_box.items():
            if old_tid in used_old:
                continue
            iou = iou_xyxy(box0, old_box)
            if iou > best_iou:
                best_old, best_iou = old_tid, iou
        if best_old is not None:
            mapping[local_tid] = best_old
            used_old.add(best_old)
    return mapping


def select_partner(local_tracks: dict, mapping: dict, t_ext_end: float, last_ego_path: dict | None):
    """Among re-identified tracks, the partner is the one with the largest LAST-seen box
    height among those still visible within END_WINDOW_S of the extension's last frame - a
    track that vanished well before the event is not a crash candidate. Ties/near-ties are
    broken by proximity to the ego path (path_x_at) when an ego_path estimate is available;
    falls back to box height alone otherwise (never silently returns "no partner" just
    because the ego-path estimate is unavailable or the box is far from it - a large box
    genuinely visible up to the last frame is still the best evidence we have).
    Returns (old_tid, diagnostics dict) or (None, diagnostics) if no candidate qualifies."""
    scored = []
    for local_tid, old_tid in mapping.items():
        pts = local_tracks[local_tid]
        t_last, box_last = pts[-1]
        if t_ext_end - t_last > END_WINDOW_S:
            continue
        h = box_last[3] - box_last[1]
        cx = (box_last[0] + box_last[2]) / 2.0
        cy = box_last[3]
        path_dx = None
        if last_ego_path is not None:
            path_dx = abs(cx - path_x_at(last_ego_path, cy))
        scored.append(dict(old_tid=old_tid, t_last=round(t_last, 4), box=box_last,
                           height=round(h, 1), path_dx=round(path_dx, 1) if path_dx is not None else None))
    if not scored:
        return None, dict(candidates=[])
    # primary: largest height; among near-ties (within 20% of the max height) prefer the
    # one closest to the ego path, when we have a path estimate.
    max_h = max(s["height"] for s in scored)
    near_top = [s for s in scored if s["height"] >= 0.8 * max_h]
    if last_ego_path is not None and any(s["path_dx"] is not None for s in near_top):
        near_top.sort(key=lambda s: (s["path_dx"] if s["path_dx"] is not None else 1e9))
    else:
        near_top.sort(key=lambda s: -s["height"])
    best = near_top[0]
    return best["old_tid"], dict(candidates=scored, chosen=best)


# =============================================================================
# Step 4: find the partner for one video
# =============================================================================

def find_partner(yp: YOLOPv2, video_id: str, aa2_out_dir: Path):
    aa1_stage2.OUT_DIR = aa2_out_dir
    row = load_event_row(video_id)
    if int(row["target"]) != 1:
        raise ValueError(f"{video_id} is not a positive video - no crash to find a partner for.")
    t_event = float(row["time_of_event"])
    t_ext_start = t_event - POS_SPAN_END_BEFORE_EVENT - EXT_OVERLAP
    t_ext_end = t_event + POST_EVENT

    frame_masks, fps, span = load_frame_masks(video_id)
    if t_ext_start < span[0] - 1e-6:
        raise RuntimeError(f"{video_id}: extension start {t_ext_start:.2f}s is before the "
                           f"cached span's own start {span[0]:.2f}s - AA.2's decode_span "
                           f"changed, or POS_SPAN_START_BEFORE_EVENT is too small for "
                           f"EXT_OVERLAP={EXT_OVERLAP}.")
    tracks = load_stitched_tracks(video_id)
    tid_pts_by_t = {tid: {round(t, 4): box for t, box, _ in pts} for tid, pts in tracks.items()}

    # the last ego_path estimated from the CACHED span (closest frame at/before the cached
    # span's own end) - see the module docstring for why this is not re-estimated from the
    # extension's own frames.
    cached_ts = sorted(t for t in frame_masks if t <= span[1] + 1e-6)
    last_ego_path = frame_masks[cached_ts[-1]]["ego_path"] if cached_ts else None

    frames_boxes, timestamps, _ = detect_extension(yp, video_id, t_ext_start, t_ext_end)
    local_tracks, first_frame_tids = track_extension(frames_boxes, timestamps)
    mapping = reidentify_tracks(local_tracks, first_frame_tids, tid_pts_by_t, t_ext_start)
    partner_tid, diag = select_partner(local_tracks, mapping, timestamps[-1] if timestamps else t_ext_end,
                                       last_ego_path)
    diag.update(video_id=video_id, t_event=t_event, t_ext_start=round(t_ext_start, 4),
               t_ext_end=round(t_ext_end, 4), n_extension_frames=len(timestamps),
               n_local_tracks=len(local_tracks), n_reidentified=len(mapping))
    return partner_tid, tid_pts_by_t, diag


# =============================================================================
# Step 5: write the `partner` array into the video's existing window .npz files
# =============================================================================

def write_partner_labels(video_id: str, rows: list, partner_tid: int | None,
                         tid_pts_by_t: dict) -> list:
    results = []
    partner_pts = tid_pts_by_t.get(partner_tid, {}) if partner_tid is not None else {}
    _, win_ends = window_ends(video_id)
    t_end_by_label = dict(win_ends)
    _, total = get_video_meta(video_id)
    for row in rows:
        win_label = LABEL_MAP[row["requested_time_to_event"]]
        t_end = t_end_by_label[win_label]
        npz_path = LABELS_DIR / f"{row['frames_dir']}.npz"
        if not npz_path.exists():
            results.append(dict(frames_dir=row["frames_dir"], status="no_base_label"))
            continue
        with np.load(npz_path) as z:
            existing = {k: z[k] for k in z.files}
        # re-derive this window's own 16-frame indices identically to aa4_token_labels.py,
        # using the SAME fps this video's stitched tracks were built at.
        partner_arr = np.zeros(N_TOKENS, dtype=np.float32)
        n_cells = 0
        if partner_pts:
            from aa1_detect_track_rank import video_fps_duration
            vfps, _ = video_fps_duration(video_id)
            indices = _window_indices(t_end, vfps, total)
            for tub in range(N_TUBELETS):
                idx_lo, idx_hi = indices[2 * tub], indices[2 * tub + 1]
                t_lo, t_hi = round(idx_lo / vfps, 4), round(idx_hi / vfps, 4)
                cells = set()
                for t_frame in (t_lo, t_hi):
                    box = partner_pts.get(t_frame)
                    if box is not None:
                        cells |= box_to_cells(box)
                for r, c in cells:
                    partner_arr[tub * GRID * GRID + r * GRID + c] = 1.0
                n_cells += len(cells)
        existing["partner"] = partner_arr
        np.savez_compressed(npz_path, **existing)
        results.append(dict(frames_dir=row["frames_dir"], status="ok",
                            has_partner=bool(n_cells > 0), n_partner_cells=n_cells))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(POOL_CFG["manifest"]))
    ap.add_argument("--aa2-out-dir", default=str(POOL_CFG["out_dir"]),
                    help="where AA.2's <vid>_yolop.json/_tracks_v2.json live. Override to "
                         "outputs/aa1_v2_18clips for the 14-clip GATE check against the "
                         "dev/gen18 hand-labeled partners (a DIFFERENT tid numbering than "
                         "the pool1761 cache).")
    ap.add_argument("--limit", type=int, default=0, help="first N positive videos, smoke test")
    ap.add_argument("--check", default=None, metavar="VIDEO_ID",
                    help="run ONE video, print the chosen partner + diagnostics, write NO "
                         ".npz files, and exit.")
    ap.add_argument("--weights", default=None, help="YOLOPv2 weights path override")
    args = ap.parse_args()

    aa2_out_dir = Path(args.aa2_out_dir)
    yp_kwargs = {}
    if args.weights:
        yp_kwargs["weights_path"] = Path(args.weights)
    yp = YOLOPv2(**yp_kwargs)

    if args.check:
        vid = args.check
        partner_tid, tid_pts_by_t, diag = find_partner(yp, vid, aa2_out_dir)
        print(json.dumps(diag, indent=2, default=str))
        print(f"\n[{vid}] chosen partner tid = {partner_tid}")
        return

    rows_by_vid = defaultdict(list)
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if row.get("event_occurs") == 1:
                    rows_by_vid[str(row["video_id"]).zfill(5)].append(row)

    vids = sorted(rows_by_vid)
    if args.limit:
        vids = vids[:args.limit]
    print(f"[data] {len(vids)} positive videos to process")

    all_video_results = []
    t0 = time.time()
    for i, vid in enumerate(vids):
        try:
            partner_tid, tid_pts_by_t, diag = find_partner(yp, vid, aa2_out_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"[{vid}] SKIP: {exc}")
            all_video_results.append(dict(video_id=vid, status="skip", reason=str(exc)))
            continue
        win_results = write_partner_labels(vid, rows_by_vid[vid], partner_tid, tid_pts_by_t)
        n_labeled = sum(1 for r in win_results if r.get("has_partner"))
        all_video_results.append(dict(video_id=vid, status="ok", partner_tid=partner_tid,
                                      n_windows=len(win_results), n_windows_labeled=n_labeled,
                                      diag=diag))
        if (i + 1) % 10 == 0 or i == len(vids) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            print(f"  [{i+1}/{len(vids)}] elapsed={elapsed/60:.1f}min  "
                 f"eta={(len(vids)-i-1)*rate/60:.1f}min", flush=True)

    n_ok = sum(1 for r in all_video_results if r["status"] == "ok")
    n_found = sum(1 for r in all_video_results if r["status"] == "ok" and r["partner_tid"] is not None)
    n_win_labeled = sum(r.get("n_windows_labeled", 0) for r in all_video_results)
    n_win_total = sum(r.get("n_windows", 0) for r in all_video_results)
    stats = dict(n_videos=len(vids), n_ok=n_ok, n_partner_found=n_found,
                n_windows_total=n_win_total, n_windows_labeled=n_win_labeled,
                window_label_rate=round(n_win_labeled / n_win_total, 4) if n_win_total else None)
    print(f"\n[stats] {json.dumps(stats, indent=2)}")
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LABELS_DIR / "partner_label_stats.json", "w", encoding="utf-8") as f:
        json.dump(dict(stats=stats, per_video=all_video_results), f, indent=2, default=str)
    print(f"[done] wrote {LABELS_DIR / 'partner_label_stats.json'}")


if __name__ == "__main__":
    main()
