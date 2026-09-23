"""
aa4_token_labels.py
====================
Stage AA.4 (child plan 2026-09-19-AA-token-relevance-aux): turn AA.2's cached detections
(outputs/aa1_pool1761/<vid>_yolop.json + <vid>_tracks_v2.json, written by
`aa1_run_set.py --set pool1761 --stages 0,1`) into the two per-window token label maps that
`semsup_train.py --aux-mode {occ,rel}` trains against - one .npz per window, keyed by the
window's own `frames_dir` (dataset/manifests/recap_v12_1761.jsonl's naming, e.g.
"00958_hires_tte05") so the training loader can look a window's labels up the same way it
already looks its 16 frames up.

Two label channels per window, each a length-2048 array over the SAME token grid V-JEPA2's
encoder uses (8 tubelets x 16x16 patches, index = tubelet*256 + row*16 + col - confirmed by
aa0_measure_geometry.py's flatten-order probe, outputs/aa0/geometry_pod.json):

  occ  uint8   1 if ANY tracked vehicle box covers that patch in that tubelet, else 0.
               AA-occ's target - "is there a car here" - no selection, every stitched track.
  rel  float32 top-5 selection-score / 2 (max score is 1*1*(1+1)=2) for the tubelet's own
               top-5 objects (aa1_collision.candidate_pool + select_top_k, recomputed
               PER TUBELET from aa1_collision.compute_track_scores' single causal pass over
               the clip - not just at the window's end), 0 elsewhere. AA-rel's target.

Both channels are computed from the union of the two raw video frames a tubelet actually pools
(sampled frames 2t and 2t+1 of the window's 16) - a track's box at EITHER frame contributes,
since the tubelet's Conv3d patch embedding sees both.

This intentionally reads the selection score as a training target for the first time in Stage
AA (recorded rev2 decision: "the selection score is never a training target" - see this child
plan's Architecture section and DECISIONS.md for why that's being relaxed here: the score is
still never fit to the crash label, so the no-leakage invariant holds; the cost is that the
trunk learns OUR heuristic, spare-tire ego-paths and all).

Usage:
  python aa4_token_labels.py                        # every video in the pool1761 clip_list.json
  python aa4_token_labels.py --limit 20              # smoke test, first 20 videos
  python aa4_token_labels.py --check 00687 00687_hires_tte05   # one label-sanity PNG, no batch run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aa1_stage2  # noqa: E402 - OUT_DIR is set below, before any loader call
from aa1_collision import (FRAME_W, FRAME_H, compute_track_scores, candidate_pool,  # noqa: E402
                           select_top_k)
from aa1_detect_track_rank import RAW_VIDEO_ROOT, window_ends  # noqa: E402
from aa1_lanes import path_x_at  # noqa: E402
from aa1_run_set import SETS  # noqa: E402
from aa1_stage2 import load_frame_masks, load_stitched_tracks  # noqa: E402
from aa1_stage3 import RANK_COLORS, GRAY  # noqa: E402 - same rank colors as the dev overlays
from semsup_extract_promptbakeoff_frames import WINDOW, STRIDE, get_video_meta  # noqa: E402

POOL_CFG = SETS["pool1761"]
AA2_OUT_DIR = POOL_CFG["out_dir"]                         # <vid>_yolop.json / _tracks_v2.json
LABELS_DIR = REPO / "dataset" / "aa_token_labels"
N_TUBELETS, GRID, N_TOKENS = 8, 16, 2048

# recap_v12_1761.jsonl's requested_time_to_event strings -> window_ends()'s own label strings.
# Kept as an explicit table (not string munging) so a change on either side fails loudly instead
# of silently mismatching. See aa1_detect_track_rank.window_ends / decode_span for the formulas
# both sides ultimately share (t_event - tte for positives, midpoint - off for negatives).
LABEL_MAP = {"TTE_1.5": "TTE1.5", "TTE_1.0": "TTE1", "TTE_0.5": "TTE0.5",
             "MID-10": "MID-10", "MID-8": "MID-8", "MID-4": "MID-4"}


def _window_indices(t_end: float, fps: float, total: int) -> list:
    """Identical formula to semsup_extract_promptbakeoff_frames._indices_for - duplicated (not
    imported, since it's a private name) so a window's 16 raw-frame indices are recomputed
    exactly as they were when its frames_dir was extracted to disk."""
    end = round(t_end * fps)
    idx = [end - (WINDOW - 1 - i) * STRIDE for i in range(WINDOW)]
    return [max(0, min(total - 1, ix)) for ix in idx]


def box_to_cells(box) -> set:
    """A box in raw 1280x720 pixels -> the set of (row, col) 16x16 patch cells it covers after
    compress256's whole-frame squash to 256x256 (x*256/1280, y*256/720, then //16). Covers the
    box's full cell RANGE, not just its corners."""
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return set()
    x1c, y1c = x1 * 256.0 / FRAME_W, y1 * 256.0 / FRAME_H
    x2c, y2c = x2 * 256.0 / FRAME_W, y2 * 256.0 / FRAME_H
    c1, c2 = sorted((int(x1c // 16), int(max(x1c, x2c - 1e-6) // 16)))
    r1, r2 = sorted((int(y1c // 16), int(max(y1c, y2c - 1e-6) // 16)))
    c1, c2 = max(0, c1), min(GRID - 1, c2)
    r1, r2 = max(0, r1), min(GRID - 1, r2)
    return {(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)}


def _label_arrays_for_window(tid_pts_by_t: dict, tracks: dict, scores: dict, fps: float,
                              indices: list) -> tuple:
    rel = np.zeros(N_TOKENS, dtype=np.float32)
    occ = np.zeros(N_TOKENS, dtype=np.uint8)
    # Two baseline features for aa1_token_probe.py's Phase-3 gate (child plan: "how much of
    # r is motion vs static geometry"), NOT used as training targets anywhere - box_h = the
    # tallest covering box's height in px (0 = no box); path_gap = the smallest side-gap
    # (box-heights to the ego path, straight from compute_track_scores) among covering boxes
    # (99.0 sentinel = no box). Both single-frame/no-history, unlike rel's alpha/closing_rate.
    box_h = np.zeros(N_TOKENS, dtype=np.float32)
    path_gap = np.full(N_TOKENS, 99.0, dtype=np.float32)
    n_top5_total, n_top5_empty = 0, 0
    for tub in range(N_TUBELETS):
        idx_lo, idx_hi = indices[2 * tub], indices[2 * tub + 1]
        t_lo, t_hi = round(idx_lo / fps, 4), round(idx_hi / fps, 4)

        boxes_hi = {tid: pts[t_hi] for tid, pts in tid_pts_by_t.items() if t_hi in pts}
        pool = candidate_pool(boxes_hi, tracks, t_hi, fps)
        scores_at_pool = {tid: scores[tid][t_hi] for tid in pool if t_hi in scores.get(tid, {})}
        top5 = select_top_k(pool, scores_at_pool, k=5)
        n_top5_total += len(top5)
        n_top5_empty += int(not top5)

        for tid, _ in top5:
            r = min(scores_at_pool[tid]["score"] / 2.0, 1.0)
            cells = set()
            for t_frame in (t_lo, t_hi):
                box = tid_pts_by_t.get(tid, {}).get(t_frame)
                if box is not None:
                    cells |= box_to_cells(box)
            for row, col in cells:
                idx = tub * GRID * GRID + row * GRID + col
                rel[idx] = max(rel[idx], r)

        for tid, pts in tid_pts_by_t.items():
            cells = set()
            h, g = 0.0, 99.0
            for t_frame in (t_lo, t_hi):
                box = pts.get(t_frame)
                if box is not None:
                    cells |= box_to_cells(box)
                    h = max(h, float(box[3] - box[1]))
                entry = scores.get(tid, {}).get(t_frame)
                if entry is not None:
                    g = min(g, entry["g"])
            for row, col in cells:
                idx = tub * GRID * GRID + row * GRID + col
                occ[idx] = 1
                box_h[idx] = max(box_h[idx], h)
                path_gap[idx] = min(path_gap[idx], g)

    return rel, occ, box_h, path_gap, n_top5_total, n_top5_empty


def process_clip(video_id: str, rows: list) -> list:
    frame_masks, fps, span = load_frame_masks(video_id)
    tracks = load_stitched_tracks(video_id)
    scores = compute_track_scores(tracks, frame_masks, fps)
    tid_pts_by_t = {tid: {round(t, 4): box for t, box, _ in pts} for tid, pts in tracks.items()}
    _, win_ends = window_ends(video_id)
    t_end_by_label = dict(win_ends)
    _, total = get_video_meta(video_id)

    results = []
    for row in rows:
        win_label = LABEL_MAP[row["requested_time_to_event"]]
        t_end = t_end_by_label[win_label]
        if not (span[0] - 1e-6 <= t_end <= span[1] + 1e-6):
            results.append(dict(frames_dir=row["frames_dir"], status="out_of_cached_span"))
            continue
        indices = _window_indices(t_end, fps, total)
        rel, occ, box_h, path_gap, n_top5, n_top5_empty = _label_arrays_for_window(
            tid_pts_by_t, tracks, scores, fps, indices)
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(LABELS_DIR / f"{row['frames_dir']}.npz",
                            rel=rel, occ=occ, box_h=box_h, path_gap=path_gap)
        results.append(dict(frames_dir=row["frames_dir"], status="ok",
                            n_object_tokens=int((occ > 0).sum()),
                            rel_mean=round(float(rel.mean()), 5), rel_max=round(float(rel.max()), 4),
                            n_top5=n_top5, n_top5_empty_tubelets=n_top5_empty))
    return results


def _draw_full_overlay(frame, t: float, frame_masks: dict, tracks: dict, tid_pts_by_t: dict,
                       scores: dict, fps: float, video_id: str):
    """Same per-frame drawing as aa1_stage3.render_overlay (lane/drivable tint, the white
    ego-path line, ranked top-5 boxes + grey candidates) - duplicated here (not imported as a
    function, aa1_stage3 only has it inlined in its own frame loop) so this file's checks show
    the SAME geometry a dev-clip overlay video shows, at full 1280x720 resolution, for the
    token labels' own tubelet frame."""
    vis = frame.copy()
    fm = frame_masks.get(t)
    boxes_now = {tid: pts[t] for tid, pts in tid_pts_by_t.items() if t in pts}
    pool = candidate_pool(boxes_now, tracks, t, fps)
    scores_now = {tid: scores[tid][t] for tid in pool if t in scores.get(tid, {})}
    top5 = select_top_k(pool, scores_now, k=5)
    top_ids = {tid: (rank + 1, sc) for rank, (tid, sc) in enumerate(top5)}

    if fm is not None:
        vis[fm["drivable"]] = (vis[fm["drivable"]] * 0.75 + np.array([0, 255, 0]) * 0.25).astype(np.uint8)
        vis[fm["lane"]] = (vis[fm["lane"]] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        ep = fm["ego_path"]
        ys = np.arange(vis.shape[0] - 1, ep["top"], -4.0)
        pts_path = np.array([[path_x_at(ep, y), y] for y in ys], dtype=np.int32)
        if len(pts_path) >= 2:
            cv2.polylines(vis, [pts_path], False, (255, 255, 255), 3)
        cv2.putText(vis, f"path: {ep['src']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    for tid, box in pool.items():
        x1, y1, x2, y2 = (int(v) for v in box)
        if tid in top_ids:
            rank, score = top_ids[tid]
            color = RANK_COLORS[min(rank - 1, len(RANK_COLORS) - 1)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)
            cv2.putText(vis, f"#{rank} id{tid} {score:.2f}", (x1, max(18, y1 - 8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), GRAY, 1)
            cv2.putText(vis, f"id{tid}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
    cv2.putText(vis, f"{video_id} t={t:.2f}s", (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
               1.1, (255, 255, 255), 3)
    return vis


def render_check(video_id: str, frames_dir_name: str, out_png: Path):
    """Label-sanity spot check (child plan verification step): the window's 8 tubelets, each
    shown three ways - the FULL 1280x720 overlay (lane/drivable tint, white ego-path line,
    ranked top-5 boxes, same as aa1_stage3.render_overlay - added 2026-09-22 so this check can
    be read against the dev-clip overlay videos directly) on top, then the rel heatmap and the
    occ mask on the compress256 (256x256) input below. Looks `frames_dir_name` up in the pool
    manifest to find its window label - NOT a string-suffix guess - then reads the same cached
    AA.2 output as process_clip. Does not require the .npz to already exist."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(POOL_CFG["manifest"], encoding="utf-8") as f:
        row = next(json.loads(l) for l in f if json.loads(l)["frames_dir"] == frames_dir_name)
    win_label = LABEL_MAP[row["requested_time_to_event"]]

    aa1_stage2.OUT_DIR = AA2_OUT_DIR
    frame_masks, fps, span = load_frame_masks(video_id)
    tracks = load_stitched_tracks(video_id)
    scores = compute_track_scores(tracks, frame_masks, fps)
    tid_pts_by_t = {tid: {round(t, 4): box for t, box, _ in pts} for tid, pts in tracks.items()}
    _, win_ends = window_ends(video_id)
    t_end = dict(win_ends)[win_label]
    _, total = get_video_meta(video_id)
    indices = _window_indices(t_end, fps, total)
    rel, occ, box_h, path_gap, n_top5, n_empty = _label_arrays_for_window(
        tid_pts_by_t, tracks, scores, fps, indices)
    rel_grid = rel.reshape(N_TUBELETS, GRID, GRID)
    occ_grid = occ.reshape(N_TUBELETS, GRID, GRID)

    cap = cv2.VideoCapture(str(RAW_VIDEO_ROOT / f"{video_id}.mp4"))
    fig, axes = plt.subplots(3, N_TUBELETS, figsize=(3 * N_TUBELETS, 9.6),
                             gridspec_kw=dict(height_ratios=[1.6, 1, 1]))
    for tub in range(N_TUBELETS):
        idx_hi = indices[2 * tub + 1]
        t_hi = round(idx_hi / fps, 4)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx_hi)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
        overlay = _draw_full_overlay(frame, t_hi, frame_masks, tracks, tid_pts_by_t, scores,
                                     fps, video_id)
        axes[0, tub].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[0, tub].set_title(f"t{tub}: full overlay (t={t_hi:.2f}s)", fontsize=8)
        axes[0, tub].axis("off")

        img = cv2.resize(frame, (256, 256))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[1, tub].imshow(img_rgb)
        axes[1, tub].imshow(rel_grid[tub], alpha=0.45, cmap="inferno", vmin=0, vmax=1,
                            extent=(0, 256, 256, 0))
        axes[1, tub].set_title(f"t{tub}: rel", fontsize=9); axes[1, tub].axis("off")
        axes[2, tub].imshow(img_rgb)
        axes[2, tub].imshow(occ_grid[tub], alpha=0.4, cmap="Greens", vmin=0, vmax=1,
                            extent=(0, 256, 256, 0))
        axes[2, tub].set_title(f"t{tub}: occ", fontsize=9); axes[2, tub].axis("off")
    cap.release()
    fig.suptitle(f"{video_id} / {frames_dir_name}  (window end t={t_end:.2f}s, "
                f"{n_top5}/{N_TUBELETS*5} top5 slots filled, {n_empty} empty tubelets)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(POOL_CFG["manifest"]))
    ap.add_argument("--limit", type=int, default=0, help="first N videos only, for a smoke test")
    ap.add_argument("--check", nargs=2, metavar=("VIDEO_ID", "FRAMES_DIR"),
                    help="render one label-sanity PNG for this video/window and exit - no batch run")
    args = ap.parse_args()

    aa1_stage2.OUT_DIR = AA2_OUT_DIR

    if args.check:
        vid, frames_dir = args.check
        render_check(vid, frames_dir, LABELS_DIR / "_checks" / f"{frames_dir}.png")
        return

    rows_by_vid = defaultdict(list)
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows_by_vid[str(row["video_id"]).zfill(5)].append(row)

    vids = sorted(rows_by_vid)
    if args.limit:
        vids = vids[:args.limit]

    all_results, n_missing_cache = [], 0
    for vid in vids:
        try:
            all_results.extend(process_clip(vid, rows_by_vid[vid]))
        except FileNotFoundError:
            n_missing_cache += 1
            print(f"[{vid}] SKIP - no AA.2 cache in {AA2_OUT_DIR} "
                 f"(run: aa1_run_set.py --set pool1761 --stages 0,1)")

    n_ok = sum(1 for r in all_results if r["status"] == "ok")
    n_oos = sum(1 for r in all_results if r["status"] == "out_of_cached_span")
    rel_means = [r["rel_mean"] for r in all_results if r["status"] == "ok"]
    obj_tok = [r["n_object_tokens"] for r in all_results if r["status"] == "ok"]
    stats = dict(n_videos_requested=len(vids), n_videos_missing_cache=n_missing_cache,
                n_windows=len(all_results), n_ok=n_ok, n_out_of_cached_span=n_oos,
                rel_mean_of_means=round(float(np.mean(rel_means)), 5) if rel_means else None,
                object_tokens_mean=round(float(np.mean(obj_tok)), 1) if obj_tok else None,
                object_tokens_median=float(np.median(obj_tok)) if obj_tok else None)
    print(f"\n[summary] {stats}")
    with open(LABELS_DIR / "label_stats.json", "w", encoding="utf-8") as f:
        json.dump(dict(stats=stats, per_window=all_results), f, indent=2)


if __name__ == "__main__":
    main()
