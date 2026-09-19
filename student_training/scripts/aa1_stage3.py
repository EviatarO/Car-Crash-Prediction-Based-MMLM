"""
aa1_stage3.py
=============
Stage 3 driver (child plan 2026-09-15-AA1-v2b): per-window object selection (aa1_collision).

2026-09-18 rewrite: scores are now computed ONCE per clip in a single causal forward pass
(aa1_collision.compute_track_scores) over every track's own detections, in time order, with
EMA smoothing carried across each track's own history. Both the per-window selection (below)
and the per-frame overlay/curves look up that same precomputed, already-smoothed score at the
relevant timestamp - nothing is recomputed per window or per frame.

Per clip, in outputs/aa1_v2_18clips/:
  <vid>_threat_v2.json       per-window top-5 {track_id, rank, score} + full score breakdown
  <vid>_overlay_threat.mp4   every frame: lane/drivable masks, the ego path (white, drawn up to
                             the first car on it) and its source, the scoring candidates only
                             (thin grey with id), current top-5 always colored + labeled
                             "#rank idN score"
  <vid>_target_curves.png/csv  5 panels: the Stage 4 target dims (alpha, side gap g, closing
                             rate, lane) + the selection score, for the 5 longest-lived tracks

2026-09-19: dropped <vid>_grid16_threat.jpg (the 16-frame grid from the overlay) - reviewing
this project's clips has settled on watching the .mp4 directly, and generating+saving the grid
was pure overhead once nobody was opening it any more.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aa1_detect_track_rank import decode_frames, window_ends  # noqa: E402
from aa1_lanes import path_x_at  # noqa: E402
from aa1_stage2 import load_frame_masks, load_stitched_tracks  # noqa: E402
from aa1_track_stage1 import VAL_E3A_IDS, PALETTE_HEX  # noqa: E402
from aa1_collision import compute_track_scores, candidate_pool, select_top_k  # noqa: E402

OUT_DIR = REPO / "outputs" / "aa1_v2_18clips"
N_CURVE_OBJECTS = 5
# BGR, rank 1..5, most relevant first
RANK_COLORS = [(0, 0, 255), (0, 128, 255), (0, 220, 255), (0, 255, 140), (255, 200, 0)]
GRAY = (140, 140, 140)


def last_causal(tracks: dict, t_end: float) -> dict:
    """{tid: (t, box, conf)} - each track's own most recent point at or before t_end."""
    out = {}
    for tid, pts in tracks.items():
        causal = [p for p in pts if p[0] <= t_end + 1e-6]
        if causal:
            out[tid] = causal[-1]
    return out


def compute_window_top5(win_ends: list, tracks: dict, scores: dict, fps: float) -> list[dict]:
    out = []
    for label, t_end in win_ends:
        last = last_causal(tracks, t_end)
        boxes_now = {tid: box for tid, (t, box, conf) in last.items()}
        pool = candidate_pool(boxes_now, tracks, t_end, fps)
        scores_at_pool = {tid: scores[tid][last[tid][0]] for tid in pool if last[tid][0] in scores.get(tid, {})}
        top5 = select_top_k(pool, scores_at_pool, k=5)
        out.append(dict(label=label, t_end=round(t_end, 3), n_candidates=len(pool),
                        top5=[dict(track_id=tid, rank=rank + 1, **scores_at_pool[tid])
                             for rank, (tid, _) in enumerate(top5)]))
    return out


def render_overlay(video_id, frames, timestamps, frame_masks, tracks, scores, out_dir: Path, fps):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_dir / f"{video_id}_overlay_threat.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame, t_raw in zip(frames, timestamps):
        t = round(float(t_raw), 4)
        vis = frame.copy()
        fm = frame_masks.get(t)
        if fm is not None:
            vis[fm["drivable"]] = (vis[fm["drivable"]] * 0.75 + np.array([0, 255, 0]) * 0.25).astype(np.uint8)
            vis[fm["lane"]] = (vis[fm["lane"]] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)

        boxes_now = {}
        for tid, pts in tracks.items():
            causal = [p for p in pts if p[0] <= t + 1e-4]
            if not causal or abs(causal[-1][0] - t) > 1e-4:
                continue
            boxes_now[tid] = causal[-1][1]

        pool = candidate_pool(boxes_now, tracks, t, fps)
        scores_now = {tid: scores[tid][t] for tid in pool if t in scores.get(tid, {})}
        top5 = select_top_k(pool, scores_now, k=5)
        # every one of the top 5 gets a rank + color, regardless of score - a low score still
        # means "one of this window's 5 candidates" and should not look identical to a box that
        # didn't even make the cut (00319 id3: real top-5 member every window, but its score
        # never crossed the old 0.1 display floor, so it rendered exactly like an ignored box)
        top_ids = {tid: (rank + 1, sc) for rank, (tid, sc) in enumerate(top5)}

        if fm is not None:
            ep = fm["ego_path"]
            # draw the full path, as far as the lane lines actually reach (ep['top']) - not
            # cut short at whatever car happens to be on it
            ys = np.arange(h - 1, ep["top"], -4.0)
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
                cv2.putText(vis, f"id{tid}", (x1, max(12, y1 - 4)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, GRAY, 1)
        cv2.putText(vis, f"{video_id}  t={t:.2f}s", (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3)
        writer.write(vis)
    writer.release()


def render_target_curves(video_id, tracks, scores, timestamps, win_ends, is_pos, out_dir: Path):
    """5 panels, 3x2 grid (row 3 spans both columns): the 4 Stage 4 target dims - alpha
    (looming), g (side gap to the ego path, in box-heights), closing_rate (= -g_rate, + =
    approaching the path), lane (LEFT/EGO/RIGHT, 5-sample majority vote) - plus the selection
    score that picked which objects to show. All 5 come straight out of
    aa1_collision.compute_track_scores' per-frame dict; nothing here is recomputed.

    2026-09-19 rewrite: was 6 stacked panels sharing one x-axis and one legend, plotting the
    OLD lane-overlap-width geometry (s/s_rate/rho/lane_state) that Stage 3's score no longer
    uses - kept "for comparison" but never actually compared to anything. Now every panel has
    its own x-axis and legend, and shows only what the score and the Stage 4 targets are built
    from."""
    top_ids = sorted(tracks, key=lambda tid: -len(tracks[tid]))[:N_CURVE_OBJECTS]
    if not top_ids:
        return
    id_labels = {tid: f"id{tid} ({len(tracks[tid])} pts)" for tid in top_ids}
    LANE = {"LEFT": -1.0, "EGO": 0.0, "RIGHT": 1.0}
    ts_all = np.array(timestamps)
    n = len(ts_all)
    dims = ["alpha", "g", "closing_rate", "lane", "threat"]
    series = {tid: {d: np.full(n, np.nan) for d in dims} for tid in top_ids}
    lane_raw = {tid: [None] * n for tid in top_ids}
    rows = []
    for f, t_raw in enumerate(ts_all):
        t = round(float(t_raw), 4)
        for tid in top_ids:
            e = scores.get(tid, {}).get(t)
            if e is None:
                continue
            s = series[tid]
            s["alpha"][f] = e["alpha"]
            s["g"][f] = e["g"]
            s["closing_rate"][f] = -e["g_rate"]
            s["threat"][f] = e["score"]
            lane_raw[tid][f] = e["lane"]
            rows.append(dict(frame=f, t=round(float(t), 3), track_id=tid, alpha=e["alpha"],
                             side_gap=e["g"], closing_rate=round(-e["g_rate"], 4), lane=e["lane"],
                             selection_score=e["score"]))
    # 5-sample majority vote on the categorical lane label, matching aa1_lanes.FLICKER_SMOOTH_N -
    # a single-frame flip near the path edge (path-width noise) shouldn't flip the plotted state
    for tid in top_ids:
        raw = lane_raw[tid]
        for f in range(n):
            window = [v for v in raw[max(0, f - 4):f + 1] if v is not None]
            if not window:
                continue
            counts = {v: window.count(v) for v in set(window)}
            top = max(counts.values())
            tied = {v for v, c in counts.items() if c == top}
            smoothed = next(v for v in reversed(window) if v in tied)
            series[tid]["lane"][f] = LANE[smoothed]
    if rows:
        with open(out_dir / f"{video_id}_target_curves.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    panels = [
        ("alpha", "1. Looming α  (1/s)", "log-size growth rate; Stage 4 target"),
        ("g", "2. Side gap g  (box-heights)", "gap from the ego path; 0 = on the path; Stage 4 target"),
        ("closing_rate", "3. Closing rate  (box-heights/s)", "-d(g)/dt; + = approaching the path; Stage 4 target"),
        ("lane", "4. Lane  (5-sample majority)", "position relative to the ego path; Stage 4 target"),
        ("threat", "5. Selection score", "closeness x side x (1+approach), EMA-smoothed - picks which objects get the targets above; never fit to the crash label"),
    ]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig = plt.figure(figsize=(13, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.55, wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
           fig.add_subplot(gs[2, :])]
    x = ts_all - ts_all[0]
    for ax, (key, title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, tid in enumerate(top_ids):
            y = series[tid][key]
            color = PALETTE_HEX[i % len(PALETTE_HEX)]
            if key == "lane":
                ax.step(x, y, where="mid", color=color, lw=2, label=id_labels[tid])
            else:
                ax.plot(x, y, color=color, lw=2, label=id_labels[tid])
        for label, te in win_ends:
            ax.axvline(te - ts_all[0], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=0)
            ax.text(te - ts_all[0], 1.0, label, transform=ax.get_xaxis_transform(),
                   color=MUTED, fontsize=7, rotation=90, va="top", ha="right")
        if key == "lane":
            ax.set_yticks([-1, 0, 1], ["LEFT", "EGO", "RIGHT"]); ax.set_ylim(-1.5, 1.5)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold", pad=14)
        if sub:
            ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")
        ax.set_xlabel("seconds into the decoded span", color=INK2, fontsize=8.5)
        ax.legend(loc="best", ncol=min(len(top_ids), 3), frameon=True, framealpha=0.7,
                 facecolor=SURFACE, edgecolor=AXIS, fontsize=7.5, labelcolor=INK2, handlelength=1.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle(f"Clip {video_id} ({'TP - positive/crash' if is_pos else 'TN - negative/normal'}): "
                f"Stage 4 targets + selection score per frame (causal; {len(top_ids)} longest-lived tracks shown)",
                x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"{video_id}_target_curves.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def run_clip(video_id: str, out_dir: Path):
    frame_masks, fps, span = load_frame_masks(video_id)
    tracks = load_stitched_tracks(video_id)
    frames, timestamps, _ = decode_frames(video_id, *span)

    scores = compute_track_scores(tracks, frame_masks, fps)

    is_pos, wins = window_ends(video_id)
    wins = [(label, te) for label, te in wins if span[0] - 1e-6 <= te <= span[1] + 1e-6]

    threat_windows = compute_window_top5(wins, tracks, scores, fps)
    for win in threat_windows:
        top_desc = ", ".join(f"#{o['rank']} id{o['track_id']} ({o['score']:.2f})" for o in win["top5"])
        print(f"[{video_id}] {win['label']} (t={win['t_end']:.2f}s): top5 = {top_desc or '(none)'}")

    render_overlay(video_id, frames, timestamps, frame_masks, tracks, scores, out_dir, fps)
    render_target_curves(video_id, tracks, scores, timestamps, wins, is_pos, out_dir)

    with open(out_dir / f"{video_id}_threat_v2.json", "w", encoding="utf-8") as f:
        json.dump(dict(video_id=video_id, span=list(span), fps=fps, windows=threat_windows), f, indent=2)
    return threat_windows


def main():
    out_dir = OUT_DIR
    all_windows = []
    for vid in VAL_E3A_IDS:
        all_windows.extend(run_clip(vid, out_dir))
    n_empty = sum(1 for w in all_windows if not w["top5"])
    print(f"\n[summary] {len(all_windows)} windows, {n_empty} with an empty top5")


if __name__ == "__main__":
    main()
