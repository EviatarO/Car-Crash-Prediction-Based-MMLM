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
  <vid>_grid16_threat.jpg    16-frame grid from that overlay
  <vid>_target_curves.png/csv  panels 1-5 unchanged (alpha/s_end/s_rate/rho/lane, from
                             aa1_lanes.track_geometry_v2, kept for comparison only); panel 6 is
                             now the new selection score
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
from aa1_lanes import track_geometry_v2, path_x_at  # noqa: E402
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
    grid_idx = set(np.linspace(0, len(frames) - 1, 16).round().astype(int).tolist())
    tiles = []
    writer = cv2.VideoWriter(str(out_dir / f"{video_id}_overlay_threat.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fi, (frame, t_raw) in enumerate(zip(frames, timestamps)):
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
        if fi in grid_idx:
            tiles.append(cv2.resize(vis, (w * 2 // 5, h * 2 // 5)))
    writer.release()
    while len(tiles) < 16:
        tiles.append(np.zeros_like(tiles[0]) if tiles else np.zeros((h * 2 // 5, w * 2 // 5, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    cv2.imwrite(str(out_dir / f"{video_id}_grid16_threat.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def render_target_curves(video_id, tracks, frame_masks, scores, timestamps, win_ends, out_dir: Path):
    top_ids = sorted(tracks, key=lambda tid: -len(tracks[tid]))[:N_CURVE_OBJECTS]
    if not top_ids:
        return
    id_labels = {tid: f"id{tid} ({len(tracks[tid])} pts)" for tid in top_ids}
    dims = ["alpha", "s_end", "s_rate", "rho", "lane", "threat"]
    LANE = {"LEFT": -1.0, "IN": 0.0, "RIGHT": 1.0}
    ts_all = np.array(timestamps)
    n = len(ts_all)
    series = {tid: {d: np.full(n, np.nan) for d in dims} for tid in top_ids}
    rows = []
    for f, t_raw in enumerate(ts_all):
        t = round(float(t_raw), 4)
        for tid in top_ids:
            pts = [p for p in tracks[tid] if p[0] <= t + 1e-4]
            if not pts or abs(pts[-1][0] - t) > 1e-4:
                continue
            g = track_geometry_v2(pts, t, frame_masks)
            if g is None or g["low_samples"]:
                continue
            score_entry = scores.get(tid, {}).get(t)
            score = score_entry["score"] if score_entry is not None else np.nan
            s = series[tid]
            s["alpha"][f] = g["alpha"]
            s["s_end"][f] = np.nan if g["s_end"] is None else g["s_end"]
            s["s_rate"][f] = np.nan if g["s_rate"] is None else g["s_rate"]
            s["rho"][f] = np.nan if g["rho"] is None else g["rho"]
            s["lane"][f] = LANE.get(g["lane_state"], np.nan)
            s["threat"][f] = score
            rows.append(dict(frame=f, t=round(float(t), 3), track_id=tid, n_samples=g["n_samples"],
                             alpha=g["alpha"], alpha_mode=g["alpha_mode"], lane_overlap_s=g["s_end"],
                             lane_overlap_rate=g["s_rate"], proximity=g["rho"], lane_state=g["lane_state"],
                             selection_score=score))
    if rows:
        with open(out_dir / f"{video_id}_target_curves.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    panels = [
        ("alpha", "1. Looming α  (1/s)", "growth rate of the object's size (horizon-free: area/width/height by which edges are clean)"),
        ("s_end", "2. Lane overlap s  (kept for comparison only, not used by the score)", "share of the traced ego path covered; <0 = outside"),
        ("s_rate", "3. Lane-overlap rate  (kept for comparison only)", ""),
        ("rho", "4. Proximity ρ  (kept for comparison only)", "0 = farthest traced row, 1 = nearest"),
        ("lane", "5. Lane state  (kept for comparison only)", ""),
        ("threat", "6. Selection score", "closeness x side x (1+approach), EMA-smoothed (tau=0.3s) - picks the top-5 objects; never fit to the crash label"),
    ]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 15.5), sharex=True, facecolor=SURFACE)
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
            ax.axvline(te - ts_all[0], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=0)
        if key == "lane":
            ax.set_yticks([-1, 0, 1], ["LEFT", "IN", "RIGHT"]); ax.set_ylim(-1.5, 1.5)
        if key == "rho":
            ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold", pad=16)
        if sub:
            ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=8.5, va="bottom")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=MUTED)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=min(len(top_ids), 5),
                  frameon=False, fontsize=9, labelcolor=INK2, handlelength=3)
    axes[-1].set_xlabel(f"seconds into the decoded span  ({n} frames)", color=INK2)
    fig.suptitle(f"Clip {video_id}: Stage 3 geometry + selection score per frame (causal; longest-lived tracks shown)",
                x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"{video_id}_target_curves.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def run_clip(video_id: str, out_dir: Path):
    frame_masks, fps, span = load_frame_masks(video_id)
    tracks = load_stitched_tracks(video_id)
    frames, timestamps, _ = decode_frames(video_id, *span)

    scores = compute_track_scores(tracks, frame_masks, fps)

    _, wins = window_ends(video_id)
    wins = [(label, te) for label, te in wins if span[0] - 1e-6 <= te <= span[1] + 1e-6]

    threat_windows = compute_window_top5(wins, tracks, scores, fps)
    for win in threat_windows:
        top_desc = ", ".join(f"#{o['rank']} id{o['track_id']} ({o['score']:.2f})" for o in win["top5"])
        print(f"[{video_id}] {win['label']} (t={win['t_end']:.2f}s): top5 = {top_desc or '(none)'}")

    render_overlay(video_id, frames, timestamps, frame_masks, tracks, scores, out_dir, fps)
    render_target_curves(video_id, tracks, frame_masks, scores, timestamps, wins, out_dir)

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
