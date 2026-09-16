"""
aa1_stage2.py
=============
Stage 2 driver (child plan 2026-09-15-AA1-v2b): lane/drivable-area path tracing (aa1_lanes.py)
+ per-object geometry (alpha, lateral position/rate, proximity - see track_geometry_v2), on top
of Stage 1's stitched tracks. NO THREAT/RANKING YET - that needs the collision check (Stage 3).

Per clip, in outputs/aa1_v2_18clips/:
  <vid>_geometry_v2.json     per-window, per-object geometry (all fields track_geometry_v2 gives)
  <vid>_overlay_lanes.mp4    every frame: drivable(green)/lane(red) masks, traced path (cyan/
                             magenta dots), tracked boxes
  <vid>_grid16_lanes.jpg     16-frame grid from that overlay
  <vid>_target_curves.png/csv  per-frame alpha/s_end/s_rate/rho/lane_state curves for the
                             clip's longest-lived tracks (no ranking exists yet to pick "top-K
                             by threat", so this stage plots the longest tracks instead); a
                             threat panel is drawn but annotated "not computed until Stage 3"
                             rather than left blank, per project convention.
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
from aa1_lanes import rle_decode, trace_path, fill_vehicle_gaps, path_bounds_at, track_geometry_v2  # noqa: E402
from aa1_track_stage1 import VAL_E3A_IDS, PALETTE_HEX, PALETTE_BGR  # noqa: E402

OUT_DIR = REPO / "outputs" / "aa1_v2_18clips"
N_CURVE_OBJECTS = 5


def load_frame_masks(video_id: str) -> dict:
    """t -> dict(drivable, lane, boxes, path). Decodes every cached YOLOPv2 frame once;
    reused across every object/window for this clip - decoding + tracing is not free
    (~5-10ms/frame), so this must not be repeated per object."""
    with open(OUT_DIR / f"{video_id}_yolop.json", encoding="utf-8") as f:
        cache = json.load(f)
    out = {}
    for fr in cache["frames"]:
        drivable = rle_decode(fr["drivable_rle"])
        lane = rle_decode(fr["lane_rle"])
        boxes = np.array(fr["boxes"], dtype=np.float32).reshape(-1, 4)
        path = trace_path(drivable, lane, vehicle_boxes=boxes)
        out[round(fr["t"], 4)] = dict(drivable=drivable, lane=lane, boxes=boxes, path=path)
    return out, cache["fps"], cache["span"]


def load_stitched_tracks(video_id: str) -> dict:
    with open(OUT_DIR / f"{video_id}_tracks_v2.json", encoding="utf-8") as f:
        rec = json.load(f)
    return {int(tid): [(t, box, conf) for t, box, conf in pts] for tid, pts in rec["tracks"].items()}


def render_overlay(video_id, frames, timestamps, frame_masks, tracks, out_dir: Path, fps):
    h, w = frames[0].shape[:2]
    color_of = {tid: PALETTE_BGR[i % len(PALETTE_BGR)] for i, tid in enumerate(sorted(tracks))}
    by_t = {}
    for tid, pts in tracks.items():
        for t, box, _ in pts:
            by_t.setdefault(round(t, 4), []).append((tid, box))

    grid_idx = set(np.linspace(0, len(frames) - 1, 16).round().astype(int).tolist())
    tiles = []
    writer = cv2.VideoWriter(str(out_dir / f"{video_id}_overlay_lanes.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fi, (frame, t) in enumerate(zip(frames, timestamps)):
        vis = frame.copy()
        fm = frame_masks.get(round(t, 4))
        if fm is not None:
            filled = fill_vehicle_gaps(fm["drivable"], fm["boxes"])
            vis[filled] = (vis[filled] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
            vis[fm["lane"]] = (vis[fm["lane"]] * 0.3 + np.array([0, 0, 255]) * 0.7).astype(np.uint8)
            for y, p in fm["path"].items():
                cv2.circle(vis, (int(p["x_left"]), y), 2, (255, 255, 0), -1)
                cv2.circle(vis, (int(p["x_right"]), y), 2, (255, 0, 255), -1)
        for tid, box in by_t.get(round(t, 4), []):
            x1, y1, x2, y2 = (int(v) for v in box)
            color = color_of[tid]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"id{tid}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(vis, f"{video_id}  t={t:.2f}s", (10, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (255, 255, 255), 3)
        writer.write(vis)
        if fi in grid_idx:
            tiles.append(cv2.resize(vis, (w * 2 // 5, h * 2 // 5)))
    writer.release()
    while len(tiles) < 16:
        tiles.append(np.zeros_like(tiles[0]) if tiles else np.zeros((h * 2 // 5, w * 2 // 5, 3), np.uint8))
    grid = np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(4)])
    cv2.imwrite(str(out_dir / f"{video_id}_grid16_lanes.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def render_target_curves(video_id, tracks, frame_masks, timestamps, win_ends, out_dir: Path):
    top_ids = sorted(tracks, key=lambda tid: -len(tracks[tid]))[:N_CURVE_OBJECTS]
    if not top_ids:
        return
    id_labels = {tid: f"id{tid} ({len(tracks[tid])} pts)" for tid in top_ids}
    dims = ["alpha", "s_end", "s_rate", "rho", "lane"]
    LANE = {"LEFT": -1.0, "IN": 0.0, "RIGHT": 1.0}
    ts_all = np.array(timestamps)
    n = len(ts_all)
    series = {tid: {d: np.full(n, np.nan) for d in dims} for tid in top_ids}
    rows = []
    for f, t_raw in enumerate(ts_all):
        # BUG FIXED 2026-09-15: track/frame-mask timestamps are cached rounded to 4dp
        # (aa1_yolop_cache.py), but decode_frames()'s raw `i/fps` timestamps are not - the two
        # differ by ~1e-4s, well past a 1e-6 tolerance, so the exact-time match below silently
        # matched nothing on every frame (confirmed: 00687's curves came out entirely empty
        # despite 5 tracks with 85-93 points each). Round to match the cache's own precision.
        t = round(float(t_raw), 4)
        for tid in top_ids:
            pts = [p for p in tracks[tid] if p[0] <= t + 1e-4]
            if not pts or abs(pts[-1][0] - t) > 1e-4:
                continue
            g = track_geometry_v2(pts, t, frame_masks)
            if g is None or g["low_samples"]:
                continue
            s = series[tid]
            s["alpha"][f] = g["alpha"]
            s["s_end"][f] = np.nan if g["s_end"] is None else g["s_end"]
            s["s_rate"][f] = np.nan if g["s_rate"] is None else g["s_rate"]
            s["rho"][f] = np.nan if g["rho"] is None else g["rho"]
            s["lane"][f] = LANE.get(g["lane_state"], np.nan)
            rows.append(dict(frame=f, t=round(float(t), 3), track_id=tid, n_samples=g["n_samples"],
                             alpha=g["alpha"], alpha_mode=g["alpha_mode"], lane_overlap_s=g["s_end"],
                             lane_overlap_rate=g["s_rate"], proximity=g["rho"], lane_state=g["lane_state"]))
    if rows:
        with open(out_dir / f"{video_id}_target_curves.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
    panels = [
        ("alpha", "1. Looming α  (1/s)", "growth rate of the object's size (horizon-free: area/width/height by which edges are clean)"),
        ("s_end", "2. Lane overlap s  (lane widths)", "share of the traced ego path covered; <0 = outside; measured from the drivable/lane mask, not a physics formula"),
        ("s_rate", "3. Lane-overlap rate  (lane widths / s)", ""),
        ("rho", "4. Proximity ρ", "0 = farthest traced row, 1 = nearest traced row (per-frame, from the traced path's own extent)"),
        ("lane", "5. Lane state", ""),
        ("threat", "6. Threat", "NOT COMPUTED UNTIL STAGE 3 - needs the collision check over alpha + lateral geometry"),
    ]
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10})
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 15.5), sharex=True, facecolor=SURFACE)
    x = ts_all - ts_all[0]
    for ax, (key, title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        if key == "threat":
            ax.text(0.5, 0.5, "not computed until Stage 3", transform=ax.transAxes,
                   ha="center", va="center", color=MUTED, fontsize=11, style="italic")
            ax.set_xticks([]); ax.set_yticks([])
        else:
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
    fig.suptitle(f"Clip {video_id}: Stage 2 geometry per frame (causal; longest-lived tracks shown)",
                x=0.06, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_dir / f"{video_id}_target_curves.png", dpi=110, facecolor=SURFACE)
    plt.close(fig)


def run_clip(video_id: str, out_dir: Path):
    frame_masks, fps, span = load_frame_masks(video_id)
    tracks = load_stitched_tracks(video_id)
    frames, timestamps, _ = decode_frames(video_id, *span)

    _, wins = window_ends(video_id)
    wins = [(label, te) for label, te in wins if span[0] - 1e-6 <= te <= span[1] + 1e-6]

    window_records = []
    for label, te in wins:
        objects = {}
        for tid, pts in tracks.items():
            causal = [p for p in pts if p[0] <= te + 1e-6]
            if not causal:
                continue
            g = track_geometry_v2(causal, te, frame_masks)
            if g is not None:
                objects[tid] = g
        window_records.append(dict(label=label, t_end=round(te, 3), n_objects=len(objects),
                                   objects={str(tid): g for tid, g in objects.items()}))
        n_invalid = sum(1 for g in objects.values() if g["lane_state"] == "INVALID")
        print(f"[{video_id}] {label} (t={te:.2f}s): {len(objects)} objects, {n_invalid} INVALID")

    render_overlay(video_id, frames, timestamps, frame_masks, tracks, out_dir, fps)
    render_target_curves(video_id, tracks, frame_masks, timestamps, wins, out_dir)

    with open(out_dir / f"{video_id}_geometry_v2.json", "w", encoding="utf-8") as f:
        json.dump(dict(video_id=video_id, span=list(span), fps=fps, windows=window_records), f, indent=2)
    return window_records


def main():
    out_dir = OUT_DIR
    all_windows = []
    for vid in VAL_E3A_IDS:
        all_windows.extend(run_clip(vid, out_dir))
    total_objects = sum(w["n_objects"] for w in all_windows)
    total_invalid = sum(sum(1 for g in w["objects"].values() if g["lane_state"] == "INVALID")
                        for w in all_windows)
    print(f"\n[summary] {len(all_windows)} windows, {total_objects} object-window geometries, "
         f"{total_invalid} INVALID ({100*total_invalid/max(total_objects,1):.1f}%)")


if __name__ == "__main__":
    main()
