"""
aa1_track_stage1.py
====================
Stage 1 driver (child plan 2026-09-15-AA1-v2b): run tracking + stitching + hood filter (see
aa1_tracks.py) on all 18 val_e3a clips from the cached YOLOPv2 detections
(outputs/aa1_v2_18clips/<vid>_yolop.json), and render the same kind of artifacts v1 produced
(outputs/aa1_smoke_18clips/), adapted to what Stage 1 actually has (tracking only - no lane
geometry or threat yet, that's Stage 2/3):

- <vid>_tracks_v2.json        raw + stitched tracks, merge log, hood-filtered ids
- <vid>_overlay_v2.mp4        every frame, boxes colored by final (post-stitch) track id
- <vid>_grid16_v2.jpg         16 evenly-spaced frames from that overlay
- <vid>_timeline_v2.png       per-track coverage over time - which raw fragments got stitched
                              into which final track, so a merge can be sanity-checked visually
                              without reading the merge log by hand

No threat/ranking numbers are produced here on purpose - Stage 1 is tracking-only, threat comes
from the lane/collision geometry built in Stage 2/3.
"""
from __future__ import annotations

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
from aa1_detect_track_rank import decode_frames, decode_span  # noqa: E402
from aa1_tracks import load_yolop_cache, track_from_yolop, stitch_fragments, is_ego_hood  # noqa: E402

VAL_E3A_IDS = ["00319", "00077", "00687", "00283", "00147", "00529", "00493", "00474",
               "00372", "01153", "01504", "01643", "01281", "01550", "01737", "02104",
               "02117", "01552"]

# Categorical palette, dataviz skill's validated 8-hue order (pre-validated for line/box
# adjacent-pair contrast) - cycle through it for however many final tracks a clip has.
PALETTE_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


PALETTE_BGR = [hex_to_bgr(h) for h in PALETTE_HEX]


def render_overlay(video_id, frames, timestamps, stitched, dropped_hood_ids, out_dir: Path, fps):
    h, w = frames[0].shape[:2]
    kept_ids = sorted(tid for tid in stitched if tid not in dropped_hood_ids)
    color_of = {tid: PALETTE_BGR[i % len(PALETTE_BGR)] for i, tid in enumerate(kept_ids)}
    by_t = {}
    for tid, pts in stitched.items():
        for t, box, _ in pts:
            by_t.setdefault(round(t, 4), []).append((tid, box))

    grid_idx = set(np.linspace(0, len(frames) - 1, 16).round().astype(int).tolist())
    tiles = []
    writer = cv2.VideoWriter(str(out_dir / f"{video_id}_overlay_v2.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for fi, (frame, t) in enumerate(zip(frames, timestamps)):
        vis = frame.copy()
        for tid, box in by_t.get(round(t, 4), []):
            x1, y1, x2, y2 = (int(v) for v in box)
            if tid in dropped_hood_ids:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (100, 100, 100), 1)
                cv2.putText(vis, f"id{tid} HOOD?", (x1, max(15, y1 - 5)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                continue
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
    cv2.imwrite(str(out_dir / f"{video_id}_grid16_v2.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])


def render_timeline(video_id, raw_tracks, stitched, merge_log, dropped_hood_ids, out_dir: Path):
    """One horizontal bar per FINAL track id, showing which raw fragment(s) it's built from
    (alternating shades) and where a stitch join happened (vertical tick)."""
    # map raw id -> final id it ended up in (final id == the "kept" id chosen by stitch_fragments)
    raw_to_final = {}
    for final_id, pts in stitched.items():
        # a stitched track's points originate from >=1 raw ids; recover via the merge log chain
        raw_to_final[final_id] = final_id
    for a, b, *_ in merge_log:
        # b was absorbed into a; a may itself later be absorbed into something else
        raw_to_final[b] = raw_to_final.get(a, a)
    # resolve chains (a<-b<-c ...) to their final root
    def resolve(tid):
        seen = set()
        while tid in raw_to_final and raw_to_final[tid] != tid and tid not in seen:
            seen.add(tid)
            tid = raw_to_final[tid]
        return tid

    final_ids = sorted(tid for tid in stitched if tid not in dropped_hood_ids)
    if not final_ids:
        return
    color_of = {tid: PALETTE_HEX[i % len(PALETTE_HEX)] for i, tid in enumerate(final_ids)}

    fig, ax = plt.subplots(figsize=(11, max(2, 0.35 * len(final_ids) + 1)), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    y = 0
    yticks, ylabels = [], []
    for final_id in final_ids:
        raw_ids_here = sorted(rid for rid in raw_tracks if resolve(rid) == final_id)
        for j, rid in enumerate(raw_ids_here):
            pts = raw_tracks[rid]
            t0, t1 = pts[0][0], pts[-1][0]
            alpha = 0.55 if j % 2 else 1.0
            ax.barh(y, t1 - t0, left=t0, height=0.6, color=color_of[final_id], alpha=alpha,
                   edgecolor="#0b0b0b", linewidth=0.5)
        yticks.append(y)
        n_frag = len(raw_ids_here)
        ylabels.append(f"id{final_id}" + (f"  ({n_frag} fragments)" if n_frag > 1 else ""))
        y += 1
    for a, b, gap, dist, sim in merge_log:
        fid = resolve(a)
        if fid in final_ids:
            gap_t = raw_tracks[b][0][0]
            yy = final_ids.index(fid)
            ax.axvline(gap_t, ymin=(yy - 0.35) / len(final_ids), ymax=(yy + 0.35) / len(final_ids),
                      color="#e34948", lw=1.5)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8, color="#0b0b0b")
    ax.set_xlabel("time (s)", color="#52514e")
    ax.set_title(f"{video_id}: track timeline (color = final id, dark tick = stitch join)",
                loc="left", color="#0b0b0b", fontsize=11, fontweight="semibold")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e1e0d9", lw=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / f"{video_id}_timeline_v2.png", dpi=110, facecolor="#fcfcfb")
    plt.close(fig)


def run_clip(video_id, out_dir: Path):
    cache = load_yolop_cache(out_dir, video_id)
    t_start, t_end = cache["span"]
    frames, timestamps, fps = decode_frames(video_id, t_start, t_end)

    raw_tracks = track_from_yolop(cache, frames)
    stitched, merge_log = stitch_fragments(raw_tracks, frames, timestamps, fps)
    dropped_hood_ids = {tid for tid, pts in stitched.items() if is_ego_hood(pts)}

    render_overlay(video_id, frames, timestamps, stitched, dropped_hood_ids, out_dir, fps)
    render_timeline(video_id, raw_tracks, stitched, merge_log, dropped_hood_ids, out_dir)

    record = dict(video_id=video_id, n_frames=len(frames), n_raw_tracks=len(raw_tracks),
                 n_stitched_tracks=len(stitched), n_merges=len(merge_log),
                 n_hood_dropped=len(dropped_hood_ids), hood_track_ids=sorted(dropped_hood_ids),
                 merge_log=[dict(kept=a, absorbed=b, gap_s=g, center_dist_frac=d, hist_sim=s)
                            for a, b, g, d, s in merge_log],
                 tracks={str(tid): [[t, box, conf] for t, box, conf in pts]
                         for tid, pts in stitched.items() if tid not in dropped_hood_ids})
    with open(out_dir / f"{video_id}_tracks_v2.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"[{video_id}] raw={len(raw_tracks)} -> stitched={len(stitched)}  "
         f"merges={len(merge_log)}  hood_dropped={len(dropped_hood_ids)}")
    return record


def main():
    out_dir = REPO / "outputs" / "aa1_v2_18clips"
    results = [run_clip(vid, out_dir) for vid in VAL_E3A_IDS]
    totals = dict(
        raw=sum(r["n_raw_tracks"] for r in results),
        stitched=sum(r["n_stitched_tracks"] for r in results),
        merges=sum(r["n_merges"] for r in results),
        hood=sum(r["n_hood_dropped"] for r in results))
    print(f"\n[summary] total raw tracks={totals['raw']}  stitched={totals['stitched']}  "
         f"merges={totals['merges']}  hood_dropped={totals['hood']}")
    with open(out_dir / "stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
