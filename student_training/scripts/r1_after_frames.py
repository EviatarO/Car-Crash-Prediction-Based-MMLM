"""
r1_after_frames.py - cut the "what happened next" frames (clip C) for the E7 hindsight prompt.

For a pilot window ending at frame f_end with spacing s (= last two window indices), clip C is f_end + s*k, k = 1.. while the frame is
<= event frame + 8 (0.27 s past the collision; for normal videos the fake event at the video midpoint), capped at 16 frames and at the
video end. Frames are written at native resolution, quality 95, to outputs/teacher_pilot_2026-10/after/<frames_dir>/frame_XXXXX.jpg
(raw, no box).  Writes after/index.json = {frames_dir: {n_after, frame_idx, event_idx}}.

    python r1_after_frames.py --from-run E3        # the windows E3 got wrong (15)
    python r1_after_frames.py --all                # every set-A window
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from build_midpoint_negatives import SRC_VIDEOS, get_video_meta, read_window_sequential  # noqa: E402

OUT = Path(__import__("os").environ.get("R1_PILOT_OUT") or ROOT / "outputs" / "teacher_pilot_2026-10")   # R1_PILOT_OUT: separate test folders
AFTER = OUT / "after"
MAX_AFTER = 16
PAST_EVENT_FRAMES = 8


def after_indices(w, total):
    idx = w["frame_idx"]
    step = idx[-1] - idx[-2]
    ev_s = w["time_of_event_s"] if w["label"] else w["fake_event_s"]
    event_idx = round(ev_s * w["fps"])
    last = min(total - 1, event_idx + PAST_EVENT_FRAMES)
    out, k = [], 1
    while idx[-1] + step * k <= last and len(out) < MAX_AFTER:
        out.append(idx[-1] + step * k)
        k += 1
    return out, event_idx


def failures(run_name):
    run = {r["frames_dir"]: r for r in map(json.loads, open(OUT / "runs" / f"{run_name}.jsonl", encoding="utf-8"))}
    return [fd for fd, r in run.items() if r.get("parsed") and (r["parsed"]["collision"] == "yes") != bool(r["label"])]


def ensure(wins):
    """Cut (if missing) the after-frames of every window and update after/index.json. Returns the index."""
    AFTER.mkdir(exist_ok=True)
    ip = AFTER / "index.json"
    index = json.load(open(ip)) if ip.exists() else {}
    for w in wins:
        fd = w["frames_dir"]
        if fd in index and (AFTER / fd).exists() and len(list((AFTER / fd).glob("frame_*.jpg"))) == index[fd]["n_after"]:
            continue
        _, total = get_video_meta(w["video_id"])
        idxs, ev = after_indices(w, total)
        d = AFTER / fd
        if not (d.exists() and len(list(d.glob("frame_*.jpg"))) == len(idxs)):
            cap = cv2.VideoCapture(str(SRC_VIDEOS / f"{w['video_id']}.mp4"))
            try:
                frames = read_window_sequential(cap, idxs)
            finally:
                cap.release()
            d.mkdir(parents=True, exist_ok=True)
            for j, fr in enumerate(frames, 1):
                cv2.imwrite(str(d / f"frame_{j:05d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        index[fd] = {"n_after": len(idxs), "frame_idx": idxs, "event_idx": ev, "label": w["label"], "horizon": w["horizon"]}
        print(f"{fd}: {len(idxs)} frames after the window (event frame {ev}, window ends {w['frame_idx'][-1]})", flush=True)
    ip.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    wins = [json.loads(l) for l in open(OUT / "windows.jsonl", encoding="utf-8")]
    wins = [w for w in wins if w["set"] in ("A", "R")]
    if args.from_run:
        keep = set(failures(args.from_run))
        wins = [w for w in wins if w["frames_dir"] in keep]
    elif not args.all:
        raise SystemExit("pass --from-run RUN or --all")
    ensure(wins)


if __name__ == "__main__":
    main()
