import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import yaml


def _normalize_id(value) -> str:
    return f"{int(float(value)):05d}"


# #region agent log
def _debug_log(message: str, data: Dict, hypothesis_id: str) -> None:
    payload = {
        "sessionId": "debug-session",
        "runId": os.environ.get("SLURM_JOB_ID", "manual"),
        "hypothesisId": hypothesis_id,
        "location": "build_teacher_manifest.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("/home/eprojuser011/.cursor/debug.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


# #endregion


def _load_config(path: Optional[str]) -> Dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _count_frames(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    return sum(1 for _ in frames_dir.glob("frame_*.jpg"))


def _build_end_indices(max_frame: int, window_step: int, max_per_video: int) -> List[int]:
    if max_frame < 0:
        return []
    end_indices = list(range(0, max_frame + 1, window_step))
    if max_per_video and len(end_indices) > max_per_video:
        step = max(1, math.ceil(len(end_indices) / max_per_video))
        end_indices = end_indices[::step]
    return end_indices


def _clamp_indices(indices: Sequence[int], min_end: int, max_end: int) -> List[int]:
    clamped = []
    for idx in indices:
        if idx < min_end:
            clamped.append(min_end)
        elif idx > max_end:
            clamped.append(max_end)
        else:
            clamped.append(int(idx))
    return clamped


def _build_event_centered_indices(
    *,
    target: int,
    time_of_event: Optional[float],
    num_frames: int,
    fps: float,
    clips_per_video: int,
    step_seconds: float,
    window_size: int,
    stride: int,
) -> List[int]:
    max_frame = max(num_frames - 1, 0)
    min_end = (window_size - 1) * stride
    step_frames = max(1, int(round(step_seconds * fps)))

    if target == 0:
        anchor_frame = max_frame // 2
    elif time_of_event is not None and not math.isnan(time_of_event):
        anchor_frame = int(round(time_of_event * fps))
    else:
        anchor_frame = max_frame

    end_indices = [anchor_frame - step_frames * i for i in range(1, clips_per_video + 1)]
    return _clamp_indices(end_indices, min_end=min_end, max_end=max_frame)


def _frame_indices(end_frame_idx: int, window_size: int, stride: int) -> List[int]:
    start = end_frame_idx - (window_size - 1) * stride
    indices = list(range(start, end_frame_idx + 1, stride))
    return [max(0, idx) for idx in indices]


def _log_progress(
    clip_idx: int,
    total: int,
    start_time: float,
    video_id: str,
    end_frame_idx: int,
    log_path: Optional[Path],
):
    elapsed = max(time.time() - start_time, 1e-6)
    clips_per_s = clip_idx / elapsed if clip_idx > 0 else 0.0
    remaining = max(total - clip_idx, 0)
    eta_s = remaining / clips_per_s if clips_per_s > 0 else float("inf")
    msg = (
        f"[{clip_idx}/{total}] video_id={video_id} end_frame_idx={end_frame_idx} "
        f"elapsed_s={elapsed:.1f} clips_per_s={clips_per_s:.3f} eta_s={eta_s:.1f}"
    )
    print(msg, file=sys.stderr, flush=True)
    if log_path:
        record = {
            "clip_idx": clip_idx,
            "total": total,
            "elapsed_s": elapsed,
            "clips_per_s": clips_per_s,
            "eta_s": eta_s,
            "video_id": video_id,
            "end_frame_idx": end_frame_idx,
            "timestamp": time.time(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--train_csv", required=False)
    parser.add_argument("--frames_root", required=False)
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress_jsonl", required=False)
    parser.add_argument("--max_per_video", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--window_step", type=int, default=None)
    parser.add_argument("--sampling_mode", type=str, default="uniform")
    parser.add_argument("--clips_per_video", type=int, default=16)
    parser.add_argument("--clip_step_seconds", type=float, default=0.25)
    parser.add_argument("--out_blind", required=False)
    parser.add_argument("--ignore_after_event", action="store_true")
    parser.add_argument("--limit_noncollision_half", action="store_true")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    # Ensure repo root is on sys.path when run as a script.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    _debug_log(
        "startup",
        {
            "cwd": os.getcwd(),
            "sys_path_0": sys.path[0],
            "pythonpath": os.environ.get("PYTHONPATH"),
            "repo_root": str(repo_root),
        },
        "H1",
    )

    _debug_log("import_compute_target_risk_start", {}, "H2")
    try:
        from data.dataset import compute_target_risk  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        _debug_log("import_compute_target_risk_failed", {"error": str(exc)}, "H2")
        raise
    _debug_log("import_compute_target_risk_ok", {}, "H2")

    cfg = _load_config(args.config)
    train_csv = args.train_csv or cfg.get("train_csv")
    frames_root = args.frames_root or cfg.get("frames_root")
    if not train_csv or not frames_root:
        raise SystemExit("train_csv and frames_root must be provided via args or config.")

    window_size = args.window_size or cfg.get("window_size", 16)
    stride = args.stride or cfg.get("stride", 4)
    window_step = args.window_step or cfg.get("window_step", 1)
    ignore_after_event = args.ignore_after_event or cfg.get("ignore_after_event", True)
    limit_noncollision_half = args.limit_noncollision_half or cfg.get("limit_noncollision_half", True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blind_path = Path(args.out_blind) if args.out_blind else out_path.parent / "teacher_clips_blind.jsonl"
    progress_path = Path(args.progress_jsonl) if args.progress_jsonl else out_path.with_suffix(".progress.jsonl")

    df = pd.read_csv(train_csv)
    df["id"] = df["id"].apply(_normalize_id)
    frames_root_path = Path(frames_root)

    per_video = []
    total = 0
    for _, row in df.iterrows():
        video_id = str(row["id"]).zfill(5)
        frames_dir = frames_root_path / video_id
        num_frames = _count_frames(frames_dir)
        if num_frames <= 0:
            continue
        target = int(row.get("target", 0))
        time_of_alert = row.get("time_of_alert", None)
        time_of_event = row.get("time_of_event", None)

        max_frame = num_frames - 1
        if args.sampling_mode == "event_centered":
            end_indices = _build_event_centered_indices(
                target=target,
                time_of_event=None if pd.isna(time_of_event) else float(time_of_event),
                num_frames=num_frames,
                fps=args.fps,
                clips_per_video=args.clips_per_video,
                step_seconds=args.clip_step_seconds,
                window_size=window_size,
                stride=stride,
            )
        else:
            if target == 1 and ignore_after_event and pd.notna(time_of_event):
                max_frame = min(max_frame, int(float(time_of_event) * args.fps))
            if target == 0 and limit_noncollision_half:
                max_frame = min(max_frame, num_frames // 2)
            end_indices = _build_end_indices(max_frame, window_step, args.max_per_video)
        if not end_indices:
            continue
        per_video.append((video_id, end_indices, target, time_of_alert, time_of_event))
        total += len(end_indices)

    start_time = time.time()
    clip_idx = 0
    with open(out_path, "w", encoding="utf-8") as fout, open(blind_path, "w", encoding="utf-8") as fblind:
        for video_id, end_indices, target, time_of_alert, time_of_event in per_video:
            for end_frame_idx in end_indices:
                clip_idx += 1
                t_seconds = end_frame_idx / max(args.fps, 1e-3)
                target_risk = compute_target_risk(
                    t_seconds=t_seconds,
                    time_of_alert=time_of_alert,
                    time_of_event=time_of_event,
                    target=target,
                    safe_risk=cfg.get("safe_risk", 0.0),
                    ramp_low=cfg.get("ramp_low", 0.1),
                    ramp_high=cfg.get("ramp_high", 0.9),
                )
                record = {
                    "video_id": video_id,
                    "end_frame_idx": int(end_frame_idx),
                    "window_size": int(window_size),
                    "stride": int(stride),
                    "t_seconds": float(t_seconds),
                    "target": int(target),
                    "time_of_alert": None if pd.isna(time_of_alert) else float(time_of_alert),
                    "time_of_event": None if pd.isna(time_of_event) else float(time_of_event),
                    "target_risk": float(target_risk),
                    "frame_indices": _frame_indices(int(end_frame_idx), int(window_size), int(stride)),
                }
                fout.write(json.dumps(record) + "\n")
                blind_record = {
                    key: value
                    for key, value in record.items()
                    if key not in {"target", "time_of_alert", "time_of_event", "target_risk"}
                }
                fblind.write(json.dumps(blind_record) + "\n")
                if args.log_every > 0 and (clip_idx % args.log_every == 0 or clip_idx == total):
                    _log_progress(clip_idx, total, start_time, video_id, int(end_frame_idx), progress_path)

    print(f"Wrote manifest: {out_path}", file=sys.stderr, flush=True)
    print(f"Wrote blind manifest: {blind_path}", file=sys.stderr, flush=True)
    print(f"Wrote progress log: {progress_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
