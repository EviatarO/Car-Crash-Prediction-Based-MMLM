import argparse
import json
import math
import os
import random
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


def _build_random_tn_index(
    num_frames: int,
    window_size: int,
    stride: int,
    fps: float,
    rng: random.Random,
) -> List[int]:
    """Return a single random end-frame index for a TN clip.

    Ensures:
      - At least `(window_size - 1) * stride` frames before the point (so 16 frames fit).
      - At least 3 seconds of video remain AFTER the point (prediction horizon).
    """
    min_end = (window_size - 1) * stride
    max_end = num_frames - 1 - int(3.0 * fps)
    if max_end < min_end:
        # Video too short to satisfy both constraints; fall back to midpoint.
        return [(min_end + num_frames - 1) // 2]
    return [rng.randint(min_end, max_end)]


def _build_tte_bucket_indices(
    *,
    time_of_event: float,
    num_frames: int,
    fps: float,
    tte_buckets: List[float],
    window_size: int,
    stride: int,
) -> List[dict]:
    """Return a list of {end_frame_idx, requested_tte} dicts for each TTE bucket.

    For each bucket (e.g. 0.5, 1.0, 1.5 seconds before event), compute the
    end-frame index and include it only if it falls within valid bounds.
    """
    min_end = (window_size - 1) * stride
    event_frame = int(round(time_of_event * fps))
    results = []
    for tte in tte_buckets:
        end_frame = int(round((time_of_event - tte) * fps))
        if end_frame < min_end or end_frame > event_frame:
            continue
        results.append({"end_frame_idx": end_frame, "requested_tte": tte})
    return results


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
    # --- v11+ args for controlled 100/500 clip generation ---
    parser.add_argument("--max_tp", type=int, default=None, help="Cap total TP clips (default: no limit)")
    parser.add_argument("--max_tn", type=int, default=None, help="Cap total TN clips (default: no limit)")
    parser.add_argument("--tn_random_sampling", action="store_true", help="Use random frame selection for TN clips (1 clip per video)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    parser.add_argument("--tte_buckets", type=str, default=None, help="Comma-sep TTE seconds for TP selection (e.g. '0.5,1.0,1.5')")
    parser.add_argument("--one_tte_per_video", action="store_true", help="Assign one TTE bucket per TP video in round-robin order (instead of all buckets per video)")
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

    # Parse tte_buckets if provided.
    tte_buckets: Optional[List[float]] = None
    if args.tte_buckets:
        tte_buckets = [float(x.strip()) for x in args.tte_buckets.split(",")]

    # Shuffle rows when caps are in effect so we sample diverse videos.
    rng = random.Random(args.seed)
    if args.max_tp is not None or args.max_tn is not None:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    tp_count = 0
    tn_count = 0
    per_video = []
    total = 0
    tp_video_idx = 0  # used for round-robin TTE bucket assignment
    for _, row in df.iterrows():
        video_id = str(row["id"]).zfill(5)
        frames_dir = frames_root_path / video_id
        num_frames = _count_frames(frames_dir)
        if num_frames <= 0:
            continue
        target = int(row.get("target", 0))
        time_of_alert = row.get("time_of_alert", None)
        time_of_event = row.get("time_of_event", None)

        # --- TP clip selection ---
        if target == 1:
            if args.max_tp is not None and tp_count >= args.max_tp:
                continue

            if tte_buckets and pd.notna(time_of_event):
                if args.one_tte_per_video:
                    # Round-robin: pick exactly 1 TTE bucket for this video.
                    tte_to_try = [tte_buckets[tp_video_idx % len(tte_buckets)]]
                    tp_video_idx += 1
                else:
                    # Original: all buckets per video.
                    tte_to_try = tte_buckets
                bucket_results = _build_tte_bucket_indices(
                    time_of_event=float(time_of_event),
                    num_frames=num_frames,
                    fps=args.fps,
                    tte_buckets=tte_to_try,
                    window_size=window_size,
                    stride=stride,
                )
                if not bucket_results:
                    continue
                # Check if adding results would exceed cap.
                remaining = args.max_tp - tp_count if args.max_tp is not None else len(bucket_results)
                bucket_results = bucket_results[:remaining]
                for br in bucket_results:
                    end_frame_idx = br["end_frame_idx"]
                    requested_tte = br["requested_tte"]
                    per_video.append((video_id, [end_frame_idx], target, time_of_alert, time_of_event, requested_tte))
                    tp_count += 1
                    total += 1
                continue
            else:
                # Original sampling modes.
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
                    if ignore_after_event and pd.notna(time_of_event):
                        max_frame = min(max_frame, int(float(time_of_event) * args.fps))
                    end_indices = _build_end_indices(max_frame, window_step, args.max_per_video)
                if not end_indices:
                    continue
                per_video.append((video_id, end_indices, target, time_of_alert, time_of_event, None))
                tp_count += len(end_indices)
                total += len(end_indices)
                continue

        # --- TN clip selection ---
        if target == 0:
            if args.max_tn is not None and tn_count >= args.max_tn:
                continue

            if args.tn_random_sampling:
                # Random frame selection: 1 clip per TN video, full video range.
                end_indices = _build_random_tn_index(num_frames, window_size, stride, args.fps, rng)
            elif args.sampling_mode == "event_centered":
                end_indices = _build_event_centered_indices(
                    target=target,
                    time_of_event=None,
                    num_frames=num_frames,
                    fps=args.fps,
                    clips_per_video=args.clips_per_video,
                    step_seconds=args.clip_step_seconds,
                    window_size=window_size,
                    stride=stride,
                )
            else:
                max_frame = num_frames - 1
                if limit_noncollision_half:
                    max_frame = min(max_frame, num_frames // 2)
                end_indices = _build_end_indices(max_frame, window_step, args.max_per_video)
            if not end_indices:
                continue
            per_video.append((video_id, end_indices, target, time_of_alert, time_of_event, None))
            tn_count += len(end_indices)
            total += len(end_indices)

    print(
        f"Selected {total} clips: {tp_count} TP + {tn_count} TN "
        f"from {len(per_video)} video entries",
        file=sys.stderr,
        flush=True,
    )

    start_time = time.time()
    clip_idx = 0
    with open(out_path, "w", encoding="utf-8") as fout, open(blind_path, "w", encoding="utf-8") as fblind:
        for video_id, end_indices, target, time_of_alert, time_of_event, requested_tte in per_video:
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
                    "gt_verdict": "YES" if target == 1 else "NO",
                    "time_of_alert": None if pd.isna(time_of_alert) else float(time_of_alert),
                    "time_of_event": None if pd.isna(time_of_event) else float(time_of_event),
                    "target_risk": float(target_risk),
                    "requested_time_to_event": requested_tte,
                    "frame_indices": _frame_indices(int(end_frame_idx), int(window_size), int(stride)),
                }
                fout.write(json.dumps(record) + "\n")
                blind_record = {
                    key: value
                    for key, value in record.items()
                    if key not in {"target", "time_of_alert", "time_of_event", "target_risk", "gt_verdict"}
                }
                fblind.write(json.dumps(blind_record) + "\n")
                if args.log_every > 0 and (clip_idx % args.log_every == 0 or clip_idx == total):
                    _log_progress(clip_idx, total, start_time, video_id, int(end_frame_idx), progress_path)

    print(f"Wrote manifest: {out_path}", file=sys.stderr, flush=True)
    print(f"Wrote blind manifest: {blind_path}", file=sys.stderr, flush=True)
    print(f"Wrote progress log: {progress_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
