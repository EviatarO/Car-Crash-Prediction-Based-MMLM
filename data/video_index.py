import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2


@dataclass
class VideoInfo:
    video_id: str
    fps: float
    num_frames: int
    frames_dir: str
    video_path: Optional[str]


def _count_frames(frames_dir: Path) -> int:
    if not frames_dir.exists():
        return 0
    return len([p for p in frames_dir.iterdir() if p.suffix.lower() == ".jpg"])


def _get_fps(video_path: Optional[Path]) -> float:
    if not video_path or not video_path.exists():
        return 30.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 0:
        return float(fps)
    return 30.0


def build_video_index(
    video_ids,
    frames_root: Path,
    videos_dir: Optional[Path] = None,
    cache_path: Optional[Path] = None,
) -> Dict[str, VideoInfo]:
    index: Dict[str, VideoInfo] = {}

    for vid in video_ids:
        video_id = str(vid).zfill(5)
        frames_dir = frames_root / video_id
        video_path = None
        if videos_dir is not None:
            candidate = videos_dir / f"{video_id}.mp4"
            if candidate.exists():
                video_path = candidate
        fps = _get_fps(video_path)
        num_frames = _count_frames(frames_dir)
        index[video_id] = VideoInfo(
            video_id=video_id,
            fps=fps,
            num_frames=num_frames,
            frames_dir=str(frames_dir),
            video_path=str(video_path) if video_path else None,
        )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({k: vars(v) for k, v in index.items()}, f, indent=2)

    return index


def load_video_index(cache_path: Path) -> Dict[str, VideoInfo]:
    with open(cache_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: VideoInfo(**v) for k, v in raw.items()}
