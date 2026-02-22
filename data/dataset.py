import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .video_index import VideoInfo, build_video_index


@dataclass
class ClipSample:
    video_id: str
    end_frame_idx: int
    fps: float
    target: int
    time_of_alert: Optional[float]
    time_of_event: Optional[float]


def compute_target_risk(
    t_seconds: float,
    time_of_alert: Optional[float],
    time_of_event: Optional[float],
    target: int,
    safe_risk: float = 0.0,
    ramp_low: float = 0.1,
    ramp_high: float = 0.9,
) -> float:
    if target == 0:
        return safe_risk
    if time_of_alert is None or math.isnan(time_of_alert):
        return safe_risk
    if time_of_event is None or math.isnan(time_of_event):
        return ramp_low
    if t_seconds < time_of_alert:
        return safe_risk
    if t_seconds >= time_of_event:
        return 1.0
    span = max(time_of_event - time_of_alert, 1e-3)
    ratio = (t_seconds - time_of_alert) / span
    return float(ramp_low + ratio * (ramp_high - ramp_low))


class MCASlidingWindowDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        frames_root: str,
        videos_dir: Optional[str],
        teacher_jsonl: Optional[str] = None,
        window_size: int = 16,
        stride: int = 4,
        frame_size: int = 224,
        max_windows_per_video: int = 128,
        window_step: int = 1,
        safe_risk: float = 0.0,
        ramp_low: float = 0.1,
        ramp_high: float = 0.9,
        ignore_after_event: bool = True,
        limit_noncollision_half: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        # Normalize ids to zero-padded string to match frame folders.
        self.df["id"] = self.df["id"].apply(lambda x: f"{int(float(x)):05d}")
        self.frames_root = Path(frames_root)
        self.videos_dir = Path(videos_dir) if videos_dir else None
        self.window_size = window_size
        self.stride = stride
        self.max_windows_per_video = max_windows_per_video
        self.window_step = window_step
        self.safe_risk = safe_risk
        self.ramp_low = ramp_low
        self.ramp_high = ramp_high
        self.ignore_after_event = ignore_after_event
        self.limit_noncollision_half = limit_noncollision_half

        self.teacher_map = {}
        if teacher_jsonl:
            with open(teacher_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    record = line.strip()
                    if not record:
                        continue
                    obj = json.loads(record)
                    key = (str(obj["video_id"]).zfill(5), int(obj["end_frame_idx"]))
                    self.teacher_map[key] = obj.get("teacher_text", "")

        self.transform = transforms.Compose(
            [
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
            ]
        )

        # #region agent log
        def _log(message, data, hypothesis_id):
            payload = {
                "sessionId": "debug-session",
                "runId": os.environ.get("SLURM_JOB_ID", "manual"),
                "hypothesisId": hypothesis_id,
                "location": "dataset.py:init",
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

        _log(
            "dataset_init",
            {
                "csv_path": csv_path,
                "frames_root": str(self.frames_root),
                "videos_dir": str(self.videos_dir) if self.videos_dir else None,
                "window_size": self.window_size,
                "stride": self.stride,
                "max_windows_per_video": self.max_windows_per_video,
            },
            "H1",
        )

        video_ids = [str(v).zfill(5) for v in self.df["id"].tolist()]
        self.video_index: Dict[str, VideoInfo] = build_video_index(
            video_ids=video_ids, frames_root=self.frames_root, videos_dir=self.videos_dir
        )
        _log(
            "video_index_summary",
            {
                "video_count": len(self.video_index),
                "nonzero_frames": sum(1 for v in self.video_index.values() if v.num_frames > 0),
            },
            "H2",
        )

        self.samples: List[ClipSample] = self._build_samples()
        _log("sample_count", {"samples": len(self.samples)}, "H3")

    def _build_samples(self) -> List[ClipSample]:
        samples: List[ClipSample] = []
        for _, row in self.df.iterrows():
            video_id = str(row["id"]).zfill(5)
            info = self.video_index.get(video_id)
            if not info or info.num_frames == 0:
                # #region agent log
                try:
                    with open("/home/eprojuser011/.cursor/debug.log", "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": os.environ.get("SLURM_JOB_ID", "manual"),
                                    "hypothesisId": "H2",
                                    "location": "dataset.py:_build_samples",
                                    "message": "skip_video_no_frames",
                                    "data": {"video_id": video_id},
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
                continue
            target = int(row.get("target", 0))
            time_of_alert = row.get("time_of_alert", None)
            time_of_event = row.get("time_of_event", None)

            max_frame = info.num_frames - 1
            if target == 1 and self.ignore_after_event and pd.notna(time_of_event):
                max_frame = min(max_frame, int(float(time_of_event) * info.fps))
            if target == 0 and self.limit_noncollision_half:
                max_frame = min(max_frame, info.num_frames // 2)

            end_indices = list(range(0, max_frame + 1, self.window_step))
            if self.max_windows_per_video and len(end_indices) > self.max_windows_per_video:
                step = max(1, len(end_indices) // self.max_windows_per_video)
                end_indices = end_indices[::step]

            if not end_indices:
                # #region agent log
                try:
                    with open("/home/eprojuser011/.cursor/debug.log", "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "sessionId": "debug-session",
                                    "runId": os.environ.get("SLURM_JOB_ID", "manual"),
                                    "hypothesisId": "H3",
                                    "location": "dataset.py:_build_samples",
                                    "message": "no_end_indices",
                                    "data": {
                                        "video_id": video_id,
                                        "num_frames": info.num_frames,
                                        "max_frame": max_frame,
                                    },
                                    "timestamp": int(time.time() * 1000),
                                }
                            )
                            + "\n"
                        )
                except Exception:
                    pass
                # #endregion
            for end_frame_idx in end_indices:
                samples.append(
                    ClipSample(
                        video_id=video_id,
                        end_frame_idx=end_frame_idx,
                        fps=info.fps,
                        target=target,
                        time_of_alert=time_of_alert,
                        time_of_event=time_of_event,
                    )
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _frame_path(self, video_id: str, frame_idx: int) -> str:
        frame_name = f"frame_{frame_idx:05d}.jpg"
        return str(self.frames_root / video_id / frame_name)

    def _load_clip(self, video_id: str, end_frame_idx: int) -> torch.Tensor:
        frames: List[torch.Tensor] = []
        start = end_frame_idx - (self.window_size - 1) * self.stride
        for idx in range(start, end_frame_idx + 1, self.stride):
            clamped = max(0, idx)
            frame_path = self._frame_path(video_id, clamped)
            if not os.path.exists(frame_path):
                img = Image.new("RGB", (256, 256), color=(0, 0, 0))
            else:
                img = Image.open(frame_path).convert("RGB")
            frames.append(self.transform(img))
        clip = torch.stack(frames, dim=0)  # (T, C, H, W)
        return clip

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        clip = self._load_clip(sample.video_id, sample.end_frame_idx)
        t_seconds = sample.end_frame_idx / max(sample.fps, 1e-3)
        target_risk = compute_target_risk(
            t_seconds=t_seconds,
            time_of_alert=sample.time_of_alert,
            time_of_event=sample.time_of_event,
            target=sample.target,
            safe_risk=self.safe_risk,
            ramp_low=self.ramp_low,
            ramp_high=self.ramp_high,
        )
        teacher_text = self.teacher_map.get((sample.video_id, sample.end_frame_idx), "")
        return {
            "video_id": sample.video_id,
            "end_frame_idx": sample.end_frame_idx,
            "clip": clip,
            "target_risk": torch.tensor(target_risk, dtype=torch.float32),
            "teacher_text": teacher_text,
            "t_seconds": torch.tensor(t_seconds, dtype=torch.float32),
            "target": torch.tensor(sample.target, dtype=torch.int64),
            "time_of_alert": torch.tensor(
                float(sample.time_of_alert) if sample.time_of_alert is not None else float("nan"),
                dtype=torch.float32,
            ),
            "time_of_event": torch.tensor(
                float(sample.time_of_event) if sample.time_of_event is not None else float("nan"),
                dtype=torch.float32,
            ),
        }
