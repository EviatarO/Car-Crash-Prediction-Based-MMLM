import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from torchvision import transforms

from data.dataset import compute_target_risk
from prompts.templates import TEACHER_PROMPT


def load_gemini_client():
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-generativeai is required for Gemini distillation") from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai


def build_clip_frames(frames_dir: Path, end_frame_idx: int, window_size: int, stride: int, frame_size: int):
    transform = transforms.Compose([transforms.Resize((frame_size, frame_size))])
    indices = list(range(end_frame_idx - (window_size - 1) * stride, end_frame_idx + 1, stride))
    frames = []
    for idx in indices:
        clamped = max(0, idx)
        frame_path = frames_dir / f"frame_{clamped:05d}.jpg"
        if not frame_path.exists():
            img = Image.new("RGB", (frame_size, frame_size), color=(0, 0, 0))
        else:
            img = Image.open(frame_path).convert("RGB")
        frames.append(transform(img))
    return frames


def build_prompt(context: Dict) -> str:
    risk = context["target_risk"]
    return (
        f"{TEACHER_PROMPT}\n"
        f"Known facts: target={context['target']} "
        f"time_of_alert={context['time_of_alert']} "
        f"time_of_event={context['time_of_event']} "
        f"t={context['t_seconds']:.3f} "
        f"target_risk={risk:.3f}\n"
        "Do not contradict the provided target risk.\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--frames_root", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--max_per_video", type=int, default=32)
    args = parser.parse_args()

    genai = load_gemini_client()
    model = genai.GenerativeModel("gemini-1.5-pro")

    df = pd.read_csv(args.train_csv)
    frames_root = Path(args.frames_root)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in df.iterrows():
            video_id = str(row["id"]).zfill(5)
            frames_dir = frames_root / video_id
            frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
            if not frame_paths:
                continue
            fps = 30.0
            target = int(row.get("target", 0))
            time_of_alert = row.get("time_of_alert", None)
            time_of_event = row.get("time_of_event", None)

            max_frame = len(frame_paths) - 1
            end_indices = list(range(0, max_frame + 1, max(1, len(frame_paths) // args.max_per_video)))

            for end_idx in end_indices:
                t_seconds = end_idx / max(1e-3, fps)
                target_risk = compute_target_risk(
                    t_seconds=t_seconds,
                    time_of_alert=time_of_alert,
                    time_of_event=time_of_event,
                    target=target,
                )
                frames = build_clip_frames(
                    frames_dir,
                    end_frame_idx=end_idx,
                    window_size=args.window_size,
                    stride=args.stride,
                    frame_size=args.frame_size,
                )
                prompt = build_prompt(
                    {
                        "target": target,
                        "time_of_alert": time_of_alert,
                        "time_of_event": time_of_event,
                        "t_seconds": t_seconds,
                        "target_risk": target_risk,
                    }
                )

                response = model.generate_content([prompt] + frames)
                text = response.text if hasattr(response, "text") else str(response)

                record = {
                    "video_id": video_id,
                    "end_frame_idx": end_idx,
                    "window_size": args.window_size,
                    "stride": args.stride,
                    "target_risk": target_risk,
                    "teacher_text": text,
                }
                fout.write(json.dumps(record) + "\n")

    print(f"Wrote teacher cache: {output_path}")


if __name__ == "__main__":
    main()
