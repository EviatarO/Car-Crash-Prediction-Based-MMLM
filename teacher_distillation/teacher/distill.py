"""
Teacher distillation: generate teacher labels via OpenRouter (vision) API.
All paths are passed via CLI (--train_csv, --frames_root, --output_jsonl) so this
script runs identically on a server or a local PC; no hardcoded paths.
"""
import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from torchvision import transforms

from data.dataset import compute_target_risk
from prompts.templates import TEACHER_PROMPT

# OpenRouter: optional .env loading (works from project root on PC or server)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
def load_gemini_client():
    try:
        import google.generativeai as genai
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("google-generativeai is required for Gemini distillation") from exc
    pass


def _encode_pil_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    """Encode a PIL Image to a data URL for OpenRouter vision API."""
    buf = BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.save(buf, format=format)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _build_messages(prompt: str, image_b64s: List[str]) -> List[Dict]:
    """Build chat messages: one user message with text + image_url parts."""
    content = [{"type": "text", "text": prompt}]
    for b64 in image_b64s:
        content.append({"type": "image_url", "image_url": {"url": b64}})
    return [{"role": "user", "content": content}]


def load_openrouter_client():
    """Create OpenAI-compatible client pointed at OpenRouter."""
    from openai import OpenAI

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Set it in the environment or in a .env file."
        )
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": os.environ.get("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": os.environ.get("OPENROUTER_APP_TITLE", "MMLM_Teacher_Distill"),
        },
    )


def call_openrouter(
    client,
    model: str,
    messages: List[Dict],
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Call OpenRouter chat completion with retries."""
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
            )
            return (response.choices[0].message.content or "") if response.choices else ""
        except Exception as exc:
            last_exc = exc
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_exc}")


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
    parser = argparse.ArgumentParser(
        description="Teacher distillation via OpenRouter (run from server or PC; all paths via CLI)."
    )
    parser.add_argument("--train_csv", required=True, help="Path to training CSV (e.g. train.csv)")
    parser.add_argument("--frames_root", required=True, help="Root directory containing per-video frame folders")
    parser.add_argument("--output_jsonl", required=True, help="Output path for teacher cache JSONL")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--max_per_video", type=int, default=32)
    parser.add_argument(
        "--model",
        default="google/gemini-2.0-flash-001",
        help="OpenRouter model ID (e.g. google/gemini-2.0-flash-001, openai/gpt-4o)",
    )

    parser.add_argument("--max_retries", type=int, default=3, help="Retries per API call")
    parser.add_argument("--retry_delay", type=float, default=2.0, help="Base delay between retries (seconds)")
    args = parser.parse_args()

    client = load_openrouter_client()
    model_id = args.model

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

                image_b64s = [_encode_pil_to_base64(f) for f in frames]
                messages = _build_messages(prompt, image_b64s)
                text = call_openrouter(
                    client,
                    model_id,
                    messages,
                    max_retries=args.max_retries,
                    retry_delay=args.retry_delay,
                )

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
