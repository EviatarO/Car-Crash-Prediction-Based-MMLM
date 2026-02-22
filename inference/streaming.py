import csv
from collections import deque
from pathlib import Path

import torch
import yaml
from PIL import Image
import cv2
from torchvision import transforms

from models.factory import build_student_model
from prompts.templates import BASE_PROMPT


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt_batch(tokenizer, batch_size: int, prompt: str, score_token: str):
    tokens = tokenizer([prompt] * batch_size, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    score_token_id = tokenizer.convert_tokens_to_ids(score_token)
    score_token_index = (input_ids == score_token_id).int().argmax(dim=1)
    return input_ids, attention_mask, score_token_index


def load_frame(path: Path, transform):
    if not path.exists():
        img = Image.new("RGB", (256, 256), color=(0, 0, 0))
    else:
        img = Image.open(path).convert("RGB")
    return transform(img)


def get_fps(video_path: Path) -> float:
    if not video_path.exists():
        return 30.0
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 0:
        return float(fps)
    return 30.0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--video_id", required=True)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_student_model(
        vision_model_id=cfg["vision_model_id"],
        llm_model_id=cfg["llm_model_id"],
        score_token=cfg["score_token"],
    ).to(device)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((cfg["frame_size"], cfg["frame_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]
    )

    video_id = str(args.video_id).zfill(5)
    frames_dir = Path(cfg["frames_root"]) / video_id
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    videos_dir = Path(cfg.get("train_videos_dir", "")) if cfg.get("train_videos_dir") else None
    fps = 30.0
    if videos_dir:
        fps = get_fps(videos_dir / f"{video_id}.mp4")

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"{video_id}_curve.csv"

    window_size = cfg["window_size"]
    stride = cfg["stride"]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "timestamp_s", "risk", "reasoning"])

        for t in range(len(frame_paths)):
            indices = list(range(t - (window_size - 1) * stride, t + 1, stride))
            frames = []
            for idx in indices:
                clamped = max(0, idx)
                frame_path = frames_dir / f"frame_{clamped:05d}.jpg"
                frames.append(load_frame(frame_path, transform))

            clip = torch.stack(frames, dim=0).unsqueeze(0).to(device)
            input_ids, attention_mask, score_idx = build_prompt_batch(
                model.tokenizer, 1, BASE_PROMPT, cfg["score_token"]
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            score_idx = score_idx.to(device)

            with torch.no_grad():
                out = model(clip=clip, input_ids=input_ids, attention_mask=attention_mask, score_token_index=score_idx)
                risk = torch.sigmoid(out.score_logits).item()

            timestamp = t / max(1e-3, fps)
            writer.writerow([t, f"{timestamp:.3f}", f"{risk:.4f}", ""])

    print(f"Wrote curve: {out_csv}")


if __name__ == "__main__":
    main()
