import time
from pathlib import Path

import torch
import yaml
from torchvision import transforms
from PIL import Image

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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--video_id", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_student_model(
        vision_model_id=cfg["vision_model_id"],
        llm_model_id=cfg["llm_model_id"],
        score_token=cfg["score_token"],
    ).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((cfg["frame_size"], cfg["frame_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]
    )

    frames_root = Path(cfg["frames_root"])
    if args.video_id:
        video_id = str(args.video_id).zfill(5)
    else:
        video_id = sorted([p.name for p in frames_root.iterdir() if p.is_dir()])[0]
    frames_dir = frames_root / video_id

    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    window_size = cfg["window_size"]
    stride = cfg["stride"]

    latencies = []
    for i in range(args.num_samples):
        t = min(i, len(frame_paths) - 1)
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

        start = time.time()
        with torch.no_grad():
            _ = model(clip=clip, input_ids=input_ids, attention_mask=attention_mask, score_token_index=score_idx)
        torch.cuda.synchronize() if device.type == "cuda" else None
        latencies.append(time.time() - start)

    avg = sum(latencies) / max(len(latencies), 1)
    print(f"Average latency over {len(latencies)} samples: {avg:.4f}s")


if __name__ == "__main__":
    main()
