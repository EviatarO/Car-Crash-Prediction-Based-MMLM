import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from PIL import Image
from torchvision import transforms

# #region agent log
def _debug_log(message: str, data: Dict, hypothesis_id: str) -> None:
    payload = {
        "sessionId": "debug-session",
        "runId": os.environ.get("SLURM_JOB_ID", "manual"),
        "hypothesisId": hypothesis_id,
        "location": "smoke_forward.py",
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


def _load_manifest(manifest_path: Path) -> List[Dict]:
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _select_balanced(records: List[Dict], num_clips: int, seed: int = 7) -> List[Dict]:
    rng = random.Random(seed)
    pos = [r for r in records if int(r.get("target", 0)) == 1]
    neg = [r for r in records if int(r.get("target", 0)) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = num_clips // 2
    selected = pos[:half] + neg[:half]
    if len(selected) < num_clips:
        remaining = [r for r in records if r not in selected]
        rng.shuffle(remaining)
        selected += remaining[: (num_clips - len(selected))]
    rng.shuffle(selected)
    return selected[:num_clips]


def _build_prompt_batch(tokenizer, batch_size: int, prompt: str, score_token: str):
    tokens = tokenizer([prompt] * batch_size, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids
    attention_mask = tokens.attention_mask
    score_token_id = tokenizer.convert_tokens_to_ids(score_token)
    score_token_index = (input_ids == score_token_id).int().argmax(dim=1)
    return input_ids, attention_mask, score_token_index


def _load_clip(frames_root: Path, record: Dict, frame_size: int, transform) -> torch.Tensor:
    video_id = str(record["video_id"]).zfill(5)
    frame_indices = record["frame_indices"]
    frames = []
    for idx in frame_indices:
        frame_path = frames_root / video_id / f"frame_{int(idx):05d}.jpg"
        if not frame_path.exists():
            img = Image.new("RGB", (frame_size, frame_size), color=(0, 0, 0))
        else:
            img = Image.open(frame_path).convert("RGB")
        frames.append(transform(img))
    return torch.stack(frames, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frames_root", required=False)
    parser.add_argument("--num_clips", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/smoke_forward_report.json")
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

    _debug_log("import_models_start", {}, "H2")
    try:
        from models.factory import build_student_model  # pylint: disable=import-outside-toplevel
        from prompts.templates import BASE_PROMPT  # pylint: disable=import-outside-toplevel
    except Exception as exc:
        _debug_log("import_models_failed", {"error": str(exc)}, "H2")
        raise
    _debug_log("import_models_ok", {}, "H2")

    cfg = _load_config(args.config)
    frames_root = args.frames_root or cfg.get("frames_root")
    if not frames_root:
        raise SystemExit("frames_root must be provided via args or config.")
    frames_root_path = Path(frames_root)

    records = _load_manifest(Path(args.manifest))
    if not records:
        raise SystemExit("Manifest is empty.")

    sample = _select_balanced(records, args.num_clips, seed=args.seed)

    frame_size = cfg.get("frame_size", 224)
    transform = transforms.Compose(
        [
            transforms.Resize((frame_size, frame_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_student_model(
        vision_model_id=cfg["vision_model_id"],
        llm_model_id=cfg["llm_model_id"],
        score_token=cfg["score_token"],
        use_4bit=cfg.get("use_4bit", False),
        lora_r=cfg.get("lora_r", 0),
        lora_alpha=cfg.get("lora_alpha", 16),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        lora_target_modules=cfg.get("lora_target_modules"),
    ).to(device)
    model.eval()

    clip_tensors = []
    targets = []
    for idx, record in enumerate(sample, 1):
        clip = _load_clip(frames_root_path, record, frame_size, transform)
        clip_tensors.append(clip)
        targets.append(float(record.get("target_risk", 0.0)))
        print(
            f"[load {idx}/{len(sample)}] video_id={record['video_id']} end_frame_idx={record['end_frame_idx']}",
            file=sys.stderr,
            flush=True,
        )

    clip_batch = torch.stack(clip_tensors, dim=0).to(device)
    input_ids, attention_mask, score_idx = _build_prompt_batch(
        model.tokenizer, clip_batch.size(0), BASE_PROMPT, cfg["score_token"]
    )
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    score_idx = score_idx.to(device)

    visual_tokens_shape = None
    score_logits_shape = None
    latency_ms = None
    forward_ok = False

    with torch.no_grad():
        if clip_batch.is_cuda:
            torch.cuda.synchronize()
        start = time.time()
        visual_tokens = model._encode_visual(clip_batch)
        visual_tokens_shape = list(visual_tokens.shape)
        out = model(
            clip=clip_batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
            score_token_index=score_idx,
        )
        if clip_batch.is_cuda:
            torch.cuda.synchronize()
        end = time.time()
        score_logits_shape = list(out.score_logits.shape)
        latency_ms = (end - start) * 1000.0
        forward_ok = True

    report = {
        "clip_count": len(sample),
        "clip_shape": list(clip_batch.shape),
        "visual_tokens_shape": visual_tokens_shape,
        "score_logits_shape": score_logits_shape,
        "target_risk_min": min(targets) if targets else None,
        "target_risk_max": max(targets) if targets else None,
        "forward_ok": forward_ok,
        "latency_ms": latency_ms,
        "device": str(device),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote smoke report: {output_path}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
