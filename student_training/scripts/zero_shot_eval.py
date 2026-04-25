"""
zero_shot_eval.py
=================
Zero-shot evaluation of InternVL3.5-4B-Flash on collision prediction clips.

No training, no LoRA — raw pre-trained model performance using PROMPT_G.
This establishes the baseline that all trained models must beat.

MODEL: OpenGVLab/InternVL3_5-4B-Flash
  - 16 frames passed as separate images (native multi-image mode)
  - PROMPT_G used as the text prompt (same as Teacher)
  - JSON output parsed → collision_verdict + confidence → float score
  - score used for AP / AUC-ROC computation

TWO EVALUATION MODES:
  1. Teacher clips (train set baseline):
       --manifest outputs/manifest_v11_100clips.jsonl
       --frames_root <path>/train_frames256
       --output outputs/zero_shot/zero_shot_teacher_100.jsonl

  2. Test clips (held-out test set):
       --manifest outputs/test_manifest_private.jsonl
       --frames_root <path>/test_frames256
       --output outputs/zero_shot/zero_shot_test.jsonl

REQUIREMENTS:
  GPU with >= 10GB VRAM (RunPod RTX 4090 recommended)
  Local PC (RTX 1000 Ada 6GB): use --load_in_4bit flag (slower, CPU-offload)

Usage:
  python student_training/scripts/zero_shot_eval.py \
    --manifest     outputs/manifest_v11_100clips.jsonl \
    --frames_root  /data/train_frames256 \
    --output       outputs/zero_shot/zero_shot_teacher_100.jsonl \
    --config       student_training/configs/zero_shot.yaml \
    [--load_in_4bit]   # force 4-bit quantization
    [--device      cuda]
    [--resume]         # skip already-processed video_ids in output file
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# ── Add project root to path so we can import from prompts/ ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.templates import PROMPT_G  # noqa: E402

# ── Image preprocessing constants (InternVL standard) ────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# =============================================================================
# Image helpers
# =============================================================================

def build_transform(input_size: int) -> T.Compose:
    """Standard InternVL image preprocessing pipeline."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_frames(frames_root: str, video_id: str, frame_indices: list[int],
                frame_size: int, pattern: str = "frame_{:05d}.jpg") -> torch.Tensor:
    """
    Load a list of frame files and return a stacked tensor.

    Returns:
        pixel_values: (N, 3, frame_size, frame_size) float tensor
    """
    transform = build_transform(frame_size)
    tensors = []
    for idx in frame_indices:
        path = os.path.join(frames_root, video_id, pattern.format(idx))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Frame not found: {path}")
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)  # (N, 3, H, W)


# =============================================================================
# Model loading
# =============================================================================

def load_model(model_id: str, load_in_4bit: bool, torch_dtype_str: str, device: str):
    """
    Load InternVL3.5-4B-Flash with trust_remote_code.
    Returns (model, tokenizer).
    """
    from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

    print(f"\nLoading model: {model_id}")
    print(f"  4-bit quantization : {load_in_4bit}")
    print(f"  dtype              : {torch_dtype_str}")
    print(f"  device             : {device}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map.get(torch_dtype_str, torch.bfloat16)

    model_kwargs = dict(
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch_dtype
        if device != "cpu":
            model_kwargs["device_map"] = {"": 0}  # explicit GPU 0, avoids transformers all_tied_weights_keys bug

    model = AutoModel.from_pretrained(model_id, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Model loaded ({total_params:.2f}B params)")
    return model, tokenizer


# =============================================================================
# Prompt building
# =============================================================================

def build_prompt_with_images(num_frames: int) -> str:
    """
    Prepend per-frame <image> tokens to PROMPT_G.

    InternVL multi-image format: each <image> token corresponds to one image
    in the pixel_values tensor, in order.

    Example output (first 3 lines):
      Frame 1: <image>
      Frame 2: <image>
      ...
      Frame 16: <image>

      ROLE: You are a senior autonomous-vehicle safety engineer...
    """
    image_prefix = "\n".join(
        [f"Frame {i + 1}: <image>" for i in range(num_frames)]
    )
    return image_prefix + "\n\n" + PROMPT_G


# =============================================================================
# Response parsing
# =============================================================================

def parse_json_response(raw_text: str) -> dict:
    """
    Extract the JSON block from the model's raw response.

    Handles:
      - Clean JSON output
      - JSON wrapped in ```json ... ``` markdown blocks
      - Partial/malformed JSON (returns empty dict with error key)
    """
    # Try to extract JSON from markdown code block
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if md_match:
        candidate = md_match.group(1)
    else:
        # Try to find the first { ... } block
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else None

    if candidate is None:
        return {"parse_error": "no JSON block found", "raw": raw_text[:200]}

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Last resort: try to fix common issues (trailing commas, single quotes)
        try:
            # Remove trailing commas before } or ]
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            return {"parse_error": str(e), "raw": candidate[:300]}


def verdict_to_score(verdict: str, confidence: str, cfg: dict) -> float:
    """
    Convert PROMPT_G output (verdict + confidence) to a float collision score.

    Used ONLY in zero-shot mode (no ScoreHead available).
    After fine-tuning, the ScoreHead provides a direct float.

    cfg['confidence_to_score'] example:
      YES: {HIGH: 0.90, MEDIUM: 0.65, LOW: 0.40}
      NO:  {HIGH: 0.10, MEDIUM: 0.35, LOW: 0.60}
    """
    verdict    = str(verdict).strip().upper()
    confidence = str(confidence).strip().upper()
    mapping    = cfg.get("confidence_to_score", {})

    if verdict not in mapping:
        # Unknown verdict → neutral score
        return 0.5

    conf_map = mapping[verdict]
    if confidence not in conf_map:
        # Unknown confidence → use MEDIUM score
        confidence = "MEDIUM"

    return float(conf_map.get(confidence, 0.5))


# =============================================================================
# Load already-processed IDs (resume support)
# =============================================================================

def load_processed_ids(output_path: str) -> set:
    """Return set of video_ids already saved in output JSONL (for --resume)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec.get("video_id", ""))
                except json.JSONDecodeError:
                    pass
    return done


# =============================================================================
# Main evaluation loop
# =============================================================================

def evaluate(args, cfg: dict):
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if not torch.cuda.is_available() and device != "cpu":
        print("WARNING: CUDA not available, falling back to CPU (inference will be slow)")
        device = "cpu"

    # ── Load model ────────────────────────────────────────────────────────────
    load_in_4bit = args.load_in_4bit or cfg.get("load_in_4bit", False)
    model, tokenizer = load_model(
        model_id       = cfg["model_id"],
        load_in_4bit   = load_in_4bit,
        torch_dtype_str= cfg.get("torch_dtype", "bfloat16"),
        device         = device,
    )

    # ── Load manifest ─────────────────────────────────────────────────────────
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Clips to evaluate: {len(records)}")

    # ── Resume support ────────────────────────────────────────────────────────
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        processed_ids = load_processed_ids(args.output)
        print(f"  Resuming — already processed: {len(processed_ids)} clips")
    remaining = [r for r in records if r["video_id"] not in processed_ids]
    print(f"  Clips to run now : {len(remaining)}")

    # ── Prepare output file ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_mode = "a" if args.resume else "w"

    # ── Build prompt (constant across all clips) ──────────────────────────────
    window_size = cfg.get("window_size", 16)
    prompt_text = build_prompt_with_images(window_size)

    # ── Generation config ─────────────────────────────────────────────────────
    gen_config = {
        "max_new_tokens": cfg.get("max_new_tokens", 600),
        "do_sample":      cfg.get("do_sample", False),
    }
    if cfg.get("temperature", 0.0) > 0.0:
        gen_config["temperature"] = cfg["temperature"]
        gen_config["do_sample"]   = True

    frame_size    = cfg.get("frame_size", 448)
    frame_pattern = cfg.get("frame_filename_pattern", "frame_{:05d}.jpg")
    save_raw      = cfg.get("save_raw_response", True)

    # ── Stats tracking ────────────────────────────────────────────────────────
    n_yes = n_no = n_parse_error = 0
    t_start = time.time()

    with open(args.output, out_mode) as out_f:
        for rec in tqdm(remaining, desc="Zero-shot eval", unit="clip"):
            vid           = rec["video_id"]
            frame_indices = rec["frame_indices"]
            # Ground truth: teacher clips use "target", test clips use "event_occurs"
            ground_truth  = rec.get("target", rec.get("event_occurs", -1))

            result = {
                "video_id":       vid,
                "ground_truth":   ground_truth,
                "group":          rec.get("group", None),
                "time_before_s":  rec.get("time_before_event_s", rec.get("requested_time_to_event", None)),
                # filled below:
                "collision_verdict": None,
                "confidence":        None,
                "score":             None,
                "scene_context":     None,
                "temporal_analysis": None,
                "verdict_reasoning": None,
                "parse_error":       None,
                "latency_s":         None,
            }
            if save_raw:
                result["raw_response"] = None

            try:
                # ── Load frames ───────────────────────────────────────────────
                pixel_values = load_frames(
                    args.frames_root, vid, frame_indices, frame_size, frame_pattern
                )
                # InternVL expects (N, 3, H, W) on the model's device
                if device != "cpu":
                    pixel_values = pixel_values.to(device=next(model.parameters()).device,
                                                   dtype=next(model.parameters()).dtype)

                # ── Run inference ─────────────────────────────────────────────
                t0 = time.time()
                # num_patches_list tells InternVL how many tiles per image
                # Flash uses ViR compression: 1 patch per image after compression
                num_patches_list = [1] * window_size

                raw_response = model.chat(
                    tokenizer        = tokenizer,
                    pixel_values     = pixel_values,
                    question         = prompt_text,
                    generation_config= gen_config,
                    num_patches_list = num_patches_list,
                    history          = None,
                    return_history   = False,
                )
                latency = time.time() - t0

                # ── Parse JSON ────────────────────────────────────────────────
                parsed = parse_json_response(raw_response)

                if "parse_error" in parsed:
                    n_parse_error += 1
                    result["parse_error"] = parsed["parse_error"]
                    result["score"]       = 0.5   # neutral fallback
                    tqdm.write(f"  PARSE ERROR [{vid}]: {parsed['parse_error'][:80]}")
                else:
                    verdict    = parsed.get("collision_verdict", "").strip().upper()
                    confidence = parsed.get("confidence", "MEDIUM").strip().upper()
                    score      = verdict_to_score(verdict, confidence, cfg)

                    result["collision_verdict"] = verdict
                    result["confidence"]        = confidence
                    result["score"]             = score
                    result["scene_context"]     = parsed.get("scene_context")
                    result["temporal_analysis"] = parsed.get("temporal_analysis")
                    result["verdict_reasoning"] = parsed.get("verdict_reasoning")
                    result["latency_s"]         = round(latency, 2)

                    if verdict == "YES":
                        n_yes += 1
                    else:
                        n_no += 1

                if save_raw:
                    result["raw_response"] = raw_response

            except FileNotFoundError as e:
                result["parse_error"] = f"FRAME_NOT_FOUND: {e}"
                result["score"]       = 0.5
                tqdm.write(f"  FILE ERROR [{vid}]: {e}")

            except Exception as e:
                result["parse_error"] = f"RUNTIME_ERROR: {type(e).__name__}: {e}"
                result["score"]       = 0.5
                tqdm.write(f"  RUNTIME ERROR [{vid}]: {e}")

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    total_run = len(remaining)
    print(f"\n{'='*55}")
    print(f"Zero-shot evaluation complete")
    print(f"  Output file    : {args.output}")
    print(f"  Clips run      : {total_run}")
    print(f"  YES verdicts   : {n_yes}  ({100*n_yes/max(total_run,1):.1f}%)")
    print(f"  NO  verdicts   : {n_no}  ({100*n_no/max(total_run,1):.1f}%)")
    print(f"  Parse errors   : {n_parse_error}  ({100*n_parse_error/max(total_run,1):.1f}%)")
    print(f"  Total time     : {elapsed/60:.1f} min  ({elapsed/max(total_run,1):.1f}s/clip)")
    print(f"\nNext step: run evaluate_metrics.py on this output file")


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot evaluation of InternVL3.5-4B-Flash on collision clips"
    )
    parser.add_argument("--manifest",    required=True,
                        help="JSONL manifest file (from build_test_manifest.py or teacher v11)")
    parser.add_argument("--frames_root", required=True,
                        help="Root directory with per-video frame folders")
    parser.add_argument("--output",      required=True,
                        help="Output JSONL path for results")
    parser.add_argument("--config",      default="student_training/configs/zero_shot.yaml",
                        help="Path to zero_shot.yaml config")
    parser.add_argument("--device",      default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run inference on")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Force 4-bit quantization (for GPUs with < 10GB VRAM)")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip clips already saved in output file")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        # Try relative to project root
        cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print(f"Config loaded: {cfg_path}")
    print(f"  Model   : {cfg['model_id']}")
    print(f"  Frames  : {cfg.get('window_size', 16)} × stride={cfg.get('stride', 4)}")

    evaluate(args, cfg)


if __name__ == "__main__":
    main()
