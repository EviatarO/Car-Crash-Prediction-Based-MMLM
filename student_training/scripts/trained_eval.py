"""
trained_eval.py
===============
Inference script for a fine-tuned InternVLCollisionModel (LoRA + ScoreHead).

For each clip:
  1. Load 16 frames → pixel_values
  2. Forward the user prompt through the model (no generation yet)
     → ScoreHead outputs P(collision) ∈ [0, 1]  (used for AP/AUC ranking)
  3. Call model.generate() with PROMPT_G to produce the full JSON reasoning
     (used for interpretability and reasoning supervision metrics)

Output JSONL format matches zero_shot_eval.py exactly, so the same
evaluate_metrics.py and zero_shot_to_xlsx.py scripts work on the output.

Usage (on RunPod after training):
  python student_training/scripts/trained_eval.py \\
    --checkpoint  outputs/checkpoints/e2_lora_100clips/step_000300 \\
    --manifest    outputs/test_manifest_private.jsonl \\
    --frames_root /data/test_frames256 \\
    --output      outputs/trained/e2_lora_100clips_test.jsonl \\
    --config      student_training/configs/train_lora.yaml \\
    [--resume]    # skip already-processed clips in output file

  python student_training/scripts/evaluate_metrics.py \\
    --results outputs/trained/e2_lora_100clips_test.jsonl \\
    --out_dir outputs/metrics/e2_lora_100clips_test \\
    --tag     "E2 LoRA 100 clips"
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.templates import PROMPT_G  # noqa: E402
from student_training.models.internvl_lora import load_from_checkpoint  # noqa: E402
from student_training.data.collision_dataset import (  # noqa: E402
    get_image_token_str,
    expand_image_placeholders,
)

# ── Image preprocessing (mirrors zero_shot_eval.py) ──────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_frames(frames_root: str, video_id: str, frame_indices: list,
                frame_size: int, pattern: str = "frame_{:05d}.jpg") -> torch.Tensor:
    transform = build_transform(frame_size)
    tensors = []
    for idx in frame_indices:
        path = os.path.join(frames_root, video_id, pattern.format(idx))
        if not os.path.exists(path):
            raise FileNotFoundError(f"Frame not found: {path}")
        img = Image.open(path).convert("RGB")
        tensors.append(transform(img))
    return torch.stack(tensors, dim=0)   # (N, 3, H, W)


# ── Response parsing (mirrors zero_shot_eval.py) ──────────────────────────────

def parse_json_response(raw_text: str) -> dict:
    md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    if md_match:
        candidate = md_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        candidate = brace_match.group(0) if brace_match else None

    if candidate is None:
        return {"parse_error": "no JSON block found", "raw": raw_text[:200]}

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            return {"parse_error": str(e), "raw": candidate[:300]}


def build_prompt_with_images(num_frames: int) -> str:
    image_prefix = "\n".join(f"Frame {i + 1}: <image>" for i in range(num_frames))
    return image_prefix + "\n\n" + PROMPT_G


# ── Resume support ────────────────────────────────────────────────────────────

def load_processed_ids(output_path: str) -> set:
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            for line in f:
                try:
                    done.add(json.loads(line).get("video_id", ""))
                except json.JSONDecodeError:
                    pass
    return done


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(args, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and device != "cpu":
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # ── Load model from checkpoint ────────────────────────────────────────
    model, tokenizer = load_from_checkpoint(
        checkpoint_dir = args.checkpoint,
        model_id       = cfg["model_id"],
        cfg            = cfg,
        device_map     = "auto" if device == "cuda" else "cpu",
    )
    model.eval()

    # ── Load manifest ─────────────────────────────────────────────────────
    print(f"\nLoading manifest: {args.manifest}")
    with open(args.manifest, "r") as f:
        records = [json.loads(line) for line in f if line.strip()]
    print(f"  Clips to evaluate: {len(records)}")

    # ── Resume support ────────────────────────────────────────────────────
    processed_ids = set()
    if args.resume and os.path.exists(args.output):
        processed_ids = load_processed_ids(args.output)
        print(f"  Resuming — already processed: {len(processed_ids)} clips")
    remaining = [r for r in records if r["video_id"] not in processed_ids]
    print(f"  Clips to run now : {len(remaining)}")

    # ── Prepare output ────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_mode = "a" if args.resume else "w"

    # ── Config values ─────────────────────────────────────────────────────
    window_size   = cfg.get("window_size", 16)
    frame_size    = cfg.get("frame_size", 448)
    frame_pattern = cfg.get("frame_filename_pattern", "frame_{:05d}.jpg")
    max_new_tokens = cfg.get("max_new_tokens", 600)   # from zero_shot.yaml if present
    prompt_text   = build_prompt_with_images(window_size)
    num_patches_list = [1] * window_size

    gen_config = {
        "max_new_tokens": max_new_tokens,
        "do_sample":      False,
    }

    # ── Stats ─────────────────────────────────────────────────────────────
    n_yes = n_no = n_parse_error = 0
    t_start = time.time()

    with open(args.output, out_mode) as out_f:
        for rec in tqdm(remaining, desc="Trained eval", unit="clip"):
            vid          = rec["video_id"]
            frame_indices = rec["frame_indices"]
            ground_truth  = rec.get("target", rec.get("event_occurs", -1))

            result = {
                "video_id":          vid,
                "ground_truth":      ground_truth,
                "group":             rec.get("group", None),
                "time_before_s":     rec.get("time_before_event_s", rec.get("requested_time_to_event", None)),
                "collision_verdict": None,
                "confidence":        None,
                "score":             None,            # from ScoreHead (continuous)
                "scene_context":     None,
                "temporal_analysis": None,
                "verdict_reasoning": None,
                "parse_error":       None,
                "latency_s":         None,
            }
            if cfg.get("save_raw_response", True):
                result["raw_response"] = None

            try:
                # ── Load frames ───────────────────────────────────────────
                pixel_values = load_frames(
                    args.frames_root, vid, frame_indices, frame_size, frame_pattern
                )
                model_device = next(model.model.parameters()).device
                model_dtype  = next(model.model.parameters()).dtype
                pixel_values = pixel_values.to(device=model_device, dtype=model_dtype)

                t0 = time.time()

                # ── Step 1: Get collision score from ScoreHead ────────────
                # The score-head pass requires `input_ids` whose <IMG_CONTEXT>
                # token slots match the vision encoder's output exactly:
                #   total <IMG_CONTEXT> count = num_frames * num_image_token
                # If we tokenize a string with a literal "<image>" placeholder,
                # the model finds 0 slots for 4096 vit_embeds and crashes:
                #   "input_embeds[selected].shape=[0, 2560], vit_embeds.shape=[4096, 2560]"
                #
                # We use the same expansion the training dataset uses
                # (collision_dataset.expand_image_placeholders) so the score-head
                # path sees the identical token layout as during training.
                image_token_str = get_image_token_str(model.model)
                expanded_prompt = expand_image_placeholders(prompt_text, image_token_str)
                prefix_text = (
                    f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{expanded_prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                enc = tokenizer(
                    prefix_text, add_special_tokens=False, return_tensors="pt"
                )
                user_input_ids = enc.input_ids.to(device=model_device)
                user_attn_mask = enc.attention_mask.to(device=model_device)


                with torch.no_grad():
                    score_pred = model.get_score(
                        pixel_values     = pixel_values,
                        input_ids        = user_input_ids,
                        attention_mask   = user_attn_mask,
                        num_patches_list = num_patches_list,
                    )
                score = float(score_pred[0].item())

                # ── Step 2: Generate reasoning text ───────────────────────
                # Derive verdict from score (threshold 0.5) for XLSX review.
                # The score itself is the primary ranking metric for AP/AUC.
                raw_response = model.generate_reasoning(
                    tokenizer        = tokenizer,
                    pixel_values     = pixel_values,
                    prompt_text      = prompt_text,
                    num_patches_list = num_patches_list,
                    generation_config= gen_config,
                )

                latency = time.time() - t0

                # ── Parse JSON reasoning ───────────────────────────────────
                parsed = parse_json_response(raw_response)

                if "parse_error" in parsed:
                    n_parse_error += 1
                    result["parse_error"] = parsed["parse_error"]
                    # Use ScoreHead score even when reasoning parse fails
                    result["score"]             = round(score, 4)
                    result["collision_verdict"] = "YES" if score >= 0.5 else "NO"
                    tqdm.write(f"  PARSE ERROR [{vid}]: {parsed['parse_error'][:80]}")
                else:
                    verdict    = parsed.get("collision_verdict", "").strip().upper()
                    confidence = parsed.get("confidence", "MEDIUM").strip().upper()

                    result["collision_verdict"] = verdict
                    result["confidence"]        = confidence
                    result["score"]             = round(score, 4)
                    result["scene_context"]     = parsed.get("scene_context")
                    result["temporal_analysis"] = parsed.get("temporal_analysis")
                    result["verdict_reasoning"] = parsed.get("verdict_reasoning")
                    result["latency_s"]         = round(latency, 2)

                    if verdict == "YES":
                        n_yes += 1
                    else:
                        n_no += 1

                if cfg.get("save_raw_response", True):
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

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed   = time.time() - t_start
    total_run = len(remaining)
    print(f"\n{'='*55}")
    print(f"Trained model evaluation complete")
    print(f"  Checkpoint     : {args.checkpoint}")
    print(f"  Output file    : {args.output}")
    print(f"  Clips run      : {total_run}")
    print(f"  YES verdicts   : {n_yes}  ({100*n_yes/max(total_run,1):.1f}%)")
    print(f"  NO  verdicts   : {n_no}  ({100*n_no/max(total_run,1):.1f}%)")
    print(f"  Parse errors   : {n_parse_error}  ({100*n_parse_error/max(total_run,1):.1f}%)")
    print(f"  Total time     : {elapsed/60:.1f} min  ({elapsed/max(total_run,1):.1f}s/clip)")
    print(f"\nNext step:")
    print(f"  python student_training/scripts/evaluate_metrics.py \\")
    print(f"    --results {args.output} \\")
    print(f"    --out_dir outputs/metrics/{Path(args.output).stem} \\")
    print(f"    --tag     \"{Path(args.checkpoint).parent.name}\"")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned InternVLCollisionModel on a clip manifest"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to checkpoint dir (contains adapter_config.json + score_head.pt)"
    )
    parser.add_argument(
        "--manifest", required=True,
        help="JSONL manifest (from build_test_manifest.py or manifest_v11_100clips.jsonl)"
    )
    parser.add_argument(
        "--frames_root", required=True,
        help="Root directory with per-video frame folders"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output JSONL path for results"
    )
    parser.add_argument(
        "--config", default="student_training/configs/train_lora.yaml",
        help="Path to train_lora.yaml (provides model_id, dtype, etc.)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip clips already saved in output file"
    )
    args = parser.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Inherit zero_shot.yaml defaults if present (for max_new_tokens etc.)
    zero_shot_cfg_path = PROJECT_ROOT / "student_training/configs/zero_shot.yaml"
    if zero_shot_cfg_path.exists():
        with open(zero_shot_cfg_path) as f:
            zs_cfg = yaml.safe_load(f)
        # Only fill in keys not already in train_lora.yaml
        for k, v in zs_cfg.items():
            cfg.setdefault(k, v)

    print(f"Config loaded: {cfg_path}")
    print(f"  Model        : {cfg['model_id']}")
    print(f"  Checkpoint   : {args.checkpoint}")

    evaluate(args, cfg)


if __name__ == "__main__":
    main()
