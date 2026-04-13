"""
collision_dataset.py
====================
PyTorch Dataset for teacher-supervised LoRA fine-tuning of InternVL3.5-4B-Flash.

Source: teacher_dataset_v11.jsonl produced by Teacher_dataset_distill_v11.py.

Each item returns:
  pixel_values    (16, 3, 448, 448)  — preprocessed frames on CPU
  input_ids       (seq_len,)         — tokenized [user prompt + assistant response]
  attention_mask  (seq_len,)         — 1 for real tokens, 0 for padding
  labels          (seq_len,)         — target tokens; -100 for user/pad positions
  score_target    scalar float        — 1.0 (collision) / 0.0 (no collision)
  asst_start_pos  scalar long         — index of first assistant-response token in input_ids
  video_id        str

Conversation format (InternVL3.5 / Qwen3-based chat template):
  <|im_start|>system
  You are a helpful assistant.<|im_end|>
  <|im_start|>user
  Frame 1: <img><IMG_CONTEXT>×N...</img>
  ...
  Frame 16: <img><IMG_CONTEXT>×N...</img>

  {PROMPT_G}<|im_end|>
  <|im_start|>assistant
  {JSON reasoning}<|im_end|>

The <image> placeholder is expanded to the model's internal image token sequence
using the model's own num_image_token property and img_context_token_id.

Usage:
  ds = CollisionDataset(
      jsonl_path   = "outputs/teacher_dataset_v11.jsonl",
      frames_root  = "/data/train_frames256",
      model        = intern_model,          # loaded InternVL model
      tokenizer    = tokenizer,
      cfg          = yaml_cfg_dict,
  )
  sample = ds[0]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

# ── Project root → prompts ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from prompts.templates import PROMPT_G  # noqa: E402

# ── Image preprocessing (identical to zero_shot_eval.py) ─────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size: int) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Image-token expansion ─────────────────────────────────────────────────────

def get_image_token_str(model) -> str:
    """
    Build the token string that replaces each <image> placeholder in the text.

    InternVL3.5 uses:   <img><IMG_CONTEXT><IMG_CONTEXT>...<IMG_CONTEXT></img>
    where the number of <IMG_CONTEXT> tokens = model.num_image_token (typically 64).

    Queries the model's own config attributes; falls back to common defaults.
    """
    num_img_tokens = getattr(model, "num_image_token", 64)

    # Try to read special-token strings from model config
    cfg = getattr(model, "config", None)

    img_start = (
        getattr(cfg, "img_start_token", None)
        or getattr(model, "IMG_START_TOKEN", None)
        or "<img>"
    )
    img_end = (
        getattr(cfg, "img_end_token", None)
        or getattr(model, "IMG_END_TOKEN", None)
        or "</img>"
    )
    img_ctx = (
        getattr(cfg, "img_context_token", None)
        or getattr(model, "IMG_CONTEXT_TOKEN", None)
        or "<IMG_CONTEXT>"
    )

    return img_start + img_ctx * num_img_tokens + img_end


def expand_image_placeholders(text: str, image_token_str: str) -> str:
    """Replace every <image> in text with the full image token sequence."""
    return text.replace("<image>", image_token_str)


# ── Reasoning JSON reconstruction ────────────────────────────────────────────

def build_full_reasoning_json(record: dict) -> dict:
    """
    Reconstruct the full PROMPT_G JSON CoT from the fields stored in the teacher JSONL.

    If a debate (Pass-2) was used and succeeded, use Pass-2 fields (p2_*).
    Otherwise use Pass-1 fields.

    The resulting dict matches the 8-key PROMPT_G output schema exactly,
    making it suitable as an LM supervision target.
    """
    mismatch    = record.get("mismatch", False)
    p2_verdict  = record.get("p2_collision_verdict")

    use_p2 = mismatch and p2_verdict is not None

    if use_p2:
        return {
            "scene_context":     record.get("p2_scene_context"),
            "dynamic_objects":   record.get("p2_dynamic_objects"),
            "temporal_analysis": record.get("p2_temporal_analysis"),
            "occlusion_check":   record.get("p2_occlusion_check"),
            "time_to_contact":   record.get("p2_time_to_contact"),
            "collision_verdict": record.get("p2_collision_verdict"),
            "confidence":        record.get("p2_confidence"),
            "verdict_reasoning": record.get("p2_verdict_reasoning"),
        }
    else:
        return {
            "scene_context":     record.get("scene_context"),
            "dynamic_objects":   record.get("dynamic_objects"),
            "temporal_analysis": record.get("temporal_analysis"),
            "occlusion_check":   record.get("occlusion_check"),
            "time_to_contact":   record.get("time_to_contact"),
            "collision_verdict": record.get("collision_verdict"),
            "confidence":        record.get("confidence"),
            "verdict_reasoning": record.get("verdict_reasoning"),
        }


# ── Dataset ───────────────────────────────────────────────────────────────────

class CollisionDataset(Dataset):
    """
    Teacher-distilled dataset for InternVL3.5-4B-Flash LoRA fine-tuning.

    Args:
        jsonl_path:   Path to teacher_dataset_v11.jsonl (or any compatible JSONL).
        frames_root:  Root directory containing per-video frame folders.
                      Paths are reconstructed as:
                        frames_root / video_id / frame_pattern.format(frame_idx)
        model:        Loaded InternVLChatModel (used to query image token properties
                      and to call tokenizer.apply_chat_template via its config).
        tokenizer:    HuggingFace tokenizer for the model.
        cfg:          Dict from train_lora.yaml.
        skip_errors:  If True, silently drop JSONL records that have an "error" field.
    """

    # Qwen2/InternVL system prompt (same as model's default)
    SYSTEM_PROMPT = "You are a helpful assistant."

    def __init__(
        self,
        jsonl_path:  str,
        frames_root: str,
        model,
        tokenizer,
        cfg:         dict,
        skip_errors: bool = True,
    ):
        self.frames_root    = frames_root
        self.tokenizer      = tokenizer
        self.cfg            = cfg
        self.window_size    = cfg.get("window_size", 16)
        self.frame_size     = cfg.get("frame_size", 448)
        self.frame_pattern  = cfg.get("frame_filename_pattern", "frame_{:05d}.jpg")
        self.max_seq_len    = cfg.get("max_seq_len", 2048)
        self.transform      = build_transform(self.frame_size)

        # Query model for image token properties
        self.image_token_str = get_image_token_str(model)
        self.num_image_token = getattr(model, "num_image_token", 64)

        # Load records
        raw_records = self._load_jsonl(jsonl_path)
        if skip_errors:
            raw_records = [r for r in raw_records if not r.get("error")]
        self.records: List[dict] = raw_records

        print(
            f"CollisionDataset: {len(self.records)} records from {jsonl_path}\n"
            f"  image_token_str length : {len(self.image_token_str)} chars "
            f"({self.num_image_token} IMG_CONTEXT tokens/image)\n"
            f"  max_seq_len            : {self.max_seq_len}"
        )

    # ── I/O helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: str) -> List[dict]:
        records = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    def _load_frames(self, video_id: str, frame_indices: List[int]) -> torch.Tensor:
        """
        Load frame images and return a stacked (N, 3, H, W) tensor.

        Frame path pattern:  frames_root / video_id / frame_00042.jpg
        """
        tensors = []
        for idx in frame_indices:
            path = os.path.join(
                self.frames_root, video_id, self.frame_pattern.format(idx)
            )
            if not os.path.exists(path):
                raise FileNotFoundError(f"Frame not found: {path}")
            img = Image.open(path).convert("RGB")
            tensors.append(self.transform(img))
        return torch.stack(tensors, dim=0)  # (N, 3, H, W)

    # ── Tokenisation ─────────────────────────────────────────────────────────

    def _build_user_text(self) -> str:
        """
        Build the user-turn text with <image> placeholders already expanded.

        Format:
          Frame 1: <img><IMG_CONTEXT>×64</img>
          ...
          Frame 16: <img><IMG_CONTEXT>×64</img>

          {PROMPT_G}
        """
        image_prefix = "\n".join(
            f"Frame {i + 1}: {self.image_token_str}"
            for i in range(self.window_size)
        )
        return image_prefix + "\n\n" + PROMPT_G

    def _tokenize_conversation(
        self,
        user_text:      str,
        assistant_text: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize the full conversation and create labels that mask the user turn.

        Returns dict with keys:
          input_ids       (seq_len,)  long
          attention_mask  (seq_len,)  long
          labels          (seq_len,)  long  — user positions set to -100
          asst_start_pos  scalar long — index of first assistant-response token
        """
        tok = self.tokenizer

        # ── Build conversation using the model's chat template ─────────────
        # apply_chat_template with add_generation_prompt=True gives us the
        # tokenised user prompt + "<|im_start|>assistant\n" suffix.
        # This works for Qwen3 and InternVL3.5 tokenisers (same <|im_start|> format).
        messages_user_only = [
            {"role": "system",    "content": self.SYSTEM_PROMPT},
            {"role": "user",      "content": user_text},
        ]

        # Tokenise the user-only prefix (without assistant response)
        try:
            user_prefix_ids = tok.apply_chat_template(
                messages_user_only,
                add_generation_prompt=True,   # appends <|im_start|>assistant\n
                tokenize=True,
                return_tensors="pt",
            ).squeeze(0)                       # (user_prefix_len,)
        except Exception:
            # Fallback: manual Qwen3 format (same <|im_start|> convention)
            prefix_text = (
                f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_text}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            user_prefix_ids = tok(
                prefix_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids.squeeze(0)

        # Tokenise the assistant response
        # Add <|im_end|> + newline as EOS for the assistant turn
        eos_token = getattr(tok, "eos_token", "<|im_end|>") or "<|im_end|>"
        asst_text = assistant_text + eos_token + "\n"
        asst_ids = tok(
            asst_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids.squeeze(0)  # (asst_len,)

        # ── Concatenate ───────────────────────────────────────────────────
        input_ids = torch.cat([user_prefix_ids, asst_ids], dim=0)

        # Truncate to max_seq_len (keep end of sequence — most important part)
        if input_ids.shape[0] > self.max_seq_len:
            input_ids = input_ids[-self.max_seq_len:]
            # Recalculate user_len after truncation — set to 0 if all user tokens removed
            user_len = max(0, user_prefix_ids.shape[0] - (input_ids.shape[0] - self.max_seq_len + user_prefix_ids.shape[0] + asst_ids.shape[0] - self.max_seq_len))
            user_len = min(user_prefix_ids.shape[0], self.max_seq_len - asst_ids.shape[0])
            user_len = max(0, user_len)
        else:
            user_len = user_prefix_ids.shape[0]

        # ── Labels: mask user positions with -100 ─────────────────────────
        labels = input_ids.clone()
        labels[:user_len] = -100   # do not compute loss on user/system tokens

        attention_mask = torch.ones_like(input_ids)

        # asst_start_pos: the index of the first assistant response token
        # = the first token AFTER the user prefix (0-indexed in the final input_ids)
        asst_start_pos = min(user_len, input_ids.shape[0] - 1)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "labels":          labels,
            "asst_start_pos":  torch.tensor(asst_start_pos, dtype=torch.long),
        }

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        video_id      = record["video_id"]
        frame_indices = record["frame_indices"]
        score_target  = float(record.get("target", 0))   # 0 or 1 (int in JSONL)

        # ── Load pixel values ─────────────────────────────────────────────
        pixel_values = self._load_frames(video_id, frame_indices)
        # pixel_values: (16, 3, 448, 448)

        # ── Build texts ───────────────────────────────────────────────────
        user_text       = self._build_user_text()
        reasoning_json  = build_full_reasoning_json(record)
        assistant_text  = json.dumps(reasoning_json, indent=2, ensure_ascii=False)

        # ── Tokenise ──────────────────────────────────────────────────────
        tok_out = self._tokenize_conversation(user_text, assistant_text)

        return {
            "pixel_values":   pixel_values,                          # (16, 3, H, W) cpu float
            "input_ids":      tok_out["input_ids"],                  # (seq_len,) long
            "attention_mask": tok_out["attention_mask"],             # (seq_len,) long
            "labels":         tok_out["labels"],                     # (seq_len,) long
            "score_target":   torch.tensor(score_target, dtype=torch.float32),
            "asst_start_pos": tok_out["asst_start_pos"],             # scalar long
            "video_id":       video_id,
        }

    # ── Collate helper ────────────────────────────────────────────────────────

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        """
        Collate a list of samples into a batch dict.

        Pads input_ids, attention_mask, and labels to the longest sequence
        in the batch.  pixel_values are stacked along dim 0 (all frames
        from all clips in order).

        num_patches_list tells InternVL how many visual tiles each image has.
        For ViR-compressed single-tile mode (our setting): always [1] per image.
        """
        pad_id = 0    # tokenizer pad_token_id; 0 is safe for most HF tokenisers

        # Find max sequence length in batch
        max_len = max(s["input_ids"].shape[0] for s in batch)

        padded_ids   = []
        padded_masks = []
        padded_labels = []
        all_pixel_values = []
        score_targets    = []
        asst_starts      = []
        video_ids        = []
        num_patches_list = []

        for s in batch:
            seq_len = s["input_ids"].shape[0]
            pad_len = max_len - seq_len

            # Pad on the RIGHT with pad_id / 0 / -100
            ids  = torch.nn.functional.pad(s["input_ids"],      (0, pad_len), value=pad_id)
            mask = torch.nn.functional.pad(s["attention_mask"], (0, pad_len), value=0)
            lbl  = torch.nn.functional.pad(s["labels"],         (0, pad_len), value=-100)

            padded_ids.append(ids)
            padded_masks.append(mask)
            padded_labels.append(lbl)
            all_pixel_values.append(s["pixel_values"])   # (16, 3, H, W)
            score_targets.append(s["score_target"])
            asst_starts.append(s["asst_start_pos"])
            video_ids.append(s["video_id"])
            # 1 tile per image, 16 images per clip → 16 ones per sample in batch
            num_patches_list.extend([1] * s["pixel_values"].shape[0])

        return {
            "pixel_values":    torch.cat(all_pixel_values, dim=0),   # (B*16, 3, H, W)
            "input_ids":       torch.stack(padded_ids,    dim=0),     # (B, seq_len)
            "attention_mask":  torch.stack(padded_masks,  dim=0),     # (B, seq_len)
            "labels":          torch.stack(padded_labels, dim=0),     # (B, seq_len)
            "score_target":    torch.stack(score_targets, dim=0),     # (B,)
            "asst_start_pos":  torch.stack(asst_starts,   dim=0),     # (B,)
            "video_ids":       video_ids,
            "num_patches_list": num_patches_list,                     # list[int], len = B*16
        }
