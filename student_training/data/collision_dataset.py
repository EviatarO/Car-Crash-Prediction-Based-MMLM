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
        self.num_image_token = getattr(model, "num_image_token", 256)

        # ── Image special token IDs ───────────────────────────────────────
        # img_context_token_id MUST be set on the model before building the
        # dataset (call _probe_and_set_img_context_token_id(model, tokenizer)
        # in load_for_training() first).  We insert these IDs directly into
        # input_ids so InternVL.forward() knows where to inject visual embeds.
        unk_id = getattr(tokenizer, "unk_token_id", None)

        self.img_ctx_id = getattr(model, "img_context_token_id", None)
        if self.img_ctx_id is None:
            # Last-chance fallback: query tokenizer directly
            self.img_ctx_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if self.img_ctx_id == unk_id:
                self.img_ctx_id = None
        if self.img_ctx_id is None:
            raise ValueError(
                "model.img_context_token_id is None and <IMG_CONTEXT> is not in "
                "the tokenizer vocabulary.  Call "
                "_probe_and_set_img_context_token_id(model, tokenizer) before "
                "constructing CollisionDataset (load_for_training does this)."
            )

        self.img_start_id = tokenizer.convert_tokens_to_ids("<img>")
        self.img_end_id   = tokenizer.convert_tokens_to_ids("</img>")

        # Validate image boundary tokens
        for name, tid in [("img_start (<img>)", self.img_start_id),
                           ("img_end (</img>)",  self.img_end_id)]:
            if tid is None or tid == unk_id:
                raise ValueError(
                    f"Token {name} not found in tokenizer vocabulary "
                    f"(got {tid!r}). Check model_id and tokenizer."
                )

        # per-image token ID list: [<img>, <IMG_CONTEXT>×N, </img>]
        self._per_img_ids: List[int] = (
            [self.img_start_id]
            + [self.img_ctx_id] * self.num_image_token
            + [self.img_end_id]
        )

        # Load records
        raw_records = self._load_jsonl(jsonl_path)
        if skip_errors:
            raw_records = [r for r in raw_records if not r.get("error")]
        self.records: List[dict] = raw_records

        tokens_per_clip = self.num_image_token * self.window_size
        print(
            f"CollisionDataset: {len(self.records)} records from {jsonl_path}\n"
            f"  img_context_token_id : {self.img_ctx_id}\n"
            f"  img_start / img_end  : {self.img_start_id} / {self.img_end_id}\n"
            f"  num_image_token      : {self.num_image_token} per image "
            f"→ {tokens_per_clip} image tokens per clip\n"
            f"  max_seq_len          : {self.max_seq_len}"
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

    def _build_ids_and_labels(self, assistant_text: str) -> Dict[str, torch.Tensor]:
        """
        Build input_ids, labels, and attention_mask by inserting image token IDs
        DIRECTLY rather than tokenizing a string that contains '<IMG_CONTEXT>'.

        This is the correct approach because InternVL's forward() looks for
        model.img_context_token_id in input_ids to know where to inject visual
        embeddings — if we let the BPE tokenizer handle '<IMG_CONTEXT>' strings,
        it may produce wrong IDs or split them incorrectly.

        Conversation structure (Qwen3 / InternVL3.5 chat format):
          <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n
          <|im_start|>user\\n
          Frame 1: <img><IMG_CONTEXT>×N</img>\\n
          ...
          Frame 16: <img><IMG_CONTEXT>×N</img>\\n\\n
          {PROMPT_G}<|im_end|>\\n
          <|im_start|>assistant\\n
          {JSON reasoning}<|im_end|>\\n
        """
        tok = self.tokenizer

        def enc(text: str) -> List[int]:
            """Encode a plain text string to token IDs (no special tokens added)."""
            return tok.encode(text, add_special_tokens=False)

        # ── System + user header ──────────────────────────────────────────
        header_ids = enc(
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n"
        )

        # ── 16 frames: "Frame N: " + [img token IDs] + "\n" ──────────────
        frame_ids: List[int] = []
        for i in range(self.window_size):
            frame_ids += enc(f"Frame {i + 1}: ")
            frame_ids += self._per_img_ids          # [<img>, ctx×N, </img>]
            frame_ids += enc("\n")

        # ── PROMPT_G + end of user turn + assistant header ─────────────────
        prompt_ids = enc(f"\n{PROMPT_G}<|im_end|>\n<|im_start|>assistant\n")

        # ── Assistant response ─────────────────────────────────────────────
        asst_ids = enc(assistant_text + "<|im_end|>\n")

        # ── Assemble ───────────────────────────────────────────────────────
        user_ids  = header_ids + frame_ids + prompt_ids
        all_ids   = user_ids + asst_ids
        user_len  = len(user_ids)

        # Truncate to max_seq_len from the END (preserve assistant response)
        if len(all_ids) > self.max_seq_len:
            all_ids  = all_ids[-self.max_seq_len:]
            user_len = max(0, self.max_seq_len - len(asst_ids))

        # ── Labels: -100 for user/system, real IDs for assistant ──────────
        labels = [-100] * user_len + all_ids[user_len:]

        input_ids      = torch.tensor(all_ids, dtype=torch.long)
        labels_tensor  = torch.tensor(labels,  dtype=torch.long)
        attention_mask = torch.ones(len(all_ids), dtype=torch.long)
        asst_start_pos = torch.tensor(min(user_len, len(all_ids) - 1), dtype=torch.long)

        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "labels":          labels_tensor,
            "asst_start_pos":  asst_start_pos,
        }

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        video_id      = record["video_id"]
        frame_indices = record["frame_indices"][:self.window_size]   # truncate to window_size
        score_target  = float(record.get("target", 0))   # 0 or 1 (int in JSONL)

        # ── Load pixel values ─────────────────────────────────────────────
        pixel_values = self._load_frames(video_id, frame_indices)
        # pixel_values: (16, 3, 448, 448)

        # ── Build assistant text + tokenise ──────────────────────────────
        reasoning_json = build_full_reasoning_json(record)
        assistant_text = json.dumps(reasoning_json, indent=2, ensure_ascii=False)
        tok_out        = self._build_ids_and_labels(assistant_text)

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
        pad_id  = 0
        max_len = max(s["input_ids"].shape[0] for s in batch)

        padded_ids, padded_masks, padded_labels = [], [], []
        all_pixel_values, score_targets, asst_starts, video_ids = [], [], [], []

        for s in batch:
            pad_len = max_len - s["input_ids"].shape[0]
            padded_ids.append(   torch.nn.functional.pad(s["input_ids"],      (0, pad_len), value=pad_id))
            padded_masks.append( torch.nn.functional.pad(s["attention_mask"], (0, pad_len), value=0))
            padded_labels.append(torch.nn.functional.pad(s["labels"],         (0, pad_len), value=-100))
            all_pixel_values.append(s["pixel_values"])
            score_targets.append(s["score_target"])
            asst_starts.append(s["asst_start_pos"])
            video_ids.append(s["video_id"])

        return {
            "pixel_values":   torch.cat(all_pixel_values, dim=0),    # (B*16, 3, H, W)
            "input_ids":      torch.stack(padded_ids,     dim=0),     # (B, seq_len)
            "attention_mask": torch.stack(padded_masks,   dim=0),     # (B, seq_len)
            "labels":         torch.stack(padded_labels,  dim=0),     # (B, seq_len)
            "score_target":   torch.stack(score_targets,  dim=0),     # (B,)
            "asst_start_pos": torch.stack(asst_starts,    dim=0),     # (B,)
            "video_ids":      video_ids,
        }
