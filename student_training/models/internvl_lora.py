"""
internvl_lora.py
================
Wraps InternVL3.5-4B-Flash with:
  1. PEFT LoRA adapters on the LLM backbone (vision encoder stays frozen).
  2. A ScoreHead — nn.Linear(hidden_size → 1) + Sigmoid — that reads the
     LLM hidden state at the first assistant-response token position and
     outputs P(collision) ∈ [0, 1].

Architecture recap (from README):
  InternViT-300M (FROZEN) → ViR Compression (FROZEN) → Projector (TRAINED)
  → Qwen3-4B + LoRA (TRAINED) → ScoreHead (TRAINED) + LM head (LoRA)

Combined forward loss:
  L = loss_alpha * BCE(score_pred, target) + (1 - loss_alpha) * LM_CE_loss

Usage — loading for training:
  model, tokenizer = load_for_training(model_id, cfg)

Usage — loading for inference from saved checkpoint:
  model, tokenizer = load_from_checkpoint(checkpoint_dir, model_id, cfg)

Usage — inference-only forward (no labels needed):
  score, reasoning = model.generate_with_score(
      pixel_values, input_ids, attention_mask, num_patches_list, tokenizer, gen_cfg
  )
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, AutoTokenizer

# ── Monkey-patch for InternVL + newer transformers compatibility ──────────────
# Newer transformers (>=4.50) both reads AND writes self.all_tied_weights_keys
# inside from_pretrained() and post_init(). InternVLChatModel (loaded via
# trust_remote_code) doesn't define it. We add a descriptor with both getter
# and setter so post_init can assign the value and later code can read it.
import transformers.modeling_utils as _tmu
if not hasattr(_tmu.PreTrainedModel, 'all_tied_weights_keys'):
    _TIED_KEYS_STORE = '_all_tied_weights_keys_store'

    @property
    def _all_tied_getter(self):
        if hasattr(self, _TIED_KEYS_STORE):
            return getattr(self, _TIED_KEYS_STORE)
        return {k: k for k in getattr(self, '_tied_weights_keys', set())}

    @_all_tied_getter.setter
    def _all_tied_getter(self, value):
        object.__setattr__(self, _TIED_KEYS_STORE, value)

    _tmu.PreTrainedModel.all_tied_weights_keys = _all_tied_getter


# ── Image-context token discovery ─────────────────────────────────────────────

def _probe_and_set_img_context_token_id(model, tokenizer) -> int:
    """
    InternVLChatModel.img_context_token_id is normally set inside chat(), not
    during from_pretrained().  We must set it manually so that forward() knows
    where to inject visual embeddings into the token sequence.

    Strategy:
      1. If model.img_context_token_id is already a valid int → done.
      2. Try known token strings used by various InternVL versions.
      3. Scan tokenizer.added_tokens_encoder for any IMG_CONTEXT-like entry.
      4. If still not found, print all added tokens and raise a clear error.

    Returns the resolved token ID (int) and sets it on model.img_context_token_id.
    """
    current = getattr(model, "img_context_token_id", None)
    if current is not None and isinstance(current, int):
        return current

    unk_id = getattr(tokenizer, "unk_token_id", None)
    added  = getattr(tokenizer, "added_tokens_encoder", {})

    def _try(tok_str: str):
        """Return token ID if tok_str is in the vocabulary, else None."""
        if not tok_str:
            return None
        # Check added_tokens_encoder first (exact match, no BPE splitting)
        if tok_str in added:
            return added[tok_str]
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if tid is not None and tid != unk_id:
            return tid
        return None

    # ── Strategy 1: read the token string from model.config ──────────────
    # InternVLChatConfig always stores the exact token string used.
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ("img_context_token", "IMAGE_CONTEXT_TOKEN", "img_pad_token"):
            tok_str = getattr(cfg, attr, None)
            if tok_str and isinstance(tok_str, str):
                tid = _try(tok_str)
                if tid is not None:
                    model.img_context_token_id = tid
                    print(
                        f"[internvl_lora] img_context_token_id = {tid}"
                        f"  (from config.{attr}: {tok_str!r})"
                    )
                    return tid

    # ── Strategy 2: try known token string variants ───────────────────────
    candidates = [
        "<IMG_CONTEXT>",
        "<img_context>",
        "<image_patch>",
        "<vit_patch>",
        "<image>",
    ]
    for tok_str in candidates:
        tid = _try(tok_str)
        if tid is not None:
            model.img_context_token_id = tid
            print(f"[internvl_lora] img_context_token_id = {tid}  (token: {tok_str!r})")
            return tid

    # ── Strategy 3: scan added_tokens_encoder for any IMG/context token ───
    for tok_str, tid in sorted(added.items(), key=lambda x: x[1]):
        low = tok_str.lower()
        if "context" in low or ("img" in low and "start" not in low and "end" not in low):
            if tid != unk_id:
                model.img_context_token_id = tid
                print(
                    f"[internvl_lora] img_context_token_id = {tid}"
                    f"  (found via scan: {tok_str!r})"
                )
                return tid

    # ── Nothing found — print full diagnostics and raise ─────────────────
    print("\n[internvl_lora] ERROR: Cannot find <IMG_CONTEXT> token.")
    print(f"  model.config attrs: { {a: getattr(cfg, a, 'N/A') for a in dir(cfg) if 'img' in a.lower()} }")
    print(f"  tokenizer.unk_token_id = {unk_id}")
    print(f"  All {len(added)} added tokens:")
    for k, v in sorted(added.items(), key=lambda x: x[1]):
        print(f"    {v:6d}: {k!r}")
    raise ValueError(
        "Cannot resolve img_context_token_id. "
        "Check that you loaded the correct model_id and tokenizer. "
        "See printed token list above."
    )


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class CollisionModelOutput:
    loss:            Optional[torch.Tensor] = None
    lm_loss:         Optional[torch.Tensor] = None
    score_loss:      Optional[torch.Tensor] = None
    score_pred:      Optional[torch.Tensor] = None   # (B,) float in [0, 1]
    logits:          Optional[torch.Tensor] = None   # LM logits if needed


# ── ScoreHead ─────────────────────────────────────────────────────────────────

class ScoreHead(nn.Module):
    """
    Single linear layer that outputs a raw logit (no sigmoid).

    Input:  hidden state at the first assistant-token position
            shape: (B, hidden_size)
    Output: logit  shape: (B,)

    Sigmoid is applied OUTSIDE this module:
      - Training:  use F.binary_cross_entropy_with_logits (autocast-safe)
      - Inference:  apply torch.sigmoid() on the returned logit
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).squeeze(-1)   # (B,) raw logit


# ── Main model wrapper ────────────────────────────────────────────────────────

class InternVLCollisionModel(nn.Module):
    """
    InternVL3.5-4B-Flash + LoRA + ScoreHead.

    Only LoRA parameters and ScoreHead are trained; everything else is frozen.

    Args:
        base_model:   Loaded InternVLChatModel (from AutoModel.from_pretrained).
        lora_config:  PEFT LoraConfig to apply to the LLM backbone.
        loss_alpha:   Weight on score BCE loss (1 - loss_alpha on LM CE loss).
    """

    def __init__(
        self,
        base_model,
        lora_config: LoraConfig,
        loss_alpha:  float = 0.5,
    ):
        super().__init__()
        self.loss_alpha = loss_alpha

        # ── Freeze everything, then apply LoRA ────────────────────────────
        for param in base_model.parameters():
            param.requires_grad = False

        # Apply LoRA to the language model backbone
        # InternVL's LLM is at base_model.language_model
        lm = base_model.language_model
        lm_with_lora = get_peft_model(lm, lora_config)
        base_model.language_model = lm_with_lora

        self.model = base_model

        # ── ScoreHead ─────────────────────────────────────────────────────
        hidden_size = self._get_hidden_size()
        self.score_head = ScoreHead(hidden_size)
        # Move ScoreHead to the same device/dtype as the model to avoid
        # device-mismatch errors during forward().
        _model_device = next(base_model.parameters()).device
        _model_dtype  = next(base_model.parameters()).dtype
        self.score_head = self.score_head.to(device=_model_device, dtype=_model_dtype)

        # ── Unfreeze projector (vision→LLM alignment layer) ───────────────
        # Projector is a small MLP and benefits from fine-tuning even without LoRA.
        projector = getattr(base_model, "mlp1", None)   # common InternVL name
        if projector is None:
            projector = getattr(base_model, "vision_proj", None)
        if projector is not None:
            for param in projector.parameters():
                param.requires_grad = True

        self._print_trainable_params()

    def _get_hidden_size(self) -> int:
        """Extract LLM hidden size from model config."""
        lm = self.model.language_model
        cfg = getattr(lm, "config", None)
        if cfg is not None:
            for attr in ("hidden_size", "d_model", "n_embd"):
                val = getattr(cfg, attr, None)
                if val:
                    return int(val)
        # Last resort: infer from embed_tokens weight shape
        try:
            emb = lm.base_model.model.embed_tokens.weight
            return emb.shape[1]
        except Exception:
            return 2048   # safe fallback for 4B models

    def _print_trainable_params(self):
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"InternVLCollisionModel trainable params: "
            f"{trained/1e6:.1f}M / {total/1e6:.1f}M "
            f"({100 * trained / total:.2f}%)"
        )

    # ── Core forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values:    torch.Tensor,           # (B*16, 3, H, W)
        input_ids:       torch.Tensor,           # (B, seq_len)
        attention_mask:  torch.Tensor,           # (B, seq_len)
        labels:          Optional[torch.Tensor], # (B, seq_len)
        score_target:    Optional[torch.Tensor], # (B,)
        asst_start_pos:  Optional[torch.Tensor], # (B,)  index into seq_len
        num_patches_list: Optional[List[int]] = None,  # kept for API compat, unused
    ) -> CollisionModelOutput:
        """
        Single forward pass that computes both LM loss and ScoreHead loss.

        InternVLChatModel.forward() uses image_flags (not num_patches_list).
        image_flags is a (B*16,) tensor of ones — all frames are real images.
        """
        # image_flags: one flag per image in pixel_values; 1 = real image
        n_images    = pixel_values.shape[0]
        image_flags = torch.ones(n_images, dtype=torch.long, device=pixel_values.device)

        # ── LM forward ────────────────────────────────────────────────────
        lm_outputs = self.model(
            pixel_values     = pixel_values,
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            image_flags      = image_flags,
            labels           = labels,
            output_hidden_states = True,
        )

        lm_loss = lm_outputs.loss   # cross-entropy over assistant tokens

        # ── Extract hidden state for ScoreHead ────────────────────────────
        # hidden_states: tuple of (B, seq_len, hidden_size) per layer
        # Take the LAST layer's hidden states
        last_hidden = lm_outputs.hidden_states[-1]   # (B, seq_len, hidden_size)

        # For each sample in the batch, read the hidden state at asst_start_pos[i]
        # This is the representation at the boundary where the LLM starts generating
        # the assistant response — it has "seen" all frames and the full PROMPT_G.
        if asst_start_pos is not None:
            # Clamp positions to valid range (in case of truncation)
            positions = asst_start_pos.clamp(0, last_hidden.shape[1] - 1)
            batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            boundary_hidden = last_hidden[batch_indices, positions, :]   # (B, hidden_size)
        else:
            # Fallback: use the last non-padded token
            seq_lengths = attention_mask.sum(dim=1) - 1   # (B,) last valid position
            seq_lengths = seq_lengths.clamp(0, last_hidden.shape[1] - 1)
            batch_indices = torch.arange(last_hidden.shape[0], device=last_hidden.device)
            boundary_hidden = last_hidden[batch_indices, seq_lengths, :]   # (B, hidden_size)

        # ── Score prediction ──────────────────────────────────────────────
        # ScoreHead outputs a raw logit (no sigmoid).
        # We use bce_with_logits for the loss (autocast-safe) and apply
        # sigmoid only on the returned score_pred for metric computation.
        score_logit = self.score_head(boundary_hidden.float())   # (B,) raw logit

        # ── Score loss ────────────────────────────────────────────────────
        score_loss = None
        if score_target is not None:
            score_loss = nn.functional.binary_cross_entropy_with_logits(
                score_logit,
                score_target.float().to(score_logit.device),
            )
        score_pred = torch.sigmoid(score_logit)   # (B,) in [0,1] for metrics

        # ── Combined loss ─────────────────────────────────────────────────
        combined_loss = None
        if lm_loss is not None and score_loss is not None:
            combined_loss = (
                self.loss_alpha * score_loss
                + (1.0 - self.loss_alpha) * lm_loss
            )
        elif lm_loss is not None:
            combined_loss = lm_loss
        elif score_loss is not None:
            combined_loss = score_loss

        return CollisionModelOutput(
            loss       = combined_loss,
            lm_loss    = lm_loss,
            score_loss = score_loss,
            score_pred = score_pred,
            logits     = lm_outputs.logits,
        )

    # ── Inference helper ──────────────────────────────────────────────────────

    @torch.no_grad()
    def get_score(
        self,
        pixel_values:     torch.Tensor,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        num_patches_list: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Inference-only: compute P(collision) without any generation or labels.

        The ScoreHead reads the hidden state at position [-1] (last input token).
        This is called once before generate() to get the collision score.

        Returns:
            score_pred (B,)  float in [0, 1]
        """
        n_images    = pixel_values.shape[0]
        image_flags = torch.ones(n_images, dtype=torch.long, device=pixel_values.device)

        lm_outputs = self.model(
            pixel_values     = pixel_values,
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            image_flags      = image_flags,
            output_hidden_states = True,
        )

        last_hidden  = lm_outputs.hidden_states[-1]          # (B, seq, H)
        seq_lengths  = attention_mask.sum(dim=1) - 1          # (B,)
        seq_lengths  = seq_lengths.clamp(0, last_hidden.shape[1] - 1)
        batch_idx    = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        last_valid   = last_hidden[batch_idx, seq_lengths, :]  # (B, H)

        logit = self.score_head(last_valid.float())             # (B,) raw logit
        return torch.sigmoid(logit)                              # (B,) P(collision)

    def generate_reasoning(
        self,
        tokenizer,
        pixel_values:     torch.Tensor,
        prompt_text:      str,
        num_patches_list: Optional[List[int]] = None,
        generation_config: Optional[dict]     = None,
    ) -> str:
        """
        Generate the full PROMPT_G JSON reasoning string using model.chat().
        This is identical to zero_shot_eval's generation call, but the model
        now has LoRA weights applied.
        """
        if num_patches_list is None:
            num_patches_list = [1] * pixel_values.shape[0]

        gen_cfg = generation_config or {
            "max_new_tokens": 600,
            "do_sample": False,
        }

        return self.model.chat(
            tokenizer        = tokenizer,
            pixel_values     = pixel_values,
            question         = prompt_text,
            generation_config= gen_cfg,
            num_patches_list = num_patches_list,
            history          = None,
            return_history   = False,
        )


# ── Factory functions ──────────────────────────────────────────────────────────

def load_for_training(
    model_id:    str,
    cfg:         dict,
    device_map:  str = "auto",
) -> Tuple[InternVLCollisionModel, object]:
    """
    Load InternVL3.5-4B-Flash, apply LoRA + ScoreHead, and return
    (model, tokenizer) ready for training.

    Args:
        model_id:   HuggingFace model ID.
        cfg:        Dict from train_lora.yaml (must contain LoRA and dtype keys).
        device_map: "auto" for multi-GPU or single GPU; "cpu" for debugging.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    print(f"Loading base model: {model_id}  dtype={torch_dtype}")
    # Do NOT pass device_map — InternVLChatModel doesn't implement all_tied_weights_keys
    # which newer transformers versions require for any device_map value.
    # Load on CPU first, then move to GPU manually.
    base_model = AutoModel.from_pretrained(
        model_id,
        torch_dtype          = torch_dtype,
        trust_remote_code    = True,
        low_cpu_mem_usage    = True,
        attn_implementation  = "flash_attention_2",   # avoids O(n²) eager attention OOM
    )
    print(f"[internvl_lora] attn_implementation = flash_attention_2")
    if device_map != "cpu" and torch.cuda.is_available():
        base_model = base_model.cuda()
    base_model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )

    # ── Set img_context_token_id ──────────────────────────────────────────
    # InternVL leaves this as None after from_pretrained(); it's normally set
    # inside chat().  We must set it before building the dataset so that
    # forward() can find image token positions in input_ids.
    _probe_and_set_img_context_token_id(base_model, tokenizer)

    # ── LoRA config ────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type     = TaskType.CAUSAL_LM,
        r             = cfg.get("lora_r", 16),
        lora_alpha    = cfg.get("lora_alpha", 32),
        lora_dropout  = cfg.get("lora_dropout", 0.1),
        target_modules= cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias          = "none",
    )

    collision_model = InternVLCollisionModel(
        base_model  = base_model,
        lora_config = lora_config,
        loss_alpha  = cfg.get("loss_alpha", 0.5),
    )

    return collision_model, tokenizer


def load_from_checkpoint(
    checkpoint_dir: str,
    model_id:       str,
    cfg:            dict,
    device_map:     str = "auto",
) -> Tuple[InternVLCollisionModel, object]:
    """
    Load a previously saved InternVLCollisionModel (LoRA adapters + ScoreHead)
    from a checkpoint directory.

    The checkpoint directory should contain:
      - adapter_config.json + adapter_model.bin  (PEFT LoRA weights)
      - score_head.pt                             (ScoreHead weights)

    Args:
        checkpoint_dir: Path to the checkpoint folder saved by train_lora.py.
        model_id:       Same HuggingFace base model ID used during training.
        cfg:            Dict from train_lora.yaml.
        device_map:     Device placement strategy.
    """
    import os
    from peft import PeftModel

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.get("torch_dtype", "bfloat16"), torch.bfloat16)

    print(f"Loading base model for inference: {model_id}")
    base_model = AutoModel.from_pretrained(
        model_id,
        torch_dtype         = torch_dtype,
        trust_remote_code   = True,
        low_cpu_mem_usage   = True,
        attn_implementation = "flash_attention_2",   # avoids O(n²) eager attention OOM
    )
    if device_map != "cpu" and torch.cuda.is_available():
        base_model = base_model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )

    # Set img_context_token_id (same reason as in load_for_training)
    _probe_and_set_img_context_token_id(base_model, tokenizer)

    # Load LoRA adapters onto the LLM backbone
    print(f"Loading LoRA adapters from: {checkpoint_dir}")
    base_model.language_model = PeftModel.from_pretrained(
        base_model.language_model,
        checkpoint_dir,
        torch_dtype = torch_dtype,
    )

    # Wrap in InternVLCollisionModel (no second LoRA application)
    # We freeze everything and just attach the ScoreHead
    for param in base_model.parameters():
        param.requires_grad = False

    hidden_size = _get_hidden_size_from_model(base_model)
    score_head = ScoreHead(hidden_size)

    # Load saved ScoreHead weights
    score_head_path = os.path.join(checkpoint_dir, "score_head.pt")
    if os.path.exists(score_head_path):
        state = torch.load(score_head_path, map_location="cpu")
        score_head.load_state_dict(state)
        print(f"ScoreHead weights loaded from {score_head_path}")
    else:
        print(f"WARNING: score_head.pt not found in {checkpoint_dir}, using random weights")

    # Build a minimal wrapper that exposes the same interface
    class _LoadedModel(InternVLCollisionModel):
        def __init__(self):
            # Bypass __init__ — manually set attributes
            nn.Module.__init__(self)
            self.model      = base_model
            self.score_head = score_head
            self.loss_alpha = cfg.get("loss_alpha", 0.5)

    loaded = _LoadedModel()
    loaded.eval()
    # Move ScoreHead to same device as model
    device = next(base_model.parameters()).device
    loaded.score_head = loaded.score_head.to(device=device, dtype=torch_dtype)

    return loaded, tokenizer


def _get_hidden_size_from_model(base_model) -> int:
    lm  = base_model.language_model
    cfg = getattr(lm, "config", None)
    if cfg:
        for attr in ("hidden_size", "d_model", "n_embd"):
            val = getattr(cfg, attr, None)
            if val:
                return int(val)
    try:
        # Try the base model's embed_tokens
        lm_base = getattr(lm, "base_model", lm)
        lm_model = getattr(lm_base, "model", lm_base)
        return lm_model.embed_tokens.weight.shape[1]
    except Exception:
        return 2048
