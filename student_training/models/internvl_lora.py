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
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig


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
    Single linear layer + Sigmoid.

    Input:  hidden state at the first assistant-token position
            shape: (B, hidden_size)
    Output: P(collision) ∈ [0, 1]  shape: (B,)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x)).squeeze(-1)   # (B,)


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
        num_patches_list: Optional[List[int]] = None,
    ) -> CollisionModelOutput:
        """
        Single forward pass that computes both LM loss and ScoreHead loss.

        The LLM is run with output_hidden_states=True so we can extract
        the hidden state at asst_start_pos for the ScoreHead.

        Returns CollisionModelOutput with combined loss, component losses,
        score predictions, and LM logits.
        """
        if num_patches_list is None:
            batch_size = input_ids.shape[0]
            num_patches_list = [1] * (batch_size * 16)   # default: 16 frames, 1 tile each

        # ── LM forward ────────────────────────────────────────────────────
        # InternVL's forward() returns a CausalLMOutputWithPast.
        # Passing output_hidden_states=True makes it include the LLM's
        # layer-by-layer hidden states in .hidden_states.
        lm_outputs = self.model(
            pixel_values     = pixel_values,
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            labels           = labels,
            output_hidden_states = True,
            num_patches_list = num_patches_list,
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
        score_pred = self.score_head(boundary_hidden.float())   # (B,) in [0,1]
        # Cast to float32 for loss stability regardless of model dtype

        # ── Score loss ────────────────────────────────────────────────────
        score_loss = None
        if score_target is not None:
            score_loss = nn.functional.binary_cross_entropy(
                score_pred,
                score_target.float().to(score_pred.device),
            )

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
        if num_patches_list is None:
            batch_size = input_ids.shape[0]
            num_patches_list = [1] * (batch_size * 16)

        lm_outputs = self.model(
            pixel_values     = pixel_values,
            input_ids        = input_ids,
            attention_mask   = attention_mask,
            output_hidden_states = True,
            num_patches_list = num_patches_list,
        )

        last_hidden  = lm_outputs.hidden_states[-1]          # (B, seq, H)
        seq_lengths  = attention_mask.sum(dim=1) - 1          # (B,)
        seq_lengths  = seq_lengths.clamp(0, last_hidden.shape[1] - 1)
        batch_idx    = torch.arange(last_hidden.shape[0], device=last_hidden.device)
        last_valid   = last_hidden[batch_idx, seq_lengths, :]  # (B, H)

        return self.score_head(last_valid.float())             # (B,)

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
    # Use {"": 0} instead of "auto" — InternVL doesn't implement all_tied_weights_keys
    # which transformers' infer_auto_device_map requires. {"": 0} puts everything on GPU 0.
    resolved_device_map = {"": 0} if device_map == "auto" else device_map
    base_model = AutoModel.from_pretrained(
        model_id,
        torch_dtype      = torch_dtype,
        trust_remote_code= True,
        low_cpu_mem_usage= True,
        device_map       = resolved_device_map,
    )
    base_model.train()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )

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
    resolved_device_map = {"": 0} if device_map == "auto" else device_map
    base_model = AutoModel.from_pretrained(
        model_id,
        torch_dtype       = torch_dtype,
        trust_remote_code = True,
        low_cpu_mem_usage = True,
        device_map        = resolved_device_map,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, use_fast=True
    )

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
