"""
vjepa_reason.py
===============
Stage B of `e4_vjepa_reason` — the V-JEPA2 -> projector -> Qwen3-4B bridge.

Three components:

  1. VJEPA2FeatureExtractor — wraps the FROZEN BADAS-Open model (loaded via the
     Stage-A loader) and taps the **patch grid** (~1568 tok x 1024-d) that feeds
     the attentive probe, via a forward-pre-hook on the `pooler` module. Also
     returns the frozen P(collision) (free byproduct) for the §5(5) per-TTE
     characterization. Used ONLY by the caching script (heavy GPU deps).

  2. ResamplerProjector — TRAINABLE Perceiver-style resampler: `num_queries`
     learned tokens cross-attend over the patch grid -> fixed `num_queries`
     visual tokens in the LLM embedding dim. PoolMLPProjector is the lower-
     variance fallback (config flag).

  3. StageBBridge — FROZEN Qwen3-4B + the trainable projector. Scatters the
     projected visual tokens into the prompt's input embeddings and computes
     teacher-forced CE on the reasoning span. Supports ablation modes
     ("none"/"zero"/"shuffle"/"mask") for the §5 gate metrics.

Only the projector trains in Stage B. The LLM, the V-JEPA2 trunk, the attentive
probe and the BADAS classifier are all frozen (LoRA is OFF — that is Stage C).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1) Frozen V-JEPA2 feature extractor (patch grid + score)  — caching only
# =============================================================================

class VJEPA2FeatureExtractor:
    """Frozen BADAS-Open wrapper exposing the pre-probe patch grid + P(collision).

    Reuses the Stage-A loader (`load_badas`, `preprocess_clip`). A forward-pre-hook
    on the attentive-probe (`pooler`) module captures its input — exactly the
    ~1568x1024 patch grid before the score-oriented bottleneck.
    """

    def __init__(self, stagea_cfg: dict, temperature: float = 2.0):
        # Import here so this module stays importable without the `badas` package.
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from e4_stageA_badas_open_eval import load_badas, preprocess_clip  # noqa: E402

        self._preprocess_clip = preprocess_clip
        self.vjepa, self.nn_model, self.device = load_badas(stagea_cfg)
        self.temperature = temperature

        # Locate the aggregation module to hook (tap point = patch grid before pooling).
        # EnhancedVideoClassifier uses `temporal_processor`; older BADAS builds use `pooler`.
        probe = getattr(self.nn_model, "temporal_processor", None)
        if probe is None:
            probe = getattr(self.nn_model, "pooler", None)
        if probe is None:
            for name, mod in self.nn_model.named_modules():
                low = name.lower()
                if ("temporal" in low or low.endswith("pooler")
                        or "probe" in low or "attentive" in low):
                    probe = mod
                    print(f"  [extractor] hooking probe module by search: '{name}'")
                    break
        if probe is None:
            raise RuntimeError(
                "Could not locate the attentive-probe module on the BADAS-Open "
                "model. Inspect `nn_model.named_modules()` and set the tap point."
            )
        self._captured = {}

        def _pre_hook(_module, args):
            # args[0] is the patch grid fed to the probe: (B, P, D)
            self._captured["patches"] = args[0].detach()

        probe.register_forward_pre_hook(_pre_hook)

    @torch.no_grad()
    def extract(self, frame_paths: list):
        """Run the frozen model on one 16-frame window.

        Returns (patch_grid (P, D) cpu fp16, score float in [0,1]).
        """
        clip = self._preprocess_clip(self.vjepa, frame_paths).to(self.device)
        self._captured.clear()
        logits = self.nn_model(clip)                       # (1, 2)
        patches = self._captured.get("patches")
        if patches is None:
            raise RuntimeError("probe pre-hook did not fire — tap point is wrong.")
        patches = patches[0].float().cpu()                 # (P, D)
        score = float(torch.softmax(logits / self.temperature, dim=1)[0, 1].item())
        return patches.half(), score


# =============================================================================
# 2) Projectors (trainable)
# =============================================================================

class ResamplerProjector(nn.Module):
    """Perceiver/Q-Former-lite resampler.

    `num_queries` learned tokens cross-attend over the patch grid, then a self-
    attention + FFN block, then a linear map to the LLM embedding dim.

    Input : (B, P, in_dim)   patch grid
    Output: (B, num_queries, out_dim)
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 2560,
                 num_queries: int = 64, n_heads: int = 8,
                 hidden_dim: int = 512, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.in_dim = in_dim
        self.out_dim = out_dim
        d = hidden_dim

        # Patches are projected to a SMALLER latent dim (Flamingo/BLIP-2 style) so
        # the attention + FFN params stay modest — ~6M total, suited to ~267 examples.
        self.in_proj = nn.Linear(in_dim, d)
        self.queries = nn.Parameter(torch.randn(num_queries, d) * 0.02)

        self.ln_q   = nn.LayerNorm(d)
        self.ln_kv  = nn.LayerNorm(d)
        self.cross  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

        self.ln_s   = nn.LayerNorm(d)
        self.selfa  = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

        self.ln_f   = nn.LayerNorm(d)
        self.ffn    = nn.Sequential(
            nn.Linear(d, d * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d * ffn_mult, d),
        )

        self.ln_out = nn.LayerNorm(d)
        self.out    = nn.Linear(d, out_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B = patches.shape[0]
        kv = self.ln_kv(self.in_proj(patches))                            # (B, P, d)
        q = self.queries.unsqueeze(0).expand(B, -1, -1).to(kv.dtype)      # (B, Q, d)

        attn, _ = self.cross(self.ln_q(q), kv, kv, need_weights=False)
        q = q + attn

        s = self.ln_s(q)
        sa, _ = self.selfa(s, s, s, need_weights=False)
        q = q + sa

        q = q + self.ffn(self.ln_f(q))
        return self.out(self.ln_out(q))                                   # (B, Q, out_dim)


class PoolMLPProjector(nn.Module):
    """Lower-variance fallback: adaptive-pool the patch grid to `num_queries`
    tokens, then a 2-layer MLP to the LLM embedding dim. (~half the params.)

    Input : (B, P, in_dim)  ->  Output: (B, num_queries, out_dim)
    """

    def __init__(self, in_dim: int = 1024, out_dim: int = 2560,
                 num_queries: int = 64, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.ln_in = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.ln_out = nn.LayerNorm(out_dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.ln_in(patches).transpose(1, 2)               # (B, in_dim, P)
        x = F.adaptive_avg_pool1d(x, self.num_queries)        # (B, in_dim, Q)
        x = x.transpose(1, 2)                                 # (B, Q, in_dim)
        return self.ln_out(self.mlp(x))                       # (B, Q, out_dim)


def build_projector(cfg: dict) -> nn.Module:
    pj = cfg["projector"]
    variant = pj.get("variant", "resampler")
    if variant == "resampler":
        return ResamplerProjector(
            in_dim=pj["in_dim"], out_dim=pj["out_dim"],
            num_queries=pj["num_queries"], n_heads=pj.get("n_heads", 8),
            hidden_dim=pj.get("hidden_dim", 512),
            ffn_mult=pj.get("ffn_mult", 4), dropout=pj.get("dropout", 0.1),
        )
    if variant == "pool_mlp":
        return PoolMLPProjector(
            in_dim=pj["in_dim"], out_dim=pj["out_dim"],
            num_queries=pj["num_queries"], dropout=pj.get("dropout", 0.1),
        )
    raise ValueError(f"unknown projector variant: {variant}")


# =============================================================================
# 3) Stage-B bridge: frozen Qwen3-4B + trainable projector
# =============================================================================

class StageBBridge(nn.Module):
    """Frozen Qwen3-4B LM + trainable projector.

    Visual tokens are scattered into the prompt's input embeddings at the
    positions flagged by `vis_mask`. Only the projector has gradients.

    `ablate` controls the gate experiments (§5):
      "none"    — real visual tokens (the trained model)
      "zero"    — visual tokens set to 0 (destroy visual signal)
      "shuffle" — use `shuffle_feats` from another clip (mismatched visual signal)
      "mask"    — attention masked off the visual positions (= text-only prior)
    """

    def __init__(self, llm, projector: nn.Module):
        super().__init__()
        self.llm = llm
        self.projector = projector
        for p in self.llm.parameters():          # freeze the LLM (LoRA OFF in Stage B)
            p.requires_grad = False
        self.llm.eval()                          # keep dropout off in the frozen LM

    @property
    def embed_dim(self) -> int:
        return self.llm.get_input_embeddings().weight.shape[1]

    def _make_inputs_embeds(self, vis_feats, input_ids, vis_mask,
                            ablate="none", shuffle_feats=None):
        embed_layer = self.llm.get_input_embeddings()
        embeds = embed_layer(input_ids)                              # (B, L, H)

        if ablate == "shuffle" and shuffle_feats is not None:
            vis_feats = shuffle_feats
        vis_tok = self.projector(vis_feats)                         # (B, Q, H)
        if ablate == "zero":
            vis_tok = torch.zeros_like(vis_tok)

        embeds = embeds.clone()
        embeds[vis_mask] = vis_tok.reshape(-1, vis_tok.shape[-1]).to(embeds.dtype)
        return embeds

    def forward(self, vis_feats, input_ids, attention_mask, labels, vis_mask,
                ablate="none", shuffle_feats=None):
        embeds = self._make_inputs_embeds(
            vis_feats, input_ids, vis_mask, ablate, shuffle_feats)
        am = attention_mask
        if ablate == "mask":
            am = attention_mask.clone()
            am[vis_mask] = 0                                          # hide visual tokens
        out = self.llm(inputs_embeds=embeds, attention_mask=am, labels=labels)
        # With labels masked to the reason span only, out.loss is the mean
        # token-level CE over that span — exactly L_B (plan §3.1).
        return out.loss, out.logits

    @torch.no_grad()
    def generate(self, vis_feats, input_ids, attention_mask, vis_mask,
                 max_new_tokens=160, **gen_kwargs):
        embeds = self._make_inputs_embeds(vis_feats, input_ids, vis_mask, "none")
        return self.llm.generate(
            inputs_embeds=embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False, **gen_kwargs)


def load_llm(model_id: str, dtype=torch.bfloat16):
    """Load the standalone Qwen3-4B LM + tokenizer (frozen base for Stage B)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True)
    return llm, tok
