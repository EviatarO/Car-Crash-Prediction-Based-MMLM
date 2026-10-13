"""
r1_bridge.py - reasoning-path bridge for week 1 (plan: 2026-10-05_Plan-Week1-GoNoGo-rev3).

    frozen V-JEPA2 tokens (8 time x 16 x 16 x 1024, cached)
      -> Merger  (Qwen3-VL's own projector STRUCTURE: LN -> 2x2 regroup -> Linear -> GELU -> Linear)  TRAINED
      -> 512 visual tokens x 2560, placed into Qwen3-VL's native video prompt
         (per-step "<t seconds>" + vision_start/end tokens, 3D M-RoPE positions)
      -> Qwen3-VL-4B language model (frozen; LoRA optional in Phase 2) -> text

Qwen3-VL's own ViT and merger weights are never used (the merger weights can optionally seed our
merger for the init A/B test). We do NOT call Qwen3VLModel.forward: it would run Qwen's vision tower.
Instead we scatter our visual tokens into inputs_embeds ourselves and call the language model with
3D position ids from the model's own `get_rope_index` - the same functions the official forward uses.
`r1_bridge_test.py` proves logit-equivalence with the official path.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_T, GRID, MERGE = 8, 16, 2            # V-JEPA2 tokens: 8 time steps x 16 x 16
TOK_PER_STEP = (GRID // MERGE) ** 2    # 64 merged tokens per time step
N_VIS = N_T * TOK_PER_STEP             # 512
WIN_FPS = 7.5                          # 16 frames over 2 s (every 4th frame of 30 fps)
IGNORE = -100
# merger input types (plan 2026-10-13): name -> (layers fed [1 = last only, 2 = layer 11 + last], attention channel appended)
INPUTS = {"last": (1, False), "last+attn": (1, True), "l11+last": (2, False), "l11+last+attn": (2, True)}


def regroup_2x2(x: torch.Tensor) -> torch.Tensor:
    """(B, 2048, C) V-JEPA tokens, index = t*256 + h*16 + w  ->  (B, 512, 4*C).
    Merged token (t, gh, gw) concatenates its 2x2 block in the order (0,0),(0,1),(1,0),(1,1) -
    Qwen3-VL's merge-group order; merged tokens are in (t, gh, gw) raster order."""
    B, N, C = x.shape
    assert N == N_T * GRID * GRID, f"expected {N_T * GRID * GRID} tokens, got {N}"
    x = x.view(B, N_T, GRID // 2, 2, GRID // 2, 2, C)       # t, gh, a, gw, b, C
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous()         # t, gh, gw, a, b, C
    return x.view(B, N_VIS, 4 * C)


class Merger(nn.Module):
    """Same structure as Qwen3VLVisionPatchMerger (use_postshuffle_norm=False):
    LayerNorm(1024) on each token -> concat a 2x2 block (4096) -> Linear -> GELU -> Linear(2560)."""

    def __init__(self, in_dim: int = 1024, out_dim: int = 2560, n_layers: int = 1, attn: bool = False):
        super().__init__()
        self.in_dim, self.n_layers, self.attn = in_dim, n_layers, attn
        self.hidden = in_dim * MERGE * MERGE
        self.norm = nn.LayerNorm(in_dim, eps=1e-6)                     # the last-layer tokens (the only input of the original model)
        if n_layers == 2 or attn:                                      # richer input (plan 2026-10-13): fuse per token back to in_dim
            self.norm_l11 = nn.LayerNorm(in_dim, eps=1e-6) if n_layers == 2 else None
            self.fuse = nn.Linear(n_layers * in_dim + int(attn), in_dim)
        self.linear_fc1 = nn.Linear(self.hidden, self.hidden)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden, out_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:        # (B, 2048, C) -> (B, 512, out)
        """C = n_layers * 1024 (+1 attention channel). Channel order: [layer-11 tokens], last-layer tokens, [attention]."""
        feats = feats.float()
        d = self.in_dim
        if self.n_layers == 1 and not self.attn:                    # original model, unchanged parameters and numbers
            return self.linear_fc2(self.act_fn(self.linear_fc1(regroup_2x2(self.norm(feats)))))
        parts, k = [], 0
        if self.n_layers == 2:
            parts.append(self.norm_l11(feats[..., :d]))
            k = d
        parts.append(self.norm(feats[..., k:k + d]))
        if self.attn:
            parts.append(feats[..., k + d:k + d + 1])
        x = self.fuse(torch.cat(parts, -1))
        return self.linear_fc2(self.act_fn(self.linear_fc1(regroup_2x2(x))))

    def load_from_qwen(self, qwen_merger: nn.Module):
        """Optional init: copy Qwen3-VL's pretrained merger weights (they translate Qwen-ViT features,
        so only the output half is meaningful for V-JEPA features - this is the init A/B test)."""
        self.load_state_dict({k: v.float().cpu() for k, v in qwen_merger.state_dict().items()})


class PromptBuilder:
    """Builds Qwen3-VL's native video prompt (same text the official processor produces) and the
    answer-only label mask. Verified token-for-token against Qwen3VLProcessor in r1_bridge_test.py."""

    def __init__(self, tokenizer):
        self.tok = tokenizer
        c = tokenizer.convert_tokens_to_ids
        self.video_id = c("<|video_pad|>")
        self.vs, self.ve = "<|vision_start|>", "<|vision_end|>"
        self.im_end_id = c("<|im_end|>")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.im_end_id

    @staticmethod
    def timestamps():
        # processor rule: average the timestamps of the 2 frames inside each temporal patch
        f = [i / WIN_FPS for i in range(2 * N_T)]
        return [(f[i] + f[i + 1]) / 2 for i in range(0, 2 * N_T, 2)]

    def video_block(self) -> str:
        return "".join(f"<{t:.1f} seconds>{self.vs}" + "<|video_pad|>" * TOK_PER_STEP + self.ve
                       for t in self.timestamps())

    def prompt_text(self, tag: str, question: str) -> str:
        q = f"{tag} {question}" if tag else question
        return f"<|im_start|>user\n{self.video_block()}{q}<|im_end|>\n<|im_start|>assistant\n"

    def encode(self, tag: str, question: str, answer: str | None = None, end: bool = True):
        """-> input_ids, labels (answer tokens + <|im_end|> only), mm_token_type_ids (2 = video).
        end=False scores an answer PREFIX (no <|im_end|>), e.g. 'Collision: yes' for the verdict probability."""
        p = self.tok(self.prompt_text(tag, question), add_special_tokens=False)["input_ids"]
        a = (self.tok(answer + ("<|im_end|>" if end else ""), add_special_tokens=False)["input_ids"] if answer is not None else [])
        ids = p + a
        labels = [IGNORE] * len(p) + a
        mm = [2 if i == self.video_id else 0 for i in ids]
        return ids, labels, mm

    def collate(self, items):
        """items: list of (ids, labels, mm). Right-padded."""
        L = max(len(i[0]) for i in items)
        pad = lambda seq, v: seq + [v] * (L - len(seq))
        ids = torch.tensor([pad(i[0], self.pad_id) for i in items])
        lab = torch.tensor([pad(i[1], IGNORE) for i in items])
        mm = torch.tensor([pad(i[2], 0) for i in items], dtype=torch.int32)
        att = torch.tensor([[1] * len(i[0]) + [0] * (L - len(i[0])) for i in items])
        return ids, lab, mm, att


class R1Bridge(nn.Module):
    """Frozen Qwen3-VL language model + trainable Merger (+ optional LoRA on the LM)."""

    def __init__(self, qwen, tokenizer, init: str = "random", in_dim: int = 1024, inp: str = "last"):
        super().__init__()
        self.qwen = qwen
        self.model = qwen.model
        out_dim = qwen.config.text_config.hidden_size
        n_layers, attn = INPUTS[inp]
        self.inp = inp
        self.merger = Merger(in_dim, out_dim, n_layers, attn)
        if init == "qwen":
            self.merger.load_from_qwen(self.model.visual.merger)
        elif init != "random":
            raise ValueError(init)
        if hasattr(self.model, "visual"):
            del self.model.visual                       # never used; frees ~0.4B params
        for p in self.qwen.parameters():
            p.requires_grad = False
        self.text = self.model.language_model            # may be replaced by a PEFT wrapper (enable_lora)
        self.prompts = PromptBuilder(tokenizer)
        self.grid = torch.tensor([[N_T, GRID, GRID]], dtype=torch.long)
        self.lm_dtype = next(self.text.parameters()).dtype

    # ---- trainable-parameter handling -------------------------------------------------
    def enable_lora(self, r=16, alpha=32, dropout=0.05,
                    targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
        from peft import LoraConfig, get_peft_model
        self.text = get_peft_model(self.model.language_model,
                                   LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                              target_modules=list(targets), bias="none"))
        for n, p in self.text.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()                 # fp32 master LoRA weights
                p.requires_grad = True

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    # ---- core forward -------------------------------------------------------------------
    def _inputs(self, vis, ids, mm, att):
        """vis (B, 512, D) merged visual tokens; ids (B, L). -> inputs_embeds, 3D position ids."""
        dev = ids.device
        emb = self.model.language_model.embed_tokens(ids)
        mask = (ids == self.prompts.video_id)
        assert int(mask.sum()) == vis.shape[0] * vis.shape[1], "video placeholder count != visual tokens"
        emb = emb.clone()
        emb[mask] = vis.reshape(-1, vis.shape[-1]).to(emb.dtype)
        grid = self.grid.to(dev).expand(ids.shape[0], 3)
        pos, _ = self.model.get_rope_index(ids, mm_token_type_ids=mm.to(dev), video_grid_thw=grid,
                                           attention_mask=att)
        return emb, pos

    def hidden_vis(self, vis, ids, mm, att):
        emb, pos = self._inputs(vis, ids, mm, att)
        return self.text(inputs_embeds=emb, position_ids=pos, attention_mask=att).last_hidden_state

    def forward_vis(self, vis, ids, mm, att):
        return self.qwen.lm_head(self.hidden_vis(vis, ids, mm, att))

    def visual_tokens(self, feats, mode="real", other=None):
        """mode: real | blank (zero V-JEPA features: the merger's 'no information' output) |
        wrong (use `other` features, i.e. another clip's tokens)."""
        if mode == "blank":
            feats = torch.zeros_like(feats)
        elif mode == "wrong":
            feats = other
        return self.merger(feats).to(self.lm_dtype)

    def loss_per_sample(self, feats, batch, mode="real", other=None):
        """Mean answer-token CE per sample (fp32). batch = prompts.collate(...) moved to device.
        The vocabulary projection is applied ONLY at answer positions (not at the ~600 prompt positions)."""
        ids, lab, mm, att = batch
        h = self.hidden_vis(self.visual_tokens(feats, mode, other), ids, mm, att)[:, :-1]
        tgt = lab[:, 1:]
        valid = tgt != IGNORE
        logits = self.qwen.lm_head(h[valid]).float()                      # (n_answer_tokens, vocab)
        ce = F.cross_entropy(logits, tgt[valid], reduction="none")
        sample_idx = torch.arange(ids.shape[0], device=ids.device).unsqueeze(1).expand_as(tgt)[valid]
        tot = torch.zeros(ids.shape[0], device=ids.device, dtype=ce.dtype).index_add(0, sample_idx, ce)
        cnt = valid.sum(1).clamp(min=1).to(ce.dtype)
        return tot / cnt

    @torch.no_grad()
    def generate(self, feats, tag, question, max_new_tokens=80, mode="real", other=None):
        """Greedy decoding for ONE window (feats (1, 2048, 1024))."""
        dev = feats.device
        ids, _, mm = self.prompts.encode(tag, question)
        ids_t = torch.tensor([ids], device=dev)
        mm_t = torch.tensor([mm], dtype=torch.int32, device=dev)
        att = torch.ones_like(ids_t)
        emb, pos = self._inputs(self.visual_tokens(feats, mode, other), ids_t, mm_t, att)
        out = self.text(inputs_embeds=emb, position_ids=pos, attention_mask=att, use_cache=True)
        pkv, nxt_pos = out.past_key_values, pos.max() + 1
        toks = []
        h = out.last_hidden_state[:, -1]
        for _ in range(max_new_tokens):
            t = self.qwen.lm_head(h).argmax(-1)
            if int(t) == self.prompts.im_end_id:
                break
            toks.append(int(t))
            e = self.model.language_model.embed_tokens(t.view(1, 1))
            p = nxt_pos.view(1, 1, 1).expand(3, 1, 1)
            att = torch.ones(1, att.shape[1] + 1, dtype=att.dtype, device=dev)
            out = self.text(inputs_embeds=e, position_ids=p, attention_mask=att,
                            past_key_values=pkv, use_cache=True)
            pkv, nxt_pos, h = out.past_key_values, nxt_pos + 1, out.last_hidden_state[:, -1]
        return self.prompts.tok.decode(toks, skip_special_tokens=True)

    # ---- checkpoints --------------------------------------------------------------------
    def save(self, path):
        sd = {"merger": self.merger.state_dict()}
        lora = {k: v.cpu() for k, v in self.text.state_dict().items() if "lora_" in k}
        if lora:
            sd["lora"] = lora
        torch.save(sd, path)

    def load(self, path, strict=True):
        sd = torch.load(path, map_location="cpu")
        self.merger.load_state_dict(sd["merger"])
        if "lora" in sd:
            missing = self.text.load_state_dict(sd["lora"], strict=False)
            assert not missing.unexpected_keys, missing.unexpected_keys
