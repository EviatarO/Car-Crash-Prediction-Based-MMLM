from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .temporal_mixer import TemporalTokenMixer


@dataclass
class ModelOutput:
    score_logits: torch.Tensor
    lm_logits: Optional[torch.Tensor]
    score_index: torch.Tensor


class MCAStudentModel(nn.Module):
    def __init__(
        self,
        vision_model_id: str,
        llm_model_id: str,
        score_token: str = "<SCORE>",
        mixer_layers: int = 2,
        mixer_heads: int = 8,
        max_frames: int = 64,
        use_4bit: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("transformers is required to load models") from exc

        self.vision = CLIPVisionModel.from_pretrained(vision_model_id)
        llm_kwargs = {}
        if use_4bit:
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            llm_kwargs["device_map"] = "auto"
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_id, **llm_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if score_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([score_token])
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.score_token = score_token

        vision_dim = self.vision.config.hidden_size
        llm_dim = self.llm.config.hidden_size

        self.temporal_mixer = TemporalTokenMixer(embed_dim=vision_dim, num_layers=mixer_layers, num_heads=mixer_heads)
        self.temporal_embeddings = nn.Embedding(max_frames, vision_dim)
        self.projector = nn.Linear(vision_dim, llm_dim)
        self.score_head = nn.Linear(llm_dim, 1)

    def _encode_visual(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, T, C, H, W)
        bsz, t, c, h, w = clip.shape
        flat = clip.view(bsz * t, c, h, w)
        vision_out = self.vision(pixel_values=flat)
        tokens = vision_out.last_hidden_state[:, 1:, :]  # drop CLS
        _, m, d = tokens.shape
        tokens = tokens.view(bsz, t, m, d)

        frame_idx = torch.arange(t, device=clip.device)
        frame_emb = self.temporal_embeddings(frame_idx).view(1, t, 1, d)
        tokens = tokens + frame_emb
        tokens = tokens.view(bsz, t * m, d)
        tokens = self.temporal_mixer(tokens)
        return tokens

    def forward(
        self,
        clip: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        score_token_index: torch.Tensor,
    ) -> ModelOutput:
        # Visual prefix tokens
        visual_tokens = self._encode_visual(clip)
        visual_embeds = self.projector(visual_tokens)
        bsz, vlen, _ = visual_embeds.shape

        # Text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        full_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

        visual_mask = torch.ones((bsz, vlen), dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([visual_mask, attention_mask], dim=1)

        outputs = self.llm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]

        # score_token_index is relative to text; shift by visual prefix length
        score_pos = score_token_index + vlen
        score_vec = last_hidden[torch.arange(bsz, device=clip.device), score_pos]
        score_logits = self.score_head(score_vec).squeeze(-1)

        return ModelOutput(score_logits=score_logits, lm_logits=outputs.logits, score_index=score_pos)


def build_student_model(
    vision_model_id: str,
    llm_model_id: str,
    score_token: str,
    mixer_layers: int = 2,
    mixer_heads: int = 8,
    max_frames: int = 64,
    use_4bit: bool = False,
    lora_r: int = 0,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
) -> MCAStudentModel:
    model = MCAStudentModel(
        vision_model_id=vision_model_id,
        llm_model_id=llm_model_id,
        score_token=score_token,
        mixer_layers=mixer_layers,
        mixer_heads=mixer_heads,
        max_frames=max_frames,
        use_4bit=use_4bit,
    )
    if lora_r and lora_r > 0:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("peft is required for LoRA") from exc
        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=lora_target_modules,
            task_type=TaskType.CAUSAL_LM,
        )
        model.llm = get_peft_model(model.llm, config)
    return model
