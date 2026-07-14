"""
Stage 2: fine-tune the ReverseBERT decoder on CRASH reasonings (text-only).

Self-supervised reconstruction (text autoencoder over a frozen embedding):
    crash reasoning TEXT -> EmbeddingGemma-300m (FROZEN) -> 768-d vector
                         -> EmbeddingProjector (TRAIN)   -> 4 soft tokens
                         -> Qwen3-0.6B-Base + LoRA (TRAIN)-> regenerate the SAME text
    loss = CE(generated tokens, original reasoning tokens)   [soft-token positions masked -100]

Encoder frozen; we train only the decoder side (projector + Qwen LoRA).
Recipe adapted from github.com/fakerybakery/ReverseBERT/main.py.

Device-aware:
  * CUDA  -> fp16 autocast + GradScaler (RunPod, the real run)
  * CPU   -> fp32, no autocast (local dry-run / correctness check via --max-steps)

No prompt/query here -- the 4 soft tokens are the entire conditioning. The collision
query belongs to the (future) Predictor stage, not the decoder.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]

ENCODER = "google/embeddinggemma-300m"
BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
EMOACT_REPO = "mrfakename/ReverseBERT-EmbeddingGemma-300M"  # warm-start source

DEFAULT_TRAIN = REPO_ROOT / "outputs" / "teacher_reasoning" / "Teacher_Reasoning_Train_All_Clips.jsonl"


class EmbeddingProjector(nn.Module):
    """Verbatim from ReverseBERT: 768 -> (num_tokens x hidden) soft tokens."""

    def __init__(self, input_dim=768, output_dim=2048, num_tokens=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim * num_tokens),
        )
        self.output_dim = output_dim

    def forward(self, embeddings):
        projected = self.proj(embeddings)
        return projected.view(-1, self.num_tokens, self.output_dim)


class ReconDataset(Dataset):
    """Holds precomputed 768-d embeddings + tokenized targets (same text)."""

    def __init__(self, embeddings, input_ids, attention_mask):
        self.embeddings = embeddings
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.embeddings.size(0)

    def __getitem__(self, i):
        return {
            "embedding": self.embeddings[i],
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
        }


def load_texts(jsonl_path: Path, field: str, limit: int = 0) -> list[str]:
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get(field)
            if isinstance(t, str) and t.strip():
                texts.append(t.strip())
    if limit:
        texts = texts[:limit]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default=str(DEFAULT_TRAIN))
    ap.add_argument("--field", default="final_reasoning")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "decoder_roundtrip" / "finetuned"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--num-tokens", type=int, default=4)
    ap.add_argument("--from-scratch", action="store_true", help="random projector + fresh LoRA (default: warm-start from emoact)")
    ap.add_argument("--limit", type=int, default=0, help="use only N texts (dry-run)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (dry-run)")
    ap.add_argument("--save", action="store_true", help="save adapter + projector at the end")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    dtype = torch.float16 if use_amp else torch.float32
    warm_start = not args.from_scratch
    print(f"[cfg] device={device} amp={use_amp} warm_start={warm_start} "
          f"epochs={args.epochs} eff_batch={args.batch_size*args.grad_accum}")

    # ---- data -----------------------------------------------------------
    texts = load_texts(Path(args.train_jsonl), args.field, args.limit)
    print(f"[data] {len(texts)} reasonings from {args.train_jsonl} (field={args.field})")

    print(f"[load] encoder {ENCODER} (frozen)")
    encoder = SentenceTransformer(ENCODER, device=device, trust_remote_code=True)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    print("[data] precomputing embeddings ...")
    with torch.no_grad():
        emb = encoder.encode(texts, convert_to_tensor=True, batch_size=16, show_progress_bar=False)
    emb = emb.to("cpu").float()  # [N, 768]

    print(f"[load] base LLM {BASE_MODEL} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tok = tokenizer(texts, max_length=args.max_len, padding="max_length",
                    truncation=True, return_tensors="pt")

    ds = ReconDataset(emb, tok["input_ids"], tok["attention_mask"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    hidden = base.config.hidden_size
    base = base.to(device)

    if warm_start:
        print(f"[init] warm-start LoRA from {EMOACT_REPO}")
        llm = PeftModel.from_pretrained(base, EMOACT_REPO, is_trainable=True)
    else:
        print("[init] fresh LoRA (from scratch)")
        lora_config = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        llm = get_peft_model(base, lora_config)
    llm.print_trainable_parameters()

    projector = EmbeddingProjector(768, hidden, args.num_tokens).to(device)
    if warm_start:
        ppath = hf_hub_download(repo_id=EMOACT_REPO, filename="reverse_bert_projector.pt")
        projector.load_state_dict(torch.load(ppath, map_location=device, weights_only=True))
        print("[init] warm-start projector from emoact checkpoint")
    projector = projector.to(dtype)

    trainable = [p for p in llm.parameters() if p.requires_grad] + list(projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    llm.train()
    projector.train()
    emb_layer = llm.get_input_embeddings()

    def run_step(batch):
        embeddings = batch["embedding"].to(device)
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        prefix = projector(embeddings)                      # [B, T, H]
        token_embeds = emb_layer(input_ids)                 # [B, L, H]
        prefix = prefix.to(token_embeds.dtype)
        inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        prefix_mask = torch.ones(embeddings.size(0), prefix.size(1), device=device, dtype=attn.dtype)
        full_mask = torch.cat([prefix_mask, attn], dim=1)
        prefix_labels = torch.full((embeddings.size(0), prefix.size(1)), -100, device=device, dtype=input_ids.dtype)
        labels = torch.cat([prefix_labels, input_ids], dim=1)
        out = llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        return out.loss

    steps = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        total = 0.0
        for i, batch in enumerate(loader):
            if use_amp:
                with torch.cuda.amp.autocast():
                    loss = run_step(batch) / args.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = run_step(batch) / args.grad_accum
                loss.backward()
            total += loss.item() * args.grad_accum

            if (i + 1) % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                steps += 1
                if args.max_steps and steps >= args.max_steps:
                    print(f"[dry-run] stop after {steps} optimizer steps; last loss={loss.item()*args.grad_accum:.4f}")
                    print(f"[dry-run] OK — loop, masking, and backward all run on {device}.")
                    return
        print(f"[epoch {epoch+1}/{args.epochs}] avg_loss={total/max(1,len(loader)):.4f}  ({time.time()-t0:.1f}s)")

        if args.save:
            ep_dir = Path(args.out_dir) / f"epoch_{epoch+1:02d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            llm.save_pretrained(str(ep_dir / "adapter"))
            torch.save(projector.state_dict(), ep_dir / "reverse_bert_projector.pt")
            print(f"[save] {ep_dir}")

    print(f"[done] {steps} optimizer steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
