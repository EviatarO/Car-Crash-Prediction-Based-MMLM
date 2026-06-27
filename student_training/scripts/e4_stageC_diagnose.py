"""
e4_stageC_diagnose.py
=====================
Diagnose why the V-JEPA2 projector fails to ground in a given LLM (e.g. Qwen3.5-4B
shows Δ-PPL ~0 vs random, while Qwen3-4B shows -48%). Tests three hypotheses, no
training, on the 18-val cache:

  (A) embedding-scale mismatch  — ||projector-token|| vs ||native token embedding||
  (B) chat-template mismatch    — do our hardcoded <|im_start|> tokens exist as
                                  single ids? does the model's own template differ?
  (C) visual contribution       — PPL with real / zeroed / text-only(masked) visual
                                  tokens (ΔCE-zero ~0 => tokens are being ignored)

Env: E4_CACHE_DIR, E4_PROJECTOR_CKPT, E4_LLM_MODEL_ID, E4_PROJECTOR_OUT_DIM.
Usage: python student_training/scripts/e4_stageC_diagnose.py --config configs/e4_stageC.yaml
"""

import argparse
import math
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from student_training.data.stageb_bridge_dataset import StageBBridgeDataset


def resolve(cfg):
    cache = os.environ.get("E4_CACHE_DIR", cfg["data"]["cache_dir"])
    model_id = os.environ.get("E4_LLM_MODEL_ID", cfg["llm"]["model_id"])
    if os.environ.get("E4_PROJECTOR_OUT_DIM"):
        cfg["projector"]["out_dim"] = int(os.environ["E4_PROJECTOR_OUT_DIM"])
    proj = os.environ.get("E4_PROJECTOR_CKPT", cfg["projector"].get("ckpt"))
    return cache, model_id, proj


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config if os.path.isabs(args.config) else PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    cache_dir, model_id, proj_ckpt = resolve(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = torch.bfloat16 if "bfloat16" in cfg.get("torch_dtype", "bfloat16") else torch.float16

    from student_training.models.vjepa_reason import load_llm, build_projector, StageBBridge
    llm, tok = load_llm(model_id, dtype=amp)
    projector = build_projector(cfg)
    if proj_ckpt and Path(proj_ckpt).exists():
        projector.load_state_dict(torch.load(proj_ckpt, map_location="cpu"))
        print(f"[proj] loaded {proj_ckpt}")
    else:
        print(f"[proj] WARNING no ckpt at {proj_ckpt}; using random init")
    bridge = StageBBridge(llm, projector, freeze_llm=True).to(device).eval()

    # ── (B) template / tokenizer check ───────────────────────────────────────
    print("\n=== (B) CHAT TEMPLATE / SPECIAL TOKENS ===")
    for t in ["<|im_start|>", "<|im_end|>", "system", "assistant"]:
        ids = tok.encode(t, add_special_tokens=False)
        print(f"  encode({t!r}) -> {ids}  ({'SINGLE' if len(ids)==1 else str(len(ids))+' tokens'})")
    try:
        msgs = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "QQQ"}]
        native = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        print("  --- model's own template (apply_chat_template) ---")
        print("  " + native.replace("\n", "\\n"))
        print("  --- our hardcoded template head ---")
        print("  <|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n<|im_start|>user\\n")
    except Exception as e:
        print(f"  apply_chat_template failed: {e}")

    # ── (A) embedding-scale check ────────────────────────────────────────────
    print("\n=== (A) EMBEDDING SCALE ===")
    emb = bridge.llm.get_input_embeddings().weight.detach().float()
    emb_norm = emb.norm(dim=1)
    print(f"  native token embedding L2 norm: mean={emb_norm.mean():.3f} "
          f"std={emb_norm.std():.3f}  (dim={emb.shape[1]}, vocab={emb.shape[0]})")

    ds = StageBBridgeDataset(os.path.join(cache_dir, "cache_manifest_val.jsonl"),
                             cache_dir, tok, num_vis_tokens=cfg["projector"]["num_queries"],
                             max_seq_len=cfg["data"]["max_seq_len"], supervise_verdict=False)
    with torch.no_grad():
        v = ds[0]["vis_feats"].unsqueeze(0).to(device)
        vtok = bridge.projector(v).float()[0]            # (Q, H)
    vnorm = vtok.norm(dim=1)
    print(f"  projector visual-token L2 norm: mean={vnorm.mean():.3f} std={vnorm.std():.3f}")
    ratio = float(vnorm.mean() / emb_norm.mean())
    print(f"  RATIO vis/native = {ratio:.3f}   "
          f"({'OK (~1)' if 0.5 < ratio < 2 else 'MISMATCH -> visual tokens off-distribution'})")

    # ── (C) visual contribution: PPL real / zero / text-only ─────────────────
    print("\n=== (C) VISUAL CONTRIBUTION (val PPL) ===")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        collate_fn=StageBBridgeDataset.collate_fn)

    def ppl(ablate):
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for b in loader:
                n = int((b["labels"] != -100).sum().item())
                if n == 0:
                    continue
                with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp):
                    loss, _ = bridge(b["vis_feats"].to(device), b["input_ids"].to(device),
                                     b["attention_mask"].to(device), b["labels"].to(device),
                                     b["vis_mask"].to(device), ablate=ablate)
                tot += float(loss.item()) * n; ntok += n
        return math.exp(tot / max(ntok, 1))

    p_real = ppl("none"); p_zero = ppl("zero"); p_text = ppl("mask")
    print(f"  PPL real visual : {p_real:.3f}")
    print(f"  PPL zeroed vis  : {p_zero:.3f}   (dCE_zero proxy: {math.log(p_zero/p_real):+.4f})")
    print(f"  PPL text-only   : {p_text:.3f}   (dCE_text proxy: {math.log(p_text/p_real):+.4f})")
    print("\n  READ: if zeroed ~= real ~= text-only -> visual tokens IGNORED (bug, see A/B).")
    print("        if text-only >> real -> visual DOES help (then Phase-0 floor was the issue).")


if __name__ == "__main__":
    main()
