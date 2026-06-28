"""
e4_stageC_train_sft.py
======================
Stage C, step 2: reasoning SFT — train LoRA adapters on the LLM so it produces
clip-specific reasoning and its OWN verdict. Score path + projector FROZEN.

Loss = teacher-forced CE over the FULL assistant JSON (verdict + reason); the
headline prediction stays vision-only (frozen BADAS score). Selection: keep the
top-k checkpoints by held-out (18-val) reasoning CE — NOT vision AP, which is
frozen/flat in Stage C.

Reuses the Stage-B bridge (StageBBridge with freeze_llm=False) and dataset
(StageBBridgeDataset with supervise_verdict=True). The V-JEPA2 features are
cached by the Stage-B caching script and reused as-is (LLM-independent).

Local validation (no GPU/model):
  ... --config student_training/configs/e4_stageC.yaml --dry_run
Full run on RunPod: drop --dry_run (see RUNPOD_E4_STAGEC.txt).

Env overrides (RunPod):
  E4_CACHE_DIR / E4_OUTPUT_DIR     cache + output dirs (off the /workspace quota)
  E4_PROJECTOR_CKPT                path to the Phase-0 projector (projector_q35_best.pt)
  E4_LLM_MODEL_ID                  override llm.model_id (e.g. fallback to Qwen3-4B)
  E4_PROJECTOR_OUT_DIM             override projector.out_dim (= LLM hidden_size)
"""

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from student_training.data.stageb_bridge_dataset import StageBBridgeDataset


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_cosine_lambda(warmup, total):
    def f(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * prog)))
    return f


def resolve_cfg(cfg):
    """Apply env overrides; return (cache_dir, out_dir, model_id, proj_ckpt)."""
    cache_dir = os.environ.get("E4_CACHE_DIR", cfg["data"]["cache_dir"])
    out_dir = os.environ.get("E4_OUTPUT_DIR", cfg["data"]["output_dir"])
    model_id = os.environ.get("E4_LLM_MODEL_ID", cfg["llm"]["model_id"])
    if os.environ.get("E4_PROJECTOR_OUT_DIM"):
        cfg["projector"]["out_dim"] = int(os.environ["E4_PROJECTOR_OUT_DIM"])
    proj_ckpt = os.environ.get("E4_PROJECTOR_CKPT", cfg["projector"].get("ckpt"))
    return cache_dir, out_dir, model_id, proj_ckpt


def dry_run(cfg):
    """No GPU: build datasets from the cache manifests, check shapes/counts and
    that supervise_verdict supervises the verdict token(s)."""
    cache_dir = resolve_cfg(cfg)[0]
    print("\n=== DRY RUN (Stage C train) — no model ===")
    cache_dir = Path(cache_dir)
    for split in ("train", "val"):
        man = cache_dir / f"cache_manifest_{split}.jsonl"
        if not man.exists():
            print(f"  [{split}] MISSING manifest: {man}  (run Stage-B cache step first)")
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
        from collections import Counter
        print(f"  [{split}] {len(rows)} windows  "
              f"target={dict(Counter(r['target'] for r in rows))}  "
              f"horizons={dict(Counter(r['horizon_label'] for r in rows))}")
    print(f"  supervise_verdict = {cfg['data'].get('supervise_verdict', True)}  "
          f"(Stage C should be True)")
    print(f"  lora = r{cfg['lora']['r']}/a{cfg['lora']['alpha']} "
          f"targets={cfg['lora']['target_modules']}")
    print("=== dry run OK ===\n")


def build_bridge(cfg, model_id, proj_ckpt, device, amp_dtype):
    import torch
    from peft import LoraConfig, get_peft_model
    from student_training.models.vjepa_reason import load_llm, build_projector, StageBBridge

    print(f"Loading LLM: {model_id}")
    llm, tok = load_llm(model_id, dtype=amp_dtype)

    lc = cfg["lora"]
    lconf = LoraConfig(
        r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc.get("dropout", 0.05),
        target_modules=lc["target_modules"], bias="none", task_type="CAUSAL_LM")
    llm = get_peft_model(llm, lconf)            # base frozen, LoRA trainable

    projector = build_projector(cfg)
    if not proj_ckpt or not Path(proj_ckpt).exists():
        raise FileNotFoundError(
            f"Stage-C needs the Phase-0 projector. Set E4_PROJECTOR_CKPT or "
            f"projector.ckpt. Got: {proj_ckpt}")
    sd = torch.load(proj_ckpt, map_location="cpu")
    projector.load_state_dict(sd)
    print(f"Loaded projector from {proj_ckpt}")

    freeze_proj = cfg["projector"].get("freeze", True)
    if freeze_proj:
        for p in projector.parameters():
            p.requires_grad = False
        print("Projector: FROZEN")
    else:
        print(f"Projector: CO-TRAIN at lr*{cfg['projector'].get('co_train_lr_scale', 0.2)}")

    bridge = StageBBridge(llm, projector, freeze_llm=False,
                          match_embed_norm=cfg["projector"].get("match_embed_norm")).to(device)
    if cfg["train"].get("gradient_checkpointing", True) and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()

    n_train = sum(p.numel() for p in bridge.parameters() if p.requires_grad)
    print(f"Trainable params (LoRA{'+proj' if not freeze_proj else ''}): {n_train/1e6:.2f}M")
    return bridge, tok, freeze_proj


def evaluate_ce(bridge, loader, device, amp_dtype):
    """Token-weighted CE/PPL over the supervised span (verdict+reason)."""
    import torch
    bridge.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for b in loader:
            n_tok = int((b["labels"] != -100).sum().item())
            if n_tok == 0:
                continue
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                loss, _ = bridge(
                    b["vis_feats"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["labels"].to(device),
                    b["vis_mask"].to(device), ablate="none")
            tot_loss += float(loss.item()) * n_tok
            tot_tok += n_tok
    mean_ce = tot_loss / max(tot_tok, 1)
    return math.exp(mean_ce), mean_ce


def save_ckpt(bridge, out_dir, epoch, freeze_proj):
    import torch
    d = Path(out_dir) / f"ckpt_ep{epoch:02d}"
    d.mkdir(parents=True, exist_ok=True)
    bridge.llm.save_pretrained(str(d))                 # LoRA adapter
    if not freeze_proj:
        torch.save(bridge.projector.state_dict(), d / "projector.pt")
    return d


def train(args, cfg):
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.get("seed", 42))
    tcfg = cfg["train"]
    cache_dir, out_dir, model_id, proj_ckpt = resolve_cfg(cfg)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    amp_dtype = torch.bfloat16 if "bfloat16" in cfg.get("torch_dtype", "bfloat16") else torch.float16
    keep_k = tcfg.get("keep_top_k", 3)

    bridge, tok, freeze_proj = build_bridge(cfg, model_id, proj_ckpt, device, amp_dtype)

    sv = cfg["data"].get("supervise_verdict", True)
    train_ds = StageBBridgeDataset(
        os.path.join(cache_dir, "cache_manifest_train.jsonl"), cache_dir, tok,
        num_vis_tokens=cfg["projector"]["num_queries"],
        max_seq_len=cfg["data"]["max_seq_len"], supervise_verdict=sv)
    val_ds = StageBBridgeDataset(
        os.path.join(cache_dir, "cache_manifest_val.jsonl"), cache_dir, tok,
        num_vis_tokens=cfg["projector"]["num_queries"],
        max_seq_len=cfg["data"]["max_seq_len"], supervise_verdict=sv)

    bs = tcfg.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=StageBBridgeDataset.collate_fn,
                              num_workers=cfg["data"].get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=StageBBridgeDataset.collate_fn,
                            num_workers=cfg["data"].get("num_workers", 0))

    # Score-consistency anchor: distill the frozen BADAS score into the LLM's
    # verdict probability (BCE of p_yes @ the verdict token vs the vision score).
    # This re-introduces the regularizing, visually-grounded signal that let the
    # InternVL student train past epoch 1 (pure reasoning-CE overfits immediately).
    acfg = cfg.get("anchor", {})
    anchor_on = bool(acfg.get("enabled", False)) and sv
    anchor_w = float(acfg.get("weight", 1.0))
    yes_id = tok.encode(acfg.get("yes_str", "YES"), add_special_tokens=False)[0]
    no_id = tok.encode(acfg.get("no_str", "NO"), add_special_tokens=False)[0]
    if anchor_on:
        print(f"Score anchor ON: + {anchor_w} * BCE(p_yes@verdict, BADAS score)  "
              f"[yes_id={yes_id} no_id={no_id}]")

    # Optimizer: LoRA params at lr; projector (if co-trained) at a lower lr.
    base_lr = tcfg["learning_rate"]
    if freeze_proj:
        params = [{"params": [p for p in bridge.parameters() if p.requires_grad], "lr": base_lr}]
    else:
        lora_p = [p for n, p in bridge.named_parameters() if p.requires_grad and "projector" not in n]
        proj_p = [p for p in bridge.projector.parameters() if p.requires_grad]
        params = [{"params": lora_p, "lr": base_lr},
                  {"params": proj_p, "lr": base_lr * cfg["projector"].get("co_train_lr_scale", 0.2)}]
    optim = torch.optim.AdamW(params, weight_decay=tcfg.get("weight_decay", 0.01))

    grad_accum = tcfg.get("gradient_accumulation_steps", 4)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * tcfg["num_epochs"]
    warmup = math.ceil(total_steps * tcfg.get("warmup_ratio", 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, get_cosine_lambda(warmup, total_steps))
    print(f"Schedule: {len(train_ds)} windows x {tcfg['num_epochs']} ep "
          f"-> {steps_per_epoch} steps/ep -> {total_steps} steps (warmup {warmup})")

    base_ppl, base_ce = evaluate_ce(bridge, val_loader, device, amp_dtype)
    print(f"Val CE/PPL @ init (LoRA=0): {base_ce:.4f} / {base_ppl:.3f}")

    epoch_log = out_dir / "epoch_metrics.jsonl"
    saved = {}                       # epoch -> (val_ce, dir)
    best_ce, best_epoch, bad = float("inf"), -1, 0
    patience = tcfg.get("early_stop_patience", 5)
    t0 = time.time(); gstep = 0

    for epoch in range(1, tcfg["num_epochs"] + 1):
        bridge.train()
        optim.zero_grad()
        run_loss, run_n, run_anchor = 0.0, 0, 0.0
        for i, b in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                loss_ce, logits = bridge(
                    b["vis_feats"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["labels"].to(device),
                    b["vis_mask"].to(device), ablate="none")
                loss = loss_ce
                loss_anchor = None
                if anchor_on:
                    vpos = b["verdict_pos"].to(device)               # (B,)
                    idx = (vpos - 1).clamp(min=0)                    # logit predicting verdict
                    sel = logits[torch.arange(logits.size(0), device=device), idx]  # (B, V)
                    pair = sel[:, [no_id, yes_id]].float()
                    p_yes = torch.softmax(pair, dim=-1)[:, 1].clamp(1e-4, 1 - 1e-4)
                    tgt = b["vision_score"].to(device).float().clamp(1e-4, 1 - 1e-4)
                    valid = (vpos >= 0).float()
                    bce = torch.nn.functional.binary_cross_entropy(p_yes, tgt, reduction="none")
                    loss_anchor = (bce * valid).sum() / valid.sum().clamp(min=1.0)
                    loss = loss_ce + anchor_w * loss_anchor
            (loss / grad_accum).backward()
            run_loss += float(loss_ce.item()); run_n += 1
            if loss_anchor is not None:
                run_anchor += float(loss_anchor.item())
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in bridge.parameters() if p.requires_grad],
                    tcfg.get("max_grad_norm", 1.0))
                optim.step(); sched.step(); optim.zero_grad(); gstep += 1

        val_ppl, val_ce = evaluate_ce(bridge, val_loader, device, amp_dtype)
        tr_loss = run_loss / max(run_n, 1)
        tr_anchor = run_anchor / max(run_n, 1)
        entry = {"epoch": epoch, "step": gstep, "train_loss": round(tr_loss, 5),
                 "train_anchor": round(tr_anchor, 5),
                 "val_ce": round(val_ce, 5), "val_ppl": round(val_ppl, 4),
                 "lr": sched.get_last_lr()[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
        with open(epoch_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # checkpoint + keep top-k by val CE
        d = save_ckpt(bridge, out_dir, epoch, freeze_proj)
        saved[epoch] = (val_ce, d)
        ranked = sorted(saved.items(), key=lambda kv: kv[1][0])
        for ep, (ce, dd) in ranked[keep_k:]:
            shutil.rmtree(dd, ignore_errors=True); del saved[ep]
        in_top = epoch in saved
        marker = "  <- top3" if in_top else ""

        if val_ce < best_ce - 1e-4:
            best_ce, best_epoch, bad = val_ce, epoch, 0
            marker += " *best*"
        else:
            bad += 1
        anc = f"  anchor={tr_anchor:.4f}" if anchor_on else ""
        print(f"  ep {epoch:>3}  train_ce={tr_loss:.4f}{anc}  val_ce={val_ce:.4f}  "
              f"val_ppl={val_ppl:.3f}{marker}")
        if bad >= patience:
            print(f"  early stop (no val_ce improvement in {patience} epochs)")
            break

    top3 = sorted(saved.items(), key=lambda kv: kv[1][0])
    summary = {
        "model_id": model_id,
        "base_val_ce_lora0": round(base_ce, 5), "base_val_ppl_lora0": round(base_ppl, 4),
        "best_val_ce": round(best_ce, 5), "best_epoch": best_epoch,
        "top3": [{"epoch": ep, "val_ce": round(ce, 5), "dir": str(d.name)}
                 for ep, (ce, d) in top3],
        "lora": cfg["lora"], "projector_frozen": freeze_proj,
        "supervise_verdict": sv,
    }
    with open(out_dir / "train_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. best_val_ce={best_ce:.4f} @ ep{best_epoch}. "
          f"Top-{keep_k}: {[ep for ep,_ in top3]}. "
          f"\nSTOP — review train/val before any test run (plan gate).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    with open(args.config if os.path.isabs(args.config) else PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)
    if args.dry_run:
        dry_run(cfg)
        return
    train(args, cfg)


if __name__ == "__main__":
    main()
