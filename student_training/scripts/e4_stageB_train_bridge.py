"""
e4_stageB_train_bridge.py
=========================
Stage B, step 2: train ONLY the projector (LoRA OFF) on cached V-JEPA2 features.
Loss = reasoning cross-entropy over the reason span (plan §3.1). Held-out val
perplexity drives early-stopping; the best projector is saved for the gate eval.

Reuses train_lora.py patterns: cosine+warmup schedule, grad-accum, epoch logging.

Local validation (no GPU/model):
  ... --config student_training/configs/e4_stageB.yaml --dry_run
Full run on RunPod: drop --dry_run (see RUNPOD_E4_STAGEB.txt).
"""

import argparse
import json
import math
import os
import random
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


def resolve_dirs(cfg):
    """Cache/output dirs, env-overridable (RunPod points these off /workspace)."""
    cache_dir = os.environ.get("E4_CACHE_DIR", cfg["data"]["cache_dir"])
    out_dir = os.environ.get("E4_OUTPUT_DIR", cfg["data"]["output_dir"])
    return cache_dir, out_dir


def dry_run(cfg):
    """No GPU: build datasets from the cache manifests, check shapes/counts."""
    cache_dir = Path(resolve_dirs(cfg)[0])
    print("\n=== DRY RUN (Stage B train) — no model ===")
    for split in ("train", "val"):
        man = cache_dir / f"cache_manifest_{split}.jsonl"
        if not man.exists():
            print(f"  [{split}] MISSING manifest: {man}  (run cache step first)")
            continue
        rows = [json.loads(l) for l in open(man, encoding="utf-8") if l.strip()]
        from collections import Counter
        print(f"  [{split}] {len(rows)} windows  "
              f"target={dict(Counter(r['target'] for r in rows))}  "
              f"horizons={dict(Counter(r['horizon_label'] for r in rows))}")
        # spot-check one cached tensor if present
        if rows:
            import torch
            p = cache_dir / f"{rows[0]['key']}.pt"
            if p.exists():
                t = torch.load(p)
                print(f"        sample feature {rows[0]['key']}: shape={tuple(t.shape)} dtype={t.dtype}")
            else:
                print(f"        (cached tensor {p.name} not present locally — expected on pod)")
    print("=== dry run OK ===\n")


def evaluate_ppl(bridge, loader, device, amp_dtype):
    """Token-weighted perplexity over the reason span (ablate='none')."""
    import torch
    bridge.projector.eval()
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
    bridge.projector.train()
    mean_ce = tot_loss / max(tot_tok, 1)
    return math.exp(mean_ce), mean_ce


def train(args, cfg):
    import torch
    from torch.utils.data import DataLoader
    from student_training.models.vjepa_reason import load_llm, build_projector, StageBBridge

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.get("seed", 42))
    tcfg = cfg["train"]
    cache_dir, out_dir = resolve_dirs(cfg)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    amp_dtype = torch.bfloat16 if "bfloat16" in cfg.get("torch_dtype", "bfloat16") else torch.float16

    # Env overrides let Phase 0 (Stage C) re-fit the projector to another LLM
    # (e.g. Qwen3.5-4B) without editing this config.
    model_id = os.environ.get("E4_LLM_MODEL_ID", cfg["llm"]["model_id"])
    if os.environ.get("E4_PROJECTOR_OUT_DIM"):
        cfg["projector"]["out_dim"] = int(os.environ["E4_PROJECTOR_OUT_DIM"])
    print(f"Loading LLM: {model_id}")
    llm, tok = load_llm(model_id, dtype=amp_dtype)
    projector = build_projector(cfg)
    bridge = StageBBridge(llm, projector).to(device)
    if tcfg.get("gradient_checkpointing", True) and hasattr(llm, "gradient_checkpointing_enable"):
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(llm, "enable_input_require_grads"):
            llm.enable_input_require_grads()

    n_proj = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    print(f"Projector ({cfg['projector']['variant']}) trainable params: {n_proj/1e6:.2f}M")

    train_ds = StageBBridgeDataset(
        os.path.join(cache_dir, "cache_manifest_train.jsonl"), cache_dir, tok,
        num_vis_tokens=cfg["projector"]["num_queries"], max_seq_len=cfg["data"]["max_seq_len"])
    val_ds = StageBBridgeDataset(
        os.path.join(cache_dir, "cache_manifest_val.jsonl"), cache_dir, tok,
        num_vis_tokens=cfg["projector"]["num_queries"], max_seq_len=cfg["data"]["max_seq_len"])

    bs = tcfg.get("batch_size", 2)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              collate_fn=StageBBridgeDataset.collate_fn,
                              num_workers=cfg["data"].get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            collate_fn=StageBBridgeDataset.collate_fn,
                            num_workers=cfg["data"].get("num_workers", 0))

    optim = torch.optim.AdamW(projector.parameters(), lr=tcfg["learning_rate"],
                              weight_decay=tcfg.get("weight_decay", 0.01))
    grad_accum = tcfg.get("gradient_accumulation_steps", 4)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = steps_per_epoch * tcfg["num_epochs"]
    warmup = math.ceil(total_steps * tcfg.get("warmup_ratio", 0.1))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, get_cosine_lambda(warmup, total_steps))

    print(f"Schedule: {len(train_ds)} windows x {tcfg['num_epochs']} ep "
          f"-> {steps_per_epoch} steps/ep -> {total_steps} steps (warmup {warmup})")

    # Baseline PPL with the untrained projector (visual-noise floor) for context.
    base_ppl, _ = evaluate_ppl(bridge, val_loader, device, amp_dtype)
    print(f"Val PPL @ init (random projector): {base_ppl:.3f}")

    step_log = open(out_dir / "train_log.jsonl", "a")
    epoch_log_path = out_dir / "epoch_metrics.jsonl"
    best_ppl, best_epoch, patience = float("inf"), -1, tcfg.get("early_stop_patience", 6)
    bad = 0
    t0 = time.time()
    gstep = 0

    for epoch in range(tcfg["num_epochs"]):
        bridge.projector.train()
        optim.zero_grad()
        run_loss, run_n = 0.0, 0
        for i, b in enumerate(train_loader):
            with torch.cuda.amp.autocast(enabled=(device == "cuda"), dtype=amp_dtype):
                loss, _ = bridge(
                    b["vis_feats"].to(device), b["input_ids"].to(device),
                    b["attention_mask"].to(device), b["labels"].to(device),
                    b["vis_mask"].to(device), ablate="none")
            (loss / grad_accum).backward()
            run_loss += float(loss.item()); run_n += 1
            if (i + 1) % grad_accum == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(projector.parameters(), tcfg.get("max_grad_norm", 1.0))
                optim.step(); sched.step(); optim.zero_grad(); gstep += 1

        val_ppl, val_ce = evaluate_ppl(bridge, val_loader, device, amp_dtype)
        tr_loss = run_loss / max(run_n, 1)
        entry = {"epoch": epoch + 1, "step": gstep, "train_loss": round(tr_loss, 5),
                 "val_ce": round(val_ce, 5), "val_ppl": round(val_ppl, 4),
                 "lr": sched.get_last_lr()[0], "elapsed_min": round((time.time() - t0) / 60, 1)}
        with open(epoch_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        step_log.write(json.dumps(entry) + "\n"); step_log.flush()
        marker = ""
        if val_ppl < best_ppl - 1e-4:
            best_ppl, best_epoch, bad = val_ppl, epoch + 1, 0
            torch.save(projector.state_dict(), out_dir / "projector_best.pt")
            marker = "  <- best"
        else:
            bad += 1
        print(f"  ep {epoch+1:>3}  train_loss={tr_loss:.4f}  val_ppl={val_ppl:.3f}{marker}")
        if bad >= patience:
            print(f"  early stop (no val_ppl improvement in {patience} epochs)")
            break

    step_log.close()
    torch.save(projector.state_dict(), out_dir / "projector_last.pt")
    summary = {"base_ppl_random_proj": round(base_ppl, 4), "best_val_ppl": round(best_ppl, 4),
               "best_epoch": best_epoch, "projector_variant": cfg["projector"]["variant"],
               "num_queries": cfg["projector"]["num_queries"]}
    with open(out_dir / "train_result.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. best_val_ppl={best_ppl:.3f} @ epoch {best_epoch} "
          f"(random-proj floor {base_ppl:.3f}). Saved projector_best.pt")


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
