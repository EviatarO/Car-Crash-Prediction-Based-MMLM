"""
r1_train.py - week-1 training for the reasoning path (plan 2026-10-05_Plan-Week1-GoNoGo-rev3).

  Phase 1 (alignment): only the Merger trains (LM frozen). Data: DADA valid crash windows (event; cause).
                       Validation = DADA val valid crash (TP-only) windows.
  Phase 2 (SFT):       Merger + LoRA on the language model. Data: DADA + Nexar train (valid crash + no-crash,
                       50/50 sampler). Validation = DADA val and Nexar val.

Loss: next-token cross-entropy on the answer tokens only (mean per sample, then per batch).
Selection / early stopping: the validation WRONG-VIDEO GAP (loss with another clip's tokens - own tokens),
not the validation loss. Evaluated every epoch together with the blank-video gap.

  python r1_train.py --phase 1 --cache-dir ../../outputs/r1_week1/cache --out-dir ../../outputs/r1_week1/phase1 \
         --init random --epochs 15 --lr-merger 1e-3
  python r1_train.py --phase 2 --cache-dir ... --out-dir ../../outputs/r1_week1/phase2 \
         --init-from ../../outputs/r1_week1/phase1/best.pt --epochs 8 --lr-merger 2e-5 --lr-lora 2e-4
  (local smoke test: add --lm tiny --device cpu --max-steps 4)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "models"))
from r1_bridge import R1Bridge  # noqa: E402
from r1_data import Cache, epoch_items, feats_for, load_items, make_batch, other_index  # noqa: E402

QWEN_ID = "Qwen/Qwen3-VL-4B-Instruct"
TOK_DIR = ROOT / "dataset" / "public_samples" / "qwen3vl_cfg"


def load_lm(name, device, dtype=torch.bfloat16):
    from transformers import AutoProcessor
    src = str(TOK_DIR) if TOK_DIR.exists() else QWEN_ID
    tok = AutoProcessor.from_pretrained(src).tokenizer
    if name == "tiny":
        from r1_bridge_test import tiny_qwen
        return tiny_qwen(len(tok)).to(device).float(), tok
    from transformers import Qwen3VLForConditionalGeneration
    qwen = Qwen3VLForConditionalGeneration.from_pretrained(name, dtype=dtype).to(device).eval()
    return qwen, tok


def bootstrap_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


@torch.no_grad()
def evaluate(bridge, items, caches, device, bs=8, modes=("real", "blank", "wrong")):
    """Per-sample answer loss under real / blank / wrong-video tokens. Returns means, gaps with 95% bootstrap CI."""
    if not items:
        return {}
    bridge.eval()
    partner = other_index(items)
    per = {m: [] for m in modes}
    for s in range(0, len(items), bs):
        chunk = items[s:s + bs]
        feats = feats_for(chunk, caches).to(device)
        other = feats_for([items[partner[s + k]] for k in range(len(chunk))], caches).to(device) if "wrong" in modes else None
        batch = [t.to(device) for t in make_batch(bridge.prompts, chunk)]
        for m in modes:
            per[m].extend(bridge.loss_per_sample(feats, batch, mode=m, other=other).float().cpu().tolist())
    out = {"n": len(items)}
    for m in modes:
        out[f"loss_{m}"] = float(np.mean(per[m]))
    real = np.array(per["real"])
    for m in ("blank", "wrong"):
        if m in per:
            d = np.array(per[m]) - real
            lo, hi = bootstrap_ci(d)
            out[f"gap_{m}"] = float(d.mean())
            out[f"gap_{m}_ci"] = [lo, hi]
            out[f"gap_{m}_frac_pos"] = float((d > 0).mean())
    return out


def cosine_lr(step, total, warm, base, floor=0.1):
    if step < warm:
        return base * (step + 1) / max(1, warm)
    p = (step - warm) / max(1, total - warm)
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2])
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--lm", default=QWEN_ID, help="HF id/path, or 'tiny' (random tiny model for smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", default="random", choices=["random", "qwen"], help="merger init (phase 1)")
    ap.add_argument("--init-from", default=None, help="phase-1 checkpoint (merger) to start phase 2 from")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr-merger", type=float, default=1e-3)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--eff-bs", type=int, default=16)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--val-split", default="val", help="smoke tests only: use another split as validation")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N optimizer steps")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)

    qwen, tok = load_lm(args.lm, dev)
    bridge = R1Bridge(qwen, tok, init=args.init if args.phase == 1 else "random").to(dev)
    if args.init_from:
        bridge.load(args.init_from)
        print(f"[init] merger from {args.init_from}")
    if args.phase == 2:
        bridge.enable_lora(r=args.lora_r)
    if args.grad_ckpt and hasattr(bridge.text, "gradient_checkpointing_enable"):
        bridge.text.gradient_checkpointing_enable()
    bridge.merger.float()

    caches = {s: Cache(args.cache_dir, s) for s in ("dada", "nexar")}
    if args.phase == 1:
        train = load_items("dada", 1, "train", caches["dada"])
        val = {"dada": load_items("dada", 1, args.val_split, caches["dada"]),
               "nexar_zeroshot": load_items("nexar", 1, args.val_split, caches["nexar"])}
        monitor = "dada"
    else:
        train = load_items("dada", 2, "train", caches["dada"]) + load_items("nexar", 2, "train", caches["nexar"])
        val = {"dada": load_items("dada", 2, args.val_split, caches["dada"]), "nexar": load_items("nexar", 2, args.val_split, caches["nexar"])}
        monitor = "nexar" if val["nexar"] else "dada"
    print(f"[data] phase {args.phase}: train {len(train)}  val " + str({k: len(v) for k, v in val.items()}))
    assert train, "no training windows found in the cache"

    groups = [{"params": list(bridge.merger.parameters()), "lr": args.lr_merger, "weight_decay": 0.0 if args.phase == 1 else 0.01}]
    lora = [p for n, p in bridge.named_parameters() if p.requires_grad and "lora_" in n]
    if lora:
        groups.append({"params": lora, "lr": args.lr_lora, "weight_decay": 0.0})
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    print(f"[params] trainable: merger {sum(p.numel() for p in bridge.merger.parameters())/1e6:.1f}M"
          + (f" + LoRA {sum(p.numel() for p in lora)/1e6:.1f}M" if lora else ""))

    accum = max(1, args.eff_bs // args.micro_bs)
    spe = math.ceil(len(epoch_items(train, random.Random(0), args.phase == 2)) / args.eff_bs)
    total, warm = spe * args.epochs, max(1, int(0.03 * spe * args.epochs))
    print(f"[sched] {spe} optimizer steps/epoch x {args.epochs} epochs = {total} (warmup {warm}); eff batch {args.eff_bs} = {accum} x {args.micro_bs}")

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    best, bad, step, t0 = -1e9, 0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        items = epoch_items(train, rng, balance=args.phase == 2)
        bridge.train()
        run, n_run = 0.0, 0
        for s in range(0, len(items), args.eff_bs):
            eff = items[s:s + args.eff_bs]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, warm, g["base_lr"])
            opt.zero_grad(set_to_none=True)
            for m in range(0, len(eff), args.micro_bs):
                chunk = eff[m:m + args.micro_bs]
                feats = feats_for(chunk, caches).to(dev)
                batch = [t.to(dev) for t in make_batch(bridge.prompts, chunk)]
                batch_mean = bridge.loss_per_sample(feats, batch).mean()
                (batch_mean * (len(chunk) / len(eff))).backward()
                run += float(batch_mean.detach()) * len(chunk)
                n_run += len(chunk)
            gn = torch.nn.utils.clip_grad_norm_(bridge.trainable_params(), 1.0)
            opt.step()
            step += 1
            if step % 25 == 0 or step == 1:
                print(f"  ep{ep} step {step}/{total} loss {run / max(1, n_run):.4f} grad-norm {float(gn):.2f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / 60:.1f}min", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        rec = {"epoch": ep, "step": step, "train_loss": run / max(1, n_run)}
        for k, v in val.items():
            rec[f"val_{k}"] = evaluate(bridge, v, caches, dev)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        bridge.save(out / f"epoch_{ep:02d}.pt")
        mon = rec.get(f"val_{monitor}", {})
        score = mon.get("gap_wrong", -1e9)
        print(f"[epoch {ep}] train {rec['train_loss']:.4f} | " + " | ".join(
            f"{k}: real {r.get('loss_real', float('nan')):.3f} blank-gap {r.get('gap_blank', float('nan')):+.3f} "
            f"wrong-gap {r.get('gap_wrong', float('nan')):+.3f} {r.get('gap_wrong_ci', '')}" for k, r in
            ((k, rec[f'val_{k}']) for k in val)), flush=True)
        if score > best + 1e-6 or (ep == 1 and best == -1e9):
            best, bad = score, 0
            bridge.save(out / "best.pt")
            print(f"  new best ({monitor} wrong-video gap {score:+.4f}) -> best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[early stop] no improvement of the {monitor} wrong-video gap for {args.patience} epochs")
                break
        if args.max_steps and step >= args.max_steps:
            break
    log.close()
    print(f"[done] {out}  best {monitor} wrong-video gap {best:+.4f}")


if __name__ == "__main__":
    main()
