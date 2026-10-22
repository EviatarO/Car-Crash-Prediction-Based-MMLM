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
from r1_common import ORDERS  # noqa: E402
from r1_data import Cache, attach_aug, epoch_items, feats_for, load_cards, load_items, load_texts, make_batch, other_index, val_pairs  # noqa: E402

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


def parse_eval_at(s, epochs):
    """--eval-at "0.5,0.75,1.0,..." -> sorted fractions in (0, epochs]; ValueError on junk."""
    fr = sorted({float(x) for x in s.split(",") if x.strip()})
    if any(f <= 0 or f > epochs for f in fr):
        raise ValueError(f"--eval-at fractions must lie in (0, {epochs}]: {fr}")
    return fr


def mid_check_steps(eval_at, epoch, steps_in_epoch):
    """Optimizer steps (1-based, inside this epoch) at which a mid-epoch check fires: {step: fraction}. Whole numbers are epoch ends (handled there)."""
    out = {}
    for f in eval_at:
        if abs(f - round(f)) > 1e-9 and epoch - 1 < f < epoch:
            out[max(1, min(steps_in_epoch - 1, round((f - (epoch - 1)) * steps_in_epoch)))] = f
    return out


def decision_auc(rec):
    """Validation decision AUC of a train-log record (verdict-first: 'first'; reason-first: 'before')."""
    dec = rec.get("decision") or {}
    for k in ("first", "before"):
        if k in dec and "auc" in dec[k]:
            return dec[k]["auc"]
    return None


def select_score(rec, select, monitor):
    """Higher is better. gap = wrong-video gap, val_loss = -own-video loss, auc = decision AUC (needs the reasoning dataset)."""
    mon = rec.get(f"val_{monitor}", {})
    if select == "auc":
        a = decision_auc(rec)
        return -1e9 if a is None else a
    return mon.get("gap_wrong", -1e9) if select == "gap" else -mon.get("loss_real", 1e9)


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
    ap.add_argument("--stop-on-val-rise", type=int, default=0,
                    help="also stop when the monitored validation loss (own video) rose this many epochs in a row (0 = off)")
    ap.add_argument("--sources", default="dada,nexar",
                    help="training/validation sources; 'nexar_reasoning' = Nexar reasoning dataset only (plan 2026-10-12), then --phase 2")
    ap.add_argument("--order", default="reason_first", choices=list(ORDERS),
                    help="nexar_reasoning answer order: Run 1 reason_first, Run 2 verdict_first, facts_first (plan 2026-10-15: closed-list Facts line first)")
    ap.add_argument("--facts-cards", default=None, help="fact_cards_teacher.jsonl (r1_fact_cards.py): needed for --order facts_first")
    ap.add_argument("--drop-prealert", action="store_true", help="leave the pre-alert windows out of training and validation (plan 2026-10-15)")
    ap.add_argument("--balance-tte", action="store_true", help="positives drawn in equal shares per TTE group each epoch (plan 2026-10-15)")
    ap.add_argument("--no-extra-train", action="store_true",
                    help="plan 2026-10-19, chain A: with the union manifest (R1_REASONING_MANIFEST), leave the extra TTE windows (row field extra) out of TRAINING; they stay in validation")
    ap.add_argument("--fixed-epochs", action="store_true",
                    help="plan 2026-10-19: no early stopping; run all --epochs (every epoch is saved as epoch_XX.pt, best.pt = the epoch with the lowest validation loss)")
    ap.add_argument("--input", default="last", choices=["last", "last+attn", "l11+last", "l11+last+attn"],
                    help="merger input (plan 2026-10-13): last-layer tokens (original), + crash-head attention channel, + layer-11 tokens")
    ap.add_argument("--aug-texts", default=None, help="paraphrases.jsonl: rephrased text versions of the training windows (plan 2026-10-14)")
    ap.add_argument("--aug-shifts", action="store_true", help="also draw frame-shifted copies of each training window (source nexar_reasoning_shifted in the cache)")
    ap.add_argument("--val-texts", default=None, help="enlarged validation with these rephrased texts (default: --aug-texts; 'none' = teacher's texts only); lets a control run be judged on the same pairs")
    ap.add_argument("--val-shifts", action="store_true", help="enlarged validation with the shifted windows (default: --aug-shifts)")
    ap.add_argument("--p-teacher", type=float, default=0.4, help="probability that the teacher's own text is drawn when --aug-texts is on")
    ap.add_argument("--box-hint", action="store_true", help="option 3: append the position of the attended road user to the prompt")
    ap.add_argument("--template-cap", type=int, default=0,
                    help="option 2: training windows whose explanation opening (first 9 words) occurs more than N times are kept with prob N/count per epoch")
    ap.add_argument("--select", default="gap", choices=["gap", "val_loss", "auc"],
                    help="best.pt by the wrong-video gap (pre-registered week-1 rule), by the lowest validation loss (option 1 lesson), or by the validation decision AUC "
                         "(reasoning dataset only; plan 2026-10-20). With --eval-at the check points inside an epoch compete too")
    ap.add_argument("--eval-at", default="",
                    help="plan 2026-10-20: comma-separated fractions of an epoch (e.g. 0.5,0.75,1.25,1.5) at which validation (own-video loss + decision AUC, no gap) is scored in the middle of training; "
                         "whole numbers are the normal epoch ends. Each check is saved as check_<frac>.pt; with --select auc/val_loss it can become best.pt")
    ap.add_argument("--patience-checks", type=int, default=0,
                    help="with --eval-at: stop after this many check points (mid-epoch checks and epoch ends together) in a row without a new best (0 = use --patience in epochs, as before)")
    ap.add_argument("--checks-log-only", action="store_true",
                    help="with --eval-at: the mid-epoch checks are scored, logged and saved (check_<frac>.pt) but take no part in the selection or the patience (best.pt and early stopping follow the epoch ends as before)")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (plan 2026-10-20 R2: 0.2)")
    ap.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay of the merger in phase 2 (plan 2026-10-20 R2: 0.05)")
    ap.add_argument("--lora-weight-decay", type=float, default=0.0, help="AdamW weight decay of the LoRA matrices (default 0 = as in all earlier runs; plan 2026-10-20 R2: 0.05)")
    # ---- stage 3 (plan 2026-10-22_Plan-Stage3-Encoder-LoRA): the V-JEPA2 encoder runs online on frames and its LoRA trains with the language model's loss only
    ap.add_argument("--online-encoder", default=None, help="stage 3: starting encoder adapter dir (the A1 / crash-model LoRA); the encoder runs ONLINE on --frames (encoder only: no predictor, no crash head) and its LoRA trains")
    ap.add_argument("--frames", default=None, help="stage 3: frames dirs (r1_frames.py build), comma-separated; the rows / crash-model P come from --cache-dir's index files")
    ap.add_argument("--lr-encoder", type=float, default=5e-5)
    ap.add_argument("--enc-precision", default="fp32", choices=["fp32", "fp32_strict", "bf16"], help="stage 3: precision of the online encoder forward while training: fp32 = TF32 matmuls (default), fp32_strict = plain fp32, bf16 = autocast (measured 13 % token change: not recommended)")
    ap.add_argument("--freeze-reader", action="store_true", help="stage 3 arm E: projector + language-model LoRA frozen (and LoRA dropout off); only the encoder LoRA trains")
    ap.add_argument("--eval-start", action="store_true", help="score the starting point (epoch 0) before any step: the parity check and the baseline")
    ap.add_argument("--expect-val-loss", type=float, default=0.0, help="with --eval-start: abort unless the epoch-0 validation loss is within --expect-tol of this (the warm-start run's loss)")
    ap.add_argument("--expect-tol", type=float, default=0.01)
    ap.add_argument("--save-every", type=float, default=0.0, help="save the trained parts every this fraction of an epoch WITHOUT scoring (ck_<frac>/enc, + bridge.pt unless --freeze-reader)")
    ap.add_argument("--refine-around", default="", help="after training: score the saved ck_* points at best epoch +- these fractions (e.g. 0.25,0.5); the overall best becomes best.pt / best_enc")
    ap.add_argument("--limit-val", type=int, default=0, help="smoke tests only: score only the first N validation windows")
    ap.add_argument("--log-every", type=int, default=25, help="monitoring: gradient norms per group, clip share, memory every N steps (monitor.jsonl)")
    ap.add_argument("--p1-min-p", type=float, default=0.5,
                    help="phase 1: train/select only on DADA crash windows with A1 P(collision) >= this (0 = all; option b)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = torch.device(args.device)

    qwen, tok = load_lm(args.lm, dev)
    bridge = R1Bridge(qwen, tok, init=args.init if args.phase == 1 else "random", inp=args.input).to(dev)
    init_has_lora = bool(args.init_from) and "lora" in torch.load(args.init_from, map_location="cpu")
    if args.init_from and not init_has_lora:
        bridge.load(args.init_from)
        print(f"[init] merger from {args.init_from}")
    if args.phase == 2:
        bridge.enable_lora(r=args.lora_r, alpha=2 * args.lora_r, dropout=args.lora_dropout)
    if init_has_lora:                                              # a Phase-2 checkpoint (stage 3 warm start): the LoRA must exist before it is loaded
        bridge.load(args.init_from)
        print(f"[init] merger + language-model LoRA from {args.init_from}")
    if args.freeze_reader:                                         # arm E: the reader is a fixed function of the tokens; no dropout noise either
        for m in bridge.text.modules():
            if hasattr(m, "lora_dropout"):
                for k in list(m.lora_dropout.keys()):
                    m.lora_dropout[k] = torch.nn.Identity()
    if args.grad_ckpt and hasattr(bridge.text, "gradient_checkpointing_enable"):
        bridge.text.gradient_checkpointing_enable()
    bridge.merger.float()

    srcs = args.sources.split(",")
    enc = None
    if args.online_encoder:
        assert srcs == ["nexar_reasoning"] and args.frames, "--online-encoder needs --sources nexar_reasoning and --frames"
        from r1_frames import FramesStore, OnlineCache, OnlineEncoder
        enc = OnlineEncoder(adapter=args.online_encoder, device=args.device, trainable=True, grad_ckpt=args.grad_ckpt, precision=args.enc_precision)
        caches = {"nexar_reasoning": OnlineCache(args.cache_dir, FramesStore(args.frames), enc)}
        print(f"[encoder] online V-JEPA2 encoder from {args.online_encoder}: {sum(p.numel() for p in enc.trainable()) / 1e6:.2f}M trainable LoRA parameters; no predictor, no crash head")
    else:
        caches = {s: Cache(args.cache_dir, s, args.input) for s in srcs}
    if args.aug_shifts or args.val_shifts:
        caches["nexar_reasoning_shifted"] = Cache(args.cache_dir, "nexar_reasoning_shifted", args.input)
    reasoning = srcs == ["nexar_reasoning"]
    cards = load_cards(args.facts_cards) if args.facts_cards else None
    if args.order == "facts_first" and cards is None:
        raise SystemExit("--order facts_first needs --facts-cards")
    if reasoning:
        train = load_items("nexar_reasoning", 2, "train", caches["nexar_reasoning"], order=args.order, box_hint=args.box_hint, facts=cards, drop_prealert=args.drop_prealert,
                           drop_extra_train=args.no_extra_train)
        val = {"nexar_reasoning": load_items("nexar_reasoning", 2, args.val_split, caches["nexar_reasoning"], order=args.order, box_hint=args.box_hint, facts=cards, drop_prealert=args.drop_prealert)}
        v_texts = None if args.val_texts == "none" else (args.val_texts or args.aug_texts)   # "none": validate on the teacher's texts only
        v_shifts = args.val_shifts or args.aug_shifts
        if args.aug_texts or args.aug_shifts or v_texts or v_shifts:    # augmentation: one random (window version, text version) per original window per epoch
            n0, v0 = len(train), len(val["nexar_reasoning"])
            if args.aug_texts or args.aug_shifts:
                sh_tr = load_items("nexar_reasoning_shifted", 2, "train", caches["nexar_reasoning_shifted"], order=args.order, box_hint=args.box_hint, facts=cards, drop_prealert=args.drop_prealert) if args.aug_shifts else []
                attach_aug(train, sh_tr, load_texts(args.aug_texts) if args.aug_texts else {}, args.p_teacher)
            if v_texts or v_shifts:                                   # validation = the deterministic enlarged pairs (the same for the control run)
                sh_va = load_items("nexar_reasoning_shifted", 2, args.val_split, caches["nexar_reasoning_shifted"], order=args.order, box_hint=args.box_hint, facts=cards, drop_prealert=args.drop_prealert) if v_shifts else []
                val["nexar_reasoning"] = val_pairs(val["nexar_reasoning"], sh_va, load_texts(v_texts) if v_texts else {})
            print(f"[aug] train originals {n0}: {sum(len(i.get('variants', [0])) for i in train)} window versions, "
                  f"{sum(len(i.get('texts', [])) + 1 for i in train)} text versions; validation {v0} -> {len(val['nexar_reasoning'])} pairs")
        if args.template_cap:
            from collections import Counter
            op = lambda it: " ".join(it["gt"]["explanation"].lower().split()[:9])  # noqa: E731
            cnt = Counter(op(it) for it in train)
            for it in train:
                c = cnt[op(it)]
                if c > args.template_cap:
                    it["keep_p"] = args.template_cap / c
            print(f"[template-cap] {sum('keep_p' in it for it in train)} windows down-weighted (openings seen > {args.template_cap} times)")
        monitor = "nexar_reasoning"
    elif args.phase == 1:
        mp = args.p1_min_p if args.p1_min_p > 0 else None
        train = load_items("dada", 1, "train", caches["dada"], min_p=mp)
        val = {"dada": load_items("dada", 1, args.val_split, caches["dada"], min_p=mp),
               "nexar_zeroshot": load_items("nexar", 1, args.val_split, caches["nexar"])}
        if mp:   # windows the encoder does NOT flag: reported every epoch, never used for selection
            val["dada_lowp"] = load_items("dada", 1, args.val_split, caches["dada"], max_p=mp)
        monitor = "dada"
    else:
        train = load_items("dada", 2, "train", caches["dada"]) + load_items("nexar", 2, "train", caches["nexar"])
        val = {"dada": load_items("dada", 2, args.val_split, caches["dada"]), "nexar": load_items("nexar", 2, args.val_split, caches["nexar"])}
        monitor = "nexar" if val["nexar"] else "dada"
    if args.limit_val:
        val = {k: v[:args.limit_val] for k, v in val.items()}
    print(f"[data] phase {args.phase}: train {len(train)}  val " + str({k: len(v) for k, v in val.items()}))
    assert train, "no training windows found in the cache"

    if args.freeze_reader:                                         # arm E: only the encoder LoRA trains
        for p in bridge.merger.parameters():
            p.requires_grad = False
        for n, p in bridge.named_parameters():
            if "lora_" in n:
                p.requires_grad = False
    groups = []
    if not args.freeze_reader:
        groups.append({"name": "projector", "params": list(bridge.merger.parameters()), "lr": args.lr_merger, "weight_decay": 0.0 if args.phase == 1 else args.weight_decay})
    lora = [p for n, p in bridge.named_parameters() if p.requires_grad and "lora_" in n]
    if lora:
        groups.append({"name": "lm_lora", "params": lora, "lr": args.lr_lora, "weight_decay": args.lora_weight_decay})
    if enc is not None:
        groups.append({"name": "encoder_lora", "params": enc.trainable(), "lr": args.lr_encoder, "weight_decay": 0.0})
    assert groups, "nothing to train"
    for g in groups:
        g["base_lr"] = g["lr"]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
    all_params = [p for g in groups for p in g["params"]]
    print("[params] trainable: " + ", ".join(f"{g['name']} {sum(p.numel() for p in g['params']) / 1e6:.2f}M (lr {g['lr']:g})" for g in groups))
    # stage 3 monitoring (plan 2026-10-22): the starting weights of every group (update ratios), the 32 fixed validation windows whose encoder tokens are tracked (drift)
    w0 = {g["name"]: [p.detach().clone() for p in g["params"]] for g in groups} if enc is not None else {}
    drift = {"ids": [], "tok0": None}
    if enc is not None and reasoning:
        drift["ids"] = [it["id"] for it in val["nexar_reasoning"][:32]]
        with torch.no_grad():
            drift["tok0"] = caches["nexar_reasoning"].get(drift["ids"]).half().cpu()

    accum = max(1, args.eff_bs // args.micro_bs)
    spe = math.ceil(len(epoch_items(train, random.Random(0), args.phase == 2 or reasoning, args.balance_tte)) / args.eff_bs)
    total, warm = spe * args.epochs, max(1, int(0.03 * spe * args.epochs))
    print(f"[sched] {spe} optimizer steps/epoch x {args.epochs} epochs = {total} (warmup {warm}); eff batch {args.eff_bs} = {accum} x {args.micro_bs}")

    log = open(out / "train_log.jsonl", "a", encoding="utf-8")
    step, t0 = 0, time.time()
    prev_vl, rises = None, 0
    eval_at = parse_eval_at(args.eval_at, args.epochs) if args.eval_at else []
    use_checks = bool(eval_at) and not args.checks_log_only          # the mid-epoch checks take part in the selection and the patience
    if use_checks and args.patience_checks <= 0 and not args.fixed_epochs:
        raise SystemExit("--eval-at needs --patience-checks N (or --fixed-epochs, or --checks-log-only)")
    if args.select == "auc" and not reasoning:
        raise SystemExit("--select auc needs --sources nexar_reasoning")
    sel_name = {"gap": "wrong-video gap", "val_loss": "-val loss", "auc": "decision AUC"}[args.select]
    state = {"best": -1e9, "bad": 0, "stop": False, "n": 0}
    if eval_at:
        print(f"[checks] extra validation checks at epoch fractions {eval_at} (whole numbers = epoch ends); "
              + ("logged only (selection and patience follow the epoch ends)" if args.checks_log_only else f"stop after {args.patience_checks} checks in a row without a new best"))

    def abort(msg):
        print(f"[abort] {msg}", flush=True)
        log.write(json.dumps({"abort": msg, "step": step}) + "\n")
        log.flush()
        sys.exit(3)

    gp = {g["name"]: g["params"] for g in groups}

    def group_norm(name):
        return float(torch.sqrt(sum((p.grad.detach().float().norm() ** 2) for p in gp[name] if p.grad is not None)).item()) if any(p.grad is not None for p in gp[name]) else 0.0

    def diagnostics():
        """Stage 3 monitoring at a scored check: update-to-weight ratio per group, encoder-token drift on 32 fixed validation windows, peak memory."""
        d = {}
        if enc is None:
            return d
        with torch.no_grad():
            d["update_ratio"] = {n: float(torch.sqrt(sum(((p.detach() - q) ** 2).sum() for p, q in zip(gp[n], w0[n]))) / torch.sqrt(sum((q ** 2).sum() for q in w0[n])).clamp_min(1e-12)) for n in gp}
            if drift["tok0"] is not None:
                t = caches["nexar_reasoning"].get(drift["ids"]).half().cpu().float()
                d["token_cosine_vs_start"] = float(torch.nn.functional.cosine_similarity(t, drift["tok0"].float(), dim=-1).mean())
        d["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0.0
        return d

    def consider(rec, label, frac=None):
        """Selection bookkeeping of one check point (mid-epoch check or epoch end): new best -> best.pt; else count it. Early stop is decided here."""
        score = select_score(rec, args.select, monitor)
        state["n"] += 1
        if score > state["best"] + 1e-6 or state["n"] == 1:
            state["best"], state["bad"], state["best_frac"] = score, 0, frac
            bridge.save(out / "best.pt")
            if enc is not None:
                enc.save_adapter(out / "best_enc")
            print(f"  new best at {label} ({monitor} {sel_name} {score:+.4f}) -> best.pt" + (" + best_enc/" if enc is not None else ""), flush=True)
        else:
            state["bad"] += 1
            limit = args.patience_checks if use_checks else args.patience
            if state["bad"] >= limit and not args.fixed_epochs:
                print(f"[early stop] no improvement of the {monitor} {sel_name} for {limit} {'checks' if use_checks else 'epochs'} (last {label})", flush=True)
                state["stop"] = True
        return score

    def check_memo():
        """During one check the encoder is fixed: each validation window is encoded once (OnlineCache memo), not once per evaluation pass; cleared on exit."""
        import contextlib
        return caches["nexar_reasoning"].memo() if enc is not None else contextlib.nullcontext()

    def score_rec(frac, train_loss, extra=None):
        """One scored check point: own-video loss + decision AUC (no wrong/blank gaps: they cost 3x) + stage-3 diagnostics; written to train_log.jsonl."""
        rec = {"epoch": frac, "mid": True, "step": step, "train_loss": train_loss, **(extra or {})}
        with check_memo():
            rec[f"val_{monitor}"] = evaluate(bridge, val[monitor], caches, dev, modes=("real",))
            if reasoning:
                from r1_reasoning import epoch_decision_report
                rec["decision"] = epoch_decision_report(bridge, val["nexar_reasoning"], caches, dev, args.order)
            if enc is not None:
                rec["diag"] = diagnostics()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        a = decision_auc(rec)
        print(f"[check {frac:.2f} epochs] step {step} train {rec['train_loss']:.4f} val loss {rec[f'val_{monitor}'].get('loss_real', float('nan')):.3f}"
              + (f" decision AUC {a:.4f}" if a is not None else "") + (f" diag {json.dumps(rec['diag'])}" if enc is not None else ""), flush=True)
        return rec

    def mid_check(frac, run, n_run):
        """Validation in the middle of an epoch (and the epoch-0 check of --eval-start), saved as check_<frac>.pt (not with the online encoder: ck_* dirs hold those)."""
        rec = score_rec(frac, run / max(1, n_run))
        if enc is None:
            bridge.save(out / f"check_{frac:.2f}.pt")
        if not args.checks_log_only:
            consider(rec, f"{frac:.2f} epochs", frac)
        bridge.train()
        return rec

    def save_ck(frac):
        """The trained parts at this point, NOT scored: the encoder adapter (+ projector / LM LoRA unless they are frozen)."""
        d = out / f"ck_{frac:.2f}"
        d.mkdir(parents=True, exist_ok=True)
        enc.save_adapter(d / "enc")
        if not args.freeze_reader:
            bridge.save(d / "bridge.pt")

    save_fracs = [round(j * args.save_every, 4) for j in range(1, int(args.epochs / args.save_every) + 1) if abs(j * args.save_every - round(j * args.save_every)) > 1e-9] if (args.save_every > 0 and enc is not None) else []
    clip_hist, gn_hist, tiny = [], [], 0
    mon_f = open(out / "monitor.jsonl", "a", encoding="utf-8") if enc is not None else None
    if args.eval_start:                                            # epoch 0: the warm start itself = the parity check and the baseline
        rec0 = mid_check(0.0, 0.0, 0)
        v0 = rec0[f"val_{monitor}"].get("loss_real")
        if args.expect_val_loss and (v0 is None or abs(v0 - args.expect_val_loss) > args.expect_tol):
            abort(f"epoch-0 validation loss {v0} differs from the warm-start run's {args.expect_val_loss} by more than {args.expect_tol}: the online encoder / the checkpoint do not reproduce the warm start")
        if enc is not None and args.save_every > 0:
            save_ck(0.0)

    for ep in range(1, args.epochs + 1):
        items = epoch_items(train, rng, balance=args.phase == 2 or reasoning, tte_balance=args.balance_tte)
        bridge.train()
        run, n_run = 0.0, 0
        mids = mid_check_steps(eval_at, ep, math.ceil(len(items) / args.eff_bs)) if eval_at else {}
        saves = mid_check_steps(save_fracs, ep, math.ceil(len(items) / args.eff_bs)) if save_fracs else {}
        local = 0
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
            if enc is not None:                                    # stage 3 monitoring BEFORE clipping (plan 2026-10-22)
                egn = group_norm("encoder_lora")
                tiny = tiny + 1 if egn < 1e-8 else 0
                if tiny >= 50:
                    abort("the encoder-LoRA gradient norm stayed below 1e-8 for 50 steps: the gradient does not reach the encoder")
                if step % args.log_every == 0 or step == 0:
                    rec_m = {"step": step + 1, "epoch_frac": round(ep - 1 + (local + 1) / math.ceil(len(items) / args.eff_bs), 3), "loss": run / max(1, n_run),
                             "grad_norm": {n: group_norm(n) for n in gp}, "clip_share_last200": (sum(clip_hist) / len(clip_hist)) if clip_hist else 0.0,
                             "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else 0.0, "min": round((time.time() - t0) / 60, 1)}
                    mon_f.write(json.dumps(rec_m) + "\n")
                    mon_f.flush()
            gn = torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            if not math.isfinite(float(gn)) or not math.isfinite(run):
                abort(f"non-finite gradient norm ({float(gn)}) or loss ({run}) at step {step + 1}")
            clip_hist = (clip_hist + [float(gn) > 1.0])[-200:]
            gn_hist = (gn_hist + [float(gn)])[-200:]
            if len(clip_hist) == 200 and sum(clip_hist) / 200 > 0.9 and sorted(gn_hist)[100] > 10.0:      # clipping alone is normal with Adam (past runs: 90th percentile ~1.0); abort only on a sustained explosion
                abort(f"gradient norm above 10 (median {sorted(gn_hist)[100]:.1f}) with clipping on more than 90 % of the last 200 steps: unstable training")
            opt.step()
            step += 1
            local += 1
            if local in saves:
                save_ck(saves[local])
            if local in mids:
                mid_check(mids[local], run, n_run)
                if state["stop"]:
                    break
            if step % 25 == 0 or step == 1:
                print(f"  ep{ep} step {step}/{total} loss {run / max(1, n_run):.4f} grad-norm {float(gn):.2f} "
                      f"lr {opt.param_groups[0]['lr']:.2e} {(time.time() - t0) / 60:.1f}min "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.1f}GB", flush=True)
            if args.max_steps and step >= args.max_steps:
                break
        if state["stop"]:
            break
        rec = {"epoch": ep, "step": step, "train_loss": run / max(1, n_run)}
        with check_memo():
            for k, v in val.items():
                rec[f"val_{k}"] = evaluate(bridge, v, caches, dev)
            if reasoning:            # decision AUC read at the label "Collision:" (r1_reasoning.py), no generation needed
                from r1_reasoning import epoch_decision_report
                rec["decision"] = epoch_decision_report(bridge, val["nexar_reasoning"], caches, dev, args.order)
                print("  [decision] " + json.dumps(rec["decision"]), flush=True)
            if enc is not None:
                rec["diag"] = diagnostics()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if enc is None:
            bridge.save(out / f"epoch_{ep:02d}.pt")
        elif args.save_every > 0:
            save_ck(float(ep))
        mon = rec.get(f"val_{monitor}", {})
        print(f"[epoch {ep}] train {rec['train_loss']:.4f} | " + " | ".join(
            f"{k}: real {r.get('loss_real', float('nan')):.3f} blank-gap {r.get('gap_blank', float('nan')):+.3f} "
            f"wrong-gap {r.get('gap_wrong', float('nan')):+.3f} {r.get('gap_wrong_ci', '')}" for k, r in
            ((k, rec[f'val_{k}']) for k in val)), flush=True)
        vl = mon.get("loss_real")
        if vl is not None and prev_vl is not None:
            rises = rises + 1 if vl > prev_vl else 0
        prev_vl = vl
        consider(rec, f"epoch {ep}", float(ep))
        if state["stop"]:
            break
        if args.stop_on_val_rise and rises >= args.stop_on_val_rise and not args.fixed_epochs:
            print(f"[early stop] {monitor} validation loss rose {rises} epochs in a row (best.pt = best wrong-video gap so far)")
            break
        if args.max_steps and step >= args.max_steps:
            break
    if args.refine_around and enc is not None and state.get("best_frac") is not None:
        # plan 2026-10-22: after training, score the saved (unscored) check points around the winning epoch: best +- the given fractions
        bf, deltas = state["best_frac"], [float(x) for x in args.refine_around.split(",") if x.strip()]
        cand = sorted({round(bf + sgn * d, 4) for d in deltas for sgn in (-1, 1)})
        print(f"[refine] best so far at {bf} epochs; scoring the saved points {cand}", flush=True)
        for f in cand:
            ck = out / f"ck_{f:.2f}"
            if f <= 0 or abs(f - round(f)) < 1e-9 or not (ck / "enc").exists():
                continue                                           # epoch ends were scored in the loop; missing points were never reached
            enc.load_adapter(ck / "enc")
            if (ck / "bridge.pt").exists():
                bridge.load(ck / "bridge.pt")
            bridge.eval()
            rec_r = score_rec(f, float("nan"), {"refine": True})
            consider(rec_r, f"refine {f:.2f} epochs", f)
    log.close()
    print(f"[done] {out}  best {monitor} {sel_name} {state['best']:+.4f}")


if __name__ == "__main__":
    main()
