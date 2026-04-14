"""
train_lora.py
=============
LoRA fine-tuning of InternVL3.5-4B-Flash on teacher-distilled collision data.

Training pipeline:
  1. Load teacher_dataset_v11.jsonl → stratified 80/20 train/val split
  2. Load InternVL3.5-4B-Flash → apply LoRA + ScoreHead
  3. Combined loss per step:
       L = alpha * BCE(score_pred, target) + (1-alpha) * CE(reasoning, teacher_text)
  4. AdamW + cosine LR schedule with linear warmup
  5. Gradient accumulation (effective batch = 8)
  6. After every epoch: validate on held-out val clips (ScoreHead only, no generation)
     → log train F1, val F1, val AP to epoch_metrics.jsonl
  7. Save LoRA adapters + ScoreHead every save_steps optimizer steps
  8. At end: print per-epoch table + suggest optimal checkpoint by val F1

Overfit design (num_epochs=50):
  We intentionally train past the optimum to observe the train/val F1 gap.
  The epoch_metrics.jsonl and end-of-run table show exactly where val F1 peaks,
  which becomes the recommended checkpoint for the final E2 evaluation.

Hardware: RunPod RTX 4090 (24 GB VRAM)
  - ~14 GB for model + optimizer states + activations (bf16 + gradient checkpointing)
  - Batch size 1 per GPU step, effective batch 8 via gradient accumulation

Usage:
  python student_training/scripts/train_lora.py \\
    --jsonl       outputs/teacher_dataset_v11.jsonl \\
    --frames_root /data/train_frames256 \\
    --config      student_training/configs/train_lora.yaml \\
    [--resume]    # continue from latest checkpoint in output_dir

After training, run the SUGGESTED checkpoint through:
  python student_training/scripts/trained_eval.py \\
    --checkpoint outputs/checkpoints/e2_lora_100clips/step_XXXXXX \\
    --manifest   outputs/test_manifest_private.jsonl \\
    --frames_root /data/test_frames256 \\
    --output     outputs/trained/e2_lora_100clips_test.jsonl
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
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# ── Add project root to sys.path ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from student_training.data.collision_dataset import CollisionDataset
from student_training.models.internvl_lora import InternVLCollisionModel, load_for_training


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Stratified train/val split ────────────────────────────────────────────────

def stratified_split(dataset: CollisionDataset, val_fraction: float, seed: int):
    """
    Split dataset indices into train and val, preserving class balance.

    Returns (train_indices, val_indices).
    """
    rng = random.Random(seed)

    pos_indices = [i for i, r in enumerate(dataset.records) if r.get("target") == 1]
    neg_indices = [i for i, r in enumerate(dataset.records) if r.get("target") == 0]

    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    n_val_pos = max(1, round(len(pos_indices) * val_fraction))
    n_val_neg = max(1, round(len(neg_indices) * val_fraction))

    val_idx   = pos_indices[:n_val_pos]   + neg_indices[:n_val_neg]
    train_idx = pos_indices[n_val_pos:]   + neg_indices[n_val_neg:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return train_idx, val_idx


# ── LR Scheduler ─────────────────────────────────────────────────────────────

def get_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int, scheduler_type: str):
    from torch.optim.lr_scheduler import LambdaLR

    if scheduler_type == "cosine":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / max(1, num_warmup_steps)
            progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    else:
        def lr_lambda(step):
            if step < num_warmup_steps:
                return float(step) / max(1, num_warmup_steps)
            return max(0.0, float(num_training_steps - step) / max(1, num_training_steps - num_warmup_steps))

    return LambdaLR(optimizer, lr_lambda)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_f1(scores, targets, threshold: float = 0.5) -> float:
    """Binary F1 at a fixed probability threshold."""
    preds = [1 if s >= threshold else 0 for s in scores]
    tp = sum(p == 1 and t == 1 for p, t in zip(preds, targets))
    fp = sum(p == 1 and t == 0 for p, t in zip(preds, targets))
    fn = sum(p == 0 and t == 1 for p, t in zip(preds, targets))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_ap(scores, targets) -> float:
    """Average Precision (area under PR curve) — primary thesis metric."""
    from sklearn.metrics import average_precision_score
    if len(set(targets)) < 2:
        return 0.0
    return float(average_precision_score(targets, scores))


# ── Validation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:      InternVLCollisionModel,
    val_loader: DataLoader,
    device:     str,
    amp_dtype:  torch.dtype,
    use_amp:    bool,
) -> dict:
    """
    Run ScoreHead forward on all val clips (no text generation).

    Returns dict: {val_f1, val_ap, val_loss, n_pos, n_neg}
    """
    model.eval()

    all_scores  = []
    all_targets = []
    total_loss  = 0.0
    n_batches   = 0

    for batch in val_loader:
        pv   = batch["pixel_values"].to(device, non_blocking=True)
        ids  = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        lbl  = batch["labels"].to(device, non_blocking=True)
        stgt = batch["score_target"].to(device, non_blocking=True)
        asp  = batch["asst_start_pos"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            out = model(
                pixel_values   = pv,
                input_ids      = ids,
                attention_mask = mask,
                labels         = lbl,
                score_target   = stgt,
                asst_start_pos = asp,
            )

        if out.loss is not None:
            total_loss += out.loss.item()
            n_batches  += 1

        if out.score_pred is not None:
            all_scores.extend(out.score_pred.cpu().float().tolist())
            all_targets.extend(stgt.cpu().float().tolist())

    model.train()

    n_pos = sum(1 for t in all_targets if t >= 0.5)
    n_neg = len(all_targets) - n_pos

    return {
        "val_loss": round(total_loss / max(n_batches, 1), 5),
        "val_f1":   round(compute_f1(all_scores, all_targets), 4),
        "val_ap":   round(compute_ap(all_scores, all_targets), 4),
        "n_pos":    n_pos,
        "n_neg":    n_neg,
    }


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, step: int, epoch: int,
                    output_dir: str, cfg: dict) -> str:
    ckpt_dir = Path(output_dir) / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.model.language_model.save_pretrained(str(ckpt_dir))
    torch.save(model.score_head.state_dict(), ckpt_dir / "score_head.pt")
    torch.save({
        "step":      step,
        "epoch":     epoch,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, ckpt_dir / "training_state.pt")
    with open(ckpt_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Prune old checkpoints
    save_total_limit = cfg.get("save_total_limit", 3)
    checkpoints = sorted(
        [d for d in Path(output_dir).iterdir()
         if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    while len(checkpoints) > save_total_limit:
        import shutil
        old = checkpoints.pop(0)
        shutil.rmtree(old)

    return str(ckpt_dir)


def find_latest_checkpoint(output_dir: str):
    p = Path(output_dir)
    if not p.exists():
        return None
    ckpts = sorted(
        [d for d in p.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1])
    )
    return str(ckpts[-1]) if ckpts else None


# ── End-of-run summary ────────────────────────────────────────────────────────

def print_epoch_table(epoch_log: list):
    """Print a formatted table of per-epoch metrics and suggest the best checkpoint."""
    if not epoch_log:
        return

    print(f"\n{'='*75}")
    print(f"  Epoch metrics summary")
    print(f"{'='*75}")
    print(f"  {'Epoch':>5}  {'Step':>7}  {'Train F1':>9}  {'Val F1':>8}  {'F1 Gap':>8}  {'Val AP':>7}  {'Val Loss':>9}")
    print(f"  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*9}")

    best_val_f1   = -1.0
    best_entry    = None
    peak_train_f1 = -1.0

    for entry in epoch_log:
        gap = entry["train_f1"] - entry["val_f1"]
        peak_train_f1 = max(peak_train_f1, entry["train_f1"])

        marker = ""
        if entry["val_f1"] > best_val_f1:
            best_val_f1 = entry["val_f1"]
            best_entry  = entry
            marker = " ◀ best val"

        print(
            f"  {entry['epoch']:>5}  "
            f"{entry['step']:>7}  "
            f"{entry['train_f1']:>9.4f}  "
            f"{entry['val_f1']:>8.4f}  "
            f"{gap:>+8.4f}  "
            f"{entry['val_ap']:>7.4f}  "
            f"{entry['val_loss']:>9.5f}"
            f"{marker}"
        )

    print(f"{'='*75}")

    if best_entry:
        print(f"\n  ✓ Suggested best checkpoint:")
        print(f"    Epoch {best_entry['epoch']}  |  step_{best_entry['step']:06d}")
        print(f"    Val F1 = {best_entry['val_f1']:.4f}  |  Val AP = {best_entry['val_ap']:.4f}")
        print(f"    Train F1 at that epoch = {best_entry['train_f1']:.4f}")
        print(f"    F1 gap (overfit signal) = {best_entry['train_f1'] - best_entry['val_f1']:+.4f}")
        print(f"\n  Run with:")
        print(f"    python student_training/scripts/trained_eval.py \\")
        print(f"      --checkpoint {best_entry['ckpt_dir']} \\")
        print(f"      --manifest   outputs/test_manifest_private.jsonl \\")
        print(f"      --frames_root /data/test_frames256 \\")
        print(f"      --output     outputs/trained/e2_best_ep{best_entry['epoch']}_test.jsonl")

    print(f"{'='*75}\n")


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args, cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.get("seed", 42))

    print(f"\n{'='*65}")
    print(f"Phase 3 — LoRA Fine-tuning  (overfit-curve run)")
    print(f"  JSONL          : {args.jsonl}")
    print(f"  Frames root    : {args.frames_root}")
    print(f"  Output dir     : {cfg['output_dir']}")
    print(f"  Epochs         : {cfg['num_epochs']}  (intentional overfit)")
    print(f"  Val split      : {cfg.get('val_split', 0.2)*100:.0f}%  (stratified)")
    print(f"  Loss alpha     : {cfg['loss_alpha']}")
    print(f"  LR             : {cfg['learning_rate']}")
    print(f"  Grad accum     : {cfg['gradient_accumulation_steps']}")
    print(f"  Device         : {device}")
    print(f"{'='*65}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model, tokenizer = load_for_training(
        model_id   = cfg["model_id"],
        cfg        = cfg,
        device_map = "auto" if device == "cuda" else "cpu",
    )

    # Gradient checkpointing on LLM backbone
    # When LoRA is applied, the base model's embedding layer is frozen.
    # Gradient checkpointing requires gradients to pass through frozen inputs,
    # which PyTorch normally blocks.  enable_input_require_grads() inserts a
    # forward hook that keeps the input tensor in the autograd graph even when
    # it doesn't require_grad itself — this is mandatory for LoRA + grad-ckpt.
    lm = model.model.language_model
    if hasattr(lm, "enable_input_require_grads"):
        lm.enable_input_require_grads()

    lm_model = getattr(getattr(lm, "base_model", lm), "model", None)
    if lm_model is not None and hasattr(lm_model, "gradient_checkpointing_enable"):
        lm_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled on LLM backbone")

    # ── Dataset + split ───────────────────────────────────────────────────
    full_dataset = CollisionDataset(
        jsonl_path   = args.jsonl,
        frames_root  = args.frames_root,
        model        = model.model,
        tokenizer    = tokenizer,
        cfg          = cfg,
        skip_errors  = True,
    )

    val_fraction = cfg.get("val_split", 0.2)
    train_idx, val_idx = stratified_split(full_dataset, val_fraction, seed=cfg.get("seed", 42))

    print(
        f"Split: {len(train_idx)} train  /  {len(val_idx)} val  "
        f"(stratified {100*(1-val_fraction):.0f}/{100*val_fraction:.0f})"
    )

    num_workers = cfg.get("dataloader_num_workers", 0)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size  = 1,
        shuffle     = True,
        collate_fn  = CollisionDataset.collate_fn,
        num_workers = num_workers,
        pin_memory  = (device == "cuda"),
        drop_last   = False,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size  = 1,
        shuffle     = False,
        collate_fn  = CollisionDataset.collate_fn,
        num_workers = num_workers,
        pin_memory  = (device == "cuda"),
        drop_last   = False,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr           = cfg["learning_rate"],
        weight_decay = cfg.get("weight_decay", 0.01),
        betas        = (0.9, 0.999),
        eps          = 1e-8,
    )

    # ── LR Scheduler ──────────────────────────────────────────────────────
    grad_accum      = cfg.get("gradient_accumulation_steps", 8)
    steps_per_epoch = math.ceil(len(train_idx) / grad_accum)
    total_steps     = steps_per_epoch * cfg["num_epochs"]
    warmup_steps    = math.ceil(total_steps * cfg.get("warmup_ratio", 0.1))

    scheduler = get_scheduler(
        optimizer,
        num_warmup_steps   = warmup_steps,
        num_training_steps = total_steps,
        scheduler_type     = cfg.get("lr_scheduler", "cosine"),
    )

    print(
        f"Schedule: {len(train_idx)} train clips  ×  {cfg['num_epochs']} epochs  "
        f"→ {steps_per_epoch} steps/epoch  →  {total_steps} total steps"
    )
    print(f"  Warmup: {warmup_steps} steps")

    # ── Resume ────────────────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0

    if args.resume:
        latest = find_latest_checkpoint(cfg["output_dir"])
        if latest:
            state = torch.load(Path(latest) / "training_state.pt", map_location="cpu")
            global_step = state["step"]
            start_epoch = state["epoch"]
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            print(f"Resumed from: {latest}  (step={global_step}, epoch={start_epoch})")

    # ── Mixed precision ───────────────────────────────────────────────────
    use_amp    = (device == "cuda") and cfg.get("torch_dtype", "bfloat16") != "float32"
    amp_dtype  = torch.bfloat16 if "bfloat16" in cfg.get("torch_dtype", "") else torch.float16
    scaler     = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ── Logging setup ─────────────────────────────────────────────────────
    output_dir   = cfg["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log_steps    = cfg.get("log_steps", 10)
    save_steps   = cfg.get("save_steps", 50)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    validate_every = cfg.get("validate_every_n_epochs", 1)

    step_log_path  = Path(output_dir) / "train_log.jsonl"
    epoch_log_path = Path(output_dir) / "epoch_metrics.jsonl"

    epoch_log: list = []   # collects per-epoch dicts for the final summary table

    # ── Main training loop ────────────────────────────────────────────────
    t_start = time.time()

    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        print(f"\n── Epoch {epoch+1}/{cfg['num_epochs']} ──────────────────────────────")

        epoch_loss = epoch_lm = epoch_score = 0.0
        epoch_count  = 0
        micro_step   = 0
        # Track training predictions for per-epoch train F1
        train_scores  = []
        train_targets = []

        accum_loss = accum_lm = accum_score = accum_count = 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch+1}", unit="clip", leave=False):
            pv   = batch["pixel_values"].to(device, non_blocking=True)
            ids  = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            lbl  = batch["labels"].to(device, non_blocking=True)
            stgt = batch["score_target"].to(device, non_blocking=True)
            asp  = batch["asst_start_pos"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(
                    pixel_values   = pv,
                    input_ids      = ids,
                    attention_mask = mask,
                    labels         = lbl,
                    score_target   = stgt,
                    asst_start_pos = asp,
                )

            loss = out.loss / grad_accum
            if use_amp and amp_dtype == torch.float16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate epoch-level training metrics
            accum_loss  += out.loss.item()
            accum_lm    += (out.lm_loss.item()    if out.lm_loss    else 0.0)
            accum_score += (out.score_loss.item() if out.score_loss else 0.0)
            accum_count += 1
            epoch_loss  += out.loss.item()
            epoch_count += 1

            if out.score_pred is not None:
                train_scores.extend(out.score_pred.detach().cpu().float().tolist())
                train_targets.extend(stgt.cpu().float().tolist())

            micro_step += 1

            if micro_step % grad_accum == 0:
                if use_amp and amp_dtype == torch.float16:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                if use_amp and amp_dtype == torch.float16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg = accum_loss / max(accum_count, 1)
                    tqdm.write(
                        f"  step={global_step:5d}  loss={avg:.4f}  "
                        f"lm={accum_lm/max(accum_count,1):.4f}  "
                        f"score={accum_score/max(accum_count,1):.4f}  "
                        f"lr={lr_now:.2e}"
                    )
                    with open(step_log_path, "a") as f:
                        f.write(json.dumps({
                            "step": global_step, "epoch": epoch + 1,
                            "loss": round(avg, 5),
                            "lm_loss":    round(accum_lm    / max(accum_count, 1), 5),
                            "score_loss": round(accum_score / max(accum_count, 1), 5),
                            "lr": lr_now,
                        }) + "\n")
                    accum_loss = accum_lm = accum_score = accum_count = 0

                if global_step % save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler,
                                    global_step, epoch + 1, output_dir, cfg)

        # ── End-of-epoch validation ────────────────────────────────────────
        if (epoch + 1) % validate_every == 0:
            t_val = time.time()
            val_metrics = validate(model, val_loader, device, amp_dtype, use_amp)
            val_time    = time.time() - t_val

            train_f1 = round(compute_f1(train_scores, train_targets), 4)
            train_ap = round(compute_ap(train_scores, train_targets), 4)
            f1_gap   = round(train_f1 - val_metrics["val_f1"], 4)

            # Save checkpoint at end of each epoch (for the summary table)
            ckpt_dir = save_checkpoint(
                model, optimizer, scheduler, global_step, epoch + 1, output_dir, cfg
            )

            entry = {
                "epoch":    epoch + 1,
                "step":     global_step,
                "ckpt_dir": ckpt_dir,
                "train_f1": train_f1,
                "train_ap": train_ap,
                **val_metrics,
                "f1_gap":   f1_gap,
                "elapsed_min": round((time.time() - t_start) / 60, 1),
            }
            epoch_log.append(entry)

            with open(epoch_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

            gap_marker = f"  ← GAP GROWING (+{f1_gap:.3f})" if f1_gap > 0.05 else ""
            print(
                f"  Epoch {epoch+1:>3}  "
                f"train_f1={train_f1:.4f}  "
                f"val_f1={val_metrics['val_f1']:.4f}  "
                f"gap={f1_gap:+.4f}  "
                f"val_ap={val_metrics['val_ap']:.4f}  "
                f"val_time={val_time:.1f}s"
                f"{gap_marker}"
            )

    # ── End-of-run summary ────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print(f"\nTraining complete — {elapsed_total/60:.1f} min total")
    print_epoch_table(epoch_log)
    print(f"Full epoch log: {epoch_log_path}")
    print(f"Step log:       {step_log_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning of InternVL3.5-4B-Flash (overfit-curve run)"
    )
    parser.add_argument("--jsonl",       required=True,
                        help="Path to teacher_dataset_v11.jsonl")
    parser.add_argument("--frames_root", required=True,
                        help="Root dir with per-video frame folders")
    parser.add_argument("--config",      default="student_training/configs/train_lora.yaml",
                        help="Path to train_lora.yaml")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from latest checkpoint in output_dir")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / args.config
    if not cfg_path.exists():
        print(f"ERROR: Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    train(args, cfg)


if __name__ == "__main__":
    main()
