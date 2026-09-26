"""
aa_head_attention_diag.py
==========================
Stage AA-H, Stage 0 (child plan 2026-09-24-AA-H-head-attention-supervision): does the crash
HEAD's own attention already favor relevant-car tokens over other-vehicle/background tokens,
on a given checkpoint? Answers two things before any pod run:
  1. Sets --aux-margin for attn_rank: measured on the CONTROL checkpoint, m = current
     ln(rho_P/rho_V) [and ln(rho_P/rho_B)] + ln(1.5) - never guessed.
  2. The Stage-0 gate: if relevant tokens ALREADY receive much more attention than others
     (large existing ratio), attn_rank/attn_mass have little room left to teach - a signal to
     weight attn_gradcam more heavily in Stage 2's screen. Comparing this same measurement
     across base BADAS-Open, A1-compress256, and any trained AA-rel/AA-occ checkpoint also
     shows whether prior training already moved this ratio (a companion to
     aa1_token_probe.py's Phase-3 relevance-readability probe, which measures a DIFFERENT
     thing - whether relevance is LINEARLY READABLE at a mid-stack layer, not what the head's
     OWN attention already does with it).

Runs LOCALLY (no LoRA training - a plain forward pass per window under torch.no_grad(), same
GPU/RAM footprint as aa1_token_probe.py's caching pass). One checkpoint per invocation,
matching aa1_token_probe.py's convention - run it once per checkpoint being compared.

Usage:
  python aa_head_attention_diag.py --config ../configs/e4_stageA.yaml --checkpoint base \
      --n-windows 200 --out ../../outputs/aa_head_attn/diag_base.json
  python aa_head_attention_diag.py --config ../configs/e4_stageA.yaml --checkpoint a1compress256 \
      --lora-adapter ../../outputs/a1_compress256/train/epoch_02/lora_adapter \
      --n-windows 200 --out ../../outputs/aa_head_attn/diag_a1compress256.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import TrainableBadasWrapper, load_training_examples, clip_level_split  # noqa: E402
from aa_head_losses import token_sets_from_labels, mean_attention_received, attention_entropy, N_REAL  # noqa: E402

CAPTIONS_1761 = REPO / "outputs" / "semantic_captions" / "Caption_Train4500_Mixed_1761.jsonl"
LABELS_DIR_DEFAULT = REPO / "dataset" / "aa_token_labels"


def load_checkpoint(stagea_cfg: dict, preprocess_mode: str, lora_adapter: str | None,
                     lora_target_modules: str):
    """Same load path as aa1_token_probe.py's load_probe_model, but with
    capture_head_attention=True instead of per-layer hooks - this script reads the HEAD's
    attention, not an intermediate layer's tokens."""
    badas = TrainableBadasWrapper(stagea_cfg, lora_target_modules=None,
                                  preprocess_mode=preprocess_mode,
                                  capture_head_attention=True)
    if lora_adapter:
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file as _load_sft
        from peft.utils import set_peft_model_state_dict as _set_peft_sd
        targets = lora_target_modules[3:] if lora_target_modules.startswith("re:") \
            else [s.strip() for s in lora_target_modules.split(",") if s.strip()]
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=targets)
        badas.nn_model = get_peft_model(badas.nn_model, cfg)
        sft = Path(lora_adapter) / "adapter_model.safetensors"
        _set_peft_sd(badas.nn_model, _load_sft(str(sft)))
        print(f"[load] LoRA adapter: {sft}")
    for p in badas.nn_model.parameters():
        p.requires_grad = False
    badas.nn_model.eval()
    return badas


def measure_window(badas, ex: dict, labels_dir: Path) -> dict | None:
    """One window -> a dict of ratios/entropy, or None if no label file / no relevant tokens
    (rel>0 nowhere - a window with no detected relevant car contributes nothing to this
    measurement, same "skip, don't zero-fill" convention as load_aux_h_arrays)."""
    label_path = labels_dir / f"{ex['frames_dir']}.npz"
    if not label_path.exists():
        return None
    with np.load(label_path) as z:
        rel = torch.from_numpy(z["rel"])
        occ = torch.from_numpy(z["occ"].astype(np.float32))
    sets = token_sets_from_labels(rel, occ)
    if sets is None:
        return None
    P, V, B = sets
    with torch.no_grad():
        badas.forward(ex["frame_paths"])
        attn = badas._captured.get("head_attn")
        if attn is None:
            raise RuntimeError("head_attn was not captured - capture_head_attention must be "
                                "True on the wrapper.")
        r = mean_attention_received(attn).cpu()
        entropy = attention_entropy(attn).item()
        r_real, r_future = r[:N_REAL], r[N_REAL:]
        rho_P = r_real[P].mean().item()
        rho_V = r_real[V].mean().item() if V.any() else float("nan")
        rho_B = r_real[B].mean().item() if B.any() else float("nan")
    return dict(
        frames_dir=ex["frames_dir"], label=ex["label"],
        n_P=int(P.sum()), n_V=int(V.sum()), n_B=int(B.sum()),
        rho_P=rho_P, rho_V=rho_V, rho_B=rho_B,
        ln_rho_PV=float(np.log(rho_P / rho_V)) if rho_V == rho_V and rho_V > 0 else None,
        ln_rho_PB=float(np.log(rho_P / rho_B)) if rho_B == rho_B and rho_B > 0 else None,
        entropy=entropy,
        future_token_share=float(r_future.sum().item()),
    )


def _nan_to_null(obj):
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_null(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_null(v) for v in obj]
    return obj


def summarize(rows: list, key: str) -> dict:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return dict(n=0, mean=None, median=None, p25=None, p75=None)
    arr = np.array(vals, dtype=np.float64)
    return dict(n=len(arr), mean=float(arr.mean()), median=float(np.median(arr)),
                p25=float(np.percentile(arr, 25)), p75=float(np.percentile(arr, 75)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preprocess", default="compress256", choices=["crop", "compress256"])
    ap.add_argument("--checkpoint", required=True,
                    help="a LABEL for the report. 'base' = raw BADAS-Open (no --lora-adapter); "
                         "any other label requires --lora-adapter.")
    ap.add_argument("--lora-adapter", default=None)
    ap.add_argument("--lora-target-modules", default="query,key,value")
    ap.add_argument("--captions-path", default=str(CAPTIONS_1761))
    ap.add_argument("--labels-dir", default=str(LABELS_DIR_DEFAULT))
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split", default="val", choices=["train", "val", "all"],
                    help="which split to measure on. 'val' (default) matches Stage 0's plan "
                         "(measure on the same val split A1-compress256/AA-rel/AA-occ were "
                         "selected on) and is disjoint from every training arm's train set.")
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.checkpoint != "base" and not args.lora_adapter:
        raise ValueError(f"--checkpoint {args.checkpoint} requires --lora-adapter")
    if args.checkpoint == "base" and args.lora_adapter:
        raise ValueError("--checkpoint base must not be given --lora-adapter")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(args.config, encoding="utf-8") as f:
        stagea_cfg = yaml.safe_load(f)

    badas = load_checkpoint(stagea_cfg, args.preprocess, args.lora_adapter,
                            args.lora_target_modules)

    examples = load_training_examples(captions_path=args.captions_path)
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac, seed=args.split_seed)
    pool = dict(train=train_ex, val=val_ex, all=examples)[args.split]
    print(f"[data] {args.split} split: {len(pool)} windows available")
    if args.n_windows and args.n_windows < len(pool):
        import random as _random
        pool = _random.Random(args.sample_seed).sample(pool, args.n_windows)
    print(f"[data] measuring {len(pool)} windows (seed={args.sample_seed})")

    rows = []
    t0 = time.time()
    n_missing = 0
    for i, ex in enumerate(pool):
        row = measure_window(badas, ex, Path(args.labels_dir))
        if row is None:
            n_missing += 1
            continue
        rows.append(row)
        if (i + 1) % 25 == 0 or i == len(pool) - 1:
            elapsed = time.time() - t0
            rate = elapsed / (i + 1)
            remaining = len(pool) - (i + 1)
            print(f"  [{i+1}/{len(pool)}] measured={len(rows)} missing/no-P={n_missing}  "
                 f"{rate:.2f}s/window  elapsed={elapsed/60:.1f}min  "
                 f"eta={remaining*rate/60:.1f}min", flush=True)

    pos_rows = [r for r in rows if r["label"] == 1]
    neg_rows = [r for r in rows if r["label"] == 0]

    def report_ratio(rows_subset, tag):
        pv = summarize(rows_subset, "ln_rho_PV")
        pb = summarize(rows_subset, "ln_rho_PB")
        ent = summarize(rows_subset, "entropy")
        fut = summarize(rows_subset, "future_token_share")
        print(f"[{tag}] n={len(rows_subset)}  "
             f"ln(rho_P/rho_V): mean={pv['mean']}  median={pv['median']}  "
             f"ln(rho_P/rho_B): mean={pb['mean']}  median={pb['median']}  "
             f"entropy: mean={ent['mean']}  future_token_share: mean={fut['mean']}")
        return dict(ln_rho_PV=pv, ln_rho_PB=pb, entropy=ent, future_token_share=fut)

    report = dict(
        checkpoint=args.checkpoint, lora_adapter=args.lora_adapter, split=args.split,
        n_windows_requested=args.n_windows, n_windows_measured=len(rows),
        n_missing_or_no_P=n_missing,
        overall=report_ratio(rows, "all"),
        positives=report_ratio(pos_rows, "pos"),
        negatives=report_ratio(neg_rows, "neg"),
        # Stage 0 gate: suggested attn_rank margin = current median ratio + ln(1.5), per the
        # plan ("m is set at Stage 0 from the measured current ratio, start: current + ln(1.5)").
        suggested_margin=dict(
            from_PV=(summarize(rows, "ln_rho_PV")["median"] + float(np.log(1.5)))
                    if summarize(rows, "ln_rho_PV")["median"] is not None else None,
            from_PB=(summarize(rows, "ln_rho_PB")["median"] + float(np.log(1.5)))
                    if summarize(rows, "ln_rho_PB")["median"] is not None else None,
        ),
        per_window=rows,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_nan_to_null(report), f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
