"""
aa1_token_probe.py
===================
Stage AA.1 gate, Phase 3 of the child plan (2026-09-19-AA-token-relevance-aux): on the FROZEN
trunk (no LoRA training here), can a linear probe read the AA.4 token-relevance labels
(aa4_token_labels.py's rel/occ) out of one encoder layer's tokens? This decides two things
before any RunPod training run is started:
  1. whether the target is trivial (occ - "is a car here") or genuinely informative (rel -
     "which car matters, how much") - by comparing each to a geometric baseline that never
     sees the ViT's tokens at all;
  2. which encoder layer to tap (--aux-layer in semsup_train.py), and the FROZEN aux head
     itself (LayerNorm+Linear, semsup_train.build_aux_head's exact architecture) - this
     script's fitted head IS --aux-head-init for the real run.

RUNS ON RUNPOD, NOT LOCALLY (needs BADAS-Open + a GPU forward pass per window; the label
files themselves are produced locally by aa4_token_labels.py first). See the child plan's
Phase 3 for the full gate description and what "pass" means.

Two checkpoints, one comparison: BASE BADAS-Open (pre-training - does the pretrained trunk
already carry this?) and A1-compress256's frozen adapter (post-crash-training - did fitting
the crash task alone already produce it, incidentally?). Same probe-fitting code either way,
selected via --lora-adapter.

Cost control (~5GB cache budget for ~1,500 windows, per the child plan): every OBJECT token
(occ>0) is kept, but only up to --bg-per-window background tokens (occ==0) are kept PER
WINDOW, sampled uniformly at random (fixed seed) - not all ~1,500-1,800 background tokens a
window actually has.

Usage:
  python aa1_token_probe.py --config ../configs/e4_stageA.yaml --preprocess compress256 \
      --layers 11,17,23 --checkpoint base --out-dir ../../outputs/aa1_token_probe/base
  python aa1_token_probe.py --config ../configs/e4_stageA.yaml --preprocess compress256 \
      --layers 11,17,23 --checkpoint a1compress256 \
      --lora-adapter ../../outputs/a1_compress256/train/epoch_02/lora_adapter \
      --out-dir ../../outputs/aa1_token_probe/a1compress256
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import TrainableBadasWrapper, load_training_examples, clip_level_split  # noqa: E402
from semsup_train import build_aux_head, balanced_bce_loss  # noqa: E402

CAPTIONS_1761 = REPO / "outputs" / "semantic_captions" / "Caption_Train4500_Mixed_1761.jsonl"
LABELS_DIR_DEFAULT = REPO / "dataset" / "aa_token_labels"


# =============================================================================
# Model: frozen trunk, hooks on several layers at once (semsup_common's wrapper only
# supports one aux_layer - this script needs several in a single forward pass)
# =============================================================================

def load_probe_model(stagea_cfg: dict, preprocess_mode: str, layers: list,
                     lora_adapter: str | None, lora_target_modules: str | None):
    """Returns (badas, captured) where `captured` is a dict updated IN PLACE on every
    forward: {layer_idx: (2048, 1024) tensor}. Entirely frozen - no LoRA training, no
    aux_layer hook (that single-layer mechanism is for semsup_train.py's actual training
    run; this script needs several layers at once, so it registers its own hooks directly)."""
    badas = TrainableBadasWrapper(stagea_cfg, lora_target_modules=None,
                                  preprocess_mode=preprocess_mode)
    if lora_adapter:
        # Same load path as semsup_train.py's --lora-init, but applied to an OTHERWISE-frozen
        # wrapper: re-wrap with peft (lora_target_modules=None above left it un-wrapped) using
        # the SAME target-module pattern the checkpoint was trained with, then load its delta.
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file as _load_sft
        from peft.utils import set_peft_model_state_dict as _set_peft_sd
        targets = lora_target_modules[3:] if lora_target_modules.startswith("re:") \
            else [s.strip() for s in lora_target_modules.split(",") if s.strip()]
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, bias="none", target_modules=targets)
        badas.nn_model = get_peft_model(badas.nn_model, cfg)
        sft = Path(lora_adapter) / "adapter_model.safetensors"
        _set_peft_sd(badas.nn_model, _load_sft(str(sft)))
        print(f"[load] LoRA adapter for the probe checkpoint: {sft}")
    for p in badas.nn_model.parameters():
        p.requires_grad = False
    badas.nn_model.eval()

    captured = {}
    found = set()
    for name, mod in badas.nn_model.named_modules():
        for layer in layers:
            suffix = f".encoder.layer.{layer}"
            if name == suffix.lstrip(".") or name.endswith(suffix):
                def _hook(_m, _a, out, layer=layer):
                    captured[layer] = (out[0] if isinstance(out, (tuple, list)) else out)[0]
                mod.register_forward_hook(_hook)
                found.add(layer)
                break
    missing = set(layers) - found
    if missing:
        raise RuntimeError(f"Could not find encoder.layer.{sorted(missing)} to hook - "
                           f"run --dry-run-modules and check the real names.")
    print(f"[probe] hooked layers {sorted(found)}")
    return badas, captured


# =============================================================================
# Caching: per-window forward pass -> per-layer (features, occ, rel, box_h, path_gap) rows
# =============================================================================

def sample_rows(captured_layer: np.ndarray, occ: np.ndarray, rel: np.ndarray,
                box_h: np.ndarray, path_gap: np.ndarray, bg_per_window: int, rng) -> dict:
    """One window's contribution to the cache for one layer: every object token, plus up to
    `bg_per_window` background tokens sampled uniformly at random."""
    obj_idx = np.flatnonzero(occ > 0)
    bg_idx_all = np.flatnonzero(occ == 0)
    n_bg = min(bg_per_window, len(bg_idx_all))
    bg_idx = rng.choice(bg_idx_all, size=n_bg, replace=False) if n_bg else np.array([], dtype=int)
    idx = np.concatenate([obj_idx, bg_idx]).astype(int)
    return dict(feat=captured_layer[idx], occ=occ[idx].astype(np.float32), rel=rel[idx],
               box_h=box_h[idx], path_gap=path_gap[idx], is_object=(occ[idx] > 0))


def build_cache(badas, captured, examples: list, layers: list, labels_dir: Path,
                bg_per_window: int, seed: int, log_every: int = 25, log_prefix: str = "") -> dict:
    """{layer: {"feat": (N,1024) float32, "occ":.., "rel":.., "box_h":.., "path_gap":..,
    "is_object":..}} stacked across every window with a label file. Windows with no label
    file (aa4_token_labels.py hasn't covered that clip, or it was out_of_cached_span) are
    skipped, counted, and reported - not an error.

    Prints progress every `log_every` windows with elapsed/ETA - this loop runs one full
    ViT-L forward pass per window with no prefetch pipeline (unlike semsup_train.py's
    prefetch_clips), so at pool1761 scale it is the long step in an unattended run and needs
    a visible heartbeat, not just a final summary."""
    rng = np.random.default_rng(seed)
    per_layer_rows = {layer: [] for layer in layers}
    n_ok, n_missing = 0, 0
    t0 = time.time()
    for i, ex in enumerate(examples):
        label_path = labels_dir / f"{ex['frames_dir']}.npz"
        if not label_path.exists():
            n_missing += 1
            continue
        with np.load(label_path) as z:
            occ, rel = z["occ"], z["rel"]
            box_h, path_gap = z["box_h"], z["path_gap"]
        with torch.no_grad():
            badas.forward(ex["frame_paths"])
        for layer in layers:
            # float16 in the cache (halves RAM over ~1,761 windows x several layers); the
            # fitting/eval code below upcasts to float32 right before it hits torch/sklearn.
            feat = captured[layer].to(dtype=torch.float16).cpu().numpy()
            per_layer_rows[layer].append(
                sample_rows(feat, occ, rel, box_h, path_gap, bg_per_window, rng))
        n_ok += 1
        if n_ok % log_every == 0 or i == len(examples) - 1:
            elapsed = time.time() - t0
            rate = elapsed / n_ok
            remaining = len(examples) - (i + 1)
            print(f"  {log_prefix}[{i + 1}/{len(examples)}] cached={n_ok} missing={n_missing}  "
                 f"{rate:.2f}s/window  elapsed={elapsed/60:.1f}min  "
                 f"eta={remaining*rate/60:.1f}min", flush=True)
    print(f"[cache] {n_ok} windows cached, {n_missing} missing a label file")
    out = {}
    for layer in list(per_layer_rows):
        rows = per_layer_rows.pop(layer)   # release the per-window pieces as each layer is stacked
        out[layer] = {k: np.concatenate([r[k] for r in rows]) for k in
                      ("feat", "occ", "rel", "box_h", "path_gap", "is_object")}
        del rows
    return out, n_ok, n_missing


# =============================================================================
# Probe fitting (the SAME head architecture semsup_train.build_aux_head trains for real)
# =============================================================================

def fit_probe(feat: np.ndarray, target: np.ndarray, steps: int = 500, lr: float = 1e-2,
              tag: str = ""):
    """Fits build_aux_head() (LayerNorm-no-affine + Linear(1024,1)) by BCE against `target`
    (binary for occ, soft 0..1 for rel) - this IS the head semsup_train.py's --aux-head-init
    loads verbatim, so a probe result is a direct preview of what the frozen aux head can
    read out of this layer, not a proxy for it.

    Full-batch Adam, on the GPU when there is one. 2026-09-23: was 30 steps on CPU - too few
    for a 1,025-weight logistic probe to converge, which can make the probe falsely LOSE to
    the geometric baseline (both smoke tests showed rel's probe below baseline). The loss is
    printed at a few checkpoints so under-fitting is visible in the log, not inferred.
    Returns the head on the CPU (the caller evaluates and saves it there)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = build_aux_head(device)
    x = torch.from_numpy(feat).to(device=device, dtype=torch.float32)  # cache may be float16
    y = torch.from_numpy(target).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    report_at = {0, steps // 10, steps // 2, steps - 1}
    losses = []
    for step in range(steps):
        opt.zero_grad()
        loss = balanced_bce_loss(head(x).squeeze(-1), y)
        loss.backward()
        opt.step()
        if step in report_at:
            losses.append(f"{step}:{loss.item():.4f}")
    print(f"    [fit {tag}] train loss {' '.join(losses)}", flush=True)
    del x, y
    return head.cpu()


def fit_baseline(feat: np.ndarray, target: np.ndarray):
    """A logistic regression on the STATIC geometry features alone (box_h for occ; box_h +
    path_gap for rel) - never sees the ViT's tokens. What this baseline already achieves is
    the floor the ViT probe must clear to be worth anything; occ's ceiling is expected to be
    trivial (see --aux-mode's help in semsup_train.py) precisely because box_h is derived
    from the SAME offline boxes that define occ."""
    from sklearn.linear_model import LogisticRegression
    y_bin = (target > 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        return None
    clf = LogisticRegression(max_iter=500)
    clf.fit(feat, y_bin)
    return clf


def auroc(scores: np.ndarray, target: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y_bin = (target > 0).astype(int)
    if len(np.unique(y_bin)) < 2:
        return float("nan")
    return float(roc_auc_score(y_bin, scores))


def spearman_on_objects(scores: np.ndarray, target: np.ndarray, is_object: np.ndarray) -> float:
    from scipy.stats import spearmanr
    if is_object.sum() < 3:
        return float("nan")
    r, _ = spearmanr(scores[is_object], target[is_object])
    return float(r)


def evaluate_layer(train_cache: dict, val_cache: dict, mode: str, steps: int = 500,
                   tag: str = "") -> dict:
    """mode: 'occ' or 'rel'. Fits the probe + the geometric baseline on train_cache, reports
    both on val_cache. Returns the fitted torch head alongside the metrics so the caller can
    save the winning one."""
    target_key = mode
    y_train = train_cache[target_key]
    head = fit_probe(train_cache["feat"], y_train, steps=steps, tag=tag)
    with torch.no_grad():
        val_x = torch.from_numpy(val_cache["feat"]).to(dtype=torch.float32)
        probe_val_scores = head(val_x).squeeze(-1).sigmoid().numpy()

    if mode == "occ":
        base_feat_train = np.stack([train_cache["box_h"]], axis=1)
        base_feat_val = np.stack([val_cache["box_h"]], axis=1)
    else:
        base_feat_train = np.stack([train_cache["box_h"], train_cache["path_gap"]], axis=1)
        base_feat_val = np.stack([val_cache["box_h"], val_cache["path_gap"]], axis=1)
    baseline = fit_baseline(base_feat_train, y_train)
    base_val_scores = (baseline.predict_proba(base_feat_val)[:, 1]
                       if baseline is not None else np.full(len(val_cache["feat"]), np.nan))

    return dict(
        head=head,
        val_auroc=auroc(probe_val_scores, val_cache[target_key]),
        val_auroc_baseline=auroc(base_val_scores, val_cache[target_key]),
        val_spearman_objects=spearman_on_objects(probe_val_scores, val_cache["rel"], val_cache["is_object"]),
        val_spearman_objects_baseline=spearman_on_objects(base_val_scores, val_cache["rel"], val_cache["is_object"]),
        val_spearman_residual=spearman_residual(probe_val_scores, train_cache, val_cache),
        n_train=len(y_train), n_val=len(val_cache[target_key]),
        n_val_objects=int(val_cache["is_object"].sum()),
    )


def _static_features(cache: dict) -> np.ndarray:
    """The selection score's own STATIC inputs, per token: closeness = min(box_h/360, 1) and a
    clipped path gap (side is a monotone function of it), plus their product (the score is
    closeness x side x (1+approach), multiplicative)."""
    close = np.minimum(cache["box_h"] / 360.0, 1.0)
    gap = np.clip(cache["path_gap"], 0.0, 3.0)
    return np.stack([close, gap, close * gap], axis=1)


def spearman_residual(probe_val_scores: np.ndarray, train_cache: dict, val_cache: dict) -> float:
    """Does the probe carry the part of `rel` that static box geometry does NOT explain?
    `rel` = selection score / 2, and the score is built from box height and the path gap -
    so a baseline on those two (val_spearman_objects_baseline) nearly reproduces `rel` by
    construction and is NOT a bar the probe should be expected to clear. The informative
    question for an aux loss is the remainder (approach/looming, shielding, EMA history): fit
    rel ~ static geometry by least squares on TRAIN object tokens, take the VAL residual, and
    rank-correlate the probe's score with it on val object tokens. > 0 means the frozen trunk
    already encodes some non-static relevance; ~0 means it only reads position/size."""
    from sklearn.linear_model import LinearRegression
    tr, va = train_cache["is_object"], val_cache["is_object"]
    if tr.sum() < 10 or va.sum() < 10:
        return float("nan")
    reg = LinearRegression().fit(_static_features(train_cache)[tr], train_cache["rel"][tr])
    resid = val_cache["rel"][va] - reg.predict(_static_features(val_cache)[va])
    from scipy.stats import spearmanr
    r, _ = spearmanr(probe_val_scores[va], resid)
    return float(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--preprocess", default="compress256", choices=["crop", "compress256"])
    ap.add_argument("--layers", default="11,17,23")
    ap.add_argument("--checkpoint", required=True,
                    help="a LABEL for the report. 'base' = raw BADAS-Open (no --lora-adapter); "
                         "any other label (e.g. a1compress256, AA-rel-ep03) requires --lora-adapter "
                         "pointing at that run's saved adapter.")
    ap.add_argument("--lora-adapter", default=None)
    ap.add_argument("--lora-target-modules", default="query,key,value",
                    help="MUST match what the loaded adapter was trained with - "
                         "A1-compress256 used the legacy 'query,key,value' substring form.")
    ap.add_argument("--captions-path", default=str(CAPTIONS_1761))
    ap.add_argument("--labels-dir", default=str(LABELS_DIR_DEFAULT))
    ap.add_argument("--bg-per-window", type=int, default=256)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--cache-seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0,
                    help="passed straight to load_training_examples - takes the FIRST N rows "
                         "of the captions file, in FILE ORDER. Caption_Train4500_Mixed_1761.jsonl "
                         "is sorted by class (its first 800 rows are all positive), so this "
                         "biases hard unless N is tiny. Fine for a wiring smoke test (--limit "
                         "10-30); use --n-windows for anything meant to be read as a real result.")
    ap.add_argument("--n-windows", type=int, default=0,
                    help="random subsample of N windows, class-mixed (shuffled with "
                         "--sample-seed BEFORE the train/val split, unlike --limit's biased "
                         "prefix-slice). 0 = use every window. Applied on top of --limit if "
                         "both are set (rare - --limit would already have thrown most classes "
                         "away by then).")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--probe-steps", type=int, default=500,
                    help="full-batch Adam steps for each aux-head fit (was a fixed 30 - see fit_probe)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(l) for l in args.layers.split(",")]

    if args.checkpoint != "base" and not args.lora_adapter:
        raise ValueError(f"--checkpoint {args.checkpoint} requires --lora-adapter")
    if args.checkpoint == "base" and args.lora_adapter:
        raise ValueError("--checkpoint base must not be given --lora-adapter")

    import yaml
    with open(args.config, encoding="utf-8") as f:
        stagea_cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    badas, captured = load_probe_model(stagea_cfg, args.preprocess, layers,
                                       args.lora_adapter, args.lora_target_modules)

    examples = load_training_examples(limit=args.limit, captions_path=args.captions_path)
    if args.n_windows and args.n_windows < len(examples):
        import random as _random
        examples = _random.Random(args.sample_seed).sample(examples, args.n_windows)
        n_pos = sum(1 for e in examples if e["label"] == 1)
        print(f"[data] --n-windows {args.n_windows}: sampled {n_pos} pos / "
             f"{len(examples) - n_pos} neg (seed={args.sample_seed})")
    # SAME split A1-compress256 trained on (clip_level_split, seed 0) - the probe's train
    # rows must never include a video whose windows will be VAL for the real training run.
    train_ex, val_ex = clip_level_split(examples, val_frac=args.val_frac, seed=args.split_seed)
    print(f"[data] {len(train_ex)} train / {len(val_ex)} val windows")

    train_cache, n_ok_tr, n_missing_tr = build_cache(
        badas, captured, train_ex, layers, Path(args.labels_dir), args.bg_per_window,
        args.cache_seed, log_prefix="train ")
    val_cache, n_ok_val, n_missing_val = build_cache(
        badas, captured, val_ex, layers, Path(args.labels_dir), args.bg_per_window,
        args.cache_seed + 1, log_prefix="val ")
    # The ViT is no longer needed - release it before fitting, which moves the full train
    # feature matrix (~2-3 GB float32 at pool scale) onto the GPU; with the ~5.5 GB model still
    # resident that would not fit a 6 GB card.
    del badas, captured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = dict(checkpoint=args.checkpoint, preprocess=args.preprocess, layers=layers,
                 n_train_windows=n_ok_tr, n_train_missing_labels=n_missing_tr,
                 n_val_windows=n_ok_val, n_val_missing_labels=n_missing_val, per_layer={})
    for layer in layers:
        layer_report = {}
        for mode in ("occ", "rel"):
            result = evaluate_layer(train_cache[layer], val_cache[layer], mode,
                                    steps=args.probe_steps, tag=f"L{layer} {mode}")
            head = result.pop("head")
            head_path = out_dir / f"aux_head_layer{layer}_{mode}.pt"
            torch.save(head.state_dict(), head_path)
            result["head_path"] = str(head_path)
            layer_report[mode] = result
            print(f"[layer {layer}] {mode}: val_auroc={result['val_auroc']:.4f} "
                 f"(baseline {result['val_auroc_baseline']:.4f})  "
                 f"spearman_objects={result['val_spearman_objects']:.4f} "
                 f"(baseline {result['val_spearman_objects_baseline']:.4f})  "
                 f"spearman_residual={result['val_spearman_residual']:.4f}  "
                 f"n_val_objects={result['n_val_objects']}")
        report["per_layer"][str(layer)] = layer_report

    # json.dump's default allow_nan=True emits a bare `NaN` token (INVALID json - jq/JS/most
    # dashboards reject it), the same pitfall semsup_train.py's `_j` guard exists for. A
    # Spearman on a constant baseline (e.g. occ's box_h baseline saturating near 1.0 for every
    # object) is exactly the NaN case here, so this isn't a rare edge case to skip.
    def _nan_to_null(obj):
        if isinstance(obj, float) and obj != obj:
            return None
        if isinstance(obj, dict):
            return {k: _nan_to_null(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_nan_to_null(v) for v in obj]
        return obj

    with open(out_dir / "probe_report.json", "w", encoding="utf-8") as f:
        json.dump(_nan_to_null(report), f, indent=2)
    print(f"\n[done] wrote {out_dir / 'probe_report.json'}")
    print("Gate check: 'occ' should be near-saturated at every layer (confirms presence is "
         "trivial); 'rel' should clear its baseline by a real margin, at the layer chosen "
         "for --aux-layer, but not itself be saturated (child plan Phase 3).")


if __name__ == "__main__":
    main()
