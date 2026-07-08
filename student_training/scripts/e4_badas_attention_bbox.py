"""
e4_badas_attention_bbox.py
===========================
GATE 0 (+ Stage D-0) of the e4 Stage-D plan: extract a "where is the hazard"
bounding box from BADAS-Open's (V-JEPA2) self-attention, training-free, and
draw it on the frame InternVL will later read.

WHY THIS IS A GATE, NOT A GIVEN
--------------------------------
BADAS-Open is loaded via a CUSTOM loader (nexar-ai/nexight source), not plain
HF `transformers`. That means:
  - The BADAS-2.0 paper's instruction ("just use attn_implementation='eager'")
    is a transformers-specific flag that may not exist on this custom module.
  - Attention weights may be computed via a fused/SDPA kernel that never
    materializes a (B, heads, P, P) tensor for a hook to see.
  - The exact attention-module names/layer indices are UNCONFIRMED (we only
    know `temporal_processor` from Stage A/B). "Layers 12-20" in the paper is
    THEIR ViT-L indexing, not verified against ours.
  - The spatial/temporal factorization of the 2560-token patch grid is
    UNCONFIRMED for BADAS-Open specifically (BADAS-Open uses img_size=224,
    the paper's 256x256 + future-concat math was for BADAS-2.0).

So this script is a staged PROBE, not a one-shot extractor:

  --list_modules   Load the model, dump every module whose name/class hints
                    at attention. No image processing. Run this FIRST on the
                    pod so hooking targets real names instead of guesses.

  --probe          Try to capture attention weights on ~N known clips via
                    (a) a straightforward forward hook on discovered attention
                    modules, forcing eager/math attention where possible, and
                    (b) if that returns nothing, a MANUAL Q/K reconstruction
                    fallback: hook the Q and K sub-projections directly and
                    recompute weights = softmax(QK^T / sqrt(d)) ourselves --
                    this works regardless of which attention kernel is used
                    internally, at the cost of assuming a standard
                    q_proj/k_proj (or in_proj_weight) layout.
                    Aggregates over late layers, reshapes using an
                    AUTO-DETECTED spatial/temporal split (not hardcoded),
                    upsamples to frame resolution, thresholds to a bbox,
                    draws it, and saves everything for eyeballing.
                    Prints a clear GATE-0 PASS/FAIL summary at the end.

  --draw           (Stage D-0, only after GATE 0 passes) run the same
                    extraction over a full manifest, drawing a bold box on
                    each clip's peak-risk frame and writing the bbox cache
                    JSONL the Stage-D dataset consumes.

  --dry_run        Manifest/frame validation only. No GPU, no model.

REUSES (not duplicated): load_config, load_manifest, frame_paths_for,
load_badas, preprocess_clip from e4_stageA_badas_open_eval.py.

Env overrides: none required; paths are CLI args (mirrors Stage A/B/C scripts).

Usage (pod):
  # Step 1 -- see the real module tree before hooking anything
  python student_training/scripts/e4_badas_attention_bbox.py \
      --config student_training/configs/e4_stageA.yaml --list_modules \
      --out /root/e4_stageD/bbox_probe/

  # Step 2 -- probe on 10 known-positive val clips
  python student_training/scripts/e4_badas_attention_bbox.py \
      --config student_training/configs/e4_stageA.yaml --probe \
      --manifest dataset/manifests/val_e3a.jsonl \
      --frames_root /workspace/data/train_HiRes --n 10 \
      --out /root/e4_stageD/bbox_probe/

  # Step 3 (only if Step 2 PASSES) -- full D-0 cache build
  python student_training/scripts/e4_badas_attention_bbox.py \
      --config student_training/configs/e4_stageA.yaml --draw \
      --manifest dataset/manifests/val_e3a.jsonl \
      --frames_root /workspace/data/train_HiRes --split val \
      --out /root/e4_stageD/cache/boxed_frames/ \
      --cache /root/e4_stageD/cache/bbox_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse, don't duplicate.
from student_training.scripts.e4_stageA_badas_open_eval import (  # noqa: E402
    load_config, load_manifest, frame_paths_for, load_badas, preprocess_clip,
)

ATTN_NAME_HINTS = ("attn", "attention", "self_attn", "mha", "multihead")
QPROJ_HINTS = ("q_proj", "query", "qkv", "in_proj")
KPROJ_HINTS = ("k_proj", "key")


# =============================================================================
# Step 1 — module introspection (no GPU-heavy work beyond loading the model)
# =============================================================================

def list_modules(nn_model, out_dir: Path):
    """Dump the full module tree, flagging anything attention-shaped, so
    hooking targets are chosen from REAL names, not the paper's guesses."""
    lines = []
    flagged = []
    for name, mod in nn_model.named_modules():
        cls = type(mod).__name__
        line = f"{name or '<root>':60s} {cls}"
        lines.append(line)
        low = (name + " " + cls).lower()
        if any(h in low for h in ATTN_NAME_HINTS):
            flagged.append((name, cls))

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "named_modules_full.txt"
    flagged_path = out_dir / "named_modules_attention_candidates.txt"
    full_path.write_text("\n".join(lines), encoding="utf-8")
    flagged_path.write_text(
        "\n".join(f"{n:60s} {c}" for n, c in flagged), encoding="utf-8")

    print(f"\n=== --list_modules ===")
    print(f"  total modules            : {len(lines)}")
    print(f"  attention-hinted modules : {len(flagged)}")
    print(f"  full tree     -> {full_path}")
    print(f"  candidates    -> {flagged_path}")
    if flagged:
        print("\n  candidates (first 20):")
        for n, c in flagged[:20]:
            print(f"    {n:55s} {c}")
    else:
        print("\n  WARNING: no module name/class contains "
              f"{ATTN_NAME_HINTS} — attention is either named unusually "
              "(inspect named_modules_full.txt manually) or not exposed as "
              "a discrete submodule at all.")
    print("=== inspect the candidates file, then re-run with --probe "
          "--layer_substr '<chosen substring>' if the default doesn't match ===\n")
    return flagged


# =============================================================================
# Step 2 — attention capture (simple hook, then manual Q/K fallback)
# =============================================================================

def _looks_like_attn_weights(t) -> bool:
    """Heuristic: an attention-weight tensor has a trailing dim that is a
    probability distribution (sums to ~1) and at least 3 dims."""
    import torch
    if not torch.is_tensor(t) or t.dim() < 3:
        return False
    s = t.detach().float()
    row_sums = s.sum(dim=-1)
    return bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.05))


def attach_simple_hooks(nn_model, layer_substr: str):
    """Hook forward on modules matching `layer_substr`, capture ANY tensor
    output that looks like softmax'd attention weights."""
    captured = {}

    def make_hook(name):
        def hook(_m, _inp, out):
            candidates = out if isinstance(out, (tuple, list)) else (out,)
            for c in candidates:
                if _looks_like_attn_weights(c):
                    captured.setdefault(name, []).append(c.detach())
        return hook

    handles = []
    for name, mod in nn_model.named_modules():
        if layer_substr.lower() in name.lower():
            handles.append(mod.register_forward_hook(make_hook(name)))
    return captured, handles


def attach_manual_qk_hooks(nn_model, layer_substr: str):
    """Fallback when the simple hook captures nothing (fused/SDPA kernels
    discard weights). Hooks Q and K sub-projections directly; weights are
    reconstructed as softmax(QK^T / sqrt(d)) after the forward pass."""
    qk = {}

    def q_hook(name):
        def hook(_m, _i, out):
            qk.setdefault(name, {})["q"] = out.detach()
        return hook

    def k_hook(name):
        def hook(_m, _i, out):
            qk.setdefault(name, {})["k"] = out.detach()
        return hook

    handles = []
    for name, mod in nn_model.named_modules():
        low = name.lower()
        if layer_substr.lower() not in low:
            continue
        if any(low.endswith(h) or h in low.split(".")[-1] for h in QPROJ_HINTS):
            handles.append(mod.register_forward_hook(q_hook(name)))
        elif any(low.endswith(h) or h in low.split(".")[-1] for h in KPROJ_HINTS):
            handles.append(mod.register_forward_hook(k_hook(name)))
    return qk, handles


def reconstruct_weights_from_qk(qk: dict, n_heads: int = 8):
    """weights = softmax(QK^T / sqrt(d_head)) per captured layer. Assumes Q/K
    shape (B, P, D) or (B, heads, P, d_head); handles both defensively."""
    import torch
    import torch.nn.functional as F
    out = {}
    for name, d in qk.items():
        if "q" not in d or "k" not in d:
            continue
        q, k = d["q"], d["k"]
        if q.dim() == 3:  # (B, P, D) -> split heads
            B, P, D = q.shape
            hd = D // n_heads
            q = q.view(B, P, n_heads, hd).transpose(1, 2)
            k = k.view(B, P, n_heads, hd).transpose(1, 2)
        d_head = q.shape[-1]
        w = F.softmax((q @ k.transpose(-2, -1)) / (d_head ** 0.5), dim=-1)
        out[name] = w  # (B, heads, P, P)
    return out


def remove_hooks(handles):
    for h in handles:
        h.remove()


# =============================================================================
# Spatial aggregation -> bbox (auto-detected layout, not hardcoded)
# =============================================================================

def attn_to_saliency_map(attn_layers: dict, spatial_grid: int = 16):
    """Mean over heads and captured layers -> per-token saliency (mean
    attention RECEIVED, a standard saliency proxy) -> reshape to
    (temporal_groups, spatial_grid, spatial_grid) -> mean over temporal.

    P (num tokens) is read from the actual captured tensor, and
    temporal_groups = P // spatial_grid**2 is DERIVED, not assumed. If P does
    not divide evenly by spatial_grid**2, this is reported, not silently
    papered over -- it means our spatial_grid guess is wrong for this model
    and needs adjusting (see printed diagnostics).
    """
    import torch
    if not attn_layers:
        return None, None

    per_layer_saliency = []
    P_ref = None
    for name, w in attn_layers.items():
        # w: (B, heads, P, P) attention weights (row = query, col = key)
        if w.dim() != 4:
            continue
        P = w.shape[-1]
        P_ref = P if P_ref is None else P_ref
        if P != P_ref:
            continue  # skip layers with a mismatched token count
        received = w.mean(dim=1)[0].mean(dim=0)  # (P,) mean attention received
        per_layer_saliency.append(received)

    if not per_layer_saliency or P_ref is None:
        return None, None

    sal = torch.stack(per_layer_saliency, dim=0).mean(dim=0)  # (P,)
    spatial_sq = spatial_grid * spatial_grid
    if P_ref % spatial_sq != 0:
        return None, {"P": P_ref, "spatial_grid": spatial_grid,
                       "error": f"P={P_ref} not divisible by spatial_grid^2={spatial_sq}"}
    temporal_groups = P_ref // spatial_sq
    sal = sal.view(temporal_groups, spatial_grid, spatial_grid)
    # Default: UNIFORM mean across temporal groups. The BADAS-2.0 paper's
    # exponential temporal weighting (eq. 2) assumes a specific token
    # semantics (their 256x256 + future-concat architecture) that is
    # UNCONFIRMED for BADAS-Open's tap point -- do not blindly port it.
    sal_2d = sal.mean(dim=0)  # (spatial_grid, spatial_grid)
    info = {"P": P_ref, "spatial_grid": spatial_grid, "temporal_groups": temporal_groups}
    return sal_2d, info


def saliency_to_bbox(sal_2d, frame_wh: tuple, model_img_size: int, percentile: float = 0.75):
    """Upsample the small saliency grid to the model's square input size,
    threshold to the top (1-percentile) mass, take the largest connected
    component's bbox, then map to the ORIGINAL frame's resolution (non-
    uniform scale, since BADAS squash-resizes to a square)."""
    import cv2
    import numpy as np

    sal = sal_2d.float().cpu().numpy()
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    up = cv2.resize(sal, (model_img_size, model_img_size), interpolation=cv2.INTER_LINEAR)

    thresh = np.quantile(up, percentile)
    mask = (up >= thresh).astype(np.uint8)
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_comp <= 1:
        return None  # nothing above threshold (degenerate saliency)
    # largest non-background component
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[idx]
    # map model-square coords -> original frame coords (independent x/y scale)
    W, H = frame_wh
    sx, sy = W / model_img_size, H / model_img_size
    return (int(x * sx), int(y * sy), int((x + w) * sx), int((y + h) * sy))


def draw_bbox(frame_path: str, bbox_xyxy: tuple, out_path: str,
              color=(0, 0, 255), thickness=4):
    """Draw a bold box on the frame (BGR red by default -- bold/bright so it
    survives InternViT's pixel-shuffle downsampling downstream)."""
    import cv2
    img = cv2.imread(frame_path)
    x1, y1, x2, y2 = bbox_xyxy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)


# =============================================================================
# --probe : end-to-end on N sample clips, prints the GATE-0 verdict
# =============================================================================

def probe(cfg, records, frames_root, out_dir: Path, n: int,
          layer_substr: str, spatial_grid: int, n_heads: int, percentile: float):
    import torch

    vjepa, nn_model, device = load_badas(cfg)
    pattern = cfg["data"]["frame_filename_pattern"]
    img_size = cfg["preprocess"]["img_size"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== --probe : attempting attention capture (layer_substr='{layer_substr}') ===")
    simple_captured, simple_handles = attach_simple_hooks(nn_model, layer_substr)
    qk_captured, qk_handles = attach_manual_qk_hooks(nn_model, layer_substr)

    results = []
    n_ok = 0
    for i, r in enumerate(records[:n]):
        simple_captured.clear()
        qk_captured.clear()
        paths = frame_paths_for(r, frames_root, pattern)
        clip = preprocess_clip(vjepa, paths).to(device)

        # try disabling fused kernels so eager/math attention (if supported)
        # is more likely to expose weights to the simple hook.
        try:
            with torch.no_grad(), torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_math=True, enable_mem_efficient=False):
                nn_model(clip)
        except Exception:
            with torch.no_grad():
                nn_model(clip)

        mode = None
        attn_layers = {name: torch.stack(v).mean(dim=0) if len(v) > 1 else v[0]
                       for name, v in simple_captured.items()} if simple_captured else {}
        if attn_layers:
            mode = "simple_hook"
        else:
            recon = reconstruct_weights_from_qk(qk_captured, n_heads=n_heads)
            if recon:
                attn_layers = recon
                mode = "manual_qk"

        sal_2d, info = attn_to_saliency_map(attn_layers, spatial_grid=spatial_grid)
        row = {"video_id": r.get("video_id"), "capture_mode": mode, "layout": info}

        if sal_2d is not None:
            peak_frame = paths[-1]  # last frame = current moment (paper's peak-risk proxy)
            with Image.open(peak_frame) as im:
                wh = im.size
            bbox = saliency_to_bbox(sal_2d, wh, img_size, percentile=percentile)
            if bbox is not None:
                out_img = out_dir / f"{r.get('video_id')}_bbox.jpg"
                draw_bbox(peak_frame, bbox, str(out_img))
                row["bbox_xyxy"] = bbox
                row["boxed_frame"] = str(out_img)
                n_ok += 1
        results.append(row)
        print(f"  [{i+1}/{min(n, len(records))}] {r.get('video_id')}  "
              f"mode={mode}  layout={info}  "
              f"bbox={row.get('bbox_xyxy')}")

    remove_hooks(simple_handles)
    remove_hooks(qk_handles)

    report_path = out_dir / "probe_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n=== GATE 0 RESULT ===")
    print(f"  clips probed          : {min(n, len(records))}")
    print(f"  attention captured    : {sum(1 for r in results if r['capture_mode'])}"
          f" / {min(n, len(records))}")
    print(f"  bbox produced         : {n_ok} / {min(n, len(records))}")
    print(f"  report                : {report_path}")
    print(f"  boxed frames          : {out_dir}")
    if n_ok >= max(1, int(0.7 * min(n, len(records)))):
        print("  VERDICT: PASS-CANDIDATE — attention captured and boxes produced on "
              "most probed clips. >>> EYEBALL the boxed frames before declaring GATE 0 "
              "passed (do the boxes land on the hazard, not noise?) <<<")
    else:
        print("  VERDICT: LIKELY FAIL — attention capture and/or bbox production failed "
              "on most clips. Do NOT proceed to Stage D-0/D-1. Options: (a) inspect "
              "named_modules_attention_candidates.txt and retry with a different "
              "--layer_substr, (b) accept the fallback: score-as-text only, no bbox "
              "(revisit with user).")
    print("=== end GATE 0 ===\n")
    return results


# =============================================================================
# --draw : Stage D-0 full manifest bbox cache (only after GATE 0 passes)
# =============================================================================

def draw_full(cfg, records, frames_root, split, out_dir: Path, cache_path: Path,
              layer_substr: str, spatial_grid: int, n_heads: int, percentile: float):
    import torch

    vjepa, nn_model, device = load_badas(cfg)
    pattern = cfg["data"]["frame_filename_pattern"]
    img_size = cfg["preprocess"]["img_size"]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    simple_captured, simple_handles = attach_simple_hooks(nn_model, layer_substr)
    qk_captured, qk_handles = attach_manual_qk_hooks(nn_model, layer_substr)

    n_written = 0
    with open(cache_path, "w", encoding="utf-8") as fh:
        for i, r in enumerate(records):
            simple_captured.clear()
            qk_captured.clear()
            paths = frame_paths_for(r, frames_root, pattern)
            clip = preprocess_clip(vjepa, paths).to(device)
            with torch.no_grad():
                logits = nn_model(clip)
                score = float(torch.softmax(logits / 2.0, dim=1)[0, 1].item())

            attn_layers = {name: (torch.stack(v).mean(dim=0) if len(v) > 1 else v[0])
                           for name, v in simple_captured.items()} if simple_captured else {}
            if not attn_layers:
                recon = reconstruct_weights_from_qk(qk_captured, n_heads=n_heads)
                attn_layers = recon or {}

            sal_2d, _ = attn_to_saliency_map(attn_layers, spatial_grid=spatial_grid)
            bbox = None
            peak_frame = paths[-1]
            if sal_2d is not None:
                with Image.open(peak_frame) as im:
                    wh = im.size
                bbox = saliency_to_bbox(sal_2d, wh, img_size, percentile=percentile)

            boxed_path = str(out_dir / f"{r.get('video_id')}.jpg")
            if bbox is not None:
                draw_bbox(peak_frame, bbox, boxed_path)
            else:
                # no usable saliency this clip -- copy the frame unboxed so
                # downstream code always finds a file at the expected path.
                import shutil
                os.makedirs(os.path.dirname(boxed_path), exist_ok=True)
                shutil.copy(peak_frame, boxed_path)

            fh.write(json.dumps({
                "video_id": r.get("video_id"),
                "frames_dir": r.get("frames_dir"),
                "horizon_label": r.get("horizon_label") or r.get("requested_time_to_event"),
                "bbox_xyxy": bbox,
                "peak_frame_idx": r["frame_indices"][-1] if r.get("frame_indices") else None,
                "boxed_frame": boxed_path,
                "score": round(score, 6),
                "split": split,
            }) + "\n")
            n_written += 1
            if n_written % 20 == 0 or n_written == len(records):
                print(f"  [{n_written}/{len(records)}]  last bbox={bbox}")

    remove_hooks(simple_handles)
    remove_hooks(qk_handles)
    print(f"\nWrote {n_written} rows -> {cache_path}")
    print(f"Boxed frames -> {out_dir}")


# =============================================================================
# --dry_run : manifest/frame validation only
# =============================================================================

def dry_run(records, frames_root, pattern, n_check=5):
    print("\n=== DRY RUN (bbox script) — no model, no GPU ===")
    missing = 0
    for r in records[:n_check]:
        for p in frame_paths_for(r, frames_root, pattern):
            if not os.path.exists(p):
                missing += 1
    print(f"  records: {len(records)}  checked: {n_check}  missing frames: {missing}")
    print("=== dry run OK ===\n")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest")
    ap.add_argument("--frames_root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out", help="output dir (probe images / boxed frames); "
                    "required for --list_modules / --probe / --draw")
    ap.add_argument("--cache", help="bbox cache JSONL path (--draw mode)")
    ap.add_argument("--n", type=int, default=10, help="clips to probe (--probe mode)")
    ap.add_argument("--layer_substr", default="attn",
                    help="module-name substring to hook; refine after --list_modules")
    ap.add_argument("--spatial_grid", type=int, default=16,
                    help="assumed spatial grid side (16 -> 16x16=256, per BADAS-2.0 Sec 4.1); "
                         "override if --list_modules / --probe layout diagnostics disagree")
    ap.add_argument("--n_heads", type=int, default=8,
                    help="attention heads, for the manual Q/K fallback only")
    ap.add_argument("--bbox_percentile", type=float, default=0.75)
    ap.add_argument("--list_modules", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.list_modules:
        if not args.out:
            ap.error("--out is required for --list_modules")
        _, nn_model, _ = load_badas(cfg)
        list_modules(nn_model, Path(args.out))
        return

    if not args.manifest or not args.frames_root:
        ap.error("--manifest and --frames_root are required for --probe / --draw / --dry_run")
    manifest_path = args.manifest if os.path.isabs(args.manifest) \
        else os.path.join(PROJECT_ROOT, args.manifest)
    frames_root = args.frames_root if os.path.isabs(args.frames_root) \
        else os.path.join(PROJECT_ROOT, args.frames_root)
    records = load_manifest(manifest_path)

    if args.dry_run:
        dry_run(records, frames_root, cfg["data"]["frame_filename_pattern"])
        return

    if not args.out:
        ap.error("--out is required for --probe / --draw")
    out_dir = Path(args.out)

    if args.probe:
        probe(cfg, records, frames_root, out_dir, args.n, args.layer_substr,
              args.spatial_grid, args.n_heads, args.bbox_percentile)
        return

    if args.draw:
        if not args.cache:
            ap.error("--cache is required for --draw")
        draw_full(cfg, records, frames_root, args.split, out_dir, Path(args.cache),
                  args.layer_substr, args.spatial_grid, args.n_heads, args.bbox_percentile)
        return

    ap.error("choose one of --list_modules / --probe / --draw / --dry_run")


if __name__ == "__main__":
    main()
