"""
e4_stageA_badas_open_eval.py
============================
Stage A of `e4_vjepa_reason`: reproduce BADAS-Open's collision-prediction AP on
our Nexar test halves (Private 677 + Public 667), to anchor the frozen scorer
used by later stages.

WHAT IT DOES (per split)
  1. Loads nexar-ai/BADAS-Open via its OFFICIAL loader, so the resize +
     normalization come from BADAS's own code (neither the BADAS nor V-JEPA2
     paper fully specifies the transform — it lives in the package).
  2. For each clip: loads the 16 last-window frames named by the manifest,
     hands them to BADAS's transform (which downsizes to 256x256), runs the
     model -> P(collision).
  3. Writes a per-clip JSONL with the Stage-A schema (see OUTPUT SCHEMA).

FRAMING (see plan §3.2): the MODEL input is 256x256 regardless. We feed the
HI-RES (1280x720) source frame and let BADAS resize — V-JEPA2 eval is
resize-then-crop (not a squash), so our pre-squashed 256 frames would mismatch.

WINDOWING (see plan §3.4): the manifests already encode the last-16-frame
(~2 s) window at stride 4. We reuse them unchanged.

OUTPUT SCHEMA (one JSON object per clip)
  {video_id, ground_truth, group, time_before_s, score, collision_verdict, split}
  - ground_truth   = manifest `event_occurs` (1 collision / 0 none)
  - score          = BADAS P(collision) in [0,1]
  - collision_verdict = "YES" if score >= threshold else "NO"   (threshold cfg, default 0.5)
  - split          = "Private" | "Public"  (from --split)

RUN TARGET: RunPod GPU (needs credit). DO NOT run the scoring step locally.
  Local validation without a GPU:
    python student_training/scripts/e4_stageA_badas_open_eval.py \
        --config student_training/configs/e4_stageA.yaml \
        --manifest dataset/manifests/test_manifest_hires.jsonl \
        --frames_root dataset/test --split Private --dry_run
  Full run on RunPod: drop --dry_run (see RUNPOD_E4_STAGEA.txt).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# Config + manifest
# =============================================================================

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_manifest(path: str) -> list:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def frame_paths_for(record: dict, frames_root: str, pattern: str) -> list:
    """Absolute paths of the clip's frames. `frames_dir` falls back to video_id."""
    frames_dir = record.get("frames_dir") or record["video_id"]
    return [os.path.join(frames_root, frames_dir, pattern.format(i))
            for i in record["frame_indices"]]


# =============================================================================
# Dry-run validation (no GPU, no model)
# =============================================================================

def dry_run(cfg, records, frames_root, split, n_check=5):
    print(f"\n=== DRY RUN ({split}) — no model, no scoring ===")
    pp = cfg["preprocess"]
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]

    gts = [int(r[gt_field]) for r in records]
    from collections import Counter
    print(f"  records          : {len(records)}")
    print(f"  positives/neg    : {sum(gts)} / {len(gts) - sum(gts)}")
    print(f"  group dist       : {dict(Counter(r.get('group') for r in records))}")

    missing, sizes = 0, []
    for r in records[:n_check]:
        paths = frame_paths_for(r, frames_root, pattern)
        if len(paths) != pp["num_frames"]:
            print(f"  WARN {r['video_id']}: {len(paths)} frames != {pp['num_frames']}")
        for p in paths:
            if not os.path.exists(p):
                missing += 1
            else:
                sizes.append(Image.open(p).size)
    print(f"  checked {n_check} clips: missing frames = {missing}")
    print(f"  sample frame sizes: {set(sizes)}  (expect 1280x720 hi-res)")
    if any(s == (256, 256) for s in sizes):
        print("  NOTE: 256x256 source detected -> pre-squashed frames; prefer *_hires "
              "so BADAS owns the resize (plan §3.2).")
    print(f"  preprocess.source = {pp['source']} "
          "(verify img_size/normalization against the badas package on RunPod)")
    print("=== dry run OK (no scores computed) ===\n")


# =============================================================================
# Scoring (RunPod GPU only)
# =============================================================================

def load_badas(cfg):
    """Load BADAS-Open via its official loader + expose its frame transform."""
    import torch
    from huggingface_hub import hf_hub_download

    repo = cfg["model"]["hf_repo"]
    print(f"Loading {repo} via official loader ...")
    loader_path = hf_hub_download(repo_id=repo, filename="badas_loader.py")
    sys.path.insert(0, os.path.dirname(loader_path))
    from badas_loader import load_badas_model  # noqa: E402
    model = load_badas_model()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Prefer BADAS's own frame transform (resolves the 224/256 + normalization
    # ambiguity). Fall back to manual yaml values only if absent.
    transform = None
    if cfg["preprocess"]["source"] == "official":
        try:
            import badas
            transform = getattr(badas, "frame_transform", None) \
                or getattr(badas, "build_transform", None)
        except Exception:
            transform = None
        if transform is None:
            print("  NOTE: official frame transform not found in `badas`; "
                  "using manual yaml fallback. VERIFY img_size/norm.")
    return model, transform, device


def manual_transform(paths, pp):
    import torch
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    tfm = T.Compose([
        T.Resize((pp["img_size"], pp["img_size"]), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=pp["norm_mean"], std=pp["norm_std"]),
    ])
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths], dim=0)


def score_split(cfg, records, frames_root, split, out_path, limit=0):
    import torch
    pp = cfg["preprocess"]
    pattern = cfg["data"]["frame_filename_pattern"]
    gt_field = cfg["data"]["gt_field"]
    threshold = cfg["data"].get("verdict_threshold", 0.5)

    model, transform, device = load_badas(cfg)
    if limit:
        records = records[:limit]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"Scoring {len(records)} {split} clips -> {out_path}")
    n = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for k, r in enumerate(records):
            paths = frame_paths_for(r, frames_root, pattern)
            imgs = [Image.open(p).convert("RGB") for p in paths]
            clip = transform(imgs) if transform is not None else manual_transform(paths, pp)
            clip = clip.unsqueeze(0).to(device)  # (1, T, 3, H, W) — adjust if model differs
            with torch.no_grad():
                out = model(clip)
            score = float(out.reshape(-1).max().item()) if hasattr(out, "reshape") else float(out)
            fh.write(json.dumps({
                "video_id":          r["video_id"],
                "ground_truth":      int(r[gt_field]),
                "group":             r.get("group"),
                "time_before_s":     r.get("time_before_event_s"),
                "score":             round(score, 4),
                "collision_verdict": "YES" if score >= threshold else "NO",
                "split":             split,
            }) + "\n")
            fh.flush()
            n += 1
            if n % 50 == 0:
                print(f"  {n}/{len(records)}")

    # Quick AP/AUC sanity (authoritative metrics come from evaluate_metrics.py).
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = [json.loads(l) for l in open(out_path)]
    yt = np.array([r["ground_truth"] for r in rows])
    ys = np.array([r["score"] for r in rows])
    ap, auc = average_precision_score(yt, ys), roc_auc_score(yt, ys)
    exp, tol = cfg["acceptance"]["expected_ap"], cfg["acceptance"]["ap_tolerance"]
    print(f"\n  {split}: n={len(rows)}  AP={ap:.4f}  AUC={auc:.4f}  "
          f"(expected ~{exp}; {'PASS' if abs(ap - exp) <= tol else 'CHECK PREPROCESSING'})")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--frames_root", required=True)
    ap.add_argument("--split", required=True, choices=["Private", "Public"])
    ap.add_argument("--output", help="Output JSONL path (required unless --dry_run)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate manifest/frames only. No GPU, no scoring.")
    ap.add_argument("--limit", type=int, default=0, help="Score only first N clips (debug).")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest_path = args.manifest if os.path.isabs(args.manifest) \
        else os.path.join(PROJECT_ROOT, args.manifest)
    frames_root = args.frames_root if os.path.isabs(args.frames_root) \
        else os.path.join(PROJECT_ROOT, args.frames_root)
    records = load_manifest(manifest_path)

    if args.dry_run:
        dry_run(cfg, records, frames_root, args.split)
        return

    if not args.output:
        ap.error("--output is required unless --dry_run")
    out_path = args.output if os.path.isabs(args.output) \
        else os.path.join(PROJECT_ROOT, args.output)
    score_split(cfg, records, frames_root, args.split, out_path, args.limit)


if __name__ == "__main__":
    main()
