"""
lookahead_extract_features.py  (Stage 0b, POD side)
====================================================
Frozen BADAS-Open (no LoRA): for every training window that has an adjacent-horizon partner,
save the pooled 1024-d vector z (input of the crash head) and the head's logit.

Windows:
  positives : dataset/train/<vid>_hires_tte{05,10,15}    (full 4,446 pool, 741 videos)
  negatives : dataset/train/<vid>_hires_midtest{05,10,15} (the 905 midpoint-aligned windows)

Output: OUT/features.npz  {names[N], z[N,1024], logit[N,2]},  OUT/windows.jsonl,
        OUT/head_check.json (whether nn_model.classifier(z) reproduces the logit).
No labels or training. Analysis runs locally (lookahead_feasibility.py).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semsup_common import TrainableBadasWrapper  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HZ = {"05": 0.5, "10": 1.0, "15": 1.5}


def collect(train_root: Path):
    rows = []
    for d in sorted(train_root.iterdir()):
        m = re.fullmatch(r"(\d+)_hires_(tte|midtest)(05|10|15)", d.name)
        if not m:
            continue
        vid, kind, hz = m.groups()
        frames = sorted(d.glob("frame_*.jpg"))
        if len(frames) < 16:
            continue
        rows.append(dict(name=d.name, video_id=vid, label=1 if kind == "tte" else 0,
                         horizon=HZ[hz], frame_paths=[str(f) for f in frames[:16]]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "student_training/configs/e4_stageA.yaml"))
    ap.add_argument("--train-root", default=str(ROOT / "dataset/train"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names-file", default=None, help="only extract windows whose directory name is listed (one per line)")
    args = ap.parse_args()

    rows = collect(Path(args.train_root))
    if args.names_file:
        keep_names = {l.strip() for l in open(args.names_file, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["name"] in keep_names]
    if args.limit:
        rows = rows[: args.limit]
    print(f"windows: {len(rows)} | pos {sum(r['label'] for r in rows)} | neg {sum(1 - r['label'] for r in rows)}")

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    w = TrainableBadasWrapper(cfg, lora_target_modules=None, preprocess_mode="compress256")
    nn_model = w.nn_model

    head = getattr(nn_model, "classifier", None)
    print("children:", [n for n, _ in nn_model.named_children()])

    Z, L, keep = [], [], []
    with torch.no_grad():
        for i, ex, clip, err in w.prefetch_clips(rows, num_workers=8, prefetch=16, key="frame_paths"):
            if err is not None:
                print(f"  [warn] skip {ex['name']}: {err}")
                continue
            logits, _ = w.forward_clip(clip.to(w.device))
            z = w._captured["pooled"].float().reshape(1, -1)
            Z.append(z.cpu().numpy()[0]); L.append(logits.float().cpu().numpy()[0]); keep.append(ex)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(rows)}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    Z = np.stack(Z); L = np.stack(L)
    np.savez_compressed(out / "features.npz", names=np.array([k["name"] for k in keep]), z=Z, logit=L)
    with open(out / "windows.jsonl", "w", encoding="utf-8") as f:
        for k in keep:
            f.write(json.dumps({x: k[x] for x in ("name", "video_id", "label", "horizon")}) + "\n")

    chk = {"has_classifier_attr": head is not None}
    if head is not None:
        with torch.no_grad():
            zt = torch.from_numpy(Z[:64]).to(w.device)
            try:
                dtype = next(head.parameters()).dtype
                re_logit = head(zt.to(dtype)).float().cpu().numpy()
                chk["max_abs_diff_vs_forward"] = float(np.abs(re_logit - L[:64]).max())
            except Exception as e:  # noqa: BLE001
                chk["error"] = repr(e)
    if head is not None:
        np.savez(out / "head_params.npz", **{f"p{i:02d}": v.detach().float().cpu().numpy()
                                             for i, v in enumerate(head.state_dict().values())})
        chk["head_param_shapes"] = [list(v.shape) for v in head.state_dict().values()]
    json.dump(chk, open(out / "head_check.json", "w"))
    print("head check:", chk)
    print("saved", out)


if __name__ == "__main__":
    main()
