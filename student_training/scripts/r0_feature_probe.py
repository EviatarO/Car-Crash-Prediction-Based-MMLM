"""
r0_feature_probe.py
===================
Reasoning-path Step 0b (plan rev. 3, risk R3 "facts not in the features"): BEFORE any projector
is trained, measure which facts a linear read-out can recover from the FROZEN A1-compress256
trunk's tokens. A projector cannot verbalise what the features do not hold.

Two probe families, per encoder layer (default 5/11/17/23 - Qwen3-VL's DeepStack taps 5/11/17
on a 24-layer ViT, 23 = the layer the crash head and the reasoning merger read):

  1. TOKEN POSITION - from one token's 1024-d feature alone, predict its own tubelet t (8),
     row h (16), column w (16), and coarse column third (left / centre / right).
     Why: V-JEPA2 uses 3D-RoPE (relative positions inside attention), so absolute position may
     be weakly present in each output token. The e4 resampler fed tokens with NO positional
     embedding, so if this probe is near chance, "left / right / ahead" was unrecoverable there.
     Token layout: index = tubelet*256 + row*16 + col (aa4_token_labels.py docstring).

  2. WINDOW FACTS - from a 4x4 spatially pooled grid (all tubelets averaged), predict facts the
     explanation must state, parsed from the V12 teacher fields of the same window:
     agent side (left / ahead / right), agent type (car / large / two-wheeler / pedestrian),
     agent colour, gap_trend, and event_occurs (sanity: the crash signal must be readable).
     Labels are AI-teacher text (Gemini), so they are noisy - the comparison that matters is
     probe vs majority-class and vs shuffled-label baselines, not the absolute number.

Runs locally (6 GB GPU is enough: frozen forward only, ~1-4 s/window).

Usage (from student_training/scripts):
  python r0_feature_probe.py --config ../configs/e4_stageA.yaml \
      --lora-adapter ../../outputs/a1_compress256/train/epoch_02/lora_adapter \
      --n-windows 600 --out-dir ../../outputs/r0_feature_probe
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))

from semsup_common import load_training_examples  # noqa: E402
from aa1_token_probe import load_probe_model  # noqa: E402

CAPTIONS_1761 = REPO / "outputs" / "semantic_captions" / "Caption_Train4500_Mixed_1761.jsonl"
V12_1761 = REPO / "outputs" / "semantic_captions" / "Caption_V12_Neutral_1761.jsonl"
N_TUB, GRID = 8, 16


# ----------------------------------------------------------------------------- labels
def _side(txt: str):
    t = (txt or "").lower()
    if t in ("", "none"):
        return None
    left, right = "left" in t, "right" in t
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    if not left and not right and re.search(r"ahead|front|same lane|ego lane", t):
        return "ahead"
    return None


def _atype(txt: str):
    t = (txt or "").lower()
    if t in ("", "none"):
        return None
    if re.search(r"pedestrian|person|walker|child", t):
        return "pedestrian"
    if re.search(r"motorcycl|motorbike|scooter|cyclist|bicycl|\bbike\b|moped", t):
        return "two_wheeler"
    if re.search(r"truck|bus|van|trailer|lorry|tanker", t):
        return "large"
    if re.search(r"sedan|suv|car|hatchback|pickup|coupe|taxi|minivan|wagon", t):
        return "car"
    return None


def _colour(txt: str):
    t = (txt or "").lower()
    for name, pat in (("white", r"\bwhite\b"), ("dark", r"\bblack\b|\bdark\b"),
                      ("silver_gray", r"silver|gr[ae]y"), ("red", r"\bred\b"), ("blue", r"\bblue\b")):
        if re.search(pat, t):
            return name
    return None


def window_labels(v12_row: dict) -> dict:
    return {
        "side": _side(v12_row.get("agent_position")),
        "agent_type": _atype(v12_row.get("primary_agent")),
        "colour": _colour(v12_row.get("primary_agent")),
        "gap_trend": v12_row.get("gap_trend") or None,
        "event_occurs": str(v12_row.get("event_occurs")),
    }


# ----------------------------------------------------------------------------- probes
def softmax_probe(xtr, ytr, xva, yva, n_cls, steps=400, lr=1e-2, wd=1e-4, device="cpu"):
    """Full-batch multinomial logistic regression (LayerNorm-no-affine + Linear) in torch.
    Returns val accuracy and val balanced accuracy."""
    xtr = torch.as_tensor(xtr, dtype=torch.float32, device=device)
    xva = torch.as_tensor(xva, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, dtype=torch.long, device=device)
    lin = torch.nn.Linear(xtr.shape[1], n_cls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=wd)
    # class-balanced CE so a majority class cannot carry the score
    w = torch.bincount(ytr_t, minlength=n_cls).float().clamp(min=1)
    w = (w.sum() / w) / n_cls
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(lin(F.layer_norm(xtr, xtr.shape[1:])), ytr_t, weight=w)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = lin(F.layer_norm(xva, xva.shape[1:])).argmax(1).cpu().numpy()
    return _acc(pred, yva, n_cls)


def _acc(pred, y, n_cls):
    y = np.asarray(y)
    acc = float((pred == y).mean())
    per = [float((pred[y == c] == c).mean()) for c in range(n_cls) if (y == c).any()]
    return acc, float(np.mean(per))


def majority_baseline(ytr, yva, n_cls):
    maj = np.bincount(ytr, minlength=n_cls).argmax()
    return _acc(np.full(len(yva), maj), yva, n_cls)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lora-adapter", required=True)
    ap.add_argument("--lora-target-modules", default="query,key,value")
    ap.add_argument("--layers", default="5,11,17,23")
    ap.add_argument("--n-windows", type=int, default=600)
    ap.add_argument("--tokens-per-window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    v12 = {}
    for l in open(V12_1761, encoding="utf-8"):
        r = json.loads(l)
        v12[r["frames_dir"]] = r
    examples = [e for e in load_training_examples(captions_path=CAPTIONS_1761) if e["frames_dir"] in v12]
    rnd = random.Random(args.seed)
    examples = rnd.sample(examples, min(args.n_windows, len(examples)))
    vids = sorted({e["video_id"] for e in examples})
    rnd.shuffle(vids)
    val_vids = set(vids[:max(1, int(len(vids) * args.val_frac))])
    print(f"[data] {len(examples)} windows, {len(vids)} videos, {len(val_vids)} val videos")

    badas, captured = load_probe_model(cfg, "compress256", layers, args.lora_adapter,
                                       args.lora_target_modules)
    rng = np.random.default_rng(args.seed)
    tok = {L: {"x": [], "t": [], "h": [], "w": [], "val": []} for L in layers}
    win = {L: [] for L in layers}
    meta = []
    t0 = time.time()
    for i, ex in enumerate(examples):
        with torch.no_grad():
            badas.forward(ex["frame_paths"])
        is_val = ex["video_id"] in val_vids
        idx = rng.choice(N_TUB * GRID * GRID, size=args.tokens_per_window, replace=False)
        for L in layers:
            f = captured[L].float()                                         # (2048, 1024)
            tok[L]["x"].append(f[idx].half().cpu().numpy())
            tok[L]["t"].append(idx // 256)
            tok[L]["h"].append((idx % 256) // 16)
            tok[L]["w"].append(idx % 16)
            tok[L]["val"].append(np.full(len(idx), is_val))
            g = f.view(N_TUB, GRID, GRID, -1).mean(0).permute(2, 0, 1)      # (1024, 16, 16)
            pooled = F.adaptive_avg_pool2d(g, 4).permute(1, 2, 0).reshape(-1)  # (16*1024,)
            win[L].append(pooled.half().cpu().numpy())
        meta.append({"video_id": ex["video_id"], "frames_dir": ex["frames_dir"], "val": is_val,
                     **window_labels(v12[ex["frames_dir"]])})
        if (i + 1) % 25 == 0 or i == len(examples) - 1:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(examples)}] {el / (i + 1):.2f}s/window  "
                  f"eta={(len(examples) - i - 1) * el / (i + 1) / 60:.1f}min", flush=True)
    del badas, captured
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    report = {"n_windows": len(examples), "layers": layers, "token_position": {}, "window_facts": {}}
    # ---- 1. token position
    for L in layers:
        X = np.concatenate(tok[L]["x"]).astype(np.float32)
        val = np.concatenate(tok[L]["val"])
        res = {}
        for key, n_cls, y in (("t", 8, np.concatenate(tok[L]["t"])),
                              ("h", 16, np.concatenate(tok[L]["h"])),
                              ("w", 16, np.concatenate(tok[L]["w"])),
                              ("w_third", 3, np.minimum(np.concatenate(tok[L]["w"]) * 3 // 16, 2))):
            acc, bal = softmax_probe(X[~val], y[~val], X[val], y[val], n_cls, device=dev)
            res[key] = {"val_acc": round(acc, 4), "chance": round(1 / n_cls, 4)}
        report["token_position"][str(L)] = res
        print(f"[pos L{L}] " + "  ".join(f"{k}={v['val_acc']:.3f}(chance {v['chance']:.3f})" for k, v in res.items()))
    # ---- 2. window facts
    for fact in ("side", "agent_type", "colour", "gap_trend", "event_occurs"):
        keep = [i for i, m in enumerate(meta) if m[fact] is not None]
        classes = sorted({meta[i][fact] for i in keep})
        cls_idx = {c: j for j, c in enumerate(classes)}
        y = np.array([cls_idx[meta[i][fact]] for i in keep])
        val = np.array([meta[i]["val"] for i in keep])
        counts = {c: int((y == j).sum()) for c, j in cls_idx.items()}
        fact_rep = {"classes": counts, "n": len(keep)}
        if len(classes) < 2 or val.sum() == 0 or (~val).sum() == 0:
            report["window_facts"][fact] = {**fact_rep, "skipped": "too few classes/rows"}
            continue
        maj = majority_baseline(y[~val], y[val], len(classes))
        fact_rep["majority_bal_acc"] = round(maj[1], 4)
        shuf = np.random.default_rng(1).permutation(y[~val])
        for L in layers:
            X = np.stack([win[L][i] for i in keep]).astype(np.float32)
            acc, bal = softmax_probe(X[~val], y[~val], X[val], y[val], len(classes),
                                     steps=300, wd=1e-2, device=dev)
            _, bal_shuf = softmax_probe(X[~val], shuf, X[val], y[val], len(classes),
                                        steps=300, wd=1e-2, device=dev)
            fact_rep[f"L{L}"] = {"bal_acc": round(bal, 4), "shuffled_bal_acc": round(bal_shuf, 4)}
        report["window_facts"][fact] = fact_rep
        print(f"[fact {fact}] {counts}  majority={maj[1]:.3f}  " +
              "  ".join(f"L{L}={fact_rep[f'L{L}']['bal_acc']:.3f}/shuf{fact_rep[f'L{L}']['shuffled_bal_acc']:.3f}" for L in layers))

    (out / "probe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(out / "windows.jsonl", "w", encoding="utf-8") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"[done] {out / 'probe_report.json'}")


if __name__ == "__main__":
    main()
