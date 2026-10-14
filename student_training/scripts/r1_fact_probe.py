"""
r1_fact_probe.py - are the facts of the scene in our cached tokens? (plan 2026-10-13, Step 1.2)

A small classifier (2 layers) is TRAINED, for each fact, to predict the fact from the cached tokens of a window. No prompt, no language
model, no projector. Labels = the 4 facts extracted from the TEACHER explanation by r1_fact_eval.facts (type, side, motion, gap).
Train windows (carved 85/15 by video into fit / early-stop) -> tested on the validation windows.

Token summary per window (fp16 memmap -> one 16k vector, cached in <out>/pooled_<input>.npy):
  last   : last-layer tokens, 4x4 spatial grid mean over time and 4x4 cells   (16 x 1024)
  l11    : layer-11 tokens, same pooling                                         (16 x 1024)
  attn   : attention-weighted mean of the last-layer tokens (crash-head attention, 1024) concatenated to 'last'

Reading (per fact, test balanced accuracy vs the majority-class baseline, and vs the language model's own text agreement from r1_fact_eval):
  probe ~ baseline  -> the fact is not (linearly/shallowly) readable from these tokens: add information (layer 11, attention, object crops).
  probe >> baseline but the model's text is at the baseline -> the information is there; the text training does not extract it.

    python r1_fact_probe.py --cache-dir <cache> --out-dir <dir> [--inputs last,l11,attn] [--runs name=predictions.jsonl ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from r1_fact_eval import FACTS, facts  # noqa: E402

MANIFEST = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl"


def pooled_projector(cache_dir, source, ckpt, ids, rows, out, tag, dev):
    """Probe the PROJECTOR OUTPUT (what Qwen receives): merger(last-layer tokens) = (512, 2560) -> mean over time, 8x8 grid pooled to 4x4 -> 16 x 2560."""
    import sys as _sys
    _sys.path.insert(0, str(HERE.parent / "models"))
    from r1_bridge import Merger
    f = Path(out) / f"pooled_proj_{tag}.npy"
    if f.exists():
        return np.load(f)
    m = Merger(1024, 2560).to(dev).eval()
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["merger"])
    last = np.load(Path(cache_dir) / f"features_{source}.npy", mmap_mode="r")
    res = []
    with torch.no_grad():
        for k in range(0, len(ids), 8):
            x = torch.from_numpy(np.stack([np.asarray(last[rows[i]]) for i in ids[k:k + 8]])).float().to(dev)
            y = m(x).view(-1, 8, 8, 8, 2560).mean(1)                                  # (B, 8, 8, 2560) after the time mean
            y = y.view(-1, 4, 2, 4, 2, 2560).mean((2, 4)).reshape(y.shape[0], -1)        # (B, 4*4*2560)
            res.append(y.cpu().numpy().astype(np.float32))
            if (k // 8 + 1) % 100 == 0:
                print(f"  [pool proj {tag}] {k + 8}/{len(ids)}", flush=True)
    arr = np.concatenate(res)
    np.save(f, arr)
    return arr


def pooled(cache_dir, source, kind, ids, rows, out):
    f = Path(out) / f"pooled_{kind}.npy"
    if f.exists():
        return np.load(f)
    d = Path(cache_dir)
    last = np.load(d / f"features_{source}.npy", mmap_mode="r")
    l11 = np.load(d / f"features_{source}_l11.npy", mmap_mode="r") if kind == "l11" else None
    att = np.load(d / f"attn_{source}.npy", mmap_mode="r") if kind == "attn" else None
    res = []
    for k, i in enumerate(ids):
        r = rows[i]
        src = l11 if kind == "l11" else last
        x = torch.from_numpy(np.asarray(src[r])).float().view(8, 16, 16, 1024)
        g = x.mean(0).view(4, 4, 4, 4, 1024).mean((1, 3)).reshape(16 * 1024)           # (16 cells x 1024)
        if kind == "attn":
            a = torch.from_numpy(np.asarray(att[r])).float().clamp_min(0)
            w = (a / a.sum().clamp_min(1e-9)).view(2048, 1)
            g = torch.cat([g, (torch.from_numpy(np.asarray(last[r])).float() * w).sum(0)])      # + attention-weighted mean token
        res.append(g.numpy().astype(np.float32))
        if (k + 1) % 500 == 0:
            print(f"  [pool {kind}] {k + 1}/{len(ids)}", flush=True)
    arr = np.stack(res)
    np.save(f, arr)
    return arr


def fit_probe(Xtr, ytr, Xva, yva, n_cls, dev, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    f = lambda x: ((x - mu) / sd).to(dev)  # noqa: E731
    net = nn.Sequential(nn.Dropout(0.3), nn.Linear(Xtr.shape[1], 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, n_cls)).to(dev)
    cnt = torch.bincount(ytr, minlength=n_cls).float()
    w = (cnt.sum() / (n_cls * cnt.clamp_min(1))).to(dev)                                 # class-balanced
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr_d, ytr_d, Xva_d = f(Xtr), ytr.to(dev), f(Xva)
    best, best_state, bad = 1e9, None, 0
    for ep in range(200):
        net.train()
        perm = torch.randperm(len(Xtr_d), device=dev)
        for s in range(0, len(perm), 128):
            b = perm[s:s + 128]
            loss = nn.functional.cross_entropy(net(Xtr_d[b]), ytr_d[b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(nn.functional.cross_entropy(net(Xva_d), yva.to(dev), weight=w))
        if vl < best - 1e-4:
            best, bad, best_state = vl, 0, {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(best_state)
    return net, f


def bal_acc(y, p, n_cls):
    r = [float((p[y == c] == c).mean()) for c in range(n_cls) if (y == c).any()]
    return float(np.mean(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--source", default="nexar_reasoning")
    ap.add_argument("--inputs", default="last")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--projector-ckpt", default=None, help="probe the output of this projector checkpoint (best.pt with a 'merger' entry) instead of the encoder tokens")
    ap.add_argument("--proj-tag", default="proj")
    ap.add_argument("--runs", nargs="*", default=[], help="name=predictions.jsonl of the language-model runs, for the side-by-side with their text agreement")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    idx = {r["id"]: r for r in map(json.loads, open(Path(args.cache_dir) / f"index_{args.source}.jsonl", encoding="utf-8"))}
    man = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    man = [m for m in man if m["id"] in idx]
    rows = {m["id"]: idx[m["id"]]["row"] for m in man}
    ids = [m["id"] for m in man]
    lab = {m["id"]: facts(m["gt"]["explanation"]) for m in man}
    split = {m["id"]: m["split"] for m in man}
    video = {m["id"]: m["video_key"] for m in man}
    dev = torch.device(args.device)
    lm_agree = {}
    if args.runs:
        from r1_fact_eval import agreement
        for spec in args.runs:
            nm, pth = spec.split("=", 1)
            lm_agree[nm] = agreement([json.loads(l) for l in open(pth, encoding="utf-8")])["overall"]
    result = {}
    kinds = ["proj"] if args.projector_ckpt else args.inputs.split(",")
    for kind in kinds:
        X = torch.from_numpy(pooled_projector(args.cache_dir, args.source, args.projector_ckpt, ids, rows, out, args.proj_tag, dev) if kind == "proj"
                             else pooled(args.cache_dir, args.source, kind, ids, rows, out))
        result[kind] = {}
        for fact in FACTS:
            known = [i for i in ids if lab[i][fact] is not None]
            cnt = Counter(lab[i][fact] for i in known if split[i] == "train")
            classes = sorted(c for c, n in cnt.items() if n >= 15)                         # rare classes dropped (too few to learn)
            if len(classes) < 2:
                print(f"{kind:5s} {fact:7s} skipped: fewer than 2 classes with >= 15 training windows", flush=True)
                continue
            keep = [i for i in known if lab[i][fact] in classes]
            ci = {c: k for k, c in enumerate(classes)}
            tr = [i for i in keep if split[i] == "train"]
            va = [i for i in keep if split[i] == "val"]
            vids = sorted({video[i] for i in tr})
            rng = np.random.default_rng(0)
            stop_v = set(rng.choice(vids, size=max(1, int(0.15 * len(vids))), replace=False))
            fit = [i for i in tr if video[i] not in stop_v]
            stop = [i for i in tr if video[i] in stop_v]
            g = lambda L: (X[[ids.index(i) for i in L]], torch.tensor([ci[lab[i][fact]] for i in L]))  # noqa: E731
            (Xf, yf), (Xs, ys), (Xv, yv) = g(fit), g(stop), g(va)
            net, f = fit_probe(Xf, yf, Xs, ys, len(classes), dev)
            with torch.no_grad():
                pv = net(f(Xv)).argmax(1).cpu().numpy()
            yv = yv.numpy()
            maj = Counter(yv.tolist()).most_common(1)[0][1] / len(yv)
            result[kind][fact] = {"classes": classes, "n_train": len(fit), "n_val": len(va), "acc": float((pv == yv).mean()),
                                  "balanced_acc": bal_acc(yv, pv, len(classes)), "chance_balanced": 1 / len(classes), "majority_acc": maj}
            r = result[kind][fact]
            print(f"{kind:5s} {fact:7s} classes {len(classes)}  probe acc {r['acc']:.3f} bal {r['balanced_acc']:.3f} (chance bal {r['chance_balanced']:.2f}, majority acc {maj:.2f})"
                  + "".join(f" | {nm} text agree {a[fact]['agree']:.3f}" for nm, a in lm_agree.items()), flush=True)
    (out / "fact_probe.json").write_text(json.dumps({"probe": result, "lm_text_agreement": lm_agree}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
