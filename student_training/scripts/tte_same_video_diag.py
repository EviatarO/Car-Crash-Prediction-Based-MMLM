"""
tte_same_video_diag.py  (Stage 0a of the 1.5s look-ahead plan)
================================================================
Question: for 1.5s positives that a model misses at a fixed false-alarm rate, is the SAME source
video's 1.0s / 0.5s clip detected? If yes, the evidence exists later in the same video and a
look-ahead objective has room to help; if no, those crashes are not visible in any window.

The test manifests carry no source-video field, so clips are linked by frame matching: a 1.5s
window (ends 1.5s before the event) and a 1.0s window (ends 1.0s before) of the same video share
~1.6s of content. Two clips are matched when their thumbnails agree far better than the second-best
candidate (ratio test), restricted to the same class label and adjacent horizon.

Usage:
    python tte_same_video_diag.py --arm-glob-private ... (see --help)
Scores: mean of per-clip scores over the given (private, public) jsonl pairs, e.g. 5 seeds at epoch 1.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
THUMB = (24, 24)


def load_manifest(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def thumbs(frames_dir):
    fs = sorted(Path(frames_dir).glob("frame_*.jpg"))[:16]
    out = [np.asarray(Image.open(f).convert("L").resize(THUMB), dtype=np.float32) / 255.0 for f in fs]
    return np.stack(out).reshape(len(out), -1)


def pair_dist(a, b):
    # mean over frames of a of the min L1 distance to any frame of b (windows are time-shifted)
    d = np.abs(a[:, None, :] - b[None, :, :]).mean(-1)
    return float(d.min(1).mean())


def load_scores(paths):
    acc = {}
    for p in paths:
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l:
                r = json.loads(l)
                acc.setdefault(r["video_id"], []).append(float(r["score"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="jsonl score files (private and public, any number of seeds)")
    ap.add_argument("--fpr", type=float, default=0.10)
    ap.add_argument("--ratio", type=float, default=0.6, help="best/second-best distance ratio for a confident match")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = []
    for name, frames_root in (("test_manifest_hires.jsonl", ROOT / "dataset/test"),
                              ("test_manifest_public_hires.jsonl", ROOT / "dataset/test_public")):
        for r in load_manifest(ROOT / "dataset/manifests" / name):
            r["_dir"] = frames_root / r["frames_dir"]
            man.append(r)
    assert len(man) == 1344, len(man)

    # score = mean over score files of each clip (private files cover private ids, public cover public ids)
    sc = load_scores(args.scores)
    for r in man:
        r["score"] = sc[r["video_id"]]
    neg = np.array([r["score"] for r in man if r["event_occurs"] == 0])
    thr = float(np.quantile(neg, 1 - args.fpr))
    print(f"threshold at FPR {args.fpr:.0%} over all {len(neg)} negatives: {thr:.4f}")

    by = {(l, g): [r for r in man if r["event_occurs"] == l and r["group"] == g] for l in (0, 1) for g in (0, 1, 2)}
    T = {}
    for r in man:
        T[r["video_id"]] = thumbs(r["_dir"])

    def match(src_group, dst_group, label):
        res = {}
        cands = by[(label, dst_group)]
        for r in by[(label, src_group)]:
            ds = np.array([pair_dist(T[r["video_id"]], T[c["video_id"]]) for c in cands])
            o = np.argsort(ds)
            best, second = ds[o[0]], ds[o[1]]
            res[r["video_id"]] = (cands[o[0]], float(best), float(best / max(second, 1e-9)))
        return res

    out = {}
    for label in (1,):
        m21 = match(2, 1, label)   # 1.5s -> 1.0s
        m10 = match(1, 0, label)   # 1.0s -> 0.5s
        conf21 = {k: v for k, v in m21.items() if v[2] < args.ratio}
        print(f"1.5s->1.0s confident matches: {len(conf21)}/{len(m21)} (ratio<{args.ratio}); "
              f"median best-dist {np.median([v[1] for v in m21.values()]):.4f}")
        rows = []
        for r in by[(label, 2)]:
            vid = r["video_id"]
            c1, d1, q1 = m21[vid]
            c0, d0, q0 = m10[c1["video_id"]]
            rows.append(dict(id15=vid, s15=r["score"], id10=c1["video_id"], s10=c1["score"], q1=q1,
                             id05=c0["video_id"], s05=c0["score"], q0=q0))
        conf = [x for x in rows if x["q1"] < args.ratio and x["q0"] < args.ratio]
        print(f"full 1.5->1.0->0.5 chains with both links confident: {len(conf)}/{len(rows)}")
        missed = [x for x in conf if x["s15"] < thr]
        hit = [x for x in conf if x["s15"] >= thr]
        for nm, grp in (("1.5s MISSED", missed), ("1.5s DETECTED", hit)):
            if not grp:
                continue
            d10 = np.mean([x["s10"] >= thr for x in grp]); d05 = np.mean([x["s05"] >= thr for x in grp])
            either = np.mean([(x["s10"] >= thr) or (x["s05"] >= thr) for x in grp])
            print(f"  {nm:14s} n={len(grp):3d} | 1.0s clip detected {d10:.2f} | 0.5s clip detected {d05:.2f} "
                  f"| either {either:.2f} | mean score 1.5/1.0/0.5 = "
                  f"{np.mean([x['s15'] for x in grp]):.2f}/{np.mean([x['s10'] for x in grp]):.2f}/{np.mean([x['s05'] for x in grp]):.2f}")
        out = dict(threshold=thr, chains_confident=len(conf), chains_total=len(rows), rows=rows)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
