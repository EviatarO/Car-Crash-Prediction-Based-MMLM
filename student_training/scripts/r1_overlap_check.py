"""
r1_overlap_check.py - is any Nexar TEST clip a near-duplicate of a TRAINING clip? (plan 2026-10-18, Part A: the contamination check)

The video ids of the two sets live in separate namespaces (train 00000..., test_00001...), so comparing ids proves nothing. This compares CONTENT: the last frame of
every local test window (private 677 + public 667) against the last frame of every local training window (dataset/train/*_hires*), each reduced to a 32x18 grayscale
thumbnail and compared by normalised correlation. For calibration it also prints how similar two DIFFERENT training videos get (the same dashcam, road and car
hood make unrelated clips look alike). Test windows whose best match is above the 99.9th percentile of that calibration set, or above 0.99, are listed.

    python r1_overlap_check.py [--max-show 15]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def thumb(p):
    im = cv2.imread(str(p), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if im is None:
        return None
    t = cv2.resize(im, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 1e-6 else None


def collect(dirs, name):
    keys, mats = [], []
    for d in dirs:
        t = thumb(d / "frame_00016.jpg")
        if t is not None:
            keys.append(d.name)
            mats.append(t)
    print(f"[{name}] {len(keys)} windows", flush=True)
    return keys, np.stack(mats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-show", type=int, default=15)
    a = ap.parse_args()
    tr_dirs = sorted(d for d in (ROOT / "dataset" / "train").iterdir() if d.is_dir() and "_hires" in d.name)
    te_dirs = sorted(d for d in (ROOT / "dataset" / "test").iterdir() if d.is_dir() and d.name.endswith("_hires"))
    kt, Mt = collect(tr_dirs, "train")
    ke, Me = collect(te_dirs, "test (private)")
    # the public windows have no local frame folders (extracted on the pod), but their mp4s are local: decode the window's last frame from the video
    vids = Path(r"C:\Users\eviatar.ohayon\Ramon Space\PycharmProjects\Thesis\Data-Centric-Crash-Prediction-Using-3LC-and-MViT\src\Nexar_DataSet\test")
    pub_keys, pub_mats = [], []
    for l in open(ROOT / "dataset" / "manifests" / "test_manifest_public.jsonl", encoding="utf-8"):
        r = json.loads(l)
        cap = cv2.VideoCapture(str(vids / f"{r['video_id']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, r["frame_indices"][-1])
        ok, fr = cap.read()
        cap.release()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        t = cv2.resize(g, (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        t -= t.mean()
        n = np.linalg.norm(t)
        if n > 1e-6:
            pub_keys.append(f"public_{r['video_id']}")
            pub_mats.append(t / n)
    print(f"[test (public)] {len(pub_keys)} windows decoded from the local mp4s", flush=True)
    ke, Me = ke + pub_keys, np.concatenate([Me, np.stack(pub_mats)])
    # calibration: each train window against train windows of a DIFFERENT video
    vid = np.array([k.split("_")[0] for k in kt])
    best_tt = np.empty(len(kt), np.float32)
    for i in range(0, len(kt), 512):
        s = Mt[i:i + 512] @ Mt.T
        s[vid[i:i + 512, None] == vid[None, :]] = -1                  # exclude the same video (its own windows overlap)
        best_tt[i:i + 512] = s.max(1)
    cal = float(np.percentile(best_tt, 99.9))
    s_te = Me @ Mt.T
    best, arg = s_te.max(1), s_te.argmax(1)
    print(f"\ntrain -> other train video, best match correlation: median {np.median(best_tt):.3f}, 99th {np.percentile(best_tt, 99):.3f}, 99.9th {cal:.3f}, max {best_tt.max():.3f}")
    print(f"test  -> train,             best match correlation: median {np.median(best):.3f}, 99th {np.percentile(best, 99):.3f}, max {best.max():.3f}")
    thr = max(cal, 0.99)
    hit = np.argsort(-best)
    flagged = [i for i in hit if best[i] >= thr]
    print(f"\ntest windows with a best match >= {thr:.3f} (the calibration 99.9th percentile or 0.99): {len(flagged)} of {len(ke)}")
    for i in hit[:a.max_show]:
        print(f"  {ke[i]:20s} <-> {kt[arg[i]]:28s} corr {best[i]:.4f}{'  FLAGGED' if best[i] >= thr else ''}")
    print("\nVerdict:", "no near-duplicates found" if not flagged else f"{len(flagged)} windows to look at by eye")


if __name__ == "__main__":
    main()
