"""
lookahead_feasibility.py  (Stage 0b, LOCAL analysis)
=====================================================
Input: features.npz + windows.jsonl from lookahead_extract_features.py (frozen BADAS-Open).
Question: is the pooled vector 0.5s later predictable from the current window's vector, better than
just copying it? Held out BY VIDEO (GroupKFold), ridge on the change (z_next - z_now), so the
"copy" baseline is the zero prediction.

Reports, per (label, source horizon):
  * rel_mse = |z_hat - z_next|^2 / |z_now - z_next|^2      (<1: beats copying)
  * cosine(z_hat, z_next) vs cosine(z_now, z_next)
  * learning curve on 25/50/100% of training videos (is more data still helping?)
Payoff test (no network training): the real frozen crash head applied to z_now, to the predicted z_hat,
to their mix, and to the true +0.5s vector (oracle upper bound), AP on 1.5s windows, pos vs neg.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score as AP
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--balance", action="store_true", help="weight negative pairs so pos and neg contribute equally")
    args = ap.parse_args()
    d = Path(args.dir)
    f = np.load(d / "features.npz", allow_pickle=True)
    names, Z, LG = list(f["names"]), f["z"].astype(np.float64), f["logit"].astype(np.float64)
    meta = {json.loads(l)["name"]: json.loads(l) for l in open(d / "windows.jsonl", encoding="utf-8")}
    idx = {n: i for i, n in enumerate(names)}
    W = defaultdict(dict)  # (video, label) -> {horizon: row}
    for n in names:
        m = meta[n]
        W[(m["video_id"], m["label"])][m["horizon"]] = idx[n]

    pairs = []  # (i_now, i_next, video, label, src_horizon)
    for (vid, lab), hs in W.items():
        for h in (1.5, 1.0):
            if h in hs and round(h - 0.5, 1) in hs:
                pairs.append((hs[h], hs[round(h - 0.5, 1)], vid, lab, h))
    P = np.array([(p[0], p[1]) for p in pairs]); vids = np.array([p[2] for p in pairs])
    lab = np.array([p[3] for p in pairs]); src = np.array([p[4] for p in pairs])
    print(f"windows {len(names)} | pairs {len(pairs)}: "
          + ", ".join(f"label{l} h{h}={int(((lab == l) & (src == h)).sum())}" for l in (1, 0) for h in (1.5, 1.0)))

    X, Y = Z[P[:, 0]], Z[P[:, 1]]
    D = Y - X
    sw = np.where(lab == 0, (lab == 1).sum() / max((lab == 0).sum(), 1), 1.0) if args.balance else np.ones(len(lab))
    zhat = np.zeros_like(Y)
    for tr, te in GroupKFold(args.folds).split(X, D, vids):
        sc = StandardScaler().fit(X[tr])
        r = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(sc.transform(X[tr]), D[tr], sample_weight=sw[tr])
        zhat[te] = X[te] + r.predict(sc.transform(X[te]))
    print("\nheld-out (by video) prediction of the vector 0.5s later:")
    print("label  src  n     rel_mse  cos(z_hat,z_next)  cos(z_now,z_next)")
    for l in (1, 0):
        for h in (1.5, 1.0):
            m = (lab == l) & (src == h)
            if m.sum() < 5:
                continue
            rel = ((zhat[m] - Y[m]) ** 2).sum() / ((X[m] - Y[m]) ** 2).sum()
            print(f"  {'pos' if l else 'neg'}  {h}  {int(m.sum()):4d}  {rel:7.3f}  {cos(zhat[m], Y[m]).mean():.4f}"
                  f"            {cos(X[m], Y[m]).mean():.4f}")

    print("\nlearning curve (fraction of training videos used; all pairs held-out-by-video test):")
    uv = np.unique(vids)
    rng = np.random.default_rng(0)
    for frac in (0.25, 0.5, 1.0):
        num = den = 0.0
        for tr, te in GroupKFold(args.folds).split(X, D, vids):
            tv = np.unique(vids[tr]); rng.shuffle(tv)
            keep = np.isin(vids[tr], tv[: max(2, int(len(tv) * frac))])
            trk = tr[keep]
            sc = StandardScaler().fit(X[trk])
            r = RidgeCV(alphas=np.logspace(0, 4, 9)).fit(sc.transform(X[trk]), D[trk])
            ph = X[te] + r.predict(sc.transform(X[te]))
            num += ((ph - Y[te]) ** 2).sum(); den += ((X[te] - Y[te]) ** 2).sum()
        print(f"  {int(frac * 100):3d}% of train videos: rel_mse {num / den:.3f}")

    # ---- payoff test on 1.5s windows: the REAL frozen head applied to z_hat ----
    hp = np.load(d / "head_params.npz")
    P_ = [hp[k] for k in sorted(hp.files)]  # Linear w,b | LN w,b | Linear w,b | LN w,b | Linear w,b
    import torch
    T = lambda a: torch.from_numpy(a).float()
    def head(z):
        z = T(z)
        h = torch.nn.functional.linear(z, T(P_[0]), T(P_[1]))
        h = torch.nn.functional.layer_norm(torch.nn.functional.gelu(h), (768,), T(P_[2]), T(P_[3]))
        h = torch.nn.functional.linear(h, T(P_[4]), T(P_[5]))
        h = torch.nn.functional.layer_norm(torch.nn.functional.gelu(h), (768,), T(P_[6]), T(P_[7]))
        o = torch.nn.functional.linear(h, T(P_[8]), T(P_[9]))
        return (o[:, 1] - o[:, 0]).numpy()
    chk = np.abs(head(Z[:64]) - (LG[:64, 1] - LG[:64, 0])).max()
    print(f"\nlocal head reproduces stored logit diff: max abs diff {chk:.2e}")
    m15 = src == 1.5
    y = lab[m15]
    now = head(X[m15]); pred = head(zhat[m15]); true_next = head(Y[m15])
    print(f"1.5s windows: {int(y.sum())} pos / {int((1 - y).sum())} neg (pos and neg come from different pools)")
    print(f"  AP head(z_now)                  : {AP(y, now):.4f}   <- today's model, frozen BADAS-Open")
    print(f"  AP head(z_hat)   (predicted +0.5s): {AP(y, pred):.4f}")
    print(f"  AP head(true z at +0.5s) [oracle] : {AP(y, true_next):.4f}   <- upper bound for a perfect look-ahead")
    for g in (0.25, 0.5, 1.0):
        print(f"  AP head(z_now) + {g} * head(z_hat)   : {AP(y, now + g * pred):.4f}")


if __name__ == "__main__":
    main()
