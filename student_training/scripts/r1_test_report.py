"""
r1_test_report.py - P2V vs the crash model on the Nexar test halves (2026-10-18): private, public and pooled (1,344 windows).

Reads the decision-only predictions.jsonl of r1_eval_reasoning.py --no-ref (fields label, tte, a1_p = crash model P, s_first = P2V log-odds at "Collision:").
Per set and model: AP, AUC, AP per TTE (crash windows of that TTE vs the normal windows of the same TTE group), Kaggle mAP (their mean, the BADAS papers'
metric), false-alarm rate at 85% recall. Paired bootstrap (the same resampled windows for both models, 2,000 draws, seed 0) of P2V - crash model for
AP, AUC and Kaggle mAP. Caveat printed with the result: the test clips come from fewer source videos than clips, so window-level intervals are a bit narrow.

    python r1_test_report.py --private <predictions.jsonl> [--public <predictions.jsonl>] [--out <dir>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

TTES = (0.5, 1.0, 1.5)


def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    y = np.array([r["label"] for r in rows]); t = np.array([float(r["tte"]) for r in rows])
    crash = np.array([r["a1_p"] for r in rows], float); p2v = np.array([r["s_first"] for r in rows], float)
    return {"y": y, "t": t, "crash": crash, "P2V": p2v, "n": len(rows)}


def metrics(y, t, s):
    per = [AP(y[t == k], s[t == k]) for k in TTES]
    thr = np.quantile(s[y == 1], 0.15)                       # 85% of the crash windows score at or above it
    return {"AP": AP(y, s), "AUC": AUC(y, s), "AP_tte": per, "kaggle_mAP": float(np.mean(per)), "FPR_at_85recall": float((s[y == 0] >= thr).mean())}


def boot(y, t, a, b, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    out = {"AP": [], "AUC": [], "kaggle_mAP": []}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yy, tt = y[i], t[i]
        if yy.min() == yy.max() or any(len(set(yy[tt == k])) < 2 for k in TTES):
            continue
        ma, mb = metrics(yy, tt, a[i]), metrics(yy, tt, b[i])
        for k in out:
            out[k].append(ma[k] - mb[k])
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private", required=True)
    ap.add_argument("--public", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sets = {"private": load(a.private)}
    if a.public:
        sets["public"] = load(a.public)
        sets["pooled"] = {k: np.concatenate([sets["private"][k], sets["public"][k]]) for k in ("y", "t", "crash", "P2V")}
        sets["pooled"]["n"] = sets["private"]["n"] + sets["public"]["n"]
    rep, lines = {}, ["| Set | n | Model | AP | AUC | Kaggle mAP | AP at TTE 0.5 / 1.0 / 1.5 s | False alarms at 85% recall |", "|---|---|---|---|---|---|---|---|"]
    for name, d in sets.items():
        rep[name] = {"n": int(d["n"])}
        for m in ("crash", "P2V"):
            r = metrics(d["y"], d["t"], d[m])
            rep[name][m] = r
            lines.append(f"| {name} | {d['n']} | {'crash model' if m == 'crash' else 'P2V'} | {r['AP']:.4f} | {r['AUC']:.4f} | {r['kaggle_mAP']:.4f} | "
                         + " / ".join(f"{x:.3f}" for x in r["AP_tte"]) + f" | {r['FPR_at_85recall']:.3f} |")
        rep[name]["delta_P2V_minus_crash"] = boot(d["y"], d["t"], d["P2V"], d["crash"])
    lines += ["", "**P2V - crash model, paired bootstrap (mean [95% interval]):**", ""]
    for name in sets:
        b = rep[name]["delta_P2V_minus_crash"]
        lines.append(f"- {name}: " + "; ".join(f"{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in b.items()))
    lines += ["", "Test noise only: a single P2V training seed. Window-level intervals are a bit narrow (the test clips come from fewer source videos)."]
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        o = Path(a.out); o.mkdir(parents=True, exist_ok=True)
        (o / "test_report.md").write_text(txt + "\n", encoding="utf-8")
        (o / "test_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"-> {o / 'test_report.md'}")


if __name__ == "__main__":
    main()
