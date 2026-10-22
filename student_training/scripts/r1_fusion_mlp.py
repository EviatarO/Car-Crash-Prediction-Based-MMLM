"""
r1_fusion_mlp.py - a small TRAINED fusion module (stage 1b follow-up, 2026-10-21). Two training protocols, both scored on the validation windows only:
  A  train the module on the 600-window TRAIN sample's predictions, score it on validation (the classical protocol). Caveat: both models saw those windows in training
     (train AUC ~0.99), so their scores there are over-confident and the module learns the wrong trust.
  B  5-fold cross-validation on the validation windows, grouped by video (every window is scored by a module that never saw its video).
Modules: logistic regression (linear) and a small MLP (5 inputs -> 8 -> 8 -> 1, sklearn), the MLP over 5 seeds.
Inputs per window: crash log-odds, reasoning log-odds, their absolute values, their difference.   python r1_fusion_mlp.py --val <val predictions> --train <train600 predictions>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP

NEW = ("tte0.75", "tte1.25")


def load(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    pc = np.clip([float(r["a1_p"]) for r in rows], 1e-6, 1 - 1e-6)
    c, r = np.log(pc / (1 - pc)), np.array([float(x["s_first"]) for x in rows])
    return dict(X=np.stack([c, r, np.abs(c), np.abs(r), c - r], 1), y=np.array([int(x["label"]) for x in rows]), g=np.array([str(x["group"]) for x in rows]),
                v=np.array([str(x.get("video_key") or x["id"]) for x in rows]), c=c, r=r)


def make(kind, seed):
    if kind == "lr":
        return LogisticRegression(C=1.0, max_iter=3000)
    return MLPClassifier(hidden_layer_sizes=(8, 8), alpha=1e-2, max_iter=3000, random_state=seed, early_stopping=False)


def fit_score(kind, seed, Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    m = make(kind, seed).fit((Xtr - mu) / sd, ytr)
    z = (Xte - mu) / sd
    return m.decision_function(z) if kind == "lr" else np.log(np.clip(m.predict_proba(z)[:, 1], 1e-6, 1 - 1e-6) / np.clip(m.predict_proba(z)[:, 0], 1e-6, 1))


def report(name, s, V):
    core = ~np.isin(V["g"], NEW)
    return f"| {name} | {AUC(V['y'][core], s[core]):.4f} | {AUC(V['y'], s):.4f} | {AP(V['y'][core], s[core]):.4f} |"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--train", required=True)
    a = ap.parse_args()
    V, T = load(a.val), load(a.train)
    core = ~np.isin(V["g"], NEW)
    print(f"validation {len(V['y'])} windows ({int(core.sum())} core, {int(V['y'].sum())} crash); train sample {len(T['y'])} windows ({int(T['y'].sum())} crash)")
    print("train-sample AUC of the single scorers (over-fitted): crash %.4f, reasoning %.4f; validation core: crash %.4f, reasoning %.4f"
          % (AUC(T["y"], T["c"]), AUC(T["y"], T["r"]), AUC(V["y"][core], V["c"][core]), AUC(V["y"][core], V["r"][core])))
    rows = ["| module / protocol | val core AUC | val full AUC | val core AP |", "|---|---|---|---|",
            report("crash model alone", V["c"], V), report("reasoning model alone", V["r"], V)]
    zs = lambda x: (x - x.mean()) / x.std()                                                      # noqa: E731
    rows.append(report("fixed w = 0.6 (grid on validation, optimistic)", 0.6 * zs(V["c"]) + 0.4 * zs(V["r"]), V))
    for kind, nm, seeds in (("lr", "logistic regression", [0]), ("mlp", "small MLP 5-8-8-1 (mean of 5 seeds)", [0, 1, 2, 3, 4])):
        sA = [fit_score(kind, s, T["X"], T["y"], V["X"]) for s in seeds]
        rows.append(report(f"A: {nm}, trained on the TRAIN sample", np.mean([zs(x) for x in sA], 0), V))
        sB = []
        for s in seeds:
            out = np.zeros(len(V["y"]))
            for tr, te in GroupKFold(n_splits=5).split(V["X"], V["y"], V["v"]):
                out[te] = fit_score(kind, s, V["X"][tr], V["y"][tr], V["X"][te])
            sB.append(out)
        rows.append(report(f"B: {nm}, 5-fold CV on validation", np.mean([zs(x) for x in sB], 0), V))
    print("\n".join(rows))


if __name__ == "__main__":
    main()
