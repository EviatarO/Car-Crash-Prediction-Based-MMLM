"""
r1_fusion.py - stage 1b of plan 2026-10-20: does averaging the crash model's score with the reasoning model's score rank better than either alone? VALIDATION ONLY.

Input: a decision-only / full predictions.jsonl of r1_eval_reasoning.py on the validation windows (fields label, group, video_key, a1_p = crash-model P, s_first = reasoning-model
log P(" yes") - log P(" no") at "Collision:"). Windows with no crash TTE group are the normal ones.
  (a) fixed weight   score = w * z(crash log-odds) + (1 - w) * z(reasoning log-odds), w in {0, 0.1, ..., 1}; z = standardised on the validation windows; w is ONE number for every window.
                     w is chosen on val core (the original 524 windows) and checked on val full (641, with the new 0.75 / 1.25 s windows).
  (b) learned gate   logistic regression on [crash lo, reasoning lo, |crash lo|, |reasoning lo|, crash lo - reasoning lo], 5-fold cross-validation grouped by video, so every
                     window is scored by a gate that never saw its video.
Gate (fixed before running): fusion is useful if val core AUC rises >= +0.005 over the better single model; the gate (b) beats fixed w only if its cross-validated AUC is >= +0.003 higher.
Also: the threshold at 90 % recall chosen on validation (stated as such), per-TTE AUC, and false alarms at that threshold.

    python r1_fusion.py --pred <predictions.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.model_selection import GroupKFold

NEW = ("tte0.75", "tte1.25")


def logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def zs(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


def fixed_fusion(c, r, w):
    return w * zs(c) + (1 - w) * zs(r)


def gate_cv(c, r, y, groups, k=5, seed=0):
    """Out-of-fold fused scores of the learned gate (logistic regression on 5 inputs, standardised inside each fold)."""
    X = np.stack([c, r, np.abs(c), np.abs(r), c - r], 1)
    out = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        out[te] = m.decision_function((X[te] - mu) / sd)
    return out


def select_rules(c, r, y, groups, k=5):
    """Per window pick ONE path instead of averaging (no weight to learn). Scores are put on a common scale by dividing each path's log-odds by its standard deviation.
    rule 'larger |u|'   : take the path whose scaled log-odds is further from 0 (the more confident one); the fused score is that path's scaled log-odds.
    rule 'calibrated'   : estimate, per path, P(path is right | its |log-odds|) with a logistic regression fitted on the OTHER folds (5-fold CV by video), then take the path
                          with the higher estimated probability of being right; the fused score is that path's scaled log-odds."""
    uc, ur = c / c.std(), r / r.std()
    larger = np.where(np.abs(uc) >= np.abs(ur), uc, ur)
    pick_crash = float((np.abs(uc) >= np.abs(ur)).mean())
    cal = np.zeros(len(y))
    chose_crash = np.zeros(len(y), dtype=bool)
    for tr, te in GroupKFold(n_splits=k).split(c, y, groups):
        pc = []
        for lo in (c, r):
            right = ((lo > 0) == (y == 1)).astype(int)
            X = np.log1p(np.abs(lo))[:, None]
            m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], right[tr])
            pc.append(m.predict_proba(X[te])[:, 1])
        use_c = pc[0] >= pc[1]
        chose_crash[te] = use_c
        cal[te] = np.where(use_c, uc[te], ur[te])
    return {"larger_abs": larger, "calibrated": cal}, {"larger_abs": pick_crash, "calibrated": float(chose_crash.mean())}


def at_recall(y, s, recall=0.9):
    """threshold chosen on THESE windows (validation) that keeps `recall` of the crash windows; returns (threshold, false-alarm rate)."""
    thr = float(np.quantile(s[y == 1], 1 - recall))
    return thr, float((s[y == 0] >= thr).mean())


def tte_auc(y, s, grp):
    out = {}
    for g in ("tte0.5", "tte0.75", "tte1.0", "tte1.25", "tte1.5"):
        m = (grp == g) | (y == 0)
        if (grp == g).any():
            out[g] = float(AUC(y[m], s[m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.pred, encoding="utf-8") if l.strip()]
    y = np.array([int(r["label"]) for r in rows])
    grp = np.array([str(r["group"]) for r in rows])
    vid = np.array([str(r.get("video_key") or r["id"]) for r in rows])
    c, r = logit([float(x["a1_p"]) for x in rows]), np.array([float(x["s_first"]) for x in rows])
    core = ~np.isin(grp, NEW)
    res = {"n_full": int(len(y)), "n_core": int(core.sum()), "single": {}, "fixed": {}, "gate": {}}
    for nm, s in (("crash", c), ("reasoning", r)):
        res["single"][nm] = {"core_auc": float(AUC(y[core], s[core])), "full_auc": float(AUC(y, s)), "core_ap": float(AP(y[core], s[core])), "full_ap": float(AP(y, s)),
                             "tte_auc": tte_auc(y, s, grp)}
    best_single = max(res["single"]["crash"]["core_auc"], res["single"]["reasoning"]["core_auc"])
    grid = []
    for w in np.round(np.arange(0, 1.0001, 0.1), 1):
        s = fixed_fusion(c, r, w)
        grid.append({"w": float(w), "core_auc": float(AUC(y[core], s[core])), "full_auc": float(AUC(y, s)), "core_ap": float(AP(y[core], s[core]))})
    wbest = max(grid, key=lambda g: (round(g["core_auc"], 6), -abs(g["w"] - 0.5)))["w"]
    sf = fixed_fusion(c, r, wbest)
    res["fixed"] = {"grid": grid, "w_best_on_val_core": wbest, "core_auc": float(AUC(y[core], sf[core])), "full_auc": float(AUC(y, sf)), "core_ap": float(AP(y[core], sf[core])),
                    "full_ap": float(AP(y, sf)), "tte_auc": tte_auc(y, sf, grp), "gain_core_auc_vs_best_single": float(AUC(y[core], sf[core]) - best_single)}
    sg = gate_cv(c, r, y, vid)
    res["gate"] = {"core_auc": float(AUC(y[core], sg[core])), "full_auc": float(AUC(y, sg)), "core_ap": float(AP(y[core], sg[core])), "full_ap": float(AP(y, sg)),
                   "tte_auc": tte_auc(y, sg, grp), "gain_core_auc_vs_best_single": float(AUC(y[core], sg[core]) - best_single),
                   "gain_core_auc_vs_fixed": float(AUC(y[core], sg[core]) - res["fixed"]["core_auc"])}
    sel, share = select_rules(c, r, y, vid)
    res["select"] = {}
    for nm, s in sel.items():
        res["select"][nm] = {"core_auc": float(AUC(y[core], s[core])), "full_auc": float(AUC(y, s)), "core_ap": float(AP(y[core], s[core])), "full_ap": float(AP(y, s)),
                             "tte_auc": tte_auc(y, s, grp), "share_crash_path_chosen": share[nm], "gain_core_auc_vs_best_single": float(AUC(y[core], s[core]) - best_single),
                             "gain_core_auc_vs_fixed": float(AUC(y[core], s[core]) - res["fixed"]["core_auc"])}
    thr = {}
    for nm, s in (("crash", c), ("reasoning", r), ("fixed", sf), ("gate", sg), ("select: larger |u|", sel["larger_abs"]), ("select: calibrated", sel["calibrated"])):
        t, fa = at_recall(y[core], s[core], 0.9)
        thr[nm] = {"threshold": t, "false_alarm_rate_core": fa, "note": "threshold chosen on the validation core windows to keep 90 % of the crash windows"}
    res["recall90"] = thr
    res["verdict"] = {"fixed_useful": res["fixed"]["gain_core_auc_vs_best_single"] >= 0.005, "gate_useful": res["gate"]["gain_core_auc_vs_best_single"] >= 0.005,
                      "gate_beats_fixed": res["gate"]["gain_core_auc_vs_fixed"] >= 0.003}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = [f"# Fusion of the crash model and the reasoning model (validation only; {res['n_core']} core / {res['n_full']} full windows)", "",
         "| scorer | val core AUC | val full AUC | val core AP | val full AP | " + " | ".join(f"AUC {g}" for g in ("tte0.5", "tte0.75", "tte1.0", "tte1.25", "tte1.5")) + " |",
         "|---|---|---|---|---|" + "---|" * 5]
    for nm, d in (("crash model", res["single"]["crash"]), ("reasoning model", res["single"]["reasoning"]), (f"fixed w = {wbest} (reasoning weight {1 - wbest:.1f})", res["fixed"]),
                  ("learned gate (5-fold CV by video)", res["gate"]), ("select the more confident path (larger |u|)", res["select"]["larger_abs"]),
                  ("select the path more likely right (calibrated, 5-fold CV)", res["select"]["calibrated"])):
        L.append(f"| {nm} | {d['core_auc']:.4f} | {d['full_auc']:.4f} | {d['core_ap']:.4f} | {d['full_ap']:.4f} | " + " | ".join(f"{d['tte_auc'].get(g, float('nan')):.3f}" for g in ("tte0.5", "tte0.75", "tte1.0", "tte1.25", "tte1.5")) + " |")
    L += ["", f"Gain on val core AUC over the better single model: fixed {res['fixed']['gain_core_auc_vs_best_single']:+.4f}, gate {res['gate']['gain_core_auc_vs_best_single']:+.4f}; gate vs fixed {res['gate']['gain_core_auc_vs_fixed']:+.4f}.",
          f"Gate (fixed in advance): fusion useful if >= +0.005 -> fixed {res['verdict']['fixed_useful']}, gate {res['verdict']['gate_useful']}; gate preferred over fixed only if >= +0.003 -> {res['verdict']['gate_beats_fixed']}.", "",
          f"Selection rules (one path per window, no weight): larger |u| {res['select']['larger_abs']['gain_core_auc_vs_best_single']:+.4f} over the better single model "
          f"({res['select']['larger_abs']['share_crash_path_chosen']:.0%} of windows use the crash path); calibrated {res['select']['calibrated']['gain_core_auc_vs_best_single']:+.4f} "
          f"({res['select']['calibrated']['share_crash_path_chosen']:.0%} crash path); vs the fixed weight: {res['select']['larger_abs']['gain_core_auc_vs_fixed']:+.4f} / {res['select']['calibrated']['gain_core_auc_vs_fixed']:+.4f}.", "",
          "False-alarm rate at 90 % recall (threshold chosen on the validation core windows):", "",
          "| scorer | false alarms |", "|---|---|"] + [f"| {k} | {v['false_alarm_rate_core']:.3f} |" for k, v in thr.items()]
    L += ["", "w grid (val core AUC): " + ", ".join(f"{g['w']:.1f}: {g['core_auc']:.4f}" for g in grid), ""]
    (out / "fusion.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
