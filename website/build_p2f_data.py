"""
build_p2f_data.py -> p2f_data.js   (Experiments page, tab "Reasoning model - 3 formats": night run 2026-10-15/16, plan 2026-10-16_Plan-Night-Run-Phase2-Three-Formats)

The three fine-stage runs P2R (reason-first), P2V (verdict-first), P2F (facts-first) on the 524 validation windows (pre-alert removed), next to the
crash model. Read from outputs/r1_phase2f_2026-10-15/: phase2f_2026-10-15/<run>/results/{predictions,metrics,text_metrics}.json(l),
cards_<run>.jsonl (Gemini fact card of each run's reasoning), and outputs/teacher_dataset_2026-10/fact_cards_teacher.jsonl.
Per window: frames, teacher text + card, crash-model P, and per run: written verdict, P(yes) (the decision score: s_after; s_first for P2V),
reasoning, outcome, facts line (P2F), reasoning card, and flags:
  missed / false_alarm (written verdict vs the visible-hazard label), rescue (run right where the crash model is wrong at 0.5),
  lost (crash model right, run wrong), misaligned (verdict yes but the reasoning card's gap is not 'closing', or verdict no on a CRASH window
  while the reasoning says 'closing').
Summary tables: decision (AUC, confusion at 0.5, missed per TTE), missed crashes at equal false-alarm rates, rescues, verdict-vs-reasoning,
reasoning facts vs the teacher card.

    python build_p2f_data.py
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "outputs" / "r1_phase2f_2026-10-15"
RUNDIR = OUT / "phase2f_2026-10-15"
GRIDS = ROOT / "outputs" / "teacher_dataset_2026-10"
REL = "../outputs/teacher_dataset_2026-10"
RUNS = [("P2R", "Reason-first", "Reasoning -> Collision -> Outcome", "s_after"),
        ("P2V", "Verdict-first", "Collision -> Reasoning -> Outcome", "s_first"),
        ("P2F", "Facts-first", "Facts -> Reasoning -> Collision -> Outcome", "s_after")]
SLOTS = ["road_user", "side", "motion", "gap"]
TTE = ["tte0.5", "tte1.0", "tte1.5"]


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def sig(x):
    return 1 / (1 + math.exp(-x))


def main():
    teach = {r["id"]: r for r in jl(GRIDS / "fact_cards_teacher.jsonl")}
    P = {k: {r["id"]: r for r in jl(RUNDIR / k / "results" / "predictions.jsonl")} for k, *_ in RUNS}
    C = {k: {r["id"]: r for r in jl(OUT / f"cards_{k}.jsonl")} for k, *_ in RUNS}
    base = P["P2V"]
    ids = list(base)
    assert all(set(P[k]) == set(ids) for k in P)
    y = np.array([base[i]["label"] for i in ids])
    grp = np.array([base[i]["group"] for i in ids])
    a1 = np.array([base[i]["a1_p"] for i in ids])
    score = {"crash": a1}
    for k, _, _, s in RUNS:
        score[k] = np.array([sig(P[k][i][s]) for i in ids])

    # ---- decision table (P = 0.5 for every model)
    decision = []
    for key, label in [("crash", "Crash model")] + [(k, f"{k} {n}") for k, n, *_ in RUNS]:
        s = score[key]
        pred = s >= 0.5
        TP, FN = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
        FP, TN = int((pred & (y == 0)).sum()), int((~pred & (y == 0)).sum())
        miss = {g: float((s[(grp == g)] < 0.5).mean()) for g in TTE}
        decision.append({"key": key, "label": label, "auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
                         "TP": TP, "FN": FN, "FP": FP, "TN": TN, "recall": TP / (TP + FN), "specificity": TN / (TN + FP),
                         "precision": TP / max(1, TP + FP), "missed_all": FN / (TP + FN), "missed": miss,
                         "median_p_crash": float(np.median(s[y == 1])), "median_p_normal": float(np.median(s[y == 0]))})
    # ---- missed crashes at equal false-alarm rates (threshold on these windows: comparison only)
    equal_fa = []
    for fa in (0.05, 0.10, 0.16):
        row = {"fa": fa}
        for key in score:
            thr = np.quantile(score[key][y == 0], 1 - fa)
            row[key] = float((score[key][y == 1] < thr).mean())
        equal_fa.append(row)

    # ---- per-window rows and flags
    cm_ok = (a1 >= 0.5) == (y == 1)
    rows, counts = [], collections.Counter()
    for j, i in enumerate(ids):
        b = base[i]
        fd = b["frames_dir"]
        r = {"id": i, "group": b["group"], "label": int(b["label"]), "crash_p": round(float(a1[j]), 3), "crash_wrong": bool(not cm_ok[j]),
             "t_expl": b["teacher_explanation"], "t_out": b["teacher_outcome"], "t_card": {s: teach[i][s] for s in SLOTS} if i in teach else None,
             "grid": f"{REL}/review/{fd}.jpg" if (GRIDS / "review" / f"{fd}.jpg").exists() else None,
             "grid_large": f"{REL}/grid_large/{fd}.jpg" if (GRIDS / "grid_large" / f"{fd}.jpg").exists() else None, "runs": {}}
        for k, _, _, s in RUNS:
            p = P[k][i]
            v = 1 if p["gen_collision"] == 1 else 0
            ok = v == b["label"]
            card = C[k].get(i)
            closing = card is not None and card["gap"] == "closing"
            mis = card is not None and ((v == 1 and not closing) or (v == 0 and b["label"] == 1 and closing))
            e = {"v": v, "p": round(float(score[k][j]), 3), "expl": p["gen_explanation"], "out": p["gen_outcome"],
                 "facts": p.get("gen_facts"), "card": {s_: card[s_] for s_ in SLOTS} if card else None,
                 "missed": bool(b["label"] == 1 and v == 0), "fa": bool(b["label"] == 0 and v == 1),
                 "rescue": bool(ok and not cm_ok[j]), "lost": bool(not ok and cm_ok[j]), "misaligned": bool(mis)}
            r["runs"][k] = e
            for f in ("missed", "fa", "rescue", "lost", "misaligned"):
                counts[(k, f)] += e[f]
        rows.append(r)

    # ---- verdict vs reasoning (reasoning card gap) and rescues
    vr = []
    for k, n, *_ in RUNS:
        c = collections.Counter()
        for r in rows:
            e = r["runs"][k]
            if e["card"] is None:
                continue
            c[("crash" if r["label"] else "normal", "yes" if e["v"] else "no", "closing" if e["card"]["gap"] == "closing" else "other")] += 1
        vr.append({"key": k, "label": f"{k} {n}", "cells": {"|".join(kk): vv for kk, vv in c.items()}, "misaligned": counts[(k, "misaligned")]})
    rescues = []
    for k, n, *_ in RUNS:
        res = [r for r in rows if r["runs"][k]["rescue"]]
        los = [r for r in rows if r["runs"][k]["lost"]]
        cons = sum(1 for r in res if r["runs"][k]["card"] and ((r["runs"][k]["card"]["gap"] == "closing") == (r["label"] == 1)))
        rescues.append({"key": k, "label": f"{k} {n}", "rescue_crash": sum(r["label"] == 1 for r in res), "rescue_normal": sum(r["label"] == 0 for r in res),
                        "rescue_reason_consistent": cons, "lost_crash": sum(r["label"] == 1 for r in los), "lost_normal": sum(r["label"] == 0 for r in los)})
    # ---- reasoning facts vs the teacher card (crash / normal), with the guess baseline
    facts = []
    for slot in SLOTS:
        for name, sel in (("crash", lambda r: r["label"] == 1), ("normal", lambda r: r["label"] == 0)):
            k_rows = [r for r in rows if sel(r) and r["t_card"] and r["t_card"][slot] != "not_stated" and all(r["runs"][k]["card"] for k, *_ in RUNS)]
            maj = collections.Counter(r["t_card"][slot] for r in k_rows).most_common(1)[0]
            e = {"slot": slot, "set": name, "n": len(k_rows), "guess": maj[0], "guess_acc": maj[1] / len(k_rows)}
            for k, *_ in RUNS:
                e[k] = sum(r["runs"][k]["card"][slot] == r["t_card"][slot] for r in k_rows) / len(k_rows)
            facts.append(e)
    text = []
    for k, n, *_ in RUNS:
        tm = json.load(open(RUNDIR / k / "results" / "text_metrics.json", encoding="utf-8"))
        m = json.load(open(RUNDIR / k / "results" / "metrics.json", encoding="utf-8"))
        text.append({"key": k, "label": f"{k} {n}", "bert": tm["bertscore_f1_rescaled"]["model"], "bert_floor": tm["bertscore_f1_rescaled"]["floor"],
                     "gap_wrong": m["after"]["gap_wrong"], "retrieval": m["retrieval_at_10"]["top1"], "distinct": m["text"]["distinct_explanations"]})

    data = {"n": len(rows), "n_crash": int(y.sum()), "n_normal": int((y == 0).sum()), "runs": [{"key": k, "name": n, "order": o} for k, n, o, _ in RUNS],
            "decision": decision, "equal_fa": equal_fa, "verdict_reasoning": vr, "rescues": rescues, "facts": facts, "text": text,
            "counts": {f"{k}|{f}": v for (k, f), v in counts.items()}, "rows": rows}
    out = HERE / "p2f_data.js"
    out.write_text("window.P2F_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"p2f_data.js: {len(rows)} windows x {len(RUNS)} runs, {out.stat().st_size / 1e6:.2f} MB; grids {sum(bool(r['grid']) for r in rows)}")
    for d in decision:
        print(f"  {d['label']:24s} AUC {d['auc']:.3f} TP {d['TP']} FN {d['FN']} FP {d['FP']} TN {d['TN']}")
    print("  counts", dict(sorted(data["counts"].items())))


if __name__ == "__main__":
    main()
