"""
r1_seed_report.py - stage 4 report (plan 2026-10-22_Plan-Stage4-Final-Seeds-rev3): five seeds of the final recipe, the test sets once, decision + reasoning text.
Local, no GPU (BERTScore runs on the CPU). Written and committed BEFORE the test numbers are read; the rules R1-R5 below are the plan's, fixed in advance.

Inputs  <root>/pod1/live, <root>/pod2/live   the pods' synced folders (python r1_hf_sync.py pull --dir <root>/podN/live --repo eviatarO-org/checkpoints --prefix stage4_2026-10-22/podN/live)
          S<k>/{P1,P1fixR,P2V}/train_log.jsonl, eval/S<k>/{val,train600,val_text,test_private,test_public,ground_private,ground_public}/
        seed 0 = the 2026-10-21 control run: training logs + validation evaluations from outputs/r1_b4_2026-10-21/ (CT), its test pass from pod 1 (eval/S0/)
        references: dataset/manifests/r1_nexar_reasoning_test{,_public}_refs.jsonl (r1_test_refs.py); fact cards in <root>/cards/ (made by `prepare`)
Scores  crash model: a1_p (P, yes at P >= 0.5); reasoning model: s_first = log P(" yes") - log P(" no") at "Collision:" (yes at >= 0, i.e. P = 0.5);
        ensemble = the mean of the seeds' s_first (log-odds). AP / AUC are threshold-free; every thresholded number states its threshold.

  prepare   join each seed's generated test answers with the references (agreed windows only) and read the fact cards that are missing (Gemini Flex, parallel)
  report    tables + rules + figures -> <out>/report.md, metrics.json, figures/*.png

    python r1_seed_report.py prepare --root outputs/stage4_2026-10-22 [--no-cards]
    python r1_seed_report.py report --root outputs/stage4_2026-10-22 [--boot 1000] [--no-bert]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from r1_full_report import SLOTS, TTES, Split, boot_delta, card_acc, conf, jl, metrics  # noqa: E402
from r1_phase_report import rouge_l, token_f1  # noqa: E402
from r1_review_reasoning import bleu4  # noqa: E402

B4 = ROOT / "outputs" / "r1_b4_2026-10-21"
REFS = {"private": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_test_refs.jsonl",
        "public": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_test_public_refs.jsonl"}
P2V_TEST = {"private": ROOT / "outputs" / "r1_test_eval_2026-10-17" / "test_eval_2026-10-17" / "P2V" / "results" / "predictions.jsonl",
            "public": ROOT / "outputs" / "r1_test_eval_public_2026-10-18" / "test_eval_public_2026-10-18" / "P2V" / "results" / "predictions.jsonl"}
TEACH_VAL_CARDS = ROOT / "outputs" / "teacher_dataset_2026-10" / "fact_cards_teacher.jsonl"
HALVES = ("private", "public")
VAL_TTES = (0.5, 0.75, 1.0, 1.25, 1.5)
BADAS2 = {"kaggle_mAP": 0.940, "AP_tte": {0.5: 0.943, 1.0: 0.957, 1.5: 0.921}}      # paper numbers only (no predictions): R4
# rules fixed by the plan before any test number exists
R1_GAIN, R1_MIN_ABOVE, R3_MARGIN = 0.010, 4, 0.008
NEG_OUTCOME = r"no collision|without|not collid|avoid|safe|clear|no contact|no impact|passes|remain|neither|\bnor\b|does not|doesn't"   # = r1_eval_reasoning.py


def rule_r1(seed_aps, crash_ap):
    """R1: (established, mean AP gain, seeds above, seeds needed): mean gain >= +0.010 AND >= 4 of 5 seeds above the crash model (n - 1 when fewer seeds finished)."""
    gain = float(np.mean([v - crash_ap for v in seed_aps]))
    above = sum(v > crash_ap for v in seed_aps)
    need = R1_MIN_ABOVE if len(seed_aps) >= 5 else max(1, len(seed_aps) - 1)
    return gain >= R1_GAIN and above >= need, gain, above, need


def rule_r3(ens_ap, seed_aps):
    """R3: the ensemble is the headline only if its AP >= the best seed AND >= the seed mean + 0.008."""
    return ens_ap >= max(seed_aps) and ens_ap >= float(np.mean(seed_aps)) + R3_MARGIN


def outcome_says_collision(text):
    import re
    t = (text or "").lower()
    if re.search(NEG_OUTCOME, t):
        return False
    return bool(re.search(r"collid|collision|crash|\bhit|strike|contact|impact|rear-end|runs? into|\bram(s|med)?\b|smash|plow", t))


# ----------------------------------------------------------------------------- where each seed's files are
class Seeds:
    def __init__(self, root):
        self.root = Path(root)
        self.pods = [self.root / f"pod{i}" / "live" for i in (1, 2)]

    def _find(self, rel):
        for p in self.pods:
            if (p / rel).exists():
                return p / rel
        return None

    def ids(self):
        ks = {0}
        for p in self.pods:
            ks |= {int(d.name[1:]) for d in p.glob("S[0-9]*") if d.is_dir()} | {int(d.name[1:]) for d in (p / "eval").glob("S[0-9]*") if d.is_dir()}
        return sorted(ks)

    def log(self, k, stage):
        return B4 / "CT" / stage / "train_log.jsonl" if k == 0 else self._find(f"S{k}/{stage}/train_log.jsonl")

    def ev(self, k, kind):
        """eval folder of seed k: val, train600, val_text (seed 0: the 2026-10-21 files), test_<half>, ground_<half>."""
        if k == 0 and kind in ("val", "train600", "val_text"):
            return B4 / "eval" / "CT" / kind
        return self._find(f"eval/S{k}/{kind}")

    def cards(self, name):
        return self.root / "cards" / f"{name}.jsonl"

    def joined(self, k, half):
        return self.root / "joined" / f"S{k}_{half}"


# ----------------------------------------------------------------------------- prepare
def prepare(a):
    from r1_test_refs import cmd_join
    S = Seeds(a.root)
    (S.root / "cards").mkdir(parents=True, exist_ok=True)
    todo = []                                                              # (pred file, cards file)
    for k in S.ids():
        vt = S.ev(k, "val_text")
        if vt and (vt / "predictions.jsonl").exists():
            todo.append((vt / "predictions.jsonl", B4 / "cards_CT.jsonl" if k == 0 else S.cards(f"val_S{k}")))
        for h in HALVES:
            g = S.ev(k, f"test_{h}")
            if not (g and (g / "predictions.jsonl").exists() and REFS[h].exists()):
                continue
            cmd_join(argparse.Namespace(split=h, pred=str(g / "predictions.jsonl"), refs=str(REFS[h]), out=str(S.joined(k, h))))
            todo.append((S.joined(k, h) / "predictions.jsonl", S.cards(f"test_{h}_S{k}")))
            if not any(o == S.cards(f"test_{h}_teacher") for _, o in todo):       # the teacher texts are the same for every seed: read once per half
                todo.append((S.joined(k, h) / "teacher_as_pred.jsonl", S.cards(f"test_{h}_teacher")))
    if a.no_cards:
        return
    for pred, out in todo:
        if out.parent == B4:                                               # seed 0's validation cards exist (2026-10-21)
            continue
        print(f"[cards] {pred} -> {out}", flush=True)
        subprocess.run([sys.executable, "r1_fact_cards.py", "--source", "model", "--pred", str(pred), "--out", str(out), "--tier", "flex",
                        "--concurrency", str(a.concurrency)], cwd=HERE, check=False)                  # resumable: only the missing cards are read


# ----------------------------------------------------------------------------- metrics helpers
def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def ms(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0) if v else (None, None)


def fms(v):
    m, s = ms(v)
    return "n/a" if m is None else f"{m:.3f} ± {s:.3f}"


def auc_ap(rows, key="s_first"):
    from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
    y = np.array([r["label"] for r in rows]); s = np.array([r[key] for r in rows], float)
    return (float(AUC(y, s)), float(AP(y, s))) if 0 < y.sum() < len(y) else (None, None)


def per_tte_auc(rows, ttes, key="s_first"):
    """validation: the crash windows of TTE k vs ALL normal windows (normals carry no TTE there)."""
    out = {}
    for k in ttes:
        sub = [r for r in rows if r["label"] == 0 or (r.get("tte") is not None and abs(float(r["tte"]) - k) < 1e-6)]
        out[k] = auc_ap(sub, key)[0] if any(r["label"] == 1 for r in sub) else None
    return out


def ensemble_rows(per_seed):
    """mean log-odds over the seeds that scored EVERY window of the half (the rows of the first seed carry the labels)."""
    ids = set.intersection(*[{r["id"] for r in rows} for rows in per_seed])
    by = [{r["id"]: r for r in rows} for rows in per_seed]
    return [dict(by[0][i], s_first=float(np.mean([b[i]["s_first"] for b in by]))) for i in sorted(ids)]


def text_scores(rows, bert=True):
    """model explanation vs the teacher's, each next to its floor (the same answers vs another window's teacher text, seed 0) = r1_review_reasoning.py."""
    gen = [r.get("gen_explanation") or "" for r in rows]
    ref = [r["teacher_explanation"] for r in rows]
    if len(ref) < 2:
        return None
    rng = random.Random(0)
    other = [ref[(i + rng.randrange(1, len(ref))) % len(ref)] for i in range(len(ref))]
    out = {"n": len(rows)}
    for name, fn in (("token_f1", token_f1), ("rouge_l_f1", rouge_l)):
        out[name] = {"model": float(np.mean([fn(g, r) for g, r in zip(gen, ref)])), "floor": float(np.mean([fn(g, r) for g, r in zip(gen, other)]))}
    out["bleu4"] = {"model": bleu4(gen, ref), "floor": bleu4(gen, other)}
    if bert:
        bs = _bert()
        out["bertscore_f1_rescaled"] = {"model": float(bs.score(gen, ref)[2].mean()), "floor": float(bs.score(gen, other)[2].mean())}
    return out


_BS = None


def _bert():
    global _BS
    if _BS is None:
        import torch
        from bert_score import BERTScorer
        torch.set_num_threads(4)                                           # keep the laptop usable
        _BS = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True, device="cpu")
    return _BS


def free_checks(rows):
    """reference-free checks of the generated answers (every window)."""
    ok = [r for r in rows if r.get("gen_collision") is not None]
    cons = [r["gen_collision"] == int(outcome_says_collision(r.get("gen_outcome"))) for r in ok if r.get("gen_outcome")]
    normal = [r for r in ok if r["label"] == 0]
    return {"n": len(rows),
            "format_ok": float(np.mean([r.get("gen_collision") is not None and bool(r.get("gen_explanation")) and bool(r.get("gen_outcome")) for r in rows])),
            "written_verdict_vs_score_agree": float(np.mean([r["gen_collision"] == int(r["s_first"] >= 0) for r in ok])) if ok else None,
            "verdict_vs_outcome_consistent": float(np.mean(cons)) if cons else None,
            "danger_written_on_normal_windows": float(np.mean([r["gen_collision"] == 1 or outcome_says_collision(r.get("gen_outcome")) for r in normal])) if normal else None,
            "distinct_explanations": len({r.get("gen_explanation") for r in rows}) / max(1, len(rows))}


def mean_kaggle_ci(splits, n=1000, seed=0):
    """bootstrap over windows of the MEAN Kaggle mAP of the seeds (the same resampled windows for every seed): R4."""
    from sklearn.metrics import average_precision_score as AP
    rng = np.random.default_rng(seed)
    sp0 = splits[0]
    vals = []
    for _ in range(n):
        i = rng.integers(0, sp0.n, sp0.n)
        per_seed = []
        for sp in splits:
            aps = []
            for k in TTES:
                m = sp.masks[k][i]
                y, s = sp.y[i][m], sp.p2v[i][m]
                if len(set(y)) < 2:
                    break
                aps.append(AP(y, s))
            if len(aps) < len(TTES):
                break
            per_seed.append(np.mean(aps))
        if len(per_seed) == len(splits):
            vals.append(float(np.mean(per_seed)))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def aligned(rows_list):
    """the same windows in the same order for every model (pooled bootstrap / Split arrays must line up)."""
    by = [{r["id"]: r for r in rows} for rows in rows_list]
    ids = sorted(set.intersection(*[set(b) for b in by]))
    return [[b[i] for i in ids] for b in by]


# ----------------------------------------------------------------------------- report
def report(a):
    S = Seeds(a.root)
    out = Path(a.out or S.root / "report")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ks = S.ids()
    M = {"seeds": ks, "val": {}, "train600": {}, "curves": {}, "test": {}, "text": {}, "ground": {}, "free": {}, "rules": {}}
    L = [f"# Stage 4: five seeds of the final recipe, test sets once (seeds found: {ks})", "",
         "Decision score = log-odds at 'Collision:' (reasoning model) / P (crash model). AP, AUC, Kaggle mAP are threshold-free. "
         "Every thresholded number states its threshold. Seed 0 = the 2026-10-21 control run (it won the recipe selection on validation: its VALIDATION numbers are biased upwards, its test numbers are not).", ""]
    RL = []                                                                # the rules block, put at the top at the end

    # --- validation
    L += ["## Validation (val core = 524 windows: checkpoint selection; val full = 641 incl. 0.75 / 1.25 s windows: reported only)", "",
          "| seed | val core AUC | val core AP | val full AUC | val full AUC per TTE 0.5 / 0.75 / 1.0 / 1.25 / 1.5 s | train-600 AUC | over-fit gap (train-600 - val core AUC) | best epoch P1 / P1fixR / P2V | P2V val loss |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k in ks:
        vt, vf, tr = S.ev(k, "val_text"), S.ev(k, "val"), S.ev(k, "train600")
        core = jl(vt / "predictions.jsonl") if vt else None
        full = jl(vf / "predictions.jsonl") if vf else None
        t6 = jl(tr / "predictions.jsonl") if tr else None
        c_auc, c_ap = auc_ap(core) if core else (None, None)
        f_auc = auc_ap(full)[0] if full else None
        tte = per_tte_auc(full, VAL_TTES) if full else {}
        t_auc = auc_ap(t6)[0] if t6 else None
        best, loss = [], None
        M["curves"][k] = {}
        for st in ("P1", "P1fixR", "P2V"):
            rows = jl(S.log(k, st)) if S.log(k, st) else None
            if not rows:
                best.append("n/a")
                continue
            ep = [r for r in rows if not r.get("mid")]
            v = [r["val_nexar_reasoning"]["loss_real"] for r in ep]
            M["curves"][k][st] = {"epoch": [r["epoch"] for r in ep], "train_loss": [r["train_loss"] for r in ep], "val_loss": v}
            best.append(str(ep[int(np.argmin(v))]["epoch"]))
            if st == "P2V":
                loss = float(min(v))
        M["val"][k] = {"core_auc": c_auc, "core_ap": c_ap, "full_auc": f_auc, "full_auc_tte": tte, "train600_auc": t_auc, "p2v_val_loss": loss}
        gap = None if t_auc is None or c_auc is None else t_auc - c_auc
        L.append(f"| {k} | {f3(c_auc)} | {f3(c_ap)} | {f3(f_auc)} | {' / '.join(f3(tte.get(t)) for t in VAL_TTES)} | {f3(t_auc)} | {f3(gap)} | {' / '.join(best)} | {f3(loss)} |")
    L.append(f"| **mean ± SD** | {fms([M['val'][k]['core_auc'] for k in ks])} | {fms([M['val'][k]['core_ap'] for k in ks])} | {fms([M['val'][k]['full_auc'] for k in ks])} | | "
             f"{fms([M['val'][k]['train600_auc'] for k in ks])} | | | {fms([M['val'][k]['p2v_val_loss'] for k in ks])} |")
    L.append("")

    # --- test decision
    test = {h: {} for h in HALVES}
    for h in HALVES:
        for k in ks:
            g = S.ev(k, f"test_{h}")
            if g and (g / "predictions.jsonl").exists():
                test[h][k] = jl(g / "predictions.jsonl")
    have = [k for k in ks if all(k in test[h] for h in HALVES)]
    M["test"]["seeds_with_both_halves"] = have
    if have:
        L += ["## Test decision (private 677, public 667, pooled 1,344; one pass)", "",
              "| set | model | AP | AUC | Kaggle mAP | AP 0.5 / 1.0 / 1.5 s | AUC 0.5 / 1.0 / 1.5 s | false alarms at 85 % / 90 % recall |", "|---|---|---|---|---|---|---|---|"]
        p2v = {h: jl(P2V_TEST[h]) for h in HALVES}
        sets = {}
        for h in HALVES + ("pooled",):
            per = {k: (test[h][k] if h != "pooled" else test["private"][k] + test["public"][k]) for k in have}
            ens = ensemble_rows(list(per.values()))
            ref = p2v["private"] + p2v["public"] if h == "pooled" else p2v[h]
            al = aligned(list(per.values()) + [ens] + ([ref] if ref else []))
            sets[h] = {"seeds": {k: Split(f"{h} S{k}", r) for k, r in zip(have, al)}, "ens": Split(f"{h} ensemble", al[len(have)]),
                       "p2v": Split(f"{h} P2V", al[len(have) + 1]) if ref else None}
            first = sets[h]["seeds"][have[0]]
            rows_m = {"crash model": metrics(first, "crash model")}
            rows_m.update({f"seed {k}": metrics(sp, "P2V") for k, sp in sets[h]["seeds"].items()})
            rows_m["ensemble (mean log-odds)"] = metrics(sets[h]["ens"], "P2V")
            if sets[h]["p2v"] is not None:
                rows_m["P2V (2026-10-15, reference)"] = metrics(sets[h]["p2v"], "P2V")
            M["test"][h] = rows_m
            for name, m in rows_m.items():
                L.append(f"| {h} | {name} | {f3(m['all']['AP'])} | {f3(m['all']['AUC'])} | {f3(m['kaggle_mAP'])} | {' / '.join(f3(m[t]['AP']) for t in TTES)} | "
                         f"{' / '.join(f3(m[t]['AUC']) for t in TTES)} | {f3(m['FPR_at_85_recall'])} / {f3(m['FPR_at_90_recall'])} |")
            sm = [rows_m[f"seed {k}"] for k in have]
            L.append(f"| {h} | **seeds mean ± SD** | {fms([m['all']['AP'] for m in sm])} | {fms([m['all']['AUC'] for m in sm])} | {fms([m['kaggle_mAP'] for m in sm])} | | | |")
        L.append("")

        # paired bootstrap vs the crash model (pooled)
        L += ["### Reasoning model minus crash model, pooled test (paired bootstrap over windows, 95 % interval)", "",
              "Each draw resamples the 1,344 windows with replacement and scores both models on the same draw; the interval is where 95 % of the differences fall. "
              "The windows come from fewer source videos than windows, so the intervals are a little too narrow.", "",
              "| model | delta AP | delta AUC | delta Kaggle mAP |", "|---|---|---|---|"]
        M["test"]["boot_vs_crash"] = {}
        for name, sp in [(f"seed {k}", s) for k, s in sets["pooled"]["seeds"].items()] + [("ensemble", sets["pooled"]["ens"])]:
            b = boot_delta(sp, n=a.boot)
            M["test"]["boot_vs_crash"][name] = b
            L.append(f"| {name} | " + " | ".join(f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" for v in (b["AP"], b["AUC"], b["kaggle_mAP"])) + " |")
        L.append("")

        # thresholded tables: P = 0.5 and the train-600 threshold of each seed
        L += ["### Thresholded (pooled test): confusion at P = 0.5 (log-odds 0) and at each seed's train-600 threshold (max balanced accuracy on 600 training windows)", "",
              "| model | threshold | group | TP | FN | FP | TN | recall | specificity | precision | F1 | accuracy | balanced accuracy |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        M["test"]["confusion"] = {}
        pooled = sets["pooled"]
        for name, sp, model, thrs in ([("crash model", pooled["seeds"][have[0]], "crash model", [("P = 0.5", 0.5)])] +
                                      [(f"seed {k}", sp, "P2V", [("log-odds 0 (P = 0.5)", 0.0)] + _train_thr(S, k)) for k, sp in pooled["seeds"].items()] +
                                      [("ensemble", pooled["ens"], "P2V", [("log-odds 0 (P = 0.5)", 0.0)])]):
            for tname, thr in thrs:
                for grp in ("all",) + TTES:
                    c = conf(sp, model, thr, sp.masks[grp])
                    acc = (c["TP"] + c["TN"]) / max(1, c["TP"] + c["TN"] + c["FP"] + c["FN"])
                    M["test"]["confusion"][f"{name}|{tname}|{grp}"] = dict(c, accuracy=acc, balanced_accuracy=(c["recall"] + c["specificity"]) / 2)
                    L.append(f"| {name} | {tname} | {grp if grp == 'all' else f'TTE {grp} s'} | {c['TP']} | {c['FN']} | {c['FP']} | {c['TN']} | {c['recall']:.3f} | "
                             f"{c['specificity']:.3f} | {c['precision']:.3f} | {c['F1']:.3f} | {acc:.3f} | {(c['recall'] + c['specificity']) / 2:.3f} |")
        L.append("")

        # rules
        crash_ap = M["test"]["pooled"]["crash model"]["all"]["AP"]
        aps = {k: M["test"]["pooled"][f"seed {k}"]["all"]["AP"] for k in have}
        r1, gain, above, need_above = rule_r1(list(aps.values()), crash_ap)
        m_ap, sd_ap = ms(list(aps.values()))
        p2v_ap = M["test"]["pooled"].get("P2V (2026-10-15, reference)", {}).get("all", {}).get("AP")
        ens_ap = M["test"]["pooled"]["ensemble (mean log-odds)"]["all"]["AP"]
        r3 = rule_r3(ens_ap, list(aps.values()))
        kag = [M["test"]["pooled"][f"seed {k}"]["kaggle_mAP"] for k in have]
        lo, hi = mean_kaggle_ci(list(pooled["seeds"].values()), n=a.boot)
        r4 = "above" if lo > BADAS2["kaggle_mAP"] else ("below" if hi < BADAS2["kaggle_mAP"] else "within")
        M["rules"] = {"R1": {"mean_AP_gain": gain, "seeds_above_crash": above, "of": len(have), "needed_above": need_above, "established": r1},
                      "R2": {"p2v_AP": p2v_ap, "seed_mean": m_ap, "seed_sd": sd_ap, "seed_min": min(aps.values()), "seed_max": max(aps.values()),
                             "inside_range": None if p2v_ap is None else min(aps.values()) <= p2v_ap <= max(aps.values()),
                             "z": None if p2v_ap is None or not sd_ap else (p2v_ap - m_ap) / sd_ap},
                      "R3": {"ensemble_AP": ens_ap, "best_seed_AP": max(aps.values()), "seed_mean_AP": m_ap, "helps": r3},
                      "R4": {"mean_kaggle_mAP": float(np.mean(kag)), "ci95": [lo, hi], "badas2_paper": BADAS2["kaggle_mAP"], "position": r4}}
        R = M["rules"]
        RL += ["## Rules fixed before the test pass (plan rev 3)", "",
                  f"- **R1 decision vs the crash model:** mean pooled-AP gain {gain:+.3f} (needed >= {R1_GAIN:+.3f}), {above} of {len(have)} seeds above the crash model "
                  f"(needed {need_above}) -> **{'gain established' if r1 else 'not established'}**.",
                  f"- **R2 P2V (one seed, 2026-10-15):** pooled AP {f3(p2v_ap)} vs the seeds {m_ap:.3f} ± {sd_ap:.3f} (range {min(aps.values()):.3f}-{max(aps.values()):.3f}) -> "
                  f"**{'inside the seed range: consistent with seed noise' if R['R2']['inside_range'] else 'outside the seed range'}**.",
                  f"- **R3 ensemble:** AP {ens_ap:.3f} vs best seed {max(aps.values()):.3f} and seed mean + {R3_MARGIN} = {m_ap + R3_MARGIN:.3f} -> "
                  f"**{'the ensemble is the headline' if r3 else 'the seed mean is the headline'}**.",
                  f"- **R4 BADAS-2.0 (paper Kaggle mAP {BADAS2['kaggle_mAP']}, no paired test possible):** our mean Kaggle mAP {np.mean(kag):.3f} [{lo:.3f}, {hi:.3f}] -> **{r4}** the paper's number."]

    # --- video dependence (loss on the real clip vs a blank / wrong clip, retrieval among 10)
    L += ["## Does the text depend on the clip? (loss of the teacher text on the real clip vs a blank and a wrong clip; right clip among 10)", "",
          "| seed | set | n | loss real | gap blank (95 % CI) | gap wrong (95 % CI) | perplexity | right clip among 10 (chance 0.10) |", "|---|---|---|---|---|---|---|---|"]
    for k in ks:
        for name, d in (("val core", S.ev(k, "val_text")),) + tuple((f"test {h} (referenced)", S.ev(k, f"ground_{h}")) for h in HALVES):
            m = json.load(open(d / "metrics.json", encoding="utf-8")) if d and (d / "metrics.json").exists() else None
            if not m or not m.get("after"):
                continue
            x, r = m["after"], m.get("retrieval_at_10") or {}
            M["ground"][f"S{k}|{name}"] = {"after": x, "retrieval_at_10": r}
            L.append(f"| {k} | {name} | {x['n']} | {x['loss_real']:.3f} | {x['gap_blank']:+.3f} {[round(v, 3) for v in x['gap_blank_ci']]} | "
                     f"{x['gap_wrong']:+.3f} {[round(v, 3) for v in x['gap_wrong_ci']]} | {x['perplexity_real']:.2f} | {f3(r.get('top1'))} |")
    L.append("")

    # --- text vs the teacher
    L += ["## Reasoning text vs the teacher's (model / floor = the same answers against another window's teacher text)", "",
          "Test references = the teacher's pass-1 text, only where its verdict agrees with the label (private 431 of 677; public see n): easier windows on average, "
          "so test text scores are not directly comparable with validation.", "",
          "| seed | set | n | token F1 | ROUGE-L | BLEU-4 | BERTScore | road user | side | motion | gap | mean of 4 (majority baseline) |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    teach_val = {r["id"]: r for r in jl(TEACH_VAL_CARDS)} if TEACH_VAL_CARDS.exists() else {}
    bert_cache_p = S.root / "bert_cache.json"
    bert_cache = json.load(open(bert_cache_p, encoding="utf-8")) if bert_cache_p.exists() else {}
    for k in ks:
        sets_t = [("val core", S.ev(k, "val_text") / "predictions.jsonl" if S.ev(k, "val_text") else None,
                   B4 / "cards_CT.jsonl" if k == 0 else S.cards(f"val_S{k}"), teach_val)]
        for h in HALVES:
            tc = S.cards(f"test_{h}_teacher")
            sets_t.append((f"test {h}", S.joined(k, h) / "predictions.jsonl", S.cards(f"test_{h}_S{k}"),
                           {r["id"]: r for r in jl(tc)} if tc.exists() else {}))
        for name, pred, mc, tcards in sets_t:
            if not pred or not Path(pred).exists():
                continue
            rows = jl(pred)
            key = f"S{k}|{name}"
            if key in bert_cache and not a.no_bert:
                ts = bert_cache[key]
            else:
                ts = text_scores(rows, bert=not a.no_bert)
                if ts and not a.no_bert:
                    bert_cache[key] = ts
                    bert_cache_p.write_text(json.dumps(bert_cache, indent=1), encoding="utf-8")
            mcards = {r["id"]: r for r in jl(mc)} if mc and Path(mc).exists() else {}
            ca = card_acc(mcards, tcards, {r["id"] for r in rows}) if mcards and tcards else {}
            mean4 = float(np.mean([v["acc"] for v in ca.values()])) if len(ca) == 4 else None
            base4 = float(np.mean([v["baseline"] for v in ca.values()])) if len(ca) == 4 else None
            M["text"][key] = {"scores": ts, "facts": ca, "mean4": mean4, "mean4_baseline": base4}
            if not ts:
                continue
            mf = lambda d: "n/a" if not d else f"{d['model']:.3f} / {d['floor']:.3f}"   # noqa: E731
            L.append(f"| {k} | {name} | {ts['n']} | {mf(ts['token_f1'])} | {mf(ts['rouge_l_f1'])} | {mf(ts['bleu4'])} | {mf(ts.get('bertscore_f1_rescaled'))} | "
                     + " | ".join(f3(ca.get(s, {}).get("acc")) for s in SLOTS) + f" | {f3(mean4)} ({f3(base4)}) |")
    L.append("")
    # R5
    r5 = []
    for h in HALVES:
        for k in have if have else []:
            ts = (M["text"].get(f"S{k}|test {h}") or {}).get("scores") or {}
            b = ts.get("bertscore_f1_rescaled")
            r5.append(None if not b else b["model"] - b["floor"] > 0)
    if r5:
        ok = all(x is True for x in r5)
        M["rules"]["R5"] = {"bertscore_above_floor_all": ok, "checked": len(r5), "missing": sum(x is None for x in r5)}
        for h in HALVES:
            M["rules"]["R5"][h] = {"bertscore": fms([((M["text"].get(f"S{k}|test {h}") or {}).get("scores") or {}).get("bertscore_f1_rescaled", {}).get("model") for k in have]),
                                   "mean4": fms([(M["text"].get(f"S{k}|test {h}") or {}).get("mean4") for k in have])}
        RL.append(f"- **R5 text (descriptive, no gate):** BERTScore above its floor for every seed on both halves: **{'yes' if ok else 'no'}** "
                  f"({len(r5)} checks, {sum(x is None for x in r5)} missing); test BERTScore private {M['rules']['R5']['private']['bertscore']}, "
                  f"public {M['rules']['R5']['public']['bertscore']}; facts mean of 4 private {M['rules']['R5']['private']['mean4']}, public {M['rules']['R5']['public']['mean4']}.")

    # --- reference-free checks (every test window)
    L += ["## Reference-free checks of the generated answers (every test window)", "",
          "| seed | set | n | format ok | written verdict = score verdict (log-odds 0) | verdict / outcome sentence consistent | danger written on normal windows | distinct explanations |",
          "|---|---|---|---|---|---|---|---|"]
    for k in ks:
        for h in HALVES:
            if k in test[h]:
                c = free_checks(test[h][k])
                M["free"][f"S{k}|{h}"] = c
                L.append(f"| {k} | test {h} | {c['n']} | {f3(c['format_ok'])} | {f3(c['written_verdict_vs_score_agree'])} | {f3(c['verdict_vs_outcome_consistent'])} | "
                         f"{f3(c['danger_written_on_normal_windows'])} | {f3(c['distinct_explanations'])} |")
    L.append("")

    if RL:
        L[4:4] = RL + [""]
    figures(M, sets if have else None, out / "figures")
    L += ["## Figures", "", "`figures/loss_curves.png` (train / val loss per epoch, every seed and stage), `figures/roc_pr_pooled.png`, `figures/ap_by_tte.png` "
          "(seeds mean ± SD vs the crash model and BADAS-2.0's paper numbers), `figures/reliability_ensemble.png`."]
    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=str), encoding="utf-8")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def _train_thr(S, k):
    vt = S.ev(k, "val_text")
    m = json.load(open(vt / "metrics.json", encoding="utf-8")) if vt and (vt / "metrics.json").exists() else {}
    t = (m.get("train_threshold") or {}).get("log_odds_threshold")
    return [] if t is None else [(f"train-600 log-odds {t:+.2f} (P = {1 / (1 + math.exp(-t)):.2f})", float(t))]


# ----------------------------------------------------------------------------- figures
def figures(M, sets, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, st in enumerate(("P1", "P1fixR", "P2V")):
        for k, c in M["curves"].items():
            if st in c:
                ax[j].plot(c[st]["epoch"], c[st]["val_loss"], "o-", label=f"seed {k} val")
                ax[j].plot(c[st]["epoch"], c[st]["train_loss"], ":", color=ax[j].lines[-1].get_color(), alpha=0.6)
        ax[j].set_title(f"{st}: val loss (solid) / train loss (dotted)"), ax[j].set_xlabel("epoch"), ax[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(d / "loss_curves.png", dpi=120); plt.close(fig)
    if not sets:
        return
    sp = sets["pooled"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    first = next(iter(sp["seeds"].values()))
    for name, y, s, kw in ([("crash model", first.y, first.crash, dict(color="black", lw=2))] +
                           [(f"seed {k}", x.y, x.p2v, dict(alpha=0.35, lw=1)) for k, x in sp["seeds"].items()] +
                           [("ensemble", sp["ens"].y, sp["ens"].p2v, dict(color="crimson", lw=2))]):
        f, t, _ = roc_curve(y, s); p, r, _ = precision_recall_curve(y, s)
        ax[0].plot(f, t, label=name, **kw); ax[1].plot(r, p, label=name, **kw)
    ax[0].set_title("ROC, pooled test"), ax[0].set_xlabel("false-positive rate"), ax[0].set_ylabel("true-positive rate"), ax[0].legend(fontsize=7)
    ax[1].set_title("precision-recall, pooled test"), ax[1].set_xlabel("recall"), ax[1].set_ylabel("precision")
    fig.tight_layout(); fig.savefig(d / "roc_pr_pooled.png", dpi=120); plt.close(fig)
    pooled = M["test"]["pooled"]
    seeds = [v for n, v in pooled.items() if n.startswith("seed ")]
    x = np.arange(len(TTES)); w = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w, [pooled["crash model"][t]["AP"] for t in TTES], w, label="crash model", color="gray")
    ax.bar(x, [np.mean([s[t]["AP"] for s in seeds]) for t in TTES], w, yerr=[np.std([s[t]["AP"] for s in seeds], ddof=1) if len(seeds) > 1 else 0 for t in TTES],
           label="reasoning model, seeds mean ± SD", color="steelblue")
    ax.bar(x + w, [BADAS2["AP_tte"][t] for t in TTES], w, label="BADAS-2.0 (paper)", color="orange")
    ax.set_xticks(x, [f"{t} s" for t in TTES]), ax.set_ylim(0.8, 1.0), ax.set_ylabel("AP (pooled test)"), ax.set_title("AP by time to impact"), ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(d / "ap_by_tte.png", dpi=120); plt.close(fig)
    e = sp["ens"]; p = 1 / (1 + np.exp(-e.p2v)); bins = np.linspace(0, 1, 11); idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k:")
    ax.plot([p[idx == b].mean() for b in range(10) if (idx == b).any()], [e.y[idx == b].mean() for b in range(10) if (idx == b).any()], "o-", color="crimson")
    ax.set_xlabel("ensemble P (sigmoid of the mean log-odds)"), ax.set_ylabel("observed crash share"), ax.set_title("reliability, pooled test")
    fig.tight_layout(); fig.savefig(d / "reliability_ensemble.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "report"])
    ap.add_argument("--root", default=str(ROOT / "outputs" / "stage4_2026-10-22"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-bert", action="store_true")
    ap.add_argument("--no-cards", action="store_true")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    prepare(a) if a.cmd == "prepare" else report(a)
