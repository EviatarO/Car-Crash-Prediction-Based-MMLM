"""
r1_teacher_prompt_study_report.py - report of the teacher prompt study (plan 2026-10-08_Plan-Teacher-Prompt-Study.md)
-> outputs/teacher_prompt_study/report.md

Runs (all Gemini 3.8 Flash on Flex, the same 51 set-A windows, blind to the label): E1 V12 neutral raw frames, E2 V12 + box,
E3 v6_balanced re-balanced + box, E4 debate on E3's mistakes (label-routed), and E34 = E3 with E4 substituted on the routed windows.

Definitions used everywhere: TP = crash windows (label 1) where the teacher said collision = yes; TN = normal windows (label 0) where it
said no; both are read at the teacher's own verdict (risk_score >= 50). Intervals are 95 % Wilson. McNemar compares "verdict equals the
label" window by window on the same windows (exact two-sided binomial on the discordant pairs). risk AUC is the rank AUC of risk_score
for crash vs normal. Importable: compute() is reused by website/build_pilot_data.py.

    python r1_teacher_prompt_study_report.py
"""
from __future__ import annotations

import collections
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PILOT = ROOT / "outputs" / "teacher_pilot_2026-10"
OUT = ROOT / "outputs" / "teacher_prompt_study"
NAMES = {"E1": "V12 neutral · raw frames", "E2": "V12 neutral · boxed frames", "E3": "v6_balanced re-balanced · boxed frames",
         "E4": "debate on E3 mistakes (label-routed) · boxed", "E34": "E3 with E4 on its mistakes (final)"}
PAIRS = [("E1", "E2"), ("E2", "E3"), ("E1", "E3"), ("E3", "E34")]


def load(name):
    p = PILOT / "runs" / f"{name}.jsonl"
    return {r["frames_dir"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else {}


def load_all():
    runs = {n: load(n) for n in ("E1", "E2", "E3", "E4")}
    e34 = {}
    for fd, r in runs["E3"].items():
        x = runs["E4"].get(fd)
        e34[fd] = dict(x, run="E34", from_e4=True) if x and x.get("parsed") else dict(r, run="E34", from_e4=False)
    runs["E34"] = e34
    return runs


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 2), round(c + h, 2))


def mcnemar(a, b, labelled):
    n01 = sum(a[k] and not b[k] for k in labelled)
    n10 = sum(b[k] and not a[k] for k in labelled)
    n = n01 + n10
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(n01, n10) + 1)) / 2 ** n)
    return n01, n10, round(p, 3)


def rank_auc(pos, neg):
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def rec(r):
    p = r.get("parsed") or {}
    risk = p.get("risk_score")
    return {"yes": p.get("collision") == "yes", "risk": risk if isinstance(risk, (int, float)) else None, "p": p}


def stats_one(name, run, windows):
    rows = [(w, rec(run[w["frames_dir"]])) for w in windows if w["frames_dir"] in run and run[w["frames_dir"]].get("parsed")]
    P = [x for w, x in rows if w["label"] == 1]
    N = [x for w, x in rows if w["label"] == 0]
    tp, tn = sum(x["yes"] for x in P), sum(not x["yes"] for x in N)
    out = {"n": len(rows), "P": len(P), "N": len(N), "tp": tp, "tn": tn, "tp_ci": wilson(tp, len(P)), "tn_ci": wilson(tn, len(N)),
           "acc": round((tp + tn) / max(1, len(rows)), 3), "bal": round((tp / max(1, len(P)) + tn / max(1, len(N))) / 2, 3)}
    out["tp_by_tte"] = {str(h): [sum(x["yes"] for w, x in rows if w["label"] == 1 and w["horizon"] == h),
                                 sum(1 for w, x in rows if w["label"] == 1 and w["horizon"] == h)] for h in (1.5, 1.0, 0.5)}
    pr = [x["risk"] for x in P if x["risk"] is not None]
    nr = [x["risk"] for x in N if x["risk"] is not None]
    a = rank_auc(pr, nr)
    out["risk_auc"] = round(a, 3) if a is not None else None
    best = None
    for t in range(0, 101, 5):
        tpr = sum(r >= t for r in pr) / max(1, len(pr))
        tnr = sum(r < t for r in nr) / max(1, len(nr))
        if best is None or (tpr + tnr) / 2 > best[0] + 1e-9:
            best = ((tpr + tnr) / 2, t, sum(r >= t for r in pr), sum(r < t for r in nr))
    out["best_thr"] = {"t": best[1], "bal": round(best[0], 3), "tp": best[2], "tn": best[3]} if pr and nr else None
    out["median_risk"] = {"crash": st.median(pr) if pr else None, "normal": st.median(nr) if nr else None}
    ex = [str(x["p"].get("explanation", "")) for _, x in rows]
    out["words_median"] = st.median([len(e.split()) for e in ex]) if ex else None
    opens = collections.Counter(" ".join(e.lower().split()[:3]) for e in ex)
    out["opener_top"] = opens.most_common(1)[0] if opens else None
    out["opener_unique"] = f"{len(opens)}/{len(ex)}"
    out["problems"] = sum(bool(run[w["frames_dir"]]["problems"]) for w, _ in rows)
    bo = [x["p"].get("box_ok") for _, x in rows if "box_ok" in x["p"]]
    out["box_ok"] = f"{sum(b is True for b in bo)}/{len(bo)}" if bo else None
    # same verdict across the three TTE of a crash video
    byv = collections.defaultdict(list)
    for w, x in rows:
        if w["label"] == 1:
            byv[w["video_id"]].append(x["yes"])
    full = [v for v in byv.values() if len(v) == 3]
    out["same_verdict_across_tte"] = f"{sum(len(set(v)) == 1 for v in full)}/{len(full)}"
    return out


def compute():
    runs = load_all()
    windows = [json.loads(l) for l in open(PILOT / "windows.jsonl", encoding="utf-8") if json.loads(l)["set"] == "A"]
    windows = [w for w in windows if w["frames_dir"] in runs["E3"]]
    stats = {n: stats_one(n, runs[n], windows) for n in ("E1", "E2", "E3", "E34")}
    right = {n: {w["frames_dir"]: (rec(runs[n][w["frames_dir"]])["yes"] == bool(w["label"])) for w in windows
                 if w["frames_dir"] in runs[n] and runs[n][w["frames_dir"]].get("parsed")} for n in ("E1", "E2", "E3", "E34")}
    mc = {}
    for a, b in PAIRS:
        common = [k for k in right[a] if k in right[b]]
        n01, n10, p = mcnemar(right[a], right[b], common)
        mc[f"{a}|{b}"] = {"only_first": n01, "only_second": n10, "p": p}
    # E4 flips
    flips = []
    for fd, r in runs["E4"].items():
        w = next(x for x in windows if x["frames_dir"] == fd)
        e3, e4 = rec(runs["E3"][fd]), rec(r)
        flips.append({"frames_dir": fd, "label": w["label"], "variant": r["variant"], "e3_risk": e3["risk"], "e4_risk": e4["risk"],
                      "e4_yes": e4["yes"], "flipped": e4["yes"] == bool(w["label"]),
                      "e3_text": e3["p"].get("explanation"), "e4_text": e4["p"].get("explanation")})
    cost = {}
    for n in ("E1", "E2", "E3", "E4"):
        sp = PILOT / "runs" / f"{n}.summary.json"
        s = json.load(open(sp)) if sp.exists() else None
        cost[n] = s and {k: s[k] for k in ("n", "elapsed_total_s", "latency_median_s", "latency_p95_s", "cost_total_usd", "cost_per_window_usd",
                                          "served_tiers", "with_problems")}
    return {"stats": stats, "mcnemar": mc, "flips": flips, "cost": cost, "windows": windows, "runs": runs}


def md(c):
    s, W = c["stats"], c["windows"]
    npos, nneg = s["E3"]["P"], s["E3"]["N"]
    L = ["# Teacher prompt study — E1 · E2 · E3 · E4 (2026-10-08)", "",
         f"Same {len(W)} set-A windows for every run ({npos} crash = 9 videos × TTE 1.5/1.0/0.5 s; {nneg} normal = 8 videos × 3 mid-video windows), "
         "Gemini 3.8 Flash on OpenRouter Flex, native-resolution 16 frames, temperature 0.1, blind to the label. Nexar train videos OUTSIDE the "
         "1,761-window pool.", "",
         "**Definitions.** TP = crash windows where the teacher said `collision = yes` (its own verdict, risk_score ≥ 50); TN = normal windows where it said "
         "`no`. Brackets are 95 % Wilson intervals. *bal* = mean(TP rate, TN rate). *risk AUC* = rank AUC of `risk_score`, crash vs normal. "
         "*best threshold* = the risk threshold (steps of 5) with the best bal, a diagnostic only — it is chosen on these same windows.", "",
         "| run | what changes | TP | TN | accuracy | bal | risk AUC | best threshold → TP / TN (bal) |", "|---|---|---|---|---|---|---|---|"]
    for n in ("E1", "E2", "E3", "E34"):
        x = s[n]
        bt = x["best_thr"]
        L.append(f"| {n} | {NAMES[n]} | {x['tp']}/{x['P']} {x['tp_ci']} | {x['tn']}/{x['N']} {x['tn_ci']} | {x['acc']:.2f} | {x['bal']:.2f} | "
                 f"{x['risk_auc']} | {bt['t']} → {bt['tp']}/{x['P']} · {bt['tn']}/{x['N']} ({bt['bal']}) |" if bt else f"| {n} | – |")
    L += ["", "E34 is **not a perception measure**: E4 is told which windows to re-review by the label (crash missed → TP-recovery prompt, normal flagged → "
          "TN-recovery prompt), so it can only fix mistakes. It shows how much a second look can recover, not how good the first look is.", "",
          "## Paired tests (same windows; verdict = label)", "", "| comparison | only first right | only second right | exact p |", "|---|---|---|---|"]
    for k, v in c["mcnemar"].items():
        a, b = k.split("|")
        L.append(f"| {a} vs {b} | {v['only_first']} | {v['only_second']} | {v['p']} |")
    L += ["", "With 51 windows only large differences reach p < 0.05; p ≥ 0.1 means *not distinguishable*, not *equal*.", "",
          "## Crash TP by time-to-event", "", "| run | TTE 1.5 s | TTE 1.0 s | TTE 0.5 s | same verdict across the 3 TTE of a video |", "|---|---|---|---|---|"]
    for n in ("E1", "E2", "E3", "E34"):
        t = s[n]["tp_by_tte"]
        L.append(f"| {n} | {t['1.5'][0]}/{t['1.5'][1]} | {t['1.0'][0]}/{t['1.0'][1]} | {t['0.5'][0]}/{t['0.5'][1]} | {s[n]['same_verdict_across_tte']} |")
    L += ["", "## Text quality, cost, time", "",
          "| run | explanation words (median) | most common 3-word opener | distinct openers | windows with soft problems | box_ok | tier | total time | cost / window | cost total |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in ("E1", "E2", "E3", "E4"):
        x = s.get(n)
        k = c["cost"][n]
        if x:
            L.append(f"| {n} | {x['words_median']} | “{x['opener_top'][0]}” ×{x['opener_top'][1]} | {x['opener_unique']} | {x['problems']} | {x['box_ok'] or '–'} | "
                     f"{','.join(k['served_tiers'])} | {k['elapsed_total_s']:.0f} s | ${k['cost_per_window_usd']:.4f} | ${k['cost_total_usd']:.2f} |")
        else:
            L.append(f"| {n} | – | – | – | {k['with_problems']} | – | {','.join(k['served_tiers'])} | {k['elapsed_total_s']:.0f} s | "
                     f"${k['cost_per_window_usd']:.4f} | ${k['cost_total_usd']:.2f} |")
    tot = sum(k["cost_total_usd"] for k in c["cost"].values() if k)
    L += ["", f"Total spend of the study: ${tot:.2f}. Soft problems = explanation length outside 25–45 words, outcome/colour/time words, a mention of the box, "
          "or verdict inconsistent with risk_score; they are flags for review, no answer was discarded.", "",
          "## E4 debate — the 15 windows E3 got wrong", "",
          "| window | label | prompt | E3 risk → E4 risk | E4 verdict | fixed? |", "|---|---|---|---|---|---|"]
    for f in c["flips"]:
        L.append(f"| {f['frames_dir']} | {f['label']} | {f['variant']} | {f['e3_risk']} → {f['e4_risk']} | {'yes' if f['e4_yes'] else 'no'} | {'**flipped**' if f['flipped'] else ''} |")
    fx = collections.Counter((f["variant"], f["flipped"]) for f in c["flips"])
    L += ["", f"TP-recovery prompt fixed {fx[('e4_tp', True)]} of {fx[('e4_tp', True)] + fx[('e4_tp', False)]} missed crashes; TN-recovery prompt fixed "
          f"{fx[('e4_tn', True)]} of {fx[('e4_tn', True)] + fx[('e4_tn', False)]} false alarms. Flipped windows need a human check that the new explanation is "
          "grounded in the frames rather than invented to fit the second prompt (the August debate produced invented agents on 02117 and 00474).", "",
          "## Caveats", "",
          "- 51 windows from 17 videos (3 windows per video are not independent); the intervals above are wide and the tests are paired but low-powered.",
          "- E1 vs E2 isolates the box; E2 vs E3 changes the whole prompt (structure, gates, wording), so it does not say which part of E3 helped.",
          "- Explanations were scored only by rules here; their quality is judged by reading them on the website tab (Boxed-teacher pilot → run picker).",
          "- The best-threshold column uses the evaluation windows to pick the threshold; treat it as a ceiling, not a result."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ anchored / hindsight runs (E5, E6, E7, chain, routed)
sys.path.insert(0, str(HERE))
from r1_phase_report import token_f1  # noqa: E402

CHAIN_RUN = "E6C"
CAVEAT_MULTI = ("**E5, E6 and E7 are label-informed by design:** the anchor's correct verdict is the video's label, and E7 is told the outcome. "
                "Their TP/TN on these windows is not a perception measure. What they test is the text (grounded or copied, right road user) "
                "and the visibility reading.")


def routed_run(runs):
    """The final answer per window by the routing flow: E3 where it was right; E6 for failures with a correct sibling;
    for videos with no correct E3 window: E7 at TTE 0.5, then the E6 chain at 1.0 and 1.5."""
    out = {fd: dict(r, final_from="E3") for fd, r in runs["E3"].items()}
    for fd, r in runs["E6"].items():
        if r.get("parsed"):
            out[fd] = dict(r, final_from="E6")
    e3 = runs["E3"]
    for fd, r in runs["E7"].items():
        sib_ok = any(is_ok(e3[s]) for s in e3 if s.split("_")[0] == fd.split("_")[0] and s != fd)
        if not sib_ok and r.get("horizon") == 0.5 and r.get("parsed"):
            out[fd] = dict(r, final_from="E7")
    for fd, r in runs.get(CHAIN_RUN, {}).items():
        if r.get("parsed") and not r.get("skipped"):
            out[fd] = dict(r, final_from=CHAIN_RUN)
    return out


def is_ok(r):
    return bool(r.get("parsed")) and (r["parsed"].get("collision") == "yes") == bool(r["label"])


def compute_multi(c):
    runs = c["runs"]
    for n in ("E5", "E6", "E7", CHAIN_RUN):
        runs[n] = load(n)
    runs["ROUTED"] = routed_run(runs)
    wins = c["windows"]
    stats = {"ROUTED": stats_one("ROUTED", runs["ROUTED"], wins)}
    e3, per = runs["E3"], {}
    anchor_text = {}
    for n in ("E5", "E6", CHAIN_RUN):
        for fd, r in runs[n].items():
            if not r.get("parsed"):
                continue
            a = r.get("anchor_frames_dir")
            src = runs["E7"] if r.get("anchor_from") == "E7" else runs[CHAIN_RUN] if r.get("anchor_from") == CHAIN_RUN else e3
            anchor_text[(n, fd)] = (src.get(a) or {}).get("parsed", {}).get("explanation", "")
    for n in ("E5", "E6", "E7", CHAIN_RUN):
        rows = [r for r in runs[n].values() if r.get("parsed")]
        ok = sum(is_ok(r) for r in rows)
        per[n] = {"n": len(rows), "correct": ok, "problems": sum(bool(r["problems"]) for r in rows)}
        if n in ("E5", "E6", CHAIN_RUN):
            cs = [token_f1(r["parsed"]["explanation"], anchor_text[(n, r["frames_dir"])]) for r in rows]
            per[n]["copy_f1_mean"] = round(sum(cs) / max(1, len(cs)), 3)
            per[n]["copy_f1_max"] = round(max(cs), 3) if cs else None
            per[n]["same_agent_true"] = sum(bool(r["parsed"].get("same_agent")) for r in rows)
        if n in ("E6", "E7", CHAIN_RUN):
            per[n]["hazard_visible"] = dict(collections.Counter(r["parsed"].get("hazard_visible") for r in rows))
    # text shift vs E3's own wrong explanation of the same window
    for n in ("E5", "E6", "E7"):
        sh = [token_f1(r["parsed"]["explanation"], e3[fd]["parsed"]["explanation"]) for fd, r in runs[n].items() if r.get("parsed") and fd in e3]
        per[n]["f1_vs_e3_text_mean"] = round(sum(sh) / max(1, len(sh)), 3)
    flips = []
    for fd in sorted(runs["E7"]):
        e3r = e3[fd]
        row = {"frames_dir": fd, "label": e3r["label"], "e3_risk": rec(e3r)["risk"], "e3_yes": rec(e3r)["yes"]}
        for n in ("E5", "E6", "E7", CHAIN_RUN):
            r = runs[n].get(fd)
            if r and r.get("parsed"):
                p = r["parsed"]
                row[n] = {"yes": p.get("collision") == "yes", "risk": p.get("risk_score"), "hazard": p.get("hazard_visible"),
                          "first": p.get("first_visible_frame"), "same_agent": p.get("same_agent"), "explanation": p.get("explanation"),
                          "prediction": p.get("prediction"), "anchor": r.get("anchor_frames_dir")}
        flips.append(row)
    cost = {}
    for n in ("E5", "E6", "E7", CHAIN_RUN):
        sp = PILOT / "runs" / f"{n}.summary.json"
        s = json.load(open(sp)) if sp.exists() else None
        cost[n] = s and {k: s[k] for k in ("n", "elapsed_total_s", "latency_median_s", "cost_total_usd", "cost_per_window_usd", "served_tiers", "with_problems")}
    c["multi"] = {"stats": stats, "per_run": per, "windows": flips, "cost": cost}
    return c


def md_multi(c):
    m, s = c["multi"], c["stats"]
    r = m["stats"]["ROUTED"]
    L = ["", "## Anchored and hindsight prompts on E3's 15 failures (E5, E6, E7, chain)", "", CAVEAT_MULTI, "",
         "Routing (two stages): E3 is run on all 3 windows of every video first; a failed window then takes the nearest correct sibling as anchor "
         "(E5 = reference clip + E3's reasoning for it; E6 = E5 + a timing check and `hazard_visible`); a video with no correct window "
         "(00997) takes E7 (frames after the window + the outcome in text) at TTE 0.5, then E6 chained to 1.0 and 1.5. Final answer per window = "
         "`ROUTED`.", "",
         "| run | windows | verdict = label | copy score vs anchor text (F1 mean / max) | same_agent = true | hazard_visible | text overlap with E3's own explanation (F1) | soft problems |",
         "|---|---|---|---|---|---|---|---|"]
    for n, p in m["per_run"].items():
        L.append(f"| {n} | {p['n']} | {p['correct']}/{p['n']} | {p.get('copy_f1_mean', '–')} / {p.get('copy_f1_max', '–')} | "
                 f"{p.get('same_agent_true', '–')} | {p.get('hazard_visible', '–')} | {p.get('f1_vs_e3_text_mean', '–')} | {p['problems']} |")
    L += ["", "Copy score = token F1 between the new explanation and the anchor's explanation (1.0 = identical wording); the E3 overlap column shows how "
          "much the text changed from the wrong E3 answer. A copy score near the max of ordinary same-scene pairs means the teacher paraphrases "
          "the anchor instead of describing its own frames: those windows are the ones to read first.", "",
          "### Routed result on the 51 windows (E3 kept where right)", "",
          "| run | TP | TN | accuracy | bal | risk AUC |", "|---|---|---|---|---|---|",
          f"| E3 | {s['E3']['tp']}/{s['E3']['P']} {s['E3']['tp_ci']} | {s['E3']['tn']}/{s['E3']['N']} {s['E3']['tn_ci']} | {s['E3']['acc']:.2f} | {s['E3']['bal']:.2f} | {s['E3']['risk_auc']} |",
          f"| ROUTED | {r['tp']}/{r['P']} {r['tp_ci']} | {r['tn']}/{r['N']} {r['tn_ci']} | {r['acc']:.2f} | {r['bal']:.2f} | {r['risk_auc']} |", "",
          "ROUTED is label-informed on its 15 replaced windows and says nothing about how well a blind teacher sees; it measures how clean a training "
          "set the routing can produce. The 36 windows E3 got right are untouched.", "",
          "### Per-window answers (the 15 failures)", "",
          "| window | label | E3 risk | E5 | E6 (hazard, first frame) | E7 (hazard, first frame) | chain |", "|---|---|---|---|---|---|---|"]
    for w in m["windows"]:
        def cell(n, hz=False):
            x = w.get(n)
            if not x:
                return "–"
            t = f"{'yes' if x['yes'] else 'no'}/{x['risk']}"
            return t + (f" ({x['hazard']}, {x['first']})" if hz else "")
        L.append(f"| {w['frames_dir']} | {w['label']} | {w['e3_risk']} | {cell('E5')} | {cell('E6', True)} | {cell('E7', True)} | {cell(CHAIN_RUN, True)} |")
    L += ["", "### Explanation + prediction sentence of the routed answers (read these)", ""]
    for w in m["windows"]:
        pick = "E6" if "E6" in w else CHAIN_RUN if CHAIN_RUN in w else "E7"
        x = w[pick]
        L += [f"- **{w['frames_dir']}** (label {w['label']}, {pick}, anchor {x.get('anchor') or 'hindsight'}): {x['explanation']} "
              f"*Prediction: {x['prediction']}*"]
    L += ["", "### Cost and time", "", "| run | calls | time | cost / call | cost |", "|---|---|---|---|---|"]
    for n, k in m["cost"].items():
        if k:
            L.append(f"| {n} | {k['n']} | {k['elapsed_total_s']:.0f} s | ${k['cost_per_window_usd']:.4f} | ${k['cost_total_usd']:.2f} |")
    tot = sum(k["cost_total_usd"] for k in m["cost"].values() if k)
    L += ["", f"Spend of this stage: ${tot:.2f}. Hand labels (\"hitter first visible\") are pending: the website tab collects them and the visibility "
          "agreement with the teacher's `hazard_visible` is computed from the exported notes in the next pass."]
    return "\n".join(L) + "\n"


def main():
    c = compute_multi(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    text = md(c) + md_multi(c)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps({k: c[k] for k in ("stats", "mcnemar", "flips", "cost", "multi")}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
