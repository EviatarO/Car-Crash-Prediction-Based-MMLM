"""
r1_teacher_pilot_report.py - summary of the boxed-teacher pilot (runs R1 R2 R3) -> outputs/teacher_pilot_2026-10/summary.md + review.xlsx

Sections: (1) cost and time per run, (2) order effect R1 vs R3 on the SAME 51 windows, (3) cross-TTE consistency and label
contradictions, (4) box quality, (5) side-by-side texts for the user's review.
"""
from __future__ import annotations

import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from r1_phase_report import token_f1  # noqa: E402

OUT = ROOT / "outputs" / "teacher_pilot_2026-10"


def load(name):
    p = OUT / "runs" / f"{name}.jsonl"
    return {r["frames_dir"]: r for r in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else {}


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "n/a"


def verdict_table(run):
    rows = [r for r in run.values() if r["parsed"]]
    out = {}
    for name, sel in (("crash TTE 1.5", lambda r: r["label"] == 1 and r["horizon"] == 1.5),
                      ("crash TTE 1.0", lambda r: r["label"] == 1 and r["horizon"] == 1.0),
                      ("crash TTE 0.5", lambda r: r["label"] == 1 and r["horizon"] == 0.5),
                      ("normal", lambda r: r["label"] == 0), ("all", lambda r: True)):
        S = [r for r in rows if sel(r)]
        ok = sum((r["parsed"]["collision"] == "yes") == bool(r["label"]) for r in S)
        out[name] = (ok, len(S))
    return out


def consistency(run, boxes):
    """Per crash video (3 windows): same class / position; gap contradictions; per normal video the same."""
    by = collections.defaultdict(list)
    for r in run.values():
        if r["parsed"]:
            by[r["video_id"]].append(r)
    res = {"videos": 0, "class_same": 0, "position_same": 0, "gap_never_opening_as_tte_shrinks": 0,
           "crash_gap_opening_or_steady": [0, 0], "normal_gap_closing": [0, 0], "class_same_normal": [0, 0]}
    flagged = []
    for vid, rs in by.items():
        if len(rs) != 3:
            continue
        lab = rs[0]["label"]
        c = len({r["parsed"]["agent_class"] for r in rs}) == 1
        p = len({r["parsed"]["position"] for r in rs}) == 1
        if lab == 1:
            res["videos"] += 1
            res["class_same"] += c
            res["position_same"] += p
            order = sorted(rs, key=lambda r: -r["horizon"])                # 1.5, 1.0, 0.5
            res["gap_never_opening_as_tte_shrinks"] += not any(order[i]["parsed"]["gap"] in ("closing", "steady") and
                                                                order[i + 1]["parsed"]["gap"] == "opening" for i in range(2))
            if not (c and p):
                flagged.append(vid)
        else:
            res["class_same_normal"][0] += c
            res["class_same_normal"][1] += 1
        for r in rs:
            if lab == 1:
                res["crash_gap_opening_or_steady"][1] += 1
                res["crash_gap_opening_or_steady"][0] += r["parsed"]["gap"] in ("opening", "steady")
            else:
                res["normal_gap_closing"][1] += 1
                res["normal_gap_closing"][0] += r["parsed"]["gap"] == "closing"
    res["flagged_crash_videos"] = flagged
    return res


def main():
    runs = {n: load(n) for n in ("R1", "R2", "R3")}
    boxes = {json.loads(l)["frames_dir"]: json.loads(l) for l in open(OUT / "boxes.jsonl", encoding="utf-8")}
    L = ["# Boxed-teacher pilot — summary", "", "Model google/gemini-3.8-flash via OpenRouter, native-resolution boxed frames, temperature 0.1. "
         "All three runs use the SAME 51 windows (set A: 9 crash videos × 3 TTE + 8 normal videos × 3 midpoint windows), all outside the 1,761 pool.", ""]
    summ = {n: json.load(open(OUT / "runs" / f"{n}.summary.json")) for n in runs if (OUT / "runs" / f"{n}.summary.json").exists()}
    desc = {"R1": "Standard · explanation → verdict", "R2": "Flex · explanation → verdict", "R3": "Standard · verdict → explanation"}
    L += ["## 1. Cost and time", "", "| run | what | windows | failed | total wall time | latency median / p95 / max | cost total | cost per window | input / output tokens (median) | served tier | with soft problems |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summ.items():
        L.append(f"| {n} | {desc[n]} | {s['n']} | {s['failed']} | {s['elapsed_total_s']:.0f} s | {s['latency_median_s']} / {s['latency_p95_s']} / {s['latency_max_s']} s | "
                 f"${s['cost_total_usd']:.2f} | ${s['cost_per_window_usd']:.4f} | {s['prompt_tokens_median']:.0f} / {s['completion_tokens_median']:.0f} | "
                 f"{', '.join(s['served_tiers'])} | {s['with_problems']} |")
    if "R1" in summ and "R2" in summ:
        ch = {n: sum((v["usage"].get("prompt_tokens_details", {}) or {}).get("cached_tokens", 0) > 0 for v in runs[n].values()) for n in runs}
        L += ["", f"Calls that got a prompt-cache discount: R1 {ch['R1']}/51, R2 {ch['R2']}/51, R3 {ch['R3']}/51. R1's lower cost comes from those cache hits "
                  "(R1 and R2 sent identical frames and text at the same time). In a real run every window is sent once, so the fair per-window prices are the "
                  "runs WITHOUT cache hits: Standard = R3, Flex = R2.",
              f"Flex vs Standard (uncached): cost per window ×{summ['R2']['cost_per_window_usd'] / summ['R3']['cost_per_window_usd']:.2f}; "
              f"latency median {summ['R2']['latency_median_s']} s (Flex) vs {summ['R3']['latency_median_s']} s (Standard R3) / {summ['R1']['latency_median_s']} s (Standard R1); "
              f"total wall time with 16 calls in parallel: Flex {summ['R2']['elapsed_total_s']:.0f} s vs Standard {summ['R1']['elapsed_total_s']:.0f} s (R1) / {summ['R3']['elapsed_total_s']:.0f} s (R3)."]
        n_full = 4446
        L += ["", f"Extrapolation to the full 4,446 windows (uncached prices): Standard ${summ['R3']['cost_per_window_usd'] * n_full:.0f}, Flex ${summ['R2']['cost_per_window_usd'] * n_full:.0f}."]
    L += ["", "## 2. Teacher verdict vs the true label (blind teacher; 'yes' = collision), by run", "",
          "| group | " + " | ".join(f"{n} ({desc[n]})" for n in runs if runs[n]) + " |", "|---|" + "---|" * sum(bool(r) for r in runs.values())]
    vt = {n: verdict_table(r) for n, r in runs.items() if r}
    for g in ("crash TTE 1.5", "crash TTE 1.0", "crash TTE 0.5", "normal", "all"):
        L.append(f"| {g} | " + " | ".join(pct(*vt[n][g]) for n in vt) + " |")
    a1 = {f: b["a1_p_collision"] for f, b in boxes.items()}
    for n, r in runs.items():
        if r:
            agree = sum((v["parsed"]["collision"] == "yes") == (a1[f] >= 0.5) for f, v in r.items() if v["parsed"])
            L.append(f"\n{n}: teacher yes/no equals the crash model's (A1, P ≥ 0.5) on {pct(agree, sum(bool(v['parsed']) for v in r.values()))}.")
    if runs["R1"] and runs["R3"]:
        both = [f for f in runs["R1"] if f in runs["R3"] and runs["R1"][f]["parsed"] and runs["R3"][f]["parsed"]]
        same_v = sum(runs["R1"][f]["parsed"]["collision"] == runs["R3"][f]["parsed"]["collision"] for f in both)
        f1 = [token_f1(runs["R1"][f]["parsed"]["explanation"], runs["R3"][f]["parsed"]["explanation"]) for f in both]
        same_c = sum(runs["R1"][f]["parsed"]["agent_class"] == runs["R3"][f]["parsed"]["agent_class"] for f in both)
        same_p = sum(runs["R1"][f]["parsed"]["position"] == runs["R3"][f]["parsed"]["position"] for f in both)
        same_g = sum(runs["R1"][f]["parsed"]["gap"] == runs["R3"][f]["parsed"]["gap"] for f in both)
        w1 = [len(runs["R1"][f]["parsed"]["explanation"].split()) for f in both]
        w3 = [len(runs["R3"][f]["parsed"]["explanation"].split()) for f in both]
        L += ["", "## 3. Order effect: R1 (explanation → verdict) vs R3 (verdict → explanation), same 51 windows", "",
              f"- same verdict: {pct(same_v, len(both))}; same agent_class: {pct(same_c, len(both))}; same position: {pct(same_p, len(both))}; same gap: {pct(same_g, len(both))}",
              f"- explanation word overlap (token F1 between the two texts of the same window): mean {np.mean(f1):.2f}",
              f"- explanation length (words): R1 median {st.median(w1)}, R3 median {st.median(w3)}",
              "- Caveat: the teacher thinks before it answers (about 5,000 hidden reasoning tokens), so the JSON field order is only a weak control of "
              "what it decides first; the order that matters is the STUDENT's, tested in training."]
    L += ["", "## 4. Cross-TTE consistency and label contradictions (crash videos: 9 per set; normal videos: 8 per set)", ""]
    for n, r in runs.items():
        if r:
            c = consistency(r, boxes)
            L += [f"**{n}** — crash videos with 3 parsed windows: {c['videos']}; same agent_class in all 3: {pct(c['class_same'], c['videos'])}; same position in all 3: "
                  f"{pct(c['position_same'], c['videos'])}; gap never turns to 'opening' as the event gets closer: {pct(c['gap_never_opening_as_tte_shrinks'], c['videos'])}; "
                  f"crash windows with gap opening/steady: {pct(*c['crash_gap_opening_or_steady'])}; normal windows with gap closing: {pct(*c['normal_gap_closing'])}; "
                  f"normal videos with the same class in all 3: {pct(*c['class_same_normal'])}; flagged crash videos (class or position changes): {c['flagged_crash_videos']}", ""]
    L += ["## 5. Boxes", ""]
    fl = collections.Counter(b["flag"] for b in boxes.values())
    L.append(f"Box flags over {len(boxes)} windows: {dict(fl)}. Boxed frames per window: median {st.median(b['frames_boxed'] for b in boxes.values())} of 16; "
             f"median attention share of the chosen track: {st.median(b['attention_share_of_tracks'] or 0 for b in boxes.values()):.2f}; "
             f"median fraction of the head's attention (on real tokens) that falls inside ANY tracked box: "
             f"{st.median(b['attn_inside_boxes'] / json.load(open(OUT / 'attn' / (f + '.json')))['attn_total_real'] for f, b in boxes.items()):.2f}.")
    for n, r in runs.items():
        if r:
            ok = sum(bool(v["parsed"]) and v["parsed"].get("box_ok") is True for v in r.values())
            L.append(f"- {n}: teacher box_ok = true in {pct(ok, sum(bool(v['parsed']) for v in r.values()))}")
    probs = collections.Counter(p.split(":")[0].split("=")[0] for r in runs.values() for v in r.values() for p in v["problems"])
    L += ["", f"Soft-validation problems (all runs): {dict(probs)}", "", "Side-by-side texts: `review.xlsx` (one row per window and run) and `review/<frames_dir>.jpg` (4×4 boxed grids)."]
    (OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))

    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "side_by_side"
    ws.append(["frames_dir", "set", "label", "TTE/horizon", "box flag", "A1 P(collision)"] + [f"{n} {k}" for n in runs for k in ("verdict", "class", "position", "motion", "gap", "box_ok", "explanation", "problems")])
    for f, b in boxes.items():
        row = [f, b["set"], b["label"], b["horizon"], b["flag"], b["a1_p_collision"]]
        for n, r in runs.items():
            v = r.get(f)
            p = v["parsed"] if v and v["parsed"] else {}
            row += [p.get("collision"), p.get("agent_class"), p.get("position"), p.get("object_motion"), p.get("gap"), p.get("box_ok"),
                    p.get("explanation"), "; ".join(v["problems"]) if v else None]
        ws.append(row)
    wb.save(OUT / "review.xlsx")


if __name__ == "__main__":
    main()
