"""
r1_test_refs_join.py - reference texts for the TEXT metrics on the Nexar PRIVATE test (plan 2026-10-18, Part A2).

The private test has no human text. The reference is the teacher's pass-1 answer (E3, the same blind prompt as the training texts, with the outcome sentence
from E3P2). Pass 1 contradicts the label on 246 of the 677 windows, so only the windows where the pass-1 verdict AGREES with the label are used (431, the
"agreed subset"). This subset is biased toward easier windows: it keeps 291 of 339 normal windows (86%) but only 74 / 47 / 19 of the 142 / 117 / 79 crash windows at
TTE 0.5 / 1.0 / 1.5 s (52% / 40% / 24%), so text numbers on the 1.5 s crashes rest on 19 windows. The report says so.

Reads   outputs/teacher_test_2026-10/runs/E3.jsonl, E3P2.jsonl
        outputs/r1_test_eval_2026-10-17/test_eval_2026-10-17/P2V/results/predictions.jsonl   (decision-only run; --no-ref left the teacher fields empty)
Writes  outputs/r1_p2v_report_2026-10-18/test_agreed/predictions.jsonl   the P2V rows of the agreed windows with teacher_explanation / teacher_outcome /
                                                                         teacher_risk filled from pass 1 (same fields as a validation predictions.jsonl)
        outputs/r1_p2v_report_2026-10-18/test_agreed/teacher_as_pred.jsonl   the same ids with gen_explanation = the TEACHER text, to read its fact cards with
                                                                         `r1_fact_cards.py --source model` (so teacher and model cards come from the same reader)

    python r1_test_refs_join.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "outputs" / "teacher_test_2026-10" / "runs"
P2V = ROOT / "outputs" / "r1_test_eval_2026-10-17" / "test_eval_2026-10-17" / "P2V" / "results" / "predictions.jsonl"
OUT = ROOT / "outputs" / "r1_p2v_report_2026-10-18" / "test_agreed"


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    e3 = {}
    for r in jl(T / "E3.jsonl"):
        if r.get("parsed"):
            e3[r["frames_dir"]] = r                                    # a retried window appears twice: keep the parsed one (the last)
    outc = {r["frames_dir"]: r.get("prediction") for r in jl(T / "E3P2.jsonl")}
    pred = jl(P2V)
    rows, drop = [], collections.Counter()
    for r in pred:
        fd = r["id"].replace("nexartest_", "", 1) + "_hires"
        a = e3.get(fd)
        if a is None:
            drop["no pass-1 answer"] += 1
            continue
        p = a["parsed"]
        if (p["collision"] == "yes") != bool(r["label"]):
            drop["pass-1 verdict disagrees with the label"] += 1
            continue
        o = outc.get(fd) or p.get("prediction")
        if not o or not str(p.get("explanation", "")).strip():
            drop["no teacher text"] += 1
            continue
        rows.append(dict(r, teacher_explanation=p["explanation"], teacher_outcome=o, teacher_risk=p.get("risk_score")))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (OUT / "teacher_as_pred.jsonl").write_text("".join(json.dumps({"id": r["id"], "gen_explanation": r["teacher_explanation"]}, ensure_ascii=False) + "\n" for r in rows),
                                               encoding="utf-8")
    by = collections.Counter((r["label"], r["group"]) for r in rows)
    allg = collections.Counter((r["label"], r["group"]) for r in pred)
    print(f"[join] {len(rows)} of {len(pred)} private windows kept; dropped {dict(drop)}")
    print("       kept / all by (label, group):", {k: f"{by[k]}/{allg[k]}" for k in sorted(allg, key=str)})
    print(f"       -> {OUT}")


if __name__ == "__main__":
    main()
