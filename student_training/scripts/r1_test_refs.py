"""
r1_test_refs.py - reference texts for the TEXT metrics on the Nexar test halves (plan 2026-10-22 stage 4; generalises r1_test_refs_join.py to both halves).

The test windows have no human text. The reference is the teacher's pass-1 answer (E3, the same blind prompt as the training texts, with the outcome sentence of
E3P2), used ONLY for scoring. A reference is kept only where the pass-1 verdict AGREES with the Nexar label ("agreed subset": 431 of the 677 private windows);
the subset is easier on average (mostly normal windows and TTE 0.5 s crashes), the report says so. No pass 2.

  manifest  writes dataset/manifests/r1_nexar_reasoning_test{,_public}_refs.jsonl: the test manifest rows of the agreed windows with gt.explanation /
            prediction / risk filled (status "pass1_agreed"). The evaluator reads it through R1_TEST_MANIFEST / R1_TEST_PUBLIC_MANIFEST (loss, gaps, retrieval).
  join      a generating run's predictions.jsonl (r1_eval_reasoning.py --no-ref) -> <out>/predictions.jsonl = the agreed rows with teacher_explanation /
            teacher_outcome / teacher_risk filled, and <out>/teacher_as_pred.jsonl (the teacher text as gen_explanation, for r1_fact_cards.py --source model).

    python r1_test_refs.py manifest --split private|public
    python r1_test_refs.py join --split private|public --pred <predictions.jsonl> --out <dir>
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = {"private": ROOT / "outputs" / "teacher_test_2026-10" / "runs", "public": ROOT / "outputs" / "teacher_test_public_2026-10" / "runs"}
MAN = {"private": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_test.jsonl", "public": ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_test_public.jsonl"}
REFS = {k: v.with_name(v.stem + "_refs.jsonl") for k, v in MAN.items()}


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def pass1(runs):
    """{frames_dir: (parsed E3 answer, outcome sentence)}; a retried window appears twice: the parsed one (the last) is kept."""
    e3 = {r["frames_dir"]: r["parsed"] for r in jl(Path(runs) / "E3.jsonl") if r.get("parsed")}
    p2 = Path(runs) / "E3P2.jsonl"
    outc = {r["frames_dir"]: r.get("prediction") for r in jl(p2)} if p2.exists() else {}
    return {fd: (p, outc.get(fd) or p.get("prediction")) for fd, p in e3.items()}


def reference(p1, frames_dir, label):
    """(explanation, outcome, risk) when pass 1 agrees with the label and has text, else (None, reason)."""
    a = p1.get(frames_dir)
    if a is None:
        return None, "no pass-1 answer"
    p, o = a
    if (p.get("collision") == "yes") != bool(label):
        return None, "pass-1 verdict disagrees with the label"
    if not o or not str(p.get("explanation", "")).strip():
        return None, "no teacher text"
    return (p["explanation"], o, p.get("risk_score")), None


def cmd_manifest(a):
    p1 = pass1(a.runs or RUNS[a.split])
    rows, drop, by = [], collections.Counter(), collections.Counter()
    for r in jl(MAN[a.split]):
        ref, why = reference(p1, r["frames_dir"], r["label"])
        if ref is None:
            drop[why] += 1
            continue
        g = dict(r["gt"], explanation=ref[0], prediction=ref[1], risk=ref[2], status="pass1_agreed")
        rows.append(dict(r, gt=g))
        by[(r["label"], r["tte"])] += 1
    out = Path(a.out or REFS[a.split])
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"[refs] {a.split}: {len(rows)} of {len(rows) + sum(drop.values())} windows keep a reference; dropped {dict(drop)}")
    print("       kept by (label, tte):", dict(sorted(by.items())), "->", out)


def cmd_join(a):
    refs = {r["id"]: r["gt"] for r in jl(a.refs or REFS[a.split])}               # by id: the evaluator's frames_dir of a test row is not the folder name
    pred = jl(a.pred)
    rows = []
    for r in pred:
        g = refs.get(r["id"])
        if g is not None:
            rows.append(dict(r, teacher_explanation=g["explanation"], teacher_outcome=g["prediction"], teacher_risk=g["risk"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out / "teacher_as_pred.jsonl").write_text("".join(json.dumps({"id": r["id"], "gen_explanation": r["teacher_explanation"]}, ensure_ascii=False) + "\n"
                                                       for r in rows), encoding="utf-8")
    print(f"[join] {a.split}: {len(rows)} of {len(pred)} windows have a reference -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["manifest", "join"])
    ap.add_argument("--split", choices=["private", "public"], required=True)
    ap.add_argument("--runs", default=None, help="pass-1 runs folder (default: the split's)")
    ap.add_argument("--refs", default=None, help="join: the _refs manifest (default: the split's)")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cmd_manifest(a) if a.cmd == "manifest" else cmd_join(a)
