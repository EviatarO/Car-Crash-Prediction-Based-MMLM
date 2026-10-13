"""
r1_fact_eval.py - fact-level check of generated explanations (plan 2026-10-13_Plan-Reasoning-Diagnose-and-Levers, Step 1.1).

From a text, pull out 4 facts with fixed word lists (no model):
  type   : car | suv | pickup | van | truck | bus | motorcycle | cyclist | pedestrian   (first road user mentioned; 'ego vehicle' ignored)
  side   : left | ahead | right                                                          (first position phrase)
  motion : parked | crossing | cutting_in | turning | braking | steady                    (most specific word present)
  gap    : closing | opening | stable                                                    (most specific word present)
A fact is None when no word of its list is present. Agreement of a run = share of windows where the model's fact equals the
teacher's fact, over the windows where the TEACHER fact is known (a model None counts as wrong).

    python r1_fact_eval.py --runs NAME=<predictions.jsonl> [NAME=...] [--out <dir>] [--show 30]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

TYPES = [("pedestrian", r"pedestrian|person|worker|walker|child|people"), ("cyclist", r"cyclist|bicycl|\bbike\b|scooter"),
         ("motorcycle", r"motorcycl|motorbike"), ("bus", r"\bbus\b"),
         ("truck", r"truck|lorry|dump|semi\b|trailer|tractor|tanker"), ("van", r"\bvan\b|minivan"),
         ("pickup", r"pickup|pick-up"), ("suv", r"\bsuv\b|sport utility|crossover|jeep"),
         ("car", r"\bcar\b|sedan|hatchback|coupe|wagon|taxi|cab\b|convertible")]
LEFT = r"(on|from|to|toward|towards|along|at|of|beside|by) the (far |adjacent |ego vehicle's |driver's )?left|left (lane|side|curb|shoulder|roadside|edge)|adjacent left|front-left|leftmost"
RIGHT = r"(on|from|to|toward|towards|along|at|of|beside|by) the (far |adjacent |ego vehicle's |driver's )?right|right (lane|side|curb|shoulder|roadside|edge)|adjacent right|front-right|rightmost"
AHEAD_STRONG = r"directly ahead|in front of|directly in front|same lane|ego lane|its lane|lead vehicle|leading vehicle|travel lane directly|in the ego"
AHEAD_WEAK = r"ahead|leading|in front"
MOTIONS = [("parked", r"\bpark(ed|ing)?\b"), ("crossing", r"\bcross(es|ing)?\b|across (the|its|ego)"),
           ("cutting_in", r"cut(s|ting)? (in|across|into)|merg|swerv|veer|drift|chang\w* lanes?|pull(s|ing)? (out|into)|enter(s|ing)? (the|its|ego)"),
           ("turning", r"\bturn(s|ing)?\b"), ("braking", r"brak|slow|decelerat|\bstop|stationary|halt|queue"),
           ("steady", r"maintain|steady|constant|consistent|travels? straight|same (speed|direction)|parallel")]
GAPS = [("closing", r"clos(e|es|ing) (the |in |rapidly|quickly)|gap (clos|decreas|narrow|shrink|disappear|collaps)|distance (clos|decreas|narrow|shrink)|"
                    r"decreas|narrow|shrink|diminish|collaps|approach(es|ing)? (rapidly|quickly|fast)|clos(es|ing) (rapidly|quickly)|clos(es|ing)$|rapidly clos"),
        ("opening", r"increas|widen|grow(s|ing)|pull(s|ing)? away|moves? away|recede|leaves the (frame|view)|exits"),
        ("stable", r"steady (gap|distance|separation)|constant|maintain\w* (a |the )?(safe|steady|consistent|stable)|same distance|unchanged|stable|consistent (gap|distance|separation)")]
FACTS = ("type", "side", "motion", "gap")


def _first(text, table):
    best = None
    for name, pat in table:
        m = re.search(pat, text)
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), name)
    return best[1] if best else None


def _priority(text, table):
    for name, pat in table:
        if re.search(pat, text):
            return name
    return None


def side_of(t):
    """A strong 'ahead' phrase wins only if it comes before any left / right phrase; otherwise the first left / right phrase; else a weak 'ahead'."""
    pos = {}
    for name, pat in (("left", LEFT), ("right", RIGHT), ("ahead", AHEAD_STRONG)):
        m = re.search(pat, t)
        if m:
            pos[name] = m.start()
    if pos:
        return min(pos, key=pos.get)
    return "ahead" if re.search(AHEAD_WEAK, t) else None


def facts(text):
    t = re.sub(r"\bego vehicle\b|\bthe vehicle\b", " ", (text or "").lower())
    return {"type": _first(t, TYPES), "side": side_of(t), "motion": _priority(t, MOTIONS), "gap": _priority(t, GAPS)}


def agreement(rows, key_model="gen_explanation", key_ref="teacher_explanation"):
    per, per_group = {}, defaultdict(dict)
    pairs = [(facts(r[key_ref]), facts(r[key_model]), r.get("group")) for r in rows]
    for f in FACTS:
        known = [(a[f], b[f], g) for a, b, g in pairs if a[f] is not None]
        per[f] = {"agree": sum(x == y for x, y, _ in known) / max(1, len(known)), "n_known": len(known),
                  "teacher_coverage": len(known) / max(1, len(pairs)),
                  "model_none": sum(y is None for _, y, _ in known) / max(1, len(known)),
                  "majority_baseline": Counter(x for x, _, _ in known).most_common(1)[0][1] / max(1, len(known)) if known else None}
        for g in sorted({p[2] for p in pairs}):
            kg = [(x, y) for x, y, gg in known if gg == g]
            per_group[g][f] = round(sum(x == y for x, y in kg) / max(1, len(kg)), 3) if kg else None
    return {"overall": per, "by_group": dict(per_group)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=path/to/predictions.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=0, help="print N random rows with their extracted facts (hand check)")
    args = ap.parse_args()
    res = {}
    for spec in args.runs:
        name, path = spec.split("=", 1)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        res[name] = agreement(rows)
        o = res[name]["overall"]
        print(f"{name:12s} " + "  ".join(f"{f} {o[f]['agree']:.3f} (maj {o[f]['majority_baseline']:.2f}, n {o[f]['n_known']})" for f in FACTS))
    if args.show:
        rows = [json.loads(l) for l in open(args.runs[0].split("=", 1)[1], encoding="utf-8")]
        for r in random.Random(0).sample(rows, args.show):
            print("\n", r["id"], r.get("group"), "| T:", facts(r["teacher_explanation"]), "| M:", facts(r["gen_explanation"]))
            print("   T:", r["teacher_explanation"][:220]); print("   M:", (r["gen_explanation"] or "")[:220])
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "fact_agreement.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
