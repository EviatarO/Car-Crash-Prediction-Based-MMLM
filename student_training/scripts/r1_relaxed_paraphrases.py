"""
r1_relaxed_paraphrases.py - arm D2 of plan 2026-10-20 (more diverse rewordings): the judge-accepted versions PLUS the judge-rejected versions that only broke the
"nuance" rules 3-6 (motion word, gap strength, a detail added / dropped) and do not use a contact word that the original text lacks.

Never admitted: a version with a hard-fact problem (rules 1 road-user kind, 2 position / side, 7 outcome meaning, 8 the explanation already says a collision happens),
a version with no recognisable rule number, a version with a contact-type word (bumper, meet, touch, against, ...) that the teacher's text does not have
(it would leak the outcome; the Opus audit removed such versions), duplicates, and everything the AUTOMATIC check rejected (near copies, outcome words, added type / side).
At most --max-extra relaxed versions are added per window (file order), tagged source rephrased_x1.. and judge "relaxed".

  python r1_relaxed_paraphrases.py \
      --judged ../../outputs/teacher_dataset_2026-10/paraphrases_judged.jsonl \
      --rejects ../../outputs/teacher_dataset_2026-10/paraphrase_judge_rejects.jsonl \
      --manifest ../../dataset/manifests/r1_nexar_reasoning_windows.jsonl \
      --out ../../outputs/teacher_dataset_2026-10/paraphrases_relaxed.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

NUANCE = set("3456")
CONTACT_RE = re.compile(r"\b(bumpers?|meets?|meeting|touch\w*|contact\w*|against|collid\w*|collision|hits?|hitting|strik\w*|struck|rams?|ramming|slams?|slamming|smash\w*|impacts?|crash\w*|rear-end\w*)\b", re.I)


def rule_numbers(problem: str) -> set[str]:
    """Rule numbers named at the start of a judge reason ("Rule 5/3: ...", "Rules 3 and 8: ..."); empty if none can be read."""
    m = re.match(r"\s*Rules?\s*([\d/ ,and&]+)", problem or "", re.I)
    head = m.group(1) if m else (problem or "").split(":")[0]
    return set(re.findall(r"\d", head))


def contact_words(text: str) -> set[str]:
    return {w.lower() for w in CONTACT_RE.findall(text or "")}


def norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def relaxed_rows(judged, rejects, orig_expl, max_extra=3):
    """judged: accepted rows (kept unchanged); rejects: judge-rejected rows {id, explanation, outcome, problem}; orig_expl: {window id: teacher explanation}.
    Returns (rows, stats): accepted rows followed by the admitted relaxed rows."""
    rows = list(judged)
    seen = collections.defaultdict(set)
    for r in judged:
        seen[r["id"]].add(norm(r["explanation"]))
    added = collections.Counter()
    stats = collections.Counter()
    for r in rejects:
        stats["rejects"] += 1
        rn = rule_numbers(r.get("problem", ""))
        if not rn:
            stats["no_rule_number"] += 1
            continue
        if not rn <= NUANCE:
            stats["hard_fact_rule"] += 1
            continue
        wid = r["id"]
        orig = orig_expl.get(wid)
        if orig is None:
            stats["no_original"] += 1
            continue
        if contact_words(r["explanation"]) - contact_words(orig):
            stats["new_contact_word"] += 1
            continue
        key = norm(r["explanation"])
        if key in seen[wid] or key == norm(orig):
            stats["duplicate"] += 1
            continue
        if added[wid] >= max_extra:
            stats["over_cap"] += 1
            continue
        added[wid] += 1
        seen[wid].add(key)
        rows.append({"id": wid, "source": f"rephrased_x{added[wid]}", "explanation": r["explanation"], "outcome": r["outcome"], "judge": "relaxed"})
        stats["admitted"] += 1
    stats["windows_with_extra"] = len(added)
    return rows, dict(stats)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", required=True)
    ap.add_argument("--rejects", required=True)
    ap.add_argument("--manifest", required=True, help="reasoning manifest: the teacher text of every window is gt.explanation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-extra", type=int, default=3)
    a = ap.parse_args()
    orig = {m["id"]: m["gt"]["explanation"] for m in jl(a.manifest)}
    judged = jl(a.judged)
    rows, stats = relaxed_rows(judged, jl(a.rejects), orig, a.max_extra)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per = collections.Counter(r["id"] for r in rows)
    print(f"{a.out}: {len(judged)} accepted + {stats.get('admitted', 0)} relaxed = {len(rows)} versions on {len(per)} windows (max {max(per.values())} per window)")
    print("rejected versions read:", {k: v for k, v in stats.items() if k != "admitted"})


if __name__ == "__main__":
    main()
