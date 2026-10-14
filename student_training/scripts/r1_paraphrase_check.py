"""
r1_paraphrase_check.py - text augmentation of the Nexar reasoning dataset (plan 2026-10-14_Plan-Phase1-Enlarged-Dataset, section 1a).

  export : write paraphrase batches (outputs/teacher_dataset_2026-10/paraphrase_batches/batch_NN.jsonl): one row per window with the
           original explanation + outcome sentence, the verdict, and the 4 facts the rephrasings must keep.
  check  : validate the rephrased versions written by the Claude subagents (paraphrase_out/batch_NN.jsonl) and merge the accepted ones into
           outputs/teacher_dataset_2026-10/paraphrases.jsonl; rejected versions go to paraphrase_rejects.jsonl (to rewrite), versions accepted
           but with a risky new word go to paraphrase_flags.jsonl (for the Opus verifier).

Checks per version (explanation e, outcome o):
  * 25 <= words(e) <= 45 ; words(o) <= 15
  * no outcome word in e (collide / collision / crash / contact / impact / hit / strike)
  * type and side (r1_fact_eval.facts) equal those of the original (hard); motion / gap differences and added / dropped facts are FLAGGED, not rejected
    (the word lists are noisy there): the Opus verifier reads every flagged version
  * o keeps the polarity of the original outcome (collision vs no collision)
  * not a near copy: token-F1(e, original) < 0.8 and between the versions of a window < 0.85
  * risky new content (flag, not reject): a colour / object / place word absent from the original

    python r1_paraphrase_check.py export [--batch 300]
    python r1_paraphrase_check.py check
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from r1_eval_gates import token_f1  # noqa: E402
from r1_eval_reasoning import outcome_says_collision  # noqa: E402
from r1_fact_eval import FACTS, facts  # noqa: E402

MAN = ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_windows.jsonl"
D = ROOT / "outputs" / "teacher_dataset_2026-10"
BATCHES, OUTS = D / "paraphrase_batches", D / "paraphrase_out"
OUTCOME_WORDS = r"collid|collision|crash|\bcontact|impact|\bhit\b|\bhits\b|strike|struck"
RISKY = {  # word -> group; a version may use a word of a group only if the original uses a word of the same group
    **{c: "colour:" + c for c in ("white", "black", "silver", "grey", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige", "gold", "dark", "light-colored")},
    **{w: "obj:" + g for g, ws in {"pedestrian": ("pedestrian", "person", "people", "walker", "worker"), "cyclist": ("cyclist", "bicycle", "bike", "scooter"),
                                   "motorcycle": ("motorcycle", "motorbike"), "bus": ("bus",), "truck": ("truck", "lorry", "trailer", "semi", "tanker", "dump"),
                                   "van": ("van", "minivan"), "pickup": ("pickup",), "suv": ("suv", "crossover", "jeep"), "animal": ("dog", "deer", "animal")}.items() for w in ws},
    **{w: "place:" + g for g, ws in {"intersection": ("intersection", "junction", "crossroads"), "crosswalk": ("crosswalk", "crossing"),
                                     "highway": ("highway", "freeway", "motorway", "interstate"), "parking": ("parking", "lot", "garage"),
                                     "roundabout": ("roundabout",), "bridge": ("bridge",), "tunnel": ("tunnel",), "light": ("traffic", "signal"),
                                     "rain": ("rain", "wet", "snow"), "night": ("night", "dark", "nighttime")}.items() for w in ws},
}


def words(t):
    return re.findall(r"[a-z]+(?:-[a-z]+)?", (t or "").lower())


def risky_groups(t):
    return {RISKY[w] for w in words(t) if w in RISKY}


def export(batch):
    BATCHES.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(MAN, encoding="utf-8")]
    out = []
    for r in rows:
        g = r["gt"]
        f = facts(g["explanation"])
        out.append({"id": r["id"], "split": r["split"], "explanation": g["explanation"], "outcome": g["prediction"],
                    "verdict": "yes" if g["label_vis"] else "no", "keep_facts": {k: v for k, v in f.items() if v}})
    for i in range(0, len(out), batch):
        (BATCHES / f"batch_{i // batch:02d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out[i:i + batch]), encoding="utf-8")
    print(f"[export] {len(out)} windows -> {len(range(0, len(out), batch))} batches in {BATCHES}")


def check_version(orig_e, orig_o, e, o):
    """-> (problems that reject, flags for the verifier)"""
    bad, flags = [], []
    n = len(e.split())
    if not 25 <= n <= 45:
        bad.append(f"length {n}")
    if len(o.split()) > 15:
        bad.append(f"outcome length {len(o.split())}")
    if re.search(OUTCOME_WORDS, e.lower()):
        bad.append("outcome word in explanation")
    fo, fv = facts(orig_e), facts(e)
    for k in FACTS:                                                  # type / side: reliable word lists -> hard reject; motion / gap: noisy -> flag for the verifier
        if fo[k] is not None and fv[k] is not None and fv[k] != fo[k]:
            (bad if k in ("type", "side") else flags).append(f"{k}: {fo[k]} -> {fv[k]}")
        elif fo[k] is None and fv[k] is not None:
            flags.append(f"{k} added: {fv[k]}")
        elif fo[k] is not None and fv[k] is None and k in ("type", "side"):
            flags.append(f"{k} not stated: {fo[k]} -> None")
    if outcome_says_collision(o) != outcome_says_collision(orig_o):
        bad.append("outcome polarity changed")
    if token_f1(e, orig_e) >= 0.8:
        bad.append("near copy")
    new = risky_groups(e + " " + o) - risky_groups(orig_e + " " + orig_o)
    if new:
        flags.append("new content: " + ", ".join(sorted(new)))
    return bad, flags


def check():
    orig = {json.loads(l)["id"]: json.loads(l) for p in sorted(BATCHES.glob("batch_*.jsonl")) for l in open(p, encoding="utf-8")}
    acc, rej, flg, seen = [], [], [], set()
    for p in sorted(OUTS.glob("batch_*.jsonl")):
        for l in open(p, encoding="utf-8"):
            if not l.strip():
                continue
            x = json.loads(l)
            o = orig.get(x["id"])
            if o is None or x["id"] in seen:
                continue
            seen.add(x["id"])
            kept = []
            for k, v in enumerate(x.get("versions", []), 1):
                e, oc = v.get("explanation", "").strip(), v.get("outcome", "").strip()
                bad, flags = check_version(o["explanation"], o["outcome"], e, oc)
                if not bad and any(token_f1(e, z["explanation"]) >= 0.85 for z in kept):
                    bad.append("near copy of another version")
                rec = {"id": x["id"], "version": k, "explanation": e, "outcome": oc}
                if bad:
                    rej.append({**rec, "problems": bad, "original": o["explanation"]})
                    continue
                kept.append(rec)
                if flags:
                    flg.append({**rec, "flags": flags, "original": o["explanation"], "original_outcome": o["outcome"]})
            for k, r in enumerate(kept, 1):
                acc.append({"id": r["id"], "source": f"rephrased_v{k}", "explanation": r["explanation"], "outcome": r["outcome"],
                            "flagged": any(f["id"] == r["id"] and f["version"] == r["version"] for f in flg)})
    (D / "paraphrases.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in acc), encoding="utf-8")
    (D / "paraphrase_rejects.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rej), encoding="utf-8")
    (D / "paraphrase_flags.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in flg), encoding="utf-8")
    per = Counter(r["id"] for r in acc)
    reasons = Counter(p.split(":")[0] for r in rej for p in r["problems"])
    n_ver = len(acc) + len(rej)
    print(f"[check] windows returned {len(seen)} of {len(orig)}; versions {n_ver}: accepted {len(acc)}, rejected {len(rej)} "
          f"({len(rej) / max(1, n_ver):.1%}), flagged {len(flg)}")
    print(f"        windows with 3 accepted: {sum(1 for v in per.values() if v >= 3)}; with < 3: {len(seen) - sum(1 for v in per.values() if v >= 3)}")
    print("        reject reasons:", dict(reasons.most_common(8)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "check"])
    ap.add_argument("--batch", type=int, default=300)
    a = ap.parse_args()
    export(a.batch) if a.cmd == "export" else check()
