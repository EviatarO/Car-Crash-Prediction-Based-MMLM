"""
build_aug_extra_data.py -> aug_extra_data.js   (Dataset page, tab "Reasoning dataset - new windows (augmented)"; plan 2026-10-19)

The rephrased versions of the 860 new crash windows (TTE 0.75 / 1.25 s), in the same view as the "Reasoning dataset - augmented" tab (assets/aug.js): one row per window with the
teacher's text and EVERY version Gemini 3.8 Flash wrote for it, with its status:
    used     passed the automatic check and the Sonnet 5.5 judge (tags rephrased_v1..v3; the training input for the TRAIN windows)
    spare    judge ok, a 4th+ version
    judge    rejected by the Sonnet judge, with its reason
    auto     rejected by the automatic check, with its reasons
    removed  accepted by the judge but removed after the Opus audit (a contact word such as "bumpers meet" that the original lacks; audit rule 8)
and the Opus audit verdict when the version is in the 300-version audit sample.
Sources: dataset/manifests/r1_nexar_reasoning_extra.jsonl, outputs/teacher_extra_2026-10/paraphrase/{paraphrase_out/*, paraphrase_rejects.jsonl, paraphrase_judge.jsonl,
paraphrases_judged.jsonl, removed_contact_words.json, paraphrase_audit_sample.jsonl, audit/audit_out_*.jsonl}, review/ and grid_large/ (frame sheets).

    python build_aug_extra_data.py
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
W = ROOT / "outputs" / "teacher_extra_2026-10"
P = W / "paraphrase"
REL = "../outputs/teacher_extra_2026-10"


def jl(p):
    out = []
    if Path(p).exists():
        for l in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def key_of(i, e, o):
    return hashlib.sha1(f"{i}|{e}|{o}".encode("utf-8")).hexdigest()[:12]


def main():
    man = jl(ROOT / "dataset" / "manifests" / "r1_nexar_reasoning_extra.jsonl")
    ids = {m["id"] for m in man}
    written = collections.defaultdict(list)                    # id -> [(round, explanation, outcome)] in file order
    for p in sorted((P / "paraphrase_out").glob("*.jsonl")):
        rnd = "rewrite" if "rewrite" in p.name else "first"
        for x in jl(p):
            for v in x.get("versions", []):
                if isinstance(v, dict) and v.get("explanation"):
                    written[x["id"]].append((rnd, v["explanation"].strip(), v.get("outcome", "").strip()))
    auto_rej = {(r["id"], r["explanation"].strip()): r["problems"] for r in jl(P / "paraphrase_rejects.jsonl")}
    judge = {r["key"]: r for r in jl(P / "paraphrase_judge.jsonl")}
    used = {(r["id"], r["explanation"].strip()): r["source"] for r in jl(P / "paraphrases_judged.jsonl")}
    removed = {(r["id"], r["explanation"].strip()) for r in json.load(open(P / "removed_contact_words.json", encoding="utf-8"))["removed"]}
    audit_out = {}
    for p in sorted((P / "audit").glob("audit_out_*.jsonl")):
        audit_out.update({r["key"]: r for r in jl(p)})
    rows, tally = [], collections.Counter()
    for m in man:
        g = m["gt"]
        vers, seen = [], set()
        for rnd, e, o in written.get(m["id"], []):
            if e in seen:
                continue
            seen.add(e)
            k = key_of(m["id"], e, o)
            if (m["id"], e) in removed:
                st, why = "removed", "removed after the Opus audit: a contact word (e.g. 'bumpers meet') that the original text does not have, i.e. it leaks the outcome"
            elif (m["id"], e) in used:
                st, why = used[(m["id"], e)], ""
            elif (m["id"], e) in auto_rej:
                st, why = "auto", "; ".join(auto_rej[(m["id"], e)])
            elif k in judge and not judge[k]["ok"]:
                st, why = "judge", judge[k].get("problem", "")
            elif k in judge:
                st, why = "spare", ""
            else:
                st, why = "unjudged", ""
            a = audit_out.get(k)
            vers.append({"w": rnd + " round", "e": e, "o": o, "st": st, "why": why,
                         "au": None if a is None else ("ok" if a["ok"] else f'{a.get("severity", "?")}: {a.get("problem", "")}')})
            tally[st if not st.startswith("rephrased") else "used"] += 1
        n_used = sum(v["st"].startswith("rephrased") for v in vers)
        fd = m["frames_dir"]
        rows.append({"id": m["id"], "split": m["split"], "lv": g["label_vis"], "nl": m["label"], "pre": False, "grp": f"tte{m['tte']}", "e": g["explanation"], "o": g["prediction"],
                     "nu": n_used, "sh": [], "grid": f"{REL}/review/{fd}.jpg" if (W / "review" / f"{fd}.jpg").exists() else None,
                     "gridL": f"{REL}/grid_large/{fd}.jpg" if (W / "grid_large" / f"{fd}.jpg").exists() else None, "path": g.get("path"), "v": vers})
    rows.sort(key=lambda r: r["id"])
    sample = jl(P / "paraphrase_audit_sample.jsonl")
    acc = [s for s in sample if s["judge_ok"] and s["key"] in audit_out]
    rej = [s for s in sample if not s["judge_ok"] and s["key"] in audit_out]
    sev = collections.Counter("ok" if audit_out[s["key"]]["ok"] else audit_out[s["key"]].get("severity", "?") for s in acc)
    n_tr = sum(r["split"] == "train" for r in rows)
    used_tr = sum(r["nu"] for r in rows if r["split"] == "train")
    hist = collections.Counter(r["nu"] for r in rows)
    summary = {"kind": "extra", "windows": len(rows), "tally": dict(tally), "good_hist": dict(hist), "n_train": n_tr, "used_train": used_tr,
               "written": sum(tally.values()), "audit": {"n_accepted": len(acc), "acc_ok": sev["ok"], "acc_minor": sev["minor"], "acc_major": sev["major"],
                                                          "n_rejected": len(rej), "rej_opus_ok": sum(audit_out[s["key"]]["ok"] for s in rej)}}
    out = HERE / "aug_extra_data.js"
    out.write_text("window.AUG_EXTRA_DATA = " + json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"aug_extra_data.js: {len(rows)} windows, {out.stat().st_size / 1e6:.2f} MB; versions {dict(tally)}; used in train windows {used_tr}; good per window {dict(sorted(hist.items()))}; "
          f"audit {summary['audit']}; grids {sum(bool(r['grid']) for r in rows)}")
    assert not ids - {r["id"] for r in rows}


if __name__ == "__main__":
    main()
