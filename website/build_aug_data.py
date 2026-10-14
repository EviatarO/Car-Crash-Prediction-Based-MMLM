"""
build_aug_data.py -> aug_data.js   (Dataset page, tab "Reasoning dataset - augmented"; plan 2026-10-14_Plan-Phase1-Enlarged-Dataset + Text-Judge-Addendum)

One row per original window of the Nexar reasoning dataset (4,169), with:
  the teacher's text (source tag "teacher"), every rephrased version ever written for it, each with
    writer   claude (Sonnet subagents, first 1,329 windows) | gemini (Gemini 3.8 Flash, the rest + the rewrite round)
    status   used (tag rephrased_v1..v3 = in paraphrases_judged.jsonl, the training input) | spare (judge ok, a 4th+ version) |
             judge (rejected by the Sonnet 5.5 judge, with its reason) | auto (rejected by the automatic check, with its reasons)
    audit    the Opus audit verdict when the version is in the 300-version audit sample
  the window versions: the original + its frame-shifted copies (shift 1-3 source frames earlier; frames not on the PC, ids and end times only)
Sources: dataset/manifests/r1_nexar_reasoning_{windows,shifted}.jsonl, outputs/teacher_dataset_2026-10/{paraphrase_out/*, paraphrases.jsonl,
paraphrase_rejects.jsonl, paraphrase_judge.jsonl, paraphrases_judged.jsonl, paraphrase_audit_sample.jsonl, audit/audit_out_*.jsonl}.

    python build_aug_data.py
"""
from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
D = ROOT / "outputs" / "teacher_dataset_2026-10"
MAN = ROOT / "dataset" / "manifests"
REL = "../outputs/teacher_dataset_2026-10"


def jl(p):
    out = []
    if p.exists():
        for l in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:                         # empty / marker lines of interrupted writers (same rule as r1_paraphrase_check.py)
                pass
    return out


def key_of(i, e, o):
    return hashlib.sha1(f"{i}|{e}|{o}".encode("utf-8")).hexdigest()[:12]


def main():
    man = jl(MAN / "r1_nexar_reasoning_windows.jsonl")
    shifts = collections.defaultdict(list)
    for r in jl(MAN / "r1_nexar_reasoning_shifted.jsonl"):
        shifts[r["source_id"]].append(r["shift"])
    written = collections.defaultdict(list)                     # id -> [(writer, explanation, outcome)] in file order
    for p in sorted((D / "paraphrase_out").glob("*.jsonl")):
        if p.name.startswith("gemtest"):
            writer = "gemini"
        else:
            writer = "gemini" if p.name.startswith("gem") else "claude"
        for x in jl(p):
            for v in x.get("versions", []):
                if isinstance(v, dict) and v.get("explanation"):
                    written[x["id"]].append((writer, v["explanation"].strip(), v.get("outcome", "").strip()))
    auto_rej = {(r["id"], r["explanation"]): r["problems"] for r in jl(D / "paraphrase_rejects.jsonl")}
    judge = {r["key"]: r for r in jl(D / "paraphrase_judge.jsonl")}
    used = {(r["id"], r["explanation"]): r["source"] for r in jl(D / "paraphrases_judged.jsonl")}
    audit_out = {}
    for p in sorted((D / "audit").glob("audit_out_*.jsonl")):
        audit_out.update({r["key"]: r for r in jl(p)})
    rows, tally = [], collections.Counter()
    for m in man:
        g = m["gt"]
        vers, seen = [], set()
        for w, e, o in written.get(m["id"], []):
            if e in seen:
                continue
            seen.add(e)
            k = key_of(m["id"], e, o)
            if (m["id"], e) in used:
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
            vers.append({"w": w, "e": e, "o": o, "st": st, "why": why,
                         "au": None if a is None else ("ok" if a["ok"] else f'{a.get("severity", "?")}: {a.get("problem", "")}')})
            tally[st if not st.startswith("rephrased") else "used"] += 1
        n_used = sum(v["st"].startswith("rephrased") for v in vers)
        fd = m["frames_dir"]
        rows.append({"id": m["id"], "split": m["split"], "lv": g["label_vis"], "nl": m["label"], "pre": bool(g.get("pre_alert")),
                     "grp": "normal" if not m["label"] else ("pre_alert" if g.get("pre_alert") else f"tte{m['tte']}"), "e": g["explanation"], "o": g["prediction"], "nu": n_used, "sh": sorted(shifts.get(m["id"], [])),
                     "grid": f"{REL}/review/{fd}.jpg" if (D / "review" / f"{fd}.jpg").exists() else None,
                     "gridL": f"{REL}/grid_large/{fd}.jpg" if (D / "grid_large" / f"{fd}.jpg").exists() else None, "v": vers})
    sample = jl(D / "paraphrase_audit_sample.jsonl")
    acc = [s for s in sample if s["judge_ok"] and s["key"] in audit_out]
    rej = [s for s in sample if not s["judge_ok"] and s["key"] in audit_out]
    sev = collections.Counter("ok" if audit_out[s["key"]]["ok"] else audit_out[s["key"]].get("severity", "?") for s in acc)
    summary = {
        "windows": len(rows), "shifted": sum(len(r["sh"]) for r in rows), "tally": dict(tally),
        "good_hist": dict(collections.Counter(r["nu"] for r in rows)),
        "audit": {"n_accepted": len(acc), "acc_ok": sev["ok"], "acc_minor": sev["minor"], "acc_major": sev["major"],
                  "n_rejected": len(rej), "rej_opus_ok": sum(audit_out[s["key"]]["ok"] for s in rej)},
    }
    out = HERE / "aug_data.js"
    out.write_text("window.AUG_DATA = " + json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"aug_data.js: {len(rows)} windows, {out.stat().st_size / 1e6:.1f} MB; versions {dict(tally)}; good per window {summary['good_hist']}; "
          f"audit {summary['audit']}; grids {sum(bool(r['grid']) for r in rows)}")


if __name__ == "__main__":
    main()
