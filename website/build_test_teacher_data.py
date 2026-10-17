"""
build_test_teacher_data.py -> test_teacher_data.js   (Dataset page, tab "Teacher reasoning - test (private)"; plan 2026-10-17_Plan-Test-Private-Reasoning, step 7)

One row per private-test window (677). Reads outputs/teacher_test_2026-10/:
  windows.jsonl, boxes.jsonl (r1_attention_boxes.py --stage boxes: the crash model's P and the box record), review/<frames_dir>.jpg (boxed 4x4 sheets),
  runs/E3.jsonl + runs/E3P2.jsonl (blind first look + its outcome sentence), runs/E8.jsonl / runs/E8_pilot.jsonl (label-informed second look),
  dataset_test_final.jsonl (the merged reference set), fact_cards_test.jsonl (Gemini fact cards of the final texts, optional).
Missing files are fine: the page shows what exists (before step 3 it is just the windows).

    python build_test_teacher_data.py
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
T = ROOT / "outputs" / "teacher_test_2026-10"
EVAL = ROOT / "outputs" / "r1_test_eval_2026-10-17" / "test_eval_2026-10-17"      # decision-only test evaluation of P2R / P2V (no reference text)
REL = "../outputs/teacher_test_2026-10"


def jl(p):
    out = []
    if Path(p).exists():
        for l in open(p, encoding="utf-8"):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def main():
    wins = jl(T / "windows.jsonl")
    boxes = {r["frames_dir"]: r for r in jl(T / "boxes.jsonl")}
    e3 = {r["frames_dir"]: r for r in jl(T / "runs" / "E3.jsonl") if r.get("parsed")}
    p2 = {r["frames_dir"]: r.get("prediction") for r in jl(T / "runs" / "E3P2.jsonl")}
    e8 = {r["frames_dir"]: r for r in jl(T / "runs" / "E8_pilot.jsonl") if r.get("parsed")}
    e8.update({r["frames_dir"]: r for r in jl(T / "runs" / "E8.jsonl") if r.get("parsed")})
    final = {r["frames_dir"]: r for r in jl(T / "dataset_test_final.jsonl")}
    cards = {r["id"]: {k: r[k] for k in ("road_user", "side", "motion", "gap")} for r in jl(T / "fact_cards_test.jsonl")}
    models = {}
    for name in ("P2R", "P2V"):
        for r in jl(EVAL / name / "results" / "predictions.jsonl"):
            fd = r["id"].replace("nexartest_", "", 1) + "_hires"
            sc = r.get("s_first", r.get("s_after"))
            models.setdefault(fd, {})[name] = {"p": round(1 / (1 + math.exp(-sc)), 4), "v": "yes" if sc >= 0 else "no", "expl": r.get("gen_explanation"),
                                               "out": r.get("gen_outcome"), "wv": "yes" if r.get("gen_collision") == 1 else "no"}
    rows, cnt = [], collections.Counter()
    for w in sorted(wins, key=lambda w: (w["horizon"], w["video_id"])):
        fd = w["frames_dir"]
        b = boxes.get(fd) or {}
        a, s, f = e3.get(fd), e8.get(fd), final.get(fd)
        pa, ps = (a or {}).get("parsed"), (s or {}).get("parsed")
        grp = "normal" if not w["label"] else f"tte{w['horizon']}"
        row = {"key": fd, "video": w["video_id"], "label": int(w["label"]), "tte": w["horizon"], "group": grp, "a1_p": b.get("a1_p_collision"),
               "box_flag": b.get("flag"), "grid": f"{REL}/review/{fd}.jpg" if (T / "review" / f"{fd}.jpg").exists() else None,
               "blind": None if not pa else {"v": pa["collision"], "risk": pa.get("risk_score"), "expl": pa.get("explanation"), "pred": p2.get(fd) or pa.get("prediction"),
                                             "problems": a.get("problems") or []},
               "second": None if not ps else {"v": ps["collision"], "risk": ps.get("risk_score"), "hv": ps.get("hazard_visible"), "first": ps.get("first_visible_frame"),
                                              "agent": ps.get("involved_agent"), "cues": ps.get("early_cues"), "expl": ps.get("explanation"), "pred": ps.get("prediction"),
                                              "problems": s.get("problems") or []},
               "final": None if not f else {"source": f["source"], "v": f["verdict"], "expl": f["explanation"], "pred": f["prediction"], "hv": f.get("hazard_visible"),
                                            "edited": bool(f.get("edited"))},
               "card": cards.get(fd), "models": models.get(fd)}
        row["blind_wrong"] = bool(pa and (pa["collision"] == "yes") != bool(w["label"]))
        cnt["blind"] += bool(pa)
        cnt["second"] += bool(ps)
        cnt["blind_wrong"] += row["blind_wrong"]
        rows.append(row)
    n_crash = sum(r["label"] for r in rows)
    sec = [r["second"]["hv"] for r in rows if r["second"]]
    summary = {"n": len(rows), "n_crash": n_crash, "n_normal": len(rows) - n_crash, "blind": cnt["blind"], "second": cnt["second"], "blind_wrong": cnt["blind_wrong"],
               "hv": dict(collections.Counter(sec)), "groups": dict(collections.Counter(r["group"] for r in rows)),
               "final": dict(collections.Counter(r["final"]["source"] for r in rows if r["final"]))}
    # confusion matrices at P = 0.5 per TTE (crash windows of that TTE vs the normal windows of the same TTE) and overall
    cm = []
    for key, label in (("crash", "Crash model"), ("P2R", "P2R reason-first"), ("P2V", "P2V verdict-first")):
        for tte in (0.5, 1.0, 1.5, "all"):
            sel = [r for r in rows if tte == "all" or r["tte"] == tte]
            tp = fn = fp = tn = 0
            for r in sel:
                p = r["a1_p"] if key == "crash" else ((r["models"] or {}).get(key) or {}).get("p")
                if p is None:
                    continue
                yes = p >= 0.5
                if r["label"]:
                    tp += yes; fn += not yes
                else:
                    fp += yes; tn += not yes
            if tp + fn + fp + tn:
                cm.append({"model": key, "label": label, "tte": tte, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                           "recall": tp / max(1, tp + fn), "specificity": tn / max(1, tn + fp), "precision": tp / max(1, tp + fp)})
    summary["cm"] = cm
    out = HERE / "test_teacher_data.js"
    out.write_text("window.TEST_TEACHER_DATA = " + json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"test_teacher_data.js: {len(rows)} windows, {out.stat().st_size / 1e6:.2f} MB; {summary}")


if __name__ == "__main__":
    main()
