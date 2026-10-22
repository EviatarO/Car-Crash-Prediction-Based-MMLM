"""
build_pres_data.py -> pres_data.js   (the PRESENTATION site, website_pres/; plan 2026-10-20_Plan-Presentation-Website-Home)

Reads ONLY what promoted.json lists (the lab site website/ keeps every experiment; promotion = editing promoted.json on the user's request) and writes
window.PRES_DATA = {showcase, vision, reasoning, names}. Every number is computed here from its source file, never hand-copied, and the promoted models' headline
numbers are asserted against promoted.json "expect" so a changed score file fails loudly instead of changing the page.

    python website_pres/build_pres_data.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                                   # MMLM_AI
sys.path.insert(0, str(ROOT / "website"))
sys.path.insert(0, str(ROOT / "student_training" / "scripts"))
from build_site_data import build_train  # noqa: E402
from r1_full_report import card_acc, cards, written  # noqa: E402
from r1_test_report import metrics  # noqa: E402

TTES = (0.5, 1.0, 1.5)


def jl(p):
    return [json.loads(l) for l in open(ROOT / p, encoding="utf-8") if l.strip()]


def num(x):
    return float(x) if not isinstance(x, str) else float(x.strip())


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


# ----------------------------------------------------------------------------- showcase
def build_showcase(cfg):
    clips = {c["id"]: c for c in build_train()}
    win = {w["frames_dir"]: w for w in jl(cfg["windows"])}
    out, skipped = [], {"wrong": 0, "no_video": 0, "no_window": 0}
    for r in jl(cfg["predictions"]):
        y, s = int(num(r["label"])), num(r["s_first"])
        verdict = int(num(r["gen_collision"])) if r.get("gen_collision") not in (None, "") else None
        if verdict != y or (s >= 0) != (y == 1) or not r.get("gen_explanation"):
            skipped["wrong"] += 1
            continue
        c = clips.get(r["video_key"])
        if c is None or c["video_missing"]:
            skipped["no_video"] += 1
            continue
        w = win.get(r["frames_dir"])
        if w is None:
            skipped["no_window"] += 1
            continue
        tte = num(r["tte"]) if str(r["group"]).startswith("tte") else None
        out.append({"id": r["video_key"], "video": c["video"], "label": y, "tte": tte, "t_end": round(float(w["t_end_s"]), 3),
                    "event": c["time_of_event"] if y == 1 else None, "reasoning": r["gen_explanation"].strip(), "outcome": (r.get("gen_outcome") or "").strip(),
                    "p": round(sigmoid(s), 4), "verdict": verdict})
    crash = sum(o["label"] for o in out)
    print(f"[showcase] {len(out)} correct windows with a local video ({crash} crash, {len(out) - crash} normal); skipped {skipped}")
    assert len(out) >= 500 and crash >= 150, "too few showcase windows: check the promoted predictions file"
    assert all(o["event"] is None or o["event"] > o["t_end"] for o in out), "a crash window must end before its event"
    return out


# ----------------------------------------------------------------------------- vision (Kaggle benchmark)
def build_vision(cfg):
    rows = []
    for m in cfg["measured"]:
        R = [r for f in m["files"] for r in jl(f)]
        y = np.array([int(num(r[m["label"]])) for r in R])
        t = np.array([num(r[m["tte"]]) for r in R])
        s = np.array([num(r[m["score"]]) for r in R])
        assert len(R) == 1344 and set(np.round(t, 2)) == set(TTES), f"{m['name']}: expected 1,344 test windows with TTE 0.5 / 1.0 / 1.5"
        r = metrics(y, t, s)
        for k, v in (m.get("expect") or {}).items():
            assert abs(r[k] - v) < 6e-4, f"{m['name']}: {k} {r[k]:.4f} differs from the promoted value {v} (score file changed?)"
        rows.append({"name": m["name"], "kind": m["kind"], "ap": [round(float(a), 3) for a in r["AP_tte"]], "mAP": round(float(r["kaggle_mAP"]), 3),
                     "source": "measured by us", "note": m["note"]})
        print(f"[vision] {m['name']}: AP@TTE {', '.join(f'{a:.3f}' for a in r['AP_tte'])}  mAP {r['kaggle_mAP']:.4f}  (overall AP {r['AP']:.4f})")
    for p in cfg["reported"]:
        rows.append({"name": p["name"], "kind": "paper", "ap": p["ap"], "mAP": p["mAP"], "source": p["source"], "partial": p.get("partial", False)})
    return {"benchmark": cfg["benchmark"], "rows": rows, "footnotes": cfg["footnotes"]}


# ----------------------------------------------------------------------------- reasoning
def build_reasoning(cfg):
    val, test = jl(cfg["val_predictions"]), jl(cfg["test_predictions"])
    tv, tt = (json.load(open(ROOT / cfg[k], encoding="utf-8")) for k in ("val_text_metrics", "test_text_metrics"))
    vm = json.load(open(ROOT / cfg["val_metrics"], encoding="utf-8"))
    cv = card_acc(cards(ROOT / cfg["val_cards"]), cards(ROOT / cfg["teacher_cards_val"]))
    ct = card_acc(cards(ROOT / cfg["test_cards"]), cards(ROOT / cfg["teacher_cards_test"]))
    wv, wt = written(val), written(test)
    names = {"road_user": "road user", "side": "side", "motion": "motion", "gap": "gap trend"}
    rows = []
    for s, lab in names.items():
        rows.append({"metric": f"Fact agreement with the teacher: {lab}", "group": "facts", "val": cv[s]["acc"], "test": ct[s]["acc"],
                     "base_val": cv[s]["baseline"], "base_test": ct[s]["baseline"], "base_name": "always the most common value"})
    rows.append({"metric": "Meaning similarity to the teacher (BERTScore, rescaled)", "group": "text", "val": tv["bertscore_f1_rescaled"]["model"],
                 "test": tt["bertscore_f1_rescaled"]["model"], "base_val": tv["bertscore_f1_rescaled"]["floor"], "base_test": tt["bertscore_f1_rescaled"]["floor"],
                 "base_name": "another clip's teacher text"})
    rows.append({"metric": "Written verdict agrees with the label", "group": "decision", "val": wv["verdict_acc_vs_label"], "test": wt["verdict_acc_vs_label"],
                 "base_val": None, "base_test": None, "base_name": ""})
    rows.append({"metric": "Distinct explanations (share of windows)", "group": "text", "val": wv["distinct"], "test": wt["distinct"], "base_val": None, "base_test": None, "base_name": ""})
    rows.append({"metric": "Text depends on the video: loss gap with another clip (nats)", "group": "video", "val": vm["after"]["gap_wrong"], "test": None,
                 "base_val": 0.0, "base_test": None, "base_name": "0 = the text ignores the video"})
    # pinned values from outputs/r1_p2v_report_2026-10-18/report.md (section 3)
    pin = {"Fact agreement with the teacher: side": (0.65, 0.65), "Meaning similarity to the teacher (BERTScore, rescaled)": (0.425, 0.424)}
    for r in rows:
        if r["metric"] in pin:
            assert abs(r["val"] - pin[r["metric"]][0]) < 0.006 and abs(r["test"] - pin[r["metric"]][1]) < 0.006, f"reasoning metric drifted: {r}"
    print(f"[reasoning] val n={len(val)}, test n={len(test)}; " + "; ".join(f"{r['metric'][:28]} {r['val']:.3f}/{r['test'] if r['test'] is None else round(r['test'], 3)}" for r in rows))
    return {"n_val": len(val), "n_test": len(test), "test_note": cfg["test_note"], "rows": rows}


def main():
    cfg = json.load(open(HERE / "promoted.json", encoding="utf-8"))
    data = {"names": {"reasoning": cfg["reasoning_model"], "crash": cfg["crash_model"]}, "showcase": build_showcase(cfg["showcase"]),
            "vision": build_vision(cfg["vision"]), "reasoning": build_reasoning(cfg["reasoning"])}
    out = HERE / "pres_data.js"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("window.PRES_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"[wrote] {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
