"""
build_r2_data.py -> r2_data.js   (Experiments page, tab "Reasoning model": Run 1 + the three follow-ups, with a run selector)

Results of the reasoning model: Phase-2 fine training (projector + LoRA on Qwen3-VL-4B, frozen V-JEPA2 features) on the Nexar reasoning
dataset, scored on the 612 validation windows (204 videos never trained on). Everything is read from
outputs/r1_phase2r_2026-10-12/phase2r:
  results/metrics.json       decision metrics per score, per-group confusion, paired tests (McNemar + AUC difference), text block, perplexity, gaps, retrieval
  results/text_metrics.json  token-F1 / ROUGE-L / BLEU-4 / BERTScore, model vs floor
  train_log.jsonl            per-epoch train loss, validation loss, wrong-video gap, decision AUC
  results/predictions.jsonl  one row per validation window (scores are log-odds; P(yes) = sigmoid)
Follow-up runs (outputs/r1_options_2026-10-12/options_2026-10-12), same file formats:
  opt1_epoch3  Run 1's epoch 3 (lowest validation loss), results only (its train log is Run 1's)
  opt2_cap     retrain with the template cap, results + train_log.jsonl
  opt3_box     retrain with the position of the attended road user in the prompt, results + train_log.jsonl
The follow-up metrics.json files carry no "before_phase1" block and no old-Phase-2 score; the "before Phase 2" reference numbers are taken from Run 1.
The window grids are the teacher dataset's contact sheets (outputs/teacher_dataset_2026-10/review and grid_large), same REL as build_teacher_data.py.

    python build_r2_data.py
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUN = ROOT / "outputs" / "r1_phase2r_2026-10-12" / "phase2r"
OPT = ROOT / "outputs" / "r1_options_2026-10-12" / "options_2026-10-12"
GRIDS = ROOT / "outputs" / "teacher_dataset_2026-10"
REL = "../outputs/teacher_dataset_2026-10"
GENERIC_MIN = 10                      # an explanation shared by >= this many windows counts as "generic"
# key, selector label, description, results dir, train log (None = Run 1's), epoch of the evaluated checkpoint (None = the epoch of lowest validation loss)
RUNS = [
    ("run1", "Run 1 (epoch 5)", "Run 1: best.pt = epoch 5, chosen by the wrong-video gap", RUN / "results", RUN / "train_log.jsonl", 5),
    ("opt1", "Option 1 (epoch 3)", "Option 1: Run 1's epoch 3, the epoch with the lowest validation loss (same training run, earlier checkpoint)",
     OPT / "opt1_epoch3" / "results", None, 3),
    ("opt2", "Option 2 (template cap)", "Option 2: retrain with a template cap (frequent explanation openings down-weighted)",
     OPT / "opt2_cap" / "results", OPT / "opt2_cap" / "train_log.jsonl", None),
    ("opt3", "Option 3 (box position)", "Option 3: retrain with the position of the attended road user given as words in the prompt",
     OPT / "opt3_box" / "results", OPT / "opt3_box" / "train_log.jsonl", None),
]
GROUPS = [["normal", "normal"], ["pre_alert", "pre-alert"], ["tte1.5", "TTE 1.5 s"], ["tte1.0", "TTE 1.0 s"], ["tte0.5", "TTE 0.5 s"]]
SCORES = [["s_after", "S_after", "after the model's own reasoning"],
          ["s_before", "S_before", "reasoning cut out (decision straight after the frames)"],
          ["s_teacher", "S_teacher", "after the teacher's reasoning (upper bound)"],
          ["s_old_phase2", "old Phase 2", "the previous Phase-2 model, decision only"],
          ["a1_p", "crash model", "the frozen crash model (A1), probability"]]
N_TRAIN_WINDOWS = 3557                # train windows, from outputs/r1_phase2r_2026-10-12/summary.md
SATURATED = 4.0                       # |log-odds| above this = P(yes) below 0.018 or above 0.982


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def load_run(key, label, desc, res, logp, epoch, run1_m, log1):
    m = json.load(open(res / "metrics.json", encoding="utf-8"))
    tm = json.load(open(res / "text_metrics.json", encoding="utf-8"))
    preds = [json.loads(l) for l in open(res / "predictions.jsonl", encoding="utf-8") if l.strip()]
    log = [json.loads(l) for l in open(logp, encoding="utf-8") if l.strip()] if logp else log1
    mm = m["metrics"]
    assert len(preds) == m["n_windows"] == mm["n"], (key, len(preds), m["n_windows"])
    bp1 = m.get("before_phase1") or run1_m["before_phase1"]      # follow-ups: the model before Phase 2 is the same one, Run 1 measured it

    # ---- tables (numbers only, the page formats them)
    decision = [{"key": k, "label": lab, "desc": dsc, **{f: mm["decision"][k].get(f) for f in
                ("auc_vs_label_vis", "ap_vs_label_vis", "auc_no_prealert_vs_nexar_label", "acc_at_thr", "balanced_acc_at_thr", "confusion_at_thr", "threshold")}}
                for k, lab, dsc in SCORES if k in mm["decision"]]
    paired = mm["paired"]["s_after_vs_a1_p"]
    by_group = []
    for g, lab in GROUPS:
        bg = mm["by_group"][g]
        by_group.append({"key": g, "label": lab, "n": bg["n"], "n_pos": bg["n_pos_label_vis"],
                         "crash": {"confusion": bg["a1_p"]["confusion"], "acc": bg["a1_p"]["acc"]},
                         "after": {"confusion": bg["s_after"]["confusion"], "acc": bg["s_after"]["acc"]},
                         "rescue": paired["rescues_by_group"][g]["reasoning_right_crash_model_wrong"],
                         "lost": paired["rescues_by_group"][g]["crash_model_right_reasoning_wrong"]})
    tx = m["text"]
    text = {
        "overlap": [{"key": k, "label": lab, "model": tm[k]["model"], "floor": tm[k]["floor"]} for k, lab in
                    (("bertscore_f1_rescaled", "BERTScore F1 (rescaled)"), ("rouge_l_f1", "ROUGE-L F1"), ("bleu4", "BLEU-4"), ("token_f1", "token F1"))],
        "road_user_match": tx["road_user_match"], "format_ok": tx["format_ok"], "distinct": tx["distinct_explanations"], "n": tx["n"],
        "ppl_after": m["after"]["perplexity_real"], "ppl_before": bp1["perplexity_real"],
        "gap_wrong": m["after"]["gap_wrong"], "gap_wrong_ci": m["after"]["gap_wrong_ci"],
        "gap_blank": m["after"]["gap_blank"], "gap_blank_ci": m["after"]["gap_blank_ci"],
        "gap_wrong_before": bp1["gap_wrong"], "gap_wrong_before_ci": bp1["gap_wrong_ci"],
        "gap_blank_before": bp1["gap_blank"], "gap_blank_before_ci": bp1["gap_blank_ci"],
        "retrieval_top1": m["retrieval_at_10"]["top1"], "retrieval_chance": m["retrieval_at_10"]["chance"],
    }
    curve = [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_nexar_reasoning"]["loss_real"],
              "gap_wrong": e["val_nexar_reasoning"]["gap_wrong"], "gap_wrong_ci": e["val_nexar_reasoning"]["gap_wrong_ci"],
              "gap_blank": e["val_nexar_reasoning"]["gap_blank"],
              "auc_before": e["decision"]["before"]["auc"], "auc_teacher": e["decision"]["teacher"]["auc"]} for e in log]
    val_min = min(curve, key=lambda c: c["val_loss"])["epoch"]
    best = epoch if epoch else val_min       # retrains: best.pt = lowest validation loss (options README)

    # ---- per-window rows (the run-specific part only)
    shared = collections.Counter(p["gen_explanation"] for p in preds)
    rows = []
    for p in preds:
        lbl = p["label"]
        rows.append({
            "p_before": round(sigmoid(p["s_before"]), 2), "p_after": round(sigmoid(p["s_after"]), 2),
            "verdict": "yes" if p["gen_collision"] == 1 else "no",
            "m_expl": p["gen_explanation"], "m_out": p["gen_outcome"],
            "disagree": (p["a1_p"] >= 0.5) != (p["s_after"] >= 0),
            "reason_wrong": (p["gen_collision"] == 1) != (lbl == 1),
            "generic": shared[p["gen_explanation"]] >= GENERIC_MIN,
        })
    dec = {d["key"]: d for d in decision}
    compare = {
        "key": key, "label": label,
        "val_loss": m["after"]["loss_real"], "ppl": m["after"]["perplexity_real"],
        "bert": tm["bertscore_f1_rescaled"]["model"], "bert_floor": tm["bertscore_f1_rescaled"]["floor"],
        "rouge": tm["rouge_l_f1"]["model"], "rouge_floor": tm["rouge_l_f1"]["floor"],
        "road_user_match": tx["road_user_match"], "distinct": tx["distinct_explanations"], "top_repeat": shared.most_common(1)[0][1],
        "gap_wrong": m["after"]["gap_wrong"], "gap_wrong_ci": m["after"]["gap_wrong_ci"], "retrieval_top1": m["retrieval_at_10"]["top1"],
        "auc_after": dec["s_after"]["auc_vs_label_vis"], "auc_before": dec["s_before"]["auc_vs_label_vis"],
        "acc_after": dec["s_after"]["acc_at_thr"], "bacc_after": dec["s_after"]["balanced_acc_at_thr"],
        "acc_before": dec["s_before"]["acc_at_thr"], "bacc_before": dec["s_before"]["balanced_acc_at_thr"],
        "auc_crash": dec["a1_p"]["auc_vs_label_vis"],
        "saturated_after": sum(abs(p["s_after"]) > SATURATED for p in preds) / len(preds),
    }
    run = {"key": key, "label": label, "desc": desc, "decision": decision, "by_group": by_group,
           "paired": {"mcnemar": paired["mcnemar_at_thr"], "auc_diff": paired["auc_diff"]},
           "text": text, "curve": curve, "curve_own": logp is not None or key == "run1", "best_epoch": best, "val_min_epoch": val_min,
           "rows": rows, "compare": compare,
           "counts": {"disagree": sum(r["disagree"] for r in rows), "reason_wrong": sum(r["reason_wrong"] for r in rows),
                      "generic": sum(r["generic"] for r in rows)}}
    return run, preds


def main():
    run1_m = json.load(open(RUN / "results" / "metrics.json", encoding="utf-8"))
    log1 = [json.loads(l) for l in open(RUN / "train_log.jsonl", encoding="utf-8") if l.strip()]
    runs, base = [], None
    for key, label, desc, res, logp, epoch in RUNS:
        run, preds = load_run(key, label, desc, res, logp, epoch, run1_m, log1)
        if base is None:
            base = preds
        # all runs score the same 612 windows in the same order, with the same teacher text and crash-model score
        assert len(preds) == len(base) and all(a["id"] == b["id"] and a["label"] == b["label"] and a["a1_p"] == b["a1_p"]
                                               and a["teacher_explanation"] == b["teacher_explanation"] for a, b in zip(preds, base)), key
        runs.append(run)

    # ---- window part shared by all runs
    rows, n_img = [], 0
    for p in base:
        fd = p["frames_dir"]
        g = f"{REL}/review/{fd}.jpg" if (GRIDS / "review" / f"{fd}.jpg").exists() else None
        gl = f"{REL}/grid_large/{fd}.jpg" if (GRIDS / "grid_large" / f"{fd}.jpg").exists() else None
        n_img += bool(g)
        rows.append({
            "key": p["id"], "video": p["video_key"], "group": p["group"], "tte": p["tte"], "label": p["label"], "label_nexar": p["label_nexar"],
            "pre_alert": bool(p["pre_alert"]), "grid": g, "grid_large": gl, "crash_p": round(p["a1_p"], 4),
            "t_expl": p["teacher_explanation"], "t_out": p["teacher_outcome"],
            "crash_wrong": (p["a1_p"] >= 0.5) != (p["label"] == 1),
        })
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys))

    data = {"n_windows": len(rows), "n_videos": len({p["video_key"] for p in base}), "n_train": N_TRAIN_WINDOWS, "generic_min": GENERIC_MIN,
            "groups": [{"key": g, "label": lab} for g, lab in GROUPS], "rows": rows, "runs": runs,
            "counts": {"crash_wrong": sum(r["crash_wrong"] for r in rows)}}
    out = HERE / "r2_data.js"
    out.write_text("window.R2_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"r2_data.js: {len(rows)} rows x {len(runs)} runs, {out.stat().st_size / 1e6:.2f} MB, grids {n_img}")
    for r in runs:
        c = r["compare"]
        print(f"  {r['key']}: val loss {c['val_loss']:.4f}, top repeat {c['top_repeat']}, distinct {c['distinct']}, AUC S_after {c['auc_after']:.3f}, "
              f"S_before {c['auc_before']:.3f}, saturated {c['saturated_after']:.2f}, counts {r['counts']}, best epoch {r['best_epoch']}, val-min epoch {r['val_min_epoch']}")


if __name__ == "__main__":
    main()
