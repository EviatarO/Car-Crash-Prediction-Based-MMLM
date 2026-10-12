"""
build_r2_data.py -> r2_data.js   (Experiments page, tab "Reasoning model - Run 1")

Results of the reasoning model: Phase-2 fine training (projector + LoRA on Qwen3-VL-4B, frozen V-JEPA2 features) on the Nexar reasoning
dataset, scored on the 612 validation windows (204 videos never trained on). Everything is read from
outputs/r1_phase2r_2026-10-12/phase2r:
  results/metrics.json       decision metrics per score, per-group confusion, paired tests (McNemar + AUC difference), text block, perplexity, gaps, retrieval
  results/text_metrics.json  token-F1 / ROUGE-L / BLEU-4 / BERTScore, model vs floor
  train_log.jsonl            per-epoch train loss, validation loss, wrong-video gap, decision AUC
  results/predictions.jsonl  one row per validation window (scores are log-odds; P(yes) = sigmoid)
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
GRIDS = ROOT / "outputs" / "teacher_dataset_2026-10"
REL = "../outputs/teacher_dataset_2026-10"
GENERIC_MIN = 10                      # an explanation shared by >= this many windows counts as "generic"
GROUPS = [["normal", "normal"], ["pre_alert", "pre-alert"], ["tte1.5", "TTE 1.5 s"], ["tte1.0", "TTE 1.0 s"], ["tte0.5", "TTE 0.5 s"]]
SCORES = [["s_after", "S_after", "after the model's own reasoning"],
          ["s_before", "S_before", "reasoning cut out (decision straight after the frames)"],
          ["s_teacher", "S_teacher", "after the teacher's reasoning (upper bound)"],
          ["s_old_phase2", "old Phase 2", "the previous Phase-2 model, decision only"],
          ["a1_p", "crash model", "the frozen crash model (A1), probability"]]
N_TRAIN_WINDOWS = 3557                # train windows, from outputs/r1_phase2r_2026-10-12/summary.md
EPOCH_BEST = 5                        # best.pt: chosen by the wrong-video gap


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def main():
    m = json.load(open(RUN / "results" / "metrics.json", encoding="utf-8"))
    tm = json.load(open(RUN / "results" / "text_metrics.json", encoding="utf-8"))
    preds = [json.loads(l) for l in open(RUN / "results" / "predictions.jsonl", encoding="utf-8") if l.strip()]
    log = [json.loads(l) for l in open(RUN / "train_log.jsonl", encoding="utf-8") if l.strip()]
    mm = m["metrics"]
    assert len(preds) == m["n_windows"] == mm["n"], (len(preds), m["n_windows"])

    # ---- tables (numbers only, the page formats them)
    decision = [{"key": k, "label": lab, "desc": desc, **{f: mm["decision"][k].get(f) for f in
                ("auc_vs_label_vis", "ap_vs_label_vis", "auc_no_prealert_vs_nexar_label", "acc_at_thr", "balanced_acc_at_thr", "confusion_at_thr", "threshold")}}
                for k, lab, desc in SCORES]
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
        "ppl_after": m["after"]["perplexity_real"], "ppl_before": m["before_phase1"]["perplexity_real"],
        "gap_wrong": m["after"]["gap_wrong"], "gap_wrong_ci": m["after"]["gap_wrong_ci"],
        "gap_blank": m["after"]["gap_blank"], "gap_blank_ci": m["after"]["gap_blank_ci"],
        "gap_wrong_before": m["before_phase1"]["gap_wrong"], "gap_wrong_before_ci": m["before_phase1"]["gap_wrong_ci"],
        "gap_blank_before": m["before_phase1"]["gap_blank"], "gap_blank_before_ci": m["before_phase1"]["gap_blank_ci"],
        "retrieval_top1": m["retrieval_at_10"]["top1"], "retrieval_chance": m["retrieval_at_10"]["chance"],
    }
    curve = [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_nexar_reasoning"]["loss_real"],
              "gap_wrong": e["val_nexar_reasoning"]["gap_wrong"], "gap_wrong_ci": e["val_nexar_reasoning"]["gap_wrong_ci"],
              "gap_blank": e["val_nexar_reasoning"]["gap_blank"],
              "auc_before": e["decision"]["before"]["auc"], "auc_teacher": e["decision"]["teacher"]["auc"]} for e in log]
    val_min = min(curve, key=lambda c: c["val_loss"])["epoch"]

    # ---- per-window rows
    shared = collections.Counter(p["gen_explanation"] for p in preds)
    rows, n_img = [], 0
    for p in preds:
        fd = p["frames_dir"]
        g = f"{REL}/review/{fd}.jpg" if (GRIDS / "review" / f"{fd}.jpg").exists() else None
        gl = f"{REL}/grid_large/{fd}.jpg" if (GRIDS / "grid_large" / f"{fd}.jpg").exists() else None
        n_img += bool(g)
        label = p["label"]
        crash_yes = p["a1_p"] >= 0.5
        after_yes = p["s_after"] >= 0
        rows.append({
            "key": p["id"], "video": p["video_key"], "group": p["group"], "tte": p["tte"], "label": label, "label_nexar": p["label_nexar"],
            "pre_alert": bool(p["pre_alert"]), "grid": g, "grid_large": gl, "crash_p": round(p["a1_p"], 4),
            "p_before": round(sigmoid(p["s_before"]), 2), "p_after": round(sigmoid(p["s_after"]), 2),
            "verdict": "yes" if p["gen_collision"] == 1 else "no",
            "t_expl": p["teacher_explanation"], "t_out": p["teacher_outcome"], "m_expl": p["gen_explanation"], "m_out": p["gen_outcome"],
            "disagree": crash_yes != after_yes,
            "reason_wrong": (p["gen_collision"] == 1) != (label == 1),
            "crash_wrong": crash_yes != (label == 1),
            "generic": shared[p["gen_explanation"]] >= GENERIC_MIN,
        })
    keys = [r["key"] for r in rows]
    assert len(keys) == len(set(keys))

    data = {"n_windows": len(rows), "n_videos": len({p["video_key"] for p in preds}), "n_train": N_TRAIN_WINDOWS, "generic_min": GENERIC_MIN,
            "decision": decision, "by_group": by_group,
            "paired": {"mcnemar": paired["mcnemar_at_thr"], "auc_diff": paired["auc_diff"]},
            "text": text, "curve": curve, "best_epoch": EPOCH_BEST, "val_min_epoch": val_min, "rows": rows,
            "counts": {"disagree": sum(r["disagree"] for r in rows), "reason_wrong": sum(r["reason_wrong"] for r in rows),
                       "crash_wrong": sum(r["crash_wrong"] for r in rows), "generic": sum(r["generic"] for r in rows)}}
    out = HERE / "r2_data.js"
    out.write_text("window.R2_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"r2_data.js: {len(rows)} rows, {out.stat().st_size / 1e6:.2f} MB, grids {n_img}, counts {data['counts']}, best epoch {EPOCH_BEST}, lowest val loss epoch {val_min}")


if __name__ == "__main__":
    main()
