"""
build_r2_data.py -> r2_data.js   (Experiments page, tab "Reasoning model - runs": Run 1, the three follow-ups and the four Session-1 runs, with a run selector)

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
Session-1 runs (outputs/r1_levers_2026-10-13/levers_2026-10-13/{B3,B4,B5,B45}): results/ in the same formats, plus p1/train_log.jsonl (Phase 1, projector only)
and p2/train_log.jsonl (Phase 2); metrics.json carries a "train_threshold" block (threshold for S_before chosen on 600 training windows). They have no
"before_phase1" block: the "before Phase 2" numbers are taken from the last epoch of their own Phase-1 log (best = lowest validation loss = last epoch).
Fact agreement (type / side / motion / gap vs the teacher text): outputs/r1_levers_2026-10-13/fact_agreement.json (run1, B3..B45) and
outputs/r1_fact_eval_2026-10-13/fact_agreement.json (run1, opt1..opt3); fact probe: levers_2026-10-13/probe/fact_probe.json.
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
LEV = ROOT / "outputs" / "r1_levers_2026-10-13"
LEVRUN = LEV / "levers_2026-10-13"
P1A = ROOT / "outputs" / "r1_phase1aug_2026-10-14" / "phase1aug_2026-10-14"
GRIDS = ROOT / "outputs" / "teacher_dataset_2026-10"
REL = "../outputs/teacher_dataset_2026-10"
GENERIC_MIN = 10                      # an explanation shared by >= this many windows counts as "generic"
# key, selector label, description, results dir, train log (None = Run 1's; a (phase 1, phase 2) pair for the Session-1 runs), epoch of the evaluated checkpoint (None = the epoch of lowest validation loss)
RUNS = [
    ("run1", "Run 1 (epoch 5)", "Run 1: best.pt = epoch 5, chosen by the wrong-video gap", RUN / "results", RUN / "train_log.jsonl", 5),
    ("opt1", "Option 1 (epoch 3)", "Option 1: Run 1's epoch 3, the epoch with the lowest validation loss (same training run, earlier checkpoint)",
     OPT / "opt1_epoch3" / "results", None, 3),
    ("opt2", "Option 2 (template cap)", "Option 2: retrain with a template cap (frequent explanation openings down-weighted)",
     OPT / "opt2_cap" / "results", OPT / "opt2_cap" / "train_log.jsonl", None),
    ("opt3", "Option 3 (box position)", "Option 3: retrain with the position of the attended road user given as words in the prompt",
     OPT / "opt3_box" / "results", OPT / "opt3_box" / "train_log.jsonl", None),
    ("b3", "B3 fresh Phase 1 (last layer)", "B3: a fresh Phase 1 (projector only, from random) on the Nexar reasoning texts instead of DADA, then Phase 2; last-layer tokens only (the control of Session 1)",
     LEVRUN / "B3" / "results", (LEVRUN / "B3" / "p1" / "train_log.jsonl", LEVRUN / "B3" / "p2" / "train_log.jsonl"), None),
    ("b4", "B4 + attention", "B4: B3 plus the crash head's attention per token as an extra input",
     LEVRUN / "B4" / "results", (LEVRUN / "B4" / "p1" / "train_log.jsonl", LEVRUN / "B4" / "p2" / "train_log.jsonl"), None),
    ("b5", "B5 + layer 11", "B5: B3 plus the layer-11 tokens as an extra input",
     LEVRUN / "B5" / "results", (LEVRUN / "B5" / "p1" / "train_log.jsonl", LEVRUN / "B5" / "p2" / "train_log.jsonl"), None),
    ("b45", "B45 + both", "B45: B3 plus both extra inputs (attention and layer-11 tokens)",
     LEVRUN / "B45" / "results", (LEVRUN / "B45" / "p1" / "train_log.jsonl", LEVRUN / "B45" / "p2" / "train_log.jsonl"), None),
    ("p1long", "P1long (Phase 1 only)", "P1long: Phase 1 only (projector, frozen language model, no LoRA), up to 15 epochs, original windows + teacher texts: the control of the enlargement. Its training curve was validated on 2,413 pairs (the 612 windows plus shifted copies), the results table on the 612 windows",
     P1A / "P1long" / "results", P1A / "P1long" / "train_log.jsonl", "p1only"),
    ("p1shift", "P1shift (Phase 1 only)", "P1shift: Phase 1 only, as P1long plus frame-shifted copies of the windows (no rephrased texts)",
     P1A / "P1shift" / "results", P1A / "P1shift" / "train_log.jsonl", "p1only"),
    ("p1aug", "P1aug (Phase 1 only)", "P1aug: Phase 1 only, as P1long plus frame-shifted copies and the judged rephrased texts (the teacher's text drawn 40 % of the time): the best Phase-1 model so far",
     P1A / "P1aug" / "results", P1A / "P1aug" / "train_log.jsonl", "p1only"),
]
FACT_KEYS = ["type", "side", "motion", "gap"]
FACT_RUN = {"run1": "run1", "opt1": "opt1", "opt2": "opt2", "opt3": "opt3", "b3": "B3", "b4": "B4", "b5": "B5", "b45": "B45", "p1long": "P1long", "p1shift": "P1shift", "p1aug": "P1aug"}   # run key -> key in fact_agreement.json
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


def read_log(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def curve_of(log):
    return [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val_nexar_reasoning"]["loss_real"],
             "gap_wrong": e["val_nexar_reasoning"]["gap_wrong"], "gap_wrong_ci": e["val_nexar_reasoning"]["gap_wrong_ci"],
             "gap_blank": e["val_nexar_reasoning"]["gap_blank"],
             "auc_before": e["decision"]["before"]["auc"], "auc_teacher": e["decision"]["teacher"]["auc"]} for e in log]


def load_run(key, label, desc, res, logp, epoch, run1_m, log1, fact):
    m = json.load(open(res / "metrics.json", encoding="utf-8"))
    tm = json.load(open(res / "text_metrics.json", encoding="utf-8"))
    preds = [json.loads(l) for l in open(res / "predictions.jsonl", encoding="utf-8") if l.strip()]
    two = isinstance(logp, tuple)          # Session-1 runs: (Phase 1 log, Phase 2 log)
    log = read_log(logp[1]) if two else (read_log(logp) if logp else log1)
    mm = m["metrics"]
    assert len(preds) == m["n_windows"] == mm["n"], (key, len(preds), m["n_windows"])
    log_p1 = read_log(logp[0]) if two else None
    if epoch == "p1only":          # Phase-1-only runs (no Phase 2): the evaluated model is the Phase-1 best, so "before" = "after"
        bp1 = dict(m["after"], perplexity_real=m["after"]["perplexity_real"])
        epoch = None
    elif m.get("before_phase1"):
        bp1 = m["before_phase1"]
    elif two:      # Session-1 runs: the model before Phase 2 = their own Phase-1 best (lowest validation loss; the last epoch in all four)
        e = min(log_p1, key=lambda x: x["val_nexar_reasoning"]["loss_real"])["val_nexar_reasoning"]
        bp1 = dict(e, perplexity_real=math.exp(e["loss_real"]))
    else:          # Run 1's follow-ups: the model before Phase 2 is the same one, Run 1 measured it
        bp1 = run1_m["before_phase1"]

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
    curve = curve_of(log)
    val_min = min(curve, key=lambda c: c["val_loss"])["epoch"]
    best = epoch if epoch else val_min       # retrains: best.pt = lowest validation loss (options README)
    curve_p1 = curve_of(log_p1) if two else None
    best_p1 = min(curve_p1, key=lambda c: c["val_loss"])["epoch"] if two else None
    tt = m.get("train_threshold")
    train_thr = None if not tt else {
        "n_train": tt["n_train_windows"], "p": tt["p_threshold"], "log_odds": tt["log_odds_threshold"], "bacc_train": tt["balanced_acc_on_train_windows"],
        "acc": tt["val_acc"], "bacc": tt["val_balanced_acc"], "confusion": tt["val_confusion"], "score": tt["score"]}

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
        "bacc_before_train_thr": train_thr["bacc"] if train_thr else None,
        "fact": {f: fact[FACT_RUN[key]]["overall"][f]["agree"] for f in FACT_KEYS},
    }
    run = {"key": key, "label": label, "desc": desc, "decision": decision, "by_group": by_group,
           "paired": {"mcnemar": paired["mcnemar_at_thr"], "auc_diff": paired["auc_diff"]},
           "text": text, "curve": curve, "curve_own": logp is not None or key == "run1", "best_epoch": best, "val_min_epoch": val_min,
           "curve_p1": curve_p1, "best_epoch_p1": best_p1, "train_thr": train_thr,
           "rows": rows, "compare": compare,
           "counts": {"disagree": sum(r["disagree"] for r in rows), "reason_wrong": sum(r["reason_wrong"] for r in rows),
                      "generic": sum(r["generic"] for r in rows)}}
    return run, preds


def main():
    run1_m = json.load(open(RUN / "results" / "metrics.json", encoding="utf-8"))
    log1 = read_log(RUN / "train_log.jsonl")
    fact_a = json.load(open(LEV / "fact_agreement.json", encoding="utf-8"))                                   # run1, B3, B4, B5, B45
    fact_b = json.load(open(ROOT / "outputs" / "r1_fact_eval_2026-10-13" / "fact_agreement.json", encoding="utf-8"))   # run1, opt1, opt2, opt3
    assert fact_a["run1"]["overall"] == fact_b["run1"]["overall"]
    fact_c = json.load(open(ROOT / "outputs" / "r1_phase1aug_2026-10-14" / "fact_agreement.json", encoding="utf-8"))               # P1long, P1shift, P1aug, B3_p1
    fact = {**fact_b, **fact_a, **fact_c}
    baseline = {f: fact["run1"]["overall"][f]["majority_baseline"] for f in FACT_KEYS}
    for r in fact.values():
        assert all(abs(r["overall"][f]["majority_baseline"] - baseline[f]) < 1e-12 for f in FACT_KEYS)    # same windows, same baseline
    probe_raw = json.load(open(LEVRUN / "probe" / "fact_probe.json", encoding="utf-8"))["probe"]
    probe = {k: {f: {"bacc": probe_raw[k][f]["balanced_acc"], "chance": probe_raw[k][f]["chance_balanced"]} for f in FACT_KEYS} for k in ("last", "l11", "attn")}
    runs, base = [], None
    for key, label, desc, res, logp, epoch in RUNS:
        run, preds = load_run(key, label, desc, res, logp, epoch, run1_m, log1, fact)
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
            "fact_baseline": baseline, "probe": probe,
            "counts": {"crash_wrong": sum(r["crash_wrong"] for r in rows)}}
    out = HERE / "r2_data.js"
    out.write_text("window.R2_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"r2_data.js: {len(rows)} rows x {len(runs)} runs, {out.stat().st_size / 1e6:.2f} MB, grids {n_img}")
    for r in runs:
        c = r["compare"]
        print(f"  {r['key']}: val loss {c['val_loss']:.4f}, top repeat {c['top_repeat']}, distinct {c['distinct']}, AUC S_after {c['auc_after']:.3f}, "
              f"S_before {c['auc_before']:.3f}, saturated {c['saturated_after']:.2f}, counts {r['counts']}, best epoch {r['best_epoch']}, val-min epoch {r['val_min_epoch']}"
              f", fact {({k: round(v, 3) for k, v in c['fact'].items()})}, train-thr bacc {c['bacc_before_train_thr']}")


if __name__ == "__main__":
    main()
