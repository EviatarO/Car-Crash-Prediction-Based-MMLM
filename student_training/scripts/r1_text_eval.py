"""
r1_text_eval.py - text metrics of a run's generated explanations against the teacher's, WITHOUT needing the loss / gap fields of metrics.json
(plan 2026-10-18, Part A2). For windows that have a teacher reference but no model loss: the private-test "agreed" subset (r1_test_refs_join.py).
For the validation and the train sample, r1_review_reasoning.py gives the same numbers plus the loss / gap section; the metric definitions are shared.

    python r1_text_eval.py --run-dir <dir with predictions.jsonl> [--no-bert]

predictions.jsonl needs: gen_explanation, gen_collision, gen_outcome, teacher_explanation, label, group, and s_first (verdict-first) or s_after.
Writes <run-dir>/text_metrics.json:
  token_f1 / rouge_l_f1 / bleu4 / bertscore_f1_rescaled  {model, floor}   floor = the same model texts against a random OTHER window's teacher text (seed 0)
  by_group: the same four numbers per group (normal / tte0.5 / tte1.0 / tte1.5)
  text: r1_eval_reasoning.text_metrics (road-user match, format ok, written verdict accuracy, verdict-outcome consistency, distinct explanations)
  written_verdict_vs_score_agree: the written "Collision:" line against the sign of the decision score
BERTScore needs `pip install bert-score` and roberta-large (downloaded from HF), so it runs on the pod.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from r1_eval_reasoning import text_metrics
from r1_phase_report import rouge_l, token_f1
from r1_review_reasoning import bleu4


def floor_of(ref, seed=0):
    rng = random.Random(seed)
    return [ref[(i + rng.randrange(1, len(ref))) % len(ref)] for i in range(len(ref))]


def scores(gen, ref, other, scorer=None):
    out = {}
    for name, fn in (("token_f1", token_f1), ("rouge_l_f1", rouge_l)):
        out[name] = {"model": float(np.mean([fn(g, r) for g, r in zip(gen, ref)])), "floor": float(np.mean([fn(g, r) for g, r in zip(gen, other)]))}
    out["bleu4"] = {"model": bleu4(gen, ref), "floor": bleu4(gen, other)}
    if scorer is not None:
        out["bertscore_f1_rescaled"] = {"model": float(scorer.score(gen, ref)[2].mean()), "floor": float(scorer.score(gen, other)[2].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-bert", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    rows = [json.loads(l) for l in open(run / "predictions.jsonl", encoding="utf-8")]
    scorer = None
    if not args.no_bert:
        from bert_score import BERTScorer
        scorer = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
    gen = [r["gen_explanation"] or "" for r in rows]
    ref = [r["teacher_explanation"] for r in rows]
    out = {"n": len(rows), **scores(gen, ref, floor_of(ref), scorer)}
    out["by_group"] = {}
    for g in ("normal", "tte0.5", "tte1.0", "tte1.5"):
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        if len(idx) >= 5:
            gg, rr = [gen[i] for i in idx], [ref[i] for i in idx]
            out["by_group"][g] = {"n": len(idx), **scores(gg, rr, floor_of(rr), scorer)}
    out["text"] = text_metrics(rows)
    key = "s_first" if "s_first" in rows[0] else ("s_after" if "s_after" in rows[0] else None)
    if key:
        k = [(r["gen_collision"], r[key]) for r in rows if r["gen_collision"] is not None]
        out["written_verdict_vs_score_agree"] = float(np.mean([w == int(s >= 0) for w, s in k])) if k else None
    (run / "text_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "by_group"}, indent=1))


if __name__ == "__main__":
    main()
