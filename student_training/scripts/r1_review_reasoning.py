"""
r1_review_reasoning.py - PC-side (no GPU needed for the small models) text metrics + report for the reasoning model.

    python r1_review_reasoning.py --run-dir <dir with predictions.jsonl + metrics.json> [--no-bert]

Reads what r1_eval_reasoning.py wrote on the pod. Adds, for the model's explanation vs the teacher's explanation (the reference):
  BERTScore F1 (roberta-large, rescaled with the baseline, as in the Phase-2 review)  - meaning similarity, tolerant to paraphrase  [headline]
  ROUGE-L F1                                                                          - longest common word sequence (wording and order)
  BLEU-4 (corpus, smoothed)                                                           - exact 1-4-word phrases; punishes paraphrase  [appendix]
  token-F1                                                                            - bag-of-words overlap                         [internal]
each with a FLOOR: the same model answers paired with a random OTHER window's teacher text (seed 0). METEOR / CIDEr are not computed
(extra packages; appendix-only in the plan). Perplexity, road-user match, written verdict / outcome match come from metrics.json.
Writes <run-dir>/text_metrics.json and <run-dir>/report.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np

from r1_phase_report import norm, rouge_l, token_f1


def fmt(x, k=3):
    return 'n/a' if x is None else f'{x:.{k}f}'


def bleu4(hyps, refs):
    """Corpus BLEU-4 with add-one smoothing on the higher orders (single reference)."""
    clip, tot, hl, rl = [0] * 4, [0] * 4, 0, 0
    for h, r in zip(hyps, refs):
        h, r = norm(h).split(), norm(r).split()
        hl, rl = hl + len(h), rl + len(r)
        for n in range(1, 5):
            hg = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
            rg = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
            clip[n - 1] += sum(min(c, rg[g]) for g, c in hg.items())
            tot[n - 1] += max(0, len(h) - n + 1)
    p = [(clip[i] + (i > 0)) / (tot[i] + (i > 0)) if tot[i] + (i > 0) else 0 for i in range(4)]
    if min(p) <= 0:
        return 0.0
    bp = 1.0 if hl > rl else math.exp(1 - rl / max(1, hl))
    return float(bp * math.exp(sum(math.log(x) for x in p) / 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--no-bert", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    rows = [json.loads(l) for l in open(run / "predictions.jsonl", encoding="utf-8")]
    rep = json.load(open(run / "metrics.json", encoding="utf-8"))
    gen = [r["gen_explanation"] or "" for r in rows]
    ref = [r["teacher_explanation"] for r in rows]
    rng = random.Random(0)
    other = [ref[(i + rng.randrange(1, len(ref))) % len(ref)] for i in range(len(ref))]       # floor: some other window's teacher text
    out = {"n": len(rows)}
    for name, fn in (("token_f1", token_f1), ("rouge_l_f1", rouge_l)):
        out[name] = {"model": float(np.mean([fn(g, r) for g, r in zip(gen, ref)])), "floor": float(np.mean([fn(g, r) for g, r in zip(gen, other)]))}
    out["bleu4"] = {"model": bleu4(gen, ref), "floor": bleu4(gen, other)}
    if not args.no_bert:
        from bert_score import BERTScorer
        bs = BERTScorer(model_type="roberta-large", lang="en", rescale_with_baseline=True)
        out["bertscore_f1_rescaled"] = {"model": float(bs.score(gen, ref)[2].mean()), "floor": float(bs.score(gen, other)[2].mean())}
    (run / "text_metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    d, t = rep["metrics"]["decision"], rep["text"]
    L = [f"# Reasoning model - validation report ({len(rows)} windows, order {rep['order']})", "",
         "## Decision (score = log-odds read at the label 'Collision:'; thresholded numbers at P = 0.5)", "",
         "| score | AUC vs label_vis | AUC no-pre-alert vs Nexar label | acc @ 0.5 | balanced acc @ 0.5 |", "|---|---|---|---|---|"]
    for k, v in d.items():
        L.append(f"| {k} | {fmt(v['auc_vs_label_vis'])} | {fmt(v['auc_no_prealert_vs_nexar_label'])} | {v['acc_at_thr']:.3f} | {v['balanced_acc_at_thr']:.3f} |")
    L += ["", "## Grounding", "",
          f"- loss {rep['after']['loss_real']:.3f}, perplexity {rep['after']['perplexity_real']:.2f}"
          + (f" (Phase 1 before: {rep['before_phase1']['loss_real']:.3f} / {rep['before_phase1']['perplexity_real']:.2f})" if "before_phase1" in rep else ""),
          f"- wrong-video gap {rep['after']['gap_wrong']:+.3f} {rep['after']['gap_wrong_ci']}; blank-video gap {rep['after']['gap_blank']:+.3f} {rep['after']['gap_blank_ci']}",
          f"- retrieval@1 of 10: {rep['retrieval_at_10']['top1']:.3f} (chance 0.10)", "", "## Text vs the teacher's explanation (model / floor)", ""]
    for k, v in out.items():
        if isinstance(v, dict):
            L.append(f"- {k}: {v['model']:.3f} / {v['floor']:.3f}")
    L += [f"- road-user match {fmt(t['road_user_match'])} (n={t['road_user_n']}); format ok {t['format_ok']:.3f}; written verdict vs label_vis {t['written_verdict_acc_vs_label_vis']}; "
          f"written verdict vs score agree {t['written_verdict_vs_score_agree']}; verdict/outcome consistent (heuristic) {t['verdict_vs_outcome_consistent(heuristic)']}"]
    (run / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
