"""caption_leakage_gate.py — does caption TEXT ALONE predict the crash label?

Standing QA gate for any semantic-supervision caption corpus (see
docs_agents/ARCHITECTURE.md's "Any new caption corpus must pass the label-leakage gate").
A corpus that leaks turns the semantic loss into a redundant, noisier copy of the crash
label rather than a source of scene semantics - see the V10 corpus finding
(docs_agents/EXPERIMENTS.md: text->label AUC 0.9643, driven by the GT-informed/blind
prompt branch producing two different vocabularies by class).

Method (fixed, do not change without re-validating against the V10/V12 numbers below):
TfidfVectorizer(ngram_range=(1,2), min_df=3) -> LogisticRegression, 5-fold GroupKFold
BY video_id (not row) so TTE-sibling rows of the same clip never split across train/val -
a plain K-fold would leak the group's near-duplicate captions across the split and
understate the classifier's real ability to recover the label.

This script did not exist when the V10 (0.9643) and V12 (0.7640) numbers were first
produced ad-hoc in-session; those numbers were reproduced against this implementation
before being written into the docs (see the `--verify` sample commands below the CLI).

Usage:
    python caption_leakage_gate.py \
        --captions ../../outputs/semantic_captions/Caption_V10_Mixed_1761.jsonl \
        --caption-field caption --label-field gt_verdict --positive-value YES \
        --out ../../outputs/semantic_captions/leakage_gate_v10.json

    python caption_leakage_gate.py \
        --captions ../../outputs/semantic_captions/Caption_V12_Neutral_1761.jsonl \
        --caption-field caption_neutral --label-field event_occurs --positive-value 1 \
        --out ../../outputs/semantic_captions/leakage_gate_v12.json
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score


def load_rows(path, caption_field, label_field, positive_value):
    texts, labels, groups = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cap = r.get(caption_field)
            lab = r.get(label_field)
            vid = r.get("video_id")
            if cap is None or lab is None or vid is None:
                continue
            texts.append(cap)
            labels.append(int(str(lab) == str(positive_value)))
            groups.append(vid)
    return texts, np.array(labels), groups


def run_gate(texts, labels, groups, n_splits=5, seed=0):
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros(len(texts))
    fold_aucs = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(texts, labels, groups)):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
        Xtr = vec.fit_transform([texts[i] for i in tr_idx])
        Xva = vec.transform([texts[i] for i in va_idx])
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(Xtr, labels[tr_idx])
        proba = clf.predict_proba(Xva)[:, 1]
        oof_proba[va_idx] = proba
        if len(set(labels[va_idx])) > 1:
            fold_aucs.append(roc_auc_score(labels[va_idx], proba))
    oof_auc = roc_auc_score(labels, oof_proba)
    return oof_auc, fold_aucs, oof_proba


def top_coefficients(texts, labels, k=15):
    """Full-corpus fit (not CV) for coefficient inspection only - diagnostic, not the
    reported metric. Matches the n=100 TF-IDF coefficient inspection already cited in
    docs_agents/EXPERIMENTS.md (braking +0.957, decreasing gap, path closing)."""
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3)
    X = vec.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, labels)
    names = vec.get_feature_names_out()
    order = np.argsort(clf.coef_[0])
    top_pos = [(names[i], round(float(clf.coef_[0][i]), 4)) for i in order[-k:][::-1]]
    top_neg = [(names[i], round(float(clf.coef_[0][i]), 4)) for i in order[:k]]
    return top_pos, top_neg


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captions", required=True)
    ap.add_argument("--caption-field", required=True,
                     help="e.g. 'caption' (V10) or 'caption_neutral' (V12)")
    ap.add_argument("--label-field", required=True,
                     help="e.g. 'gt_verdict' (V10, string) or 'event_occurs' (V12, 0/1)")
    ap.add_argument("--positive-value", required=True,
                     help="the label-field value meaning 'crash occurs', e.g. YES or 1")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-auc", type=float, default=0.75,
                     help="gate threshold - corpus PASSES if oof_auc < this")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = Path(args.captions)
    texts, labels, groups = load_rows(path, args.caption_field, args.label_field,
                                       args.positive_value)
    n_pos, n_neg = int(labels.sum()), int(len(labels) - labels.sum())
    n_groups = len(set(groups))
    print(f"[data] {len(texts)} rows, {n_groups} distinct video_ids, "
          f"{n_pos} positive / {n_neg} negative")
    if len(texts) == 0:
        print("[error] no rows loaded - check --caption-field/--label-field", file=sys.stderr)
        sys.exit(1)

    oof_auc, fold_aucs, _ = run_gate(texts, labels, groups, args.n_splits, args.seed)
    passed = oof_auc < args.target_auc
    print(f"[result] out-of-fold AUC = {oof_auc:.4f}  "
          f"(per-fold: {[round(a, 4) for a in fold_aucs]})")
    print(f"[gate] target < {args.target_auc}  ->  {'PASS' if passed else 'FAIL'}")

    top_pos, top_neg = top_coefficients(texts, labels)
    print("[diagnostic] top positive-class (crash-predictive) n-grams:")
    for name, coef in top_pos[:8]:
        print(f"    {name:30s} {coef:+.4f}")

    out = {
        "source_file": str(path),
        "caption_field": args.caption_field,
        "label_field": args.label_field,
        "positive_value": args.positive_value,
        "n_rows": len(texts),
        "n_groups": n_groups,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "method": "TfidfVectorizer(ngram_range=(1,2), min_df=3) + LogisticRegression, "
                  f"GroupKFold({args.n_splits}) by video_id",
        "oof_auc": round(float(oof_auc), 4),
        "fold_aucs": [round(float(a), 4) for a in fold_aucs],
        "target_auc": args.target_auc,
        "gate_passed": bool(passed),
        "top_positive_ngrams": top_pos,
        "top_negative_ngrams": top_neg,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[wrote] {out_path}")


if __name__ == "__main__":
    main()
