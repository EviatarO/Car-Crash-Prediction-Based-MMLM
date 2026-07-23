"""
metrics_core.py
===============
Pure-math classification metrics (numpy + scikit-learn only — NO pandas /
matplotlib / seaborn). Shared by the training scripts (which run on RunPod pods
that only have sklearn installed) and by evaluate_metrics.py (the local graph
pipeline). Keeping this dependency-light means a training run can emit
metrics.json without pulling the plotting stack onto the pod.

Covers the full E3 metric table: confusion matrix, accuracy, precision,
recall (sensitivity/TPR), specificity (TNR), F1, F1@optimal-threshold, AP,
AUC-ROC, Brier, ECE, and per-TTE-group AP.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """ECE: bin scores into n_bins equal-width bins, sum over bins of
    |bin_accuracy - bin_confidence| weighted by bin population. Directly
    measures the compressed-score / mis-calibration issue (lower = better).
    Standard n_bins=10."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score > lo) & (y_score <= hi) if i > 0 else (y_score >= lo) & (y_score <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(float(y_true[mask].mean()) - float(y_score[mask].mean()))
    return float(ece)


def metrics_from_arrays(y_true, y_score, groups=None, threshold: float = 0.5,
                        ece_bins: int = 10) -> dict:
    """Self-contained metric dict from raw arrays. Includes everything the E3
    metric table asks for. JSON-safe: NaN metrics (single-class splits) become
    None rather than NaN."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    n = len(y_true)
    n_pos, n_neg = int(y_true.sum()), int(n - y_true.sum())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in (cm.ravel() if cm.size == 4 else (0, 0, 0, 0)))

    def _safe(a, b):
        return float(a) / b if b else 0.0

    precision   = _safe(tp, tp + fp)
    recall      = _safe(tp, tp + fn)   # sensitivity / TPR
    specificity = _safe(tn, tn + fp)   # TNR
    accuracy    = _safe(tp + tn, n)
    f1          = _safe(2 * precision * recall, precision + recall)

    both = n_pos > 0 and n_neg > 0
    ap  = float(average_precision_score(y_true, y_score)) if both else float("nan")
    auc = float(roc_auc_score(y_true, y_score))          if both else float("nan")

    if both:
        p_arr, r_arr, thr = precision_recall_curve(y_true, y_score)
        f1_arr = np.where((p_arr + r_arr) > 0, 2 * p_arr * r_arr / (p_arr + r_arr), 0.0)
        bi = int(np.argmax(f1_arr[:-1])) if len(f1_arr) > 1 else 0
        opt_thr = float(thr[bi]) if len(thr) else threshold
        opt_f1  = float(f1_arr[bi])
    else:
        opt_thr, opt_f1 = threshold, f1

    brier = float(np.mean((y_score - y_true) ** 2))
    ece   = expected_calibration_error(y_true, y_score, ece_bins) if both else float("nan")

    def _r(x):  # JSON-safe: None for NaN, else rounded float
        return None if x != x else round(float(x), 4)

    out = {
        "n_total": n, "n_positive": n_pos, "n_negative": n_neg, "threshold": threshold,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": _r(accuracy),
        "precision": _r(precision),
        "recall_sensitivity_tpr": _r(recall),
        "specificity_tnr": _r(specificity),
        "f1": _r(f1),
        "f1_optimal": _r(opt_f1),
        "optimal_threshold": _r(opt_thr),
        "ap": _r(ap),
        "auc_roc": _r(auc),
        "brier": _r(brier),
        "ece": _r(ece),
    }
    if groups is not None:
        groups = np.asarray(groups)
        gmap = {0: "tte_0.5s", 1: "tte_1.0s", 2: "tte_1.5s"}
        per = {}
        for g, lab in gmap.items():
            m = groups == g
            cnt = int(m.sum())
            if cnt == 0:
                continue
            yt, ys = y_true[m], y_score[m]
            has_both = yt.sum() > 0 and (cnt - yt.sum()) > 0
            per[lab] = {"ap": _r(float(average_precision_score(yt, ys))) if has_both else None,
                        "n": cnt}
        out["per_tte_ap"] = per
    return out
