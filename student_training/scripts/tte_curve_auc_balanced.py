"""
tte_curve_auc_balanced.py
=========================
Two prevalence-controlled views of the anticipation curve:
  (top)    AUC-ROC vs TTE   — prevalence-INVARIANT (pool size irrelevant), uses all negatives
  (bottom) Balanced-AP vs TTE — 142 pos vs 142 group-0 negatives (chance=0.5); 0.5s anchor reproduces ~0.76

Same scenes re-cut at every offset (constant 142 positives). Bootstrap 95% CI (clip-level).
Output: outputs/e3b_student_267clips_tte/tte_curve/auc_ap_vs_tte.png  (+ .json numbers)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/e3b_student_267clips_tte/tte_curve"
OFFSETS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
N_BOOT = 5000
RNG = np.random.default_rng(42)

FULL = {
    "e3a": {"private": REPO/"outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_epoch07.jsonl",
            "public":  REPO/"outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_public_epoch07.jsonl"},
    "e3b": {"private": REPO/"outputs/e3b_student_267clips_tte/e3b_test_private/outputs/trained/e3b_test_private_step000099.jsonl",
            "public":  REPO/"outputs/e3b_student_267clips_tte/e3b_test_public/outputs/trained/e3b_test_public_step000099.jsonl"},
}
CURVE = {
    "e3a": {"private": OUT/"e3a_tte_curve_private_epoch07.jsonl", "public": OUT/"e3a_tte_curve_public_epoch07.jsonl"},
    "e3b": {"private": OUT/"e3b_tte_curve_private_step000099.jsonl", "public": OUT/"e3b_tte_curve_public_step000099.jsonl"},
}
C = {"e3a": "#1f77b4", "e3b": "#d62728"}


def load(p): return [json.loads(l) for l in open(p)]


def get_data(model, half):
    full = load(FULL[model][half]); curve = load(CURVE[model][half])
    pos = {0.5: np.array([r["score"] for r in full if r["ground_truth"] == 1 and r["group"] == 0])}
    for t in OFFSETS[1:]:
        pos[t] = np.array([r["score"] for r in curve if abs(float(r["time_before_s"]) - t) < 1e-6])
    neg_all = np.array([r["score"] for r in full if r["ground_truth"] == 0])           # 339/333
    neg_bal = np.array([r["score"] for r in full if r["ground_truth"] == 0 and r["group"] == 0])  # 142
    return pos, neg_all, neg_bal


def metric_with_ci(pos, neg, fn):
    """Point estimate + 95% CI via clip-level bootstrap."""
    yt = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    ys = np.r_[pos, neg]
    point = fn(yt, ys)
    boots = np.empty(N_BOOT)
    npos, nneg = len(pos), len(neg)
    for b in range(N_BOOT):
        pi = RNG.integers(0, npos, npos); ni = RNG.integers(0, nneg, nneg)
        yb = np.r_[np.ones(npos), np.zeros(nneg)]
        sb = np.r_[pos[pi], neg[ni]]
        boots[b] = fn(yb, sb)
    return point, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def compute(model, half, kind):
    pos, neg_all, neg_bal = get_data(model, half)
    neg = neg_all if kind == "auc" else neg_bal
    fn = roc_auc_score if kind == "auc" else average_precision_score
    out = {}
    for t in OFFSETS:
        out[t] = metric_with_ci(pos[t], neg, fn)
    return out


def compute_pooled(model, kind):
    pos_pr, na_pr, nb_pr = get_data(model, "private")
    pos_pu, na_pu, nb_pu = get_data(model, "public")
    neg = np.r_[na_pr, na_pu] if kind == "auc" else np.r_[nb_pr, nb_pu]
    fn = roc_auc_score if kind == "auc" else average_precision_score
    out = {}
    for t in OFFSETS:
        out[t] = metric_with_ci(np.r_[pos_pr[t], pos_pu[t]], neg, fn)
    return out


def main():
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [("private", "Private"), ("public", "Public"), ("pooled", "Pooled")]
    for row, kind in enumerate(["auc", "ap"]):
        for col, (half, title) in enumerate(panels):
            ax = axes[row][col]
            for model in ["e3a", "e3b"]:
                r = compute_pooled(model, kind) if half == "pooled" else compute(model, half, kind)
                results[f"{kind}_{model}_{half}"] = {str(k): v for k, v in r.items()}
                xs = OFFSETS
                ys = [r[t][0] for t in xs]
                lo = [r[t][1] for t in xs]; hi = [r[t][2] for t in xs]
                lbl = f"{model} ({'headline' if model=='e3a' else 'ablation'})"
                ax.plot(xs, ys, "o-", color=C[model], label=lbl, lw=2)
                ax.fill_between(xs, lo, hi, color=C[model], alpha=0.15)
            ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.6, label="chance")
            ax.set_xlim(0.4, 3.1); ax.set_ylim(0.3, 1.0)
            ax.invert_xaxis()
            ax.set_xlabel("seconds before collision event")
            ax.set_ylabel("AUC-ROC" if kind == "auc" else "Average Precision (balanced)")
            metric_name = "AUC-ROC (prevalence-invariant)" if kind == "auc" else "Balanced-AP (chance=0.5, anchor≈0.76)"
            ax.set_title(f"{title} — {metric_name}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Anticipation curve — same scenes re-cut, constant 142 positives per TTE\n"
                 "Top: AUC-ROC (uses all negatives) | Bottom: Balanced-AP (142 vs 142)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = OUT / "auc_ap_vs_tte.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    (OUT / "auc_ap_vs_tte_numbers.json").write_text(json.dumps(results, indent=2))
    print(f"Saved: {png}")

    # print compact tables
    for kind in ["auc", "ap"]:
        name = "AUC-ROC" if kind == "auc" else "Balanced-AP"
        print(f"\n=== {name} vs TTE (point estimate) ===")
        print(f'{"model/half":14s} ' + " ".join(f"{t:>5}s" for t in OFFSETS))
        for model in ["e3a", "e3b"]:
            for half in ["private", "public", "pooled"]:
                r = results[f"{kind}_{model}_{half}"]
                print(f'{model+" "+half:14s} ' + " ".join(f"{r[str(t)][0]:.3f}" for t in OFFSETS))


if __name__ == "__main__":
    main()
