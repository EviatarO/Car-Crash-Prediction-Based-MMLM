"""
old_vs_new_tte_curve.py
=======================
Side-by-side artifact for the thesis: the OLD (flat, artifact) vs NEW (declining, correct)
anticipation curve, so a reviewer sees exactly why the flat result was misleading.

OLD method (artifact): Nexar group-0/1/2 buckets = DISJOINT videos, each scored vs its
  OWN balanced negatives (chance=0.5 per bucket). Flat by construction; only 0.5/1.0/1.5s.
NEW method (correct):  same 142 group-0 scenes RE-CUT at every offset, vs one FIXED
  142-negative pool (balanced, chance=0.5). 0.5/1.0/.../3.0s. Shows the real decline.

Both use balanced negatives (chance=0.5) so the comparison is apples-to-apples on prevalence;
the ONLY difference is per-bucket-rebalanced+cross-scene (old) vs fixed-pool+same-scene (new).

Output: outputs/e3b_student_267clips_tte/tte_curve/old_vs_new_tte_curve.png (+ .json)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/e3b_student_267clips_tte/tte_curve"
NEW_OFF = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
OLD_OFF = [0.5, 1.0, 1.5]
G2S = {0: 0.5, 1: 1.0, 2: 1.5}

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
def ap(pos, neg):
    return average_precision_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def old_flat(model, half):
    """group-k pos vs group-k OWN neg (balanced per bucket) — the artifact."""
    full = load(FULL[model][half])
    out = {}
    for g, t in G2S.items():
        pos = [r["score"] for r in full if r["ground_truth"] == 1 and r["group"] == g]
        neg = [r["score"] for r in full if r["ground_truth"] == 0 and r["group"] == g]
        out[t] = ap(pos, neg)
    return out


def new_decline(model, half):
    """same 142 group-0 scenes re-cut, vs fixed 142 group-0 neg (balanced) — correct."""
    full = load(FULL[model][half]); curve = load(CURVE[model][half])
    neg = [r["score"] for r in full if r["ground_truth"] == 0 and r["group"] == 0]
    out = {0.5: ap([r["score"] for r in full if r["ground_truth"] == 1 and r["group"] == 0], neg)}
    for t in NEW_OFF[1:]:
        pos = [r["score"] for r in curve if abs(float(r["time_before_s"]) - t) < 1e-6]
        out[t] = ap(pos, neg)
    return out


def main():
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, half in zip(axes, ["private", "public"]):
        for model in ["e3a", "e3b"]:
            old = old_flat(model, half); new = new_decline(model, half)
            results[f"{model}_{half}_old_flat"] = {str(k): v for k, v in old.items()}
            results[f"{model}_{half}_new_decline"] = {str(k): v for k, v in new.items()}
            ax.plot(OLD_OFF, [old[t] for t in OLD_OFF], "s--", color=C[model], alpha=0.55,
                    lw=2, mfc="white", label=f"{model} OLD (per-bucket balanced, cross-scene)")
            ax.plot(NEW_OFF, [new[t] for t in NEW_OFF], "o-", color=C[model], lw=2.2,
                    label=f"{model} NEW (re-cut same scene, fixed neg)")
        ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6, label="chance")
        ax.invert_xaxis(); ax.set_ylim(0.3, 1.0); ax.set_xlim(3.1, 0.4)
        ax.set_xlabel("seconds before collision event"); ax.set_ylabel("Average Precision (balanced)")
        ax.set_title(f"{half.capitalize()} — OLD flat (artifact) vs NEW declining (correct)", fontsize=10)
        ax.legend(fontsize=7.5); ax.grid(alpha=0.3)
    fig.suptitle("Why the flat 0.76 curve was an artifact: re-balancing per bucket hides the decline\n"
                 "Same balanced chance (0.5) in both — only difference is per-bucket+cross-scene (OLD) vs fixed-pool+same-scene (NEW)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    png = OUT / "old_vs_new_tte_curve.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    (OUT / "old_vs_new_tte_curve_numbers.json").write_text(json.dumps(results, indent=2))
    print(f"Saved: {png}")
    print("\n=== OLD flat (per-bucket balanced) vs NEW declining (re-cut, balanced) ===")
    for model in ["e3a", "e3b"]:
        for half in ["private", "public"]:
            o = results[f"{model}_{half}_old_flat"]; n = results[f"{model}_{half}_new_decline"]
            print(f"{model} {half}:")
            print(f"   OLD  0.5={o['0.5']:.3f} 1.0={o['1.0']:.3f} 1.5={o['1.5']:.3f}  (flat)")
            print(f"   NEW  " + " ".join(f"{t}={n[str(t)]:.3f}" for t in NEW_OFF))


if __name__ == "__main__":
    main()
