"""
e4_stageC_figures.py
====================
Stage C figure pack. Standard per-split metrics/curves come from the existing
`evaluate_metrics.py` engine (run that separately on each eval JSONL with the
vision `score`). THIS script adds the Stage-C-specific figures that engine
doesn't cover:

  - loss_vs_epoch.png            train vs val reasoning CE (from epoch_metrics.jsonl)
  - tte_ap_bar_<split>.png       vision AP per TTE bucket (pos@TTE vs MID negs)
  - tte_cm_<split>.png           confusion matrix per TTE bucket (vision @thr)
  - score_dist_by_tte_<split>.png  vision P(collision) per TTE (overlaid)
  - roc_vision_vs_llm_<split>.png  vision ROC overlaid with LLM-verdict (P(YES)) ROC
  - llm_agreement_<split>.png    LLM verdict vs vision decision, and vs GT (2x2 each)
  - dce_shuffle_zero_B_vs_C.png  ΔCE-shuffle / ΔCE-zero, Stage B vs Stage C (headline)
  - templating_B_vs_C.png        mean pairwise gen similarity (high = templated)

Every figure is guarded by input existence and skipped (with a note) if missing,
so partial runs still produce what they can. No GPU needed.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TTE_COLORS = {"TTE_0.5": "#d7191c", "TTE_1.0": "#fdae61", "TTE_1.5": "#2c7bb6"}
TBS = {"TTE_0.5": 0.5, "TTE_1.0": 1.0, "TTE_1.5": 1.5}


def _load_jsonl(p):
    if not p:
        return None
    p = Path(p)
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _ap_auc(scores, targets):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(targets, int); s = np.asarray(scores, float)
    if len(set(y.tolist())) < 2:
        return None, None
    return float(average_precision_score(y, s)), float(roc_auc_score(y, s))


# ── per-TTE (pos@TTE vs all MID negatives) ───────────────────────────────────

def _tte_buckets(rows, neg_bucket="MID"):
    pos = defaultdict(list); negs = []
    for r in rows:
        h = r.get("horizon_label", "")
        if int(r["ground_truth"]) == 1 and h in TTE_COLORS:
            pos[h].append(r)
        elif int(r["ground_truth"]) == 0 and str(h).startswith(neg_bucket):
            negs.append(r)
    return pos, negs


def fig_tte_ap(rows, out, split, score_key="score"):
    pos, negs = _tte_buckets(rows)
    if not pos or not negs:
        print(f"  [skip] tte_ap_{split}: need pos buckets + MID negs"); return
    ttes = sorted(pos); aps = []; ns = []
    for t in ttes:
        sc = [p[score_key] for p in pos[t]] + [n[score_key] for n in negs]
        tg = [1] * len(pos[t]) + [0] * len(negs)
        ap, _ = _ap_auc(sc, tg); aps.append(ap or 0); ns.append(len(pos[t]) + len(negs))
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(ttes, aps, color=[TTE_COLORS[t] for t in ttes], edgecolor="black", width=0.5)
    for b, a, n in zip(bars, aps, ns):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"AP={a:.3f}\nn={n}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim([0, 1.05]); ax.set_ylabel("Vision AP"); ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title(f"Vision AP per TTE — {split} (frozen scorer)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def fig_tte_cm(rows, out, split, thr=0.5, score_key="score"):
    pos, negs = _tte_buckets(rows)
    if not pos or not negs:
        print(f"  [skip] tte_cm_{split}"); return
    ttes = sorted(pos)
    fig, axes = plt.subplots(1, len(ttes), figsize=(4 * len(ttes), 3.6))
    if len(ttes) == 1:
        axes = [axes]
    for ax, t in zip(axes, ttes):
        rs = pos[t] + negs
        y = np.array([int(r["ground_truth"]) for r in rs])
        pred = np.array([1 if r[score_key] >= thr else 0 for r in rs])
        cm = np.array([[int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())],
                       [int(((pred == 0) & (y == 1)).sum()), int(((pred == 1) & (y == 1)).sum())]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel("Pred"); ax.set_ylabel("True"); ax.set_title(f"{t} (n={len(rs)})")
    fig.suptitle(f"Vision CM per TTE — {split} (thr={thr})")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def fig_score_dist_tte(rows, out, split, score_key="score"):
    pos, negs = _tte_buckets(rows)
    if not pos:
        print(f"  [skip] score_dist_{split}"); return
    bins = np.linspace(0, 1, 21)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for t in sorted(pos):
        sc = [p[score_key] for p in pos[t]]
        ax.hist(sc, bins=bins, alpha=0.45, density=True, color=TTE_COLORS[t],
                label=f"{t} pos (n={len(sc)})")
    if negs:
        ax.hist([n[score_key] for n in negs], bins=bins, alpha=0.25, density=True,
                color="grey", label=f"MID neg (n={len(negs)})")
    ax.set_xlabel("Frozen vision P(collision)"); ax.set_ylabel("density")
    ax.set_title(f"Vision score by TTE — {split}"); ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def fig_roc_vision_vs_llm(rows, out, split):
    from sklearn.metrics import roc_curve
    y = np.array([int(r["ground_truth"]) for r in rows])
    if len(set(y.tolist())) < 2:
        print(f"  [skip] roc_{split}: single class"); return
    vs = np.array([float(r.get("vision_score", r.get("score"))) for r in rows])
    ls = np.array([float(r.get("llm_p_yes", np.nan)) for r in rows])
    fig, ax = plt.subplots(figsize=(5, 5))
    fpr, tpr, _ = roc_curve(y, vs); ap_v, auc_v = _ap_auc(vs, y)
    ax.plot(fpr, tpr, color="#2c3e50", lw=2, label=f"vision AUC={auc_v:.3f}")
    if not np.isnan(ls).all():
        fprl, tprl, _ = roc_curve(y, ls); ap_l, auc_l = _ap_auc(ls, y)
        ax.plot(fprl, tprl, color="#e74c3c", lw=2, label=f"LLM P(YES) AUC={auc_l:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC vision vs LLM — {split}")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def fig_agreement(rows, out, split, thr=0.5):
    vis = np.array([1 if float(r.get("vision_score", r.get("score"))) >= thr else 0 for r in rows])
    llm = np.array([1 if str(r.get("llm_verdict", "")).upper() == "YES" else 0 for r in rows])
    gt = np.array([int(r["ground_truth"]) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, (a, b, ttl, xl, yl) in zip(axes, [
            (vis, llm, "LLM verdict vs Vision decision", "LLM", "Vision"),
            (gt, llm, "LLM verdict vs GT", "LLM", "GT")]):
        cm = np.array([[int(((b == 0) & (a == 0)).sum()), int(((b == 1) & (a == 0)).sum())],
                       [int(((b == 0) & (a == 1)).sum()), int(((b == 1) & (a == 1)).sum())]])
        ax.imshow(cm, cmap="Purples")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["No", "Yes"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["No", "Yes"])
        ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=10)
    agree = int((vis == llm).sum())
    fig.suptitle(f"{split}: LLM↔vision {agree}/{len(rows)} agree")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def fig_loss_curve(epoch_metrics, out):
    rows = _load_jsonl(epoch_metrics)
    if not rows:
        print("  [skip] loss_vs_epoch: no epoch_metrics"); return
    ep = [r["epoch"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    if "train_loss" in rows[0]:
        ax.plot(ep, [r["train_loss"] for r in rows], "-o", label="train CE", color="#3498db")
    if "val_ce" in rows[0]:
        ax.plot(ep, [r["val_ce"] for r in rows], "-s", label="val CE", color="#e74c3c")
        best = min(rows, key=lambda r: r["val_ce"])
        ax.axvline(best["epoch"], color="green", ls="--", lw=1, label=f"best val @ep{best['epoch']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("reasoning CE"); ax.set_title("Loss vs epoch (Stage C)")
    ax.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def fig_dce_bar(stageb_abl, stagec_abl, out):
    b = json.load(open(stageb_abl)) if Path(stageb_abl or "").exists() else None
    c = json.load(open(stagec_abl)) if Path(stagec_abl or "").exists() else None
    if not c:
        print("  [skip] dce_bar: need Stage C ablation json"); return
    labels = ["ΔCE-zero", "ΔCE-shuffle"]
    bv = [b.get("dCE_zero", 0), b.get("dCE_shuffle", 0)] if b else [0, 0]
    cv = [c.get("dCE_zero", 0), c.get("dCE_shuffle", 0)]
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, bv, w, label="Stage B", color="#95a5a6")
    ax.bar(x + w / 2, cv, w, label="Stage C", color="#2ecc71")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("ΔCE (higher = more vision-dependent)")
    ax.set_title("Faithfulness: visual-ablation ΔCE, Stage B vs C")
    ax.legend()
    for xi, v in zip(x - w / 2, bv): ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w / 2, cv): ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def _templating_score(rows):
    """Mean pairwise Jaccard similarity of generated reasons (high = templated)."""
    texts = [set(str(r.get("llm_reasoning", "")).lower().split()) for r in rows]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return None
    sims = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            u = texts[i] | texts[j]
            sims.append(len(texts[i] & texts[j]) / len(u) if u else 0)
    return float(np.mean(sims))


def fig_templating(splits, out):
    vals = {k: _templating_score(v) for k, v in splits.items() if v}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        print("  [skip] templating"); return
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(vals.keys()), list(vals.values()), color="#9b59b6", edgecolor="black")
    ax.set_ylabel("mean pairwise gen similarity"); ax.set_ylim([0, 1])
    ax.set_title("Templating proxy (lower = more clip-specific)")
    for i, v in enumerate(vals.values()):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close(); print(f"  saved {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epoch_metrics")
    ap.add_argument("--train_jsonl"); ap.add_argument("--val_jsonl"); ap.add_argument("--test_jsonl")
    ap.add_argument("--stageb_ablation"); ap.add_argument("--stagec_ablation")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    out = Path(args.out_dir); (out / "plots").mkdir(parents=True, exist_ok=True)
    P = out / "plots"

    fig_loss_curve(args.epoch_metrics, P / "loss_vs_epoch.png")
    fig_dce_bar(args.stageb_ablation, args.stagec_ablation, P / "dce_shuffle_zero_B_vs_C.png")

    splits = {"train": _load_jsonl(args.train_jsonl), "val": _load_jsonl(args.val_jsonl),
              "test": _load_jsonl(args.test_jsonl)}
    splits = {k: v for k, v in splits.items() if v}
    for split, rows in splits.items():
        fig_tte_ap(rows, P / f"tte_ap_{split}.png", split)
        fig_tte_cm(rows, P / f"tte_cm_{split}.png", split, args.threshold)
        if split in ("train", "test"):
            fig_score_dist_tte(rows, P / f"score_dist_by_tte_{split}.png", split)
        fig_roc_vision_vs_llm(rows, P / f"roc_vision_vs_llm_{split}.png", split)
        fig_agreement(rows, P / f"llm_agreement_{split}.png", split, args.threshold)
    fig_templating(splits, P / "templating_B_vs_C.png")
    print(f"\nFigures in {P}")


if __name__ == "__main__":
    main()
