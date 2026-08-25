"""Diagnostic figures for the per-clip A0-vs-arm comparison over the 1,761 training pool.

Reads the same score files build_pool1761_comparison.py consumes, so numbers can never
diverge between the workbook and these plots. Every figure is produced twice: once over all
1,761 rows, once over the 348 held-out val rows only (the subset free of memorisation - see
the workbook's caveat). The val version is the one to trust; the all-rows version is included
for completeness, not as evidence.

Usage:
  python plot_pool1761_comparison.py --scores-dir /path/to/pool1761_scores \
      --out-dir reports/figures/pool1761_analysis
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CAP = ROOT / "outputs" / "semantic_captions"
ARMS = ["A0", "A1", "B-v1", "B-v2", "B-v3", "P1"]
ARM_COLORS = {"A0": "#888888", "A1": "#00E676", "B-v1": "#FF6B6B", "B-v2": "#FFA726",
              "B-v3": "#00BFFF", "P1": "#B266FF"}


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def clip_level_split(video_ids, val_frac=0.2, seed=0):
    vids = sorted(set(video_ids))
    random.Random(seed).shuffle(vids)
    n_val = max(1, int(len(vids) * val_frac))
    return set(vids[:n_val])


def is_ok(r, arm):
    return (r[arm] >= 0.5) == (r["gt"] == "YES")


def load_all(scores_dir):
    v12 = {r["frames_dir"]: r for r in load_jsonl(CAP / "Caption_V12_Neutral_1761_fortrain.jsonl")}
    failures = {r["frames_dir"] for r in load_jsonl(CAP / "Caption_Train4500_Failures_587.jsonl")}
    val_vids = clip_level_split([r["video_id"] for r in v12.values()])

    scores = {}
    for arm in ARMS:
        for r in load_jsonl(Path(scores_dir) / f"{arm}.jsonl"):
            scores.setdefault(r["frames_dir"], {})[arm] = r["score"]

    rows = []
    for fd, cap in v12.items():
        rows.append({
            "video_id": cap["video_id"],
            "window": cap.get("horizon_label") or cap.get("requested_time_to_event"),
            "split": "val" if cap["video_id"] in val_vids else "train",
            "mined_failure": fd in failures,
            "gap_trend": cap.get("gap_trend"),
            "primary_agent": cap.get("primary_agent"),
            "agent_visible": cap.get("agent_visible"),
            "gt": cap["gt_verdict"],
            **scores[fd],
        })
    return rows


def fig1_delta_hist(rows, out_dir, suffix):
    fig, axes = plt.subplots(len(ARMS) - 1, 2, figsize=(10, 2.2 * (len(ARMS) - 1)), sharex=True)
    pos = [r for r in rows if r["gt"] == "YES"]
    neg = [r for r in rows if r["gt"] == "NO"]
    for i, arm in enumerate(ARMS[1:]):
        for ax, subset, label, color in [(axes[i, 0], pos, "YES (want score UP)", "#00E676"),
                                         (axes[i, 1], neg, "NO (want score DOWN)", "#FF6B6B")]:
            deltas = [r[arm] - r["A0"] for r in subset]
            ax.hist(deltas, bins=30, color=color, alpha=0.8)
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title(f"{arm} - A0, {label}", fontsize=9)
    fig.suptitle(f"Delta-score vs A0, split by ground truth ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"01_delta_hist_{suffix}.png", dpi=150)
    plt.close(fig)


def fig2_scatter(rows, out_dir, suffix):
    fig, axes = plt.subplots(1, len(ARMS) - 1, figsize=(4 * (len(ARMS) - 1), 4), sharex=True, sharey=True)
    for ax, arm in zip(axes, ARMS[1:]):
        colors = ["#00E676" if r["gt"] == "YES" else "#FF6B6B" for r in rows]
        ax.scatter([r["A0"] for r in rows], [r[arm] for r in rows], c=colors, s=6, alpha=0.4)
        ax.axhline(0.5, color="gray", linewidth=0.8); ax.axvline(0.5, color="gray", linewidth=0.8)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(arm, fontsize=10); ax.set_xlabel("A0 score")
    axes[0].set_ylabel("arm score")
    fig.suptitle(f"A0 vs each arm, per window ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"02_scatter_vs_a0_{suffix}.png", dpi=150)
    plt.close(fig)


def fig3_fix_break_bar(rows, out_dir, suffix):
    a0_fp = [r for r in rows if not is_ok(r, "A0") and r["gt"] == "NO"]
    a0_fn = [r for r in rows if not is_ok(r, "A0") and r["gt"] == "YES"]
    a0_ok = [r for r in rows if is_ok(r, "A0")]
    fixed_fp, fixed_fn, broken, still = [], [], [], []
    for arm in ARMS[1:]:
        ffp = sum(1 for r in a0_fp if is_ok(r, arm))
        ffn = sum(1 for r in a0_fn if is_ok(r, arm))
        brk = sum(1 for r in a0_ok if not is_ok(r, arm))
        fixed_fp.append(ffp); fixed_fn.append(ffn); broken.append(-brk)
        still.append(-(len(a0_fp) + len(a0_fn) - ffp - ffn))
    x = np.arange(len(ARMS) - 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, fixed_fp, color="#00E676", label="fixed FP")
    ax.bar(x, fixed_fn, bottom=fixed_fp, color="#0096CC", label="fixed FN")
    ax.bar(x, broken, color="#FF6B6B", label="broken (A0 was right)")
    ax.set_xticks(x); ax.set_xticklabels(ARMS[1:])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel(f"# of {len(a0_fp)+len(a0_fn)} A0 errors / {len(a0_ok)} A0-correct")
    ax.legend(fontsize=8)
    ax.set_title(f"Fix / break relative to A0 ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"03_fix_break_{suffix}.png", dpi=150)
    plt.close(fig)


def fig4_overlap_matrix(rows, out_dir, suffix):
    fixed = {}
    a0_wrong_ids = [i for i, r in enumerate(rows) if not is_ok(r, "A0")]
    for arm in ARMS[1:]:
        fixed[arm] = {i for i in a0_wrong_ids if is_ok(rows[i], arm)}
    n = len(ARMS) - 1
    mat = np.zeros((n, n))
    for i, a in enumerate(ARMS[1:]):
        for j, b in enumerate(ARMS[1:]):
            mat[i, j] = len(fixed[a] & fixed[b])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_xticklabels(ARMS[1:], rotation=45)
    ax.set_yticks(range(n)); ax.set_yticklabels(ARMS[1:])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] < mat.max() * 0.6 else "black", fontsize=8)
    fig.colorbar(im, label="# A0-errors both arms fix")
    ax.set_title(f"Overlap of fixed A0-errors ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"04_overlap_matrix_{suffix}.png", dpi=150)
    plt.close(fig)


def fig5_fix_by_tte(rows, out_dir, suffix):
    buckets = sorted({r["window"] for r in rows if r["window"]})
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(buckets)); width = 0.14
    for k, arm in enumerate(ARMS):
        rates = []
        for b in buckets:
            sub = [r for r in rows if r["window"] == b]
            rates.append(100 * sum(is_ok(r, arm) for r in sub) / max(1, len(sub)))
        ax.bar(x + k * width, rates, width, label=arm, color=ARM_COLORS[arm])
    ax.set_xticks(x + width * (len(ARMS) - 1) / 2); ax.set_xticklabels(buckets, rotation=30)
    ax.set_ylabel("accuracy @ 0.5 (%)")
    ax.legend(fontsize=8, ncol=3)
    ax.set_title(f"Accuracy by TTE/MID bucket ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"05_fix_by_tte_{suffix}.png", dpi=150)
    plt.close(fig)


def fig6_caption_field_breakdown(rows, out_dir, suffix):
    semantic_arms = ["B-v1", "B-v2", "B-v3", "P1"]
    a0_wrong = [r for r in rows if not is_ok(r, "A0")]
    only_semantic = [r for r in a0_wrong
                     if any(is_ok(r, a) for a in semantic_arms) and not is_ok(r, "A1")]
    nobody = [r for r in a0_wrong if not any(is_ok(r, a) for a in ARMS[1:])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, field in zip(axes, ["gap_trend", "agent_visible"]):
        c1 = Counter(str(r[field]) for r in only_semantic)
        c2 = Counter(str(r[field]) for r in nobody)
        keys = sorted(set(c1) | set(c2))
        x = np.arange(len(keys)); w = 0.35
        ax.bar(x - w/2, [c1.get(k, 0) for k in keys], w, label="fixed only by semantic arms",
               color="#00BFFF")
        ax.bar(x + w/2, [c2.get(k, 0) for k in keys], w, label="nobody fixes", color="#FF6B6B")
        ax.set_xticks(x); ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.set_title(field)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Caption fields: semantic-only fixes vs unfixable ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"06_caption_field_breakdown_{suffix}.png", dpi=150)
    plt.close(fig)


def fig7_mined_vs_easy(rows, out_dir, suffix):
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(ARMS)); width = 0.35
    n_mined = sum(1 for r in rows if r["mined_failure"])
    n_easy = len(rows) - n_mined
    for offset, (label, pred, color) in zip(
            [-width/2, width/2],
            [(f"mined failure ({n_mined})", lambda r: r["mined_failure"], "#FF6B6B"),
             (f"easy ({n_easy})", lambda r: not r["mined_failure"], "#00E676")]):
        rates = [100 * sum(is_ok(r, arm) for r in rows if pred(r)) /
                 max(1, sum(1 for r in rows if pred(r))) for arm in ARMS]
        ax.bar(x + offset, rates, width, label=label, color=color)
    ax.set_xticks(x); ax.set_xticklabels(ARMS)
    ax.set_ylabel("accuracy @ 0.5 (%)")
    ax.legend()
    ax.set_title(f"Mined-failure vs easy-window accuracy, per arm ({suffix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"07_mined_vs_easy_{suffix}.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "reports" / "figures" / "pool1761_analysis"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_all(args.scores_dir)
    val_rows = [r for r in rows if r["split"] == "val"]

    for subset, suffix in [(rows, "all1761"), (val_rows, "val348")]:
        fig1_delta_hist(subset, out_dir, suffix)
        fig2_scatter(subset, out_dir, suffix)
        fig3_fix_break_bar(subset, out_dir, suffix)
        fig4_overlap_matrix(subset, out_dir, suffix)
        fig5_fix_by_tte(subset, out_dir, suffix)
        fig6_caption_field_breakdown(subset, out_dir, suffix)
        fig7_mined_vs_easy(subset, out_dir, suffix)
        print(f"  wrote 7 figures for '{suffix}' (n={len(subset)})")

    print(f"[done] figures in {out_dir}")


if __name__ == "__main__":
    main()
