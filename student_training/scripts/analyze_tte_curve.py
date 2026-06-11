"""
analyze_tte_curve.py
====================
Stage 4 (local analysis): TTE anticipation curve for e3a (headline) and e3b (ablation).

Reads:
  - Existing full-test eval JSONLs for 0.5 s scores + fixed negative pool
  - Pod-produced TTE curve JSONLs for 1.0 / 1.5 / 2.0 / 2.5 / 3.0 s scores
  - TTE curve manifests for requested_tte_s grouping

Produces (all in outputs/e3b_student_267clips_tte/tte_curve/):
  1. ap_vs_tte_curve.png    — headline AP curve, both halves + pooled, CI bands
  2. score_vs_tte_curve.png — mean score|pos per offset (threshold-free)
  3. lead_time.png          — fraction of positives exceeding FPR-matched threshold
  4. tte_curve_numbers.json — all numeric results for updating summary.md

Run:
  python student_training/scripts/analyze_tte_curve.py

Designed to run AFTER pod eval (Stage 3). If a pod JSONL is missing the script
exits with a clear message listing which file to produce first.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[2]
OUT  = REPO / "outputs" / "e3b_student_267clips_tte" / "tte_curve"

# ── Input file paths ──────────────────────────────────────────────────────────

# Full test evals (all clips — used for 0.5s positives + negative pool)
E3A_FULL = {
    "private": REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_epoch07.jsonl",
    "public":  REPO / "outputs/e3a_student_90clips/e3a_test_extracted/outputs/trained/e3a_test_public_epoch07.jsonl",
}
E3B_FULL = {
    "private": REPO / "outputs/e3b_student_267clips_tte/e3b_test_private/outputs/trained/e3b_test_private_step000099.jsonl",
    "public":  REPO / "outputs/e3b_student_267clips_tte/e3b_test_public/outputs/trained/e3b_test_public_step000099.jsonl",
}

# TTE curve eval JSONLs (from pod — only group-0 positives at 1.0..3.0 s)
E3A_CURVE = {
    "private": OUT / "e3a_tte_curve_private_epoch07.jsonl",
    "public":  OUT / "e3a_tte_curve_public_epoch07.jsonl",
}
E3B_CURVE = {
    "private": OUT / "e3b_tte_curve_private_step000099.jsonl",
    "public":  OUT / "e3b_tte_curve_public_step000099.jsonl",
}

# TTE manifests (for requested_tte_s grouping)
MANIF = {
    "private": REPO / "dataset/manifests/test_tte_curve_private_manifest.jsonl",
    "public":  REPO / "dataset/manifests/test_tte_curve_public_manifest.jsonl",
}

OFFSETS_ALL = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # x-axis (in seconds, "before event")
OFFSETS_NEW = [1.0, 1.5, 2.0, 2.5, 3.0]        # extracted new offsets
N_BOOT      = 5000
FPR_LEVELS  = [0.05, 0.10]
C_E3A, C_E3B = "#2980b9", "#e74c3c"   # blue = e3a (headline), red = e3b (ablation)

# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _check_required_files() -> None:
    missing = []
    for m, paths in [("e3a curve", E3A_CURVE), ("e3b curve", E3B_CURVE)]:
        for half, p in paths.items():
            if not p.exists():
                missing.append(f"  [{m}] {half}: {p}")
    if missing:
        print("ERROR — Missing pod eval JSONLs. Run these inference passes first:")
        for m in missing:
            print(m)
        sys.exit(1)


def _build_manifest_tte_map(half: str) -> Dict[str, float]:
    """video_id -> requested_tte_s for the curve manifest."""
    rows = _load_jsonl(MANIF[half])
    return {r["video_id"]: float(r["requested_tte_s"]) for r in rows}


def _load_half_data(half: str, model: str) -> Dict:
    """
    Returns:
      pos_05: {vid: score}  — group-0 positives at 0.5 s (from full eval)
      pos_new: {tte_s: {vid: score}} — 1.0..3.0 s (from curve eval)
      neg: {vid: score} — all safe clips from full eval
    """
    full_paths = E3A_FULL if model == "e3a" else E3B_FULL
    curve_paths = E3A_CURVE if model == "e3a" else E3B_CURVE

    full = _load_jsonl(full_paths[half])
    curve = _load_jsonl(curve_paths[half])
    tte_map = _build_manifest_tte_map(half)

    pos_05 = {r["video_id"]: r["score"]
              for r in full
              if r.get("ground_truth") == 1 and r.get("group") == 0}
    neg = {r["video_id"]: r["score"]
           for r in full
           if r.get("ground_truth") == 0}

    pos_new: Dict[float, Dict[str, float]] = {t: {} for t in OFFSETS_NEW}
    for r in curve:
        vid = r["video_id"]
        tte = tte_map.get(vid) or r.get("time_before_s")
        if tte is not None:
            tte = float(tte)
            if tte in pos_new:
                pos_new[tte][vid] = r["score"]

    return {"pos_05": pos_05, "pos_new": pos_new, "neg": neg}


def _scores_at_offset(data: Dict, offset: float) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (pos_scores, neg_scores, pos_vids) for a given offset."""
    if offset == 0.5:
        p = data["pos_05"]
    else:
        p = data["pos_new"].get(offset, {})
    n = data["neg"]
    pos_vids = sorted(p.keys())
    pos_s = np.array([p[v] for v in pos_vids])
    neg_s = np.array(list(n.values()))
    return pos_s, neg_s, pos_vids


def _ap_at_offset(data: Dict, offset: float) -> float:
    pos_s, neg_s, _ = _scores_at_offset(data, offset)
    if len(pos_s) == 0 or len(neg_s) == 0:
        return float("nan")
    yt = np.concatenate([np.ones(len(pos_s)), np.zeros(len(neg_s))])
    ys = np.concatenate([pos_s, neg_s])
    return float(average_precision_score(yt, ys))


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _bootstrap_ap_curve(data_a: Dict, data_b: Optional[Dict], half: str,
                         n_boot: int = N_BOOT, seed: int = 42
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap CI for AP vs offset curve for model A (and optionally model B).
    Returns (ap_mean, ci_lo, ci_hi) for each offset — shape (n_offsets,).
    Uses paired bootstrap if data_b is provided (same neg resample per iteration).
    Since positives vary per offset, we resample neg pool + per-offset pos pool.
    """
    rng = np.random.default_rng(seed)
    boot_aps = np.zeros((n_boot, len(OFFSETS_ALL)))

    neg_scores = np.array(list(data_a["neg"].values()))
    n_neg = len(neg_scores)

    for b in range(n_boot):
        neg_idx = rng.integers(0, n_neg, size=n_neg)
        neg_b = neg_scores[neg_idx]
        for oi, offset in enumerate(OFFSETS_ALL):
            if offset == 0.5:
                pos_dict = data_a["pos_05"]
            else:
                pos_dict = data_a["pos_new"].get(offset, {})
            pos_s = np.array(list(pos_dict.values()))
            if len(pos_s) == 0:
                boot_aps[b, oi] = float("nan")
                continue
            pos_idx = rng.integers(0, len(pos_s), size=len(pos_s))
            pos_b = pos_s[pos_idx]
            yt = np.concatenate([np.ones(len(pos_b)), np.zeros(len(neg_b))])
            ys = np.concatenate([pos_b, neg_b])
            boot_aps[b, oi] = float(average_precision_score(yt, ys))

    ap_mean = np.nanmean(boot_aps, axis=0)
    ci_lo = np.nanpercentile(boot_aps, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_aps, 97.5, axis=0)
    return ap_mean, ci_lo, ci_hi


# ── Figure 1: AP vs TTE ────────────────────────────────────────────────────────

def fig_ap_vs_tte(data: Dict) -> Dict:
    """3-panel figure: Private / Public / Pooled. Returns numeric results."""
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    xs = np.array(OFFSETS_ALL)
    results = {}

    panels = [("private", "Private (n=142 pos, 339 neg)", axes[0]),
              ("public",  "Public (n=142 pos, 333 neg)",  axes[1]),
              ("pooled",  "Pooled (n=284 pos, 672 neg)",  axes[2])]

    for half, title, ax in panels:
        for model, color, ls, lw, zorder, label in [
            ("e3a", C_E3A, "-",  2.5, 3, "e3a epoch-7 (headline)"),
            ("e3b", C_E3B, "--", 1.8, 2, "e3b ep3 (ablation)"),
        ]:
            if half == "pooled":
                # Merge both halves
                d = _pool_data(data[model]["private"], data[model]["public"])
            else:
                d = data[model][half]

            ap_mean, ci_lo, ci_hi = _bootstrap_ap_curve(d, None, half)
            ap_obs = np.array([_ap_at_offset(d, o) for o in OFFSETS_ALL])

            ax.fill_between(xs, ci_lo, ci_hi, alpha=0.15, color=color)
            ax.plot(xs, ap_obs, ls, color=color, lw=lw, label=label, zorder=zorder)
            for xi, (x, ap) in enumerate(zip(xs, ap_obs)):
                ax.annotate(f"{ap:.3f}", (x, ap), textcoords="offset points",
                            xytext=(0, 6 if model == "e3a" else -14),
                            ha="center", fontsize=7, color=color)

            results[f"{model}_{half}_ap"] = ap_obs.tolist()
            results[f"{model}_{half}_ci_lo"] = ci_lo.tolist()
            results[f"{model}_{half}_ci_hi"] = ci_hi.tolist()

        # Cross-check markers: existing group-1/2 bucket APs (different scenes)
        # These would need to be passed in separately — mark as TODO for now.

        ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
        ax.set_xticks(xs)
        ax.set_xlabel("seconds before collision event", fontsize=10)
        ax.set_title(title, fontsize=11)
        if half == "private":
            ax.set_ylabel("Average Precision (threshold-free)", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
        ax.set_ylim(0.4, 1.0)

    fig.suptitle("AP vs TTE — within-scene, confound-free anticipation curve\n"
                 "e3a = headline  |  e3b = ablation (saturated scores, compressed range)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "ap_vs_tte_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")
    return results


# ── Figure 2: Score vs TTE ─────────────────────────────────────────────────────

def fig_score_vs_tte(data: Dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    xs = np.array(OFFSETS_ALL)

    for ax, half in zip(axes, ("private", "public")):
        for model, color, ls, lw, label in [
            ("e3a", C_E3A, "-",  2.5, "e3a epoch-7 (headline)"),
            ("e3b", C_E3B, "--", 1.8, "e3b ep3 (ablation)"),
        ]:
            d = data[model][half]
            neg_mean = np.mean(list(d["neg"].values()))

            pos_means = []
            for offset in OFFSETS_ALL:
                if offset == 0.5:
                    p = list(d["pos_05"].values())
                else:
                    p = list(d["pos_new"].get(offset, {}).values())
                pos_means.append(np.mean(p) if p else float("nan"))

            ax.plot(xs, pos_means, ls, color=color, lw=lw, marker="o",
                    label=f"{label}")

        # Neg baselines
        neg_e3a = np.mean(list(data["e3a"][half]["neg"].values()))
        neg_e3b = np.mean(list(data["e3b"][half]["neg"].values()))
        ax.axhline(neg_e3a, color=C_E3A, ls=":", lw=1, alpha=0.6,
                   label=f"e3a neg mean ({neg_e3a:.3f})")
        ax.axhline(neg_e3b, color=C_E3B, ls=":", lw=1, alpha=0.6,
                   label=f"e3b neg mean ({neg_e3b:.3f})")

        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_xticks(xs)
        ax.set_xlabel("seconds before collision event", fontsize=10)
        ax.set_title(f"{half.capitalize()} — mean score | positive clips", fontsize=11)
        ax.legend(fontsize=8, loc="lower left")

    axes[0].set_ylabel("mean predicted P(collision)", fontsize=10)
    fig.suptitle("Score vs TTE — threshold-free mean score on known-positive clips\n"
                 "e3b saturation: high scores across all horizons vs e3a's spread",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "score_vs_tte_curve.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


# ── Figure 3: Lead time (FPR-matched threshold) ───────────────────────────────

def fig_lead_time(data: Dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    xs = np.array(OFFSETS_ALL)

    for row, fpr in enumerate(FPR_LEVELS):
        for col, half in enumerate(("private", "public")):
            ax = axes[row, col]
            for model, color, ls, lw, label in [
                ("e3a", C_E3A, "-",  2.5, f"e3a epoch-7"),
                ("e3b", C_E3B, "--", 1.8, f"e3b ep3"),
            ]:
                d = data[model][half]
                neg_scores = np.array(list(d["neg"].values()))
                thr = float(np.percentile(neg_scores, (1 - fpr) * 100))

                fire_rates = []
                for offset in OFFSETS_ALL:
                    if offset == 0.5:
                        pos_s = np.array(list(d["pos_05"].values()))
                    else:
                        pos_s = np.array(list(d["pos_new"].get(offset, {}).values()))
                    if len(pos_s) == 0:
                        fire_rates.append(float("nan"))
                    else:
                        fire_rates.append(float(np.mean(pos_s >= thr)))

                ax.plot(xs, fire_rates, ls, color=color, lw=lw, marker="o",
                        label=f"{label} (thr={thr:.3f})")

            ax.axhline(fpr, color="gray", ls=":", lw=1, label=f"FPR={fpr:.0%}")
            ax.set_xticks(xs)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("seconds before event", fontsize=9)
            ax.set_ylabel("fraction firing", fontsize=9)
            ax.set_title(f"{half.capitalize()} | FPR={fpr:.0%} threshold", fontsize=10)
            ax.legend(fontsize=8, loc="lower left")

    fig.suptitle("Lead-time: fraction of positives exceeding FPR-matched threshold\n"
                 "(threshold = (1−FPR) percentile of negative scores — same false-alarm budget)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = OUT / "lead_time.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"Saved: {p}")


# ── Pool helper ───────────────────────────────────────────────────────────────

def _pool_data(d_priv: Dict, d_pub: Dict) -> Dict:
    """Merge private + public data for the pooled curve panel."""
    pos_05 = {**d_priv["pos_05"], **d_pub["pos_05"]}
    pos_new = {}
    for offset in OFFSETS_NEW:
        pos_new[offset] = {**d_priv["pos_new"].get(offset, {}),
                           **d_pub["pos_new"].get(offset, {})}
    neg = {**{f"priv_{k}": v for k, v in d_priv["neg"].items()},
           **{f"pub_{k}":  v for k, v in d_pub["neg"].items()}}
    return {"pos_05": pos_05, "pos_new": pos_new, "neg": neg}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _check_required_files()
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data: Dict[str, Dict[str, Dict]] = {"e3a": {}, "e3b": {}}
    for model in ("e3a", "e3b"):
        for half in ("private", "public"):
            data[model][half] = _load_half_data(half, model)
            n_pos05  = len(data[model][half]["pos_05"])
            n_neg    = len(data[model][half]["neg"])
            n_new    = {o: len(data[model][half]["pos_new"].get(o, {})) for o in OFFSETS_NEW}
            print(f"  [{model}][{half}]: pos_0.5={n_pos05}  neg={n_neg}  "
                  f"new_pos={n_new}")

    print("\nGenerating figures (bootstrap CI may take ~30s)...")
    ap_results = fig_ap_vs_tte(data)
    fig_score_vs_tte(data)
    fig_lead_time(data)

    # Print summary table
    print("\n--- AP vs TTE summary ---")
    header = f"{'Offset':>8}  " + "  ".join(
        f"e3a-{h[:3]:>4}  e3b-{h[:3]:>4}" for h in ("private", "public", "pooled"))
    print(header)
    for oi, offset in enumerate(OFFSETS_ALL):
        row_vals = []
        for model in ("e3a", "e3b"):
            for half in ("private", "public", "pooled"):
                ap_list = ap_results.get(f"{model}_{half}_ap")
                row_vals.append(f"{ap_list[oi]:.3f}" if ap_list else "  n/a")
        print(f"  {offset:>5.1f}s:  " + "  ".join(row_vals))

    # Dump numbers
    out_json = OUT / "tte_curve_numbers.json"
    out_json.write_text(json.dumps(ap_results, indent=2), encoding="utf-8")
    print(f"\nNumerics saved: {out_json}")
    print(f"All figures  -> {OUT}")


if __name__ == "__main__":
    main()
