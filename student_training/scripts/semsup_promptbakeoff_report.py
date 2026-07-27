"""
semsup_promptbakeoff_report.py
================================
Gate 2 collation: reads each arm's b1_metrics.json (from semsup_b1_probe.py
--captions arm_X.jsonl), reports significance vs chance per arm, a PAIRED
arm-vs-arm comparison on the same val clips (reusing the paired-bootstrap
approach that produced the A1-vs-B CI - see docs_agents/EXPERIMENTS.md), and
applies the plan's decision rule mechanically. See PLAN: prompt-bakeoff-harness
(2026-07-27).

Paired, not independent, because arm_a/arm_b/arm_c are built from the SAME
manifest with the SAME --seed, so clip_level_split() (which splits by sorted
video_id, independent of caption content) produces an IDENTICAL val clip set
across arms - this script asserts that rather than assuming it. That identity
is what makes resampling clip INDICES together (not resampling each arm's hits
independently) a valid paired bootstrap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "semantic_captions" / "promptbakeoff"
PARENT_SUMMARY = PROJECT_ROOT / "outputs" / "semantic_captions" / "summary.md"


def load_arm(path: Path, label: str) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    clip_ids = d.get("val_clip_ids")
    hits = d.get("val_clip_hits")
    if clip_ids is None or hits is None:
        raise SystemExit(f"{path} has no val_clip_ids/val_clip_hits - re-run "
                          f"semsup_b1_probe.py (needs the per-clip-detail fix).")
    n = len(clip_ids)
    acc = sum(hits) / n if n else float("nan")
    return {
        "label": label, "path": str(path), "clip_ids": clip_ids, "hits": np.array(hits),
        "n_val_clips": n, "clip_acc": acc,
        "chance": 1.0 / n if n else float("nan"),
        "control_clip_acc": d.get("control_mean_embedding", {}).get("retrieval_top1_acc_clip"),
        "best_epoch": d.get("best_epoch"), "loss": d.get("loss"),
    }


def vs_chance_binomial(arm: dict) -> dict:
    n = arm["n_val_clips"]
    k = int(round(arm["clip_acc"] * n))
    res = binomtest(k, n, 1.0 / n, alternative="greater")
    return {"k": k, "n": n, "p_value": float(res.pvalue),
            "significant_at_05": bool(res.pvalue < 0.05)}


def paired_bootstrap_diff(arm_x: dict, arm_y: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    """arm_x['clip_ids'] must equal arm_y['clip_ids'] (same set, checked by caller).
    Aligns hits by clip_id (not by list position - defensive against any future
    change to save-order), then resamples clip INDICES with replacement, applying
    the SAME resampled indices to both arms each draw."""
    common = sorted(set(arm_x["clip_ids"]) & set(arm_y["clip_ids"]))
    idx_x = {c: i for i, c in enumerate(arm_x["clip_ids"])}
    idx_y = {c: i for i, c in enumerate(arm_y["clip_ids"])}
    hx = np.array([arm_x["hits"][idx_x[c]] for c in common])
    hy = np.array([arm_y["hits"][idx_y[c]] for c in common])
    n = len(common)
    point_diff = float(hy.mean() - hx.mean())

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sel = rng.integers(0, n, n)
        diffs[i] = hy[sel].mean() - hx[sel].mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"n_common_clips": n, "point_diff_y_minus_x": point_diff,
            "ci95_lo": float(lo), "ci95_hi": float(hi),
            "p_y_greater_x": float((diffs > 0).mean()),
            "crosses_zero": bool(lo <= 0 <= hi)}


def decide(a: dict, b: dict, c: dict, sig_a: dict, sig_b: dict,
           ab: dict, ac: dict, bc: dict, ref: dict | None, ref_sig: dict | None) -> str:
    """Mechanical application of the plan's decision table. 'beats' = clears the
    binomial test vs chance for itself AND the paired comparison shows it above
    the comparison arm with the CI not crossing zero in its favor."""
    a_beats_c = sig_a["significant_at_05"] and not ac["crosses_zero"] and ac["point_diff_y_minus_x"] < 0
    b_beats_a = sig_b["significant_at_05"] and not ab["crosses_zero"] and ab["point_diff_y_minus_x"] > 0

    lines = []
    if ref is not None:
        ref_beats_both = (ref["clip_acc"] > a["clip_acc"] and ref["clip_acc"] > b["clip_acc"])
        if ref_beats_both:
            lines.append("REF (incumbent 267 captions) beats BOTH new arms A and B on clip-level "
                          "retrieval@1. -> STOP: the new vision-grounded captions are worse than "
                          "the incumbent rephrasings. Diagnose the captioning before spending "
                          "anything further.")
            return "\n".join(lines)

    if a_beats_c and not b_beats_a:
        lines.append("A beats C (significant), B is NOT distinguishably better than A.")
        lines.append("-> Structure carries it; stating the verdict adds nothing. STRONGEST "
                      "THESIS CLAIM. Scale Arm A to ~4.5k.")
    elif a_beats_c and b_beats_a:
        lines.append("A beats C (significant), AND B beats A (significant).")
        lines.append("-> Both channels contribute; C quantifies how much of B's gain is just "
                      "the label. Scale Arm B, report the decomposition honestly.")
    elif not a_beats_c and b_beats_a:
        lines.append("A does NOT beat C, but B beats A (significant).")
        lines.append("-> The gain is the LABEL, not the language. Do not claim language-as-"
                      "structure - use label smoothing instead. This is a legitimate, useful "
                      "outcome: it stops a false claim before it's published.")
    else:
        lines.append("Neither A beats C, nor does B beat A.")
        lines.append("-> Nothing works at this scale, even with a fixed objective and clean "
                      "sampling. Deprioritize the thread (see docs_agents/DECISIONS.md option 4).")
    return "\n".join(lines)


def render_summary_md(arms: dict, sig: dict, pairs: dict, ref, ref_sig, verdict_text: str) -> str:
    lines = ["# Prompt bake-off report (Gate 2)", ""]
    lines.append("| arm | n_val_clips | clip-level retrieval@1 | chance | vs-chance p | control (collapse) |")
    lines.append("|---|---|---|---|---|---|")
    order = ["A", "B", "C"] + (["REF"] if ref else [])
    all_arms = dict(arms)
    if ref:
        all_arms["REF"] = ref
    for name in order:
        a = all_arms[name]
        p = sig.get(name, {}).get("p_value")
        p_str = f"{p:.4f}" if p is not None else "-"
        lines.append(f"| {name} | {a['n_val_clips']} | {a['clip_acc']:.4f} | "
                      f"{a['chance']:.4f} | {p_str} | {a['control_clip_acc']:.4f} |")
    lines.append("")
    lines.append("## Paired comparisons (same val clips, 5000-resample bootstrap)")
    lines.append("| pair | diff | 95% CI | crosses zero |")
    lines.append("|---|---|---|---|")
    for name, p in pairs.items():
        lines.append(f"| {name} | {p['point_diff_y_minus_x']:+.4f} | "
                      f"[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}] | {p['crosses_zero']} |")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(verdict_text)
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-a", required=True)
    ap.add_argument("--arm-b", required=True)
    ap.add_argument("--arm-c", required=True)
    ap.add_argument("--ref", default=None, help="incumbent 267-caption b1_metrics.json, optional")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--update-parent-summary", action="store_true",
                     help="append a pointer line to the parent outputs/semantic_captions/"
                          "summary.md. Off by default so smoke/toy runs don't pollute it.")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    a = load_arm(Path(args.arm_a), "A")
    b = load_arm(Path(args.arm_b), "B")
    c = load_arm(Path(args.arm_c), "C")
    ref = load_arm(Path(args.ref), "REF") if args.ref else None

    for other in (b, c):
        if set(a["clip_ids"]) != set(other["clip_ids"]):
            print(f"WARNING: arm {other['label']}'s val clip set differs from arm A's - "
                  f"the paired comparison will only use the {len(set(a['clip_ids']) & set(other['clip_ids']))} "
                  f"clips in common. Check --seed and the manifest match across arms.")

    sig = {"A": vs_chance_binomial(a), "B": vs_chance_binomial(b), "C": vs_chance_binomial(c)}
    if ref:
        sig["REF"] = vs_chance_binomial(ref)

    print("=" * 90)
    print("GATE 2: prompt bake-off report")
    print("=" * 90)
    for name, arm in (("A", a), ("B", b), ("C", c)) + ((("REF", ref),) if ref else ()):
        s = sig[name]
        print(f"  arm {name:4s} n_clips={arm['n_val_clips']:3d}  clip_acc={arm['clip_acc']:.4f}  "
              f"chance={arm['chance']:.4f}  vs-chance p={s['p_value']:.4f}  "
              f"{'SIGNIFICANT' if s['significant_at_05'] else 'not significant'}")

    pairs = {
        "A_vs_C": paired_bootstrap_diff(c, a, args.n_boot, args.seed),
        "B_vs_A": paired_bootstrap_diff(a, b, args.n_boot, args.seed),
        "B_vs_C": paired_bootstrap_diff(c, b, args.n_boot, args.seed),
    }
    print()
    for name, p in pairs.items():
        print(f"  {name}: diff={p['point_diff_y_minus_x']:+.4f}  "
              f"95% CI=[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]  "
              f"crosses_zero={p['crosses_zero']}")

    verdict_text = decide(a, b, c, sig["A"], sig["B"],
                           pairs["B_vs_A"], pairs["A_vs_C"], pairs["B_vs_C"],
                           ref, sig.get("REF"))
    print()
    print("--- DECISION ---")
    print(verdict_text)
    print("=" * 90)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_summary_md({"A": a, "B": b, "C": c}, sig, pairs, ref, sig.get("REF"), verdict_text)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print(f"\nwrote {out_dir / 'summary.md'}")

    report_json = {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk not in ("hits",)}
                 for k, v in {"A": a, "B": b, "C": c, **({"REF": ref} if ref else {})}.items()},
        "significance_vs_chance": sig, "paired_comparisons": pairs, "decision": verdict_text,
    }
    (out_dir / "report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    if args.update_parent_summary and PARENT_SUMMARY.exists():
        with open(PARENT_SUMMARY, "a", encoding="utf-8") as f:
            f.write(f"\n- Prompt bake-off (Gate 2) report updated -> "
                    f"see `outputs/semantic_captions/promptbakeoff/summary.md`\n")
        print(f"appended pointer line to {PARENT_SUMMARY}")


if __name__ == "__main__":
    main()
