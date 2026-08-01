"""
mine_train_failures.py
========================
Deliverable 4 of ~/.claude/plans/but-if-b-a1-it-woolly-metcalfe.md. Consumes
the frozen BADAS-Open scorer's output on the train4500 manifest and produces
the failure set + taxonomy that decides whether caption budget should be
failure-targeted or uniform.

JOIN KEY, read before modifying: the scorer's output schema (see
e4_stageA_badas_open_eval.py) is {video_id, ground_truth, group,
time_before_s, score, collision_verdict, split} - it does NOT carry
horizon_label/frames_dir, so this script joins back to the original
train4500_hires.jsonl manifest to recover them. The join key is
(video_id, group), NOT row position. Position would break across chunked
scoring (deliverable 2b writes one scores_chunk_NN.jsonl per chunk, and
whoever concatenates them could reorder). (video_id, group) is safe because
build_train4500_manifest.py assigns each video's 3 rows DISTINCT group values
0/1/2 by construction (TTE_0.5/1.0/1.5 -> 0/1/2 for positives; MID/MID-4/
MID-8 -> 0/1/2 for negatives) - see that script's "GROUP FIELD DESIGN NOTE"
docstring section for why group means different things for the two classes.

Failure definition: binary at threshold 0.5 (matches
data.verdict_threshold in e4_stageA.yaml and the collision_verdict field the
scorer already writes) - the user's explicit choice over a margin-ranked
definition.

Sanity check performed on every row: ground_truth from the scores file must
equal event_occurs from the manifest for the same (video_id, group). A
mismatch means the scores file and manifest are out of sync (wrong chunk
concatenation order, stale manifest, etc.) - this is treated as a hard
failure of the whole run, not a per-row skip, since it would silently corrupt
every downstream number.

Outputs (outputs/train4500_inference/):
  failures.jsonl   - one row per FP/FN: video_id, horizon_label, frames_dir,
                      ground_truth, score, error_type (FP|FN), margin
  taxonomy.json     - failure rate by class, by horizon_label bucket, and
                      score-margin distribution per error type
  (stdout)          - the same taxonomy, printed as tables
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "student_training" / "scripts"))
from evaluate_metrics import compute_group_metrics, compute_metrics  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "dataset" / "manifests" / "train4500_hires.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "train4500_inference"

# A0's real test-set confusion matrix (677 clips, threshold 0.5) - printed as
# the sanity anchor every run should be compared against (see plan doc's
# verification step 5: a large train/test error-rate gap means a
# preprocessing or manifest bug, not a real finding, since A0 trains nothing).
A0_TEST_TP, A0_TEST_FP, A0_TEST_TN, A0_TEST_FN = 308, 130, 209, 30
A0_TEST_ERROR_RATE = (A0_TEST_FP + A0_TEST_FN) / (A0_TEST_TP + A0_TEST_FP + A0_TEST_TN + A0_TEST_FN)


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_scores(paths: list) -> list:
    rows = []
    for p in paths:
        rows.extend(load_jsonl(Path(p)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scores", nargs="+", required=True,
                     help="one or more scorer-output JSONL files (e.g. all scores_chunk_*.jsonl)")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    manifest_rows = load_jsonl(Path(args.manifest))
    manifest_by_key = {(r["video_id"], r["group"]): r for r in manifest_rows}
    if len(manifest_by_key) != len(manifest_rows):
        print(f"WARNING: manifest has {len(manifest_rows)} rows but only "
              f"{len(manifest_by_key)} distinct (video_id, group) keys - "
              f"duplicate keys will silently collide.", file=sys.stderr)

    score_rows = load_scores(args.scores)
    print(f"Loaded {len(score_rows)} scored rows from {len(args.scores)} file(s); "
          f"manifest has {len(manifest_rows)} rows.")

    joined, unjoined, mismatched = [], [], []
    for s in score_rows:
        key = (s["video_id"], s.get("group"))
        m = manifest_by_key.get(key)
        if m is None:
            unjoined.append(s)
            continue
        if int(s["ground_truth"]) != int(m["event_occurs"]):
            mismatched.append((s, m))
            continue
        joined.append({**s, "horizon_label": m["horizon_label"], "frames_dir": m["frames_dir"],
                       "t_seconds": m.get("t_seconds")})

    if unjoined:
        print(f"ERROR: {len(unjoined)} scored rows had no matching (video_id, group) in the "
              f"manifest - sample: {[(r['video_id'], r.get('group')) for r in unjoined[:5]]}",
              file=sys.stderr)
    if mismatched:
        print(f"ERROR: {len(mismatched)} rows have ground_truth != event_occurs for the same "
              f"(video_id, group) - scores file and manifest are OUT OF SYNC. Sample:", file=sys.stderr)
        for s, m in mismatched[:5]:
            print(f"    {s['video_id']} group={s.get('group')}: "
                  f"scores.ground_truth={s['ground_truth']} vs manifest.event_occurs={m['event_occurs']}",
                  file=sys.stderr)
    if unjoined or mismatched:
        print("Refusing to compute failure taxonomy on an out-of-sync join. Fix the input files "
              "and re-run.", file=sys.stderr)
        sys.exit(1)

    print(f"Joined {len(joined)}/{len(score_rows)} rows cleanly.\n")

    df = pd.DataFrame(joined)
    metrics = compute_metrics(df, args.threshold)
    group_metrics = compute_group_metrics(df, args.threshold)

    error_rate = (metrics["fp"] + metrics["fn"]) / metrics["n_total"]
    print("=" * 70)
    print(f"HEADLINE (train4500, threshold={args.threshold})")
    print("=" * 70)
    print(f"  n={metrics['n_total']}  AP={metrics['ap']:.4f}  AUC={metrics['auc_roc']:.4f}")
    print(f"  TP/FP/TN/FN = {metrics['tp']}/{metrics['fp']}/{metrics['tn']}/{metrics['fn']}"
          f"  error_rate={error_rate:.1%}")
    print(f"  --- sanity anchor: A0 test-set error_rate = {A0_TEST_ERROR_RATE:.1%} "
          f"(TP/FP/TN/FN={A0_TEST_TP}/{A0_TEST_FP}/{A0_TEST_TN}/{A0_TEST_FN}) ---")
    gap = abs(error_rate - A0_TEST_ERROR_RATE)
    if gap > 0.05:
        print(f"  ** GAP = {gap:.1%} exceeds the 5pp stop-and-diagnose threshold from the plan's "
              f"verification step 5. A0 trains nothing, so a gap this large likely means a "
              f"manifest/preprocessing bug, not a real finding. **")
    else:
        print(f"  gap = {gap:.1%} - within the expected range for a frozen, untrained scorer.")

    print(f"\n  Per-bucket AP (compute_group_metrics; group labels are cosmetic for negatives, "
          f"see build_train4500_manifest.py's GROUP FIELD DESIGN NOTE):")
    for label, v in group_metrics.items():
        print(f"    {label}: AP={v['ap']} (n={v['n']})")

    # ---- failure set ----
    y_true = df["ground_truth"].astype(int).values
    y_score = df["score"].astype(float).values
    y_pred = (y_score >= args.threshold).astype(int)

    failures = []
    for i, row in enumerate(joined):
        gt, pred, score = int(y_true[i]), int(y_pred[i]), float(y_score[i])
        if gt == pred:
            continue
        error_type = "FP" if pred == 1 else "FN"
        failures.append({
            "video_id": row["video_id"], "horizon_label": row["horizon_label"],
            "frames_dir": row["frames_dir"], "ground_truth": gt, "score": round(score, 4),
            "error_type": error_type, "margin": round(abs(score - args.threshold), 4),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failures_path = out_dir / "failures.jsonl"
    with open(failures_path, "w", encoding="utf-8") as f:
        for r in failures:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(failures)} failures -> {failures_path}")

    # ---- taxonomy ----
    fp_list = [r for r in failures if r["error_type"] == "FP"]
    fn_list = [r for r in failures if r["error_type"] == "FN"]

    by_bucket_total = Counter(r["horizon_label"] for r in joined)
    by_bucket_wrong = Counter(r["horizon_label"] for r in failures)
    bucket_taxonomy = {
        bucket: {"n_total": by_bucket_total[bucket], "n_wrong": by_bucket_wrong.get(bucket, 0),
                 "error_rate": round(by_bucket_wrong.get(bucket, 0) / by_bucket_total[bucket], 4)}
        for bucket in by_bucket_total
    }

    def margin_stats(rows):
        if not rows:
            return {"n": 0}
        margins = [r["margin"] for r in rows]
        return {"n": len(rows), "mean": round(statistics.mean(margins), 4),
                "median": round(statistics.median(margins), 4),
                "min": round(min(margins), 4), "max": round(max(margins), 4)}

    taxonomy = {
        "threshold": args.threshold,
        "n_total": len(joined), "n_failures": len(failures), "error_rate": round(error_rate, 4),
        "a0_test_error_rate": round(A0_TEST_ERROR_RATE, 4), "gap_vs_a0_test": round(gap, 4),
        "by_class": {
            "FP": {"n": len(fp_list), "rate_of_negatives": round(
                len(fp_list) / max(1, sum(1 for r in joined if r["ground_truth"] == 0)), 4)},
            "FN": {"n": len(fn_list), "rate_of_positives": round(
                len(fn_list) / max(1, sum(1 for r in joined if r["ground_truth"] == 1)), 4)},
        },
        "by_bucket": bucket_taxonomy,
        "margin_distribution": {"FP": margin_stats(fp_list), "FN": margin_stats(fn_list)},
    }
    taxonomy_path = out_dir / "taxonomy.json"
    taxonomy_path.write_text(json.dumps(taxonomy, indent=2), encoding="utf-8")

    print(f"\n{'='*70}\nFAILURE TAXONOMY\n{'='*70}")
    print(f"  FP: {len(fp_list)} ({taxonomy['by_class']['FP']['rate_of_negatives']:.1%} of negatives)")
    print(f"  FN: {len(fn_list)} ({taxonomy['by_class']['FN']['rate_of_positives']:.1%} of negatives)")
    print(f"\n  By bucket:")
    for bucket, v in sorted(bucket_taxonomy.items()):
        print(f"    {bucket:10} n={v['n_total']:4}  wrong={v['n_wrong']:4}  rate={v['error_rate']:.1%}")
    print(f"\n  Margin (|score-threshold|, how confidently wrong):")
    for et, stats in taxonomy["margin_distribution"].items():
        if stats["n"]:
            print(f"    {et}: n={stats['n']} mean={stats['mean']} median={stats['median']} "
                  f"range=[{stats['min']}, {stats['max']}]")
    print(f"\nWrote taxonomy -> {taxonomy_path}")

    # ---- allocation-decision hint (per the plan's stated purpose for this table) ----
    bucket_rates = [v["error_rate"] for v in bucket_taxonomy.values()]
    spread = max(bucket_rates) - min(bucket_rates) if bucket_rates else 0.0
    print(f"\n{'='*70}")
    if spread > 0.15:
        print(f"  Bucket error-rate spread = {spread:.1%} (max-min across the 6 buckets) - "
              f"SYSTEMATIC. Failures concentrate in specific buckets, which argues FOR "
              f"failure-targeted caption allocation over uniform.")
    else:
        print(f"  Bucket error-rate spread = {spread:.1%} - DIFFUSE. Failures are not "
              f"concentrated in particular buckets, which argues FOR uniform caption "
              f"allocation over failure-targeting.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
