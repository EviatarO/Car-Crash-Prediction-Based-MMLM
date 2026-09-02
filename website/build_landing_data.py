"""
build_landing_data.py
======================
Generates website/landing_data.js (window.LANDING_DATA = {...}) for the landing page:
  - showcase: all 321 V12-captioned A1-failure windows, each joined to its clip's
    train.xlsx metadata (time_of_event/time_of_alert/response_time/target) and its
    video/thumb paths - so the landing page's random showcase needs no separate
    video-resolution logic.
  - metrics: TP/FN/FP/TN/Prec/Rec/F1/Acc/AP/AUC at threshold 0.5 for A0 (frozen
    baseline), A1 (crash-only control), v12 (semantic, epoch 10) on the SAME 677
    test clips - computed here, never hand-copied, so the landing page can never
    drift from the actual score files.
  - lambda: the current semantic-loss weight, carried as DATA so the architecture
    diagram's caption can show it without re-drawing the SVG when it changes.

Reuses build_site_data.py's build_train()/url_of()/find_thumb() for path resolution
instead of re-deriving train.xlsx parsing and the thumbnail fallback chain here.

    python website/build_landing_data.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site_data import MMLM_AI, build_train  # noqa: E402

# Every metric on this page comes from the same function the training pipeline itself
# uses, so the website can never disagree with the run reports by using a different
# formula, threshold convention or rounding.
sys.path.insert(0, str(MMLM_AI / "student_training" / "scripts"))
from metrics_core import metrics_from_arrays  # noqa: E402

OUT = Path(__file__).resolve().parent / "landing_data.js"

CAPTIONS = MMLM_AI / "outputs" / "a1fail321" / "Caption_a1fail321_V12.jsonl"
E4 = MMLM_AI / "outputs" / "e4_vjepa_reason"

# Explicit timeline order: the landing page renders the experiments from oldest to newest.
# Keeping the order as data avoids relying on dict iteration or hand-sorting in the browser.
#
# A1 deliberately reads its ORIGINAL evaluation (a1_1761, epoch 4 - the run the 0.900
# headline number was published from), not the a1fail321 re-score. The two differ: the
# re-score flips one clip (fp 124 vs 123) and lands AP at 0.8995. Where a `summary` file
# exists it wins for AP/AUC, because the per-clip dumps round `score` to 4 dp
# (semsup_train.py: `round(s, 4)`) and those ties perturb average precision - AUC and the
# confusion matrix are insensitive to them, which is why the two sources agree there and
# not on AP. See the f1/recall cross-check in compute_metrics().
EXPERIMENTS = [
    dict(order=0, arm="A0", label="A0 · Baseline",
         path=E4 / "StageA_scorer" / "badas_open_private.jsonl",
         gt_key="ground_truth", summary=None,
         source="e4_vjepa_reason/StageA_scorer/badas_open_private.jsonl"),
    dict(order=1, arm="A1", label="A1 · Crash-only (control)",
         path=E4 / "a1_1761" / "test_results_ep04.jsonl",
         gt_key="ground_truth", summary=E4 / "a1_1761" / "test_summary.json",
         source="e4_vjepa_reason/a1_1761 test_results_ep04.jsonl + test_summary.json (epoch 4)"),
    dict(order=2, arm="v12", label="V12 · Semantic",
         path=MMLM_AI / "outputs" / "a1fail321" / "test_scores" / "v12_ep10.jsonl",
         gt_key="gt_verdict", summary=None,
         source="a1fail321/test_scores/v12_ep10.jsonl (epoch 10)"),
]
SEMANTIC_LAMBDA = 0.2   # current --semantic-weight; a landing-page field, not baked into the SVG

# Expected values, verified against the source artifacts - asserted below so a future
# score-file change is caught loudly rather than silently propagating a stale table.
EXPECTED = {
    "A0":  dict(n=677, tp=308, fn=30, fp=130, tn=209),
    "A1":  dict(n=677, tp=320, fn=18, fp=123, tn=216),
    "v12": dict(n=677, tp=253, fn=85, fp=39, tn=300),
}


def to01(v):
    """Labels arrive as 0/1 ints ('ground_truth') or YES/NO strings ('gt_verdict')."""
    return int(v) if not isinstance(v, str) else (1 if v == "YES" else 0)


def compute_metrics():
    table = []
    for e in EXPERIMENTS:
        arm, path = e["arm"], e["path"]
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        y = [to01(r[e["gt_key"]]) for r in rows]
        s = [float(r["score"]) for r in rows]
        m = metrics_from_arrays(y, s, threshold=0.5)

        exp = EXPECTED[arm]
        got = dict(n=m["n_total"], tp=m["tp"], fn=m["fn"], fp=m["fp"], tn=m["tn"])
        assert got == exp, f"{arm}: metrics drifted from the expected/verified values.\n" \
                            f"  expected {exp}\n  got      {got}\n" \
                            f"  (source: {path})"

        ap, auc, note = m["ap"], m["auc_roc"], None
        if e["summary"]:
            # The run's own published report. Cross-check the metrics that ARE reproducible
            # from the rounded dump before trusting its AP - if f1/recall disagree, the two
            # files describe different runs and the override would be silently wrong.
            best = json.load(open(e["summary"], encoding="utf-8"))["checkpoints"][0]
            for key, mine in (("f1", m["f1"]), ("recall", m["recall_sensitivity_tpr"])):
                assert abs(best[key] - mine) < 1e-3, \
                    f"{arm}: {e['summary'].name} {key}={best[key]} disagrees with the " \
                    f"per-clip dump ({mine}) - the two files are not the same run."
            ap, auc = round(float(best["test_ap"]), 4), round(float(best["auc_roc"]), 4)
            note = f"AP/AUC from test_summary.json (dump reproduces AUC exactly; " \
                   f"its AP reads {m['ap']} because scores are stored rounded to 4 dp)"

        table.append({
            "arm": arm, "label": e["label"], "timeline_order": e["order"],
            "n": m["n_total"], "tp": m["tp"], "fn": m["fn"], "fp": m["fp"], "tn": m["tn"],
            "precision": m["precision"], "recall": m["recall_sensitivity_tpr"],
            "f1": m["f1"], "accuracy": m["accuracy"], "ap": ap, "auc": auc,
            "threshold": 0.5, "source": e["source"], "note": note,
        })
        print(f"[metrics] {arm:<4} n={m['n_total']} TP={m['tp']} FN={m['fn']} "
              f"FP={m['fp']} TN={m['tn']}  Prec={m['precision']:.4f} "
              f"Rec={m['recall_sensitivity_tpr']:.4f} F1={m['f1']:.4f} "
              f"Acc={m['accuracy']:.4f} AP={ap:.4f} AUC={auc:.4f}")
        if note:
            print(f"           {note}")
    return table


def build_showcase():
    train_by_id = {c["id"]: c for c in build_train()}
    rows = [json.loads(l) for l in open(CAPTIONS, encoding="utf-8") if l.strip()]
    showcase, missing = [], 0
    for r in rows:
        clip = train_by_id.get(r["video_id"])
        if clip is None or clip["video_missing"]:
            missing += 1
            continue
        showcase.append({
            "id": clip["id"], "video": clip["video"], "thumb": clip["thumb"],
            "time_of_event": clip["time_of_event"], "time_of_alert": clip["time_of_alert"],
            "response_time": clip["response_time"], "target": clip["target"],
            "caption": r["caption"], "horizon_label": r.get("horizon_label"),
        })
    print(f"[showcase] {len(showcase)}/{len(rows)} rows usable (missing video: {missing})")
    assert len(showcase) > 0, "no usable showcase clips - check Caption_a1fail321_V12.jsonl"
    return showcase


def main():
    metrics = compute_metrics()
    showcase = build_showcase()
    data = {
        "generated_from": "build_landing_data.py",
        "semantic_lambda": SEMANTIC_LAMBDA,
        "metrics": metrics,
        "showcase": showcase,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.LANDING_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"[wrote] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
