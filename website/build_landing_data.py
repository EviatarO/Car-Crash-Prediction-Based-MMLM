"""
build_landing_data.py
======================
Generates website/landing_data.js (window.LANDING_DATA = {...}) for the landing page:
  - showcase: all 321 V12-captioned A1-failure windows, each joined to its clip's
    train.xlsx metadata (time_of_event/time_of_alert/response_time/target) and its
    video/thumb paths - so the landing page's random showcase needs no separate
    video-resolution logic.
  - metrics: TP/FN/FP/TN/Prec/Rec/F1/Acc/AP/AUC at threshold 0.5 for every arm that
    has been scored on the SAME 677 test clips - A0, A1, B-v1/v2/v3, P1, and the four
    a1fail321 recovery arms (a1cont, V10, V12, v12shuf). Computed here, never
    hand-copied, so the landing page cannot drift from the actual score files.
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
SELECTION = MMLM_AI / "outputs" / "a1fail321" / "selection_a1fail321.jsonl"
E4 = MMLM_AI / "outputs" / "e4_vjepa_reason"

A1F = MMLM_AI / "outputs" / "a1fail321"
A1C = MMLM_AI / "outputs" / "a1_compress256"

# A1 deliberately reads its ORIGINAL evaluation (a1_1761, epoch 4 - the run the published
# 0.900 came from), not the a1fail321 re-score of the same weights. The two differ: the
# re-score flips one clip (fp 124 vs 123) and lands AP at 0.8995.

# Every arm that has been scored on the 677-clip test set, oldest first. `timeline_order`
# is carried as data so the page never has to hard-code an arm list to sort by.
#
# Two sources of truth, on purpose: where a run wrote a `test_summary.json` that file wins
# for AP/AUC, because the per-clip dumps store `score` rounded to 4 dp
# (semsup_train.py: `round(s, 4)`) and those ties perturb average precision. AUC and the
# confusion matrix are insensitive to the ties, which is why the two sources agree there
# and not on AP - and the f1/recall cross-check in compute_metrics() is what proves the
# summary describes the same checkpoint as the dump before its AP is used.
#
# The four a1fail321 arms and B-v1 have no summary file and use their dump throughout.
EXPERIMENTS = [
    dict(order=0, arm="A0", label="A0 · Frozen baseline",
         path=E4 / "StageA_scorer" / "badas_open_private.jsonl",
         gt_key="ground_truth", summary=None,
         source="e4_vjepa_reason/StageA_scorer/badas_open_private.jsonl"),
    dict(order=1, arm="A1", label="A1 · Crash-only (control)",
         path=E4 / "a1_1761" / "test_results_ep04.jsonl",
         gt_key="ground_truth", summary=E4 / "a1_1761" / "test_summary.json", epoch=4,
         source="e4_vjepa_reason/a1_1761 (epoch 4)"),
    dict(order=1.5, arm="A1-compress256", label="A1-compress256 · Full-frame preprocessing",
         path=A1C / "train" / "test_results_ep02.jsonl",
         gt_key="ground_truth", summary=A1C / "train" / "test_summary.json", epoch=2,
         source="a1_compress256/train (epoch 2)"),
    dict(order=2, arm="B-v1", label="B-v1 · Crash + semantic (V10 captions)",
         path=E4 / "b_1761_par" / "test_results_ep04.jsonl",
         gt_key="ground_truth", summary=None,
         source="e4_vjepa_reason/b_1761_par (epoch 4)"),
    dict(order=3, arm="B-v2", label="B-v2 · Same, neutral V12 captions",
         path=E4 / "b_v2_1761" / "test_results_ep02.jsonl",
         gt_key="ground_truth", summary=E4 / "b_v2_1761" / "test_summary.json", epoch=2,
         source="e4_vjepa_reason/b_v2_1761 (epoch 2)"),
    dict(order=4, arm="B-v3", label="B-v3 · Execution defects fixed",
         path=E4 / "b_v3_1761" / "test_results_ep10.jsonl",
         gt_key="ground_truth", summary=E4 / "b_v3_1761" / "test_summary.json", epoch=10,
         source="e4_vjepa_reason/b_v3_1761 (epoch 10)"),
    dict(order=5, arm="P1", label="P1 · Two-stage (semantic → crash)",
         path=E4 / "p1_stageB" / "test_results_ep02.jsonl",
         gt_key="ground_truth", summary=E4 / "p1_stageB" / "test_summary.json", epoch=2,
         source="e4_vjepa_reason/p1_stageB (epoch 2)"),
    dict(order=6, arm="a1cont", label="A1-cont · Recovery control (no captions)",
         path=A1F / "test_scores" / "a1cont_ep10.jsonl",
         gt_key="gt_verdict", summary=None,
         source="a1fail321/test_scores/a1cont_ep10.jsonl (epoch 10)"),
    dict(order=7, arm="V10", label="V10 · Recovery, GT captions",
         path=A1F / "test_scores" / "v10_ep10.jsonl",
         gt_key="gt_verdict", summary=None,
         source="a1fail321/test_scores/v10_ep10.jsonl (epoch 10)"),
    dict(order=8, arm="v12", label="V12 · Recovery, neutral captions",
         path=A1F / "test_scores" / "v12_ep10.jsonl",
         gt_key="gt_verdict", summary=None,
         source="a1fail321/test_scores/v12_ep10.jsonl (epoch 10)"),
    dict(order=9, arm="v12shuf", label="V12-shuffled · Content control",
         path=A1F / "test_scores" / "v12shuf_ep10.jsonl",
         gt_key="gt_verdict", summary=None,
         source="a1fail321/test_scores/v12shuf_ep10.jsonl (epoch 10)"),
]
SEMANTIC_LAMBDA = 0.2   # current --semantic-weight; a landing-page field, not baked into the SVG

# Expected values, verified against the source artifacts - asserted below so a future
# score-file change is caught loudly rather than silently propagating a stale table.
EXPECTED = {
    "A0":     dict(n=677, tp=308, fn=30, fp=130, tn=209),
    "A1":     dict(n=677, tp=320, fn=18, fp=123, tn=216),
    "A1-compress256": dict(n=677, tp=286, fn=52, fp=55, tn=284),
    "B-v1":   dict(n=677, tp=317, fn=21, fp=130, tn=209),
    "B-v2":   dict(n=677, tp=285, fn=53, fp=76, tn=263),
    "B-v3":   dict(n=677, tp=267, fn=71, fp=55, tn=284),
    "P1":     dict(n=677, tp=278, fn=60, fp=92, tn=247),
    "a1cont":  dict(n=677, tp=238, fn=100, fp=34, tn=305),
    "V10":     dict(n=677, tp=253, fn=85, fp=39, tn=300),
    "v12":     dict(n=677, tp=253, fn=85, fp=39, tn=300),
    "v12shuf": dict(n=677, tp=244, fn=94, fp=36, tn=303),
}
# Every arm must be pinned - an unpinned arm silently skips the drift assert, which is the
# only thing standing between a changed score file and a wrong number on the landing page.
assert {e["arm"] for e in EXPERIMENTS} <= set(EXPECTED),     f"unpinned arms: {sorted({e[chr(34)+chr(34)] if False else e['arm'] for e in EXPERIMENTS} - set(EXPECTED))}"


def to01(v):
    """Labels arrive as 0/1 ints ('ground_truth') or YES/NO strings ('gt_verdict')."""
    return int(v) if not isinstance(v, str) else (1 if v == "YES" else 0)


def compute_metrics():
    table, unpinned = [], []
    for e in EXPERIMENTS:
        arm, path = e["arm"], e["path"]
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        y = [to01(r[e["gt_key"]]) for r in rows]
        s = [float(r["score"]) for r in rows]
        m = metrics_from_arrays(y, s, threshold=0.5)

        got = dict(n=m["n_total"], tp=m["tp"], fn=m["fn"], fp=m["fp"], tn=m["tn"])
        exp = EXPECTED.get(arm)
        if exp is None:
            # A newly-scored arm has nothing to compare against yet. Print it in
            # paste-ready form rather than silently trusting it - the assert is the only
            # thing standing between a changed score file and a wrong number on the page.
            unpinned.append(f'    "{arm}":'.ljust(14) +
                            f'dict(n={got["n"]}, tp={got["tp"]}, fn={got["fn"]}, '
                            f'fp={got["fp"]}, tn={got["tn"]}),')
        else:
            assert got == exp, f"{arm}: metrics drifted from the expected/verified values.\n" \
                                f"  expected {exp}\n  got      {got}\n" \
                                f"  (source: {path})"

        ap, auc, note = m["ap"], m["auc_roc"], None
        if e["summary"]:
            # The run's own published report. Select the CHECKPOINT MATCHING the dump's
            # own epoch (project review §5.5) rather than trusting checkpoints[0] to be
            # rank-ordered to the right one - summaries happen to be sorted by val_ap
            # today, but nothing guarantees that stays true. Then cross-check the metrics
            # that ARE reproducible from the rounded dump before trusting its AP - if
            # f1/recall/specificity disagree, the two files describe different runs and
            # the override would be silently wrong. (specificity added alongside
            # build_experiments_data.py's gate - f1+recall alone can tie across two
            # epochs that differ only in how FP/TN split, per the project review.)
            checkpoints = json.load(open(e["summary"], encoding="utf-8"))["checkpoints"]
            best = next((c for c in checkpoints if c["epoch"] == e["epoch"]), None)
            assert best is not None, \
                f"{arm}: epoch {e['epoch']} not found in {e['summary'].name} " \
                f"(has epochs {[c['epoch'] for c in checkpoints]})"
            for key, mine in (("f1", m["f1"]), ("recall", m["recall_sensitivity_tpr"]),
                              ("specificity", m["specificity_tnr"])):
                assert abs(best[key] - mine) < 1e-3, \
                    f"{arm}: {e['summary'].name} epoch {e['epoch']} {key}={best[key]} " \
                    f"disagrees with the per-clip dump ({mine}) - the two files are not " \
                    f"the same run."
            ap, auc = round(float(best["test_ap"]), 4), round(float(best["auc_roc"]), 4)
            note = f"AP/AUC from test_summary.json epoch {e['epoch']} (dump reproduces " \
                   f"AUC exactly; its AP reads {m['ap']} because scores are stored " \
                   f"rounded to 4 dp)"

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
    if unpinned:
        print("\n[PIN ME] add these to EXPECTED, then re-run so the assert locks them:")
        for line in unpinned:
            print(line)
    return table


def horizon_of(row, sel_by_dir):
    """The sampling horizon (TTE_0.5 / MID-4 / ...). The three corpora disagree about
    which key holds it - the V12 caption file leaves `horizon_label` null and the
    selection file populates `requested_time_to_event` - so read both, then fall back to
    the selection manifest keyed by frames_dir. Same fallback chain as
    build_compare_data.py::bucket_of."""
    direct = row.get("requested_time_to_event") or row.get("horizon_label")
    if direct:
        return direct
    sel = sel_by_dir.get(row.get("frames_dir"), {})
    return sel.get("requested_time_to_event") or sel.get("horizon_label")


def build_showcase():
    train_by_id = {c["id"]: c for c in build_train()}
    sel_by_dir = {r["frames_dir"]: r for r in
                  (json.loads(l) for l in open(SELECTION, encoding="utf-8") if l.strip())}
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
            "caption": r["caption"], "horizon_label": horizon_of(r, sel_by_dir),
        })
    print(f"[showcase] {len(showcase)}/{len(rows)} rows usable (missing video: {missing})")
    assert len(showcase) > 0, "no usable showcase clips - check Caption_a1fail321_V12.jsonl"
    # The horizon was silently null for all 321 rows before the fallback above; assert so
    # a future corpus that drops the field fails loudly instead of blanking the field again.
    no_h = sum(1 for c in showcase if not c["horizon_label"])
    assert no_h == 0, f"{no_h}/{len(showcase)} showcase rows have no horizon - check the fallback"
    print(f"[showcase] horizon resolved for all {len(showcase)} rows")
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
