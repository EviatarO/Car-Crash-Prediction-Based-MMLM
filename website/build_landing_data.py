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

from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site_data import MMLM_AI, build_train  # noqa: E402

OUT = Path(__file__).resolve().parent / "landing_data.js"

CAPTIONS = MMLM_AI / "outputs" / "a1fail321" / "Caption_a1fail321_V12.jsonl"
SCORE_FILES = {
    "A0": (MMLM_AI / "outputs" / "e4_vjepa_reason" / "StageA_scorer" / "badas_open_private.jsonl",
           "video_id", "ground_truth", "score"),
    "A1": (MMLM_AI / "outputs" / "a1fail321" / "test_scores" / "A1.jsonl",
           "video_id", "gt_verdict", "score"),
    "v12": (MMLM_AI / "outputs" / "a1fail321" / "test_scores" / "v12_ep10.jsonl",
            "video_id", "gt_verdict", "score"),
}
LABELS = {"A0": "A0 · Baseline", "A1": "A1 · Crash-only (control)", "v12": "v12 · Semantic"}
SEMANTIC_LAMBDA = 0.2   # current --semantic-weight; a landing-page field, not baked into the SVG

# Expected values, verified this session (outputs/a1fail321/test_scores + StageA_scorer) -
# asserted below so a future score-file change is caught loudly rather than silently
# propagating a stale table to the page.
EXPECTED = {
    "A0":  dict(n=677, tp=308, fn=30, fp=130, tn=209),
    "A1":  dict(n=677, tp=320, fn=18, fp=124, tn=215),
    "v12": dict(n=677, tp=253, fn=85, fp=39, tn=300),
}


def load_labels(path, gt_key):
    """int(0/1) from either 'ground_truth' (already 0/1) or 'gt_verdict' (YES/NO)."""
    def _to01(v):
        if gt_key == "ground_truth":
            return int(v)
        return 1 if v == "YES" else 0
    return _to01


def compute_metrics():
    table = []
    for arm, (path, id_key, gt_key, score_key) in SCORE_FILES.items():
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        to01 = load_labels(path, gt_key)
        y = [to01(r[gt_key]) for r in rows]
        s = [float(r[score_key]) for r in rows]
        pred = [1 if v >= 0.5 else 0 for v in s]
        tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
        fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
        fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
        tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
        n = len(rows)

        exp = EXPECTED[arm]
        got = dict(n=n, tp=tp, fn=fn, fp=fp, tn=tn)
        assert got == exp, f"{arm}: metrics drifted from the expected/verified values.\n" \
                            f"  expected {exp}\n  got      {got}\n" \
                            f"  (source: {path})"

        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = (tp + tn) / n
        ap = average_precision_score(y, s)
        auc = roc_auc_score(y, s)

        table.append({
            "arm": arm, "label": LABELS[arm], "n": n,
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round(acc, 4), "ap": round(ap, 4), "auc": round(auc, 4),
        })
        print(f"[metrics] {arm:<4} n={n} TP={tp} FN={fn} FP={fp} TN={tn}  "
              f"Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} Acc={acc:.4f} "
              f"AP={ap:.4f} AUC={auc:.4f}")
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
