"""
build_compare_data.py
=====================
Generates website/compare_data.js (window.COMPARE_DATA) - the per-clip tables behind the
Experiments page's Cross-Experiment Comparison view.

Three comparable datasets, because "the same dataset" is not one thing here:

  test677     the 677-clip held-out test set. Shared by every arm that was ever scored on
              it (all but V10), so this is the only place all families meet.
  pool1761    the 1,761-window training pool. Shared by A0/A1/B-v1/B-v2/B-v3/P1.
  a1fail321   the A1-failure recovery pool, shared by V10/V12 and their crash-only control.
              Restricted to the 61 VAL windows: that is the only subset where every arm
              has a score, and mixing in train-split rows would make the success counts
              incomparable across arms with different denominators.

The two training pools share no windows, so an arm from one cannot be compared against an
arm from the other on training data - the page enforces that rather than silently joining
on video_id and producing a meaningless table.

Column layout mirrors build_pool1761_comparison.py's `per_clip` sheet
(video_id, window, split, mined_failure, caption_V10, caption_V12, gt, then one column per
arm), so the page and the workbook can be read side by side.

    python website/build_compare_data.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_site_data import MMLM_AI  # noqa: E402

# clip_level_split is imported, not re-implemented: the train/val column must match what
# training actually did, and that module already replicates semsup_common's partition.
sys.path.insert(0, str(MMLM_AI / "student_training" / "scripts"))
from build_pool1761_comparison import clip_level_split  # noqa: E402

OUT = Path(__file__).resolve().parent / "compare_data.js"
E4 = MMLM_AI / "outputs" / "e4_vjepa_reason"
A1F = MMLM_AI / "outputs" / "a1fail321"
CAPS = MMLM_AI / "outputs" / "semantic_captions"
TEST_MANIFEST = MMLM_AI / "dataset" / "manifests" / "test_manifest_hires.jsonl"

THRESHOLD = 0.5
GROUP_LABEL = {0: "tte_0.5s", 1: "tte_1.0s", 2: "tte_1.5s"}

# Where each arm's per-clip scores on the 677-clip test set live. V10 is absent on
# purpose - it was never scored there, and an estimate would be worse than a gap.
TEST_SCORES = {
    "A0":   (E4 / "StageA_scorer" / "badas_open_private.jsonl", "ground_truth"),
    "A1":   (E4 / "a1_1761" / "test_results_ep04.jsonl", "ground_truth"),
    "B-v1": (E4 / "b_1761_par" / "test_results_ep04.jsonl", "ground_truth"),
    "B-v2": (E4 / "b_v2_1761" / "test_results_ep02.jsonl", "ground_truth"),
    "B-v3": (E4 / "b_v3_1761" / "test_results_ep10.jsonl", "ground_truth"),
    "P1":   (E4 / "p1_stageB" / "test_results_ep02.jsonl", "ground_truth"),
    "V12":  (A1F / "test_scores" / "v12_ep10.jsonl", "gt_verdict"),
}
POOL1761_ARMS = ["A0", "A1", "B-v1", "B-v2", "B-v3", "P1"]
# a1cont is the crash-only control started from the identical A1 weights - it is what V10
# and V12 must be read against, so it is offered here even though it is not one of the
# eight headline arms.
A1FAIL_ARMS = ["a1cont", "v10", "v12", "v12shuf"]
A1FAIL_LABEL = {"a1cont": "A1-cont (control)", "v10": "V10", "v12": "V12",
                "v12shuf": "V12-shuffled (control)"}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def to01(v):
    return int(v) if not isinstance(v, str) else (1 if v == "YES" else 0)


def bucket_of(row):
    """V12's corpus populates requested_time_to_event and leaves horizon_label null;
    V10's does the reverse. Read both rather than trusting one file's convention."""
    return row.get("requested_time_to_event") or row.get("horizon_label")


# --------------------------------------------------------------------------- test677
def build_test():
    manifest = load_jsonl(TEST_MANIFEST)
    by_vid = {r["video_id"]: r for r in manifest}

    scores, arms = {}, []
    for arm, (path, gt_key) in TEST_SCORES.items():
        rows = load_jsonl(path)
        assert len(rows) == 677, f"{arm}: {len(rows)} rows, expected 677"
        for r in rows:
            scores.setdefault(r["video_id"], {})[arm] = round(float(r["score"]), 4)
        arms.append(arm)

    out = []
    for r in manifest:
        vid = r["video_id"]
        out.append({
            "key": vid,          # one window per clip here, so video_id is unique
            "video_id": vid,
            "window": GROUP_LABEL[r["group"]],
            "gt": "YES" if r["event_occurs"] == 1 else "NO",
            "scores": scores.get(vid, {}),
        })
    missing = [r for r in out if len(r["scores"]) != len(arms)]
    assert not missing, f"{len(missing)} test clips missing a score from some arm"
    return {
        "key": "test677", "name": "Test set · 677 held-out clips",
        "clip_split": "test",          # which window.SITE_DATA.splits bucket the video is in
        "arms": arms, "rows": out,
        "columns": ["video_id", "window", "gt"],
        "note": "Held-out Nexar private test set. Every arm here was scored on exactly "
                "these 677 clips. V10 is absent: it was never scored on the test set.",
    }


# ------------------------------------------------------------------------- pool1761
def build_pool1761():
    v10 = {r["frames_dir"]: r for r in load_jsonl(CAPS / "Caption_Train4500_Mixed_1761.jsonl")}
    v12 = {r["frames_dir"]: r for r in
           load_jsonl(CAPS / "Caption_V12_Neutral_1761_fortrain.jsonl")}
    failures = {r["frames_dir"] for r in
                load_jsonl(CAPS / "Caption_Train4500_Failures_587.jsonl")}
    assert set(v10) == set(v12), "V10/V12 corpora cover different windows"
    assert len(v12) == 1761, f"expected 1761 windows, got {len(v12)}"

    val_vids = clip_level_split([r["video_id"] for r in v12.values()])

    scores = {}
    for arm in POOL1761_ARMS:
        rows = load_jsonl(E4 / "pool1761_scores" / f"{arm}.jsonl")
        assert len(rows) == 1761, f"{arm}: {len(rows)} rows, expected 1761"
        for r in rows:
            scores.setdefault(r["frames_dir"], {})[arm] = round(float(r["score"]), 4)

    out = []
    for fd, c12 in v12.items():
        out.append({
            # a clip contributes up to 3 windows, so video_id is NOT unique here -
            # frames_dir is, and it is what per-row UI state (notes) must key on.
            "key": fd,
            "video_id": c12["video_id"],
            "window": bucket_of(c12),
            "split": "val" if c12["video_id"] in val_vids else "train",
            "mined_failure": fd in failures,
            "caption_V10": v10[fd]["caption"],
            "caption_V12": c12["caption"],
            "gt": c12["gt_verdict"],
            "scores": scores[fd],
        })
    n_val = sum(1 for r in out if r["split"] == "val")
    assert (len(out) - n_val, n_val) == (1413, 348), \
        f"split drifted from training's 1413/348: got {len(out)-n_val}/{n_val}"
    return {
        "key": "pool1761", "name": "Training pool · 1,761 windows",
        "clip_split": "train",
        "arms": POOL1761_ARMS, "rows": out,
        "columns": ["video_id", "window", "split", "mined_failure",
                    "caption_V10", "caption_V12", "gt"],
        "note": "The pool A1 and every B/P1 arm trained on. Both splits are shown — filter "
                "on split=val for the honest held-out view, since train rows were seen "
                "during fitting.",
    }


# ------------------------------------------------------------------------ a1fail321
def build_a1fail321():
    sel = {r["frames_dir"]: r for r in load_jsonl(A1F / "selection_a1fail321.jsonl")}
    v10 = {r["frames_dir"]: r for r in load_jsonl(A1F / "Caption_a1fail321_V10.jsonl")}
    v12 = {r["frames_dir"]: r for r in load_jsonl(A1F / "Caption_a1fail321_V12.jsonl")}

    # Restricted to the val split: it is the only subset every arm has a score for, and
    # equal denominators are what make the success-count bars comparable.
    scores, arms = {}, []
    for arm in A1FAIL_ARMS:
        p = A1F / "results" / arm / "fold_01" / "val_scores_ep10.jsonl"
        if not p.exists():
            continue
        for r in load_jsonl(p):
            scores.setdefault(r["frames_dir"], {})[arm] = round(float(r["score"]), 4)
        arms.append(arm)

    out = []
    for fd, r in sel.items():
        if fd not in scores:
            continue                     # train-split window: no per-arm score was dumped
        row_scores = dict(scores[fd])
        # A1's own score on every window of this pool is recorded in the selection file
        # itself (it is how the pool was mined), so the baseline needs no extra source.
        row_scores["A1"] = round(float(r["a1_score"]), 4)
        out.append({
            "key": fd,
            "video_id": r["video_id"],
            "window": bucket_of(r),
            "split": r.get("split"),
            "mined_failure": True,       # every window in this pool is an A1 failure
            "caption_V10": v10.get(fd, {}).get("caption"),
            "caption_V12": v12.get(fd, {}).get("caption"),
            "gt": r["gt_verdict"],
            "scores": row_scores,
        })
    assert out, "no a1fail321 rows with scores - check val_scores_ep10.jsonl"
    return {
        "key": "a1fail321", "name": "A1-failure pool · 61 val windows",
        "clip_split": "train",
        "arms": ["A1"] + arms,
        "arm_labels": {**A1FAIL_LABEL, "A1": "A1 (mining baseline)"},
        "rows": out,
        "columns": ["video_id", "window", "split", "mined_failure",
                    "caption_V10", "caption_V12", "gt"],
        "note": "Every window here is one A1 gets wrong, so A1 scores 0 correct by "
                "construction. Restricted to the 61 held-out val windows — the only "
                "subset where all arms have a score, and the only one where the counts "
                "share a denominator.",
    }


def main():
    datasets = [build_test(), build_pool1761(), build_a1fail321()]
    for d in datasets:
        keys = [r["key"] for r in d["rows"]]
        assert len(keys) == len(set(keys)),             f"{d['key']}: row keys are not unique - per-row notes would collide"
    for d in datasets:
        gt = Counter(r["gt"] for r in d["rows"])
        buckets = Counter(r["window"] for r in d["rows"])
        print(f"[dataset] {d['key']:<10} rows={len(d['rows']):<5} arms={len(d['arms'])} "
              f"YES={gt['YES']} NO={gt['NO']}  buckets={dict(buckets)}")

    data = {
        "generated_from": "build_compare_data.py",
        "threshold": THRESHOLD,
        "datasets": {d["key"]: d for d in datasets},
        "order": [d["key"] for d in datasets],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("window.COMPARE_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"[wrote] {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
