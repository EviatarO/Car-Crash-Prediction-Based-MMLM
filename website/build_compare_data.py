"""
build_compare_data.py
=====================
Generates website/compare_data.js (window.COMPARE_DATA) - the per-clip tables behind the
Experiments page's Cross-Experiment Comparison view.

Three comparable datasets, because "the same dataset" is not one thing here:

  test677     the 677-clip held-out test set. Shared by every arm that was ever scored on
              it (all but V10), so this is the only place all families meet.
  pool1761    the 1,761-window training pool. Shared by A0/A1/B-v1/B-v2/B-v3/P1.
  a1fail321   the A1-failure recovery pool - ALL 321 windows A1 gets wrong. Coverage is
              deliberately uneven and the `split` column is what makes it readable: the
              six pool-1761 arms were scored on the whole pool so they cover all 321,
              while the four recovery arms only ever dumped their VALIDATION split, so
              they carry scores on 61 rows and blanks on the other 260. Those 260 are
              the recovery arms' own training windows, so even if they were scored the
              numbers would be train-set numbers - filter split=val for the honest
              arm-vs-arm read.

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
A1C = MMLM_AI / "outputs" / "a1_compress256"
CAPS = MMLM_AI / "outputs" / "semantic_captions"
TEST_MANIFEST = MMLM_AI / "dataset" / "manifests" / "test_manifest_hires.jsonl"

THRESHOLD = 0.5
GROUP_LABEL = {0: "tte_0.5s", 1: "tte_1.0s", 2: "tte_1.5s"}

# Where each arm's per-clip scores on the 677-clip test set live. All ten arms are now
# present: the four a1fail321 recovery arms were scored 2026-09-05, which closed the V10
# gap and added the two controls (a1cont = same weights with no caption term, v12shuf =
# same captions scrambled within class) that make the content-vs-presence question
# answerable on held-out data rather than only on the 61-window recovery val split.
TEST_SCORES = {
    "A0":   (E4 / "StageA_scorer" / "badas_open_private.jsonl", "ground_truth"),
    "A1":   (E4 / "a1_1761" / "test_results_ep04.jsonl", "ground_truth"),
    "A1-compress256": (A1C / "train" / "test_results_ep02.jsonl", "ground_truth"),
    "B-v1": (E4 / "b_1761_par" / "test_results_ep04.jsonl", "ground_truth"),
    "B-v2": (E4 / "b_v2_1761" / "test_results_ep02.jsonl", "ground_truth"),
    "B-v3": (E4 / "b_v3_1761" / "test_results_ep10.jsonl", "ground_truth"),
    "P1":   (E4 / "p1_stageB" / "test_results_ep02.jsonl", "ground_truth"),
    "V12":  (A1F / "test_scores" / "v12_ep10.jsonl", "gt_verdict"),
    "a1cont":  (A1F / "test_scores" / "a1cont_ep10.jsonl", "gt_verdict"),
    "V10":     (A1F / "test_scores" / "v10_ep10.jsonl", "gt_verdict"),
    "v12shuf": (A1F / "test_scores" / "v12shuf_ep10.jsonl", "gt_verdict"),
}
POOL1761_ARMS = ["A0", "A1", "A1-compress256", "B-v1", "B-v2", "B-v3", "P1"]
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
        pos = r["event_occurs"] == 1
        out.append({
            "key": vid,          # one window per clip here, so video_id is unique
            "video_id": vid,
            # `group` stratifies BOTH classes into the same three horizons - test.xlsx
            # assigns 0/1/2 to negatives exactly as it does to positives - so the
            # horizon is a real property of how the set was sampled, not a claim about
            # an event, and it is shown on every row.
            "window": GROUP_LABEL[r["group"]],
            "group": r["group"],
            "gt": "YES" if pos else "NO",
            "scores": scores.get(vid, {}),
        })
    missing = [r for r in out if len(r["scores"]) != len(arms)]
    assert not missing, f"{len(missing)} test clips missing a score from some arm"
    return {
        "key": "test677", "name": "Test set · 677 held-out clips",
        "clip_split": "test",          # which window.SITE_DATA.splits bucket the video is in
        "arms": arms, "rows": out,
        "columns": ["video_id", "window", "gt"],
        "note": "Held-out Nexar private test set - every arm was scored on exactly these "
                "677 clips. The recovery family is complete here: a1cont (same weights, "
                "no caption term), V10 and V12 (real captions), and v12shuf (the same "
                "captions permuted within class). v12 vs v12shuf isolates caption "
                "CONTENT; a1cont is the floor with no captions at all.",
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
        "columns": ["video_id", "window", "split",
                    "caption_V10", "caption_V12", "gt"],
        "note": "The pool A1 and every B/P1 arm trained on. Both splits are shown — filter "
                "on split=val for the honest held-out view, since train rows were seen "
                "during fitting.",
    }


# ------------------------------------------------------------------------ a1fail321
def build_a1fail321():
    """All 321 A1-failure windows.

    Two score sources with different coverage, joined on frames_dir:
      - the six pool-1761 arms were scored across the whole 1,761-window pool, and the
        321 are a subset of it, so they cover every row here;
      - the four recovery arms (a1cont / v10 / v12 / v12shuf) only ever wrote
        `val_scores_ep10.jsonl`, i.e. their held-out split - 61 of the 321.

    The gap is left as blanks rather than hidden by trimming the table to 61 rows: the
    260 missing rows are those arms' own TRAINING windows, so scoring them later would
    produce train-set numbers, not a fair comparison. The downstream sections already
    skip a row for any arm that has no score there, so an arm-vs-arm read automatically
    falls back to the 61 they share.
    """
    sel = load_jsonl(A1F / "selection_a1fail321.jsonl")
    v10cap = {r["frames_dir"]: r for r in load_jsonl(A1F / "Caption_a1fail321_V10.jsonl")}
    v12cap = {r["frames_dir"]: r for r in load_jsonl(A1F / "Caption_a1fail321_V12.jsonl")}

    # TWO different splits partition these same 321 windows, and confusing them is the
    # easiest way to misread this table:
    #   `split`      - the a1fail321 video-level split. Held-out for the RECOVERY arms.
    #   `pool_split` - the pool-1761 split. Held-out for A0..P1.
    # They disagree badly: only 15 of the 61 a1fail-val rows are also pool-val, so
    # filtering split=val does NOT give a held-out set for B-v3 and friends. Both are
    # exposed as columns so the genuinely-held-out-for-everything subset is reachable.
    pool_val = clip_level_split([r["video_id"] for r in
                                 load_jsonl(CAPS / "Caption_V12_Neutral_1761_fortrain.jsonl")])

    # pool arms - full coverage of all 321
    scores = {}
    for arm in POOL1761_ARMS:
        for r in load_jsonl(E4 / "pool1761_scores" / f"{arm}.jsonl"):
            scores.setdefault(r["frames_dir"], {})[arm] = round(float(r["score"]), 4)

    # recovery arms - validation split only. FAILS LOUDLY on a missing arm (project
    # review §5.4): this pool is where v12 vs v12shuf - the content-vs-presence
    # control the thesis's headline claim rests on - is decided. A silent `continue`
    # here would render the comparison page missing its control with no indication
    # anything was wrong.
    rec_arms = []
    missing_arms = []
    for arm in A1FAIL_ARMS:
        p = A1F / "results" / arm / "fold_01" / "val_scores_ep10.jsonl"
        if not p.exists():
            missing_arms.append((arm, p))
            continue
        for r in load_jsonl(p):
            scores.setdefault(r["frames_dir"], {})[arm] = round(float(r["score"]), 4)
        rec_arms.append(arm)
    assert not missing_arms, (
        "a1fail321: missing val_scores_ep10.jsonl for recovery arm(s) "
        f"{[a for a, _ in missing_arms]} - the page would silently render without "
        f"the control (v12shuf) or another recovery arm. Paths checked: "
        f"{[str(p) for _, p in missing_arms]}")

    out = []
    for r in sel:
        fd = r["frames_dir"]
        out.append({
            "key": fd,
            "video_id": r["video_id"],
            "window": bucket_of(r),
            "split": r.get("split"),
            "pool_split": "val" if r["video_id"] in pool_val else "train",
            "mined_failure": True,       # every window in this pool is an A1 failure
            "caption_V10": v10cap.get(fd, {}).get("caption"),
            "caption_V12": v12cap.get(fd, {}).get("caption"),
            "gt": r["gt_verdict"],
            "scores": scores.get(fd, {}),
        })
    assert len(out) == 321, f"expected all 321 A1-failure windows, got {len(out)}"
    # COVERAGE first, then correctness (project review §5.3): a total frames_dir join
    # failure makes every r["scores"] == {} for A1, so "A1 has zero WRONG rows among
    # rows it scored" would trivially and silently pass with A1 scored on nothing at
    # all. Assert A1 actually joined onto all 321 rows before trusting the
    # by-construction correctness check below.
    a1_covered = sum(1 for r in out if "A1" in r["scores"])
    assert a1_covered == 321, (
        f"A1 joined onto only {a1_covered}/321 a1fail321 windows - frames_dir key "
        f"drift between selection_a1fail321.jsonl and pool1761_scores/A1.jsonl. "
        f"(The A1-should-be-wrong-on-all-321 check below would pass vacuously "
        f"if this ran with a broken join, since an uncovered row scores as neither "
        f"right nor wrong.)")
    # A1 is wrong on every row by construction - the cheapest possible check that the
    # join is right, and it would break loudly if the selection ever drifted.
    a1_right = sum(1 for r in out
                   if "A1" in r["scores"] and (r["scores"]["A1"] >= THRESHOLD) == (r["gt"] == "YES"))
    assert a1_right == 0, f"A1 should be wrong on all 321 by construction, but is right on {a1_right}"
    n_val = sum(1 for r in out if r["split"] == "val")
    assert (len(out) - n_val, n_val) == (260, 61), \
        f"a1fail321 split drifted from 260/61: got {len(out)-n_val}/{n_val}"

    return {
        "key": "a1fail321", "name": "A1-failure pool · all 321 windows",
        "clip_split": "train",
        "arms": POOL1761_ARMS + rec_arms,
        "arm_labels": A1FAIL_LABEL,
        "rows": out,
        "columns": ["video_id", "window", "split", "pool_split",
                    "caption_V10", "caption_V12", "gt"],
        "note": "Every window here is one A1 gets wrong, so A1 scores 0 correct by "
                "construction - the floor any recovery arm has to beat. The six pool-1761 "
                "arms cover all 321 rows; the four recovery arms show scores on their 61 "
                "held-out windows only, because training never dumped scores for the 260 "
                "windows they were fitted on. NOTE the two split columns are different "
                "partitions of the same 321: `split` is held-out for the recovery arms, "
                "`pool split` is held-out for A0..P1, and only 15 rows are val in both. "
                "Set split=val AND pool split=val for the subset no arm was trained on.",
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
