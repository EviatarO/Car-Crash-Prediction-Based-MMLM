"""build_semsup_captions_587.py -- converts the 587 raw V10-hybrid captions
into the caption-schema JSONL semsup_common.load_training_examples() expects,
for the A1_587 / B_587 training run.

WHY THIS CONVERSION IS NEEDED (not just a reformat)
------------------------------------------------------
load_training_examples() reads: video_id, requested_time_to_event, gt_verdict
("YES"/"NO"), caption (a single string), and optionally an explicit
frames_dir (bypasses the default 267-key index lookup, which this data isn't
in - see that function's docstring). The raw captioner output
(raw_v10_gemini_gt_pos.jsonl / raw_v10_gemini_blind_neg.jsonl) already
carries frames_dir/event_occurs/t_seconds/horizon_label directly (written by
semsup_caption_promptbakeoff.py's out_row construction), but still needs
converting:
  - no single `caption` field - only caption_neutral + risk_clause separately
  - no reliable `gt_verdict` - GT-mode rows carry the (correct) `gt_label`,
    but BLIND-mode rows only carry the model's own predicted `verdict`,
    which is exactly the field these 318 windows are selected for being
    wrong about (they're A0's false positives). Using the model's predicted
    verdict as the training label would silently corrupt every row - this
    script uses each row's own `event_occurs` field instead.
  - `requested_time_to_event` is present but always null on both files (a
    separate, lower-priority bug - the runner writes `row.get(
    "requested_time_to_event")` for the out_row's own metadata field, but
    the train4500 manifest's field is actually named `time_before_event_s`,
    so that particular lookup always misses). Harmless for training -
    load_training_examples() only uses it for the (unused-by-the-trainer)
    informational `tte` field once frames_dir is already resolved - so it is
    carried through as-is (null) here rather than blocking on it.

CAPTION FIELD CHOICE: `caption` = caption_neutral ALONE (not
caption_neutral + risk_clause). risk_clause is explicitly the one field
where evaluative/outcome language is allowed ("high risk of collision
likely") - concatenating it into the semantic-loss target risks making the
caption label-collinear (arm C from the original bake-off plan: the
Predictor could learn to recover the crash label from language cues instead
of scene content, which would make B indistinguishable from a redundant
copy of the crash loss). caption_neutral alone also matches the existing
267-row Caption_Train_All_Clips.jsonl's convention (a single descriptive
sentence, no risk language) for direct comparability. Pass --with-risk-clause
to build the alternate arm if you want to test that variant later.

Usage:
    python build_semsup_captions_587.py
    python build_semsup_captions_587.py --with-risk-clause   # caption_neutral + risk_clause
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

FAILURES_DIR = PROJECT_ROOT / "outputs" / "semantic_captions" / "failures587"
POS_JSONL = FAILURES_DIR / "raw_v10_gemini_gt_pos.jsonl"
NEG_JSONL = FAILURES_DIR / "raw_v10_gemini_blind_neg.jsonl"
# The 2026-08-05 balanced-mixing round (see the A1_587-overfitting finding in
# outputs/semantic_captions/failures587/summary.md): 587 pos + 587 neg sampled
# from the A0-CORRECT pool (not just failures), same V10-hybrid captioning
# recipe, to stop training exclusively on adversarially-hard windows.
POS_JSONL_BALANCED = FAILURES_DIR / "raw_v10_gemini_gt_pos_balanced.jsonl"
NEG_JSONL_BALANCED = FAILURES_DIR / "raw_v10_gemini_blind_neg_balanced.jsonl"
FRAMES_ROOT = PROJECT_ROOT / "dataset" / "train"
OUT_PATH = PROJECT_ROOT / "outputs" / "semantic_captions" / "Caption_Train4500_Failures_587.jsonl"

REQUIRED_SOURCE_KEYS = ("video_id", "frames_dir", "event_occurs", "caption_neutral", "risk_clause")


def _load_jsonl(p: Path) -> list:
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def convert(caption_rows: list, with_risk_clause: bool) -> list:
    out = []
    bad = []
    for r in caption_rows:
        missing = [k for k in REQUIRED_SOURCE_KEYS if r.get(k) is None]
        if missing:
            bad.append((r.get("video_id"), missing))
            continue
        caption = r["caption_neutral"]
        if with_risk_clause:
            caption = f"{caption}, {r['risk_clause']}"
        out.append({
            "video_id": r["video_id"],
            "t_seconds": r.get("t_seconds"),
            "requested_time_to_event": r.get("requested_time_to_event"),
            "horizon_label": r.get("horizon_label"),
            "gt_verdict": "YES" if int(r["event_occurs"]) == 1 else "NO",
            "caption": caption,
            "frames_dir": r["frames_dir"],
        })
    if bad:
        raise RuntimeError(
            f"{len(bad)}/{len(caption_rows)} caption rows are missing a required source "
            f"field ({REQUIRED_SOURCE_KEYS}) - refusing to silently shrink the training "
            f"pool. First few: {bad[:5]}"
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-risk-clause", action="store_true",
                     help="caption = caption_neutral + ', ' + risk_clause instead of "
                          "caption_neutral alone (see module docstring for the collinearity "
                          "risk this defaults away from)")
    ap.add_argument("--pos", nargs="+", default=[str(POS_JSONL)],
                     help="one or more raw GT-mode (positive) caption JSONLs to merge")
    ap.add_argument("--neg", nargs="+", default=[str(NEG_JSONL)],
                     help="one or more raw blind-mode (negative) caption JSONLs to merge")
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()

    pos_rows = [r for p in args.pos for r in _load_jsonl(Path(p))]
    neg_rows = [r for p in args.neg for r in _load_jsonl(Path(p))]

    dup_pos = len(pos_rows) - len({(r["video_id"], r.get("frames_dir")) for r in pos_rows})
    dup_neg = len(neg_rows) - len({(r["video_id"], r.get("frames_dir")) for r in neg_rows})
    if dup_pos or dup_neg:
        raise RuntimeError(
            f"duplicate (video_id, frames_dir) rows across --pos/--neg inputs "
            f"({dup_pos} pos dupes, {dup_neg} neg dupes) - merging overlapping files would "
            f"double-count those windows. Check for accidental overlap between input files."
        )

    pos_out = convert(pos_rows, args.with_risk_clause)
    neg_out = convert(neg_rows, args.with_risk_clause)
    all_out = pos_out + neg_out

    if len(all_out) != len(pos_rows) + len(neg_rows):
        raise RuntimeError("row count mismatch after conversion - this should be unreachable "
                            "given convert() raises on any unresolved row")

    # Verify every frames_dir resolves to 16 real frames on disk BEFORE writing -
    # load_training_examples() would silently skip a bad row (require_frames=True
    # default), which would quietly shrink the pool exactly like the join failure above.
    missing = []
    for r in all_out:
        frame_dir = FRAMES_ROOT / r["frames_dir"]
        paths = [frame_dir / f"frame_{i:05d}.jpg" for i in range(1, 17)]
        if not all(p.exists() for p in paths):
            missing.append(r["video_id"])
    if missing:
        raise RuntimeError(f"{len(missing)} rows have missing frames on disk: {missing[:10]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for r in all_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pos = sum(1 for r in all_out if r["gt_verdict"] == "YES")
    n_neg = sum(1 for r in all_out if r["gt_verdict"] == "NO")
    print(f"Wrote {args.out}")
    print(f"  {len(all_out)} rows ({n_pos} YES / {n_neg} NO), "
          f"caption={'caption_neutral+risk_clause' if args.with_risk_clause else 'caption_neutral only'}")
    print(f"  all {len(all_out)} rows verified to resolve to 16 real frames on disk")


if __name__ == "__main__":
    main()
