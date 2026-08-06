# train4500 failure-window captioning (587 windows) — DONE (2026-08-04)

**Goal:** caption the 587 windows where the frozen A0 scorer disagrees with the true label
(269 FN + 318 FP, from `outputs/train4500_inference/`), as the caption pool for a paired
`A1_587` vs `B_587` semantic-supervision comparison — see
`outputs/prompt_bakeoff/semsup_val18_gt/summary.md` for the bake-off that chose this config.

**Config:** Gemini 3.6 Flash, native 1280×720, `detail="high"`, `temperature=0.1`, hybrid mode —
`--gt-mode gt` on the 269 positives (`PROMPT_SEMSUP_V10_GT`), `--gt-mode blind` on the 318
negatives. Manifest reconstructed from `outputs/train4500_inference/monitor_train4500_coverage.xlsx`
joined to `dataset/manifests/train4500_hires.jsonl` (587/587 matched, 0 unmatched) via
`student_training/scripts/build_failure_manifest.py`.

## Result: 587/587 captioned, verified complete

- Positives: 269/269, 0 failures.
- Negatives: 318/318, 0 failures (1 empty-response row retried and recovered).
- **Integrity check**: captured `frames_dir` set == manifest `frames_dir` set for both halves,
  exactly — 0 missing, 0 duplicates, 0 extra.

## Two resume-key bugs found and fixed during this run (same bug class, twice)

1. **In the captioning runner**: resume/skip logic keyed on `video_id` alone. 76/269 positive
   and 70/318 negative `video_id`s repeat (a video can fail A0 at more than one TTE/offset
   bucket) — would have silently skipped ~146/587 windows as "already captioned" after the
   first bucket for each video landed. Fixed by keying on `frames_dir` (unique per window).
2. **In `build_failures587_review_xlsx.py`'s file-merge step**: same mistake, independently —
   deduped input rows by `video_id` when merging JSONL files, silently collapsing 587 rows to
   396 the first time the review sheet was built. Fixed the same way (dedup key → `frames_dir`).

## Quality signals (all GT-free — no `gt_reasoning_en` exists for these 587 windows)

| Signal | Result |
|---|---|
| `mechanism_visible=false` on positives | 36/269 (13.4%) — model's own admission it found no mechanism even with GT given |
| `double_failure` on negatives (teacher's blind verdict also disagrees with label) | 9/318 (2.8%) |
| Duplicate `caption_neutral` across all 587 | **0** — no template collapse |
| Internal contradiction (`hazard_agent` vs its own `caption_neutral`) | 2/587 |
| Rare vehicle type (≤2 occurrences in 587) | 5/587 (taxi, motorcycle, tractor — plausible) |
| Empty/invalid `evidence_frames` | 9/587 |
| Banned outcome-word in `caption_neutral` | 2/587 (both "safe following distance" — minor, not label-collinear) |
| Caption length | mean 21.8 words, max 36 (cap 40) |

**Decision (2026-08-04): skip building a manual-review queue from these signals.** Zero
duplicates is the reassuring signal — that was the failure mode that would have collapsed the
semantic branch (identical captions give SigLIP nothing to separate). The remaining flags
(~55 rows total) are weak/proxy signals, not confirmed hallucinations, and the real open
question (does the semantic loss help at all) is downstream in training, not here. The
`fabrication_check()` scorer built for the 18-clip val set **cannot run on this data** — both
its rules require `gt_reasoning_en`, which doesn't exist for these windows; a GT-free variant
would be needed to revisit this later.

## Cost

Estimated ~$27 pre-run (live OpenRouter pricing check); not re-verified against actual billing
post-run.

## Files

- `raw_v10_gemini_gt_pos.jsonl` (269), `raw_v10_gemini_blind_neg.jsonl` (318)
- `review_failures587.xlsx` — full review sheet, `double_failure` column, blank `manual_review`
  column for hand annotation
- `check10_gt_pos.jsonl` / `check10_blind_neg.jsonl` / `review_check10.xlsx` — the 10-clip
  stratified sanity check that preceded the full run (superseded, kept as a record)
- Manifests: `dataset/manifests/train4500_failures_pos_269.jsonl`,
  `train4500_failures_neg_318.jsonl`, `train4500_failures_587.jsonl`
- Builder: `student_training/scripts/build_failure_manifest.py`
- Review sheet builder: `teacher_distillation/scripts/build_failures587_review_xlsx.py`

## Next step

`A1_587` vs `B_587`, paired, on this exact 587-window caption pool. **Not started** — needs a
RunPod GPU (the previous pod was terminated after the train4500-inference run). Per project
convention, this needs explicit go-ahead before spinning up compute.

`semsup_train.py`'s training pool is defined by whichever caption file it's pointed at
(`load_training_examples(captions_path=...)`), so **A1's existing test AP=0.8638 is an n=267
number and is not a valid control for B_587** — A1 must be re-run on this identical 587-window
pool with `--semantic-weight 0.0` before comparing to B_587's `--semantic-weight >0`.
