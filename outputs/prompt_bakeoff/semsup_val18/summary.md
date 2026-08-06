# PROMPT_SEMSUP_V2 / V3_COT vs v6 — 18-clip hallucination screen

> **Superseded by the GT-informed round (2026-08-02):** this thread's whole premise (predict
> the verdict, score prediction accuracy) turned out to be the wrong axis for a captioning
> target whose GT label already comes from `train.csv`. See
> `../semsup_val18_gt/summary.md` — giving the teacher the GT label and asking it to explain
> the mechanism eliminated positive-clip contradictions. That round's winner —
> **Gemini 3.6 Flash + `PROMPT_SEMSUP_V10_GT`, hybrid mode (GT on positives, blind on
> negatives)** — is what ships to the 587 train4500 failure windows, not anything below.

> **Correction notice:** an earlier version of this document concluded V2/V3 underperform v6.
> That conclusion was **confounded** and is retracted below — see "The control experiment."
> Re-running v6's own original prompt today, at identical settings, no longer reproduces its
> historical result either. The corrected, fair conclusion follows.

**Settings:** native 1280×720, `detail="high"`, `temperature=0.1`, `google/gemini-3.1-pro-preview`
— matched exactly to the original `v6_hires_full18.py` run (verified: identical `_encode_image`
logic, identical model slug/temperature, and one real difference found and fixed —
`max_tokens=8192`, which the original run set explicitly and this repo's newer scripts had
dropped; the control rerun below restores it). Data: `dataset/manifests/val_e3a.jsonl` (18 clips
with human-written `gt_reasoning_en`). Full per-clip data:
`reasoning_analysis_semsup_val18.xlsx`.

**Scorer disclosure:** Claude wrote both `PROMPT_SEMSUP_V2` and `PROMPT_SEMSUP_V3_COT` and
assigned every 0-10 score below, including the v6-rerun scores. This is not an independent
evaluation. Every score has a published rationale in the xlsx so it can be audited.

## The control experiment (why the first version of this analysis was wrong)

The initial comparison (V2/V3 vs the historical v6 result: 83.3% accuracy, mean score 6.78)
made an unstated assumption: that re-running the same model behind the same OpenRouter alias
today gives the same result it gave when v6 was originally validated. **That assumption was
never tested — until asked to.** `semsup_v6_control_rerun.py` re-ran `PROMPT_G_OPT_v6_balanced`
verbatim, today, at the exact original settings. Result:

| Run | n | Mean score | Median | Verdict accuracy |
|---|---|---|---|---|
| v6 **original** (2026-07, historical, after debate) | 18 | 6.78 | 8 | **83.3%** (15/18) |
| v6 **rerun today** (same prompt, same settings, control) | 18 | 4.61 | 7 | **50.0%** (9/18) |

**Re-running v6's own unmodified prompt today does not reproduce its own historical result.**
The drop (83.3% → 50.0%, mean 6.78 → 4.61) is essentially the entire gap the first version of
this analysis attributed to "V2/V3 are worse than v6." It isn't the prompt — it's something
about *today's environment* relative to whenever v6 was validated. The most likely explanation
is **model drift on the `google/gemini-3.1-pro-preview` alias**: OpenRouter aliases can resolve
to an updated underlying snapshot without the string changing, and preview-tagged models in
particular are not guaranteed stable over time. Sampling noise at `temperature=0.1` (non-zero)
could contribute a little, but not a 33-point swing on the same 18 clips with the same prompt.

**This is a real, separate finding worth acting on**: any result tied to a specific
`preview`-tagged OpenRouter model slug should be treated as a snapshot-in-time, not a
reproducible baseline, unless the run records the model's actual resolved version (not just the
alias) — OpenRouter's response `usage`/model metadata may expose this; worth checking before the
next teacher-distillation run of any kind.

## The corrected, fair comparison: same day, same environment

With v6 re-baselined today, all three arms are directly comparable:

| Arm | n | Mean score | Median | Verdict accuracy | BERTScore |
|---|---|---|---|---|---|
| v6 rerun today (control) | 18 | 4.61 | 7 | 50.0% (9/18) | 0.864 |
| **V2** (direct caption) | 18 | 4.61 | 5 | 50.0% (9/18) | 0.864 |
| **V3** (CoT-then-distill) | 18 | 4.78 | 6 | 50.0% (9/18) | 0.865 |

**All three are statistically indistinguishable.** Verdict accuracy is identical (9/18 for all
three arms) and mean fidelity scores are within 0.17 of each other. **The original concern —
that V2/V3 might be materially worse than v6 — is not supported once the comparison is made
fairly.** V3 has a slight, consistent edge (higher mean, higher median, better recall — see
below) but n=18 cannot distinguish a 0.17-point gap from noise.

### Confusion matrices

| Arm | TP | FP | TN | FN | Recall | Precision |
|---|---|---|---|---|---|---|
| v6 rerun today | 6 | 6 | 3 | 3 | 0.67 | 0.50 |
| V2 | 2 | 2 | 7 | 7 | 0.22 | 0.50 |
| V3 | 5 | 5 | 4 | 4 | 0.56 | 0.50 |

All three land at precision 0.50 — of every "collision" call, half are right. V2 is markedly
more conservative (recall 0.22) than either v6-today (0.67) or V3 (0.56); this specific
asymmetry (not the overall accuracy) is the one place V2 looks meaningfully different from the
other two, and it's the same under-calling pattern flagged below (V2's `risk_clause` frequently
implies danger while `verdict=0`).

## Hallucination audit: convergent failures across ALL FOUR runs (original v6, v6-today, V2, V3)

**`02117` (GT=NO)** — a gray sedan holds constant distance; a black van is *stopped* before a
crosswalk; no conflict. **All four independent runs** (original v6, v6-rerun-today, V2, V3)
hallucinate the same wrong object doing the same wrong thing: *"a black SUV... merges/executes a
lane change into the ego vehicle's lane... rapidly closing."* Four runs, two different prompt
authors, at least one likely different underlying model snapshot, converging on one specific
wrong reading — this is very strong evidence of a **genuinely ambiguous or mislabeled clip**,
not a prompt or model issue. Worth inspecting the raw 16 frames directly.

**`00474` (GT=YES)** — a white van's left turn into the ego lane is missed by **all four runs**
identically (v6: "stable parallel trajectories"; v6-today: "no sudden maneuvers"; V2: "van
stopped"; V3: "van moving parallel"). Same conclusion: a model-level blind spot on this specific
late-emerging maneuver, present regardless of which prompt or which day.

**New this round — `00529`, `01281`, `01504`, `01552`, `02104`**: v6-rerun-today introduces
**new** hallucinations not present in the original v6 run (which got several of these right).
E.g. `00529`: original v6 correctly identified "SUV forced to merge right due to construction
scaffolding" (score 10); today's rerun says "stable, parallel trajectories" (score 1) — a direct
regression on a clip the same prompt used to get right. This is further evidence the *model*
changed, not just noise: these are wholesale scene misreadings, not borderline verdict flips.

## What's actually different between V2 and V3

V3 (mean 4.78, recall 0.56) modestly beats V2 (mean 4.61, recall 0.22) — the gap that's real is
**recall**, not overall accuracy (both tie v6-today at 50%). V3 catches real positives V2 misses:
`00319` (car crossing into path, V3=9 vs V2=1), `00283` (V3=9 vs V2=7), `00687` (V3=8 vs V2=6,
correct verdict). The CoT scaffold appears to help the model commit to what it noticed rather
than defaulting to a generic safe-traffic description — consistent with the original hypothesis,
just a smaller effect than the first (confounded) analysis suggested.

**V2's specific, fixable defect**: on `00687`, `01281`, `02104`, V2's own `risk_clause` states
"moderate/high risk" while `verdict=0` — internally inconsistent on 3 of 18 clips. V3 has zero
such contradictions, plausibly because its decision gates (STEP 7) are resolved before the
caption is written, not independently of it.

## Answers to your three questions (revised)

### 1. What should change in the prompt?

Unchanged from before, since these are about caption *content*, not the confound:
- Add an explicit "state the causal event, not just current position" instruction — the
  recurring gap (`00147`, `00493`'s EGO-merge context, `00529`) is describing *where* agents are
  while omitting *what one is about to do*.
- Keep verdict-before-caption ordering (V3's structure) — plausibly why V3 has zero
  verdict/risk_clause contradictions where V2 has three.
- Do not raise the 40-word cap (hard SigLIP constraint, unrelated to this finding).

### 2. Should we adopt the debate/recovery flow?

**Still no**, and this finding makes it more relevant, not less: recovery reasoning quality
*collapsed* relative to first-pass v6 in the original data (mean 3.6 vs 6.28) even when it fixed
verdicts. If the underlying model is now less reliable than when v6 was validated, layering a
second unreliable pass on top compounds the problem rather than fixing it — the recovery
prompt's design assumed a stable, self-consistent first-pass model to push back against.

### 3. Should we adopt v6's CoT-then-distill structure?

**Yes, but the evidence for it is now modest rather than dramatic.** V3 still beats V2 on every
axis (mean score, recall, zero self-contradictions), at identical cost. The corrected margin is
smaller than first reported, but it's the right default given it's free.

## Recommendation

**The 498-clip production run should not be gated on this screen's absolute numbers** — 50%
verdict accuracy looked alarming against a stale 83% baseline; against a fair same-day baseline,
it's the same starting point v6 itself gets today. **Adopt V3 as the production prompt** (small
consistent edge, zero cost difference), apply the Q1 causal-event instruction, and **before
spending the 498-clip budget, resolve the model-drift question**: check whether OpenRouter
exposes the actually-resolved model version for `google/gemini-3.1-pro-preview`, and consider
whether a versioned (non-`preview`) model slug would give a more stable target to develop the
prompt against.

## Files

- Per-clip data + scores (4 arms: v6 original, v6 rerun, V2, V3): `reasoning_analysis_semsup_val18.xlsx`
- Raw captions: `raw_v2_native.jsonl`, `raw_v3_cot.jsonl`, `raw_v6_control_rerun.jsonl` (18 rows each)
- v6 original reference (reused, not re-run): `../reasoning_analysis_v6_debate.xlsx`, `../leaderboard_v6_debate.md`
- New prompt: `prompts/PROMPT_SEMSUP_V3_COT.py`
- Analysis scripts: `teacher_distillation/scripts/reasoning_analysis_semsup_val18.py`,
  `student_training/scripts/semsup_v6_control_rerun.py`
