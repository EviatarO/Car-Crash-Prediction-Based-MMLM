# PROMPT_SEMSUP_V5_BALANCED vs qwen/qwen3-vl-235b-a22b-thinking — 18-clip screen

**Question:** three straight rounds (Gemini/v6-rerun, Qwen3.7 Flash/v6, GPT-5.6 Luna Pro/v6,
Qwen3-VL-235B/V4) converged on the *identical* confusion matrix — TP=2, FP=0, TN=9, FN=7,
recall 0.22 — regardless of model or of an explicit "under-calling is as serious as a false
alarm" instruction (V4). Does replacing the binary verdict with a continuous `risk_score`
(0-100, band-anchored), a mandatory pre-mortem (`counter_evidence`, forced before scoring), and
an explicit uncertainty band (40-59) move recall, without corrupting `caption_neutral` as a
SigLIP training target?

**Settings:** same model as the V4 round (`qwen/qwen3-vl-235b-a22b-thinking`), native 1280×720,
`detail="high"`, `temperature=0.1` (deliberately *not* raised — see prompt docstring), `max_tokens=20000`.
18/18 succeeded first attempt, 0 failures, 0 verdict/risk_score derivation mismatches (every
`verdict` correctly equalled `1 iff risk_score >= 50`).

**Scorer disclosure:** all hand-scores assigned by Claude, who wrote `PROMPT_SEMSUP_V5_BALANCED`.
Full rationale per clip in `reasoning_analysis_v5_val18.xlsx`.

## Headline: recall moved for the first time in four rounds — at the cost of one FP

| Teacher / prompt | Verdict acc | Recall | Precision | TP/FP/TN/FN |
|---|---|---|---|---|
| Gemini-today / v6 | 50.0% | 0.67 | 0.50 | 6/6/3/3 |
| Qwen3.7 Flash / v6 | 61.1% | 0.22 | 1.00 | 2/0/9/7 |
| GPT-5.6 Luna Pro / v6 | 61.1% | 0.22 | 1.00 | 2/0/9/7 |
| Qwen3-VL-235B / V4 | 61.1% | 0.22 | 1.00 | 2/0/9/7 |
| **Qwen3-VL-235B / V5 (this round)** | 61.1% | **0.33** | 0.75 | **3/1/8/6** |

Accuracy is unchanged (the +1 TP is offset by a new FP, `01153`), but this is the first prompt
of five tried on this screen that moved the confusion matrix off the exact 2/0/9/7 fixed point.
The pre-mortem + banded scale did something the semantic anti-under-calling instruction in V4
did not.

**The number that actually matters here is the score-based ranking, not the 50-cut accuracy**,
because V5's entire premise is that AP/AUC (this project's stated headline metric, see
`CLAUDE.md`) should be measured on the continuous `risk_score`, not on an arbitrary binary cut:

| Metric on `risk_score` (9 pos / 9 neg) | Value |
|---|---|
| AUC | 0.648 |
| AP | 0.677 |

Meaningfully better than chance (0.5), and — unlike every prior round — gives us a real lever:
the 50-point cut is a placeholder, and other thresholds trade the single FP against different
FN/TP combinations. None of the binary-only prompts (v6, V4) offered this option at all.

### Important caveat: the "continuous" score is actually 6 discrete values

Sorting all 18 clips by `risk_score` shows the model did not use the 0-100 range continuously —
it snapped to one representative value per anchor band, and **left two of the five bands
completely empty**:

```
78, 78, 75, 75   |   25, 25, 25   |   12, 12, 10, 10, 8, 8, 8, 8, 8   |   5
```

Only 6 distinct values appear across 18 clips (5, 8, 10, 12, 25, 75, 78). Zero clips landed in
the 40-59 "genuine uncertainty" band — the one purpose-built for exactly the ambiguous cases
driving the FNs — and zero landed in 85-100. Five clips (`00529`, `00493`, `01504`, `01643`,
`01552`) are tied at exactly `risk_score=8`, three GT=NO and two GT=YES — no threshold can
separate them, they are indistinguishable to the scorer. **The anchored-band instruction bought
structure at the cost of resolution**: giving the model 5 labelled bands with example values
taught it to pick a band and emit its round-number midpoint, not to place each clip on a genuine
continuum. The AUC/AP numbers above are real, but they are being computed on what is functionally
a 6-level ordinal variable, not a fine-grained score — worth fixing before trusting AP more than
`n=18` already limits it (this is a prompt-behavior finding, not a proposal to re-derive AP with
different tie-breaking).

### Threshold sweep (same run, no new API calls)

| Threshold | TP | FP | TN | FN | Recall | Precision | Acc |
|---|---|---|---|---|---|---|---|
| ≥8.5 | 7 | 5 | 4 | 2 | 0.78 | 0.58 | 0.61 |
| ≥12.5 | 4 | 4 | 5 | 5 | 0.44 | 0.50 | 0.50 |
| ≥25.5 | 3 | 1 | 8 | 6 | 0.33 | 0.75 | 0.61 |
| ≥50 (prompt default) | 3 | 1 | 8 | 6 | 0.33 | 0.75 | 0.61 |
| ≥76 | 2 | 0 | 9 | 7 | 0.22 | 1.00 | 0.61 |

Because of the value-clustering above, `25.5` and `50` are the same cut in practice (nothing
scored between 26 and 74), and `76` reproduces V4's exact 2/0/9/7 matrix. The only threshold that
meaningfully changes the picture is `≥8.5` — pulling in almost every non-floor clip — which
trades 5 new FPs for 4 new TPs (recall 0.33→0.78, precision 0.75→0.58). None of these operating
points dominates; which one is preferable depends on the FP/FN cost ratio for the eventual
downstream use, not something this 18-clip screen can settle alone.

## Caption quality: comparable to V4, not better, with one clip regressing

| | Mean hand score | Green | Orange | Red |
|---|---|---|---|---|
| Qwen3.7 Flash / v6 | 4.72 | — | — | — |
| GPT-5.6 Luna Pro / v6 | 4.83 | — | — | — |
| Qwen3-VL-235B / V4 | 5.11 | 8/18 | 5/18 | 5/18 |
| **Qwen3-VL-235B / V5** | 4.50 | 7/18 | 4/18 | 7/18 |

Slightly *lower* than V4, not higher. Two things are happening inside that number, and they
pull in opposite directions:

- **`00319` flipped from a clean loss to a lucky win**: V4 got both verdict and caption wrong on
  this clip (red, score 1). V5's verdict is now correct (risk_score=78) but the caption
  hallucinates a stationary "large truck ahead" and still completely misses GT's actual
  mechanism (a car entering the intersection from the right without slowing). Scored orange (2)
  here rather than green, specifically because a hallucinated caption is not a usable training
  target regardless of which side of the threshold it landed on — this is the clearest instance
  yet of verdict-correctness and caption-correctness decoupling.
- **`00687` regressed**: this was V4's best piece of evidence for the calibration-not-perception
  finding — the model *correctly* perceived the gray SUV drifting into ego's lane, then
  discounted it in the verdict. In V5, the caption itself now says the gray SUV is "parked on
  right side" — the drift is no longer perceived at all. This is a genuine perception miss, not
  a calibration one, and it is the reason `00687` counts as a straightforward FN this round
  instead of the interesting calibration-miss case it was in V4.

Net: V5 is not a strict caption-quality upgrade over V4. The recall gain did not come packaged
with better captions — it came from the scoring mechanism (band anchors + pre-mortem) doing real
work on the risk axis, somewhat independent of what the caption axis was doing.

## The one new false positive: `01153`

GT: ego makes a smooth, uncontested right turn at a green light; other traffic is stopped or
parallel; "no collision or accident is expected." V5's `counter_evidence` invents "left-turning
sedan crossing ego path with no visible braking or yielding" — a conflict that GT's reasoning
does not describe at all. This looks like the pre-mortem step doing exactly what it was designed
to do (actively search for a collision mechanism) on a clip where none exists, and finding one
anyway. That is the expected failure mode of a forced-search instruction: it will occasionally
manufacture a mechanism rather than report "none found." The prompt does have an explicit escape
hatch for this ("If after genuine search no such visible mechanism exists, say so explicitly")
and 8 of 9 true negatives correctly used it (`counter_evidence: "No visible mechanism..."`) — this
is the one clip where it didn't.

## `02117`: fourth model in a row to solve it correctly

Continues to resolve cleanly (caption: *"Car ahead in same lane maintaining consistent following
distance... green traffic light..."*), without the "black SUV merges into ego lane"
hallucination that broke every Gemini attempt. Now 4/4 non-Gemini teacher/prompt combinations
tested have solved this clip; 0/4 Gemini runs have.

## Recommendation

**Recall did move, but the effect is small at the prompt's own threshold (0.22 → 0.33), it did
not come with a caption-quality improvement, and the score turned out to be a 6-level ordinal
variable rather than a genuine continuum.** This is meaningfully different from V4's outcome
(identical confusion matrix to two prior teachers) but is not yet a result to build the 498-clip
production run on. Three threads worth pulling before committing to a teacher:

1. **Fix the score-clustering before trusting the ranking further.** The threshold sweep above
   is already done (free, no new API calls) and shows the only real alternative operating point
   is `≥8.5` (recall 0.78 / precision 0.58) — everything between is empty. That gap is itself the
   finding: get the model to actually spread scores within a band (e.g. require the score to
   differ from the nearest worked example's anchor value, or drop the named band boundaries and
   keep only qualitative anchors) before re-measuring AUC/AP as if it were a fine-grained signal.
2. **`00687`'s regression is worth understanding before scaling**: if the pre-mortem step
   trades off against baseline perception quality on some clips (gaining recall on `00319` at
   the cost of perception on `00687`), that trade needs to be characterized on a larger sample
   before this becomes the production prompt — 18 clips is too small to tell if it nets positive.
3. Not yet tried: loosening or removing the decision-gate language from v6-style prompts
   directly (the alternative explanation from the V4 round that predicted the gates themselves,
   not the surrounding instructions, are the calibration anchor) — V5 sidesteps this by removing
   gates entirely in favor of the anchored scale, which is itself indirect evidence for that
   explanation, but a direct A/B on the gates was still never run.

## Files

- Per-clip data + scores: `reasoning_analysis_v5_val18.xlsx`
- Raw captions: `raw_v5_qwen3vl235b.jsonl`
- New prompt: `prompts/PROMPT_SEMSUP_V5_BALANCED.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v5_val18.py`
- Prior rounds (referenced above): `summary.md` (Gemini reproducibility), `teacher_bakeoff_summary.md`
  (Qwen3.7 Flash / GPT-5.6 Luna Pro on unmodified v6), `qwen3vl_v4_summary.md` (V4)
