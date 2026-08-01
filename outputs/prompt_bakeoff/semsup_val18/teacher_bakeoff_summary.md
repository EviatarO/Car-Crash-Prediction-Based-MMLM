# Teacher model bake-off — Qwen3.7 Flash & GPT-5.6 Luna Pro vs current teacher

**Question:** does replacing the teacher model (currently `google/gemini-3.1-pro-preview`) with
`qwen/qwen3.7-flash` or `openai/gpt-5.6-luna-pro` improve performance, using the identical,
unmodified `PROMPT_G_OPT_v6_balanced` prompt on the same 18 GT validation clips? Baseline is
**Gemini rerun today** (not the historical 83.3% number — see `summary.md` in this folder for
why that baseline isn't reproducible), so this is a same-day, same-prompt, model-only swap.

**Settings:** native 1280×720, `detail="high"`, `temperature=0.1`, `max_tokens=20000` (raised
from the original 8192 after Qwen's more verbose responses triggered empty completions at that
budget — see the note on run reliability below).

**Scorer disclosure:** all 0-10 scores assigned by Claude. Rationale published per clip in
`reasoning_analysis_teacher_bakeoff.xlsx` for audit.

## Headline numbers

| Teacher | n | Mean score | Verdict accuracy | BERTScore | Recall | Precision |
|---|---|---|---|---|---|---|
| Gemini 3.1 Pro Preview (today, current) | 18 | 4.61 | 50.0% (9/18) | 0.864 | 0.67 | 0.50 |
| **Qwen3.7 Flash** | 18 | **4.72** | **61.1% (11/18)** | 0.870 | 0.22 | 1.00 |
| **GPT-5.6 Luna Pro** | 18 | **4.83** | **61.1% (11/18)** | 0.853 | 0.22 | 1.00 |

Both new teachers beat Gemini on verdict accuracy and mean score. **Read the next section before
concluding they're better** — the accuracy gain has a specific, important cause.

## The critical caveat: both new teachers are extremely conservative

Qwen predicted collision on **2 of 18 clips** (`00077`, `00687`). GPT-5.6 Luna Pro predicted
collision on **2 of 18 clips** (`00077`, `00283`) — a *different* pair, same count. The dataset
is 9 positive / 9 negative. Both models:

- Get **all 9 negatives right** (perfect specificity, 0 false positives) — precision 1.00.
- Catch only **2 of 9 positives** (recall 0.22) — same recall as Gemini's *worst* case.

**The 61.1% accuracy is arithmetic, not insight**: on a 50/50 dataset, a model that says "no
collision" almost every time scores 50% by construction (9/18, all the true negatives) plus
whatever it happens to catch on top. Both models added exactly 2 correct positives to that floor.
This is not evidence of better scene understanding in general — it's evidence of a **strong
prior toward "safe"**, which happens to score well here because the val set is balanced, but
would be a serious liability in deployment (a system that misses 78% of real collisions is not
usable regardless of how it scores on this metric). Compare to Gemini-today's recall of 0.67 —
noisier (6 false positives), but it actually attempts to flag danger far more often.

**Do not read this table as "switch to Qwen/GPT-5.6, done."** It shows these models are more
conservative under v6's decision-gate framing, not that they perceive collision risk better.

## What IS a genuine positive finding: `02117`

Every prior run — original v6, Gemini-today, V2, V3 — hallucinated the identical wrong scene on
`02117` (GT=NO: gray sedan at constant distance, van *stopped* before a crosswalk; all four
independently invented "a black SUV merges/crosses into the ego lane, collision imminent").

**Both Qwen and GPT-5.6 Luna Pro get this one right** — and not by accident: both explicitly
perceive the black SUV/van's motion near the lane, but correctly judge it as a benign merge or
diverging motion rather than escalating to a collision call. This is the first time in this
entire investigation any model has resolved this specific clip correctly. Genuinely suggests
better spatial/motion disambiguation on this category of near-crosswalk scenario, independent of
the general conservative-bias finding above (it's a *recall-preserving* correct NO — the models
had a real chance to hallucinate danger here, the way every other run did, and didn't).

## Run reliability note (operational, not scientific)

Qwen3.7 Flash returned empty (unparseable) responses on 4/18 clips at `max_tokens=8192`; raising
to 20000 fixed some but introduced empty responses on 2 *different* clips, and one retry round
was needed to get a clean 18/18. This pattern (different clips failing on different attempts,
not the same ones) points to provider-side flakiness (OpenRouter load-balances a model slug
across backend hosts) rather than a fixable token-budget issue. `semsup_v6_control_rerun.py` now
supports `--resume` to retry only missing clips without re-running successes. **If Qwen is
adopted for production-scale captioning, budget for retry passes** — this isn't a one-off.
GPT-5.6 Luna Pro had zero failures at `max_tokens=20000` in a single pass.

## Cost, for context

| Teacher | Input / output $ per 1M tokens | Relative to Gemini |
|---|---|---|
| Gemini 3.1 Pro Preview | $2.00 / $12.00 | 1x |
| GPT-5.6 Luna Pro | $0.50 / $3.00 | ~4-6x cheaper |
| Qwen3.7 Flash | $0.03 / $0.13 | ~65-90x cheaper |

## Recommendation

**Not enough evidence to switch teachers on this screen alone**, and specifically: do not
interpret "61.1% > 50.0%" as "better teacher" without carrying forward the recall caveat. What
this bake-off actually established:

1. Both candidates are viable and meaningfully cheaper, with mean fidelity scores at least as
   good as Gemini-today's.
2. Both show a real, non-obvious strength on the `02117`-style scenario (correctly avoiding a
   danger hallucination that fooled every Gemini run).
3. Both show a real, concerning weakness: severe under-calling of actual collisions (recall
   0.22 vs Gemini's 0.67) under this exact prompt.

**Next step, if pursuing this further**: don't decide from verdict accuracy on a small balanced
set — it rewards conservatism. Either (a) re-run on a recall-stress set (mostly true positives,
so a "say NO always" strategy can't hide behind specificity), or (b) look at whether v6's
decision gates (STEP 7's "predict YES ONLY if...") are being applied unusually strictly by these
two models specifically — that's a prompt-calibration question, not necessarily a model-quality
one, and might be fixable by loosening the gate language for these specific teachers rather than
switching away from them.

## Files

- Per-clip data + scores (3 teachers x GT): `reasoning_analysis_teacher_bakeoff.xlsx`
- Raw reasoning: `raw_v6_control_rerun.jsonl` (Gemini, reused), `raw_v6_qwen37flash.jsonl`,
  `raw_v6_gpt56lunapro.jsonl`
- Runner (generalized, any OpenRouter model): `student_training/scripts/semsup_v6_control_rerun.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_teacher_bakeoff.py`
