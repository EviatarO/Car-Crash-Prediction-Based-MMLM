# PROMPT_SEMSUP_V4_QWEN vs qwen/qwen3-vl-235b-a22b-thinking — 18-clip screen

**Question:** does a prompt written in Qwen's own recommended structure (forced step-by-step
thinking, worked examples, explicit anti-under-calling instruction) fix the severe under-calling
found last round (Qwen3.7 Flash and GPT-5.6 Luna Pro both predicted collision on only 2/18
clips, recall 0.22)? Run against a different, reasoning-native Qwen model
(`qwen/qwen3-vl-235b-a22b-thinking`, $0.40/$4.00 per 1M, 131k context).

**Settings:** native 1280×720, `detail="high"`, `temperature=0.1`, `max_tokens=20000`. 18/18
succeeded on the first attempt, zero failures (unlike Qwen3.7 Flash last round, which needed two
retry passes at this same token budget).

**Scorer disclosure:** all scores assigned by Claude, who wrote `PROMPT_SEMSUP_V4_QWEN`. Full
rationale per clip in `reasoning_analysis_qwen3vl_val18.xlsx`.

## Headline: the under-calling was NOT fixed — and the pattern is now suspiciously exact

| Teacher / prompt | Verdict acc | Mean score | Recall | Precision | Predicted YES on |
|---|---|---|---|---|---|
| Gemini-today / v6 | 50.0% | 4.61 | 0.67 | 0.50 | 6 clips |
| Qwen3.7 Flash / v6 | 61.1% | 4.72 | **0.22** | 1.00 | 2 clips |
| GPT-5.6 Luna Pro / v6 | 61.1% | 4.83 | **0.22** | 1.00 | 2 clips |
| **Qwen3-VL-235B / V4 (this round)** | 61.1% | **5.11** | **0.22** | 1.00 | **2 clips** |

**Three different models, two different prompts (v6 unmodified, and V4 — which explicitly
instructs "do NOT default to NO... under-calling is as serious an error as a false alarm" plus
two worked examples calibrating the threshold) — identical confusion matrix: TP=2, FP=0, TN=9,
FN=7.** The explicit counter-instruction had no measurable effect on recall. This is now the
single most robust finding across three rounds of teacher testing: whatever is driving the
conservative bias is not fixable by prompt-level calibration language alone, at least not the
version tried here. Two candidate explanations, not yet distinguished:

1. **A property of these specific models** on visual risk-assessment — genuinely more
   conservative than Gemini regardless of instruction.
2. **A property of the task framing itself** for any model given v6-derived decision gates
   ("predict YES only if A, B, or C clearly hold") — the gates themselves may be the calibration
   anchor, and no amount of surrounding instruction moves it if the model treats the gates as
   the actual decision rule.

Distinguishing these needs a prompt that changes the *decision gates themselves* (e.g. lowering
the evidence bar, or removing the gate structure entirely), not just adding surrounding
instructions — not attempted this round.

## But caption quality is the best measured yet

| | Mean score | Green | Orange | Red |
|---|---|---|---|---|
| V2 | 4.61 | — | — | — |
| V3 | 4.78 | — | — | — |
| Qwen3.7 Flash / v6 | 4.72 | — | — | — |
| GPT-5.6 Luna Pro / v6 | 4.83 | — | — | — |
| **Qwen3-VL-235B / V4** | **5.11** | 8/18 | 5/18 | 5/18 |

Highest mean fidelity score of every teacher/prompt combination tested so far, despite recall
being unchanged. This matters because **the caption is the actual SigLIP training target** —
verdict is QA-only. A teacher that writes better captions while being verdict-conservative is
still a better *captioning* teacher, even if it needs a separate fix (or a different prompt) for
the verdict/QA signal specifically.

## `02117`: third model in a row to solve it correctly

`02117` (GT=NO: gray sedan at constant distance, van stopped before a crosswalk) broke every
single Gemini run (original v6, v6-rerun-today, V2, V3) — all four hallucinated an identical
"black SUV merges into ego lane" event. **Qwen3.7 Flash, GPT-5.6 Luna Pro, and now Qwen3-VL-235B
have all three independently resolved it correctly**, this time with the caption *"Sedan ahead in
ego lane maintaining consistent following distance through green traffic light with parallel
traffic flow"* — a clean, accurate read. Three-for-three across non-Gemini candidates on the one
clip that broke every Gemini attempt is strong, now well-replicated evidence of a genuine
Gemini-specific weakness on this scenario type, not a fluke.

## A genuinely good catch worth naming: `00687`

Caption: *"Gray SUV merging from right lane into ego lane while black sedan maintains position
ahead"* — this **correctly identifies GT's exact causal mechanism** (a gray SUV drifting into the
ego lane). But `risk_clause` says "normal merging traffic" and `verdict=NO`. The model perceived
the hazard accurately and then discounted it — direct evidence that the recall problem is a
**calibration/decision-threshold issue downstream of correct perception**, not a perception
failure. This is the clearest single-clip evidence for explanation #2 above (the gates or
surrounding calibration language, not the vision itself, are suppressing the verdict).

## Recommendation

**For captioning quality alone**: `qwen/qwen3-vl-235b-a22b-thinking` + `PROMPT_SEMSUP_V4_QWEN` is
the best-scoring combination found across all three rounds (5.11 vs the next-best 4.83), at
$0.40/$4.00 per 1M — cheaper than Gemini, pricier than Qwen3.7 Flash. Worth carrying forward for
the SigLIP-target caption use case specifically.

**For the verdict/QA signal**: do not use any of the three alternative teachers as-is for
anything relying on the collision verdict — all three currently under-call by the same amount,
and it survived an explicit corrective instruction. Two concrete next steps if this is worth
pursuing further, in order of information value per cost:
1. **Isolate perception vs. calibration directly**: ask for a bare risk *score* (0-100, no
   verdict, no gates) instead of a binary verdict, and threshold it after the fact at several
   cutoffs — `00687`'s finding suggests the raw signal may already discriminate better than the
   binary output shows.
2. Re-test with the decision gates removed or loosened (e.g. drop "clearly hold" language,
   lower the evidence bar) — the cheapest way to test explanation #2 above directly.

## Files

- Per-clip data + scores: `reasoning_analysis_qwen3vl_val18.xlsx`
- Raw captions: `raw_v4_qwen3vl235b.jsonl`
- New prompt: `prompts/PROMPT_SEMSUP_V4_QWEN.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_qwen3vl_val18.py`
- Prior rounds (referenced above): `summary.md` (Gemini reproducibility), `teacher_bakeoff_summary.md`
  (Qwen3.7 Flash / GPT-5.6 Luna Pro on unmodified v6)
