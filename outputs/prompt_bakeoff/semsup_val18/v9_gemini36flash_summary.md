# PROMPT_SEMSUP_V9_MINIMAL vs google/gemini-3.6-flash — 18-clip screen

**Question:** V9 (866 tokens) produced the best verdict metrics of any round tested on
qwen3-vl-235b-a22b-thinking (72.2% acc, recall 0.56, AUC 0.821), but that result was earned on a
reasoning-native "thinking" model performing its own internal chain-of-thought — the exact
condition V9's minimal-scaffolding design bet was built for. Does the same prompt, unchanged,
transfer to a Flash-class model, or was the result specific to Qwen's reasoning mode?

**Settings:** `google/gemini-3.6-flash` ($1.50/M in, $7.50/M out — 3.75× Qwen3-VL-235B's input
rate), native 1280×720, `detail="high"`, `temperature=0.1`, `max_tokens=20000`. 18/18 succeeded
first attempt, 0 failures. Wall 180s — roughly 4-10× faster than every Qwen3-VL round, consistent
with Flash having no internal reasoning pass to run before emitting output.

**Scorer disclosure:** hand-scores assigned by Claude, who wrote the prompt.

## Headline: verdict accuracy collapsed to 50%, and the reason is specific, not diffuse

| | Qwen3-VL-235B / V9 | **Gemini 3.6 Flash / V9** |
|---|---|---|
| Accuracy | 72.2% | **50.0%** |
| Recall | 0.56 | 0.22 |
| Precision | 0.83 | 0.50 |
| TP/FP/TN/FN | 5/1/8/4 | 2/2/7/7 |
| AUC | 0.821 | 0.778 |
| AP | 0.786 | 0.757 |

AUC (0.778) and AP (0.757) are close to Qwen's numbers — the model is still ranking clips in
roughly the right relative order. The accuracy collapse comes from **miscalibration at the
derived 50 threshold**, not from an unusable ranking (see the threshold sweep below). This is a
different failure mode from anything seen in the Qwen rounds, and it is diagnosable from the
captions themselves.

## The actual finding: Gemini states ego's reaction backwards, systematically

V9 mandates a closing clause naming what ego did — this was the single highest-value instruction
in the whole prompt lineage on Qwen3-VL. On Gemini 3.6 Flash it still fires reliably (ego-response
mentioned in 15/18 captions, matching Qwen's rate exactly) — **but the stated direction is wrong
far more often**:

| | reasoning CONTRADICT | of which: ego-reaction direction inverted |
|---|---|---|
| Qwen3-VL / V9 | 5/18 | not tracked separately |
| **Gemini Flash / V9** | **9/18 (50%)** | **7/18 (39%)** |

Concrete pairs, same clips, same prompt, different model:

| Clip | GT | Qwen3-VL / V9 | **Gemini Flash / V9** |
|---|---|---|---|
| `00493` | "the EGO vehicle **does not** slow down" | "ego not slowing" — correct | "ego **braking** in response" — **contradicts GT** |
| `01550` | ego closes gap "**in a controlled manner**" | "ego braking in response" — correct | "ego **not** slowing" — **contradicts GT** |
| `01504` | "the EGO vehicle... **also braked** in time" | (TN, correct) | "ego **not** slowing" — **contradicts GT**, caused this FP |

`01504` and `01550` are not incidental — they are the two clips whose entire GT narrative *is*
"ego reacted correctly, which is why nothing happened." Gemini reports the opposite reaction on
both, which is exactly what turned them into the run's two false positives. Seven of the nine
CONTRADICT clips follow this same pattern: the model names a change and an actor correctly, then
asserts ego's response in the wrong direction. This is a narrow, specific, and actionable finding
— not "Gemini is worse at this task" in general, but "Gemini's estimate of whether ego reacted is
unreliable in a way Qwen3-VL's was not," on this exact prompt.

## The ranking is usable even though the threshold is not

| Threshold | TP | FP | TN | FN | Recall | Precision | Acc |
|---|---|---|---|---|---|---|---|
| ≥12 | 9 | 6 | 3 | 0 | 1.00 | 0.60 | 66.7% |
| **≥14** | **8** | **3** | **6** | **1** | **0.89** | **0.73** | **77.8%** |
| ≥18 | 5 | 2 | 7 | 4 | 0.56 | 0.71 | 66.7% |
| ≥50 (prompt default) | 2 | 2 | 7 | 7 | 0.22 | 0.50 | 50.0% |

At threshold 14 (vs the prompt's built-in 50), accuracy is 77.8% — better than Qwen3-VL's default
cut. Six of nine real positives cluster in the 12-23 range, well below the 50-point cut the prompt
assumes; only two land above 80. Gemini's absolute score scale runs low relative to Qwen's for
this prompt, which is itself informative: **the same "risk_score 0-100" instruction produces a
different effective scale on a different model**, and any production use of V9 across model
families would need a per-model threshold, not a shared constant.

## What worked cleanly, and is worth keeping in mind before writing this off

- **`00283`**: the best-aligned reading of this clip across every round and every model tested to
  date (correctly identifies a pickup truck, not the SUV every Qwen round reported, entering from
  the right).
- **`01153`**: the fabricated crossing-sedan hallucination that has appeared in the Qwen3-VL V5,
  V6, V8 and V9 runs on this identical clip **does not reproduce here** — Gemini correctly reports
  the sedan staying in its own lane. This is evidence the hallucination is a Qwen3-VL-specific
  prior on this scene type, not a defect of the prompt or the task.
- `01737`, `02117`, `02104` are all read at least as well as, and in `02104`'s case better than,
  the equivalent Qwen3-VL captions (no "no changes observed" contradiction).

## Recommendation, and answer to the original question

**v6_balanced was never a live option for this evaluation** — its output schema
(`collision_verdict`/`scene_context`/`dynamic_objects`/`temporal_analysis`, no `caption_neutral`
or `risk_clause`) cannot feed the captioning pipeline `semsup_caption_qa.py` builds arms from, so
this run only ever tested V9. Verdict: **do not adopt Gemini 3.6 Flash with V9's default 50-point
threshold for anything relying on the verdict** — it is materially worse than Qwen3-VL on this
prompt (50.0% vs 72.2%), and the failure is systematic (ego-reaction direction), not random noise
that a bigger sample would average out.

If Gemini 3.6 Flash is worth pursuing at all — its cost is higher, not lower, than Qwen3-VL-235B
here ($1.50/M vs $0.40/M input), so the case would have to rest on latency or a different
strength — the fix is narrow: either recalibrate the threshold per-model (14 recovers most of the
gap), or add a single line to V9 asking the model to double-check the ego-reaction claim against
what the gap is actually doing (closing vs stopped-changing) before committing to it, since that
is the one specific place this model's judgment breaks.

## Files

- Per-clip data + reasoning-alignment verdicts: `reasoning_analysis_v9_gemini36flash_val18.xlsx`
- Raw output: `raw_v9_gemini36flash.jsonl`
- Prompt (unchanged from the Qwen3-VL run): `prompts/PROMPT_SEMSUP_V9_MINIMAL.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v9_gemini36flash_val18.py`
- Same-prompt comparison point: `v9_minimal_summary.md` (Qwen3-VL-235B / V9)
