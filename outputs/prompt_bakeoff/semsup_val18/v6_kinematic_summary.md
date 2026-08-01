# PROMPT_SEMSUP_V6_KINEMATIC vs qwen/qwen3-vl-235b-a22b-thinking — 18-clip screen

**Question:** V5's per-clip data showed the 6 remaining false negatives were not a threshold
problem (the low-score region was 5 YES/5 NO, exactly chance) but a specific perceptual blind
spot — all 6 FN captions used the word "maintaining" on clips whose GT mechanism was a lateral
drift or an ego-motion event. V6 adds three mandatory observation fields (`ego_motion`,
`lateral_watch`, `final_delta`) targeting exactly that axis, and decomposes the score into four
independent 0-25 sub-scores to fix V5's band-clustering. Does it work?

**Settings:** same model as V4/V5 (`qwen/qwen3-vl-235b-a22b-thinking`), native 1280×720,
`detail="high"`, `temperature=0.1`, `max_tokens=20000`. 18/18 succeeded first attempt, 0
failures, 0 verdict/risk_score derivation mismatches, 0 risk_score/sub-score-sum mismatches.

**Scorer disclosure:** all hand-scores assigned by Claude, who wrote `PROMPT_SEMSUP_V6_KINEMATIC`.
Full rationale per clip in `reasoning_analysis_v6_val18.xlsx`.

## Headline: recall improved again, but net accuracy went DOWN — this is not a clean win

| Teacher / prompt | Verdict acc | Recall | Precision | TP/FP/TN/FN | Mean hand-score |
|---|---|---|---|---|---|
| Qwen3-VL-235B / V4 | 61.1% | 0.22 | 1.00 | 2/0/9/7 | 5.11 |
| Qwen3-VL-235B / V5 | 61.1% | 0.33 | 0.75 | 3/1/8/6 | 4.50 |
| **Qwen3-VL-235B / V6 (this round)** | **55.6%** | **0.44** | 0.57 | **4/3/6/5** | **4.33** |

Recall kept climbing (0.22 → 0.33 → 0.44), continuing the trend from V5. But accuracy and mean
caption-fidelity both **fell** relative to V5, not just relative to V4. This round traded away
more than it gained: 2 real FNs flipped to TPs (`00147`, `00372`), but 2 clips that had been
clean true negatives in **every single prior round** (`01281`, `01504`) flipped to false
positives, on top of the FP already present from V5 (`01153`, which V6 did not fix).

## Root cause: the lateral-focus instruction is being satisfied by template completion, not observation

This is the actual finding of the round, and it's a specific, diagnosable failure, not vague
"still biased." Look at what `lateral_watch` produced on the three false positives:

| Clip | GT (verbatim) | `lateral_watch` output |
|---|---|---|
| `01153` | "All vehicles remain in their respective lanes, so no collision... is expected" | *"White sedan left side turning across intersection into ego's path"* |
| `01281` | vehicles "maintain reasonable distances from each other" (a braking-pickup, purely longitudinal scenario) | *"Black SUV right lane drifting rightward toward container truck"* |
| `01504` | "the EGO vehicle noticed [the braking] and also braked in time" (again purely longitudinal) | *"Dark SUV on right turning across ego's path from side street"* — there is no side street in GT at all |

None of these events exist in the ground truth. All three are confabulated, and all three follow
the **identical template**: *"[vehicle] on [side] turning/crossing [direction] into ego's path...
ego holding lane centre... no braking or steering input."* Two clips with completely different
real content — `01504` (dark SUV, side street, night) and `00147` (white sedan, green light,
correct GT match) — produced **byte-for-byte identical sub-score vectors** (`5, 22, 20, 21`) and
near-identical sentence structure. That is not two independent visual analyses arriving at the
same numbers by coincidence; it is the model completing a learned narrative template that the
prompt primed too strongly, not deriving the observation from the specific 16 frames in front of
it.

**Likely direct cause, and it's on this prompt, not the model:** two of V6's three worked
examples are lateral-drift-positive cases (Examples 1 and 2), and only one (Example 3) is a
clean non-event — and Example 3 is purely longitudinal (braking handled), not "scanned for
lateral movement and found none." There is no worked example showing the specific output pattern
"searched for lateral movement per the STEP 2 instruction, found none, and correctly said so
without inventing anything." The few-shot prior is unbalanced in exactly the direction that
produces this failure mode.

**The `counter_evidence` citation guard did not catch this, and structurally cannot.** V6 requires
`counter_evidence` to cite a detail from `ego_motion`/`lateral_watch`/`final_delta` specifically
to prevent the pre-mortem from inventing an unsupported mechanism (this fixed V5's version of the
problem, where the pre-mortem invented things directly). But if the fabrication happens one step
earlier — in `lateral_watch` itself — then `counter_evidence` citing it is not a check, it's the
model citing its own hallucination back to itself. The guard verifies internal consistency, not
grounding in the actual frames, and V6 has no field that verifies *that*.

## What actually worked

Not a wash — the targeted fix did fire correctly on real lateral-drift clips:

- **`00687` — best result recorded on this clip across every round tested.** V4 saw the drift but
  miscalibrated the verdict (NO); V5 regressed the perception entirely (called the SUV "parked");
  V6 correctly identifies *"Gray SUV turning left across the lane line into ego's path"* — GT's
  exact mechanism — and gets the verdict right. Clean green, score 8.
- `00147` and `00372` both flip from FN to TP. `00147`'s caption still has an inverted causality
  (a recurring issue on this clip across multiple rounds) and `00372`'s caption names the wrong
  mechanism entirely (verdict right, caption wrong — scored orange, not green, for exactly that
  reason) — so these are partial wins, not clean ones.
- `00493`'s caption improved concretely versus V5 (now reports the lead sedan braking; V5 said
  only "maintaining following distance") even though the verdict is still wrong.
- `01552` and `01643` reproduce the exact same minor fabrications (a "school bus", "parked cars")
  seen in the V4 and V5 rounds on these identical clips — now a stable 3-for-3 pattern, worth
  treating as a persistent artifact of this model/scene pair rather than noise.
- `02117` solved cleanly for the fifth consecutive non-Gemini teacher/prompt combination.

## Score clustering: improved but not solved

11 distinct `risk_score` values across 18 clips (up from V5's 6), and the 40-59 uncertainty band
is finally populated (`01281`, `50.0`). But `00147` and `01504` — one real positive, one
fabricated positive — landed on the exact same score via identical sub-score vectors, which is
the clustering problem re-appearing as a symptom of the template-completion issue above rather
than a genuinely separate problem.

## Recommendation

**Do not carry V6 forward as-is for the 498-clip production run — on this screen it is a step
sideways, not forward, on the metric this project treats as headline (`CLAUDE.md`: trust
AP/AUC).** `risk_score` AUC (0.698) is actually the best of the three rounds, but that number is
propped up by score separation between fabricated-high and genuine-low clips, not by real
discrimination — it would be a mistake to read the AUC alone as validating this prompt.

The fix path is narrow and specific, which is the useful part of this result:

1. **Rebalance the worked examples.** Add a fourth example (or replace Example 3) that explicitly
   walks through "STEP 2 lateral scan performed, genuinely found nothing" on a busy multi-agent
   scene — not just an empty one — so the model has a template for confidently reporting a null
   result under visual complexity, not just under simplicity.
2. **A grounding constraint on `lateral_watch` itself**, not just on `counter_evidence`: e.g.
   require it to name a specific frame range and a specific visual cue (brake light, wheel angle,
   lane-line pixel position) rather than a free-text claim. Free text with no evidence pointer is
   exactly what let 3 clips converge on the same unsupported sentence.
3. Worth checking directly: rerun the 3 new-FP clips (`01281`, `01504`, `01153`) at a different
   sampling seed / temperature to see whether the fabrication is stable or a low-probability
   sample that happened to land three times — `n=1` per clip on a thinking model doesn't
   distinguish "this is what the model does on this input" from "this is what it did once."

## Files

- Per-clip data + scores: `reasoning_analysis_v6_val18.xlsx`
- Raw captions: `raw_v6_qwen3vl235b.jsonl`
- New prompt: `prompts/PROMPT_SEMSUP_V6_KINEMATIC.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v6_val18.py`
- Prior rounds: `summary.md` (Gemini reproducibility), `teacher_bakeoff_summary.md` (Qwen3.7
  Flash / GPT-5.6 Luna Pro on unmodified v6), `qwen3vl_v4_summary.md` (V4),
  `v5_balanced_summary.md` (V5)
