# PROMPT_SEMSUP_V8_NARRATIVE vs qwen/qwen3-vl-235b-a22b-thinking — 18-clip screen

**Question:** a caption-level review of V5/V6/V7 (not a metric, a manual read of all 18×3
captions against GT) found every prior prompt produces a furniture inventory ("vehicle X ahead,
maintaining Y") where GT is always a causal narrative (what changed, why, how ego responded).
Measured: static vocabulary 12-17/18, change vocabulary 1-2/18, ego-reaction mentioned 2-3/18
across V5/V6/V7. V8 changes the caption's required GRAMMAR — `delta` / `true_movers` / `cause` /
`ego_response` fields, path-relative motion vocabulary with bare left/right banned for motion,
mandatory ego-response closing clause with explicit perceptual cues. Does the caption's reasoning
now align with GT?

**This round's focus, per instruction: reasoning alignment against GT, not verdict accuracy.**
Each clip is judged MATCH / PARTIAL / MISS / CONTRADICT against what `gt_reasoning_en` actually
asserts, independent of whether the binary verdict landed on the right side.

**Settings:** same model as V4-V7, native 1280×720, `detail="high"`, `temperature=0.1`,
`max_tokens=20000`. 18/18 first attempt, 0 failures, 0 derivation mismatches. Wall 715s.

**Scorer disclosure:** hand-scores assigned by Claude, who wrote the prompt.

## The vocabulary fix worked exactly as designed

| | static words | change words | ego-response mentioned | bare left/right for motion |
|---|---|---|---|---|
| V5 | 12/18 | 3/18 | 3/18 | 2/18 |
| V6 | 17/18 | 3/18 | 2/18 | 5/18 |
| V7 | 16/18 | 3/18 | 3/18 | 11/18 |
| **V8** | 12/18 | **8/18** | **6/18** | **0/18** |

Change-vocabulary usage roughly tripled, ego-response mentions doubled, and the direction-word ban
worked completely — not one caption uses "moving left/right" or "-ward" for motion, down from 11/18
in V7. Mechanically, V8's structural instructions fire.

## But reasoning alignment did not follow the vocabulary, and verdict accuracy fell

| Reasoning alignment (headline) | n |
|---|---|
| MATCH — substantively correct | 5/18 |
| PARTIAL — right shape, wrong detail | 6/18 |
| MISS — misses GT's mechanism | 3/18 |
| CONTRADICT — states the opposite of GT | **4/18** |

| Verdict metrics (secondary) | V5 | V6 | V7 | **V8** |
|---|---|---|---|---|
| Accuracy | 61.1% | 55.6% | 66.7% | **50.0%** |
| Recall | 0.33 | 0.44 | 0.33 | 0.22 |
| Precision | 0.75 | 0.57 | 1.00 | 0.50 |
| TP/FP/TN/FN | 3/1/8/6 | 4/3/6/5 | 3/0/9/6 | 2/2/7/7 |
| AUC / AP | .648/.677 | .698/.641 | .796/.844 | .654/.611 |
| mean hand-score | 4.50 | 4.33 | 4.83 | **4.22** |

McNemar V7 vs V8: 3 clips V7-only-correct, 0 V8-only-correct, **p = 0.250** — not significant at
n=18, consistent with everything else in this thread, but the direction is the wrong one and it is
the lowest verdict accuracy of any Qwen3-VL round tested. **The vocabulary fix and the
reasoning-quality fix are not the same fix**, and 4 clips actively regressed into stating the
opposite of GT — worse than V7 producing a wrong caption, because a CONTRADICT caption is actively
misleading as a training target, not merely uninformative.

## Two genuine, specific wins

**1. Ego's non-reaction is finally being detected — on the clips it matters most.** `00077` and
`00493` both correctly report "no reaction, gap continues closing, no nose dip" matching GT's exact
stated mechanism ("EGO fails to brake in time" / "EGO does not slow down"). Neither had been
captured by V5, V6, or V7. Both land at risk_score 49 and 47 — one point under the derived-verdict
cut. This is evidence the underlying signal improved even though the binary call didn't: the
STEP 4 corollary (attribute convergence to ego when the other agent holds static position) worked
correctly on `00493` specifically — `true_movers` states "pickup truck holds constant position...
ego converging with lead sedan," exactly the intended mechanism, with no turn-direction guess
required.

**2. Two persistent artifacts from V4-V7 did not reproduce.** The "school bus" fabrication on
`01552` (present in all four prior rounds) is absent. `01737`'s wrong turn-direction claim (V7 said
"turning left," GT says a right curve) is avoided entirely — not by getting the direction right, but
by the path-relative vocabulary rule making the claim unnecessary in the first place.

## Two new, specific failures — both traceable to this round's own design choices

**1. `01153`'s false-crossing-sedan hallucination is back, at the highest confidence in the run
(score 88).** This is the same fabricated "sedan turning into my path" template that broke V5 and
V6 on this identical clip, and it resurfaces despite `true_movers`' grounding language ("moving
relative to buildings and power poles") — the mechanism claims to be checking against the static
background but the check itself is unreliable here. V7's `apparent_vs_true` test suppressed this
exact failure on this exact clip; V8's `true_movers` field does the same job in principle but did
not reproduce the fix. Worth naming plainly: this is the single most damaging result in the run,
since it is the highest-confidence output and it is fabricated.

**2. A new false positive, `01550`, directly caused by the `ego_response` cue list.** GT's stated
reason for NO is that ego closes the gap "in a controlled manner while maintaining distance" — a
real but *gentle* reaction. The cue list this round (nose dip, drop in lead-vehicle expansion rate)
is tuned for hard braking; a controlled, gradual deceleration does not trip those cues, so the
model reports "No reaction -- gap continues closing steadily, no nose dip" when GT says ego was in
fact managing the approach. TN in V5, V6, and V7; FP here. This is a false negative on the *ego
reaction* judgment translating directly into a wrong verdict.

**3. `00687` regressed from a correct read in both V6 and V7** to reporting the SUV moving away
(gap *opening*) — the literal opposite of GT's SUV-drifts-in, gap-closing-rapidly mechanism.

## What this means, plainly

The prompt-grammar fix changed *what vocabulary* the model uses, and did not reliably change
*what the model perceives*. Two of the clearest per-clip wins (`00077`, `00493`) show the intended
mechanism working exactly as designed. Two of the clearest failures (`01153`, `01550`) show the
same class of mechanism — grounding against a static background, reading a subtle motion cue —
failing in ways specific to this round's exact instruction wording, not generic noise. That is a
more informative result than a flat "no improvement," because both directions are diagnosable, but
it is not evidence V8 should replace V7 as the production candidate. If anything, V7's
false-positive suppression (3→0, and it held on `01153` specifically) is a capability V8 lost.

## Recommendation

**Do not adopt V8 over V7 based on this screen.** Verdict accuracy is lower and not significantly
so in the other direction; caption reasoning-alignment is mixed (5 MATCH, but 4 CONTRADICT is the
worst outright-wrong count of any round). The caption-grammar hypothesis (state change + cause +
ego response, path-relative vocabulary) is not refuted — `00077` and `00493` are real, specific
evidence it can work — but the current cue list for detecting ego's reaction needs to distinguish
"no reaction" from "gentle, controlled reaction," and the static-background grounding check needs
to be more reliable specifically on the `01153`-type scene (an uncontested turn near other stopped
traffic), not just generically similar scenes.

**This is the seventh round of hand-tuning on the same 18 clips, and the numbers keep landing
inside each other's confidence intervals while flipping direction round to round.** The next
action with real information value is not an eighth prompt version — it is measuring V5, V6, V7 and
V8 together on a genuinely held-out slice, because at n=18 this thread cannot distinguish "V8 is
worse" from "V8 is noise," and the caption-quality claims specifically cannot be checked at scale
without `gt_reasoning_en`, which only exists for these 18 clips. A B1-probe run (video-caption
signal, not a verdict) on captions from these candidate prompts over ~100-200 held-out clips is the
next step with actual statistical power.

## Files

- Per-clip data + reasoning-alignment verdicts: `reasoning_analysis_v8_val18.xlsx`
- Raw output: `raw_v8_qwen3vl235b.jsonl`
- Prompt: `prompts/PROMPT_SEMSUP_V8_NARRATIVE.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v8_val18.py`
- Prior rounds: `summary.md`, `teacher_bakeoff_summary.md`, `qwen3vl_v4_summary.md`,
  `v5_balanced_summary.md`, `v6_kinematic_summary.md`, `v7_egoframe_summary.md`
