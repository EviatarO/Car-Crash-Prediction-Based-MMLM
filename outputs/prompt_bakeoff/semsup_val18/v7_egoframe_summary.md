# PROMPT_SEMSUP_V7_EGOFRAME vs qwen/qwen3-vl-235b-a22b-thinking — 18-clip screen

**Question:** the V6 run showed that on the 6 clips where GT says the EGO VEHICLE ITSELF turns,
the model reported "holding lane centre / proceeding straight" on 4, with that boilerplate in
18/18 responses. The hypothesis: when ego rotates, static objects sweep across the image, and a
model that thinks it is going straight blames the objects — producing false positives (parked
cars read as "cutting across my path") and false negatives (ego's own manoeuvre never named)
from the *same* defect. V7 fixes the world frame from static features first, then subtracts ego
motion before judging any vehicle. Does it work?

**Settings:** same model as V4/V5/V6, native 1280×720, `detail="high"`, `temperature=0.1`,
`max_tokens=20000`. 18/18 first attempt, 0 failures, 0 derivation mismatches, 0 off-enum
`conflict_source` values. Wall 845s.

**Examples are principle-based, not scene-based.** After the overfitting review, V7's four worked
examples were rewritten into settings and actor types absent from the val set entirely (rural
roundabout/tractor, motorway slip road/coach, tunnel/box van, country lane/cyclist) so a gain
here must come from applying the rule, not recognising a rehearsed scene.

**Scorer disclosure:** hand-scores assigned by Claude, who wrote the prompt.

## Headline: best numbers of any round — and not statistically significant

| Round | Acc | Recall | Prec | TP/FP/TN/FN | AUC | AP | distinct scores | mean caption |
|---|---|---|---|---|---|---|---|---|
| V4 | 11/18 | 0.22 | 1.00 | 2/0/9/7 | — | — | — | 5.11 |
| V5 | 11/18 | 0.33 | 0.75 | 3/1/8/6 | 0.648 | 0.677 | 6 | 4.50 |
| V6 | 10/18 | 0.44 | 0.57 | 4/3/6/5 | 0.698 | 0.641 | 11 | 4.33 |
| **V7** | **12/18** | 0.33 | **1.00** | **3/0/9/6** | **0.796** | **0.844** | **14** | 4.83 |

**But the significance tests say do not celebrate:**
- V6 vs V7: McNemar **p = 0.625**. V5 vs V7: **p = 1.000**.
- Bootstrap 95% CI on AUC: V7 [0.542, 0.981] vs V5 [0.361, 0.882] vs V6 [0.431, 0.938] — heavily
  overlapping.

The one defensible statistical claim: **V7 is the only round whose AUC confidence interval
excludes 0.5** (lower bound 0.542). V5 and V6 cannot be distinguished from random ranking; V7
can. That is a weak claim, but it is the first non-trivial one in seven rounds.

**Threshold sweep** (same run, free) — the 50 cut is not the best operating point:

| Threshold | TP | FP | TN | FN | Recall | Prec | Acc |
|---|---|---|---|---|---|---|---|
| ≥12 | 7 | 3 | 6 | 2 | 0.78 | 0.70 | 72.2% |
| **≥42** | **5** | **0** | **9** | **4** | **0.56** | **1.00** | **77.8%** |
| ≥50 (prompt default) | 3 | 0 | 9 | 6 | 0.33 | 1.00 | 66.7% |

At threshold 42: 5 TPs with zero false positives, 14/18. The four highest-scoring clips are all
true positives, and `00077` (49) and `00493` (43) are true positives sitting just under the
default cut — the ranking is genuinely better even where the binary call is not.

## What actually worked: false-positive suppression

**All three of V6's false positives are gone** (`01153`, `01281`, `01504` — the last two had been
clean TNs in every round before V6). The confabulated "vehicle turning across ego's path"
template does not appear anywhere in the V7 output. On `01153` the model now correctly notes the
sedan turns into a *different* lane. The `apparent_vs_true` test did its job: it is a grounding
check (position relative to fixed background) rather than V6's internal-consistency check, and
that difference is exactly what V6 structurally could not provide.

Boilerplate collapsed: `"lane centre/position"` went from **18/18 → 0/18**.

## What did NOT work — two findings that matter more than the score

**1. Ego-path estimation is a coin flip: 9/18 correct.** V6 always said "straight"; V7 now says
"turning left" on 7 clips where GT has ego straight or turning right. **The prompt swapped one
default for another** rather than producing genuine estimation. Direction is inverted on `01153`
(GT right → reported left), `01737` (GT right curve → left), and `00687` reverts to "straight"
where GT has a left turn. Detection of *whether* ego manoeuvres improved (5/6 vs 2/6); inference
of *which way* did not. The stated rule ("static sweeps left ⇒ ego turns right") is correct, so
the failure is in reading the sweep direction off the frames, not in the logic.

**2. `conflict_source` never once returned `ego_into_other`** — distribution was 11 `none`,
4 `other_into_ego`, 3 `longitudinal`. The entire purpose of adding the enum was to let the model
attribute a conflict to ego's own manoeuvre, and it did not use that option a single time, even
on the clips where it correctly identified ego as turning. **So V7's accuracy gain did not come
from the mechanism V7 was designed around.** It came from the FP-suppression half.

**Consequence — the suppression is too aggressive.** The same `apparent_vs_true` test that killed
the false positives also labels genuine lateral events as `APPARENT ONLY`: `00529` (GT: SUV
drifts into ego lane) and `00474` (GT: van turns sharply into ego lane) are both now dismissed as
apparent, and `00687` — which V6 got *right* — regressed to "gray SUV parked on right". That is
why recall fell back to 0.33 while precision went to 1.00.

## Caption quality

Mean 4.83 (V4 5.11, V5 4.50, V6 4.33) — mid-pack, differences well inside noise. 4 green / 10
orange / 4 red. Two notes:
- `00493` produced the best caption-vs-GT match in the whole set ("*ego turning left … following
  a silver sedan with brake lights illuminated … gap closing*" — GT's exact mechanism) yet scored
  43 and reads as a false negative. Caption quality and verdict correctness are visibly decoupled.
- Persistent per-clip artifacts reproduce again: the `01552` "school bus" (now 4-for-4 across
  V4/V5/V6/V7) and the `01643` parked-cars fabrication. These are stable model/scene artifacts,
  not prompt-dependent noise.

## Recommendation

**Do not treat 12/18 as evidence that V7 is the best prompt.** With p = 0.625 against V6 and
p = 1.000 against V5, this screen cannot rank these prompts, and seven rounds have now produced
9, 11, 11, 11, 11, 10, 12 — a spread entirely consistent with noise around a flat ~60%.

What V7 does establish, because they are within-run observations rather than between-round score
deltas:
- The static-frame grounding check **removes** the confabulated-crossing failure mode (3 FP → 0).
- Ego-rotation *direction* inference is unreliable (9/18) and is the next real target.
- The model will not attribute a conflict to its own manoeuvre even when told to and even when it
  has correctly identified the manoeuvre — `ego_into_other` was never emitted.

Next step should be the **frozen comparison on held-out data**, not an eighth prompt: V5, V6, V7
(and ideally a deliberately minimal variant, since every round so far has only added complexity)
run once on ~120 clips from `dataset/manifests/semsup_promptbakeoff.jsonl` — 249/249 balanced,
83 per horizon bucket, zero overlap with these 18, never captioned. At n=120 the accuracy CI
narrows from roughly ±21% to ±9%, which is the difference between being able to rank these
prompts and not.

## Files

- Per-clip data + scores: `reasoning_analysis_v7_val18.xlsx`
- Raw output: `raw_v7_qwen3vl235b.jsonl`
- Prompt: `prompts/PROMPT_SEMSUP_V7_EGOFRAME.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v7_val18.py`
- Prior rounds: `summary.md`, `teacher_bakeoff_summary.md`, `qwen3vl_v4_summary.md`,
  `v5_balanced_summary.md`, `v6_kinematic_summary.md`
