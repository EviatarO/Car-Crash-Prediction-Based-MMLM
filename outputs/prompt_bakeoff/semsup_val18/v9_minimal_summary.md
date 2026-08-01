# PROMPT_SEMSUP_V9_MINIMAL vs qwen/qwen3-vl-235b-a22b-thinking — 18-clip screen

**Question:** seven prompt versions grew from 619 to 4,072 tokens and produced verdict
accuracies of 9, 9, 11, 11, 10, 12, 9 out of 18 — none distinguishable from the others at n=18.
Is the scaffolding itself the problem — is the model spending capacity satisfying instructions
instead of looking at the frames? V9 keeps only the six insights with direct evidence behind
them (delta-not-state, ego-response-as-closing-clause, path-relative motion vocabulary,
ignore-apparent-only-agents, a single unbanded continuous score) and drops every observation
field, sub-score decomposition, and all but one worked example. 866 tokens — 21% of V7's length,
1.4× V2's.

**Settings:** same model as V4-V8, native 1280×720, `detail="high"`, `temperature=0.1`,
`max_tokens=20000`. 18/18 succeeded (one transient connection retry, self-healed). Wall 1935s —
notably slower than every prior round despite the shorter prompt, consistent with the model
doing more of its own unprompted reasoning rather than following an externally imposed pipeline.

**Scorer disclosure:** hand-scores assigned by Claude, who wrote the prompt.

## Headline: best verdict metrics of any round tested, by a clear margin

| | V4 | V5 | V6 | V7 | V8 | **V9** |
|---|---|---|---|---|---|---|
| tokens | 1,386 | 2,273 | 3,257 | 4,072 | 3,357 | **866** |
| Accuracy | 61.1% | 61.1% | 55.6% | 66.7% | 50.0% | **72.2%** |
| Recall | 0.22 | 0.33 | 0.44 | 0.33 | 0.22 | **0.56** |
| Precision | 1.00 | 0.75 | 0.57 | 1.00 | 0.50 | 0.83 |
| TP/FP/TN/FN | 2/0/9/7 | 3/1/8/6 | 4/3/6/5 | 3/0/9/6 | 2/2/7/7 | **5/1/8/4** |
| AUC | — | 0.648 | 0.698 | 0.796 | 0.654 | **0.821** |
| AP | — | 0.677 | 0.641 | 0.844 | 0.611 | **0.786** |

V9 is the first round to break out of the recall/precision trade-off every prior version was
stuck in — it has both the best recall (0.56, previous best 0.44) and near-best precision (0.83,
second only to V7's 1.00), with only one false positive. McNemar vs V7 (the prior best): 3
V7-only-correct, 4 V9-only-correct, **p = 1.000** — not significant, consistent with every
comparison in this thread, but the direction and the AUC gap (0.821 vs 0.796, and V9's 95% CI
would need to be computed to know if it clears V5/V6's — worth doing before treating this as
settled) are the best signal seen in nine rounds.

## But reasoning alignment is mid-pack, not best — and this matters more than the verdict

| | MATCH | PARTIAL | MISS | CONTRADICT | mean hand-score |
|---|---|---|---|---|---|
| V4 | — | — | — | — | 5.11 |
| V5 | — | — | — | — | 4.50 |
| V6 | — | — | — | — | 4.33 |
| V7 | — | — | — | — | 4.83 |
| V8 | 5 | 6 | 3 | 4 | 4.22 |
| **V9** | 4 | 7 | 2 | **5** | 4.33 |

V9 ties V6 for the lowest mean caption-fidelity score of any round with a hand-scored breakdown,
and it has the **most** CONTRADICT clips (5, worse than V8's 4) — captions that state something
GT explicitly contradicts, or fabricate a hazard GT does not describe. **The verdict metrics
improved substantially; the caption — the actual SigLIP training target — did not.** These two
axes moved in opposite directions in the same run, which is the central finding of this round.

## The most important single fact in this round: the highest-confidence prediction is fabricated

`00529` scores 82 — the highest risk_score in the entire run — and the caption invents a
pedestrian "entering crosswalk on left into ego's path." GT's reasoning for this clip does not
mention a pedestrian intent on entering the road at all; the actual cause is a silver SUV
drifting into ego's lane after the left lane becomes obstructed, which V9 misses entirely. The
verdict is correct (YES) but for a completely wrong, invented reason. A downstream system relying
on the highest-scored clips as the clearest signal would be relying on a fabrication first.

## `01153`'s hallucination is now confirmed as a stable, prompt-independent failure

The same fabricated "sedan crossing my path" narrative — GT explicitly states all vehicles remain
in their own lanes while ego makes an uncontested right turn — has now appeared in **V5, V6, V8,
and V9**. It was absent only in V7, whose specific grounding mechanism (`apparent_vs_true`,
checking agent position against the static background before allowing a lateral claim) happened
to suppress it there. V9 has no equivalent explicit check and the hallucination returned at high
confidence (score 73, tied with four genuine positives). This is evidence the failure lives in
the model's prior on this scene type, not in prompt length or scaffolding — shortening the prompt
did not remove it, and only one specific grounding mechanism (not yet reproduced reliably) has
ever suppressed it.

## A hypothesis from the V9 design doc was falsified by this run

V9's docstring predicted that removing V5's *named* numeric bands (which caused score-snapping to
6 distinct values in that round) would fix the clustering problem, since nothing in V9 labels
specific ranges. It did not:

| | mechanism | distinct risk_score values |
|---|---|---|
| V5 | 5 named bands | 7 |
| V6 | 4 summed sub-scores, no bands | 11 |
| V7 | 4 summed sub-scores, no bands | 14 |
| V8 | 4 summed sub-scores, no bands | 16 |
| **V9** | **1 unbanded integer, no bands** | **6** |

V9 has the *second-fewest* distinct values of any round, behind only V5. The real driver of
score granularity was the **sub-score decomposition** (asking for 4 independent numbers and
summing them), not the presence or absence of named bands — a single free-form integer output
snaps just as hard as a labeled band did. Worth correcting explicitly rather than letting the
wrong explanation stand: V9's compact form has a real cost here, and any future minimal-prompt
iteration should keep some form of decomposition if score resolution matters.

## What genuinely worked

- **`00687`**: best-aligned reasoning on this clip across every round tested — correctly
  identifies the gray SUV merging into ego's lane with closing distance and no reaction, matching
  GT directly.
- **`01550`**: a real, specific fix of a V8 regression. GT says ego closes the gap "in a
  controlled manner" — a gentle reaction. V8 missed this nuance and produced a false positive;
  V9's caption ("gap closing then steady with ego braking in response") captures it correctly and
  the verdict lands right.
- **`00077`, `00493`**: both correctly identify ego's non-reaction as the decisive mechanism,
  landing comfortably above the verdict cut (73) rather than the one-point-under near-misses seen
  on these same clips in V7 (43) and V8 (47/49).
- The two persistent minor fabrications tracked since V4 — the `01552` "school bus" and the
  `01643` invented road-work sign — do not reproduce this round.

## Recommendation

**Do not treat this as a clean resolution to the length question.** The honest read is that V9
produced the best-calibrated *verdict* signal of any round while producing reasoning that is no
more (and by CONTRADICT-count, slightly less) faithful to GT than the longest, most scaffolded
version tried. That is a genuinely informative result — it suggests the four-round trend of
"more structure helps the score, not the caption" continues even at 21% of the length — but it is
not evidence that shorter is simply better across both axes, and the two specific failure modes
that matter most for a training-target caption (fabricated highest-confidence predictions,
persistent scene-specific hallucination) are both still present.

**This is the ninth prompt version and the pattern from round three onward has held: verdict
metrics and caption fidelity move independently, sometimes in opposite directions, and n=18 is
not powered to resolve which prompt is actually best on either axis.** The next action with real
information value is unchanged from what was recommended after V7 and V8: a frozen comparison —
no further edits — of the strongest candidates on each axis (V9 for verdict/AUC, V4 or V7 for
caption fidelity) against held-out clips from `dataset/manifests/semsup_promptbakeoff.jsonl`,
scored where possible by something other than hand-scoring at n=18.

## Files

- Per-clip data + reasoning-alignment verdicts: `reasoning_analysis_v9_val18.xlsx`
- Raw output: `raw_v9_qwen3vl235b.jsonl`
- Prompt: `prompts/PROMPT_SEMSUP_V9_MINIMAL.py`
- Analysis script: `teacher_distillation/scripts/reasoning_analysis_v9_val18.py`
- Prior rounds: `summary.md`, `teacher_bakeoff_summary.md`, `qwen3vl_v4_summary.md`,
  `v5_balanced_summary.md`, `v6_kinematic_summary.md`, `v7_egoframe_summary.md`,
  `v8_narrative_summary.md`
