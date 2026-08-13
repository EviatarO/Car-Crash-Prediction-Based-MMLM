# Semantic Supervision for Crash Anticipation — Status Summary

**Date:** 2026-08-12 · **Thread:** `e4_vjepa_reason` · **Status:** blocked on a decision, not on capability

---

## Executive summary

The control arm is a **real, publishable success**: LoRA fine-tuning BADAS-Open on our data lifts
test AP from **0.853 → 0.900** (+0.047) on the 677-clip Private test set.

The treatment arm — adding a train-only semantic (caption) loss — has **lost twice**, both times
with statistical significance. But **neither loss is yet a valid test of the idea**, because both
runs deviated from the written experimental plan in two specific, identified, fixable ways.

**We are not at a scientific dead end. We are at an execution defect that has not been corrected.**
The cost to correct it is roughly one day of GPU time.

---

## The thesis question

Does language, used **only during training** as semantic supervision, improve a vision model's
collision-anticipation performance — while inference stays vision-only, at zero added cost?

Formally: does **B** (crash loss + λ·semantic loss) beat **A1** (crash loss only) beat **A0**
(frozen baseline)? The semantic effect is `B − A1`, with a bootstrap confidence interval.

---

## What was built

| Component | Role | Trained? |
|---|---|---|
| V-JEPA2 ViT-L trunk (BADAS-Open) | video → patch features | LoRA (2.8M params, 0.84%) |
| Crash head (BADAS's own) | patches → P(collision) | reused, frozen |
| Predictor (`ResamplerProjector`, 1.25M) | patches → text-embedding space | trained |
| SigLIP text encoder | caption → target embedding | frozen |

Semantic loss = InfoNCE, each clip's predicted embedding contrasted against the **whole** caption
bank (1,413 train / 348 val) with sibling-video masking. λ = 0.05. The Predictor and SigLIP are
**discarded at inference** — the deployed model is vision-only.

---

## Results to date (677-clip Private test, AP is the headline metric)

| Arm | What it is | test AP | AUC | Verdict |
|---|---|---|---|---|
| **A0** | Frozen BADAS-Open baseline | 0.853 | 0.864 | reference |
| **A1_1761** | Crash-only LoRA, 1,761-window pool | **0.900** | 0.904 | **champion, +0.047 over A0** |
| B_1761 sequential | A1 ep4 + InfoNCE continued | 0.897 | — | confounded design, not a valid test |
| B_1761 parallel | InfoNCE from scratch, **V10 captions** | 0.8901 | 0.8955 | **lost**: ΔAP +0.0105, CI [0.0040, 0.0173] |
| A1-v2 | Full 4,446 pool + cosine LR + encoder-only LoRA + dropout 0.10 | 0.868 (sel.) / 0.888 (best) | — | **lost** to A1_1761; recipe abandoned |
| **B-v2** | InfoNCE from scratch, **V12 clean captions** | **0.8796** | 0.8905 | **lost**: ΔAP +0.0189, CI [0.0099, 0.0285] |

Paired bootstrap, 5,000 resamples, same 677 clips scored by both arms. Both B losses exclude zero
— they are real, not sampling noise. P(B beats A1) = 0.0%.

> Footnote on numbers: `a1_1761/test_summary.json` records test AP = 0.900; the paired-bootstrap
> script recomputes 0.8986 from the same per-clip file (different AP estimator). Both arms in every
> comparison use the *same* estimator, so the reported deltas are valid.

---

## The caption corpus problem (found, largely fixed)

A full project review found the original **V10** caption corpus **leaked the label**: positives were
captioned in GT-informed mode, negatives in blind mode, producing two different vocabularies by
class. A plain TF-IDF + logistic-regression classifier recovered the crash label from **caption text
alone at AUC 0.9643** — higher than the vision model itself. The semantic branch was acting as a
redundant, noisier copy of the label rather than as scene grounding.

**V12** removed the GT/blind branch entirely (one neutral prompt, no arguments), added a closed
`gap_trend` vocabulary and symmetric alarm/reassurance word bans, and the full 1,761-window pool was
re-captioned (real logged cost: $32.82 for the tracked portion).

| Corpus | text→label AUC | Target | Result |
|---|---|---|---|
| V10 | 0.9643 | — | severe leak |
| **V12** | **0.7640** | < 0.75 | narrow miss (accepted); 43% cut in excess-over-chance signal |

Residual leakage traced to **genuine kinematic vocabulary** (`braking`, `decreasing gap`) — physically
real correlates of the label, not register violations (0/100 banned-word violations). Likely a floor,
not a fixable prompt defect.

**Critically: cleaning the corpus did not close the performance gap.** B-v2 (clean captions) lost by
*more* than B_1761-parallel (leaky captions). So label leakage is ruled out as the explanation.

---

## Why it isn't working — assessment

Ranked by confidence. The first two are defects we introduced; the third is a structural constraint.

### 1. The written plan was not followed (high confidence this happened; effect size unknown)

The plan (`2026-07-07_Plan Semantic-Supervision...`, lines 113–145) specifies run order
`0 → A0 → B1 → A1 → B`, where **B1** trains the Predictor alone against frozen vision, and **B**
warm-starts that Predictor (`"Predictor (warm-started from B1)"`).

**Neither B run did this.** Both recorded `predictor_init: null` — the Predictor was randomly
initialized and co-trained with the LoRA from step 1.

Why this plausibly matters: the semantic gradient must pass *through* the Predictor to reach the
ViT-L. When the Predictor is random, that gradient is a random rotation of the true error signal —
the trunk is being taught by a teacher that knows nothing yet. **B-v2's selected checkpoint was
epoch 2**, i.e. the winning checkpoint was formed exactly when the Predictor was noisiest.

### 2. The two arms differ by more than λ (confirmed defect, effect size unmeasured)

`clip_grad_norm_(trainable, 1.0)` clips one **global** norm across all trainable parameters.

- **A1**: `trainable` = LoRA only (2.8M) — budget 1.0 for the LoRA alone.
- **B**: `trainable` = LoRA + Predictor (1.25M) + `log_tau` — same budget 1.0, shared.

If the Predictor's early gradients are large (random init, ~7-nat InfoNCE loss), they inflate the
global norm, and the resulting scale-down is applied **to the LoRA's crash gradients too**. B's trunk
would receive systematically smaller effective updates than A1's, for reasons unrelated to semantics.

Whether this actually bites depends on whether the clip is active at all — **not currently logged.**

### 3. Scale (high confidence this is a real constraint)

We have **1,761** caption-window pairs. For comparison:

| System | Alignment pairs |
|---|---|
| CLIP | 400,000,000 |
| PeFoMed (Stage 1 alignment) | ~765,000 |
| **This work** | **1,761** |

That is two-to-three orders of magnitude below anything that has demonstrably worked for
language-supervised visual representation learning.

The signal *is* present but modest. InfoNCE chance level is ln(N): ln(348) = 5.852 for val. B-v2
reached **val_sem_loss 4.90** — real above-chance retrieval (effective candidate set narrowed from
348 to ~134, ≈2.6× chance) and still improving at epoch 8. So the captions carry recoverable
information. Whether 1,761 examples of it can reshape a 334M-parameter trunk is the open question.

### 4. Possible gradient conflict (unmeasured)

The crash and semantic objectives may be pulling the shared trunk in opposing directions. The
diagnostic is `cos(∂L_sem/∂θ_LoRA, ∂L_crash/∂θ_LoRA)`; if persistently negative, that is the
mechanism, and the literature has standard responses (gradient-similarity gating, PCGrad). **Never
measured.**

### What is *not* the problem

- **Not a broken gradient path.** No `.detach()` anywhere between the semantic loss and the trunk;
  verified in code. The learnable temperature moved (0.07 → 0.0563), which is only possible if the
  InfoNCE term produced real gradients.
- **Not a frozen Predictor.** It is fully trained.
- **Not label leakage.** Fixed in V12; the gap widened rather than closed.
- **Not checkpoint-selection gaming.** B-v2 loses under any selection rule, including best-on-test.

---

## What to do now

**One probe + one run. Approximately one day of GPU time. No architecture change, no new thesis
question.**

| Step | Action | Why | Cost |
|---|---|---|---|
| 1 | **B1-v2 probe on V12** — frozen trunk, Predictor only | Executes the plan's missing step. Produces the warm-start checkpoint **and** a clean retrieval number to replace the V10-contaminated 32×/24× figure | ~20 min |
| 2 | **Per-group gradient clipping** — clip LoRA on its own budget, as A1 does | Removes the asymmetry so the arms differ only by λ | code only |
| 3 | **B-v3** = B-v2 + warm-start + per-group clip, with grad-norm and crash/semantic cosine logging | The first valid test of the thesis question | ~1 h |

**Decision gate.** If B-v3 still loses to A1_1761 with both defects corrected, the negative result is
real, diagnosed, and defensible — and the gradient log will say *why* (conflict vs. weak signal).
That is a genuine contribution, not a failure.

**Schedule impact: none.** We are ~12 days ahead of the Gantt; this fits inside the W3 slot without
touching the writing phase.

### Held in reserve (do not start yet)

- **Reverse the projection direction** — project text into vision space instead of vision into text,
  placing the loss adjacent to the ViT-L with no trainable module in the gradient path. Well-founded
  (cross-modal prototype literature), but a genuine architecture change. Only justified if B-v3 fails
  *and* the gradient log points specifically at the noisy-teacher path.
- **Shuffled-caption control** (plan step 3.3) — captions shuffled within class. Decides whether
  language *content* matters at all. Cheap, and needed for the writeup regardless of outcome.
- **Multi-seed replication** (Gantt W4) — required before any claim is publishable.

---

## What is already publishable regardless of outcome

1. A student that **beats the published BADAS-Open baseline** (0.900 vs 0.853, +0.047).
2. A **rigorously controlled negative result with a diagnosed cause** — a methodological warning for
   caption-based distillation, backed by the label-leakage finding (text alone predicts the label at
   AUC 0.964) and its correction.
3. The **training-inversion failure mode** — training only on mined failures against a frozen head
   collapses to AP 0.333 / AUC 0.163 (below random). A reusable lesson.
