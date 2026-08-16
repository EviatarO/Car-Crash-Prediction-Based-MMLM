# Semantic Supervision — Open Problems, Proposed Solutions, Priority Order

**Written 2026-08-15.** Covers everything surfaced 2026-08-13 → 15. Companion to
`summary.md` (status) — this file is the **problem register and forward plan**.

**Goal reminder:** the target is a **positive, publishable method**, not a negative result.
Nothing here is a recommendation to cancel the approach. The design space has been tested at
exactly **one point** (one λ, one tap point, one direction, one training paradigm), four times,
with progressively better plumbing.

---

## Where we actually stand

| Arm | test AP | ΔAP vs A1 (paired bootstrap, 677 clips) | Verdict |
|---|---|---|---|
| A0 frozen baseline | 0.853 | — | reference |
| **A1_1761 (control, champion)** | **0.900** | — | **+0.047 over A0. Real, banked** |
| B_1761 parallel (V10 leaky corpus) | 0.8901 | +0.0105 [0.0040, 0.0173] | lost |
| B-v2 (V12 clean corpus) | 0.8796 | +0.0189 [0.0099, 0.0285] | lost |
| B-v3 (warm-start + per-group clip) | 0.8768 | +0.0218 [0.0117, 0.0325] | lost |
| B-v3 @12 epochs (extension) | 0.8655 | +0.0330 [0.0156, 0.0519] | lost, overfit |

**The gap widened monotonically across all four attempts.** Every "fix" so far was
*execution hygiene* (warm-start, clipping, corpus cleaning), none touched the *design*.
That pattern is itself the strongest evidence that the problem is structural.

**Two positive findings already banked, independent of whether B ever wins:**
1. A1 beats published BADAS-Open: **0.900 vs 0.853 (+0.047)**.
2. Frozen crash-video features linearly encode teacher-caption semantics at **31× chance**,
   scaling as ~n^0.63 with **no saturation** — a claim, not an anecdote.

---

## Priority table

| # | Problem | Proposed solution | Cost | Why this priority (few words) |
|---|---|---|---|---|
| **P1** | Gradient reaches trunk but is **orthogonal** to crash (cos≈0, 44-56% conflict) — joint-λ co-training has failed 4× | **Two-stage: semantic pre-train → crash fine-tune**, pooler unfrozen in stage 2 | 1-2 d | The paradigm that *actually works* in the literature (CLIP/SLIP/LiT). Our joint-λ variant is the unproven one. **Never tried in this order** |
| **P2** | Semantic loss attaches to the 2560×1024 patch grid; the classifier reads only a 1024-d vector | **Move the semantic loss to the pooler's output** (`forward_hook` not `pre_hook`) | 0.5-1 d | Structurally forces every semantic gradient through the decision path. Probe confirms the target is learnable there (22× chance) |
| **P3** ✅ | **RESOLVED 2026-08-16.** Unknown whether the semantic gradient moves *high-weight* or *low-weight* tokens | Measured: real ratio 1.8× the random-noise control (0.0034 vs 0.0019, n=40, no CI). Gradient reaches the decision-relevant representation at least as well as chance — **P2 downgraded**, **P1 upgraded** | done | See elaboration below |
| **P4** | λ=0.05 was never swept; chosen by a magnitude-matching heuristic | **λ sweep {0.01, 0.05, 0.2, 0.5}**, short runs | 1 d | A reviewer *will* ask whether a flat B is just bad λ. Cannot be defended without it |
| **P5** | The leakage gate punished **legitimate** predictive signal as if it were contamination | **Re-frame the gate**: grounding check + outcome-word ban, drop the blanket AUC ceiling | 0.5 d + API | Cleaning V10→V12 made B *worse* (0.0105→0.0189). Strong hint we stripped the useful part |
| **P6** | Never established whether caption **content** matters at all, vs just caption **class** | **B-shuffle control** (permute captions within class) | 1 d | Needed for the writeup *whatever* the outcome. Cleanest single control in the whole design |
| **P7** | Scaling curve still rising at 1,413 (13×/19×/31×, ~n^0.63, no flattening) | **Caption the full 4,446 pool with Qwen3-VL** (~$11 vs ~$162) | 0.5 d + ~$11 | Only lever that adds *information* rather than re-routing it. Now affordable |
| **P8** | Predictor trains jointly — free to re-aim at convenient features | **Freeze predictor after warm-start** (one-line ablation) | 0.5 d | Closes one bypass route from the other end. Nearly free to test |
| **P9** | Full-bank InfoNCE won't scale past ~10k captions (peaked softmax, not compute) | **SigLIP sigmoid loss** or MoCo-style fixed queue | 1 d | Only matters *if* P7 succeeds and we scale. Not blocking now |
| **P10** | A1 control could be stronger (crash head is frozen) | **A1-v3: unfreeze head at 0.1× LR** | 1 d | Raises the *control* bar — makes B's job harder. Optional, do last |

---

## Closed / eliminated (do not re-open)

| Problem | Outcome |
|---|---|
| **InfoNCE false negatives** (near-duplicate captions punished as negatives) | **ELIMINATED 2026-08-15.** Cross-video caption cosine: mean 0.701, p99 0.870. At a 0.90 threshold only **~4 of 1,413** negatives per anchor would be masked (0.3%). Cannot explain a 0.02 AP gap. **Do not implement masking.** Bonus finding: V12 captions are genuinely clip-specific despite the constrained vocabulary |
| **"1024-d is too small to carry captions"** | **REFUTED.** Pooled probe = 22× chance. Info survives the 2560× compression |
| **"Train longer / 12 epochs"** | **ANSWERED: no.** Pure overfitting — `train_val_gap` climbs 0.63 → 1.04 across epochs 9-12, train crash loss → 0.119 while val → 1.133. 8 epochs was already past the useful window |
| **Gradient gating / PCGrad** | **NOT APPLICABLE.** These fix cos ≪ 0 (opposition). We measured cos ≈ 0 (orthogonality). Gating would do nothing |
| **Bigger λ to strengthen a weak signal** | **WON'T WORK ALONE.** cos is scale-invariant — λ cannot change direction, only magnitude. Pushing harder sideways is still sideways. (λ sweep in P4 is for *defensibility*, not as the fix) |
| **Bypass hypothesis (strong and weak form)** | **REFUTED, both forms, 2026-08-16.** Strong form (info can't survive the bottleneck): refuted 2026-08-15, pooled retrieval stayed at 22× chance. Weak form (the trained gradient specifically avoids pooler-visible directions): refuted by P3 — the real A1→B weight difference reaches the pooled space at 1.8× the rate a random perturbation would, not less. The bottleneck is not where the problem is |

---

## Elaboration

### P1 — Two-stage: semantic pre-train → crash fine-tune

**The problem.** Across 8 epochs of B-v3 the crash/semantic gradient angle drifts from
+0.0165 (45.2% conflicting steps) to −0.0244 (55.9%), while the semantic term grows
*relatively louder* (λ·|g_sem|/|g_crash|: 0.048 → 0.130) as the crash loss saturates.
Joint co-training with a fixed λ has now failed four times.

**Why two-stage is better motivated.** CLIP, SLIP and LiT do **not** co-train a language
objective against a downstream task with a weight λ. They **pre-train** with the language
objective, then transfer. Our joint-λ design is the *less-proven* variant and it's the one
that keeps failing.

**Critically: the reverse order was already tested and failed.** `B_1761 sequential`
(continue A1 epoch 4, then add InfoNCE) was flat/declining. **Semantic → crash has never
been run.** Different experiment, better-motivated direction.

**Steps:**
1. **Stage 1 — semantic only.** Train LoRA encoder + Predictor with **only** the InfoNCE loss
   (`--semantic-weight 1.0`, crash loss disabled). No crash head involvement at all.
   Target: maximize held-out clip retrieval. Checkpoint on retrieval, not loss.
2. **Measure what stage 1 achieved.** Record retrieval@1 on the held-out 221 clips.
   Also record A0-style crash AP with the stage-1 encoder + frozen head — this tells us
   whether semantic pre-training alone *already* moved crash performance (either direction).
3. **Stage 2 — crash fine-tune.** Load the stage-1 encoder. Train with crash loss only.
   **Unfreeze the pooler** (this is required — see below). Lower LR than stage 1
   (start 0.1×, i.e. 2e-5) to reduce catastrophic forgetting.
4. **Track forgetting.** After stage 2, re-measure caption retrieval with the stage-2 encoder.
   If it collapses back to chance, stage 2 erased the semantic structure and we need a
   smaller LR / fewer epochs / partial freezing.
5. **Compare** final test AP against A1_1761 (0.900) with the standard paired bootstrap.

**Why the pooler must be unfrozen in stage 2.** The pooler is currently frozen and was fitted
to the *original* encoder's feature geometry. If stage 1 moves those features, a frozen pooler
is reading a representation it was never fitted to — it cannot benefit from the new structure.
This is the one place where "unfreeze the pooler" becomes genuinely necessary rather than
neutral.

**Risk:** catastrophic forgetting in stage 2 — the crash signal is direct and strong and may
simply overwrite stage 1. Mitigations above (step 3-4). The retrieval-after-stage-2 metric is
the early-warning indicator.

---

### P2 — Move the semantic loss to the pooler output

**The problem.** Today the hook is a `register_forward_pre_hook` capturing the pooler's
**input** — the full (2560, 1024) grid. The Predictor has *learnable* queries and can read any
directions it likes; the crash head is a *frozen* pooler reading fixed directions. Verified
shapes: input `(1, 2560, 1024)` → output `(1, 1024)`, a 2560× compression, 0.04% of values
retained.

**Why this should help.** The pooled vector is a **bottleneck**: it is the only thing the
classifier sees. Attaching the semantic loss there makes it structurally impossible for the
semantic gradient to shape anything the classifier can't read.

**Why we know the target is learnable.** The pooled-tap B1 probe (2026-08-15) reached
**9.95% clip retrieval = 22× chance** (vs 31× from the full patch grid, and 0.45% chance).
Caption information survives the bottleneck comfortably. The `meanpool` control gave 18×,
statistically indistinguishable from 22× at n=221 (±1.9pp), so the crash-tuned attention is
**not** selecting against caption semantics.

**Steps:**
1. Switch the semantic tap in `semsup_train.py` from `_captured["patches"]` to
   `_captured["pooled"]` (the post-hook already exists in `semsup_common.py` as of 2026-08-15).
2. Replace `ResamplerProjector` with the `_VectorMLP` (1024 → 512 → 768) already written for
   the probe — the Resampler's cross-attention is a no-op on a single vector.
3. Add a `--semantic-tap {patches,pooled}` flag so both are runnable and A/B-comparable.
4. Run with B-v3's exact recipe otherwise (warm-start from the pooled-tap B1 checkpoint,
   per-group clip, grad-angle logging on).
5. **Leading indicator to watch:** `grad_cos_mean` should rise well above the ~0.02 we see
   today. If it does, that alone is a publishable mechanistic result *regardless of final AP*:
   "relocating the alignment target to the classifier bottleneck raises crash/semantic gradient
   alignment from ~0 to X."

---

### P3 — Δpatches vs Δpooled — ✅ RESOLVED 2026-08-16

**The question.** The pooled-tap probe (§5b) proved caption info *exists* in the pooled
vector. It did **not** prove that the semantic gradient, applied at patch level in the
runs we've actually trained, produces changes that *show up* in the pooled vector. Those
are different claims, and this was run to settle the second one.

**What was run** (`student_training/scripts/p3_delta_patches_vs_pooled.py`): loaded
A1_1761 epoch 4 and the fully-corrected Stage-B run's epoch 2 LoRA weights on the same
frozen base, captured `patches` and `pooled` for the same 40 held-out clips under each,
computed `‖Δpooled‖ / ‖Δpatches‖`, and compared it against the same ratio for a random
perturbation of `patches` with equal norm (the control that separates "the real weight
difference reaches the pooled space" from "any equal-sized change would").

**Result:**

| | mean ratio ‖Δpooled‖ / ‖Δpatches‖ |
|---|---|
| Real (A1 → B) | 0.00341 |
| Random-noise control | 0.00186 |

Real is ~1.8× the random baseline — higher, but a modest effect, and this is one point
estimate over 40 clips with no confidence interval. Read plainly: the weight difference
between the crash-only and crash+semantic checkpoints is **not** being preferentially
routed away from the pooler; if anything it leans mildly toward directions the pooler keeps.

**Consequence.** Combined with §7c's gradient-angle finding (cos≈0, drifting mildly
negative — not the signature of a blocked signal, but of a genuinely weak-or-conflicting
one), the bypass framing no longer explains the result well. The semantic signal's
influence does appear to reach the classifier-relevant representation. **P2 (move the loss
to the pooled tap) is downgraded** — it would likely relocate an already-arriving gradient
rather than unblock a stuck one. **P1 (two-stage training) is upgraded** to the leading
hypothesis, since it's the one untested lever that changes *what* the gradient says, not
*where* it lands.

**Reading it:** if the semantic-training ratio is **much lower** than random, the gradient moved
patches in directions the pooler ignores → P2 is necessary. If it's **comparable**, the gradient
is already reaching the decision path → P2 is redundant and P1 becomes the sole priority.

Uses existing checkpoints, no training, ~30 minutes.

---

### P5 — Re-frame the caption corpus gate

**The problem.** The current gate is "TF-IDF text → crash label, require AUC < 0.75." It
punishes **all** label-predictability. But there are two very different sources:

| | Source A — contamination | Source B — the actual goal |
|---|---|---|
| Origin | Teacher was told/guessed the outcome | Teacher observed real physical precursors |
| Wording | "collision occurs", "impact" | "gap decreasing", "brake lights on" |
| Grounded in pixels? | No | Yes |
| Want it? | **No** | **Yes — this is the distillation target** |

**Evidence V12's residual is mostly Source B:** 0/100 banned-word violations, and TF-IDF
coefficient inspection found the top features were `braking` (+0.957), `decreasing gap`,
`path closing` — all physical observables. V12 already removed the GT/blind branch, which was
the actual contamination *mechanism*. So its 0.764 is probably legitimate signal we suppressed.

**Corroborating:** V10 (AUC 0.964) → V12 (0.764) made B **worse** (0.0105 → 0.0189).
For a distillation objective that is backwards — you *want* the teacher signal to be predictive.

**Steps:**
1. **Drop the blanket AUC ceiling.** Keep the outcome-word ban (that's the real leakage control).
2. **Add a grounding gate instead** — reuse `score_val18_neutral.py`'s grounding machinery:
   is each claim verifiable from the frames?
3. **Make captions more kinematically explicit, not less** — quantitative where the frames
   support it ("gap closes ~15 m → ~4 m over 16 frames"), front-loaded to survive SigLIP's
   hard 64-token truncation.
4. **Let B-shuffle (P6) adjudicate.** It empirically separates Source A from Source B: if
   captions are mostly label-copies, shuffling within class costs nothing; if they carry real
   clip-specific detail, shuffling destroys it.

---

### P6 — B-shuffle control

Permute captions among clips **of the same class**. Preserves any residual label signal,
destroys clip-specific semantics.

- **B-shuffle ≈ B** → language *content* contributes nothing; only the class signal matters.
  That's a clean, decisive finding and it reframes the whole thesis honestly.
- **B-shuffle ≪ B** → content genuinely matters, and the mechanism is real even if the current
  routing wastes it.

Needed for the writeup either way. Planned since W3, never run.

---

### P7 — Scale up the corpus

The scaling curve is the strongest *positive* evidence we have:

| Train fraction | Train clips | Retrieval@1 | × chance |
|---|---|---|---|
| 25% | 221 | 5.88% | 13× |
| 50% | 443 | 8.60% | 19× |
| 100% | ~886 | 14.03% | **31×** |

Still rising, ~n^0.63, **no flattening**. Extrapolating to the full 4,446-window pool
(~2.5× more) suggests ~24% retrieval — speculative from three points, but it's the only lever
that adds *information* rather than re-routing what we have.

**Steps:** caption the remaining ~2,685 windows with Qwen3-VL-32B (~$11 total vs ~$162 with
Gemini — verify with one logged call first, since our token estimate uses Gemini's vision
tokenizer), run the same grounding/QA gates, rebuild the pool, re-run the winning arm.

---

### P9 — If we scale, the loss must change

Full-bank InfoNCE is fine at 1,413 (one 1413×768 matvec per anchor — negligible next to a
~100 GFLOP ViT-L forward). The problem at 100k is **statistical, not computational**: the
softmax becomes extremely peaked and per-negative gradient vanishes.

Standard fixes — **don't scale negatives with dataset size**:
- **CLIP**: in-batch negatives only, fixed by batch size
- **MoCo**: momentum queue, fixed size (~65k)
- **SigLIP**: pairwise **sigmoid** loss, no global normalization — most attractive here since
  we already use SigLIP embeddings

---

## Suggested execution order

1. ~~**P3** (30 min, gate)~~ — **done 2026-08-16.** Gradient reaches the decision path;
   P2 downgraded, P1 upgraded (see P3's elaboration above).
2. **P1** (1-2 d) — now the leading hypothesis, and the highest-expected-value experiment
3. **P6** + **P4** — can run unattended alongside
4. **P7** → then **P9** if scaling works
5. **P5** — feeds the next captioning round
6. **P2** — demoted, not eliminated: worth a cheap ablation later if P1 also underperforms,
   since a mild routing effect (1.8×, not overwhelming) could still compound with a better
   training order
7. **P8**, **P10** — cheap ablations, fill gaps

**Standing rule:** every arm must match A1_1761's recipe except the one variable under test,
and final Private-677 scoring happens **once** per chosen configuration.
