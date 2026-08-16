# Architecture — block-by-block reference

Companion to `reports/figures/semsup_architecture_2026-07-21.png`. One section per block
in that diagram: purpose, tensor shapes, internal architecture, frozen/trainable status,
and the governing equations.

**Two corrections to the figure itself** (it was generated 2026-07-21 and is stale):

1. It shows `total loss = crash loss + 0.3 × semantic loss`. The weight is now **λ = 0.05**,
   because the semantic loss was replaced (cosine → InfoNCE) and the two operate on
   different scales — see §7.
2. It labels the semantic loss "meaning match", which describes the cosine formulation.
   The current default for Stage B is contrastive.

**Confidence marks used below:** ✅ verified in code or by measurement this session ·
⚠️ documented but not independently confirmed.

---

## Notation

| Symbol | Meaning | Value |
|---|---|---|
| `T` | frames per clip | 16 |
| `P` | patch tokens out of the encoder | 2560 ⚠️ |
| `D` | encoder width | 1024 ✅ |
| `d` | Predictor latent width | 256 ✅ |
| `Q` | Predictor learned queries | 8 ✅ |
| `Dt` | SigLIP text embedding dim | 768 ✅ |
| `N` | caption-bank size (pool size) | 1413 train / 348 val ✅ |
| `λ` | semantic loss weight | 0.05 ✅ |
| `τ` | InfoNCE temperature (learned) | init 0.07 → 0.0576 ✅ |

---

## 0. Input — 16 dashcam frames

**Purpose.** One ~2 s temporal window ending at the prediction horizon (0.5 / 1.0 / 1.5 s
before the event for positives; midpoint offsets `MID-10 / MID-4 / MID-8` for negatives,
which have no event to count down to).

**Shape.** 16 JPEGs at native 1280×720 on disk → processed to `(1, 16, 3, H, W)`.

**Architecture.** None — data loading plus BADAS's own `AutoVideoProcessor`
(squash-resize + ImageNet normalisation). Frames are stored HiRes deliberately: BADAS
resizes internally, so pre-squashed frames would incur a lossy double-resize.

**Frozen.** N/A.

✅ **Discrepancy resolved 2026-08-13/15.** Directly measured on the pod (a purpose-built
inspection script hooking the pooler with both a pre-hook and a post-hook, run against one
real preprocessed clip): patches tensor shape is `(1, 2560, 1024)`. This confirms the
non-square 256×320 resize path, **not** the 224×224 figure `ARCHITECTURE.md` records — that
statement in `ARCHITECTURE.md` is the stale one and should be corrected to match.

---

## 1. V-JEPA2 vision encoder (ViT-L)

**Purpose.** Turn the clip into a spatiotemporal patch representation. V-JEPA2 is
self-supervised and learns dynamics (closing speed, relative motion) directly in latent
space — which is precisely the information the earlier InternVL3.5 design lost on its way
to the language model.

**Input / output.** `(1, 16, 3, H, W)` → `(P, D) = (2560, 1024)` ⚠️

**Architecture.** ✅ Confirmed from the runtime module dump:
- `VJEPA2PatchEmbeddings3D` — a single `Conv3d` doing joint spatial+temporal patchification
  (tubelet), so temporal structure enters at the embedding, not by frame-stacking.
- **24 × `VJEPA2Layer`**, each: `LayerNorm → VJEPA2RopeAttention → DropPath → LayerNorm →
  VJEPA2MLP`. Attention uses **rotary position embeddings**, and exposes the separate
  `query` / `key` / `value` Linears that LoRA attaches to.
- Width `D = 1024`, pre-norm residual blocks.

Per layer, with `x ∈ R^{P×D}`:

```
x ← x + DropPath( RoPEAttn( LN₁(x) ) )
x ← x + DropPath( MLP( LN₂(x) ) )
```

**Frozen.** ✅ **Frozen base weights, adapted by LoRA.** `get_peft_model` freezes every
base parameter; only injected LoRA matrices train.

⚠️ **Also present, and not on the classification path:** `backbone.predictor` — a further
**12-layer** `VJEPA2Predictor` (width 384), the latent-forecasting head used during
self-supervised pretraining. The classifier consumes encoder output, not this. It matters
because substring LoRA targeting hits it too — see §2.

---

## 2. LoRA adapters — the *only* trained part of the vision path

**Purpose.** Adapt a 334M-parameter trunk on ~1.4–3.6k training windows without
overfitting, by constraining updates to a low-rank subspace.

**Architecture.** For each targeted Linear `W₀ ∈ R^{out×in}`, LoRA adds a rank-`r`
residual and leaves `W₀` untouched:

```
h = W₀x + (α / r) · B A x ,    A ∈ R^{r×in},  B ∈ R^{out×r},  r = 16, α = 32
```

`A` is Gauss-initialised, `B` is zero-initialised — so at step 0 the model is *exactly*
BADAS-Open. Dropout 0.05 on the LoRA branch. Scaling α/r = 2.

**Where it attaches.** ✅ Measured against the real module dump:

| Target spec | Linears matched | Encoder | Predictor | Params |
|---|---|---|---|---|
| `query,key,value` (legacy substrings) | 108 | 72 | **36** | 2,801,664 |
| `re:backbone\.encoder\.layer\.\d+\.attention\.(query\|key\|value)` | 72 | 72 | 0 | **2,359,296** |

Encoder-only arithmetic: `24 layers × 3 projections × 2 matrices × r=16 × D=1024 = 2,359,296`.
The predictor stack adds `12 × 3 × 2 × 16 × 384 = 442,368` — **15.8%** of the legacy total,
sitting on a module that is not on the classification path. Every run to date used the
legacy form.

**Frozen.** ✅ **Trainable** — and the only trainable thing in the vision path.
0.71% of the 334,355,842-parameter model (encoder-only) / 0.84% (legacy).

✅ **Verified:** zero LoRA adapters land on `temporal_processor.*` or `classifier.*` —
those names contain no `query`/`key`/`value` substring. LoRA structurally cannot touch the
crash head.

---

## 3. Crash head — `temporal_processor` + `classifier`

**Purpose.** BADAS-Open's own pretrained collision classifier. Reused *unchanged* so
results stay comparable to the published A0 baseline (AP 0.853).

**Input / output.** `(P, D) = (2560, 1024)` → `(1, 2)` logits.

**Architecture.** ✅ From the module dump, and ✅ shapes now confirmed by direct measurement
(2026-08-13/15, see below):

- `temporal_processor` (`AttentionProcessor`) — an **attentive-probe pool**: one
  `nn.MultiheadAttention` + `LayerNorm`. A learned query attends over the `P` patch tokens
  and collapses them to a single vector. This is the standard V-JEPA2/DINOv2 evaluation
  recipe, not mean-pooling.
  **Measured I/O**, via a `forward_pre_hook` (captures the pooler's input) and a
  `forward_hook` (captures its output) on the same real clip: `(1, 2560, 1024) → (1, 1024)`
  — a **2560×  compression, 0.04% of values retained**. All 2560 spatiotemporal patch tokens
  collapse to a single 1024-d vector, and that vector is the *entire* basis for the crash
  decision — nothing downstream of it ever sees the patch grid again. This is the exact
  bottleneck the §5b tap-point experiments below test.
- `classifier` (`MLPHead`) — a `Sequential` of exactly 9 modules:

```
Linear → GELU → LayerNorm → Dropout → Linear → GELU → LayerNorm → Dropout → Linear
```

i.e. a **3-layer MLP** (two hidden blocks + output), ending in 2 logits.

Probability:

```
p(collision) = softmax(z)₁ ,   z ∈ R²
```

⚠️ **Temperature inconsistency.** The A0 scorer applies `z/2.0` before softmax
(`e4_stageA_badas_open_eval.py:182`); `semsup_train.py` does not. Softmax temperature is
strictly monotone, so **AP and AUC are unaffected** and the 0.853-vs-0.900 comparison is
valid. But Brier, ECE, F1@0.5, precision, recall, specificity and the optimal threshold
all shift — those are **not** comparable across A0 and A1/B as currently stored.

**Frozen.** ✅ **Fully frozen, always** — in A0, A1, B1 and B alike.

> **Why this matters more than it looks.** The head was fitted to BADAS's *original*
> feature distribution. As LoRA moves the trunk's features, the decision boundary cannot
> follow. The only way to reduce loss is to bend the representation to fit a fixed
> boundary. On an all-failure training set that has a degenerate solution — invert the
> features — which is exactly what produced **AUC = 0.163** in the `A1_587` run. Mixing in
> correctly-classified data removed the degenerate optimum but not the constraint.

---

## 4. Crash loss

**Purpose.** The primary objective — the only one present at all in arm A1.

```
L_crash = CE(z, y) = −log softmax(z)_y ,    y ∈ {0, 1}
```

Standard cross-entropy over 2 logits, computed per window (batch size 1), averaged over
the gradient-accumulation window of 8.

**Frozen.** N/A — loss term.

---

## 5. Predictor — `ResamplerProjector` (train-only)

**Purpose.** Map the patch grid to a single vector living in SigLIP's text space, so the
trunk can be asked to make its representation *describable*. A Perceiver/Q-Former-lite
resampler: fixed-size output regardless of `P`.

**Input / output.** ✅ Shape trace verified by execution:

```
(1, P, 1024)  →  ResamplerProjector  →  (1, 8, 768)  →  mean(dim=1)  →  (1, 768)
```

The mean over the `Q = 8` query tokens is what makes it comparable to the single SigLIP
caption vector. (It was `.squeeze(1)` back when `Q = 1`; using `.squeeze` now would be a
silent shape bug.)

**Architecture.** ✅ `num_queries=8, hidden_dim=256, n_heads=8, ffn_mult=2, dropout=0.1`.
Patches are projected **down** to `d = 256` (Flamingo/BLIP-2 style) to keep the block small
relative to the data:

```
kv = LN_kv( W_in · patches )                  # (1, P, 256)
q  = queries                                   # (8, 256), learned
q  = q + CrossAttn( LN_q(q), kv, kv )          # queries read the patch grid
q  = q + SelfAttn( LN_s(q) )                   # queries exchange information
q  = q + FFN( LN_f(q) )                        # 256 → 512 → 256, GELU
out = W_out( LN_out(q) )                       # (8, 768)
```

Measured parameter breakdown — **1,253,632 total**:

| Sub-block | Type | Params |
|---|---|---|
| `in_proj` | Linear 1024→256 | 262,400 |
| `cross` | MultiheadAttention(256, 8 heads) | 263,168 |
| `selfa` | MultiheadAttention(256, 8 heads) | 263,168 |
| `ffn` | Linear 256→512 → GELU → Dropout → Linear 512→256 | 262,912 |
| `out` | Linear 256→768 | 197,376 |
| `queries` | `nn.Parameter(8, 256)` | 2,048 |
| 5 × LayerNorm | — | 2,560 |

The self-attention block is **conditionally constructed** (`use_selfattn = num_queries > 1`):
at `Q = 1` softmax over a single key is identically 1.0, so the block degenerates to a plain
affine map — it is skipped rather than carrying ~1M dead parameters.

**Frozen.** ✅ **Trainable**, and **discarded entirely at inference.** Not constructed at
all when `--semantic-weight 0.0` (arm A1), which is why A1 never reads the caption field.

**Gradient path.** ✅ Verified intact. BADAS may run fp16 while the Predictor is fp32; the
boundary cast is `.to(dtype=torch.float32)`, which is differentiable, and the patch tap
hook stores `args[0]` **without** `.detach()`. So the semantic loss really does reach the
LoRA weights — it is not a detached side-branch.

**Warm-starting (`--predictor-init`).** ✅ The written experimental plan
(`2026-07-07_Plan Semantic-Supervision...`, lines 113/122-129/141) mandates a specific run
order: a standalone "B1" probe first (frozen trunk, frozen SigLIP, train **only** the
Predictor against the InfoNCE loss), then warm-start Stage B's Predictor from that
checkpoint via `--predictor-init <path>` before training the crash+semantic arm. **This was
not followed for the first Stage-B runs on the 1,761-window pool** (both the V10-corpus and
first V12-corpus runs used `predictor_init: null`, i.e. a randomly-initialized Predictor
trained from scratch alongside the LoRA). It was corrected in the run following it, warm-
started from a B1 probe trained on the corrected corpus. Note the plan does **not** ask for
the Predictor to be frozen after warm-starting — it keeps training jointly in Stage B either
way (§8's "Trainable in arm B" figure already includes it); no `--freeze-predictor` flag
exists.

---

## 5b. Alternate semantic-loss tap points — `--tap {patches, pooled, meanpool}`

**Purpose.** The default Predictor (above) reads from the pooler's **input** — the full
`(P, D)` patch grid — which the crash classifier never sees directly (§3). Whether a
semantic gradient applied there can shape the classifier's actual decision was an open
question; these tap points test it directly. Implemented in the standalone B1 probe
(`semsup_b1_probe.py --tap ...`), not (yet) in the joint Stage-B trainer.

**The three taps**, all evaluated against the identical 221-clip held-out set used for the
patches-tap retrieval number in §7b (chance level = 1/221 ≈ 0.45%):

| Tap | Predictor reads from | Shape | Predictor architecture | Params | Clip retrieval@1 | × chance |
|---|---|---|---|---|---|---|
| `patches` (default, §5 above) | pooler **input** | `(P, 1024)` | `ResamplerProjector` (§5) | 1,253,632 | 14.03% | **31×** |
| `pooled` | pooler **output** — the classifier's actual input | `(1024,)` | `_VectorMLP` (below) | 1,181,440 | 9.95% | **22×** |
| `meanpool` (control) | uniform mean over the same 2560 patch tokens | `(1024,)` | `_VectorMLP` (below) | 1,181,440 | 8.14% | **18×** |

**`_VectorMLP`** (new, `semsup_b1_probe.py`): a plain 3-layer MLP, `1024 → 512 → 512 → 768`
with GELU activations, used for both single-vector taps. `ResamplerProjector`'s
cross-attention is mathematically a no-op on a single input token (softmax over one key —
the same dead-weight trap documented for the old `num_queries=1` config in §5), so it is not
reused here.

**Reading the result.** Caption-recoverable information **survives** the 2560× pooling
bottleneck comfortably (22× chance, not collapsed to ~1×), and the `meanpool` control (18×)
is statistically indistinguishable from `pooled` (22×) at n=221 (±1.9pp) — so the
crash-tuned attention is **not** specifically discarding caption-relevant directions
relative to uniform pooling. This partially refutes the strong form of the "bypass"
hypothesis (that semantic information lives entirely outside what the classifier can read)
and gives the `pooled` tap a confirmed, learnable target for a future Stage-B run that
attaches the semantic loss there instead of at the patch grid — which would make it
structurally impossible for the semantic gradient to shape anything the classifier doesn't
see. Whether the *gradient produced by patches-tap training* actually reaches the pooled
representation (as opposed to merely "the information exists there") is a distinct,
still-open question — see the "not yet measured" note in §7c.

---

## 6. SigLIP text encoder — the supervision target

**Purpose.** Convert a teacher caption into a fixed vector that defines "what this clip
means". Chosen because it is a strong contrastively-trained text tower whose space is
already shaped for image-text alignment.

**Input / output.** caption string → `(1, 768)`, L2-normalised.

**Architecture.** `google/siglip-base-patch16-224`, text tower only
(`get_text_features`). Standard transformer text encoder; the vision tower is never used.

**Frozen.** ✅ **Fully frozen** — `eval()` and `requires_grad = False` on every parameter,
every call wrapped in `no_grad`.

⚠️ **Hard 64-token truncation.** `max_length=64, truncation=True`. Current captions measure
max 36 words (p95 = 28), so nothing is being cut — but a longer prompt design would
silently lose its tail, and in an earlier draft that tail was the outcome clause.

> **The frozen-ness is load-bearing, not incidental.** Because no gradient ever flows to
> the text side, all `N` caption embeddings are constants and can be precomputed **once**
> into a bank (~4 MB for 1,413×768 fp32). That is what makes contrastive learning possible
> despite the trunk running at batch size 1 — see §7.

---

## 7. Semantic loss

Two implementations; `infonce` is current.

### 7a. Cosine (original — **proven degenerate**, retained only for reproducibility)

```
L_sem = 1 − cos( p, t )
```

with `p` the Predictor output and `t` the caption embedding. For a predictor that ignores
the video entirely and emits a constant, the optimal constant is `E[t]/‖E[t]‖`, giving

```
L_sem* = 1 − ‖E[t]‖
```

Measured `‖E[t]‖ = 0.8648`. The real B1 run reached 0.8655 mean cosine — beating the
video-blind floor by **0.53% of the available range**, with retrieval@1 exactly at chance.
The objective was satisfiable without looking at the video.

### 7b. InfoNCE over a frozen bank (current)

For anchor `i` against the full precomputed bank `{t_j}`:

```
L_sem = − log [ exp(pᵢ·tᵢ / τ) / Σ_{j ∈ 𝒩(i)} exp(pᵢ·tⱼ / τ) ]
```

- `τ = exp(log τ)`, **learned**, clamped to [1e-2, 1.0]; must be in the optimiser's
  parameter list or it silently stays at its initial value. ✅ Verified it moves:
  0.07 → 0.0576.
- `𝒩(i)` **excludes other TTE windows of the same `video_id`** (sibling masking). Those
  have near-duplicate captions, so scoring them as negatives punishes a correct near-match.
- Chance level is `log N` — 7.25 nats for the 1,413-row train bank, 5.85 for val.

Why this fixes 7a: the shared mean direction appears in both numerator and denominator and
**cancels**, so the collapse solution scores at chance instead of getting a free ride.

✅ Measured effect: retrieval@1 went from exactly chance (cosine) to **32× chance** at row
level and **24× chance** at clip level.

**Frozen.** N/A — loss term. Note `λ` does **not** transfer between 7a and 7b: cosine's
magnitude is ~0.13, InfoNCE's is ~log N ≈ 5–7. Reusing the old λ = 0.3 would over-weight the
semantic term by roughly 20–40×.

---

## 7c. Gradient-angle diagnostic (`--grad-cosine-every`)

**Purpose.** §5b established that caption-recoverable information reaches the pooled
representation. It does **not** establish whether the semantic loss's gradient, computed at
the patch grid, pulls the LoRA trunk toward or away from the crash objective. This
diagnostic measures that directly rather than inferring it from AP deltas.

**Mechanism.** ✅ Every `N`th window (default `N=8`), computes `∂L_crash/∂θ` and
`∂L_sem/∂θ` **separately** via `torch.autograd.grad()` — restricted to `lora_params`, the
LoRA weights captured **before** the Predictor's parameters are appended to the optimizer's
trainable list. This restriction matters: the Predictor sits only on the semantic path, so
its `∂L_crash/∂θ_predictor` is identically zero, which would drag any cosine computed over
the full trainable set toward 0 for a reason having nothing to do with gradient conflict.
`torch.autograd.grad()` does **not** populate `.grad` or touch optimizer state, so the probe
is bit-identical to having it off — pure observation, verified by construction, not just
claimed. Logged per epoch: `grad_cos_mean` (mean cosine of the two gradients),
`grad_cos_frac_neg` (fraction of sampled steps where they point in opposing directions),
`grad_norm_crash`, `grad_norm_sem`.

**Measured, first fully-corrected Stage-B run** (warm-started predictor + per-group
clipping — see §8's clipping note):

| Epoch | cos(crash, sem) | conflicting steps | λ·‖g_sem‖ / ‖g_crash‖ |
|---|---|---|---|
| 1 | +0.0165 | 45.2% | 0.048 |
| 4 | −0.0007 | 49.2% | 0.089 |
| 8 | −0.0244 | 55.9% | 0.089 |

Reading: the two gradients start **near-orthogonal with a mild positive lean** (cos≈0.02,
well above the ~0.0006 expected from two random vectors in this ~2.36M-dim space, but far
from aligned) and drift toward **mild opposition** by epoch 8, with the fraction of
outright-conflicting steps crossing 50%. Simultaneously the semantic term's *relative*
magnitude after λ-weighting roughly doubles (0.048 → 0.089) as the crash loss saturates —
so the semantic gradient becomes both louder and more adversarial later in training, which
is offered as a mechanistic account of why later checkpoints in this run underperformed
earlier ones (a pattern that repeated, more severely, in an unplanned 12-epoch extension of
the same run: cos continued oscillating between −0.04 and +0.01, the ratio spiked to ~0.19
as `‖g_crash‖` collapsed toward zero, and `train_val_gap` climbed from 0.63 to 1.04 —
overfitting, not further learning).

✅ **Measured 2026-08-16** (`student_training/scripts/p3_delta_patches_vs_pooled.py`,
40 held-out clips, A1_1761 epoch 4 vs. the fully-corrected Stage-B run's epoch 2): loaded
both checkpoints' LoRA weights on the same frozen base, captured `patches` and `pooled` for
the same 40 clips under each, and compared `‖Δpooled‖ / ‖Δpatches‖` against the same ratio
for a random perturbation of the patch grid with equal norm.

| | mean ratio (‖Δpooled‖ / ‖Δpatches‖) |
|---|---|
| Real (A1 → B weight difference) | **0.00341** |
| Random-noise control (same norm) | 0.00186 |

The real ratio is ~1.8× the random-noise baseline — higher, but not dramatically so, and
this is a single point estimate over 40 clips with no confidence interval computed. Read
together with §5b's pooled-tap probe (22× chance, comparable to `meanpool`'s 18×), this is a
second, independent line of evidence that the weight difference between the crash-only and
crash+semantic checkpoints is **not** being preferentially routed into directions the pooler
discards — if anything it leans mildly the other way. **This weakens the "bypass" framing
further**: the semantic signal's influence on the trunk does appear to reach the
classifier-relevant representation, at least to a comparable-or-greater degree than a random
weight perturbation would. Combined with §7c's own finding (cos≈0, drifting mildly negative,
not a bypass signature), the more likely account of B's underperformance is **near-orthogonal
or mildly conflicting objectives that already reach the decision path**, not a routing
problem — which reprioritizes the two-stage training design (P1 in
`outputs/e4_vjepa_reason/problems_and_solutions_2026-08-15.md` — pretrain the semantic
branch, then fine-tune on crash) over the §5b pooled-tap architecture change as the more
promising next step.

---

## 8. Total objective

```
L = L_crash + λ · L_sem ,    λ = 0.05
```

- **A1 (control):** `λ = 0`. No Predictor is constructed; the caption field is never read.
- **B (treatment):** `λ = 0.05`, chosen to land the semantic term at roughly a third of the
  crash-loss magnitude given InfoNCE's scale.

**Optimisation.** AdamW, lr 2e-4, batch size 1 with gradient accumulation 8 (effective
batch 8, ~176 optimiser steps per epoch at the 1,761 pool). No LR schedule currently
(a separate `--lr-schedule cosine` option exists and was tested in a rejected recipe
variant — see DECISIONS.md — but is not the default).

**Gradient clipping — two modes** (`--clip-grad-per-group`, default off): ✅
- **Default (off):** ONE `clip_grad_norm_(trainable, 1.0)` over every trainable parameter
  combined. In arm A1, `trainable` = LoRA only. In arm B, `trainable` = LoRA + Predictor +
  `log τ`, all sharing the same budget of 1.0.
- **`--clip-grad-per-group` (on):** LoRA and the semantic branch (Predictor + `log τ`) are
  clipped against **separate** budgets of 1.0 each.

This distinction is not cosmetic: under the default mode, a large early Predictor gradient
can inflate the *combined* norm and scale down the LoRA trunk's effective update below what
arm A1 receives for an identical crash loss — a confound unrelated to `λ` that makes A1 and
B not strictly comparable. Every run before this fix used the default (shared-budget) mode;
the fix is applied starting with the first fully-corrected run cited in §7c.

---

## 9. Inference path

```
16 frames → V-JEPA2 ViT-L + merged LoRA → crash head → p(collision)
```

The Predictor and SigLIP are **not loaded**. Language contributes zero parameters, zero
latency and zero dependencies at run time — the entire point of the design. LoRA can also
be merged into the base weights (`W₀ + (α/r)BA`), leaving a model architecturally identical
to stock BADAS-Open.

---

## Summary — what trains and what doesn't

| Block | Params | Status | At inference |
|---|---|---|---|
| Input / preprocessing | — | — | present |
| V-JEPA2 ViT-L (base weights) | 334.4 M | **frozen** | present |
| LoRA adapters | 2.36 M encoder-only (2.80 M legacy) | **trainable** | merged in |
| `backbone.predictor` (V-JEPA2 SSL head) | — | frozen; ⚠️ legacy LoRA hit 36 of its Linears | not on class. path |
| Crash head (`temporal_processor` + `classifier`) | — | **frozen, always** | present |
| Predictor (`ResamplerProjector`) | 1.25 M | **trainable** | **discarded** |
| SigLIP text encoder | ~110 M (base) | **frozen** | **discarded** |
| InfoNCE temperature `log τ` | 1 | **trainable** | discarded |

**Trainable in arm B:** LoRA (2.36 M) + Predictor (1.25 M) + `log τ` (1) ≈ **3.61 M**.
**Trainable in arm A1:** LoRA only ≈ **2.36 M**.

⚠️ `log τ` is *not* checkpointed — a resumed InfoNCE run silently restarts it at
`--infonce-tau-init` while `--optimizer-init` restores stale Adam moments for that slot.

**Note on the table above:** the Predictor row (1.25 M, `ResamplerProjector`) is the
default `patches`-tap architecture used by the joint Stage-B trainer. The standalone B1
probe's `pooled`/`meanpool` taps (§5b) use a different, smaller `_VectorMLP` (1.18 M) —
not yet wired into Stage B, so arm B's live trainable-param count is still governed by this
table as written.
