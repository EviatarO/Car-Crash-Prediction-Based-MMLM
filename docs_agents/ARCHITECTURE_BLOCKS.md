# Architecture blocks — shapes, frozen status, equations

Written 2026-09-09. Referenced from `ARCHITECTURE.md`'s opening paragraph and its files
table, which have pointed here since before this file existed — written now, from the
same investigation that resolved most of the 2560-vs-2048 token question (see
`NEXT_LORA_PLACEMENT.md` and `EXPERIMENTS.md`'s recovery-family section). This is a fresh
document, not a reconstruction of a lost original — it does not use `§5b/7c`-style
section numbers a stale reference to it once implied.

Matches the diagram at `reports/figures/semsup_architecture_2026-07-21.png`, with the two
deliberate deviations `WEBSITE.md` already documents for the website's redrawn SVG version
(no separate "patch grid" box; teacher caption sits left of SigLIP). That diagram is stale
in two spots per `ARCHITECTURE.md`'s note: it shows semantic weight `0.3×` (actual default
`0.05`, later raised to `0.2` for the A1-failure-recovery run) and labels the loss "meaning
match" (describes cosine, not the current InfoNCE default) — this doc uses the current
values throughout.

## 1. Input → preprocessing

```
16 JPEG frames (native 1280×720, stride-4 sampled over ~60 source frames of the
                clip — see CODE_GUIDE.md's corrected "window" definition, NOT 16
                back-to-back frames)
        │
        ▼  preprocess_clip(vjepa, paths)  — e4_stageA_badas_open_eval.py
   vjepa.processor(videos=frames_np, return_tensors="pt")
        │  (AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2"),
        │   crop_size 256×256, do_center_crop=True, do_resize=True — MEASURED directly
        │   2026-09-09, not inferred: config `img_size=224` in badas_loader.py's
        │   VJEPAModel constructor is DEAD for this path, see below)
        ▼
   (1, 16, 3, 256, 256)   — (batch, T, C, H, W), confirmed by direct measurement
```

**⚠️ `img_size=224` vs `img_size=256` — resolved 2026-09-09.** `nexar-ai/BADAS-Open`'s
`badas_loader.py` constructs `VJEPAModel(model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
img_size=224, ...)`. That `img_size` value is used only by `VJEPAModel`'s *fallback*
albumentations transform (`get_transform_for_model`); the `.processor` attribute — the one
this project's `preprocess_clip` actually calls — is built by
`get_processor_for_model(model_name)`, which calls
`AutoVideoProcessor.from_pretrained(model_name)` with **no `img_size` argument at all**.
Loading that processor directly (public Meta model, not gated) and running it on 16 dummy
1280×720 frames measured `pixel_values_videos.shape == (1, 16, 3, 256, 256)` — **256, not
224**. This matches the BADAS-2.0 paper and the HF model card, not the GitHub inference
wrapper's constructor default; the `img_size: 224` value in `e4_stageA.yaml` is dead
config, confirmed doubly now (both "never read by our own scoring path" and "wouldn't be
256 anyway if it were read, per BADAS's own source").

## 2. Trunk — V-JEPA2 ViT-L (`nexar-ai/BADAS-Open` / `facebook/vjepa2-vitl-fpc16-256-ssv2`)

Config read directly via `AutoConfig.from_pretrained` 2026-09-09 (not previously in this
repo's docs from inference — from the actual HF config):

| field | value |
|---|---|
| `patch_size` | 16 |
| `tubelet_size` | 2 |
| `num_hidden_layers` (encoder) | 24 |
| `num_attention_heads` | 16 |
| `pred_num_hidden_layers` (SSL predictor stack) | 12 |
| `pred_hidden_size` | 384 |
| `pred_num_attention_heads` | 12 |
| `num_pooler_layers` | 3 |
| `mlp_ratio` | 4 |

**Trainable status**: frozen base weights everywhere; LoRA (r=16, α=32, dropout 0.05) on
`query,key,value` projections. Default target-module spec (changed 2026-09-08, see
`semsup_train.py --lora-target-modules`'s help): the encoder-only regex
`re:backbone\.encoder\.layer\.\d+\.attention\.(query|key|value)`, 72 adapters. The legacy
bare-substring form `query,key,value` matches those 72 **plus** 36 more on
`backbone.predictor.layer.{0-11}` — the 12-layer, 384-dim SSL latent-forecast stack above,
not on the classification path (15.8% of every historical arm's LoRA params landed there,
common-mode, cannot explain the A-vs-B gap).

```
(1, 16, 3, 256, 256)
        │  ViT-L encoder, 24 layers, LoRA on q/k/v
        ▼
patches: (1, P, 1024)     P = temporal_groups × spatial_patches_per_frame
                           = (16/tubelet_size) × (256/patch_size)²
                           = (16/2) × (256/16)²  =  8 × 256  =  2048   [derived, this doc]
```

**⚠️ Still open**: this project's own runtime hook (`semsup_common.py`'s pre-hook,
comment "verified 2026-08-13 on BADAS-Open") measured **P = 2560**, not the 2048 this
section derives from the stock config + measured input shape. 2560/2048 = 1.25, not a
clean relationship — ruling out both "the derivation has an arithmetic error" (checked
twice) and "it's a units/batch mixup" (2560 and 2048 are both plausible per-clip patch
counts, not off by a factor of 16 or similar). Two live hypotheses, needing an actual
pod-side forward pass to settle: (a) the 2026-08-13 measurement was taken under different
conditions than this section assumes (a different `img_size` code path, or a stale/
mislabeled tap point) and 2048 is correct; or (b) `badas_open.pth`'s fine-tuning changed
the effective patch grid away from the stock `facebook/vjepa2-vitl-fpc16-256-ssv2`
pretrained config this section reads (e.g. interpolated position embeddings at a
resolution other than 256×256) — only checkable by loading the real weights. Next pod
session: re-run `e4_badas_attention_bbox.py --list_modules` and additionally print
`patches.shape` right where `semsup_common.py`'s pre-hook fires.

## 3. Crash head (`temporal_processor` + `classifier`) — always frozen

```
patches: (1, P, 1024)
        │  temporal_processor — an "attentive probe": per BADAS-Open's own README/HF
        │  model card, 12 learned queries cross-attend over the P patch tokens, then a
        │  3-layer GELU MLP (per the HF card's stated head design)
        ▼
pooled: (1, 1024)          — the ENTIRE basis for the crash decision; everything a
        │                      semantic loss shapes OUTSIDE this vector's span is
        │  classifier          structurally invisible to it (see §5 below)
        ▼
logits: (1, 2)
        │  softmax(logits / temperature)   temperature = 2.0 for A0's published
        │                                   scorer (e4_stageA_badas_open_eval.py,
        │                                   BADAS's own apply_temperature_scaling
        │                                   default) ; temperature = 1.0 for every
        │                                   A1/B/recovery-family arm's own scorer
        │                                   (semsup_train.py, score_checkpoints_
        │                                   on_test.py) — see EXPERIMENTS.md /
        │                                   WEBSITE.md for why this is safe for
        │                                   AP/AUC/CM@0.5 (monotone transform) but
        │                                   NOT for Brier/ECE (project review §4.5)
        ▼
P(collision) ∈ [0,1]
```

**Frozen status**: `temporal_processor` + `classifier` are frozen in every arm by default
— no q/k/v LoRA substring matches those module names, so LoRA structurally cannot reach
them. `--unfreeze-head` (added for SemTest-200, 2026-08-26) opens both to training in a
separate optimizer param group at `lr * head_lr_mult`; `--head-lr-schedule constant`
(added 2026-08-29, still not the default) is required to make that movement real — the
default `cosine` schedule combined with the default `head_lr_mult=0.1` was measured to
move the head <0.05% relative magnitude over 200 steps (SemTest-200-v1's confound, still
not fixed by the defaults as of this writing).

## 4. Semantic branch (train-only, patch-tap — the existing design)

```
patches: (1, P, 1024)                      (SAME tensor as §2/§3 — no extra
        │                                    forward pass)
        │  ResamplerProjector (vjepa_reason.py) — NOT part of BADAS; written for
        │  this project's own e4 Stage-B V-JEPA2→Qwen bridge and reused here at
        │  num_queries=8, hidden_dim=256, n_heads=8, ffn_mult=2 (~1.25M params)
        ▼
(1, 8, Dt)          Dt = 768 (google/siglip-base-patch16-224's text-embedding dim)
        │  .mean(dim=1)  — averages the 8 query outputs into one vector
        ▼
(1, Dt)
        │  F.normalize
        ▼
pred: (1, Dt)  ──┐
                 │  InfoNCE (default) or cosine (legacy, degenerate optimum — see
                 │  semsup_train.py's --semantic-loss help) against a frozen
                 │  caption-bank of SigLIP text embeddings
                 ▼
            sem_loss_patch
```

**Trainable status**: Predictor is trained from scratch (or warm-started via
`--predictor-init`) every run; SigLIP text encoder is frozen throughout. Both are
**discarded at inference** — enforced by construction (`forward_clip`'s inference branch
never calls them), not a runtime flag.

**Equation**: `loss = crash_weight · CE(logits, label) + semantic_weight · sem_loss`,
where (as of 2026-09-08) `sem_loss = sem_patch_weight · sem_loss_patch +
sem_pooled_weight · sem_loss_pooled` (§5).

## 5. Semantic branch, pooled tap (`--sem-pooled-weight`, added 2026-09-08) — default off

The single architectural finding from the 2026-09-06 project review (§3.2): every prior
semantic arm attaches its loss to `patches` (§4) — the tensor **before** the frozen
`temporal_processor`'s fixed attention-pool. Whatever the semantic loss shapes outside
that pool's span is structurally invisible to the classifier, which reads only `pooled`
(§3). This makes "captions add no new information" and "the channel to the classifier is
too narrow" **observationally identical** under the patch-only design — every null this
project has collected (B-v1/v2/v3, P1, SemTest-200, the a1fail321 recovery family) cannot
distinguish them.

```
pooled: (1, 1024)     — SAME tensor §3 taps (badas._captured["pooled"], from the SAME
        │                forward_clip() call — zero extra GPU compute)
        │  pooled_head: LayerNorm(1024) → Linear(1024, Dt)   (~0.8M params, only
        │                                                       constructed when
        │                                                       --sem-pooled-weight > 0)
        ▼
(1, Dt)
        │  F.normalize
        ▼
pred_pooled: (1, Dt)  ──┐
                        │  InfoNCE, own learnable temperature (log_tau_pooled,
                        │  independent of the patch-tap's log_tau) against the
                        │  SAME frozen caption bank as §4
                        ▼
                  sem_loss_pooled
```

**Default OFF** (`--sem-pooled-weight 0.0`): at the default, `pooled_head` is never even
constructed — no new params, no new RNG draws, byte-identical to the pre-2026-09-08 code
path. **Not yet run** as of this writing (staged for the Phase D GPU runbook). Falsifiable
outcome: if enabling it moves test AP where the patch-tap design didn't, the channel was
too narrow, not empty, and language supervision is alive again; if it stays flat, "captions
add no information the model doesn't already have" wins **with a mechanism**, closing the
direction as a fourth independent negative rather than a fifth unexplained one.

## 6. Gradient-angle diagnostic (`--grad-cosine-every`, existing) and per-layer grouping
(`--per-layer-grads`, added 2026-09-08)

```
crash_loss.backward-able graph  ──┐
                                   │  torch.autograd.grad(loss, lora_params,
sem_loss_combined.backward-able ──┤     retain_graph=True, allow_unused=True)
graph                             │  — returns gradients WITHOUT accumulating into
                                   │    .grad, so training is bit-identical with the
                                   ▼    probe on or off
              g_crash, g_sem  (per-parameter lists, positionally aligned with
                                lora_params)
                    │
      ┌─────────────┴──────────────┐
      ▼ (existing)                 ▼ (--per-layer-grads, 2026-09-08 — SAME
 flatten ALL lora_params            g_crash/g_sem, no extra backward pass)
 into one vector each          group by encoder-layer index (regex on param
      │                        name) or "predictor_stack" bucket (36 adapters,
      ▼                        NOT comparable to encoder buckets — different
 cos(g_crash, g_sem)           shapes, 384-dim vs 1024-dim, per
 — the GLOBAL number           NEXT_LORA_PLACEMENT.md); "other" bucket, defensive
 reported throughout                 │
 EXPERIMENTS.md                      ▼
 (measured: -0.04 to +0.05,    per-layer cos(g_crash[layer], g_sem[layer])
  near-orthogonal, not         — answers whether the near-zero GLOBAL cosine
  opposed)                     hides per-layer structure (mildly-unrelated-
                                everywhere vs conflicting-in-some-layers-
                                cancelling-in-others), and where the crash
                                gradient concentrates at all (is LoRA placement
                                a lever) — see NEXT_LORA_PLACEMENT.md
```

**Not yet run** as of this writing (default off, `--grad-cosine-every > 0` required;
staged for Phase D). Cross-reference against per-layer **weight-change norm** before
reading a bare gradient-norm ranking as importance — a converged layer has small
gradients *because* it is already adapted, not because it doesn't matter.

## Cross-references

| Topic | Where the fuller story lives |
|---|---|
| Two sources of truth for AP (summary vs dump) | `WEBSITE.md` |
| Temperature convention and why it's safe for AP/AUC/CM but not Brier/ECE | `EXPERIMENTS.md` (recovery-family section), project review §4.5 |
| Full pairwise bootstrap CIs on the recovery family | `EXPERIMENTS.md` (recovery-family section) |
| LoRA targeting defect and the encoder-only regex fix | `NEXT_LORA_PLACEMENT.md` |
| Prior work on LoRA placement (AdaLoRA, Surgical Fine-Tuning, etc.) | `NEXT_LORA_PLACEMENT.md` |
| Why the crash head is frozen / `--unfreeze-head`'s confound | `PROJECT_STATE.md`, `ARCHITECTURE.md`'s "Crash-head unfreezing" section |
