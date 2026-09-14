# Detection-Guided Auxiliary Supervision — Design Summary (2026-09-10 → 09-13)

Summary of a 72-hour design discussion that moved the semantic-supervision thread from
clip-level caption alignment to **detection-guided, object-level auxiliary supervision**.
The executable program is the father plan:
`~/.claude/plans/CCP based BADAS/2026-09-13_Father-Plan-Stage-AA-BB.md`
(Stage AA: detection-derived kinematic targets, vision-only; Stage BB: + per-object captions).

---

## I.1 Where it started

A code-reading guide (`docs_agents/CODE_GUIDE.md`) for `e4_vjepa_reason` and `semtest200_v2`,
and a review plan for five code changes (`~/.claude/plans/transient-purring-comet.md`). That
review produced the questions that drove everything below.

## I.2 Why clip-level caption supervision cannot work — the mechanism

The semantic branch is: 2048 patch tokens → `ResamplerProjector` (8 queries cross-attend all
tokens) → `.mean(dim=1)` → one 768-d vector → InfoNCE against one SigLIP caption vector.

1. **Position is destroyed before the loss.** Cross-attention collapses 2048 → 8, the mean
   collapses 8 → 1. After that, no gradient can reference *where* in the frame something is.
2. **The single global target is the real problem, not the mean.** Keeping 8 vectors against one
   target just makes them redundant copies.
3. **The correction is a diffuse smear.** Per example the patch-space update is one direction;
   ~160–1,400 examples constrain 2.6M activations. Gradient descent takes the minimum-norm
   solution: a faint change everywhere, never "the truck on the right."
4. **Scale alone is already falsified.** 160 (SemTest-200) and ~1,600 (pool1761) both lost.
5. **Explaining-away.** The Predictor (1.25M params, adjacent to the loss) and the LoRA trunk
   (24 layers upstream) are both trainable, so the optimizer can reduce `sem_loss` by moving the
   Predictor instead of the trunk. `semsup_train.py` adds Predictor params unconditionally —
   **freezing the Predictor has never been tested** (user's proposal).

This is a fourth, more fundamental explanation alongside the three already on record (corpus
scale vs CLIP-style alignment; SigLIP as bag-of-words; frozen crash head). The B1 probe
(caption info reaches `pooled`, ~24× chance retrieval), P3 (it moves the representation) and the
grad-cosine probe (≈0, not opposed) are all consistent with it: the signal arrives, but as a
global rotation with no spatial commitment.

## I.3 Corrections that came out of checking the record

| Belief | Record |
|---|---|
| "B1→B warm-start was run on V12" | It was run on **V10** (`b_1761_seq`). The V12 arm had no warm-start. |
| "Warm-start = frozen Predictor" | No. Every arm kept training the Predictor. Freeze is untested in all 7 arms. |
| "Token grid is 2560" | Config derivation gives **2048 = 8 × 16×16** (256×256 input, patch 16, tubelet 2); the runtime hook measured 2560. **Unresolved — gates everything spatial.** |
| "iFinder beats us / is comparable" | Not comparable (see I.8). **72 of iFinder's 100 Nexar eval videos are in our V12 training corpus.** |
| "The pool is 4,500 minus clips with short event−alert time" | The pool is **(1,500 − 18) × 3 = 4,446**; the 18 missing videos are the deliberately held-out `val_e3a` clips. |

## I.4 The mechanism that localizes — region masking

Build a boolean mask over the token grid for each tracked object, and feed the aux branch only
that object's tokens:

```
patches (2048, 1024)  ──►  crash head (ALL tokens, unchanged)
        │
        └──► patches[mask_i] ──► Predictor ──► loss vs object i's own target
```

`patches[m]` is a differentiable gather; its backward is a scatter. So
**∂loss_i/∂patch_j = 0 exactly for every token j outside object i's mask** — enforced by the
graph, not learned. Summing over objects is safe because their gradients land on (mostly)
disjoint tokens — unlike the 8-query mean, which shared input *and* target. The pixel→token map
is **pure arithmetic** fixed by the architecture (strided patch embedding), so masks need no
ViT forward pass.

**Do not draw boxes on the student's input frames.** Visual prompting is a real technique
(RedCircle ICCV'23, Set-of-Mark, ViP-LLaVA), but applying it **at training only** creates a
train/test input mismatch and a possible shortcut (box presence ↔ label). Masks live in the
*loss*, not the *data*: the student sees byte-identical frames, inference is unchanged.

## I.5 Detection stack decisions

| Decision | Reason |
|---|---|
| **Grounding DINO** detector (Apache 2.0) | Best accuracy, text-promptable. Prompts are **class-level** (recall); its relational grounding ("the closest car") is unreliable → narrowing to 5–8 comes from **geometry**, not the prompt. |
| **BoT-SORT** tracker (Roboflow Trackers) | G-DINO detects per frame with no identity. Tracks are required for per-object targets, looming rate and tube masks. |
| **Lane detector** — CLRerNet (CLRNet family) | Boxes are the wrong representation for lanes; a dedicated polyline model is needed. |
| **No SAM-2** | Masks vs boxes differ by less than one 16×16 cell. Not worth the cost. |
| Detect at **full frame rate** (~60 frames / 2 s), then subsample | At the student's 7.5 fps effective rate, IoU association degrades; at ~30 fps it is easy and kinematics are smoother. |
| **Keep only objects present in the last ~0.5 s** of the window (tolerant of dropout) | Objects that left earlier cannot be the collision partner. |
| `minimum_consecutive_frames=1` + bidirectional tracking | Default `min_hits=3` silently drops late-appearing objects — exactly the dangerous cut-ins. |

## I.6 The cut-in problem and the threat signal

The first threat formula `max(α,0)·(1−e)` **zeroed out cut-ins** (low looming, high
eccentricity). Corrected design uses two complementary signals:

- **Looming** (longitudinal): log-linear fit of apparent size, `log s(t) = α·t + β`,
  `TTC_long ≈ 1/α` — Lee's τ, no depth or calibration needed.
- **Crossing CLRerNet's lane lines** (lateral, user's refinement): fraction of the object's box
  width past the ego-lane polylines at its ground row, and its rate. An interval test fires as
  soon as *any* part of the car intrudes — collisions happen when bodies touch.

`threat = 1 / min(TTC_long, TTC_lat)`. Edge-truncated boxes use **height only** for looming.

## I.7 What would actually be learned — attribution and novelty

- **Numeric vs caption targets.** If the useful content is kinematics, regress on it directly
  (no SigLIP, no 64-token limit, no leakage risk). Language can only add what numbers lack:
  object category, scene context, relational structure. It may also *lose* information
  (SigLIP already failed the v12≈v12shuf control). → Hence **Stage AA = numeric targets**,
  **Stage BB = numeric + caption**, which tests additivity directly.
- **Weights `a_i`:** start uniform. Geometric (threat-based) second, as a separate ablation.
  Learned last, and only constrained (softmax / floor), or it becomes another explaining-away
  path. Never weight by "is the collision partner" — that leaks the label.
- **Novelty, honestly:** object-centric accident anticipation (DSA, ACCV'16) and
  detection→VLM reasoning (iFinder, SafeVL) are established. What is defensible: the rigorous
  negative-result methodology, data efficiency (A1 at 0.900 inside BADAS-2.0's 0.86–0.91 range
  with ~1.4k windows), and language as a **train-only** auxiliary with vision-only inference.
  If BB ≈ AA, "language adds nothing over structured geometric supervision" is a clean,
  publishable result.

## I.8 External work reviewed — iFinder (NeurIPS 2025)

Its Nexar row (Acc 62.0, F1 0.59) is not comparable to A1: **post-hoc** accident detection on
whole videos vs **anticipation**; **n=100** (±~10 pt CI) sampled from Nexar's *train* split;
**zero-shot** vs fine-tuned; accuracy vs AP (our comparable balanced accuracy is ~0.73–0.74).
Reusable idea only: per-object structured cues. Extractors should come from their own SOTA
sources, not iFinder.

## I.9 Facts verified this session that constrain the plan

| Fact | Consequence |
|---|---|
| **Train pool = 4,446 windows from 1,482 videos** = (1,500 − 18 held-out `val_e3a`) × 3. | Stage AA pool size. The 18 clips are clean held-out demo clips. |
| Positives: `TTE_0.5/1.0/1.5` (741 each). **Negatives: `MID-10/MID-4/MID-8`** (741 each) — offsets from clip midpoint, not TTEs. The 18 `val_e3a` negatives use the retired `TN_MIDPOINT` (offset 0). | Case analysis and horizon reporting differ by class; re-cut the 18 negatives. |
| 16-frame window = 61 raw frames at **stride 4**, ≈2 s. fps varies **28.9–30.6**. Raw indices not stored in the train manifest — reconstruct via the extractor formula. | Detection must use each video's real fps. |
| Raw MP4s are local: `…/Data-Centric-Crash-Prediction-Using-3LC-and-MViT/src/Nexar_DataSet/{train,test}` | Full-frame-rate detection is feasible. |
| ⚠️ **`a1_v2_full` (trained on all 4,446) scored AP 0.868 val-selected / 0.888 best — below A1's 0.900.** It bundled pool + cosine LR + dropout 0.10 + encoder-only LoRA, ran 6/12 epochs, with a resume bug. | **Moving to 4,446 is itself a risk.** Stage AA needs a matched crash-only control on the same pool and recipe. |
| **`a1cont`** (A1 warm-start, crash-only continuation): AP 0.8956; A1−a1cont ΔAP +0.0039, CI [−0.0039, +0.0117]. | Continued training alone does not raise AP. |
| On the 4,446 split, **val_ap saturates at ~0.98** and picked the wrong epoch (val-best ep3 → test 0.868; test-best ep6 → 0.888). | val_ap is a weak checkpoint selector at this scale. |
| **Public-test scores exist only for A0** (0.871). | A1 must be scored on public before any public comparison. |
| `--keep-top-k` default is **3** (never 5). A1_1761 actually kept 8. Selection = clip-level `val_ap`. | Answers "3 or 5". |
| `--per-layer-grads` **exists** (commit fcb91fb) but only runs when a semantic Predictor exists. Weight-change norm is not implemented. | Needs generalizing to any aux loss. |
| `ResamplerProjector` has **no `key_padding_mask`**. | Required for batching variable-size object token sets. |
| Website asserts **exactly 677 test rows** (private only). Shows TP/FN/FP/TN, P, R, F1, Acc, AP, AUC @0.5, ROC, CM, per-TTE, vs-baseline, score distribution, agreement. Brier/ECE/specificity computed but not rendered. No 18-clip demo. | Public results and a detection demo need site extensions. |
| ⚠️ **BADAS preprocessing (per config derivation): resize shortest edge 720→292, center-crop 256×256.** That keeps only original x ≈ 324–955 of 1280 (**~49% of frame width**) and y ≈ 44–676. | **Cars entering from the frame corners may be invisible to the student until they reach the central crop.** Needs verifying against the 2048-vs-2560 question. Possibly an independent explanation for missed cut-ins. |
