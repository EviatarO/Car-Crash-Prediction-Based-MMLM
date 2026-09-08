# Next stage — LoRA placement, and what it can / cannot buy

Written 2026-09-03. Trigger: "who said LoRA is spread in the right place? can we change
their position?" Everything below is either verified in this repo or cited; the
confidence level is stated where it matters.

## Three findings that came out of auditing the config

1. **15.8% of LoRA capacity is wasted in every arm.** All arms except `a1_v2_full` used
   the plain substring `--lora-target-modules query,key,value`, which matches **108**
   Linear layers, not the intended 72: 72 in `backbone.encoder.layer.{0-23}` (wanted) plus
   36 in `backbone.predictor.layer.{0-11}` — V-JEPA2's SSL latent-forecast head, not in the
   classification path. That is 442,368 of 2,801,664 params (r=16, α=32).
   **It is common-mode**: A1 and every B arm carry it identically, so it shifts them
   equally and **cannot explain the A-vs-B gap**. Do not retire the negative result on it.
   Fix: copy `a1_v2_full`'s regex
   `re:backbone\.encoder\.layer\.\d+\.attention\.(query|key|value)`.

2. **Preprocessing is clean — no train/eval mismatch.** `preprocess_clip` delegates to
   BADAS's own HF `AutoVideoProcessor`; the yaml's `preprocess.img_size: 224` and norm
   values are **dead config** (only `num_frames` is read, for a warning). Training and
   evaluation import the *same* `preprocess_clip` from `e4_stageA_badas_open_eval`, so the
   most dangerous class of bug is ruled out. The stale yaml keys should be deleted or
   commented as unused.

3. **The 2560-token layout — substantially resolved 2026-09-09, locally, without a pod.**
   Contrary to the earlier note here, `nexar-ai/BADAS-Open`'s HF repo IS readable without
   gating (confirmed: `huggingface_hub.list_repo_files`/`hf_hub_download` both succeed with
   no auth prompt). Read the actual source directly instead of guessing:

   - `badas_loader.py` constructs `VJEPAModel(model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
     img_size=224, ...)`. **But `img_size` never reaches the actual preprocessing path.**
     `VJEPAModel.load()` builds `self.processor = get_processor_for_model(self.model_name)`
     (in `src/utils/video.py`) which calls `AutoVideoProcessor.from_pretrained(model_name)`
     with **no `img_size` argument at all** — `img_size` is used only by the *fallback*
     `self.transform` path (`get_transform_for_model`), which our own `preprocess_clip` never
     exercises (it always uses `vjepa.processor`, same as BADAS's own `_predict_sliding_window`
     path via `processor(videos=frames_array, return_tensors="pt")`). `img_size=224` is
     confirmed **dead** for the actual scoring path, exactly as the yaml note already said —
     but now with the mechanism, not just the symptom.
   - Loaded `facebook/vjepa2-vitl-fpc16-256-ssv2`'s `AutoVideoProcessor` directly (public Meta
     model, no gating) and printed its config: `crop_size: {height:256, width:256},
     do_center_crop: true, size: {shortest_edge: 292}`. Ran it on 16 dummy 1280×720 frames
     (the same call shape `preprocess_clip` makes) and got **`pixel_values_videos.shape ==
     (1, 16, 3, 256, 256)`** — confirmed **256×256, not 224×224**, matching the BADAS-2.0 paper
     and the HF model card, not the GitHub wrapper's constructor default.
   - Loaded the model's `AutoConfig` and read the real numbers instead of guessing:
     **`patch_size=16, tubelet_size=2, num_hidden_layers=24, pred_hidden_size=384,
     pred_num_hidden_layers=12, num_pooler_layers=3`** — confirms the 24 encoder layers, the
     12 V-JEPA2-predictor-stack layers, and the predictor stack's 384-dim (vs the encoder's
     1024-dim) cited elsewhere in this doc, all from the actual config now, not inference.
   - With `patch_size=16, tubelet_size=2` on a `(1,16,3,256,256)` input, the arithmetic is
     forced: temporal groups = 16/2 = **8**; spatial patches/frame = (256/16)×(256/16) =
     16×16 = **256**; total = 8×256 = **2048** — exactly BADAS's own documented "2048 patches
     × 1024 dim". Read `EnhancedVideoClassifier._extract_video_features`'s V-JEPA2 branch
     (`src/train/video_training.py`) directly: it calls
     `self.backbone.get_vision_features(pixel_values_videos=x)` (or `.last_hidden_state`) with
     **no reshaping specific to V-JEPA2** — the `seq_len // frame_count` reshape a few lines
     away is gated behind `elif self.model_info['is_videomae']`, a different code path entirely.

   **⚠️ Still open, and this is now the interesting part**: this derivation gives **2048**, but
   this project's own runtime hook (`semsup_common.py`'s `_pre_hook`, per its comment "verified
   2026-08-13 on BADAS-Open: input (1, 2560, 1024)") measured **2560** on the actual pod, on the
   actual loaded `badas_open.pth` checkpoint. 2560/2048 = 1.25 — not a clean relationship. Two
   readings, and only a fresh pod-side measurement decides between them: (a) the earlier 2560
   measurement was taken under different conditions than assumed here (a different img_size
   path, a stale/mislabeled tap point) and 2048 is actually correct; or (b) BADAS's fine-tuning
   of `badas_open.pth` changed the effective patch grid away from the stock pretrained
   config this derivation assumes (e.g. interpolated position embeddings at a different
   resolution than `facebook/vjepa2-vitl-fpc16-256-ssv2`'s stock 256×256) — checkable only by
   loading the actual weights, which needs the pod. **Next pod session: re-run
   `e4_badas_attention_bbox.py --list_modules` (still the direct way to get the number that
   matters) and additionally print `patches.shape` right where `semsup_common.py`'s pre-hook
   fires, to settle 2048-vs-2560 for real** — this local investigation narrows the search space
   (settles img_size=256, rules out img_size=224 and the 10×(16×16) hypothesis, since the
   temporal factor is now confirmed pinned at 8 = 16/tubelet_size(2), not derivable as
   2560/256=10) but does not close the file.

## The diagnostic to add

Per-layer LoRA gradient norms. `semsup_train.py:922` already calls
`torch.autograd.grad(crash_loss, lora_params)`, which returns a **per-parameter** list and
then immediately flattens it. Grouping those same tensors by encoder-layer index before
reducing is nearly free — no extra backward pass.

It yields:
- **Where the crash gradient concentrates** → is placement even a lever?
- **Per-layer crash-vs-semantic cosine.** The headline "near-orthogonal (−0.04…+0.05)" is
  one average over 2.8M params across 24 layers. That average looks identical whether the
  objectives are mildly unrelated *everywhere*, or conflicting in some layers and agreeing
  in others and cancelling. Different mechanisms, different fixes — currently
  indistinguishable. **This is the strongest reason to run it.**
- **Confirms the 15.8%** from the run itself (predictor-head adapters should show ~zero
  crash gradient).

Comparability caveat: norms are comparable across the 24 encoder layers (identical q/k/v
shapes) but **not** against the predictor-head adapters (384-dim vs 1024-dim) — report
those separately. And gradient norm ≠ importance: a converged layer has small gradients
*because* it is already adapted. Log the per-layer **weight-change norm** alongside.

## The bound that limits the semantic side

On the a1fail321 recovery pool, the crash-only control (`a1cont`) and the semantic arms
land on the **same** held-out accuracy (39/61 = 0.6393). Those are the two endpoints of the
weight-separation spectrum — fully shared and fully separate. Any per-layer scheme that
routes the semantic gradient to a subset of layers interpolates between two endpoints that
already agree, so there is no room between them for a gain. **Treat per-layer work as a
route to a stronger champion (A1), not as a rescue of the semantic hypothesis.**

## Prior work — has anyone done this, and did it work?

| Work | What they did | Did it work? |
|---|---|---|
| **LoRA** — Hu et al., ICLR 2022 | Ablated *which* matrices to adapt at a fixed budget | Yes. Adapting **Wq+Wv** beat Wq alone at equal params; very low rank (r=1–4) often sufficed. Supports q/k/v being a reasonable default, and that budget *allocation* matters more than budget *size*. |
| **AdaLoRA** — Zhang et al., ICLR 2023 | Allocates **rank** across modules by an importance score (sensitivity ≈ \|w·∇w\|), SVD-parameterised, pruning unimportant directions **continuously during training** | Yes — gains over uniform LoRA, largest at low budgets. **The closest published answer to this question.** Note it uses sensitivity, not raw grad norm, and reallocates on a schedule rather than one-shot. |
| **Surgical Fine-Tuning** — Lee et al., ICLR 2023 | Tune only a *subset* of layers | Yes, and instructively: the best subset **depends on the shift type** — early layers for input-level shift, later layers for label-level shift. Sometimes beat full fine-tuning. Implies our answer is task-specific and must be measured, not copied. *(moderate confidence on the exact layer/shift mapping)* |
| **Fine-Tuning Distorts Pretrained Features** — Kumar et al., ICLR 2022 | Why fine-tuning can underperform linear probing OOD | Already cited in this project (SemTest-200 motivation). Relevant because our crash head is frozen while the trunk moves. |
| **ViTs Need Registers** — Darcet et al., ICLR 2024 | High-norm artifact tokens in background regions | Relevant to the heat map, not placement: **do not use raw token norm** as a saliency map — it will look confident and mean nothing. Use attention or projection onto `pooled`. |

**Method warning from AdaLoRA's design:** one-shot top-k selection using gradients measured
with LoRA on *all* layers is not self-consistent — remove the others and the landscape
shifts; quiet layers may absorb the load. AdaLoRA avoids this by reallocating continuously.
If we do a one-shot cut, validate it against the all-layers baseline rather than assuming.

## Recommended order

1. **Fix the targeting regex.** Free, strictly better, no measurement needed.
2. **Per-layer diagnostic on A1** (~1 epoch). Crash-only, so it gives the clean "where does
   the crash objective want capacity" profile. If the profile is uniform, placement is not
   a lever — stop here.
3. **Per-layer cosine on B-v3 or v12.** For mechanism only; the bound above says do not
   expect a fix from it.

## Open questions

- Is the crash gradient concentrated or uniform across the 24 layers? (2 answers it.)
- Does the near-zero global cosine hide per-layer structure? (3 answers it.)
- What is the real 2560 factorisation? (`--list_modules`, prerequisite for any heat map.)
- Should LoRA move off q/k/v into the MLP blocks? Not answerable from these runs — LoRA is
  only on q/k/v, so there is no gradient signal for modules that have no adapter. Would
  need a separate run with MLP targets.
