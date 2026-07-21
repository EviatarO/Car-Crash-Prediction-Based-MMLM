# Semantic-Supervision Experiment Summary

Tests whether a language-derived auxiliary loss (SigLIP caption-embedding alignment)
improves BADAS-Open's (V-JEPA2 ViT-L) crash-prediction representation, while keeping
inference vision-only. Staged: **A0** (frozen baseline) → **B1** (predictor-only probe,
diagnostic) → **A1** (crash-only LoRA control) → **B** (crash+semantic LoRA, treatment).

Update this table (and append a dated entry below) every time a stage is re-run or the
data scale changes.

## Status

| Stage | Data scale | Status | Key result |
|---|---|---|---|
| A0 | 677 clips (Private test) | **DONE** (2026-06-24) | AP=0.853, AUC=0.864 (within 0.86±0.03 target) |
| Module discovery | — | **DONE** | LoRA targets `query,key,value` under `backbone.encoder.layer.{0-23}`, zero overlap with crash head |
| Stage 0 (captions) | 267 rows / 89 clips | **PARTIAL** — rephrased from existing teacher `final_reasoning`, not fresh vision-captioning. Target ~4.5k (1500 clips × 3 TTE), not yet scaled |
| B1 (predictor-only probe) | 216 train / 51 val (267 total) | **DONE** (2026-07-21, real GPU run) | retrieval_top1_acc=0.0196 == chance/control (0.0196) — **no evidence of learned video-caption alignment at this scale** |
| A1 (crash-only LoRA control) | — | **NOT STARTED** — not even locally smoke-tested | — |
| B (crash+semantic LoRA treatment) | — | **NOT STARTED** — needs A1 first + B1's predictor checkpoint (now available) | — |

## B1 real run — 2026-07-21

**Setup:** 216 train / 51 val clips (clip-level split), BADAS + SigLIP fully frozen,
only the small `ResamplerProjector` predictor trains. Loss = `1 − cos(pred, SigLIP(caption))`.
Batch size 16, AdamW lr=1e-4, up to 100 epochs with early stopping (patience 15 on val_loss),
frozen features cached once up front (~122s), 3 best checkpoints kept by val_loss.

**Result:**
- Early-stopped at epoch 23; best checkpoint = epoch 8.
- Best: `val_loss=0.1345`, `mean_cosine=0.8655`, `retrieval_top1_acc=0.0196`.
- Collapse control (constant mean-caption-embedding baseline): `mean_cosine=0.8648`,
  `retrieval_top1_acc=0.0196` (= chance, 1/51).
- **Verdict: NO evidence beyond the constant-embedding baseline** — the predictor did not
  learn video-specific caption alignment detectable above chance at n=267.
- Diagnosis, not failure: `train_loss` fell steadily (0.48→0.11) while `val_loss` bottomed
  at epoch 8 and rose after — classic overfitting on a small caption set, not a training bug.
  SigLIP embeddings of 267 near-synonymous crash captions are anisotropic, which is exactly
  why the collapse control was added — without it, `mean_cosine≈0.87` alone would look like
  a positive result.
- **Interpretation:** this was an explicit plumbing check ("NOT meaningful science" per the
  runbook) at 267 examples — the mechanism (caching, early-stop, checkpointing, controls) is
  now verified correct end-to-end. Whether semantic supervision carries real signal is
  undetermined until re-tested at ~4.5k captions.
- Artifacts: `/workspace/semsup/b1/predictor_b1.pt` (best, epoch 8), `b1_metrics.json`
  (full per-epoch history + control), top-3 checkpoints `predictor_b1_ep{008,009,010}.pt`.
  `predictor_b1.pt` remains usable as Stage B's warm-start even though it learned little —
  B's own crash loss will dominate there.

## Infra notes (current as of 2026-07-21)

- Working repo on RunPod: **`/workspace/MMLM_AI`** (persistent network volume — NOT `/root`,
  which is wiped on every new pod).
- Frame data: `dataset/train` and `dataset/test` are **symlinks** into the pod's existing
  `/workspace/data/train_HiRes` and `/workspace/data/test_HiRes` — confirmed all 267/267
  needed training folders and all test folders already present there, so no manual data
  transfer was needed (avoided moving a prebuilt 780MB tar bundle).
- `/workspace` network volume was enlarged 24GB → 30GB mid-session after hitting a hard
  per-user quota (RunPod's `df -h` cluster-wide free space is misleading — the real per-pod
  cap is separate and much smaller). `/workspace/data/extracted_clips` (1.1G, a superseded
  raw-frame-index intermediate, confirmed unused by any current script) was deleted to free
  headroom.
- Checkpoints at `/workspace/Car-Crash-Prediction-Based-MMLM/outputs/checkpoints/` from
  earlier (unrelated) distillation experiments are **NOT backed up to HF Hub** except
  `e3a-epoch7-lora` and `e3b-ep3-lora`: `e2_lora_100clips` (4.4G), `e3a_lora_89clips` (832M),
  `e3b_lora_267clips` (704M) exist only on this pod. Not deleted — awaiting explicit
  confirmation they're disposable.
- Caption labels (`Caption_Train_All_Clips.jsonl`/`.xlsx`, small text/xlsx, not weights) are
  force-added to git despite the blanket `outputs/` gitignore, so they can be `git pull`ed
  onto a pod instead of transferred by hand.

## Next step
A1 (crash-only LoRA control) — local CPU smoke test first (not yet run even at small scale),
then a real pod run, before B.
