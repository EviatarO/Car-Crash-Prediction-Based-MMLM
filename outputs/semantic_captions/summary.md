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
| A1 (crash-only LoRA control) | 216 train / 51 val (267 total) | **DONE** (2026-07-23, real GPU run) | test_AP=0.865 (best ckpt, ep7) vs A0's 0.853 — **essentially flat, within noise** |
| B (crash+semantic LoRA treatment) | 216 train / 51 val (267 total) | **DONE** (2026-07-23, real GPU run) | val-selected test_AP=0.8574 — **slightly below A1 (0.8638)**, still above A0. 3.5× more variance across checkpoints than A1 |

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

## A1/B script hardening — 2026-07-23

Local CPU smoke test of `semsup_train.py` (shared by A1 and B — `--semantic-weight 0`
vs `>0`) surfaced two real bugs, both fixed and re-verified before any pod time was spent:

1. **`save_pretrained()` crashed on every checkpoint save.** `peft` auto-generates a
   model card before writing adapter weights, assuming the base model's `.config`
   supports `in` (a HF `PretrainedConfig`). BADAS's V-JEPA2 uses a plain `ModelArgs`
   dataclass instead, so this crashed before any weights were ever written. Fixed by
   skipping the (unneeded) model-card step.
2. **Test-set scoring silently used the LAST epoch's weights, not the best one.** The
   script tracked `best_epoch` but never reloaded its checkpoint before scoring —
   always scored whatever was left in memory post-training. Fixed: reloads via
   `set_peft_model_state_dict`, verified correct with a synthetic save/reload
   round-trip test (tiny dummy LoRA model, confirms reload picks the intended
   epoch's weights, not the last).

Also expanded to the full E3 metric table (confusion matrix, accuracy, precision,
recall, **specificity**, F1, F1@optimal, AP, AUC-ROC, **Brier**, **ECE**, per-TTE AP —
new pure-math module `metrics_core.py`, importable on the pod without
matplotlib/seaborn/pandas), and now keeps + scores the **top-3** checkpoints by
val_ap independently (not just one "best" pick), matching B1's convention. Found and
fixed one more bug this surfaced: degenerate (single-class val) runs wrote invalid
JSON (`NaN` instead of `null`) — caught by strict-parsing every output file.

Smoke test itself ran at `--limit 8` (only positive-class rows in file order, by
design — this exercises the single-class/`val_ap=nan` fallback path, not a
meaningful AP number). Confirms mechanics only.

## A1 real run — 2026-07-23

**Setup:** 216 train / 51 val clips (same split as B1), BADAS LoRA (r=16/α=32,
`query,key,value`) unfrozen, crash-only loss (CE), 8 epochs, grad-accum=8, lr=2e-4.
No semantic loss (`semantic_weight=0.0`). Top-3 checkpoints by val_ap each scored
on the full 677-clip Private test set.

**Result:**

| Checkpoint | val_ap (n=51) | test_AP | AUC | F1 | recall | specificity | ECE |
|---|---|---|---|---|---|---|---|
| ep8 (best val) | 0.9751 | 0.8638 | 0.8728 | 0.7971 | 0.8195 | 0.7640 | 0.1478 |
| ep7 | 0.9682 | **0.8647** | 0.8718 | 0.8021 | 0.8994 | 0.6578 | 0.1658 |
| ep1 | 0.9679 | 0.8600 | 0.8694 | 0.7932 | 0.8964 | 0.6372 | 0.1839 |

- A0 (frozen baseline) reference: AP=0.853, AUC=0.864.
- **All three checkpoints land within ~+0.01 AP of A0 — no meaningful movement.**
  Expected at this data scale: crash-only LoRA fine-tuning on 216 clips isn't enough
  signal to move a model already well-trained on the full Nexar dataset. The value of
  this run isn't the AP number itself — it sets the **control bar B must clear (or at
  minimum not fall below)** to claim the semantic loss helps.
- Best-val checkpoint (ep8) is NOT the best-test checkpoint (ep7) — expected given the
  51-clip val set's known gap from the 677-clip test distribution (val_ap 0.96-0.98 vs
  test_AP ~0.86 across all three).
- Optimal threshold is unstable across epochs (0.1116 / 0.4592 / 0.2431) and ECE is
  fairly high (~0.15-0.18) — score calibration is noisy at this scale, not yet
  something to read into.
- Artifacts: `/workspace/semsup/a1/{epoch_01,epoch_07,epoch_08}/lora_adapter/`,
  `test_summary.json`, per-checkpoint `metrics_ep{01,07,08}.json` +
  `test_results_ep{01,07,08}.jsonl`.

## B real run — 2026-07-23

**Setup:** identical to A1 (216/51 split, LoRA r=16/α=32 on `query,key,value`, 8 epochs,
grad-accum=8, lr=2e-4), plus `semantic_weight=0.3` and predictor warm-started from B1's
real checkpoint (`/workspace/semsup/b1/predictor_b1.pt`).

**Result:**

| Checkpoint | val_ap (n=51) | test_AP | AUC | F1 | recall | specificity | ECE |
|---|---|---|---|---|---|---|---|
| ep1 (best val) | 0.9742 | 0.8574 | 0.8685 | 0.7903 | 0.8639 | 0.6785 | 0.1750 |
| ep8 | 0.9701 | 0.8742 | 0.8816 | 0.8005 | 0.8669 | 0.7021 | 0.1573 |
| ep2 | 0.9688 | 0.8592 | 0.8687 | 0.7979 | 0.9112 | 0.6283 | 0.1790 |

- **Per the pre-registered selection rule (best val_ap)**: B = 0.8574 vs A1 = 0.8638 —
  B is slightly *below* the control. This is the number that counts; reporting ep8
  (0.8742, val-rank #2) instead because it scores best on test would be post-hoc
  cherry-picking and is explicitly rejected as a comparison.
- **The more informative signal is variance, not the mean**: B's 3 checkpoints span
  0.8574–0.8742 (spread 0.0168) vs A1's 0.8600–0.8647 (spread 0.0047) — B is ~3.5×
  noisier across epochs. Consistent with the framing going in: at n=267 the semantic
  loss adds instability, not signal — not damaging (still above A0=0.853), not helping.
- **val_ap vs test_AP rank order is inverted for B** (val's best checkpoint is test's
  worst) — the real trigger for the val-split investigation below.
- Artifacts: `/workspace/semsup/b/{epoch_01,epoch_02,epoch_08}/{lora_adapter,predictor.pt}`,
  `test_summary.json`, per-checkpoint `metrics_ep{01,02,08}.json`.

## Val-split diagnostic — 2026-07-23

Investigated why val_ap fails to predict test_AP (triggered by B's inverted ranking
above). Computed the actual `clip_level_split(seed=0, val_frac=0.2)` composition locally:

- **51 val rows are only 17 unique clips** (9 positive / 8 negative — well balanced,
  not a class-skew problem) — each clip contributes 2-3 correlated rows (same video,
  different TTE offset), so the *effective* independent sample size is ~17, not 51.
- TTE-bucket proportions in val closely match train (no skew there either).
- **Root cause: sample-size ceiling, not stratification.** With only ~17 independent
  clips, a reasonably-trained model ranks them almost perfectly regardless of small
  checkpoint differences, which is why val_ap saturates at 0.96-0.98 for both A1 and B
  and can't discriminate between epochs — it's measuring near the ceiling, not signal.
- **Implication:** val-based checkpoint selection is unreliable at this data scale for
  both A1 and B equally, so it doesn't bias the A1-vs-B comparison in one direction,
  but it does make any single "best" pick (by either method) untrustworthy.
- **Fixes**: (a) free now — use a fixed selection rule (e.g. last epoch) applied
  identically to both arms instead of trusting val_ap; (b) structural — val naturally
  grows ~15x at the 4.5k caption scale, which should resolve the saturation on its own.

## Infra notes

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

A0→B1→A1→B is now fully validated end-to-end at n=267: mechanism correct, no crashes,
no evidence of a semantic-supervision benefit at this scale, no evidence it hurts either
(B1: retrieval == chance; A1 vs A0: flat; B vs A1: flat/inconclusive, dominated by
checkpoint-selection noise). This was the explicit purpose of the 267-scale pass per the
original runbook — decide now whether to scale captions to ~4.5k. Open, undecided:

1. **Full scale-up (267 → ~4.5k)** — the originally planned target. Real cost: fresh
   vision-captioning of ~1500 clips × 3 TTE, previously flagged as having hit real
   friction (wrong TTE window selected, a silent unattended-run failure, higher
   per-clip token cost than estimated — see DECISIONS.md).
2. **Intermediate scale-up first (e.g. ~500-1000 captions)** — an order of magnitude
   cheaper than (1), re-run B1's retrieval-vs-control diagnostic only (no LoRA needed)
   to check for an upward trend before committing to the full 4.5k captioning effort.
3. **Fix checkpoint selection first, independent of scale** — switch to a fixed rule
   (e.g. last epoch, applied identically to both arms) so any future A1-vs-B result
   isn't compromised by the val-selection noise documented above.
4. **Deprioritize this thread** — per the earlier literature check, no crash-anticipation
   precedent trains language as a strictly train-only signal with vision-only inference;
   the domain's best AP (LATTE) uses no language at all. Redirect effort to the
   faithfulness-focused e4 work, which has stronger existing footing.

(3) is cheap and worth doing regardless of which of (1)/(2)/(4) is chosen. The
1-vs-2-vs-4 choice is a cost/timeline call — not yet made.
