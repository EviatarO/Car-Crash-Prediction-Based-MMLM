# Experiments

## Accepted baseline (reference point, unrelated to the semantic-supervision work)
InternVL3.5-4B-Flash student (LoRA), test AP = **0.762** on 677 clips. See project `CLAUDE.md`.

## Semantic-supervision route

All stages share: 267-row caption set → `clip_level_split(val_frac=0.2, seed=0)` →
216 train / 51 val rows. Test set = the 677-clip Private set
(`dataset/manifests/test_manifest_hires.jsonl`, 338 pos / 339 neg, `group` 0/1/2 =
TTE 0.5s/1.0s/1.5s with n=284/233/160).

### A0 — frozen BADAS-Open baseline (2026-06-24)
Config `student_training/configs/e4_stageA.yaml`. Output
`outputs/e4_vjepa_reason/StageA_scorer/badas_open_private.jsonl` + `metrics_private/`.

| Metric | Value |
|---|---|
| AP | **0.853** |
| AUC | 0.864 |
| F1 @ thr 0.5 | 0.794 |
| Per-TTE AP (0.5 / 1.0 / 1.5 s) | 0.862 / 0.864 / 0.856 |

Within the 0.86±0.03 acceptance band. Do not re-run.

### B1 — predictor-only probe, REAL GPU run (2026-07-21)
BADAS + SigLIP fully frozen; only the `ResamplerProjector` predictor trains. Loss
`1 − cos(pred, SigLIP(caption))`. Batch 16, AdamW lr=1e-4, ≤100 epochs, early stop
patience 15 on val_loss, frozen features cached once (~122s), top-3 checkpoints kept.

Early-stopped at epoch 23; best = epoch 8.

| Metric (held-out, n_val=51) | B1 predictor | Constant mean-embedding control |
|---|---|---|
| val_loss | 0.1345 | — |
| mean_cosine | 0.8655 | 0.8648 |
| retrieval_top1_acc | **0.0196** | **0.0196** (= chance, 1/51) |

**Verdict: no evidence beyond the collapse control.** `train_loss` fell 0.48→0.11 while
`val_loss` bottomed at epoch 8 and rose — overfitting, not a training bug.

Why the control exists and matters: SigLIP embeddings of 267 near-synonymous crash captions
are highly anisotropic, so a predictor that **ignores the video entirely** and emits the mean
caption embedding still scores mean_cosine ≈ 0.865. Without the control, `mean_cosine=0.8655`
reads as a success. `retrieval_top1_acc` is the metric the control cannot fake.

Artifacts: `/workspace/semsup/b1/predictor_b1.pt` (best, ep8), `b1_metrics.json` (full
per-epoch history + control), `predictor_b1_ep{008,009,010}.pt`. Local copy of the metrics:
`outputs/semantic_captions/b1_metrics.json`.

### A1 — crash-only LoRA control, REAL GPU run (2026-07-23)
LoRA r=16/α=32 on `query,key,value`; crash CE loss only (`--semantic-weight 0.0`); 8 epochs,
grad-accum 8, lr 2e-4. Top-3 checkpoints by val_ap, each scored on all 677 test clips.

| Checkpoint | val_ap (n=51) | test_AP | AUC | F1 | recall | specificity | ECE |
|---|---|---|---|---|---|---|---|
| ep8 (best val) | 0.9751 | 0.8638 | 0.8728 | 0.7971 | 0.8195 | 0.7640 | 0.1478 |
| ep7 | 0.9682 | 0.8647 | 0.8718 | 0.8021 | 0.8994 | 0.6578 | 0.1658 |
| ep1 | 0.9679 | 0.8600 | 0.8694 | 0.7932 | 0.8964 | 0.6372 | 0.1839 |

All three within ~+0.01 AP of A0 (0.853) — **flat**. Expected: 216 clips of crash-only
fine-tuning cannot move a model already trained on the full Nexar set. Value of this run is
as the **control bar for B**, not the AP number.

Artifacts: `/workspace/semsup/a1/` — `epoch_{01,07,08}/lora_adapter/`, `test_summary.json`,
`metrics_ep{01,07,08}.json`, `test_results_ep{01,07,08}.jsonl`.

### B — crash + semantic-aux LoRA, REAL GPU run (2026-07-23)
Identical to A1 plus `--semantic-weight 0.3` and predictor warm-started from B1's real ep8
checkpoint. `sem_loss` confirmed active and decreasing (0.298 → 0.171), vs A1 where it is
structurally 0.0 (no predictor is even constructed when `semantic_weight == 0`).

| Checkpoint | val_ap (n=51) | test_AP | AUC | F1 | recall | specificity | ECE |
|---|---|---|---|---|---|---|---|
| ep1 (best val) | 0.9742 | 0.8574 | 0.8685 | 0.7903 | 0.8639 | 0.6785 | 0.1750 |
| ep8 | 0.9701 | 0.8742 | 0.8816 | 0.8005 | 0.8669 | 0.7021 | 0.1573 |
| ep2 | 0.9688 | 0.8592 | 0.8687 | 0.7979 | 0.9112 | 0.6283 | 0.1790 |

Artifacts: `/workspace/semsup/b/` — `epoch_{01,02,08}/{lora_adapter,predictor.pt}`,
`test_summary.json`, `metrics_ep{01,02,08}.json`.

### A1 vs B — the actual comparison (INCONCLUSIVE)
**The conclusion flips depending on which defensible aggregation rule is used**, which is
itself the finding:

| Rule | A1 | B | Says |
|---|---|---|---|
| Best val_ap (pre-registered) | 0.8638 | 0.8574 | B slightly worse |
| Mean of the 3 scored checkpoints | 0.8628 | 0.8636 | Indistinguishable |
| Last epoch (ep8), same for both | 0.8638 | 0.8742 | B better |

No directional claim survives. **Do not report B's ep8 = 0.8742 as "B's result"** — it was
val-rank #2, and promoting it because it scored best on *test* is selecting on the test set.
(Caveat on the "mean" row: the 3 checkpoints come from one trajectory, so they are correlated
— it understates true run-to-run variance.)

The one real signal is **variance**: B's checkpoints span 0.0168 AP, A1's span 0.0047 (~3.5×).
At n=267 the semantic loss adds instability, not signal. Nothing is damaged (both stay above
A0), nothing is gained.

### C1 + T-2 — real per-clip test scores pulled off the pod, paired bootstrap CI (2026-07-25)
Pod was stopped (network volume persisted the results); restarted once, 6 files pulled
directly from `/workspace/semsup/{a1,b}/test_results_ep*.jsonl` (677 rows each — one row per
Private-test clip, `{video_id, ground_truth, group, score}`). Local copy:
`outputs/semantic_captions/Pod_Run_Results/`. Verified before trusting: 677/677 unique clips
per file, zero `ground_truth`/`group` mismatches against `dataset/manifests/test_manifest_hires.jsonl`
for either arm, identical ground-truth vector across A1 and B (same test set, as expected).
Pod stopped again afterward — nothing further needed from it until the InfoNCE re-run.

**Caveat — scores are stored rounded to 4 decimals.** Recomputing AP directly from these files
does not exactly reproduce the headline numbers recorded above (e.g. A1 ep8: recomputed 0.8561
vs the recorded 0.8638) — 677 clips collapse to only 366 unique score values, so the
rank-sensitive AP shifts when near-ties merge. The headline numbers (computed at full precision
during the run, written to `metrics_ep*.json`) stay authoritative; everything below is computed
on the rounded archive, since that's the only per-clip data that exists on disk.

**T-2 — paired bootstrap CI (5000 resamples, seed 0)** on the pre-registered comparison (A1 ep8
vs B ep1, both best-val checkpoints; the same 677 resampled clip indices are applied to both
arms together, preserving the pairing):

| | Value |
|---|---|
| Point estimate (B − A1), rounded-score basis | −0.0099 AP |
| 95% CI | [−0.0239, +0.0030] |
| P(B > A1) across resamples | 7.4% |

**Verdict: confirms, doesn't overturn, the existing "no directional claim survives" call.** The
CI crosses zero, so B isn't distinguishable from A1 at 95% confidence — but only barely (B wins
in ~1 of 13 resamples), and the direction/magnitude agree with the official point estimate
(official: −0.0064; here: −0.0099 — same sign, consistent with the rounding caveat above). The
earlier "no claim survives" framing was based only on the aggregation-rule sensitivity table;
it's now also backed by an actual resampling estimate of the noise floor.

### Val-split diagnostic (2026-07-23) — why val_ap can't select checkpoints
Triggered by B's val-rank order being nearly the *reverse* of its test-rank order. Computed
the real `clip_level_split(seed=0, val_frac=0.2)` composition locally:

- 51 val rows = **only 17 unique clips** (9 positive / 8 negative — well balanced).
- TTE-bucket proportions in val match train closely.
- Each clip contributes 2–3 rows (same video, different TTE offset) → correlated, so the
  **effective independent sample size is ~17, not 51**.

**Root cause is sample-size ceiling, not stratification** — a class/TTE-skew hypothesis was
tested and ruled out. With ~17 independent clips, val_ap saturates at 0.96–0.98 for both arms
and cannot discriminate between epochs. Affects A1 and B equally, so it does not bias the
comparison in one direction, but it makes any single "best" pick untrustworthy.

Reproduce: `python student_training/scripts/semsup_common.py` (prints the split), or call
`clip_level_split` directly and count `{e["video_id"] for e in val_ex}`.

## 2026-07-25 project review — the null result may be a broken-objective artifact
`/project-review` (new user-level skill) audited the semantic-supervision thread end to end
(docs-first, then 2 parallel expert agents: ML-architecture + software-engineering). Full
report: `reports/project_reviews/2026-07-25_project_review.md` (12 sections, severity-ranked;
gitignored, not tracked). Two Critical findings reframed the whole thread:

**A-1 — the cosine-regression objective sits at its own degenerate optimum.** Verified by
hand from the recorded B1 metrics: for a predictor that learns nothing about the video, the
analytic-optimal output is `target_mean/‖target_mean‖`, with loss `1-‖target_mean‖`. Using the
constant-mean control's `mean_cosine=0.8648` (`b1_metrics.json`), that floor is `0.1352065`.
The real trained run reached `0.1344893` — beat the degenerate solution by **0.0007171, i.e.
0.53% of the available range** — with `retrieval_top1_acc` exactly at chance. This means
scaling captions 267→4.5k without changing the loss would very likely reproduce the same null,
at ~17x the cost, without ever testing the actual hypothesis.

**T-1 — `semsup_train.py` had no seed anywhere** (confirmed: B1 did; the A1/B trainer didn't).
The recorded A1-vs-B delta (B slightly below A1) is therefore confounded with different LoRA
init and data order, not attributable to `semantic_weight` alone.

### Fixes implemented and verified this session (all committed)
- **InfoNCE loss** (`--loss infonce` in `semsup_b1_probe.py`, default stays `cosine`): the
  target-mean direction contributes equally to every softmax column and cancels, so the
  collapse solution scores at chance instead of getting a free ride. Sibling-TTE rows of the
  same clip are masked out of the negative set. Verified via 3 synthetic tests before touching
  real data: (1) a constructed video-blind predictor scores retrieval@1 exactly at chance
  (0.0250 = 1/40) under InfoNCE vs 0.865 under cosine on the same synthetic targets; (2) the
  sibling mask tensor checked directly (8 structural assertions: diagonal never masked,
  cross-video pairs never masked, same-video off-diagonal always masked); (3) the loss is
  genuinely trainable — recovers a known synthetic video→target relationship (loss
  7.1→0.0005, retrieval 0%→100% over 300 steps). Then a real end-to-end smoke run (actual
  BADAS+SigLIP+real captions with genuine sibling-TTE groups): no crash, train_loss fell
  1.16→0.62→0.20 across 3 epochs.
- **`--seed` added to `semsup_train.py`** (random/torch/cuda RNG + `clip_level_split`).
- **Predictor resized** `num_queries=1→8, hidden_dim=512→256` (~5.13M→~1.25M params, now
  smaller than the ~2.8M LoRA trunk as originally intended). `ResamplerProjector`'s
  self-attention is now skipped when `num_queries≤1` (verified mathematically a no-op there —
  softmax over one key). Verified: param counts match hand calculation exactly, forward/
  backward correct at both `num_queries=1` and `=8`, and the existing `num_queries=64` caller
  elsewhere in the codebase is unaffected.
- **Per-clip val AP/retrieval** (T-3): `evaluate_crash_ap` (A1/B) and a new
  `clip_level_retrieval_acc` (B1) pool a clip's 2-3 correlated TTE-window rows into one point
  before scoring, instead of treating 51 rows as 51 independent samples when the real count is
  ~17 clips. Verified with a synthetic 7-row/4-clip case showing aggregated AP (0.8333) genuinely
  differs from row-level AP (0.8875) — not a no-op — and a synthetic clip-retrieval case
  (5 well-separated clips retrieve perfectly once pooled; a single-clip case returns NaN
  rather than crashing).
- Also fixed same session (Critical/High, cheap): frames_dir index now defaults to the one
  label file that covers all 267 keys instead of globbing 28 (raises on a genuine conflict
  instead of silent last-writer-wins — 58 overlapping keys were confirmed to exist already);
  `evaluate_metrics.py` no longer duplicates `metrics_core.py`'s formulas (they had diverged:
  single-class AP was `0.0` in one, `null` in the other); per-epoch `epoch_metrics.jsonl`;
  full run config recorded in `train_metrics.json`; test-set scoring streams+flushes per clip
  instead of losing all 677 scores on a mid-run failure; per-clip try/except + `--min-examples`
  guard against a silently-shrunk dataset.

**Not yet done: the actual re-run.** All of the above is verified-correct tooling, not a new
result. The real B1-with-InfoNCE diagnostic at n=267 (or whatever scale is chosen next) has
not been executed — that's the next step, and it's the test that actually distinguishes
"the objective was broken" from "the data is too small."

### A real bug found while running these verifications (not in the reviewed code)
`| tail -N` on a backgrounded shell command reports **`tail`'s** exit code, not the piped
process's. A concurrent-BADAS-loading resource-contention crash was masked this way —
"completed, exit 0" with an empty output directory. Caught by checking the output directory
directly rather than trusting the reported exit code. Fix going forward: redirect straight to
a file (`> log.txt 2>&1`) and check `$?` explicitly; don't pipe through `tail` for anything
where the exit code matters. Separately: two processes both loading BADAS-Open concurrently
on this machine can silently crash one of them (same on-disk HF cache) — run local smoke
tests sequentially, not as parallel background jobs, when they both load BADAS.

## Literature check (2026-07-23, web)
- **BADAS-2.0** (arXiv 2604.05767, Apr 2026) tested general VLMs against their specialised
  architecture and both lost clearly: Cosmos-BADAS F1 0.817 and Gemini-BADAS F1 0.662 (tuned)
  vs BADAS-2.0's 0.964. Their stated conclusion: V-JEPA2's dense temporal prediction suits
  collision anticipation better, and VLMs belong as *explanation generators, not predictors*.
  They also hit the missing-projector problem and worked around it the same way — BADAS-Reason
  is a **separate** Qwen3-VL-4B fine-tune fed peak-risk frames + attention boxes, not a fused
  encoder→LLM. So no published V-JEPA2→LLM projector exists, including from Nexar.
- **LATTE** (arXiv 2504.04103, 2025): AP 89.74 on DAD — current domain best, **no language at
  all**, purely architectural.
- General cross-modal / privileged-information precedent is real (ViLD, VirTex, ICMLM, LUPI /
  "generalized distillation") but gains typically appear at 10^5–10^7 pairs.
- **No crash-anticipation paper found trains language as a strictly train-only signal with
  vision-only inference AND publishes an ablation isolating that component.** A null result at
  n=267 is therefore consistent with the literature, not a contradiction of it.

## Paused parallel thread — ReverseBERT decoder round-trip (reasoning-generation route)
Separate from the above; kept in case the route is revisited. Fine-tuned
`ReverseBERT-EmbeddingGemma-300M` on the same 267 teacher captions (crash-domain fine-tune;
the public checkpoint was domain-locked to emotional-speech captions and unusable as-is).

| Test | BERTScore F1 (baseline-rescaled) |
|---|---|
| Random-pairing floor | 0.185 |
| Accepted epoch-7 InternVL student vs human GT (calibration anchor) | 0.236 |
| Decoder: unseen held-out teacher-style clips | **0.457** |
| Decoder: seen (memorized) teacher clips | 0.656 |

Decoder de-risked (0.457 ≈ 2× the accepted-student bar). **But no video-side Predictor was
ever built for this route** — it would need to map video features into EmbeddingGemma space,
mirroring the Predictor now used against SigLIP. Paused.
