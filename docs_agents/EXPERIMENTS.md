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

### B1-InfoNCE re-run — the objective was broken, not (only) the data (2026-07-25/26)
Real GPU run on the pod (`--loss infonce`, same 267 captions, same 216/51 clip-level split,
`num_queries=8` predictor). Early-stopped at epoch 28, best checkpoint = **epoch 13** (val_loss
0.8918). Cache step (135.7s) reproduced the original B1 cosine run's collapse-control numbers
exactly (`mean_cosine=0.8648`, `retrieval_top1_acc=0.0196`), confirming same data/frozen
features — only the loss changed.

| Metric (best ckpt, epoch 13, n_val=51 / n_clips=17) | InfoNCE | Collapse control | Chance |
|---|---|---|---|
| Clip-level retrieval@1 | **0.2353** (4/17) | 0.0588 | 0.0588 |
| Row-level retrieval@1 | 0.0784 (4/51) | 0.0196 | 0.0196 |
| Sibling-tolerant retrieval@1 | 0.1765 (9/51) | 0.0588 | 0.0588 |
| mean_cosine | 0.1082 | 0.8648 | — |

**Verdict, printed by the script itself: "LEARNED something video-specific" (decided on
clip-level retrieval).** Exact one-sided binomial tests against chance (not just "beats the
control"): clip-level p=0.0154, row-level p=0.0178, sibling-tolerant p=0.0027 — real signal
across all three aggregation rules, which is itself notable since A1-vs-B's aggregation-rule
sensitivity was exactly what made that comparison inconclusive. `mean_cosine` dropping to 0.108
(from 0.865 under cosine) is *expected and correct* — InfoNCE optimizes relative ranking, not
absolute cosine, so the metric that mattered under the old (broken) objective is no longer the
one to read.

**Caveat, stated plainly:** n_clips=17 is tiny. 4/17 correct vs ~1 expected by chance is a real
effect, but the *size* of the effect (0.24 retrieval@1) is not a reliable estimate at this n —
one flipped clip moves it by ~6 points. Epoch 13 was also both the val_loss optimum and the
retrieval peak (legitimate — selection was on val_loss — but worth flagging as a mild lucky
alignment, not a fully independent confirmation).

**What this resolves:** A-1's analytic argument (the cosine objective's own degenerate optimum
explained 99.47% of the original null) is now confirmed empirically, not just by hand-calculation.
The original B1 null was an artifact of the loss function, not proof that no video↔caption
signal exists at n=267. This unblocks the scale-up decision — see DECISIONS.md.

Artifacts: `/workspace/semsup/b1_infonce/{predictor_b1.pt, b1_metrics.json,
predictor_b1_ep{013,020,024}.pt}`. Not yet pulled locally — a stale copy of the *old* cosine
run's `b1_metrics.json` was pulled by mistake and lives at
`outputs/semantic_captions/b1_metrics2.json` (ignore/replace it, don't cite it as InfoNCE data).

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

## Prompt bake-off harness built + calibration-verified (2026-07-27)
Before any caption scale-up spend, built the measurement harness to choose a caption style at
n≈300 rather than guess. Full design in
`~/.claude/plans/CCP based MMLM - Student/2026-07-27_Plan-Prompt-bakeoff-harness-semantic-captions-Gates-0-2.md`.
Two findings changed the original two-prompt plan before any code was written:

- **SigLIP truncates at 64 tokens** (confirmed: `tok.model_max_length == 64`). The incumbent
  267 captions measure 12-24 tokens; a representative 70-120-word caption in the originally
  drafted prompt style measured **128 tokens — 50% discarded**, always losing the outcome
  clause (written last). Redesigned to a single prompt, `caption_neutral` capped at 40 words
  (measured 24-30 tokens on realistic examples), outcome-relevant content stated first.
- **Positives and negatives use different windowing conventions on disk**: positives are
  pre-extracted at `TTE_0.5/1.0/1.5`; negatives (no event exists) at `MID/MID-4/MID-8`. This
  matches how the incumbent 267-caption set already works (`teacher_dataset_e3b.jsonl`:
  47 negatives at MID/MID-4/MID-8, 42 positives at TTE_0.5/1.0/1.5 — the origin of "267" is
  exactly `42×3 + 47×3`).

**Design change**: two separate prompts → **one prompt, three arms** built from its structured
JSON output (`caption_neutral`, `risk_clause`, `verdict`, `confidence`). Arm A = neutral
description only; Arm B = neutral + risk clause (identical descriptive content, isolating one
variable); Arm C = a zero-cost label-only template (falsification control). This makes the
comparison **paired** (same descriptive content between A/B) instead of independent, which is
where the statistical power comes from at n≈300 — plain end-to-end AP was already shown
underpowered for a much bigger intervention (A1-vs-B's CI above).

**Calibration tests, all run for real against the incumbent 267-caption set (not synthetic):**
- Gate 0 self-test (`semsup_caption_qa.py --input Caption_Train_All_Clips.jsonl`): reproduced
  0% token truncation and correctly **flagged** the known 267/267 verdict-leakage artifact
  (documented 2026-07-25) rather than treating it as success — proves the leakage check fires.
- Gate 1 (`semsup_caption_geometry.py`): reproduced anisotropy `‖mean(E)‖ = 0.8547` **exactly**,
  matching the earlier hand-calculated figure. Also produced new numbers not measured before:
  effective rank 27.73, `nn_purity_by_class = 0.8914` (embeddings cluster by pos/neg) alongside
  `centroid_separation = 0.0369` (small) — flagged as the exact structure-vs-leakage ambiguity
  Arm C is designed to resolve.
- Sampler preflight (`semsup_sample_clips.py --n 300 --dry-run`, local run): **correctly
  refused** rather than silently shrinking — the achievable local pool is 89 videos / 267 rows,
  identical to what already exists (42 pos, 47 neg). Zero new distinct videos are reachable
  locally; the real ceiling depends on the pod's 295 `train_HiRes` folders, not yet checked.
- Gate 2 mechanics (toy run, n=14 real clips from the incumbent set, real BADAS+SigLIP forward
  passes, CPU): `semsup_b1_probe.py --captions arm_X.jsonl` ran end-to-end for A/B/C, and
  `semsup_promptbakeoff_report.py` correctly computed exact-binomial and paired-bootstrap
  statistics from real per-clip hit data and rendered `summary.md`. No real conclusion drawn
  (n=14 toy set) — mechanics only, per plan verification step 6.
- `_build_promptbakeoff_xlsx.py`: banned-word (amber) and over-token (lavender) cell coloring
  verified by direct cell inspection on the toy set, alongside the reused green/red verdict
  coloring convention from `_build_caption_xlsx.py`.

**Not yet done**: real captioning against the prompt, and everything downstream of it. The
harness is proven correct; no new scientific result exists yet.

## 500-clip extraction unblocked (2026-07-28)
Raw Nexar MP4s were found locally in a sibling project folder not previously checked, unlocking
sampling from the full 750/750-balanced `train.csv` pool instead of the 89-video local frame
ceiling. `semsup_extract_promptbakeoff_frames.py` extracted 500 new distinct-video clips
(250 pos at TTE_0.5/1.0/1.5 from `train.csv`'s `time_of_event`, 250 neg at MID/MID-4/MID-8,
matching the incumbent set's convention) — 500/500 succeeded, 0 failures, ~15.5 min wall.
Sampled manifest `dataset/manifests/semsup_promptbakeoff.jsonl`: 498 rows (250/3=83.33 → 249+249
after integer-division bucket filling), 498 distinct videos, zero overlap with the incumbent
267, the 677-clip test set, or `val_e3a.jsonl`. No pod trip was needed for this step after all.

## Teacher-model bake-off (2026-07-28/29)
Before spending the above 498-clip budget for real, tested whether the current teacher
(`google/gemini-3.1-pro-preview` via OpenRouter) is still the right choice, using the 18-clip
`val_e3a.jsonl` GT set as a cheap screen. Three rounds, each building on the last.

### Round 0 — the historical v6 baseline is not reproducible (model drift)
Re-ran `PROMPT_G_OPT_v6_balanced`, **completely unmodified**, at the exact original settings
(native 1280x720, `detail=high`, `temp=0.1`, same model slug) that produced the recorded
83.3% verdict accuracy / mean reasoning score 6.78. Result **today**: **50.0% / 4.61** — a
33-point accuracy swing on the identical prompt and clips. Confirmed not a code/settings
confound (image encoding, model slug, and temperature all verified byte-for-byte identical to
the original run; one real difference found - the original capped `max_tokens=8192` to reserve
room for reasoning tokens, this repo's newer scripts had dropped it - included in the rerun for
fidelity). Most likely cause: silent drift on the `preview`-tagged OpenRouter alias. **Every
comparison below uses this same-day rerun as the baseline, not the historical number.**
Full detail + the two clips that broke every subsequent run: `outputs/prompt_bakeoff/
semsup_val18/summary.md`.

### Round 1 — Qwen3.7 Flash & GPT-5.6 Luna Pro, v6 prompt unmodified
Same 18 clips, same v6 prompt, two new OpenRouter models (`qwen/qwen3.7-flash` $0.03/$0.13 per
1M; `openai/gpt-5.6-luna-pro` $0.50/$3.00 per 1M, vs Gemini's $2/$12).

| Teacher | Verdict acc | Mean score | Recall | Precision | Predicted YES on |
|---|---|---|---|---|---|
| Gemini (same-day baseline) | 50.0% | 4.61 | 0.67 | 0.50 | 6/18 |
| Qwen3.7 Flash | 61.1% | 4.72 | **0.22** | 1.00 | **2/18** |
| GPT-5.6 Luna Pro | 61.1% | 4.83 | **0.22** | 1.00 | **2/18** |

**The higher accuracy is arithmetic, not insight**: both new teachers predict "collision" on
only 2 of 18 clips, getting all 9 negatives right (precision 1.00) but missing 7 of 9 real
positives (recall 0.22, same as Gemini's *worst* case). Do not read this as "switch teachers" -
it's evidence of a strong conservative prior, not better scene understanding.

**Genuine positive finding**: both independently solved `02117` (GT=NO: gray sedan at constant
distance, van *stopped* before a crosswalk) - the one clip every Gemini run (original v6,
same-day rerun, V2, V3) hallucinated identically wrong ("black SUV merges into ego lane").

**Operational note**: Qwen3.7 Flash returned 4 empty/unparseable responses at
`max_tokens=8192`; raising to 20000 fixed some but broke 2 *different* clips on the same
attempt (provider-side flakiness, not a deterministic budget issue) - needed a `--resume` retry
pass to reach 18/18. GPT-5.6 Luna Pro had zero failures at 20000 in one pass.

Full detail: `outputs/prompt_bakeoff/semsup_val18/teacher_bakeoff_summary.md`,
`reasoning_analysis_teacher_bakeoff.xlsx`.

### Round 2 — a from-scratch prompt (V4) + Qwen3-VL-235B-A22B-Thinking
Hypothesis: maybe the under-calling is fixable with better prompting. Wrote
`prompts/PROMPT_SEMSUP_V4_QWEN.py` from scratch in Qwen's own recommended structure
(Role/Task/Context/Instructions-with-forced-step-by-step-thinking/worked-examples/Do-NOT/
Priority), explicitly instructing *"Do NOT default to NO... under-calling a real collision is
as serious an error as a false alarm"* plus two worked examples (one YES, one NO) to calibrate
the threshold. Ran against a different, reasoning-native model,
`qwen/qwen3-vl-235b-a22b-thinking` ($0.40/$4.00 per 1M, 131k context). 18/18 succeeded on the
first attempt at `max_tokens=20000`, zero failures.

| Teacher / prompt | Verdict acc | Mean score | Recall | Precision | Predicted YES on |
|---|---|---|---|---|---|
| Qwen3-VL-235B / V4 | 61.1% | **5.11** | **0.22** | 1.00 | **2/18** |

**The explicit counter-instruction had zero measurable effect on recall** - identical confusion
matrix (TP=2, FP=0, TN=9, FN=7) to both Round 1 candidates, on a different model *and* a
different prompt. This is now a 3x-replicated finding: whatever drives the conservative bias
survives explicit corrective instruction. Two explanations, not yet distinguished: (a) a
property of these specific models' calibration on visual risk assessment, or (b) a property of
v6-style decision-gate framing itself ("predict YES ONLY if... clearly hold") regardless of
surrounding instruction.

**The clearest single-clip evidence for explanation (b)**: on `00687` (GT=YES: gray SUV drifts
into ego lane), the caption correctly says *"Gray SUV merging from right lane into ego lane
while black sedan maintains position ahead"* - an accurate read of the actual hazard - but
`risk_clause` calls it "normal merging traffic" and `verdict=NO`. **The model perceived the
hazard correctly and the decision layer discounted it anyway.** This reframes the open question
from "can the model see the danger" to "why does the decision layer override correct
perception," which points toward testing a bare risk score instead of a binary verdict+gates
(not yet tried).

**`02117` solved a third time**, independently, with an accurate caption ("Sedan ahead in ego
lane maintaining consistent following distance") - three different non-Gemini models now agree
on the correct read of the one clip that broke every Gemini attempt.

**Also the best caption-fidelity score of the whole investigation** (mean 5.11 vs the prior
best 4.83) - worth keeping in mind purely for the SigLIP-target captioning use case, independent
of the unresolved verdict/recall problem.

Full detail: `outputs/prompt_bakeoff/semsup_val18/qwen3vl_v4_summary.md`,
`reasoning_analysis_qwen3vl_val18.xlsx`.

### Cross-round summary table

| Teacher / prompt | Verdict acc | Mean score | Recall | Precision |
|---|---|---|---|---|
| Gemini 3.1 Pro Preview / v6 (same-day) | 50.0% | 4.61 | 0.67 | 0.50 |
| Gemini 3.1 Pro Preview / V2 | 50.0% | 4.61 | - | - |
| Gemini 3.1 Pro Preview / V3 (CoT) | 50.0% | 4.78 | - | - |
| Qwen3.7 Flash / v6 | 61.1% | 4.72 | 0.22 | 1.00 |
| GPT-5.6 Luna Pro / v6 | 61.1% | 4.83 | 0.22 | 1.00 |
| Qwen3-VL-235B-Thinking / V4 | 61.1% | **5.11** | 0.22 | 1.00 |

**Not yet done**: (a) bare 0-100 risk score instead of binary verdict+gates, thresholded
post-hoc - directly tests the `00687` finding; (b) loosened/removed decision gates, same
prompt otherwise - cheaper test of the same hypothesis. Either should run on the 18-clip
screen before any teacher/prompt is chosen for the 498-clip production captioning run.

### Rounds 3-9 (2026-07-30/31, 2026-08-01) — V5 through V9, plus cross-model checks

Continued the same 18-clip screen through 6 more prompt versions, chasing the recall problem
via progressively different mechanisms: V5 (0-100 risk score + mandatory pre-mortem, verdict
derived mechanically from the score), V6 (kinematic decomposition — ego-motion/lateral-drift
observation fields + 4 summed 0-25 sub-scores), V7 (explicit ego-frame vs world-frame motion
separation, since V6 showed the model conflating its own turning with other agents moving), V8
(narrative delta/cause/ego-response caption structure), V9 (deliberately minimal — ~800 tokens,
no observation scaffolding, betting that a reasoning-native model's own internal CoT makes
external scaffolding redundant).

**Final result: statistically inconclusive, all of it.** Verdict accuracy across V4-V9 on
`qwen/qwen3-vl-235b-a22b-thinking`: 61.1%, 61.1%, 55.6%, 66.7%, 50.0%, 72.2%. Every pairwise
comparison's 95% CI overlaps; McNemar exact test between any two rounds never drops below
p=0.125. **n=18 cannot rank these prompts** — this was true from V4 onward, not just
discovered at the end.

**Cross-model checks (2026-08-01)**, the two results worth keeping:
- **`PROMPT_G_OPT_v6_balanced` (unmodified, the *original* teacher prompt) on
  `google/gemini-3.6-flash`**: 72.2% acc, **0 false positives**, best caption-fidelity mean of
  every round/model combination tested this entire investigation. The sharpest single
  comparison recorded: McNemar vs V9 on the same model gave 4 clips flipping in v6_balanced's
  favor, 0 the other way (p=0.125 — still short of significance at n=18, but the least
  ambiguous result of the whole thread).
- **The same unmodified v6_balanced prompt on `qwen/qwen3-vl-235b-a22b-thinking`**: **0/18 YES
  predictions** — complete collapse. Its `verdict_reasoning` field echoed the prompt's own
  "prefer NO"/"base-rate principles favor safe outcome" language back almost verbatim on every
  clip. Confirms this is a **model-family × prompt interaction**, not a property of "heavy
  CoT + conservative gates" in general — the same structure produced Qwen3.7 Flash's and
  GPT-5.6 Luna Pro's under-calling (Round 1) but not Gemini's.

**Decision (2026-08-01): stop here.** Full detail per round: `outputs/prompt_bakeoff/
semsup_val18/{v5_balanced,v6_kinematic,v7_egoframe,v8_narrative,v9_minimal,v9_gemini36flash,
v6_balanced_gemini36flash}_summary.md`. Superseded by the train4500-inference pipeline below —
see PROJECT_STATE.md.

## train4500-inference pipeline (2026-08-01)

**Goal**: score the real ~4,500-window train pool through the frozen A0 scorer (inference
only, nothing trains) to find where BADAS-Open actually fails, informing whether caption
budget should be uniform or failure-targeted — a direct answer instead of an 18-clip proxy.

**Setup**: `build_train4500_manifest.py` → 4,446 rows = 741 pos × 3 TTE buckets (0.5/1.0/1.5s)
+ 741 neg × 3 offset buckets, excluding val_e3a's 18 clips (drawn from the same train.csv pool,
used for Stage-C checkpoint selection — confirmed via contamination guard, which correctly
fired on the first attempt before the exclusion was added). Chunked into 3 groups of ~500
videos (class-interleaved, not plain-sorted — see ARCHITECTURE.md gotcha) for pipelined
extraction/scoring.

### Chunk 0 (500 videos / 1,500 windows) — before the MID fix
First real scoring run on train data. `n=1500 AP=0.9034 AUC=0.9094 accuracy=81.7%
TP/FP/TN/FN=655/179/571/95 error=18.3%`. Compared against A0's known test error (23.6%,
TP/FP/TN/FN=308/130/209/30) — gap 5.4%, just over the pre-registered 5pp stop-and-diagnose
threshold. Investigation (not assumption) found the gap entirely attributable to the `MID`
bucket: **107/250 MID-bucket windows were false positives (42.8% error), 0 false negatives**,
at 0.99+ confidence — see the MID-10 fix in ARCHITECTURE.md for the diagnosis and repair.

### Chunk 0 — after the MID-10 fix
Only the 250 affected windows needed re-extraction/re-scoring (the other 1,250 rows in chunk 0
were untouched and reused as-is). **`n=1500 AP=0.9555 AUC=0.9504 accuracy=86.7%
TP/FP/TN/FN=655/104/646/95 error=13.3%`.**

| Bucket | n | wrong | error rate |
|---|---|---|---|
| MID-10 (was MID) | 250 | 32 | 12.8% (was 42.8%) |
| MID-4 | 250 | 36 | 14.4% |
| MID-8 | 250 | 36 | 14.4% |
| TTE_0.5 | 250 | 17 | 6.8% |
| TTE_1.0 | 250 | 31 | 12.4% |
| TTE_1.5 | 250 | 47 | 18.8% |

Bucket-error spread dropped from 36.0% (systematic — one bucket clearly broken) to 12.0%
(diffuse — `mine_train_failures.py`'s own classifier now recommends **uniform** caption
allocation, not failure-targeted, based on this chunk alone).

**Still open, not a bug as far as verified**: chunk 0's corrected error (13.3%) is well below
A0's test-set error (23.6%) — the checkpoint gap actually *grew* (5.4%→10.4%) once the MID
artifact was removed, because MID's errors had been coincidentally padding train's rate closer
to test's. Test is FP-dominated (130:30, ~4.3:1); chunk 0 is nearly balanced (104:95, ~1.1:1).
Real distributional difference between the pools, not yet explained — pipeline mechanics were
checked (sequential-decode extraction verified byte-identical to the old per-frame-seek
method on 6 real videos) and nothing pointed to a mechanical bug. **Chunks 1-2 will show
whether this is a chunk-0-specific fluke or a stable property of the full train pool.**

### Chunks 1-2 (982 videos / 2,946 windows) — DONE
First transfer attempt corrupted 1,674/2,946 dirs (0-byte content from a RunPod storage quota
hit mid-`tar`, caught by a size-aware re-check after a count-only check falsely passed). User
raised the quota (+15GB); re-transfer verified clean (all 2,946 dirs correct size, cross-checked
via `du -sh` delta) and both chunks scored successfully.

Per-chunk AP/AUC held stable across all three independently-sampled 500-video chunks
(0.9555/0.9513/0.9535 AP, 0.9504/0.9454/0.9462 AUC) — the pattern is a real property of the
train pool, not chunk-0 noise.

### Combined result, all 3 chunks (n=4,446) — FINAL
`AP=0.9535 AUC=0.9474 accuracy=86.8% TP/FP/TN/FN=1954/318/1905/269 error=13.2%`
(587 failures: 318 FP + 269 FN). Bucket-error spread 13.9% (worst TTE_1.5 19.6%, best TTE_0.5
5.7%) → **DIFFUSE**, confirming chunk 0's own classification at 3× the sample. Decision:
**uniform caption allocation**, not failure-targeted. Coverage check
(`monitor_train4500_coverage.xlsx`): 213/4,446 windows already captioned (4.8%).

**Still open, confirmed real (not chunk-0 noise)**: train's 13.2% error vs A0's known 677-clip
test error of 23.6% — held at the same magnitude across all 3 chunks. Test is FP-dominated
(130:30, ~4.3:1); train is nearly balanced (318:269, ~1.2:1). Pipeline mechanics checked and
ruled out (byte-identical sequential-decode extraction). Not investigated further this session —
see DECISIONS.md.

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

---

## B_1761 parallel — InfoNCE semantic-aux vs crash-only, matched init (on the V10 corpus)
- Config: from-scratch LoRA (not continued from A1's checkpoint, unlike the earlier sequential
  attempt), same seed=0 as A1_1761, same recipe (`query,key,value`, constant LR, dropout 0.05),
  `--semantic-weight 0.05 --semantic-loss infonce`, V10 corpus (`Caption_Train4500_Mixed_1761
  .jsonl`, GT-informed/blind branch prompt).
- Result: test_AP=0.8901, AUC=0.8955, vs A1_1761's test_AP=0.900, AUC=0.904.
- Paired bootstrap (677 test clips, per-clip scores): ΔAP = A1 − B = **+0.0105**, 95% CI
  [0.0040, 0.0173] (excludes zero) — **B is significantly worse than A1**, not noise.
- Caveat found later (see `/project-review` below): this result is confounded by corpus label
  leakage and should not be read as "semantic supervision doesn't help" without qualification.

## `/project-review` audit (2026-08-08)
- Full ML+code review of the semantic-supervision thread, triggered by the B_1761-parallel
  negative result.
- **Key finding**: TF-IDF (1-2gram, min_df≥3) + LogisticRegression, 5-fold GroupKFold by
  `video_id`, predicting the crash label from V10 caption text alone → **AUC=0.9643**. The V10
  corpus's caption text alone is a near-perfect proxy for the label — driven by the GT-informed
  vs. blind prompt branch producing systematically different vocabulary by class, not by
  semantic content per se.
- Secondary finding: `semsup_b1_probe.py`'s `evaluate()` selects checkpoints on cosine loss even
  when `--loss infonce` is passed — the B1_1761 probe's selected checkpoint
  (`predictor_b1_ep028.pt`) is not actually the best one by the metric that matters
  (`val_retrieval_top1_acc_clip`: 0.1086 selected vs 0.1267 available at epoch 43). Not yet
  fixed; low priority since B-v2 doesn't warm-start from this checkpoint.
- Report: `reports/project_reviews/2026-08-08_project_review.md` (gitignored, not in repo
  history).

## A1-v2 — full pool + cosine LR + encoder-only LoRA + dropout 0.10
- Config: full 4,446-window pool (natural 13.2% hard-example distribution, built via
  `build_pool_from_manifest.py` with placeholder captions — crash-only, `--semantic-weight 0`),
  `--lr-schedule cosine --warmup-frac 0.05`, `re:`-regex encoder-only LoRA target modules (72
  adapters, vs A1_1761's 108 encoder+predictor), `--lora-dropout 0.10`, seed=0, 12 epochs
  planned, resumed mid-run after the I/O fix (epochs 1-2 pre-fix at ~97-102 min/epoch, epochs
  3+ post-fix at 15-20 min/epoch).
- Result (test_AP by checkpoint):

  | Epoch | val_ap | test_AP |
  |---|---|---|
  | selected (by val) | — | 0.868 |
  | 6 (best test, not selected) | — | 0.888 |
  | A1_1761 reference | — | **0.900** |

  Even the single best test-set checkpoint across all 12 epochs (0.888) did not beat A1_1761's
  0.900. Val-based checkpoint selection also picked a worse-on-test checkpoint than epoch 6 —
  flagged as a val/test selection-rule mismatch, not fixed (small-val-set noise, expected at
  this scale).
- Verdict: **negative result, recipe/pool bundle not adopted.** Root cause not isolated — could
  be the natural (non-enriched) pool distribution, could be the recipe bundle (cosine LR /
  dropout / encoder-only), could be both. Deprioritized in favor of the core B-vs-A1 test — see
  DECISIONS.md.
- Outputs: `outputs/e4_vjepa_reason/a1_v2_full/{train_metrics.json, test_summary.json,
  epoch_metrics.jsonl}` (pulled locally).

## I/O bottleneck diagnosis + fix (infrastructure, not a model experiment)
- Direct on-pod profiling over 20 real windows: raw file read = 670ms/window, +decode/resize =
  503ms/window (total ~1.17s/window), +GPU forward = ~0ms (unmeasurable against the I/O cost).
- Fix: `TrainableBadasWrapper.prefetch_clips()` concurrent pipeline (see ARCHITECTURE.md).
- Verified via isolated pod benchmark: workers=0 (serial) vs 8 vs 16 → **5.3× speedup at 8
  workers**. Verified via live resumed A1-v2 run: epochs 1-2 (pre-fix) ~97 min avg → epoch 3
  (post-fix) 15.5s... i.e. 929.4s = 15.5 min → **6.3× speedup**, GPU utilization 24-33% → 83-94%.

## Captioning concurrency fix + real cost logging (infrastructure)
- Serial baseline: 11.8s/clip. At `--concurrency 16`: ~1s/clip. **~12× speedup.**
- Cost: verified cost-neutral (OpenRouter bills per-token processed, not per-request or
  wall-clock; confirmed via the `usage` field in real API responses, now logged to
  `<out>.usage.jsonl` instead of discarded).
- Real logged cost for the post-fix portion of the V12 1,761-window recaption: 900 calls,
  $32.758 tokens + $0.058 other = **$32.82 tracked** (the pre-fix portion of the run, ~861
  calls, was not covered by the usage-logging fix since it predates it — total real spend for
  the full V12 run is not fully reconstructable from logs, only the post-fix tail).

## V12 neutral prompt — leakage-gate validation cascade
Three stages, increasing scale/rigor, run in sequence with an explicit stop/go decision at each:

| Stage | n | Design | Result | Verdict |
|---|---|---|---|---|
| Leakage judge, val18 | 18 | Text-only judge (fresh context, captions only) predicts crash/no-crash | 12/18 = 66.7% correct | p≈0.12 (one-sided exact binomial vs chance) — **not significant** |
| Leakage judge, val18+82 | 100 | Same judge, +82 balanced fresh-sampled clips (`sample_val_check_clips.py`, seed=0, excludes val18's video_ids) | 72/100 = 72.0% correct | p<0.0001 — **real, significant residual leakage** |
| Full-corpus TF-IDF gate | 1,761 | TfidfVectorizer(1,2-gram)+LogisticRegression, GroupKFold(5) by video_id, `caption_neutral` vs `event_occurs` | **AUC=0.7640** (target <0.75) | **Narrow miss** (0.014 over target); reduction vs V10's 0.9643 = 43% cut in excess-over-chance signal (i.e. (0.9643−0.5) → (0.7640−0.5)) |

- Residual-leak source (TF-IDF coefficient inspection on the n=100 sample): driven by genuine
  kinematic vocabulary (`braking` +0.957 strongest coefficient, `decreasing gap`, `path
  closing`) — physically real correlates of the crash label, not register violations. **0/100
  and 0/18 banned-word violations found** — V12's word bans worked completely; the residual
  signal is a different phenomenon (physics correlation) that prompt engineering alone likely
  can't remove without degrading caption accuracy.
- User's decision (via AskUserQuestion): **accept the near-miss, proceed to full recaption +
  B-v2, report the residual leak honestly.**
- Outputs: `outputs/prompt_bakeoff/semsup_val18_neutral/{raw_v12_gemini.jsonl,
  raw_v12_extra82.jsonl, review_val18_neutral.xlsx, summary.md, leakage_judge_n100.md}`.

## V12 full recaption (1,761 windows)
- Ran `semsup_caption_promptbakeoff.py --prompt v12 --concurrency 16` over the full pool
  (`dataset/manifests/recap_v12_1761.jsonl`).
- Verified: 1,761 distinct `frames_dir` (no duplicates/dropped rows), all captions ≤40 words,
  all `gap_trend` values from the closed vocabulary, label balance 905/856 — matches the
  original V10 pool's balance exactly (same underlying clip set, different caption text).
- Output: `outputs/semantic_captions/Caption_V12_Neutral_1761.jsonl` (raw schema) +
  `..._fortrain.jsonl` (with `caption`/`gt_verdict` aliases added for `load_training_examples`
  compatibility) + `.usage.jsonl` (cost sidecar, post-fix portion only).

## B-v2 — InfoNCE semantic-aux vs crash-only, matched init, on the corrected V12 corpus
- Config: from-scratch LoRA, seed=0, A1_1761's exact recipe (`query,key,value`, constant LR,
  dropout 0.05, grad-accum 8, 8 epochs), `--semantic-weight 0.05 --semantic-loss infonce
  --infonce-tau-init 0.07`, captions = `Caption_V12_Neutral_1761_fortrain.jsonl`.
- **Result: lost.** Selected checkpoint (epoch 2, best val_ap): test_AP=0.8796, AUC=0.8905.
  Paired bootstrap vs A1_1761: ΔAP=+0.0189, 95% CI [0.0099, 0.0285], excludes zero. **Wider**
  than B_1761-parallel's gap on the leaky V10 corpus (+0.0105) — cleaning the caption leak did
  NOT close the gap, ruling out leakage as the sole explanation.
- **But this run had two real execution defects** (found 2026-08-12, present in this run and
  B_1761-parallel both): (1) Predictor cold-started, contrary to the written plan's B1→B
  warm-start requirement; (2) shared gradient-clip budget across LoRA+Predictor, unlike A1's
  LoRA-only budget. See B-v3 below, which fixes both.
- Output dir: `/workspace/semsup/b_v2_1761/`.

## B-v3 — B-v2 with both execution defects fixed (2026-08-13)
- Same recipe as B-v2, plus: `--predictor-init` from a B1 probe trained on the V12 corpus
  (warm-start, per the written plan), and `--clip-grad-per-group` (LoRA and Predictor clipped
  on separate 1.0 budgets, matching A1's effective LoRA budget).
- **Result: lost, and by MORE than B-v2** — fixing the defects made it worse, not better.
  Selected checkpoint (epoch 2): test_AP=0.8768, AUC=0.8877. Paired bootstrap vs A1_1761:
  ΔAP=+0.0218, 95% CI [0.0117, 0.0325], excludes zero.
- **Crash-vs-semantic gradient-angle probe** (new instrumentation, `--grad-cosine-every`,
  `torch.autograd.grad()` on shared LoRA params only — bit-identical to off, no `.grad`
  accumulation): cos(crash,sem) drifted from +0.0165 (epoch 1) to −0.0244 (epoch 8); the
  fraction of conflicting sampled steps climbed from 45.2% to 55.9%; the semantic term's
  relative magnitude after λ-weighting grew from 0.048 to 0.089. Reading: the two objectives
  are **near-orthogonal, drifting mildly adversarial**, not strongly opposed — and getting
  relatively louder as the crash loss saturates.
- A 12-epoch exploratory extension (resumed from epoch 8) confirmed pure overfitting past
  epoch 8, not further learning: train_val_gap climbed from 0.63 to 1.04, best-on-test ΔAP vs
  A1_1761 widened further to +0.0330 (95% CI [0.0156, 0.0519]).
- Output dirs: `/workspace/semsup/b_v3_1761/` (8-epoch), `/workspace/semsup/b_v3_1761_ext12/`
  (12-epoch extension). **Known bug**: the 8-epoch `test_summary.json`/`epoch_metrics.jsonl`
  were accidentally overwritten locally by the ext12 pull — the correct 8-epoch files still
  exist on the pod's persistent volume, not yet re-pulled.

## Caption leakage gate — persisted as a script (2026-08-16)
`teacher_distillation/scripts/caption_leakage_gate.py`: TF-IDF(1,2-gram, min_df≥3) +
LogisticRegression + GroupKFold(5) by `video_id`, previously run ad-hoc and never saved.
Reproduces both prior numbers **exactly**: V10 AUC=0.9643, V12 AUC=0.7640. Writes results
(per-fold AUCs, top predictive n-grams) to JSON — `outputs/semantic_captions/
leakage_gate_{v10,v12}.json`.

## Pooled-tap B1 probe — does the classifier's own bottleneck carry caption info? (2026-08-15)
The crash classifier reads a single 1024-d pooled vector, not the full 2560×1024 patch grid
the semantic loss has always attached to (`semsup_b1_probe.py --tap {patches,pooled,
meanpool}`, `_VectorMLP` predictor for the single-vector taps). All measured against the same
221-clip held-out set, chance=0.45%:

| Tap | retrieval@1 | × chance |
|---|---|---|
| `patches` (default) | 14.03% | 31× |
| `pooled` (classifier's actual input) | 9.95% | 22× |
| `meanpool` (control — uniform pooling) | 8.14% | 18× |

Caption info survives the 2560× compression comfortably, and `pooled` ≈ `meanpool` (within
noise at n=221) — the crash-tuned attention is **not** specifically discarding caption-relevant
directions relative to uniform pooling. Refutes the *strong* form of the bypass hypothesis
(information can't reach the classifier).

## InfoNCE false-negative check — eliminated as a concern (2026-08-15)
Concern: near-duplicate captions across different clips get punished as false negatives.
Measured directly: cross-video caption cosine (SigLIP embeddings) averages 0.701 (p99=0.870).
At a 0.90 masking threshold, only ~4 of 1,413 negatives per anchor would be masked (0.3%) —
cannot explain a 0.02 AP gap. **Not implemented; not needed.** Bonus: confirms V12 captions
are genuinely clip-specific despite the constrained vocabulary.

## P3 — does the semantic gradient reach the classifier's representation? (2026-08-16, corrected 2026-08-17)
`student_training/scripts/p3_delta_patches_vs_pooled.py`: loaded A1_1761 (epoch 4) and B-v3
(epoch 2) LoRA weights on the same frozen base, captured `patches`+`pooled` for the same 40
held-out clips under each, computed `‖Δpooled‖/‖Δpatches‖` vs the same ratio for a random
patch-grid perturbation of equal norm.

**First pass (2026-08-16) was under-powered**: single noise draw per clip, no per-clip data
saved, paired design analyzed as independent means — "1.8×" reported with no error bar.
**Corrected 2026-08-17**: 20 noise draws per clip (averaged), per-clip arrays saved, paired
bootstrap CI (5,000 resamples). Same point estimate, now quantified: real ratio 0.00341 vs
random-control 0.00186 (~1.8×), **paired diff mean=0.00152, 95% CI [0.00143, 0.00163],
excludes zero.**

**Refutes the *weak* form of the bypass hypothesis too**: the real weight difference reaches
the pooled representation at least as well as (if anything slightly better than) a random
perturbation of equal size would — not preferentially routed away from it. Combined with the
gradient-angle finding above (near-orthogonal, not opposed), the account of B's underperformance
is "the signal reaches the decision path but doesn't help there," not a routing problem.

## P1 — two-stage (semantic-pretrain → crash-finetune) training (2026-08-17)
Implemented in `semsup_train.py`: `--crash-weight` (0.0 = Stage A, semantic-only, no crash
gradient reaches the trunk at all) and `--select-by {val_ap,retrieval}` (Stage A requires
`retrieval` — val_ap is uninformative when nothing optimizes it). `evaluate_val()` extended
with clip-level retrieval@1, a per-epoch collapse control, retrieval vs the full 1,761-caption
bank, similarity-tolerant retrieval, and embedding-health diagnostics (margin, softmax
saturation, similarity spread, predictor collapse) — all from tensors already in hand, no
extra forward passes. `semsup_b1_probe.py`'s retrieval helpers lifted to module level so
`semsup_train.py` can import them. New `p1_stageA_gate.py`: scores a Stage-A checkpoint's
encoder against the **unchanged frozen crash head** on the 677-clip test set, no training —
the cheap check before committing to Stage B.

**Stage A** (12 epochs, semantic-only, `--select-by retrieval`, full 1,761-window corpus):
retrieval@1 climbed to a peak at **epoch 10 (20.81%, 46× chance)**, then declined (epoch 11:
15.38%, epoch 12: 15.84%) — the held-out retrieval metric caught overfitting directly, with
`train_val_gap` corroborating (crosses from negative to positive right at epoch 10). Selected
epoch 10 (correctly, by the ranking — not just "most recent").

**Gate** (epoch 10 encoder + frozen head, 677-clip test, no training): test_AP=0.8448,
AUC=0.8595 — a small, expected dip below A0 (0.853), not a catastrophic collapse. **Passed.**

**Stage B** (8 epochs, crash-only, LoRA warm-started from Stage A epoch 10, otherwise
identical to A1_1761's recipe): selected epoch 2 (val_ap=0.9029), **test_AP=0.8266,
AUC=0.8481**. Paired bootstrap vs A1_1761: **ΔAP=+0.0716, 95% CI [0.0477, 0.0977], excludes
zero** — the **largest negative result in the entire thread** (>3× the prior worst, B-v3's
+0.0218), and **below the frozen A0 baseline**. Even the best-on-test checkpoint across all 8
(epoch 1, illegitimate to select on) only reaches 0.8538, essentially tying A0, nowhere near
A1's 0.900.

**Mechanism, not just a number**: Stage B's `train_val_gap` grew to more than double
A1_1761's under an identical LR schedule (0.870 vs 0.370 by epoch 8; train crash_loss 0.192
vs 0.314, val_crash_loss 1.062 vs 0.684). Warm-starting LoRA from Stage A's already-adapted
weights and reusing A1's from-scratch learning rate (2e-4) overfits much faster — a specific,
measured mechanism, not unexplained forgetting.

**Not yet run**: the retention probe (does Stage B's final encoder still retain Stage A's
semantic structure, measured via retrieval@1 using Stage A's frozen Predictor paired with
Stage B's LoRA weights) — Stage B never constructs a Predictor (`semantic_weight=0`), so this
needs a small standalone script, not yet written.

Output dirs: `/workspace/semsup/p1_stageA/`, `/workspace/semsup/p1_stageB/`. Local copies:
`outputs/e4_vjepa_reason/p1_stageB/{test_summary.json, epoch_metrics.jsonl, train_metrics.json,
test_results_ep0{1,2}.jsonl, bootstrap_vs_a1_1761.json}`.


---

# Per-clip arm comparison over the 1,761-window training pool (2026-08-23/24)

**Motivation.** Aggregate AP says which arm is better; it cannot say *which clips* move. Every
arm had only ever been scored on the 677-clip test set, so per-window training-pool scores did
not exist for any of them.

**Run.** `score_arms_on_pool1761.py`, 6 configurations × 1,761 windows, **inference only**, on
the pod. ~5 min per arm. A0 = frozen baseline, no adapter attached. Checkpoints used are the
same epochs the reported test numbers came from: A1 `a1_1761/epoch_04`, B-v1
`b_1761_par/epoch_04`, B-v2 `b_v2_1761/epoch_02`, **B-v3 `b_v3_1761_ext12/epoch_10`** (note:
the ext12 directory, not `b_v3_1761/` — that one only holds epochs 1–8), P1
`p1_stageB/epoch_02`. All 6 returned 1,761/1,761 rows, zero skips.

**Integrity check (the strongest one available):** by construction A0 must be wrong on exactly
the 587 mined-failure windows and right on the other 1,174. The re-score reproduces this
**exactly**, confirming the scoring path matches the original mining run.

## Pool and split structure

```
1,761 windows = 587 mined A0-failures + 587 TP + 587 TN, from 1,107 unique clips
  (578 clips give 1 window, 404 give 2, 125 give 3 — TTE_0.5/1.0/1.5 or MID-10/-8/-4)
split by CLIP: 221 val clips -> 348 val windows / 1,413 train windows
  of the 348 val windows: 117 are mined failures (51 YES / 66 NO), 231 are easy
```

Split is identical across all five trained arms (verified: V10 and V12 caption files contain
the same 1,107 video_ids, so `clip_level_split(val_frac=0.2, seed=0)` partitions them
identically). A0 never trained on any of it.

## Train/val, per arm (threshold-free AP)

| Arm | train AP | val AP | gap | train acc | val acc |
|---|---|---|---|---|---|
| A0 | 0.8435 | 0.8579 | −0.014 | 0.667 | 0.664 |
| **A1** | 0.9395 | **0.8770** | 0.063 | 0.839 | 0.730 |
| B-v1 | 0.9409 | 0.8751 | 0.066 | 0.850 | 0.739 |
| B-v2 | 0.8972 | 0.8670 | 0.030 | 0.786 | 0.764 |
| **B-v3** | **0.9990** | 0.8741 | **0.125** | **0.955** | 0.759 |
| P1 | 0.9326 | 0.8575 | 0.075 | 0.834 | 0.753 |

**B-v3 has memorised the training rows (train AP 0.9990).** Its large fix counts on train rows
are recall of memorised labels, not capability. On val it sits *below* A1.

**Accuracy and AP disagree here, and AP is right.** At threshold 0.5 B-v3 makes fewer val
errors than A1 (84 vs 94 of 348) yet has lower AP. Cause: B-v3 is extremely confident in both
directions (median val score 0.991 on YES, 0.002 on NO — spread 0.989, the widest of any arm,
wider than A0's 0.979), so its errors are *confident* errors, which AP punishes heavily.
⚠️ B-v3 is **not** "scoring everything lower" — an earlier characterisation that the data
contradicts.

⚠️ **The pool is adversarial against A0 at threshold 0.5 by construction** (one third selected
*because* A0 fails it), so every arm beats A0 on accuracy almost automatically. Accuracy
comparisons against A0 on this pool carry little information.

## The core finding: false-alarm recovery up, missed-crash recovery down

On the **677-clip held-out test set** (A0's 130 false alarms + 30 misses), Wilson 95% CIs:

| Arm | recovers false alarms | recovers **missed crashes** |
|---|---|---|
| **A1** | 21.5% [15.3, 29.4] | **56.7%** [39.2, 72.6] |
| B-v2 | 44.6% [36.3, 53.2] | 10.0% [3.5, 25.6] |
| **B-v3** | **60.0%** [51.4, 68.0] | 20.0% [9.5, 37.3] |

**McNemar, A1 vs B-v3 on A0's 30 misses (paired, same clips):**
`A1 right & B-v3 wrong = 11`, `B-v3 right & A1 wrong = 0`, **p = 0.0026.**
B-v3's correct set is a strict subset of A1's — a pure loss on the safety-critical axis.

Replicated on the independent 348-window val split (117 mined-failure windows there):

| Arm | correct / 117 | FP-type / 66 | FN-type / 51 |
|---|---|---|---|
| A0 | 0 | 0 | 0 |
| A1 | 37 | 14 | **23** |
| B-v1 | 40 | 18 | 22 |
| B-v2 | 44 | 41 | 3 |
| B-v3 | **57** | **46** | 11 |
| P1 | 50 | 38 | 12 |

## Regression: what the arms BREAK

Of the 231 A0-correct windows in val:

| Arm | broke | of which were YES (**detected crash → miss**) |
|---|---|---|
| A1 | 14 | 4 (29%) |
| B-v2 | 9 | 8 (89%) |
| **B-v3** | **24** | **21 (88%)** |
| **P1** | 19 | **19 (100%)** |

Semantic arms damage A0's correct predictions almost exclusively by converting detected
collisions into misses.

## Accuracy by horizon (val)

| Bucket | A0 | A1 | B-v3 |
|---|---|---|---|
| MID-10 / MID-4 / MID-8 (safe) | 58–68% | 58–71% | **84–93%** |
| TTE_0.5 | 88% | 92% | 87% |
| TTE_1.0 | 73% | **88%** | 69% |
| **TTE_1.5** (earliest warning) | 52% | **66%** | **41%** |

Semantic arms win on safe windows, collapse on crash windows, worst at the longest horizon.

## Val vs test gains — NOT overfitting to val

| Arm | gain vs A0 (val) | gain vs A0 (test) |
|---|---|---|
| A1 | +0.019 | **+0.047** |
| B-v1 | +0.017 | +0.037 |
| B-v2 | +0.009 | +0.027 |
| B-v3 | +0.016 | +0.024 |
| P1 | −0.000 | **−0.026** |

Every arm gains *more* on test than on val — the val pool merely looks worse because it is
failure-enriched. **P1 is the only arm that goes backwards on test.**

## Label noise in the pool (measured 2026-08-24)

Using the teacher's own admissions, independent of any model of ours. Note the V10 schema
differences: **blind-mode rows carry `verdict`/`confidence`/`risk_score`; gt-mode rows do not
(the teacher was told the answer); V12 dropped verdict entirely.**

| Signal | Mined (587) | Easy (1,174) |
|---|---|---|
| **Positives with `mechanism_visible=false`** — teacher told GT=crash, still saw no mechanism | **36 / 269 (13.4%)** | 20 / 587 (3.4%) |
| Blind-mode verdict disagrees with the label | 9 / 318 | 5 / 587 |

⚠️ `mechanism_visible=false` on a **negative** is normal (44–59% of them) — a safe clip has no
collision mechanism. Only positives are meaningful. **Unexplainable positives are ~4× enriched
in the mined failures**, consistent with off-camera impacts or mislabels. A0 scores 0.001–0.007
on several of these YES-labelled clips.

Also measured: **230 / 587 (39.2%)** of mined failures are windows where A0 is >0.95 confident
and the label disagrees. This is *not* a usable filter — removing them would be circular.

**70 suspect windows** (45 mined + 25 easy, 13 in val) itemised with frame paths, both caption
versions and A0's score in `outputs/e4_vjepa_reason/suspect_windows_for_review.xlsx`.

## Outputs

| Path | Contents |
|---|---|
| `outputs/e4_vjepa_reason/pool1761_scores/{A0,A1,B-v1,B-v2,B-v3,P1}.jsonl` | 1,761 per-window scores per arm |
| `outputs/e4_vjepa_reason/pool1761_arm_comparison.xlsx` | 4 sheets: `per_clip`, `summary`, `val_only`, `failures_only` |
| `outputs/e4_vjepa_reason/pool1761_findings_2026-08-24.md` | Findings write-up incl. figure/sheet walkthroughs |
| `outputs/e4_vjepa_reason/suspect_windows_for_review.xlsx` | The 70 label-noise candidates |
| `reports/figures/pool1761_analysis/` | 14 figures (7 × `_all1761` / `_val348`) |

**Read the `_val348` figures, not `_all1761`** — the latter includes memorised training rows.

## Captioning cost, recomputed from the real usage log

Measured per caption on the V12 run: **18,756 input tokens** (image-dominated) + **1,150
output**. At `gemini-3.7-flash` batch rates ($0.1875/M in, $0.9375/M out) that is **$0.0046
per caption → $20.43 for all 4,446 windows**, versus the $0.0365/caption the V12 run actually
paid (= $162 for 4,446).

## Calibration re-analysis of the 1,761-pool arms (2026-08-27) -- corrects the leading hypothesis
Prompted by the "broken" columns showing near-100% TP-to-FN breakage vs A1. Measured on val
(n=348, 170 YES/178 NO):

| arm | AP | AUC | Cohen's d | mean YES | mean NO | own optimal threshold | acc@0.5 | acc@own-threshold |
|---|---|---|---|---|---|---|---|---|
| A0 | 0.8579 | 0.8372 | -- | 0.727 | 0.332 | 0.979 | 0.664 | 0.767 |
| A1 | 0.8770 | 0.8621 | 1.505 | 0.801 | 0.353 | 0.812 | 0.730 | 0.782 |
| B-v1 | 0.8751 | 0.8622 | 1.530 | 0.792 | 0.348 | 0.749 | 0.739 | 0.779 |
| B-v2 | 0.8670 | 0.8544 | 1.564 | 0.684 | 0.246 | 0.541 | 0.764 | 0.782 |
| B-v3 | 0.8741 | 0.8602 | 1.352 | 0.644 | 0.143 | 0.173 | 0.759 | 0.779 |
| P1 | 0.8575 | 0.8443 | 1.476 | 0.676 | 0.218 | 0.368 | 0.753 | 0.773 |

AP/AUC/Cohen's-d spread is inside CI width at this n for every arm -- ranking quality is
unchanged. What moves is the optimal threshold (0.812 to 0.173 across arms) -- a pure monotone
rescaling, invisible to AP/AUC (Guo et al., ICML 2017), fully visible at any fixed cut like 0.5.
Re-deriving fixed/broken at each arm's OWN threshold instead of 0.5 (vs A1 at 0.812):

| arm | thr* | brokenFN@thr* | brokenFP@thr* | fixedFN@thr* | fixedFP@thr* | net@thr* |
|---|---|---|---|---|---|---|
| B-v1 | 0.749 | 4 | 4 | 5 | 2 | -1 |
| B-v2 | 0.541 | 9 | 5 | 7 | 7 | 0 |
| B-v3 | 0.173 | 6 | 18 | 17 | 6 | -1 |
| P1 | 0.368 | 5 | 21 | 20 | 3 | -3 |

B-v3's "31 broken crashes" (at shared 0.5) becomes 6 at its own threshold, and net collapses to
about 0 for every arm. Conclusion: the semantic arms did not un-learn crash detections -- the
frozen crash head just never recalibrated to LoRA's shifted feature distribution. See
DECISIONS.md for the corrected/refuted mechanism entry and PROJECT_STATE.md for the full
write-up. This is the direct motivation for --unfreeze-head (below).

## SemTest-200 -- 4-arm controlled experiment with an unfrozen crash head (2026-08-26/27)
Setup: 200 windows (160 train/40 val, one window per video), selected via
select_semtest200_recovery.py's 3-tier priority fill from a fresh A0 re-score of the full
4,446-window pool (outputs/semtest200/A0_full4446.jsonl, integrity-verified: reproduces the
known 587 mined failures exactly). Positive tiers: FN near-boundary [0.3,0.5) RT-eligible (all
of them), then TP fill [0.5,--tp-fill-max=0.85) lowest-score-first, then FN wide (<0.3)
highest-score-first, filled tier-globally across all 3 TTE buckets at once (a per-bucket loop
starved TTE_1.5 by video-sharing across buckets -- fixed). Negative tiers: FP near-boundary
[0.5,0.7) (all of them) then FP fill [0.7,1.0) lowest-first -- 100% FP by design, zero TN.
Captions: V10 (leaky), V12 (clean), V12-shuffled (make_semtest200_shuffled.py -- derangement
within class, seed 0). 164/200 captions reused from the 1,761-pool corpus; 36 newly generated
(discovered mid-flight: 22 of those 36 were accidentally captioned with the WRONG teacher --
gemini-3.1-pro-preview via the DEFAULT_MODEL bug -- regenerated on gemini-3.7-flash before
training; see PROJECT_STATE.md).

Training: all 4 arms identical except --captions-path/--semantic-weight:
--lora-target-modules query,key,value --lora-r 16 --lora-alpha 32 --lora-dropout 0.05
--unfreeze-head --head-lr-mult 0.1 --clip-grad-per-group --lr 1e-4 --lr-schedule cosine
--warmup-frac 0.05 --epochs 10 --keep-top-k 10 --seed 0 --val-video-ids <fixed 40-clip val>
--grad-cosine-every 8 --dump-val-scores; semantic arms add --semantic-loss infonce
--semantic-weight 0.2 --infonce-tau-init 0.07. Dry-run gate (--epochs 1, vision + semantic
paths) passed before the real batch; ran sequentially on one pod (concurrent BADAS loads risk
crashing each other, documented gotcha).

Results (val, n=40, threshold 0.5 unless noted):

| arm | selected epoch | val AP | val AUC | acc@0.5 |
|---|---|---|---|---|
| vision | 8 | 0.5424 | 0.5025 | 0.525 |
| v10 | 8 | 0.5204 | 0.4975 | 0.450 |
| v12 | 10 | 0.5154 | 0.4900 | 0.475 |
| v12shuf | 10 | 0.5172 | 0.4900 | 0.475 |

Full val_ap trajectories (all 10 epochs, monotonic-ish rise to a plateau, no earlier peak --
epoch selection is not masking a better checkpoint):
```
vision   0.4644 0.4545 0.5045 0.4826 0.5155 0.5201 0.5345 0.5424 0.5416 0.5416
v10      0.4823 0.4105 0.4599 0.4687 0.4887 0.4956 0.5117 0.5204 0.5193 0.5193
v12      0.4823 0.4088 0.4589 0.4699 0.4892 0.5017 0.5089 0.5086 0.5154 0.5154
v12shuf  0.4812 0.4105 0.4580 0.4717 0.4955 0.5047 0.5150 0.5170 0.5172 0.5172
```
Train AUC 0.85-0.87 for every arm (pure memorization). Mean score by source-tier, val (A0 vs
vision vs v12) shows every arm regressing scores toward 0.4-0.55 regardless of true label
(TP_fill correct-highs drop toward 0.5; FN_wide confident-lows barely rise past 0.35-0.40) --
the regression-to-mean signature, not class-conditional learning.

Primary endpoint -- paired per-clip delta (delta_arm minus delta_vision, signed toward truth) on val:

| arm vs vision | mean signed delta | sign test | Wilcoxon p |
|---|---|---|---|
| v10 | -0.0043 | 17 vs 23, p=0.43 | 0.52 |
| v12 | -0.0046 | 20 vs 20, p=1.00 | 0.49 |
| v12shuf | -0.0041 | 20 vs 20, p=1.00 | 0.51 |

No arm beats vision-only; v10/v12/v12shuf are statistically indistinguishable from each other
-- v12 approx v12shuf is the cleanest available evidence that caption content isn't reaching
the score at this scale, confirmed structurally in the summary_vs_vision sheet (fixed/broken/net
numerically identical between v12 and v12shuf on val).

Confound found post-hoc (code review, 2026-08-27) -- the run doesn't cleanly test an open
head: head_state.pt's total L2 norm agrees to 4 decimal places across all 4 differently-
trained arms (70.3568 vision vs 70.3566 for the other three); the final classifier bias moved
by about 1e-6. LoRA moved fine for comparison (144/144 lora_B tensors nonzero, zero-init to
mean-norm 0.114-0.118 -- real signal). Mechanism: head LR=1e-5, cosine-decayed alongside the
trunk toward 0, clip-grad-per-group budget 1.0/step -- Adam's per-step movement at this LR over
200 steps is bounded to about 1e-4 to 2e-3 in each parameter's own units, matching the
measurement exactly. The head was unfrozen in name, not in practice.

Outputs: outputs/semtest200/ -- selection.jsonl, Caption_semtest200_{V10,V12,
V12_shuffled}.jsonl, per-arm results/{arm}/{epoch_metrics.jsonl,val_scores_ep*.jsonl,
train_metrics.json}, scores/{A0,vision,v10,v12,v12shuf}.jsonl,
semtest200_arm_comparison.xlsx (per_clip/summary_vs_A0/summary_vs_vision/metrics sheets),
figures/{loss_curves_2x2,val_ap_vs_epoch}.png, code_review_findings_2026-08-27.md (full
correctness audit + ML-design critique + literature review).

## Architecture literature review (2026-08-27)
Full report + reference list in outputs/semtest200/code_review_findings_2026-08-27.md (Part
C). Verdict: abandon trunk-level SigLIP-InfoNCE alignment as an accuracy-lift mechanism;
retarget language supervision to post-hoc explanation. Key evidence: (1) Nexar's own BADAS/
BADAS-2.0 fully fine-tune V-JEPA2 end-to-end at 178,500 labeled videos for their accuracy
gains -- LoRA-on-frozen-trunk is a reasonable small-data compromise, not the bottleneck; (2)
CLIP-style contrastive alignment (LiT, SLIP) is only validated at hundreds-of-millions-of-pairs
scale, 5-6 orders of magnitude above this thesis's corpus; (3) SigLIP-family text encoders are
documented (ARO, Winoground) to behave close to bag-of-words, missing exactly the relational/
motion semantics ("closing distance") this task needs; (4) the shuffled-caption control
empirically confirms no signal is being extracted, not merely a mis-weighted one -- there is
nothing for gradient-conflict mitigation (PCGrad/GradNorm) to rescue.

## V13 causal-caption redesign -- full 4,446-window pool (2026-08-27/28)
Motivation, measured before writing the prompt: SigLIP's real limit is 64 tokens (not the
about-40-word V12 rule); across all 2,161 existing V10/V12 captions, max is 43 tokens, 0%
truncated -- about 3x headroom unused. Caption-length-vs-SigLIP-distinctiveness correlation on
the existing V12 corpus: -0.0017 (zero) -- more words of the same KIND of content don't
separate better.

Prompt (prompts/PROMPT_SEMSUP_V13_CAUSAL.py): V12's anti-leak machinery (blind, closed
vocab, symmetric bans) plus 5 new closed-vocabulary causal-cue fields (lead_vehicle_lighting,
ego_maneuver, road_geometry, signal_state, occluded_or_peripheral), colour banned.
First iteration used a word CEILING only (<=45) and "at least 1 causal cue" -- a 15-clip gate
measured mean 26.7 words/30.4 tokens, half the intended budget, fields recorded but not
verbalized. Fixed to a 42-52 word FLOOR+ceiling band and mandatory verbalization of every
populated field (validate_parsed's v13 branch: word-count check + per-field keyword-coverage
check via a _COVERAGE dict) before the full-pool spend.

Also hit for real: semsup_caption_promptbakeoff.py:87's stale DEFAULT_MODEL =
"google/gemini-3.1-pro-preview" silently captioned 36 SemTest-200 clips with the wrong teacher
when --model was omitted. Fixed: default changed to google/gemini-3.7-flash.

Full run: 4,446/4,446 windows, gemini-3.7-flash, pinned to the Google Vertex provider
(--provider-order google-vertex, allow_fallbacks=False) for a 75%-off launch discount
confirmed live via real per-call billing ($0.375/M in / $1.875/M out, vs the base
$1.50/M/$7.50/M). Concurrency 16. Wall time 3,921s (about 65 min), cost $24.85 ($0.0056/
clip real, stable across the whole run, no jump at the midnight boundary into 28/08). Zero
failures. 4,446 unique frames_dir, 2,223/2,223 class balance. 28/4,446 (0.6%) rows exceed the
58-token cap (reported by --token-cap 58, not auto-regenerated).

QC:
- Leakage gate: AUC=0.7774 (up from V12's 0.7640 -- expected: top predictive n-grams are
  "brake lights"/"distance decreasing"/"lead sedan", genuine causal signal, not register leak).
- Mean words 45.9, mean SigLIP tokens 51.5 (target band hit).
- Decisive check FAILED its pre-registered go/no-go: mean cross-caption SigLIP cosine
  0.7974 (worse than V12's 0.7010); mean distinctiveness 0.2026 (vs V12's 0.3003,
  -32.5%).
- Root cause diagnosed: 73.5% of all 4,446 captions open with the literal phrase "Ego moves
  straight...", 16.8% with "Ego travels straight...", 6.6% with "Ego remains stopped..." --
  96.9% total share one of 3 near-identical 3-word openers. Top-20 words account for 47.8%
  of all tokens in the corpus. Template collapse from the prompt's one worked example plus
  "always verbalize ego_maneuver" instruction, not genuine content homogeneity -- SigLIP's
  bag-of-words sensitivity (per the literature review above) means this dominates the
  embedding regardless of what differs afterward in each sentence.

Outputs: outputs/semantic_captions/v13/{raw_v13_4446.jsonl, Caption_V13_Causal_4446_fortrain
.jsonl, Caption_V13_Causal_4446.xlsx, leakage_gate_v13.json, raw_v13_gate15.jsonl,
v13_gate15_review.xlsx}.

Not yet done (open decision, see DECISIONS.md/PROJECT_STATE.md): fix the opener-template-
collapse and re-gate before any full re-run, or stop here and report the failed distinctiveness
check as a completed negative result.

## `--head-lr-schedule` fix + SemTest-200-v2 (2026-08-29) -- secondary data point

`semsup_train.py` gains `--head-lr-schedule {cosine,constant}` (default `cosine` = old
behavior; `constant` keeps the head's LR flat after warmup instead of decaying it alongside the
trunk's shared cosine schedule) -- fixes SemTest-200-v1's bug where `--unfreeze-head` moved the
head <0.05% relative magnitude over 200 steps because its already-small LR (0.1x the trunk's)
was ALSO decayed by the trunk's shared schedule. `head_lr` now logged per epoch in
`epoch_metrics.jsonl` as an audit trail. Also `--bank-captions <corpus>` (widens the InfoNCE
train bank with extra distractors from a wider corpus while preserving each anchor's own
`_bank_idx` position -- must append after the anchor's own-caption block, never replace it).

SemTest-200-v2 (4 arms, 300-clip pool = the original 200-clip pool + 100 easy A0-correct anchor
clips added via `select_semtest200_easy.py`/`merge_semtest200_v2.py`/
`merge_semtest200_v2_captions.py`, addressing SemTest-200-v1's 100%-adversarial/zero-true-
negative composition) fold-1 was run with the head-LR fix applied. Result: same qualitative
finding as SemTest-200-v1 -- v12approxv12shuf, no transfer. This result is **superseded in
relevance by the A1-failure-recovery run below**, which answers the same underlying question
(does semantic supervision transfer once the confounds are removed?) via a different, cleaner,
mechanism-explained route -- kept here as a secondary/earlier data point, not the headline.
Also fixed this session: `aggregate_semtest200_cv.py`'s `metrics()` read a `gt_verdict` key
that doesn't exist in `--dump-val-scores` output (which actually uses `label`, int 0/1) --
fixed. Also fixed (pre-existing, unrelated): unescaped `%` in `semsup_train.py`'s argparse help
strings crashed `--help` entirely (adjacent string-literal concatenation produced runtime
content like `"...0.53%" + "of..."`) -- now `%%`-escaped throughout.

New CV infrastructure: `make_semtest200_folds.py` (stratified 5-fold split by `video_id` +
source tier, self-asserts exact partition), `aggregate_semtest200_cv.py` (pools per-fold
val_scores into a full-pool readout), `plot_semtest200_cv_curves.py` (mean+-std-band loss
curves across folds, shared y-axis, right axis color-keyed to its own series, `--mark-epoch`/
`--init-note` for annotating a selected checkpoint), `siglip_bottleneck_probe.py` (measures how
much crash-relevant signal survives text->SigLIP-embedding vs raw text; ran on V10/V12/V13
corpora -- SigLIP retains 86-96% of the text's own crash-AUC, ruling out the encoder as the
bottleneck for prior negative results).

## A1-failure-recovery -- starting from A1's own 321 test-pool-style failures (2026-08-29)

**Question**: starting from A1 (crash-only LoRA, current champion, test AP=0.900/AUC=0.904 on
677 clips), can semantic supervision recover the specific clips A1 gets wrong, and does trying
damage the headline test score?

**Pool**: all 321 windows (240 unique videos) A1 scores wrong at threshold 0.5, from the
1,761-pool. A1's own AUC on this pool is **exactly 0.0 by construction** (every row is on the
wrong side of the boundary -- expected, not a bug, and must be stated whenever this pool's
in-pool numbers are read). Split 260 train / 61 val by `video_id` (seed 0). Selection script:
`select_a1fail321.py`, writes `outputs/a1fail321/selection_a1fail321.jsonl` + per-arm caption
files (`Caption_a1fail321_{V10,V12,V12_shuffled}.jsonl`, 321 rows each, joined from the existing
1,761-pool V10/V12 corpora plus 72 freshly-captioned clips where needed). Real captioning cost
for this specific 321-pool was **$0** -- every needed caption already existed in the 1,761 corpus.

**4 arms**, all initialized from A1's own LoRA weights
(`/workspace/semsup/a1_1761/epoch_04/lora_adapter`, r=16/alpha=32/dropout=0.05, config verified
against `adapter_config.json` before loading), head **frozen** (deliberate -- not unfrozen; see
DECISIONS.md), predictor warm-started from B-v3's B1 checkpoint
(`/workspace/semsup/b1_v2_100pct/predictor_b1.pt`, the same one B-v3 used, shared across all 3
semantic arms to hold initialization constant and vary only the caption file): `a1cont`
(crash-only control, `--semantic-weight 0.0`), `v10` (leaky captions), `v12` (clean captions),
`v12shuf` (v12 captions shuffled within class -- content-vs-presence control). Config: `--lr
2e-5` (5x below A1's own 1e-4 -- refining, not retraining from scratch), `--lr-schedule cosine
--warmup-frac 0.1 --epochs 10 --keep-top-k 10 --semantic-weight 0.2` (3 semantic arms),
`--bank-captions` = each arm's own full 1,761-row corpus (v12shuf banks against a
freshly-shuffled 1,761 corpus, NOT the unshuffled one -- banking against the wrong corpus would
silently break the content-vs-presence control). Driver `run_a1fail321_4arms.sh` runs the 4
arms strictly sequentially (concurrent BADAS loads can crash each other -- known gotcha). Ran on
RunPod, 1 fold only (fold_01), all 4 arms, results in `outputs/a1fail321/results/<arm>/fold_01/`.

### RESULT 1 (in-pool, val split, 61 clips) -- ALL FOUR ARMS BIT-IDENTICAL

fixed_FP=39, fixed_FN=0, still_wrong=22, acc@0.5=0.6393 -- literally the same per-clip
predictions whether there's no semantic branch at all, real captions, or scrambled captions.
AP/AUC vary by ~0.02 (noise at n=61): a1cont AP=0.1941/AUC=0.1190, v10 AP=0.1920/AUC=0.1040,
v12 AP=0.1914/AUC=0.0990, v12shuf AP=0.1937/AUC=0.1159. Workbook:
`outputs/a1fail321/a1fail321_arm_comparison.xlsx` (built by `build_a1fail321_comparison.py`).
Read the in-pool AP/AUC values with the pool's AUC=0.0-by-construction caveat always attached.

### RESULT 2 (predictor health -- the semantic branch demonstrably WORKS, mechanistically)

v10/v12 retrieval@1 reaches 35-44% (peak across the run) vs a 2.1% collapse control (same
magnitude ballpark both arms). v12shuf sits at ~0.0% (at or below its own collapse control) for
the ENTIRE run -- the cleanest real-vs-scrambled separation this project has produced. This is
the opposite of the earlier SemTest-200 (pre-A1fail321) result where the predictor was
collapsed at chance for ALL arms including real captions -- the difference is this run's wider
InfoNCE bank (full 1,761 rows via `--bank-captions`, vs only 160 train-split captions in the
earlier run) plus the warm start from B-v3's B1 checkpoint.

### RESULT 3 (test set, 677 clips, the number that actually matters)

Via new script `score_checkpoints_on_test.py` (loads BADAS once, swaps LoRA adapters between
checkpoints; uses `softmax(logits)[0,1]` with NO `/2.0` divisor, unlike
`e4_stageA_badas_open_eval.py`'s published-scorer convention -- confirmed this does NOT affect
AP/AUC or the confusion matrix at threshold 0.5, since dividing logits by a constant is a
monotone transform that preserves the 0.5 crossing point; it would only matter for calibration
metrics at other thresholds). Scored A1 itself through this same scorer as a validation check
(reproduced 0.8995/0.9034, matching its documented 0.900/0.904 to 3 decimals -- confirms the
scorer is trustworthy) and v12's epoch-10 checkpoint:

**⚠️ Corrected 2026-09-08: the 0.8995 vs 0.9000 gap is NOT the temperature convention.**
`a1_1761/test_results_ep04.jsonl` (0.9000/published) and this section's own A1 validation-check
score (0.8995) are the SAME checkpoint scored on the SAME 677 clips by two divisor-free scorers
(`semsup_train.py` and `score_checkpoints_on_test.py` both use bare `softmax(logits)[0,1]`, no
`/2.0` -- confirmed by reading both scorers directly). Measured directly: 677/677 clips differ
between the two files (mean |delta| 0.0078, max 0.0973), 5 clips flip the 0.5 decision, and the
signed mean is +0.00118 (median +0.00001) -- zero-centred noise, not a systematic divisor
effect, consistent with unmanaged GPU-inference nondeterminism across the two scoring runs
(neither had `torch.use_deterministic_algorithms` set at the time). `--deterministic` (default
on) was added to both scorers 2026-09-08 to close this gap going forward. See
`ARCHITECTURE.md`'s matching correction and the 2026-09-06 project review §3.1.

| arm | n | AP | AUC | acc@0.5 | acc@own-best-threshold |
|---|---|---|---|---|---|
| A0 | 677 | 0.8530 | 0.8642 | 0.7637 | 0.8287 (thr 0.68) |
| A1 | 677 | 0.8995 | 0.9034 | 0.7903 | 0.8287 (thr 0.68) |
| v12 | 677 | 0.8972 | 0.9027 | 0.8168 | 0.8331 (thr 0.43) |

v12 vs A1: AP -0.0023, AUC -0.0007 -- flat, within noise. acc@0.5 looks like +2.65pp for v12
but this is **calibration, not discrimination**: v12's mean test score is 0.488 vs A1's 0.660
(training shifted the whole distribution down, landing it near 0.5 by coincidence); at each
arm's own optimal threshold the gap collapses to +0.004. Score files:
`outputs/a1fail321/test_scores/{A1,v12_ep10}.jsonl`.

### Mechanism -- why semantic supervision doesn't transfer, despite genuinely working

`grad_cos_mean` (cosine between crash-loss and semantic-loss gradients on the shared LoRA
params, logged via the existing `--grad-cosine-every 8`) sits at -0.04 to +0.05 across the whole
run, sign-flipping epoch to epoch, in ALL THREE semantic arms. This is 10-100x above the
pure-random-orthogonality floor for a ~2.8M-param space (so not literally independent) but far
below what conflict would look like (persistently negative cosine, `frac_neg` -> 1.0).
Interpretation: the two objectives want mildly overlapping but mostly orthogonal features --
captions are a lossy function of the same 16 frames the student sees, so semantic supervision
was never adding NEW information, only a reorganization pressure, and that pressure happens to
point in a direction the frozen crash head's fixed linear readout is largely blind to.

### Conclusion

A1's 0.900 test benchmark survives training on its own failures intact -- no catastrophic
forgetting (the real risk this run tested for). Semantic supervision neither helps nor hurts it.
The semantic branch is now proven to work end-to-end (retrieval, real-vs-shuffled separation)
for the first time this project -- so the null is a clean, mechanistically-explained
non-transfer, not an artifact of a broken predictor (which is what earlier runs in this project
could not rule out).

### Presentation deck

`reports/presentations/2026-08_a1-failure-recovery.pptx`, generator
`build_a1fail_presentation.py` -- 6 slides, house style matching the existing `2026-08-22` deck
(reuses its palette/helper-function conventions, does not modify that file). Has a `verify()`
gate that re-derives every embedded number from the actual score/result files and asserts
against known-good values before writing -- run it after ANY score-file change, never hand-edit
numbers into the deck. Also regenerates `make_arch_figures_2026-08-22.py`'s `fig_L3()` (now
parameterized by `lam`/`out_name` so a different `semantic_weight` can be drawn without
overwriting the original 0.05-weight figure other decks depend on) -- produced
`reports/figures/arch_L3_training_a1fail_2026-08-29.png` (lambda=0.2 variant).

## Recovery family on the 677-clip test set — the content-vs-presence control (2026-09-05)

`score_checkpoints_on_test.py`, one BADAS load + three adapter swaps, epoch 10 for all.
Scores: `outputs/a1fail321/test_scores/{a1cont,v10,v12shuf}_ep10.jsonl`, 677 rows each.

**⚠️ Correction 2026-09-08**: "epoch 10 is each arm's own best `val_ap`" is not quite right --
re-derived from `epoch_metrics.jsonl` (project review §4.3): a1cont and v12 tie epochs 8/9/10,
v12shuf's true best is epoch 10, but **v10's best is epoch 9** (val_ap 0.2096 vs epoch 10's
0.2094) -- the 0.0002 gap is immaterial but the stated justification was inaccurate. Separately,
these curves move only ~0.01 total across 10 epochs and are tied to 4 decimals across the top 3
epochs for 3 of 4 arms -- checkpoint selection here is close to arbitrary at this scale.

| arm | captions | AP | AUC | acc@0.5 | TP | FN | FP | TN |
|---|---|---|---|---|---|---|---|---|
| A1 (reference) | — | **0.9000** | 0.9042 | 0.7917 | 320 | 18 | 123 | 216 |
| a1cont | none | 0.8956 | 0.9011 | 0.8021 | 238 | 100 | 34 | 305 |
| V10 | real (GT-conditioned) | 0.8976 | 0.9033 | 0.8168 | 253 | 85 | 39 | 300 |
| V12 | real (neutral) | 0.8972 | 0.9027 | 0.8168 | 253 | 85 | 39 | 300 |
| **v12shuf** | **scrambled within class** | **0.8963** | 0.9017 | 0.8080 | 244 | 94 | 36 | 303 |

## Bootstrap CIs + the measurement-precision correction (2026-09-09, Phase B of the
## post-review remediation plan) — READ THIS BEFORE CITING THE TABLE ABOVE AS A RANKING

The 2026-09-06 project review (§3.1) found the pipeline's own run-to-run scoring noise is
comparable in size to the differences this table reports as a ranking. This section closes
that out with real numbers: `paired_bootstrap_ab.py` (fixed 2026-09-09 to also read
`gt_verdict`-schema files — it previously read only `ground_truth`, so it could not run on
`outputs/a1fail321/test_scores/*.jsonl` at all) was run pairwise across the whole family, plus
a direct measurement of the noise floor itself.

**The noise floor**, measured from two independent scoring runs of the literal SAME A1
checkpoint (`a1_1761/epoch_04`) on the SAME 677 clips —
`outputs/e4_vjepa_reason/a1_1761/test_results_ep04.jsonl` vs
`outputs/a1fail321/test_scores/A1.jsonl`:

| | value |
|---|---|
| ΔAP (run1 − run2) | **−0.0009** |
| 95% bootstrap CI | [−0.0035, +0.0009] — crosses zero |
| clips with any score difference | 677 / 677 |
| mean \|Δ\| per clip | 0.0078 |
| max \|Δ\| per clip | 0.0973 |
| clips flipping the 0.5 decision | 5 / 677 |

**The between-arm pairwise matrix** (all n=677, 5000-resample paired bootstrap, seed 42;
raw JSON in `outputs/a1fail321/bootstrap/`):

| pair (A − B) | ΔAP | 95% CI | excludes 0? | P(B>A) |
|---|---|---|---|---|
| **V12 − v12shuf** (the decisive control) | +0.0009 | [−0.0007, +0.0026] | no | 13.4% |
| A1 − a1cont | +0.0039 | [−0.0039, +0.0117] | no | 15.6% |
| a1cont − V10 | −0.0020 | [−0.0032, −0.0010] | **yes** | 100.0% |
| a1cont − V12 | −0.0016 | [−0.0033, −0.0001] | **yes (barely)** | 97.9% |
| a1cont − v12shuf | −0.0007 | [−0.0018, +0.0004] | no | 90.4% |
| V10 − V12 | +0.0004 | [−0.0010, +0.0017] | no | 29.2% |
| V10 − v12shuf | +0.0013 | [+0.0002, +0.0026] | **yes** | 1.1% |

**Reading this honestly requires separating two different kinds of uncertainty that are
easy to conflate.** The bootstrap CI answers "would this ΔAP survive scoring a *different*
random sample of 677 clips" (clip-sampling variance). The noise floor above answers a
different question: "would this ΔAP survive scoring the exact same 677 clips *again* with
the exact same weights" (scorer nondeterminism — neither scorer had
`torch.use_deterministic_algorithms` set at the time these arms were scored). A pairwise CI
excluding zero says nothing about the second source.

**The decisive result: V12 vs v12shuf's own bootstrap CI [−0.0007, +0.0026] already crosses
zero on its own terms** — sampling variance alone cannot rule out these two being identical,
independent of the noise-floor question. This is the number the thesis's content-vs-presence
claim rests on, and it is not resolvable in either arm's favor at n=677.

**Three pairs *do* have bootstrap CIs excluding zero** (a1cont vs V10, a1cont vs V12, V10 vs
v12shuf) — meaning those specific deltas would likely survive a different clip sample. But
every one of their point estimates (−0.0020, −0.0016, +0.0013) is within 1–2× of the single
measured noise-floor point estimate (−0.0009, itself inside a CI of half-width up to 0.0035),
and none of these five arms was scored more than once. With only one noise-floor sample in
hand, there is no way to tell whether 0.0009 is a typical draw from the scorer's own noise or
an unusually small one — so "excludes zero under resampling" cannot be read as "distinguishable
from what the SAME arm would show if scored again." **None of the pairwise orderings in this
family should be treated as established.** The specific claim that stays intact is the
weaker, correctly-hedged one already in this doc: caption content shows no measurable,
noise-floor-clearing effect — not the stronger claim that any one arm is reliably ranked above
another.

**What would actually resolve this**: `--deterministic` (default on, added 2026-09-08) removes
the scorer-nondeterminism source going forward — a re-score of all five arms under it would
produce numbers with no cross-run noise at all, at which point the bootstrap CIs above would be
the complete picture rather than a lower bound on the real uncertainty. That re-score is GPU
work, staged for the Phase D runbook, not run as part of this (local-only) research pass.

**What survives regardless of all the above** (measured independently of AP, immune to this
noise-floor concern since it is read directly off each arm's own confusion matrix, not compared
across arms via a difference statistic):

**Clip-level agreement.** Across a1cont / V12 / v12shuf the three arms make the *same* call on
**657 of 677 clips (97.0%)** — only 20 clips disagree at all (6 / 4 / 5 at TTE 0.5s / 1s / 1.5s,
5 among negatives). Of those, no clip is uniquely fixed by V12 at any TTE bucket; V12 is
uniquely right on 2 negatives, a1cont on 2 positives.

**V10 and V12 have IDENTICAL confusion matrices** (253/85/39/300) despite different caption
corpora — identical decisions at threshold 0.5, differing only in AP ranking (0.8976 vs 0.8972),
which per the above is itself noise-floor-level.

This is the test-set confirmation of what the recovery pool already showed: at epoch 10 on the
61 held-out recovery windows, a1cont, v10, v12 and v12shuf **all** score 39/61 = 0.6393. Two
independent evaluation sets, same conclusion: presence and content are indistinguishable at
this measurement precision, and no arm has been shown to reliably beat any other.

**Interpretation.** Combined with the prior mechanism results — B1 (caption info
survives the 2560→1 pooling at 22× chance), P3 (the semantic gradient reaches the
pooled representation at least as well as an equal-norm random perturbation, paired CI
excludes zero) and the near-zero `grad_cos` — the account is now closed on the
"does the signal arrive" question: **it arrives, it is decodable, and it does not
help.** A shuffled-caption arm matching the real one is the cleanest available
evidence that the effect is caption *presence* (a text-shaped auxiliary regulariser),
not caption *meaning*. Any future claim that language supervision helps this task has
to beat v12shuf, not A0.

Website: all ten arms now appear in the landing results table, the Cross-Experiment
Comparison test-set arm list, and the detail view.

## A1-compress256 — full-frame preprocessing, NEW CHAMPION (2026-09-14)

Separate thread from the semantic-supervision work above — see PROJECT_STATE.md's
top status block for the full evidence table and decision reasoning; this entry is the
experiment-log summary.

**Finding**: every arm above (A0 through v12shuf) was trained/scored on a preprocessing bug
- BADAS's V-JEPA2 processor center-crops to 256x256 by default, keeping only source
`x∈[321,953]` of a 1280-wide frame (~49% of width). Fix: `--preprocess compress256` (full
frame resized to 256x256, no crop) added to `preprocess_clip()` and threaded through the
trainer/scorers; `crop` stays default, verified byte-identical to every historical run.

| Arm | Preprocess | Recipe | Private AP | Public AP |
|---|---|---|---|---|
| A0 | crop | frozen, no training | 0.8535 | 0.8711 |
| **A0-compress256** | compress256 | frozen, no training | **0.9067** | **0.9042** |
| A1 | crop | LoRA r16/a32 q,k,v, crash CE only, 8ep | 0.8995 (this session's re-score) | 0.9083 |
| **A1-compress256** | compress256 | **identical recipe to A1**, only preprocess changed | **0.9128** | **0.9096** |

Bootstrap comparisons (`paired_bootstrap_ab.py`, 5000 resamples):
- A0 crop vs compress256, private: ΔAP +0.0532, 95% CI [0.0356, 0.0727] — **excludes zero**
- A0 crop vs compress256, public: ΔAP +0.0331, 95% CI [0.0133, 0.0547] — **excludes zero**
- A1-compress256 vs A1-recorded, private: ΔAP +0.0137, CI [-0.0015, 0.0292] — crosses zero
- A1-compress256 vs A1-crop, pooled private+public (n=1344): ΔAP +0.0071, CI [-0.0034, 0.0180]

**A1-compress256 is the new champion.** `compress256` is now the project default. Outputs:
`outputs/a1_compress256/` (train dir, all score files, bootstrap JSONs, `summary.md`,
`RUNBOOK_pod.md` — all gitignored, not in git history). Website: `A1-compress256` registered
in all three builders and `POOL1761_ARMS`.

A planned same-environment control run (`A1-crop-rerun`) was started then deliberately
cancelled when the pod was paused mid-run — see DECISIONS.md for why that was the right call,
not a loose end.

## Stage AA.1 — detection pipeline (v1 baseline 2026-09-14; v2b Stages 0–3 + generalization set 2026-09-19)

All runs local (RTX 1000 Ada 6 GB). Dev set = 18 held-out `val_e3a` clips (9 pos / 9 neg). Positives decode
[event − 3.5 s, event − 0.5 s] (~91 frames), windows TTE 1.5/1.0/0.5; negatives decode ~8 s, windows
MID-10/-8/-4.

### v1 baseline — G-DINO + horizon corridor (`outputs/aa1_smoke_18clips/`, commit `812e884`)
G-DINO-tiny 0.61–0.79 s/frame; horizon fit fell back on 6/18 clips → threat 0 on 3 of 9 positives
(00319/00077/00283). Rejected.

### v2b Stage 0 — YOLOPv2
`aa1_yolop_cache.py --all`, img 640, NMS IoU 0.45, cached down to conf 0.10 (was 0.30). 22–95 ms/frame;
all boxes class id 3 (vehicle only). Recall vs the v1 G-DINO tracks, IoU ≥ 0.4: mean 0.814 (was 0.776 at
cache conf 0.30); lowest 01737 0.14 (near-empty night highway), 00474 0.60, 00283 0.66.

### v2b Stage 1 — tracking
BoT-SORT `lost_track_buffer=90`; totals on the 18 dev clips: raw 1062 → stitched 676 (386 merges), 0 hood
tracks. `extend_edge_tracks` keeps a track alive with low-confidence boxes at the frame edge (00319 id3 now
tracked to the last frame, 12 boxes added; its crash car had genuinely started leaving the frame, box
grows 36→231 px tall).

### v2b Stage 3 — selection score (dev 18 + gen18)
Config in ARCHITECTURE.md. Steps that changed the numbers (dev set, partner = drafted GT, see PROJECT_STATE):
- box height alone: partner in top-5 23/23 visible windows, #1 14/23;
- path-overlap score (2026-09-17): top-5 20/23, #1 15/23 — small far in-lane boxes and stale frozen tracks
  outranked the partner (fixed by recency gate, closeness, shielding);
- closeness × side × (1+approach) with EMA (2026-09-18): top-5 23/24, #1 15/24 (00319 excluded);
- with a lane-line ego path (2026-09-18/19): top-5 26/27, #1 18/27 (00319 partner id3 counted).

### Generalization set (gen18, `outputs/aa1_v2_18clips/clip_list.json`)
9 pos + 9 neg drawn with seed 20260919 from the training pool (741 pos / 741 neg after excluding the 18
dev clips and all `test_*.jsonl` manifests, 1362 ids). Nothing was tuned on them.
- Positives: 00136, 00195, 00486, 00505, 00583, 00803, 00903, 00932, 01035.
- Negatives: 01075, 01131, 01140, 01196, 01478, 01532, 01696, 01862, 01939.
- Eye-labelled partners (5 clips, 15 windows): 00136 id0, 00486 id0, 00505 id9, 00583 id0, 00803 id1 →
  #1 in 15/15 windows, top-5 15/15 (before the last ego-path fixes; unchanged after).
- 00195: the car about to be hit never gets a box (late-entering car missed by the detector) — a recall
  failure, not a scoring failure. 00903/00932/01035 ambiguous by eye — not labelled.
- Ego-path problems found on gen18 (fixed/open list in PROJECT_STATE.md): 01075, 00932 fixed; 00486,
  00505, 01532, 01478 open.

### Combined result after the last fixes (36 clips)
- 14 clips with drafted/eye labels: partner #1 in 33/42 windows, top-5 in 41/42.
- Top-1 selection score median 0.59 over 54 crash windows vs 0.19 over 54 normal windows (not a crash
  classifier; the score picks objects only).
- Ego-path sources, e.g. 01075 lane_pair 31 / lane_single 184; 01552 lane_pair 204 (was 0); 00077 lane_pair
  86 / lane_single 4; 00486 held 88; 01532 held 132 + backfilled 112; 00529 default 93.
- Score-term diagnostics worth remembering: partner box heights in the last window are 172–591 px on the 9 dev
  positives (00283 truck only 30 px at TTE1.5); lane-overlap `s` was noise (values −10…−127), ρ was a
  constant 0/1 bug before 2026-09-17.

Outputs: `outputs/aa1_v2_18clips/` (see ARCHITECTURE.md file table).

## Stage AA token-relevance auxiliary loss (2026-09-19 to 2026-09-24) — negative result

Question: does a per-token BCE aux loss, teaching the ViT-L's intermediate tokens which detected
car is relevant to a possible collision, beat **A1-compress256** (test AP 0.9128 private / 0.9096
public — the reference control throughout)? Architecture/data-flow: ARCHITECTURE.md. Design
history: `~/.claude/plans/CCP based BADAS/2026-09-19_Child-Plan-AA-token-relevance-aux.md`.

### Groundwork
- **AA.2 detection**: `aa1_run_set.py --set pool1761 --stages 0,1` on the full 1,761-window
  training pool = 1,107 unique videos (543 pos / 564 neg, counted from the manifest, not the
  "~587" estimate in an earlier plan draft), local, `outputs/aa1_pool1761/` (1.7 GB).
- **AA.4 labels**: `aa4_token_labels.py`, 1,761/1,761 windows, `dataset/aa_token_labels/` — `rel`
  (soft, top-5 selection score / 2) and `occ` (hard, any tracked vehicle) per-token targets, plus
  `box_h`/`path_gap` static-geometry baseline features.
- **Phase 3 probe** (`aa1_token_probe.py`, frozen trunk, local GPU): fit + evaluate a
  LayerNorm+Linear(1024,1) probe per layer (11/17/23 checked) on base BADAS-Open and on
  A1-compress256, train/val split matching the 1,761-pool's `clip_level_split`. Confirms the
  relevance signal is present and above a static-geometry baseline (not saturated) — the go-ahead
  for training. Local speed 1.1-3.7 s/window on a quiet machine (an earlier ~20 s/window estimate
  was measured under contention, not representative — see the "no repeated polling" memory note).

### Training runs (all pod, RTX PRO 4500; recipe = A1-compress256's exact config — LoRA
r16/α32/dropout0.05 on `query,key,value`, lr 2e-4 constant, grad_accum 8, compress256, pool1761,
seed 0 — plus the aux loss, unless noted)

**Full 8-epoch runs, frozen crash head:**
| Arm | aux_layer | λ (targets ~0.1 grad-norm pull) | val-selected rank-1 test AP | mean over 8 ckpts |
|---|---|---|---|---|
| AA-rel | 17 | 3.6 | 0.9125 | 0.8957 |
| AA-occ (control) | 17 | 0.73 | 0.9152 | 0.8995 |
| AA-rel-L23 | 23 | 2.76 | 0.8931 | 0.9007 |

**3-epoch checks, `--unfreeze-head --head-lr-mult 1.0 --head-lr-schedule constant`:**
| Arm | λ | val-selected rank-1 test AP (epoch 2 every time) |
|---|---|---|
| AA-ctrl-unfrozen (no aux) | — | 0.9070 |
| AA-occ-unfrozen | 0.59 | 0.9101 |
| AA-rel-unfrozen | 2.43 | 0.9068 |

**None separates from its control.** Stopped by user decision (2026-09-24) rather than extending
the unfrozen arms to 8 epochs — layer choice (17→23) and head-unfreeze both failed to produce the
separation the "frozen head is the bottleneck" hypothesis predicted, so extending training further
without new evidence had nothing to select on.

### Diagnostics — why, not just that it failed (frozen-head AA-rel/AA-occ, epoch-3 checkpoints,
base = BADAS-Open pretrained, probed on a held-out 400-window sample unless noted)

1. **Test-score correlation vs A1-compress256**: AA-rel pearson=0.99 vs A1-compress256, mean
   |Δ|=0.02-0.03, 16-17/677 flips at threshold 0.5 — almost no change reached the final
   prediction. Reference/noise floor: A1-compress256 epoch2 vs its own epoch3 (same run) pearson
   =0.96, 65 flips — ordinary epoch-to-epoch drift is *bigger* than the whole aux loss's effect.
2. **LoRA weight-change vs A1-compress256 at epoch 3**: layers 0-17 (aux-reached) differ by
   80-170% of A1's own update magnitude; layers 18-23 (crash-loss-only) differ by only 30-57%. No
   same-seed A1 rerun exists, so the 30-57% figure is also this comparison's noise-floor upper
   bound. Reading: the aux loss's weight change is real in the early layers and does not
   propagate to the later ones.
3. **Token-relevance probe on AA-rel/AA-occ's own trained epoch-3 weights, layers 17 vs 23**
   (`aa1_token_probe.py --checkpoint <label> --lora-adapter ...`): AA-rel's layer-17
   rel-Spearman gain over base (**+0.098**) clearly exceeds AA-occ's control gain (**+0.054**) —
   the aux loss did teach something layer-17-specific. At layer 23 the two arms' gains are nearly
   equal (**+0.056 vs +0.049**) — AA-rel's specific gain does not survive to where the crash head
   actually reads. Tapping layer 23 directly (AA-rel-L23) did not fix this either — moving the tap
   closer to the output is not, by itself, the fix.
4. **Gradient-norm vs loss-value sizing disagree** — see ARCHITECTURE.md's "λ sizing" note. λ was
   sized to a defensible ~10-12% gradient-norm pull, but the weighted loss VALUE
   (`λ*aux_loss`) exceeded `crash_loss` for most of AA-rel's 8 epochs (0.80 vs 0.58 at epoch 1,
   0.72 vs 0.21 at epoch 8) — worth stating plainly in any write-up even though gradient-norm is
   the correct criterion for what moves the weights.
5. **BADAS crash head architecture, corrects an earlier wrong guess** (read directly from
   `nexight/src/train/video_training.py`): `nn.MultiheadAttention` (8 heads) self-attention over
   2560 tokens → LayerNorm → **plain mean pool** (no single learned query) → MLP classifier. Also
   resolves the long-open 2048-vs-2560/concat-order question: `torch.cat([present, future], dim=1)`
   — 2048 real + 512 predicted, present-then-future, confirmed from source.

**Not run**: an attention-mass diagnostic (does the crash head's attention to the relevant car
actually shift under any condition — designed, not built); paired bootstrap CIs for any arm vs
A1-compress256; continuing the 3-epoch unfrozen checks to 8 epochs; a layer 13-19 probe sweep
(deferred by the user — "keep it for future work if we will not succeed with layer 17").

### Code fixes found and landed during this thread (both matter beyond Stage AA)
- **`f3c064a`** — the global (non-per-layer) crash-vs-aux gradient-norm diagnostic silently never
  ran for any run before this fix: filtering `None` independently on the crash- and aux-gradient
  lists before concatenating them meant their lengths never matched, so the reporting guard was
  always false and `epoch_metrics.jsonl`'s `aux_grad_norm_crash`/`aux_grad_norm_aux` stayed `null`
  for every run. The separate per-layer `grad_trace.jsonl` diagnostic (used for all numbers above)
  was unaffected. Found by the user asking a pointed diagnostic question, not caught proactively.
- **`1c56fe2`** — `--unfreeze-head` resume via `--lora-init` reloaded LoRA correctly but silently
  reset the head to its frozen starting weights (`load_head_state()` was only ever called at
  test-scoring time, never on resume). Found proactively while answering "can we continue if we
  see good training partway through" — fixed via new `--head-init`, required alongside
  `--unfreeze-head` when resuming past epoch 1, before it could corrupt any real
  check-then-continue workflow.
- **`1cb7d48`** — `aa1_token_probe.py --checkpoint` widened from a closed `{base,a1compress256}`
  choice to a free-form label, needed to run the post-hoc probe (diagnostic #3 above) on AA-rel's
  and AA-occ's own trained checkpoints.

## Stage AA-H — head-attention/decision-gradient supervision (2026-09-24/25), Stage 1: noise floor

New direction after Stage AA's null result: instead of a side probe on an intermediate ViT-L
layer, supervise what the crash HEAD's own attention (or decision gradient) reads — see the
child plan `~/.claude/plans/CCP based BADAS/2026-09-24_Child-Plan-AA-H-head-attention-
supervision.md` for the full design (3 loss variants: attn_rank, attn_mass, gradcam) and the
literature review that motivated it (RARE, FAX, GAIN, CAMAL — methods that supervise the
classifier's own attention/gradient, not a side probe, are the ones that worked in the wild).

**Stage 1 (noise floor, pod, RTX PRO 4500)**: 2 runs of A1-compress256's exact recipe, split-seed
held at 0 (same train/val partition), only init_seed varied (1, 2). No new losses — this measures
how much test AP moves from LoRA-init randomness alone, before any Stage 2/3 arm can be judged.

| Arm | init_seed | val-selected rank-1 test AP (private) | mean AP over 8 ckpts | public AP |
|---|---|---|---|---|
| A1-compress256 (control) | 0 | 0.9128 | 0.8957 | 0.9096 |
| AA-ctrl-seed1 | 1 | 0.8924 | 0.8910 | 0.9017 |
| AA-ctrl-seed2 | 2 | 0.9111 | 0.8958 | 0.9087 |

**Rank-1 noise floor is large (range 0.0204)** — bigger than the plan's own "+0.01 real
progress" bar. **Mean-over-8 noise floor is tight (range 0.0048)** — a much more trustworthy
signal for whether a Stage 2/3 arm actually moved something, since it isn't sensitive to which
single epoch validation happens to pick. Recommendation carried into Stage 2/3: lead with
mean-over-8 when judging an arm, treat rank-1 as secondary.

Outputs: `outputs/aa_head_attn/AA-ctrl-seed{1,2}/train/test_summary.json` (all 8 checkpoints,
per-TTE breakdown), `.../scores_public/AA-ctrl-seed{1,2}.metrics.json`. Ran in ~65 min total
(much faster than the ~3.5h estimate — ~536-540s/epoch, at or below A1-compress256's own
historical ~1,165s/epoch average). Large per-epoch LoRA adapters stay pod-only
(`/workspace`, persistent volume); only metadata synced locally.

**Two real bugs found and fixed during Stage 0 verification** (both would have silently
corrupted a real run — full detail in the child plan's status block):
1. **gradcam's autograd-graph-ancestry bug** — differentiated against `patches[0]`
   (`forward_clip`'s post-hoc return slice) instead of `badas._captured["patches"]` (the actual
   ancestor tensor in the graph). PyTorch raised loudly ("not used in the graph"); fixed.
2. **partner-label re-identification's float-timestamp tolerance bug** — `decode_frames` snaps
   to the nearest real frame (~12-33ms drift from the requested time); a `1e-6` tolerance
   silently zeroed re-identification (0/70 matches, no error). Fixed by tracking "started in the
   real first frame" as an index-based flag instead of comparing floats.

## Stage AA-H, Stage 2a — screening `attn_rank` across 3 label choices (2026-09-25)

Same recipe as Stage 1 (A1-compress256's exact config), seed 0, `--aux-mode attn_rank`,
`--aux-margin 0.5` (a thin-sample approximation — see PROJECT_STATE.md's disclosed caveat), each
label's own λ sized by a 1-epoch full-pool pilot targeting a 0.3x gradient-norm pull, then the
loss ramped on for epochs 1-3 and off for 4-8 (`--aux-schedule warm_on_off`, the default).

| Label (what counts as "relevant") | λ* | rank-1 test AP (private) | mean AP over 8 ckpts | public AP | public FP (n=667) |
|---|---|---|---|---|---|
| A1-compress256 / AA-ctrl-seed1 (control) | — | 0.8924 | 0.8910 | 0.9017 | 62 |
| AA-ctrl-seed2 (control) | — | 0.9111 | 0.8958 | 0.9087 | 54 |
| **R_all** — existing rel score, every clip | 0.1645 | 0.9159 | 0.8991 | 0.9119 | 67 |
| **R_pos** — same score, crash clips only | 0.6965 | 0.9124 | **0.9018** | 0.9102 | 66 |
| **partner_pos** — hindsight crash-partner, crash clips only | 0.5036 | 0.9130 | 0.8951 | 0.9092 | 66 |

**Verdict: attn_rank fails the plan's own screen gate (AP clears noise floor AND mechanism moved
AND no FP increase) for all three labels — the FP condition is what breaks it.** R_pos's
mean-over-8 (0.9018) edges the control range (0.8910-0.8958) by +0.006, R_all by +0.003 — both
marginal (roughly one noise-spread-width), not a clean win. partner_pos (0.8951) sits *inside*
the control range: no AP lift at all from the label closest to the user's actual FP concern. All
three land at public FP=66-67 vs both controls' 54-62 — every attn_rank arm is worse on false
alarms than either control seed. Making the label more precise did not fix this; if anything
partner_pos (most precise) gave the least AP lift of the three. Full reasoning and the
mechanism-vs-FP explanation in the child plan's 2026-09-25 status block.

**Mechanism check (per-epoch `epoch_metrics.jsonl`, all 3 labels)**: `rho_P/V` (mean attention
received by relevant-car tokens ÷ mean attention received by other-vehicle tokens) and `rho_P/B`
(same, vs background) both rise sharply during the aux-on window (epochs 1-3) and partly persist
after it switches off (epochs 4-8), confirming the loss does move the head's own attention as
designed. But R_all/R_pos show strong **P-vs-vehicle** separation (rho_pv up to ~4.6 for R_pos),
while partner_pos's rho_pv barely moves (2.3→2.3) and its entire gain is P-vs-**background**
(rho_pb up to ~5.7) — partner_pos never teaches the head to prefer the actual collision partner
over other nearby traffic, only over empty scenery, which lines up with why it produced no AP
lift and did not reduce FP. `val_ap` stayed in the normal 0.94-0.95 band throughout for all three
(no crash-task damage). `n_aux_missing=733/1761` (R_pos) and similar for partner_pos confirm the
negative-clip skip is working as designed.

**Third real bug, found live on the pod (not caught by local testing)**: the crash-vs-aux
gradient-cosine probe — what the λ pilot reads via `epoch_metrics.jsonl`'s
`aux_grad_norm_crash`/`aux_grad_norm_aux` — was gated to `aux_head is not None`, always False for
the new AA-H modes (no trainable projector). Widened the gate, which then exposed a second,
worse bug underneath: whenever a window's aux term is skipped (R_pos/partner_pos on a negative
clip - about half the pool), `aux_loss` is a disconnected `torch.tensor(0.0)` with no `grad_fn`;
the probe tried to differentiate through it, crashed, and **permanently disabled itself for the
rest of the epoch** from the first such window. R_pos's first pilot attempt got
`aux_grad_cos_n_sampled=0` for the entire epoch and silently fell back to an **uncalibrated
lambda=1.0**, which then trained a full 8-epoch run on that bad value before being caught (by
manually checking the sample count printed in the log, not by any automated guard — that run
was killed once found). Fixed by requiring `aux_loss.requires_grad` before the probe attempts
anything. Verified two ways: (1) an isolated control-flow test reproducing the exact
`RuntimeError` message from the pod and confirming the fix skips cleanly instead, (2) live
re-runs on the pod: R_pos's corrected pilot got `aux_grad_cos_n_sampled=80` (lambda*=0.6965),
`partner_pos`'s got 74 (lambda*=0.5036). The corrected driver script also now aborts a label's
full run outright if a pilot ever returns 0 samples again, rather than silently using a
fallback lambda.

Outputs: `outputs/aa_head_attn/AA-H-rank-{R_all,R_pos,partner_pos}/{pilot,train,scores_public}/`.
Pod driver scripts: `run_stage2a_1_2.sh` (R_all + the FIRST, buggy R_pos attempt - its `train/`
output was deleted once the bug was found), `run_stage2a_3plus.sh` (the corrected R_pos re-run +
partner_pos, includes the 0-samples abort guard).

## Stage AA-H, Stage 2b — `attn_mass` with R_pos (2026-09-25/26): worse FP than `attn_rank`

Same recipe as Stage 2a, `--aux-mode attn_mass --aux-label R_pos --aux-weight 0.5518` (1-epoch
pilot, same 0.3x gradient-norm target), `--aux-schedule warm_on_off` (on epochs 1-3, off 4-8).
This was the user's own follow-up after `attn_rank` failed - `attn_mass` (FAX-style, maximizes
attention *mass* on relevant tokens directly) is mechanistically different from `attn_rank`'s
margin/ranking objective, and the hypothesis was that it might not share the same FP failure.

| Checkpoint | private test AP | public AP | public FP (n=667) | recall | specificity |
|---|---|---|---|---|---|
| control (2 seeds) | 0.8924 / 0.9111 | 0.9017 / 0.9087 | 62 / 54 | 0.811/0.802 | 0.816/0.836 |
| **val-selected epoch 8 (operational pick)** | 0.8841 (worst of 8) | 0.8802 | **79** | 0.832 | 0.763 |
| epoch 4 (best private test_ap, diagnostic only) | 0.9163 | 0.9029 | **121** | 0.949 | 0.637 |

**Mean-over-8 AP = 0.9020** - nominally the best mean-over-8 in the whole AA-H investigation
(edges `attn_rank`/R_pos's 0.9018). **This is misleading: at threshold 0.5, this arm has the worst
FP of anything tried, at every checkpoint tested.** The mechanism hypothesis was right (this loss
does force genuine attention reallocation, not a cheap margin trick) but the outcome hypothesis
was wrong (it doesn't help FP - it's worse).

**Two compounding problems:**
1. **Epoch selection broke.** `val_ap` (51-clip val set) selected epoch 8 as best
   (val_ap=0.9432, the highest of all 8 epochs), but epoch 8 has the *worst* private test_ap
   (0.8841 of 677 clips) and a degenerate `optimal_threshold=0.0857`. The small val set stopped
   tracking the larger test set once this loss is present - the most extreme case yet of the
   rank-1 noise problem documented since Stage 1.
2. **Even the genuinely best checkpoint (epoch 4) is worse on FP than anything else tried.**
   FP=121 - almost double `attn_rank`'s worst arm (67) and double both control seeds. Recall
   0.949 at the cost of specificity 0.637 - the model says "crash" much more often, not more
   correctly.

**Mechanism reading (epoch_metrics.jsonl):** `rho_P/V` explodes to **195.85 at epoch 3**
(entropy collapsing 6.86→4.31 - attention goes from broad to sharply peaked on relevant tokens),
far beyond anything `attn_rank` produced (peak ~4.6), and **doesn't unwind** after the loss turns
off at epoch 4 (`rho_P/V` stays at 105→40→26→17→20 through epochs 4-8 with `lambda_last=0.0`).
The aggressive, persistent attention concentration this loss produces by design is the same thing
driving the FP blowout - a head that's learned to attend almost exclusively to "relevant" tokens
appears to fire more readily on anything that pattern-matches "relevant," the same close-car-
triggers-alarm failure mode as the original Stage AA side-probe, reached by a different mechanism
and worse in degree.

**Pod-handling note:** the driver script's `runpodctl stop pod` call fired correctly at job end
(confirmed via RunPod API: `desiredStatus: EXITED` at the matching timestamp) - but the pod was
then deleted (not just stopped) by a separate manual dashboard action before results were synced.
No data lost: `/workspace` is on network volume `0hnvco2s4j` (EU-RO-1), independent of any
specific pod; a fresh pod attached to the same volume retrieved everything intact.

Outputs: `outputs/aa_head_attn/AA-H-mass-R_pos/{pilot,train,scores_public,
scores_public_ep04_diag}/`. Driver script: `run_stage2b_mass_Rpos.sh`.

**Reading for what's next:** two independent loss families (`attn_rank`, `attn_mass`) now agree
that pushing the crash head's attention toward the labeled-relevant object does not reduce false
alarms - `attn_mass` makes it measurably worse. **User decision 2026-09-26: close Stage AA-H, do
not run `gradcam`.**

### Post-mortem: which negatives does `attn_mass` turn into false positives? (2026-09-26)

Cheap post-hoc check, no pod time - cross-referenced `attn_mass`-epoch4's 121 public FPs against
`AA-ctrl-seed2`'s per-clip scores on the same 333 public negatives (both jsonl already synced
locally):

- 67 of the 333 negatives flip from TN (control, score<0.5) to FP (attn_mass-ep4, score>=0.5).
- **Median control score among the 67 flipped clips: 0.212. Median control score across ALL 333
  negatives: 0.051** - a ~4x elevation. These are not random clips; they are disproportionately
  the negatives the plain classifier already found borderline/ambiguous (still correctly called
  safe, but with noticeably elevated risk score), not clips it was confidently correct on.

**This directly confirms the diagnosis change below**: `attn_mass` isn't failing on arbitrary
clips - it's taking exactly the negatives that already look like "close call" cases (a car got
near or entered the lane but didn't collide) and tipping them over the threshold. Forcing more
attention onto the labeled-relevant object amplifies the existing proximity-to-risk association
the classifier already had, rather than teaching it to distinguish "closing in dangerously" from
"nearby but not converging" - because the relevance/partner label used as the aux target encodes
*which object matters*, not *whether its trajectory implies a hit*, so there's no signal in the
loss that could teach that distinction in the first place.

## Re-analysis 2026-09-26: pooled 1,344-clip test, val-vs-test gap, literature (CORRECTS parts of the Stage 2a/2b verdicts above)

**Protocol.** Each arm at its val-selected epoch; private `test_results_epNN.jsonl` (677) + public
`scores_public/*.jsonl` (667) pooled = the full Nexar test set (1,344 clips, 672/672, no id overlap).
Paired bootstrap (5,000 resamples of clips) vs **A1-compress256 (same init/split seed 0 as every
AA-H arm)**. FP compared at **matched recall** (FPR at TPR=0.85/0.90), not at threshold 0.5.
Pooled per-arm files: scratchpad only (rebuild from the two sources above).

| Arm | AP (1344) | Kaggle mAP | ΔAP vs A1-c256 [95% CI] | FPR@TPR.85 | FP/FN @0.5 |
|---|---|---|---|---|---|
| A0 (frozen BADAS-Open) | 0.861 | 0.869 | — | — | — |
| **A1-compress256** (seed 0) | 0.911 | 0.916 | — | 0.188 | 126/103 |
| AA-ctrl-seed1 | 0.897 | 0.899 | −0.014 [−0.021,−0.008] | 0.238 | 122/130 |
| AA-ctrl-seed2 | 0.910 | 0.912 | −0.001 [−0.004,+0.002] | 0.196 | 99/137 |
| attn_rank R_all | 0.914 | 0.918 | +0.003 [+0.000,+0.005] | 0.177 | 124/100 |
| attn_rank R_pos | 0.911 | 0.916 | +0.000 [−0.003,+0.003] | 0.185 | 121/103 |
| attn_rank partner_pos | 0.911 | 0.915 | −0.000 [−0.003,+0.002] | 0.185 | 121/102 |
| attn_mass ep8 (val-selected) | 0.881 | 0.884 | −0.030 [−0.042,−0.018] | 0.265 | 156/117 |
| attn_mass ep4 (diag) | 0.910 | 0.915 | −0.001 [−0.011,+0.006] | 0.176 | 218/40 |

**Corrections to the verdicts above:**
1. **`attn_rank` does NOT increase FP.** At matched recall its FPR equals or slightly beats its seed-0
   twin (0.177-0.185 vs 0.188), and FP/FN at 0.5 are ~identical to A1-compress256. The earlier "66-67
   vs 54-62" comparison was against seeds 1/2 on the public half only — seed-to-seed differences in
   operating point (seed2: 99 FP / 137 FN) masquerading as a treatment effect. Correct verdict:
   **null effect** (R_all +0.003 is paired-significant but ~5x smaller than training-seed variance).
2. **`attn_mass` ep4's FP=121/218 is a calibration shift, not worse discrimination** (same AP, FPR at
   matched recall 0.176 ≈ control). The genuinely harmful part is late-epoch degradation (ep8: −0.030,
   CI excludes 0) plus val_ap selecting ep8.
3. **The "close-call amplification" flip evidence is weak**: a monotone upward shift of all scores
   also flips the highest-scoring negatives first, so median-ctrl-score 0.21 vs 0.05 among flipped
   negatives is expected from calibration alone. It does not by itself support the proximity-confound
   diagnosis.
4. **What survives:** attention moved 2x (rank) to 40x (mass) with **zero change in ranking quality**
   (AP flat within CI) → attention placement is not the bottleneck. The kinematic direction remains a
   hypothesis, not a demonstrated one.

**Resolution / noise (answers "is 677 enough?"):** single-arm AP 95% CI on 1344 ≈ ±0.017; paired ΔAP
CI ≈ ±0.003 — pooling makes *test* noise small. The binding constraint is **training-seed variance**:
seed1 vs seed0 = −0.014 (CI excludes 0), seed2 vs seed0 = −0.001. Caveat: the 1,344 clips come from
only **568 source videos** (up to 3 TTE crops per video, Nexar paper §4.1), so clip-level bootstrap
CIs are somewhat optimistic. Any future claim needs ≥3 seeds per arm, compared on seed means.

**Why val_ap reads 0.94-0.95 while test is ~0.90:**
- **Metric definition (~+0.03):** `evaluate_val` averages the scores of a video's 1-3 windows
  (clip-level, 221 clips); test is one window per clip. Window-level val AP from `val_scores_ep*` is
  0.91-0.93, not 0.95.
- **In-distribution val, blind to late-epoch drift:** early epochs val≈test (seed2 ep1: 0.918 vs
  0.911); later val stays flat while test falls (seed1 ep8: 0.918 vs 0.854). Val is drawn from the
  same pool and sampling rules as train, so it cannot see the train→test shift that grows with
  training → rank-1 selection picks late, degraded epochs.
- **Negative-sampling mismatch (confirmed from the Nexar paper §4.1):** test negatives end at a fake
  event time = **video midpoint + Gaussian noise**. `build_train4500_manifest.py` deliberately moved
  our negatives *away* from the midpoint (MID-10/-8/-4) because midpoint windows produced 43% FP at
  0.99+ confidence. So train/val never contain the kind of negative window the test set samples. A0
  (untrained) has equal FPR@0.5 on val and test negatives (0.37/0.38); after fine-tuning val FPR drops
  to 0.04-0.15 but test only to 0.13-0.30 — training fixes train-like negatives, not test-like ones.
  (Nexar labels near-misses as POSITIVE; test negatives are regular driving, midpoint-cut.)

**Literature context (Nexar test, 1,344 clips):**
- BADAS-Open (1.5k videos): AP 0.86 (matches our A0 0.861). BADAS-1.0 (40k proprietary videos): AP
  0.91, Kaggle mAP 0.925. BADAS-2.0 (178.5k labeled videos ≈2M clips + 2.25M unlabeled videos for
  SSL): Kaggle mAP 0.940, FPR 10.9%→4.6%, largely attributed to mined **hard negatives**. Data scaling
  is logarithmic (BADAS Fig. 6).
- FLaRA (2026, same 1.5k train): AP 0.866; its aux future-latent loss adds +1.1 AP (0.855→0.866),
  single run, no seed/CI analysis.
- An LLM-agent search over 10,469 configurations (arXiv 2603.15916, V-JEPA2 features) converges to
  AP 0.9245 on dashcam collision detection (power-law convergence).
- **Our A1-compress256 (1.5k videos) = AP 0.911 / Kaggle mAP 0.916 ≈ BADAS-1.0 (40k videos).** We are
  already at the level that cost Nexar ~27x our data; the next +0.025 cost them ~4.5x more labeled
  data plus 2.25M videos of domain SSL. Realistic headroom from method changes at our scale: ~+0.01.

**Follow-ups (same day):**
- **Window-level vs clip-level val AP as epoch selector** (6 runs with per-epoch val+test dumps):
  clip-level tracks test better (Spearman vs test AP higher in 5/6 runs; selected-epoch test AP 0.9056
  clip vs 0.9018 window). `semsup_train.py` now LOGS `val_ap_window` (comparable to test, one window
  per clip) in the epoch line and epoch_metrics.jsonl, but still SELECTS by clip-level `val_ap`.
- **Test AP falls with every epoch past ~1-2** (AA-ctrl-seed2 private: 0.911, 0.906, 0.908, 0.902,
  0.890, 0.891, 0.880, 0.872). Averaging epochs 1-3 = 0.9105; all 8 = 0.9040. Longer training hurts.
- **3-seed ensemble** (seed0/1/2 controls, pooled 1,344): 0.9087 vs mean single seed 0.9057 vs best
  seed 0.9109. Ensembling removes bad-seed risk but does not raise the ceiling.

**Gradient propagation across training (2026-09-26, from epoch_metrics.jsonl / grad_trace.jsonl):**
no aux loss had a vanishing gradient. On the shared LoRA parameters, |g_aux| stayed at 0.8-4.2 vs
|g_crash| 2.6-5.8 for every AA-H arm over all 8 epochs (λ-weighted pull 0.25-0.55x while on, as
designed). Stage AA probes: aux/crash norm ratio decays ~5x from the tap layer down to layer 0
(e.g. AA-rel ep1 0.075 → 0.014) — attenuation, not vanishing. The consistent signal is
**direction**: cos(g_crash, g_aux) ≈ 0 for every probe arm and every attn_rank arm (|cos| ≤ 0.03,
~50% of steps conflicting = random sign), and only +0.05..+0.13 for attn_mass. The aux gradient
arrives with full magnitude but points almost entirely in directions the crash loss is
indifferent to — the mechanistic reason attention moved while AP did not. attn_mass's small
positive alignment matches it being the only arm that changed outputs (calibration shift, late
drop). Caveats: AA-H-rank-R_all has only 1-57 probe samples/epoch (pre-fix probe bug) — noisy;
AA-H arms did not log per-layer grads (--per-layer-grads off); gradcam was never trained.
**Use:** cos(g_crash, g_aux) from the 1-epoch λ pilot is a cheap pre-screen for any future aux
target — near-zero predicts no AP effect.

## Overnight run 2026-09-27: midpoint negatives beat control, full pool loses to it

Two crash-only experiments, 3 seeds each (init_seed/seed 0/1/2, split_seed=0), 3 epochs each
(keep-top-k 3), A1-compress256's exact recipe otherwise (LoRA r=16/α=32 on query,key,value,
lr 2e-4 constant, compress256, crash head frozen, `lora_init=None` — fresh random LoRA init each
run, same as every arm this session). Scored on BOTH private 677 and public 667 (pooled 1,344,
no overlap) at every one of the 3 kept epochs. Driver: `outputs/overnight_2026-09-27/
run_overnight.sh`; results: `outputs/overnight_2026-09-27/{midneg,fullpool}-seed{0,1,2}/`.

**midneg**: `outputs/semantic_captions/Caption_Train4500_MidpointNeg_1761.jsonl` — same 1,761
windows, same 564 negative videos, only the 905 negative windows are re-cut. Built by
`build_midpoint_negatives.py`: per negative video, one fake-event time = true midpoint +
N(0, 1.0s) (drawn once per video, seeded, so a video's multiple old buckets share one consistent
fake event), then 3 sub-windows cut 0.5/1.0/1.5s before it — replacing MID-10/-4/-8's fixed
off-midpoint offsets with the Nexar test protocol's own negative-sampling method (dataset paper
§4.1). Old→new bucket mapping keeps the group index fixed (MID-10→grp0/0.5s, MID-4→grp1/1.0s,
MID-8→grp2/1.5s). Extracted locally from the raw mp4s (905/905 windows, 0 errors, 0 floored).

**fullpool**: `outputs/semantic_captions/Pool_Train4500_Full_4446.jsonl` — the full 4,446-window
pool (1,482 videos, 2,223/2,223 balanced) instead of the mined 1,761. Original MID-10/-4/-8
negative sampling, unchanged.

**Result (mean over the 9 checkpoints per arm; ΔAP is PAIRED against the same seed's control at
the same epoch — A1-compress256=seed0, AA-ctrl-seed1, AA-ctrl-seed2 — using their epoch 1-3
checkpoints, newly scored on public this same day, see below):**

| Arm | mean AP (pooled 1,344) | paired ΔAP vs matched control | positive pairs |
|---|---|---|---|
| control (A1-compress256 + ctrl-seed1/2) | 0.9077 | — (reference) | — |
| **midpoint negatives** | **0.9158** | **+0.0082** | **9/9** |
| full pool | 0.8986 | −0.0091 | 1/9 |

Holds at every individual epoch too (3-seed mean per epoch): control 0.9070/0.9085/0.9076 (flat —
seed noise, not a trend, since only 3 epochs were run here); midneg 0.9185/0.9157/0.9133 (beats
control at all 3); fullpool 0.8998/0.9051/0.8907 (loses to control at all 3).

**Confusion matrix @ threshold 0.5, pooled 1,344, averaged over 9 checkpoints:**

| Arm | TP | FN | FP | TN | Recall | Specificity | Precision | Accuracy |
|---|---|---|---|---|---|---|---|---|
| control | 581.1 | 90.9 | 156.4 | 515.6 | 0.865 | 0.767 | 0.788 | 0.816 |
| midpoint negatives | 535.4 | 136.6 | 103.8 | 568.2 | 0.797 | 0.846 | 0.838 | 0.821 |
| full pool | 606.2 | 65.8 | 195.7 | 476.3 | 0.902 | 0.709 | 0.756 | 0.805 |

Midpoint negatives cut FP 34% (156→104) and raise specificity 8pt, trading for more misses
(91→137, i.e. a genuine operating-point shift toward caution, not just better ranking); overall
accuracy still rises (0.816→0.821). Full pool moves the opposite way (more FP, worse specificity)
— **correction to an earlier same-day note**: full pool's FPR at MATCHED RECALL (TPR 0.85) is
0.208, essentially equal to control's 0.209 — its problem is lower AP/ranking quality, not a
distinct FP-calibration defect. Per-checkpoint spread is wide for both non-control arms (midneg
FN range 79-265, fullpool FP range 157-225) — not yet a single stable deployment threshold.

**Per-TTE-horizon AP (pooled, mean over 9 checkpoints; each bucket's negatives are the fake-event
protocol for ALL arms including control here, since these are Nexar's own test buckets, not a
training-side split):**

| Arm | TTE 0.5s (n=568) | TTE 1.0s (n=464) | TTE 1.5s (n=312) | Kaggle mAP (mean of 3) |
|---|---|---|---|---|
| control | 0.9262 | 0.9216 | 0.8852 | 0.9110 |
| midpoint negatives | 0.9395 | 0.9265 | 0.8705 | 0.9122 |
| full pool | 0.9094 | 0.9142 | 0.8890 | 0.9042 |

Midpoint negatives win clearly at 0.5s/1.0s but LOSE at 1.5s (0.8705 vs control's 0.8852, largest
seed spread of the three, sd 0.0114) — on overall AP the gain is +0.008, but on the officially
weighted Kaggle mAP (equal weight per horizon) it shrinks to +0.001, since the 1.5s loss cancels
most of the 0.5s/1.0s gain. **1.5s TTE is the weakest bucket for every arm** (0.87-0.89 vs
0.91-0.94 elsewhere) and is now the highest-value target: a method that lifts 1.5s without
regressing the other two would move both overall AP and Kaggle mAP together.

**Selection-bias note**: the single best checkpoint observed anywhere this session is
midneg-seed0-epoch1 at pooled AP 0.9237 — this is NOT the number to report. It is 1 of 9
midneg draws (3 seeds x 3 epochs) and sits within ordinary seed noise at epoch 1 alone (3-seed
sd=0.0045). The reportable number is the mean (0.9158 over 9, or 0.9185 at epoch 1 across 3
seeds), stated with its spread.

**Loss curves (crash_loss) confirm real learning, not stagnation**: every seed's train crash_loss
drops monotonically through all 3 epochs (e.g. midneg-seed0: 0.692→0.499→0.446), while val/test AP
plateaus or wobbles after epoch 1 and `train_val_gap` grows for 2/3 seeds — classic early
overfitting: epoch 1 captures most of the transferable signal, epochs 2-3 mostly sharpen
confidence on training-specific patterns. Consistent with Stage 1's 8-epoch finding (test AP
declines almost every epoch past 1-2).

**Comparison with literature, same pooled 1,344-clip Nexar test set:**

| Model | Training data | Overall AP | Kaggle mAP |
|---|---|---|---|
| BADAS-Open (paper 0.86; our reproduction) | 1.5k videos | 0.861 | 0.869 |
| **Ours, midpoint negatives (mean of 9)** | **same 1.5k videos** | **0.916** | **0.912** |
| BADAS-1.0 (paper) | 40k videos | 0.91 | 0.925 |
| BADAS-2.0 (paper) | 178.5k labeled + 2.25M unlabeled (SSL) | — | 0.940 |

We reproduce BADAS-Open almost exactly (0.861 vs 0.86 paper), which calibrates the pipeline for
this cross-paper comparison. Our overall AP matches/slightly beats BADAS-1.0 on 1/27th its data;
Kaggle mAP is 0.013 behind, entirely attributable to the 1.5s weak point above.

**Architecture note (2026-09-27, in response to "did we change the architecture"): no.** Every
arm this entire project uses BADAS-Open's published architecture unmodified at inference — LoRA
(r=16, on query/key/value, 0.84% of params) is a training-time adapter that merges into the
existing weight matrices (W = W0 + (alpha/r)*B*A); nothing is added to the forward graph. Gains
to date come from preprocessing (full-frame vs crop, +0.013 AP), data curation (curated 1,761 >
full 4,446), and negative-sampling protocol (this entry). No BADAS-2.0 architecture element
(distillation, SSL pretraining) has been adopted — see DECISIONS.md's open items for which of
BADAS-2.0's §5.3 (two-phase KD) and §6.1 (training-free attention heatmaps) ideas are candidates.

### Overnight 2026-09-27 — per-seed breakdown (added 2026-09-29)

**Per-TTE AP by seed** (pooled 1,344, each cell = mean over that seed's epochs 1-3):

| midneg seed | TTE 0.5s | TTE 1.0s | TTE 1.5s |
|---|---|---|---|
| seed0 | 0.9402 | 0.9266 | 0.8782 |
| seed1 | 0.9361 | 0.9229 | 0.8641 |
| seed2 | 0.9421 | 0.9300 | 0.8692 |
| mean ± sd | 0.9395 ± 0.0025 | 0.9265 ± 0.0029 | 0.8705 ± 0.0058 |

The "0.9395 / 0.9265 / 0.8705" row reported elsewhere is this mean — an aggregate of 9
checkpoints, not any single run (closest single checkpoint: midneg-seed1-ep01, 0.9390/0.9203/0.8700).

**Confusion matrices, seed x epoch x TTE** (pooled 1,344, threshold 0.5; cell = TP/FN/FP/TN;
bucket sizes 284+/284−, 232+/232−, 156+/156−):

| seed | ep | TTE 0.5s | TTE 1.0s | TTE 1.5s | FN | FP | AP |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 262/22/66/218 | 204/28/34/198 | 106/50/17/139 | 100 | 117 | 0.9237 |
| 0 | 2 | 246/38/46/238 | 190/42/32/200 | 83/73/15/141 | 153 | 93 | 0.9133 |
| 0 | 3 | 260/24/59/225 | 205/27/46/186 | 106/50/21/135 | 101 | 126 | 0.9152 |
| 1 | 1 | 253/31/49/235 | 188/44/28/204 | 90/66/15/141 | 141 | 92 | 0.9126 |
| 1 | 2 | 248/36/53/231 | 196/36/28/204 | 94/62/18/138 | 134 | 99 | 0.9175 |
| 1 | 3 | 253/31/65/219 | 203/29/50/182 | 111/45/26/130 | 105 | 141 | 0.9056 |
| 2 | 1 | 227/57/20/264 | 134/98/10/222 | 46/110/3/153 | 265 | 33 | 0.9193 |
| 2 | 2 | 253/31/40/244 | 188/44/28/204 | 80/76/12/144 | 151 | 80 | 0.9163 |
| 2 | 3 | 262/22/69/215 | 211/21/55/177 | 120/36/29/127 | 79 | 153 | 0.9191 |

**Seed-2 epoch-1 finding:** same ranking quality (AP 0.9193, 2nd of 9) but a global downward score
shift — median positive score 0.671 (others 0.86-0.97), median negative 0.032, threshold for 85%
recall 0.15 (others 0.37-0.62). FN at 0.5 is therefore 265, concentrated at 1.5s (110/156), whose
positives sit closest to 0.5. Epoch 3 of the same seed flips to FN 79 / FP 153 at equal AP.
Across checkpoints, 1.5s FN ranges 36-110 while AP moves < 0.02 — most 1.5s FN variation at 0.5 is
score offset, not detection. Val-score dumps were not enabled for these runs (`--dump-val-scores`
off), so no val-side calibration check exists.

**Val-selected epochs (train_metrics.json best_epoch)**: midneg seed0/1/2 = 2/2/1; fullpool
seed0/1/2 = 1/1/3. These are what the website comparison page currently shows (mixed epochs).

**Experiment dates (result-file mtimes, for the website date column):** A0 2026-06-24, A1 08-06,
B-v1 08-08, B-v2 08-11, B-v3 08-13, P1 08-17, V12 08-29, a1cont/V10/v12shuf 09-05,
A1-compress256 09-14, AA-rel/AA-occ/AA-rel-L23 09-23, AA-occ-unfrozen/AA-rel-unfrozen/
AA-ctrl-unfrozen 09-24, AA-ctrl-seed1/2 + all AA-H arms 09-25, midneg/fullpool 09-27.

## 2026-09-29/30 — 5-seed confirmation, 1.5s-TTE plan Stage 0 and Stage 1

All test numbers: pooled 1,344 clips (private 677 + public 667), per-TTE AP = that horizon's positives
vs its negatives, Kaggle mAP = mean of the 3, FPR@85 = false-alarm rate at the threshold giving 85%
recall, FN/FP at threshold 0.5. Paired = same seed, same epoch. Epoch 1 pre-registered before seeds 3-4.
Analysis script: `student_training/scripts/stage_compare.py` (generic, `--arm NAME=priv|pub` with
`{seed}`/`{ep}` placeholders, `--ref`, prints the pass rule).

### Midpoint negatives, seeds 3-4 (confirmation run, 2026-09-29)
- Driver `outputs/confirm_seeds_2026-09-29/run_confirm.sh`; runs midneg-seed{3,4} + ctrl-seed{3,4}
  (ctrl = old 1,761 Mixed pool), both 3-epoch recipe, `--dump-val-scores` on. Outputs (no weights)
  in `outputs/confirm_seeds_2026-09-29/`.
- Epoch 1, paired midneg − ctrl, seeds 0-4: pooled AP **+0.0092 ± 0.0052 (5/5)**; Kaggle mAP
  +0.0037 ± 0.0067 (4/5; seed 4 −0.0063); 1.5s AP −0.0071 ± 0.0115 (2/5); FPR@85 −0.017.
  Epoch 2: Kaggle ≈0; epoch 3: −0.005.
- 5-seed epoch-1 means: midneg AP 0.9174±0.005, TTE 0.9427/0.9274/0.8742, Kaggle 0.9148±0.007;
  control AP 0.9082±0.003, TTE 0.9285/0.9235/0.8813, Kaggle 0.9111±0.002.
- Caveat: ctrl seeds 0-2 are the old 8-epoch runs' epochs 1-3; seeds 3-4 are 3-epoch runs. Same picture.

### Chart diagnosis — TP per TTE at matched FPR (private 677)
| arm | AP 0.5/1.0/1.5 | TP @FPR10% | TP @FPR36% |
|---|---|---|---|
| A1 (8-ep, ep4) | .909/.910/.884 | 117/87/43 | 135/112/72 |
| A1-compress256 | .928/.923/.902 | 125/91/41 | 139/114/71 |
| midneg-seed1 | .937/.939/.871 | 116/88/37 | 136/115/72 |
Median negative score 0.34 (A1) vs 0.07-0.10 → the website chart's TP drop at 0.5 is a score shift.

### Stage 0a — same-source-video test check (local, no GPU)
- Script `student_training/scripts/tte_same_video_diag.py`: links each 1.5s test clip to the same
  video's 1.0s and 0.5s clips by frame matching (manifests have no source field); 156/156 positive
  chains linked confidently. Scores = 5-seed mean, midneg epoch 1; threshold at FPR 10% (0.568).
- 1.5s MISSED (n=79): same video's 1.0s clip detected 67%, 0.5s clip 92%, either **94%**; mean score
  1.5/1.0/0.5 = 0.33/0.66/0.87. 1.5s DETECTED (n=77): 96% either. Output `outputs/tte_diag_0a/`.

### Stage 0b — is the +0.5 s pooled vector predictable? (frozen BADAS-Open, no LoRA)
- Extraction `lookahead_extract_features.py` (pod) → pooled 1024-d `z` + logits; analysis
  `lookahead_feasibility.py` (local): ridge on the change z_next − z_now, GroupKFold by video,
  `--balance` weights negatives. Frozen head reproduces stored logits to 2e-6.
- Negatives needed completing: `build_midpoint_negatives.py --complete-horizons` cut the 787 missing
  midpoint windows (all 3 horizons for the 564 pool negative videos; same seeded fake event, so the
  905 existing windows reproduce exactly; manifest `outputs/semantic_captions/LookAhead_NegWindows_564x3.jsonl`).
- Final run, **1,761-pool videos only**: 543 pos + 564 neg videos x 3 horizons = 3,321 windows,
  pairs 1.5→1.0 and 1.0→0.5 = 1,086 pos / 1,128 neg. Merged features `outputs/lookahead_0b_merged/`.
- Error ratio (ours ÷ "copy the current vector"; <1 = beats copy): pos 1.5→1.0 **0.710**, pos
  1.0→0.5 **0.729**, neg 1.049 / 0.998. Learning curve 25/50/100% of videos: 0.876/0.848/0.828.
- Frozen head AP on 1.5s windows (pool, not test): z_now 0.769, predicted z_hat **0.794**,
  true z(+0.5 s) oracle **0.894**; mix z_now + z_hat 0.780.
- Earlier imbalanced run (full-pool positives, 276 neg pairs) inflated AP (~0.93) and made negatives
  worse than copy (1.27) — superseded.

### Stage 1 — horizon-weighted crash loss (baseline for Stage 2), 2026-09-29
- Flags `--horizon-weights 0.5:1.0,1.0:1.5,1.5:2.0` and `--horizon-weights-scope {pos,all}`; midneg
  recipe, 3 epochs. Driver `outputs/stage1_horizon_weights_2026-09-29/run_stage1{,_sym}.sh`.
- **Crash-only weights (scope pos), seeds 0-1** (seed 2 stopped by design): epoch 1 1.5s AP
  −0.0024 vs midneg (0/2); FN 241→150, FP 209→329 (sum 2 seeds) — pure score shift up.
- **Symmetric weights (scope all), seed 0:** epoch 1 AP 0.9229 / TTE 0.9457/0.9312/0.8918 /
  Kaggle 0.9229 / FPR@85 0.171 / FN 78 FP 142, vs midneg 0.9237 / 0.9467/0.9323/0.8920 / 0.9237 /
  0.174 / 100 117. Ties at 1.5s; half the score shift of crash-only. Seeds 1-2: see pod outputs
  `outputs/stage1_horizon_weights_2026-09-29/sym/` once pulled (pending at handoff).
- Pre-registered rule (1.5s AP above matched control): fails for both so far.
