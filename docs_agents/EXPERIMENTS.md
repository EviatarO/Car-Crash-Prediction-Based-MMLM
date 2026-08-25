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
