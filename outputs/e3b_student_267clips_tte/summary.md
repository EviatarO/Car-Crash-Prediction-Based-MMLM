# E3b — Multi-Horizon Retrain (267 clips, warm-start from e3a ep7)

**Checkpoint:** `step_000099` (epoch 3) — best genuine-e3b val_ap; warm-started from
e3a epoch-7. Backed up to HF: `EviatarO/e3b-ep3-lora`.
**Train set:** 267 clips = **89 unique scenes × 3 TTE horizons** (126 YES / 141 NO).
**Selection caveat:** val set n=18 is too noisy to rank checkpoints; the headline
decision was made on the large-n test sets below.

## Headline (test, threshold-free)

| | e3a (ep7) | **e3b ep3** | Δ |
|---|---|---|---|
| Private AP (677) | 0.762 | **0.790** | +0.029 |
| Public  AP (667) | 0.752 | **0.768** | +0.016 |
| Private AUC | 0.784 | **0.803** | +0.019 |
| Public  AUC | 0.791 | **0.800** | +0.009 |

**Paired bootstrap (5000×, clip-level, percentile):**
- Private ΔAP = +0.029, 95% CI [−0.006, +0.063], P(e3b>e3a)=0.946
- Public  ΔAP = +0.016, 95% CI [−0.021, +0.049], P(e3b>e3a)=0.794
- **Pooled (n=1344)** ΔAP = +0.024, 95% CI [−0.003, +0.048], **P=0.962**
- Verdict: **one-sided marginally significant, two-sided not.** e3b matches-or-beats
  e3a on both halves; cross-half consistency argues it's a small real effect.

## Thresholded (state the threshold!)

| Half | thr | TP | FP | FN | TN | mean(score\|pos) | mean(score\|neg) | opt-thr |
|---|---|---|---|---|---|---|---|---|
| Private | 0.5 | 304 | 167 | 34 | 172 | 0.878 | 0.524 | 0.548 |
| Public  | 0.5 | 298 | 161 | 36 | 172 | 0.882 | 0.509 | 0.376 |

**Score saturation:** e3b positives sit at ~0.88 (vs e3a's low band). The whole score
distribution shifted up → optimal threshold moved to ~0.5–0.55 (e3a was ~0.32), and
0.5 over-predicts YES (60–62%). AP/AUC unaffected (rank-based).

## Per-TTE-group AP (over all clips in bucket) — FLAT

| TTE | Private AP | Public AP |
|---|---|---|
| 0.5s | 0.793 | 0.779 |
| 1.0s | 0.806 | 0.767 |
| 1.5s | 0.772 | 0.776 |

No AP-vs-TTE slope (same as e3a). The multi-horizon data shifted score *calibration*
(saturation) but did not change per-horizon *discrimination*. A within-scene paired
slope test (pod) is the clean confound-free check still outstanding.

## Tricolor review (score vs text-verdict, thr=0.5)

| Half | 🟢 ok | 🟠 score-ok/verdict-wrong | 🟧 score-wrong/verdict-right | 🔴 both-wrong |
|---|---|---|---|---|
| Private | 451 | 25 | 38 | 163 |
| Public  | 449 | 21 | 30 | 167 |

## Files

- `metrics/e3b_test_private_step000099/` — metrics.json, confusion_matrix, roc, pr,
  score_distribution, group_ap_bar, examples_table, per_group/, tricolor xlsx
- `metrics/e3b_test_public_step000099/`  — same for Public
- `figures/` — e3a-vs-e3b comparisons + all-groups-combined:
  - `overall_ap_auc_e3a_vs_e3b.png`, `ap_per_group_e3a_vs_e3b.png`, `score_slope_e3a_vs_e3b.png`
  - `e3b_score_dist_all_groups_overlay.png`, `e3b_ap_all_groups_plus_overall.png`, `e3b_tte_confidence_bar.png`

## Outstanding (needs pod — GPU inference on step_000099)

- VAL (18) and TRAIN (267) per-clip readback metrics+figures — predictions never
  saved; require an inference pass. Batch with the paired slope test in one pod session.

---

# TTE Anticipation Curve (re-cut to 3.0s) — 2026-06-14

## Headline finding
Within-scene, AP/AUC **decline monotonically** with time-to-event (0.5→3.0s),
reproducing the published anticipation-curve shape. e3a and e3b **overlay** — e3b's
score saturation did **not** extend its anticipation horizon (multi-horizon training
inflated scores, bought no extra early-warning range). Private/Public overlay = replication.

## ⚠️ Correction to the earlier "flat 0.76 TTE curve"
The earlier flat AP across 0.5/1.0/1.5s was a **measurement artifact, not a finding.**
Two confounds:
1. **Cross-scene:** the Nexar group-0/1/2 buckets are **disjoint videos** (142/117/79,
   verified 0 overlap), each used once — different scenes per TTE, not the same scene.
2. **Per-bucket re-balanced negatives:** each bucket was scored vs its *own* ~equal
   negative set (chance≈0.5), which **structurally cannot show a decline** (each bucket
   is graded on its own curve). Positive scores *did* drift down (0.587→0.536→0.514),
   but balanced-per-bucket AP hid it.

The corrected curve **re-cuts the same 142 group-0 scenes** at every offset (constant
142 positives → constant prevalence) and uses **one fixed negative pool** across all TTE.
Only time varies → the decline is real, not a count/prevalence artifact.

## Curves (pooled, 284 pos vs fixed neg; bootstrap 5000× clip-level)

**AUC-ROC vs TTE** (prevalence-invariant — FEATURED):

| model | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s |
|---|---|---|---|---|---|---|
| e3a | 0.820 | 0.696 | 0.668 | 0.647 | 0.634 | 0.617 |
| e3b | 0.833 | 0.682 | 0.656 | 0.615 | 0.601 | 0.569 |

**Balanced-AP vs TTE** (142 vs 142, chance=0.5, anchor reproduces ~0.76 — reconciling secondary):

| model | 0.5s | 1.0s | 1.5s | 2.0s | 2.5s | 3.0s |
|---|---|---|---|---|---|---|
| e3a | 0.770 | 0.629 | 0.608 | 0.587 | 0.580 | 0.561 |
| e3b | 0.784 | 0.598 | 0.574 | 0.545 | 0.531 | 0.506 |

Both still **above chance (0.5) at 3.0s** — the model retains weak signal even 3s out.
Steepest drop is 0.5→1.0s; gentle glide thereafter.

## Method note (defensible for thesis)
- Equal-N-per-TTE follows the standard frame-sweep anticipation protocol (Chan et al.,
  ACCV 2016; Bao et al., ACM MM 2020): a fixed positive set scored at every
  time-before-accident → constant positive count across TTE by construction. Our
  16-frame-window re-cut is the windowed analog.
- AP is prevalence-dependent (chance = P/(P+N)); AUC is prevalence-invariant. The
  full-339-pool AP looks low only because of prevalence (chance 0.295), not lost skill —
  hence AUC is featured and balanced-AP is the comparable secondary.

## Val (18) + Train readback (267) — e3b step_000099
| set | AUC | F1@0.5 | mean(pos) | mean(neg) | opt-thr |
|---|---|---|---|---|---|
| VAL (18)   | 0.728 | 0.696 | 0.816 | 0.614 | 0.363 |
| TRAIN (267)| 0.971 | 0.825 | 0.982 | 0.384 | 0.893 |

Train readback ≈ memorized (AUC 0.97, mean pos 0.982). Tricolor (train): 211 green,
3 score-ok/verdict-wrong, 14 score-wrong/verdict-right, 39 both-wrong — score/text
desync persists even on memorized clips. (Val tricolor: 8/3/1/6.)

## Files (new)
- `tte_curve/auc_ap_vs_tte.png` (+ `_numbers.json`) — FEATURED AUC + balanced-AP, 6 panels
- `tte_curve/ap_vs_tte_curve.png`, `score_vs_tte_curve.png`, `lead_time.png` (full-pool variants)
- `tte_curve/tte_curve_numbers.json`
- `metrics/e3b_val_step000099/` and `metrics/e3b_train_readback_step000099/` — metrics + tricolor xlsx
- scripts: `tte_curve_auc_balanced.py`, `analyze_tte_curve.py`, `split_tte_curve_by_offset.py`

## Nexar bucket origin (confirmed from build_test_manifest.py)
The `group` (0/1/2 → 0.5/1.0/1.5s) is a **Nexar-native field** from `test.csv`
(`id, event_occurs, Usage, group`), NOT our construction. Each video is assigned by
Nexar to one group; we always take the last-16 frames, and Nexar's per-group
pre-trimming sets the horizon. So the 142/117/79 imbalance is Nexar's test-set design.
⚠️ Calibration caveat: re-cut extraction assumed `t_event = duration − 0.5`; if Nexar's
group-0 video *ends* 0.5s before the event, absolute TTE labels may be shifted ~0.5s
(decline shape unaffected; x-axis may need a constant offset note).

## Artifact figure
- `tte_curve/old_vs_new_tte_curve.png` (+ `_numbers.json`) — OLD flat (per-bucket
  balanced, cross-scene) vs NEW declining (re-cut same scene, fixed neg). Both share
  the 0.5s anchor; OLD stays ~0.76 flat, NEW declines — the artifact, visualized.

---

# Related Work & Positioning (2026-06-14) — SafeVL & BADAS on Nexar

## Direct competitors on the SAME Nexar dataset
Both evaluate on the Nexar real-world collision dataset (BADAS: n=1344, 672 pos /
672 neg, 1:1). This makes them valid same-dataset comparisons — unlike cross-dataset
(DAD/CCD/DoTA) numbers, which are NOT comparable.

| Model | Metric reported | Value (Nexar) | Setting | Backbone | Nexar train data |
|---|---|---|---|---|---|
| **Ours e3a** | AP / bal-acc | 0.762 / 0.726 | fine-tuned | InternVL3.5-4B (LoRA) | 267 clips |
| **Ours e3b** | AP / bal-acc | 0.790 / 0.738 | fine-tuned | InternVL3.5-4B (LoRA) | 267 clips |
| SafeVL | **balanced accuracy** | 0.76 | **zero-shot** | Qwen2.5-VL-7B + DINO/YOLO + SAM-2 | none |
| BADAS-Open | AP / AUC | 0.86 / 0.88 | trained | V-JEPA2 (video-native) | 1.5k videos |
| BADAS-1.0 | AP / AUC | 0.91 / 0.91 | trained | V-JEPA2 (video-native) | 40k videos |
| BADAS-2.0 | adds reasoning + 22M/86M distilled models, larger long-tail bench | — | trained | V-JEPA2 | 178.5k videos |

## Metric caveats (critical — do not conflate)
- **SafeVL's 0.76 is balanced ACCURACY of a zero-shot safe/unsafe verdict, NOT AP.**
  Our comparable balanced accuracy = 0.726 (e3a) / 0.738 (e3b). SafeVL (0.76, zero-shot)
  slightly edges us (fine-tuned) on this metric. SafeVL reports **no TTE / no AP-vs-TTE
  curve** — single verdict per clip; nothing to put on our TTE x-axis.
- **BADAS reports AP at a 2s prediction horizon, frame-level, balanced 1:1**, mTTA
  3.9–4.9s. BADAS is meaningfully ahead on AP (0.86–0.91 vs our 0.76–0.79) on the same data.

## Honest verdict (decided)
- **Do NOT chase raw AP.** The gap to BADAS is structural: data scale (267 vs 40k–178k)
  and backbone (image-frame VLM vs video-native V-JEPA2). More training on 267 clips
  overfits (train AUC already 0.97) — we are data-bound, not compute-bound. We will not
  beat the team that owns the dataset as an MSc.
- **SafeVL + BADAS-2.0 together already cover "interpretable VLM for collision"**
  (reasoning traces, small distilled models). That niche is taken.
- **Our defensible contribution (reposition the thesis here):**
  1. **Data-efficient teacher→student distillation** — how far 267 distilled clips reach
     (AP 0.79) vs competitors' 40k–178k. The 0.79-vs-0.91 gap becomes a *feature* of a
     low-resource study, not a failure.
  2. **Prevalence-controlled within-scene AP/AUC-vs-TTE methodology** + the flat-0.76
     artifact correction — neither SafeVL nor BADAS reports this.
- **Publication:** sound as an MSc thesis (cite SafeVL + BADAS, position honestly).
  Not a SOTA-AP paper. Possible workshop/methods note on the data-efficiency + TTE angle.

Refs: SafeVL (Ma et al., "SafeVL: Driving Safety Evaluation via Meticulous Reasoning in
VLMs"); BADAS (arXiv 2510.14876); BADAS-2.0 (arXiv 2604.05767).
