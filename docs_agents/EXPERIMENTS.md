<!-- handoff-month: 2026-10 -->
# Experiments

**Index (full logs, every run's config and tables):**
- `history/2026-09_EXPERIMENTS.md` — Jun → 30 Sep 2026, complete (superset of `history/2026-08_EXPERIMENTS.md`, a 16-Sep snapshot).
- Outputs live under `outputs/<experiment>/` (gitignored; adapters stay on the pod, locally only metadata/scores).

Conventions: AP/AUC threshold-free; any CM/P/R/F1/Acc is at **threshold 0.5**. "private" = 677-clip test, "public" = 667-clip test, "pooled" = both (1,344).

## 1. Result ledger (final numbers; detail in the September archive)

### 1a. Crash-score path
| arm | preprocess | private AP | public AP | note |
|---|---|---|---|---|
| A0 (frozen BADAS-Open) | crop | 0.8535 | 0.8711 | published BADAS-Open ≈ 0.86 (reproduced 0.861) |
| A0-compress256 | compress256 | 0.9067 | 0.9042 | zero training; ΔAP vs crop +0.0532 [0.0356, 0.0727] private, +0.0331 [0.0133, 0.0547] public — both exclude 0 |
| A1 (crash-only LoRA) | crop | 0.8995 | 0.9083 | recorded 0.900/AUC 0.904 |
| **A1-compress256 (champion)** | compress256 | **0.9128** | **0.9096** | identical recipe to A1; vs A1 pooled ΔAP +0.0071 [−0.0034, 0.0180] |
| control (A1-c256 + 2 ctrl seeds) | compress256 | pooled mean-of-9 0.9077 | | |
| **midpoint negatives** (3 seeds × 3 ep) | compress256 | pooled mean-of-9 **0.9158** | | paired ΔAP +0.0082, 9/9; FP@0.5 156→104, specificity +8 pt, FN 91→137 |
| full pool 4,446 (old negative sampling) | compress256 | pooled mean-of-9 0.8986 | | ΔAP −0.0091, 1/9 — more data with the old sampling does not help |
- 5-seed confirmation of midneg (epoch 1 fixed in advance): pooled AP +0.0092 ± 0.0052, 5/5 seeds (p=0.03) → confirmed; Kaggle mAP +0.0037 (4/5) and 1.5 s AP −0.007 (2/5) → **not confirmed**. Midneg lifts 0.5/1.0 s TTE and slightly hurts 1.5 s (0.8852→0.8705).
- Seed spread is a **calibration offset**, not detection: e.g. midneg-seed2 ep1 misses 265/672 crashes at 0.5 yet has the 2nd-best AP; at matched FPR all arms find the same TP per TTE. At FPR 10 % only ~50 % of 1.5 s crashes are caught by any arm.
- Seed variance ~0.014 AP ≫ paired test noise ~0.003; rank-1 val-selected checkpoints have a 0.0204 AP range across two seeds, mean-over-8 only 0.0048.

### 1b. Closed negative results (crop-era unless stated)
| thread | result |
|---|---|
| Semantic supervision, 1,761 pool (B-v1/v2/v3, P1) | test AP 0.8880 / 0.8797 / 0.8784 / 0.8266 vs A1 0.900; P1 below A0. Mechanism checks: caption info survives pooling (B1 probe 22× chance), semantic gradient reaches `pooled` ≥ noise (P3), gradients near-orthogonal (cos −0.04…+0.05). Arms differ mostly by optimal threshold (A1 0.812, B-v3 0.173), not ranking |
| SemTest-200 v1/v2 (unfrozen head) | all arms val AUC ≈ 0.49–0.50; head moved < 0.05 % (v1 confound), v2 repeated the null |
| A1-failure recovery (321 windows, 4 arms from A1's weights) | on 61 val windows all four arms bit-identical (39/61); test AP a1cont 0.8956, V10 0.8976, V12 0.8972, v12shuf 0.8963 |
| Recovery-family bootstrap (n=677, 5000 resamples) | V12−v12shuf ΔAP +0.0009 [−0.0007, +0.0026]; a1cont−V10 −0.0020 [−0.0032, −0.0010]; a1cont−V12 −0.0016 [−0.0033, −0.0001]; V10−v12shuf +0.0013 [+0.0002, +0.0026]; others cross 0. **Noise floor** (same A1 checkpoint scored by two code paths): ΔAP −0.0009 [−0.0035, +0.0009], 677/677 clips differ, mean\|Δ\| 0.0078, max 0.0973, 5 flips of the 0.5 decision → no ordering in this family is established; the null is not threatened. Raw JSON `outputs/a1fail321/bootstrap/` |
| Stage AA token-relevance aux loss (6 runs) | none separated from its control; frozen-trunk probe shows a real layer-17 gain (rel-Spearman +0.098 vs +0.054 control) that does not reach layer 23 / the head input (+0.056 vs +0.049) |
| Stage AA-H head-attention supervision | `attn_rank` null at matched recall; `attn_mass` ep4 calibration shift only, ep8 −0.030 AP (val_ap picked it); attention was moved (ρ_P/V up to 195) with AP flat → attention placement is not the bottleneck. Pooled re-analysis corrected the earlier FP-based verdicts |
| Look-ahead head (1.5 s) | Stage 0a passes (94 % of missed 1.5 s crashes are caught by the same video's later clip); 0b offline passes but its control was a frozen untuned readout (trained probe on z_now AP 0.7955 > 0.7782). Stage 2 λ=15.3: la-full ≈ la-auxonly < plain midneg (1.5 s AP 0.8699 vs 0.8776; Kaggle 0.9120 vs 0.9164); λ=1.0 pre-registered follow-up FAILS (1.5 s −0.0032 vs midneg, −0.0074 vs symmetric weights). Closed |
| Horizon-weighted crash loss | positives-only: no gain, FP +57 %; symmetric (3 seeds, ep1): 1.5 s AP +0.0042, n.s. (t≈1.5), Kaggle +0.0019 — the baseline bar, inside seed noise |

### 1c. Reasoning-path prior art in this repo (June 2026, e4; archived)
- Stage B (projector only, 267 windows/89 clips): Qwen3-4B-Instruct Δ-PPL −48 % vs random projector — gate passed; Qwen3.5-4B only +4.9 %.
- Stage C v2 on Qwen3-4B (LoRA r8, anchor weight 1.0, 18 val clips): best at epoch 1 (val CE 2.071, val PPL 7.93 → 12.03 by ep7); **88.9 % verdict agreement with the vision score**, 55.6 % GT agreement (BADAS's own false positives propagate), free generation = **two templates** keyed on the verdict. On Qwen3.5: 72.2 % agreement, confabulated "white van from rear" on 8+ unrelated clips.
- Teacher text audit: 267 windows/89 scenes, 0 duplicate reasons, distinct-3 0.53, mean 45 words — the teacher is not the source of the collapse. ep5/ep7 generation eval **never run**.

## 2. Data audits for the reasoning path (2026-10-02/03; all re-runnable)
**CAViAR (`nec-labs-ma/CAViAR`, `data/test.json`) — released human reasoning on our Nexar train positives.**
- 749 videos, 7,407 QA (MCQ + open). Ids are zero-padded Nexar ids (`video_path` e.g. `00776`).
- **749/749 are in our Nexar train set and are positives (`target=1`; we have 750 positives); overlap with private 677 = 0, public 667 = 0.**
- Human-written (GPT-4 only fixed grammar). Open-answer length (words, median/max): detailed step-by-step description 40/84; accident reason 37/93; violation 37/93; summary 12/26; at-fault ~10/26; victim ~11/26.
- Accident type answers: none (near-miss) 282, rear-end 197, T-bone 165, side-by-side 100, head-on 5. Items per benchmark: Dense Captioning 1,475, Weather&Light 1,498, Accident Type/Road Conditions 749, Accident Reason/Violation 744, Faulter/Victim 724.
- Not released: its CCD train/holdout annotations and any video files (videos come from HF `nexar-ai/nexar_collision_prediction`, Open Data License).

**Local label files (short text, confirmed):**
- BDD-X `BDD-X-Annotations_v1.csv`: 12,997 rows; 26,538 action+justification pairs; median 6 / 7 words (mean 6.0 / 7.9); 6,198 unique actions of 26,538 (top "the car is stopped" ×1,820); ego-behaviour only, no hazards. 6,999 rows carry a video URL — the first (`s3-us-west-2…/samples-1k/*.mov`) returns **404**; videos must come from BDD100K.
- `Q&A_labels/data/dada/*.xls` (train 1,100 rows) and `cap/*.xls` (train 6,837 rows) = **MM-AU accident-reason MCQ**: one fixed question "What is the cause of the accident?", 5 options (6–12-word phrases from a closed set), answer = index; also accident-window anchors (abnormal start / accident frame / end; `tai/tco/tae`) and weather/lighting/scene/location codes (Chinese headers). Classification dressed as QA — useful as anchors/timing only.

**Public datasets checked (✓ = page fetched; ~ = search snippet only):**
| dataset | verified facts |
|---|---|
| MM-AU (CVPR 2024) ✓ | HF `JeffreyChou/MM-AU` public, not gated, 526 GB, cc-by-nc-4.0; CAP-DATA ~214 GB (8,218 video seqs, JPG) + DADA-DATA ~131 GB (1,962 seqs, PNG); per-video `texts` (~8 words), `causes` (~14), `measures` (longer) in `video_metadata.json/csv`; `t_ai/t_co/t_ae`; GitHub README says "ONLY free for academic use … contact lotvsmmau@gmail.com" (conflicts with the open HF listing) |
| RoadSafe365 (arXiv 2602.07212, 2026) ✓ | 36,196 clips (dashcam + CCTV, split unreported), 10–20 s, dense 150–200-word captions (GPT-4o) + Gemini-2.5-Pro attributes + 200 K MCQ, all human-verified; caption prompt in App. B ("describe only what is visually present … one coherent paragraph"); **no download link or release statement found** |
| WTS (ECCV 2024) ✓ | Google-Form request → Dropbox/Drive; 255 staged scenarios/1.2 k segments + ~4,800 BDD100K pedestrian videos (`BDD_PC_5K`, **annotations only — videos from BDD100K**); 58.7 words/segment, 5 phases (pre-recognition, recognition, judgement, action, avoidance); built as human checklist → GPT-3.5 phrasing → manual check |
| DRAMA (WACV 2023) ✓ | `usa.honda-ri.com/drama`, university-email request, non-commercial; 17,785 two-second clips, Tokyo (left-hand traffic), risk-object boxes + free-form captions + VQA. The `github.com/mtli/DRAMA` link from the Gemini plan is **404** |
| LLaVA-Video-178K (2024) ~ | HF `lmms-lab/LLaVA-Video-178K`, academic use; 178,510 captions + 960,792 open QA, synthetic |
| VRU-Accident ✓ | HF `kyh9191/VRU-Accident`, Apache-2.0, 1,000 dashcam videos, 6 k MCQ + 1 k dense captions, creation method undocumented |
| TAR ~, AccidentBench ~, CrashSight ~, TAU-106K, SUTD-TrafficQA, LingoQA ~ | evaluated and excluded as training sources (CCTV-heavy / MCQ / unverifiable annotation method / 2021 / 4 s @ 1 Hz) — reasons in DECISIONS.md. TAR (2026, Gemini 3.1 Pro + Gemma-4 CoT traces): Qwen3-VL-8B answer-only 52.9 → reasoning-SFT 53.9 → +CoT inference 54.4 |

**Literature facts used by the plan:**
- BADAS-2.0 (arXiv 2604.05767): BADAS-Reason = Qwen3-VL-4B QLoRA on 6,862 samples (8,680 videos; Gemini converted human descriptions to `{reasoning, action}`, 99.6 % parse), test n=887: loss 2.657 → 0.612, PPL 14.25 → 1.84, action-match 12.2 % → 43.6 %; bbox-crop input beats heatmap overlay (val loss 0.612 vs 0.628). Heatmap pointing-game accuracy: random 11.5 %, BADAS-1.0 49.8 %, BADAS-2.0 52.4 %, Flash-Lite 69.8 %, Flash 72.1 %.
- V-JEPA 2 → LLM (Meta): projector (typically MLP), 3-stage progressive alignment (projector on image captions → full model on image QA → video caption/QA), **88.5 M** image/video-text pairs (scaled from 18 M, which was worse), PerceptionTest 84.0 with Llama 3.1 8B, frozen-encoder variants reported. Prismatic VLMs: language-free encoders (DINOv2) underperform language-aligned ones (SigLIP) under the same projector/data; fusing both helps 5–10 %.
