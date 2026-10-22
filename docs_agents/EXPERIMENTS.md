<!-- handoff-month: 2026-10 -->
# Experiments

**Index:** `history/2026-09_EXPERIMENTS.md` — Jun → 30 Sep 2026, complete. Outputs live under `outputs/<experiment>/` (git-ignored).
Conventions: AP/AUC threshold-free; any CM/P/R/F1/Acc is at **threshold 0.5**. "private" = 677-clip test, "public" = 667, "pooled" = 1,344.

## 1. Crash-score path (frozen) — final ledger
| arm | preprocess | private AP | public AP | note |
|---|---|---|---|---|
| A0 (published BADAS-Open) | crop | 0.8535 | 0.8711 | paper 0.86 |
| A0-compress256 | compress256 | 0.9067 | 0.9042 | untrained; ΔAP vs crop excludes 0 on both |
| A1 (crash-only LoRA) | crop | 0.8995 | 0.9083 | |
| **A1-compress256 (champion)** | compress256 | **0.9128** | **0.9096** | |
| midpoint negatives (5 seeds, epoch 1) | compress256 | pooled +0.0092 ± 0.0052 vs control | | Kaggle mAP gain not confirmed |
- Correct at threshold 0.5, private, A0 (crop) → A1-compress256: TTE 0.5 s 135→129 /142, 1.0 s 104→101 /117, 1.5 s 69→56 /79, negatives 209→284 /339.
  Public: 127→126 /142, 105→105 /115, 71→52 /77, 196→262 /333. Score files: `outputs/a1_compress256/scores/` (private A1 = pooled file minus public ids).
- Closed negative results (semantic supervision, AA token-relevance, AA-H head attention, look-ahead λ 15.3 / 1.0, horizon weights) and e4 June
  prior art: see the September archive. e4 Stage B Δ-PPL −48 % (Qwen3-4B, 267 windows); Stage C collapsed to 2 templates.

## 2. Reasoning path — measurements (2026-10-03…05)

### 2a. Feature probe on frozen A1-compress256 tokens (`r0_feature_probe.py`, `outputs/r0_feature_probe/n600/`)
600 random windows of the 1,761 pool, 75/25 split by video, linear probes, labels from V12 fields (AI-written, noisy).
| probe | L5 | L11 | L17 | L23 | chance / majority |
|---|---|---|---|---|---|
| token column third (left/centre/right) | 0.933 | 0.963 | 0.915 | 0.857 | 0.333 |
| token tubelet t | 0.959 | 0.911 | 0.795 | 0.676 | 0.125 |
| agent side (bal. acc) | 0.501 | 0.527 | 0.596 | 0.595 | 0.333 (shuffled ~0.37) |
| gap trend (bal. acc) | 0.337 | 0.434 | 0.533 | 0.506 | 0.250 (shuffled ~0.23) |
| event occurs (bal. acc) | 0.604 | 0.643 | 0.660 | 0.740 | 0.500 |
| agent type / colour (bal. acc) | ≈ chance | | 0.327 / 0.258 | 0.290 / 0.248 | 0.250 / 0.200 |
Reading: absolute position, agent side and closing-gap trend are linearly readable; type/colour inconclusive (13 pedestrian, 3 two-wheeler windows, noisy labels). 2.3 h on the laptop GPU (~14 s/window).

### 2b. Window counts after the visibility rule (`r1_build_manifests.py`)
- DADA (1,945 accident videos; official ArA split train/val/test = 1,090/358/497 videos): valid crash windows **3,299** of 4,574
  (TTE 0.5/1.0/1.5 = 1,560/1,118/621); per split crash valid/all train 1,870/3,270, val 629/1,074, test 800/1,491; valid no-crash 717/237/329.
- Nexar V12 pool: with the 8-frame margin (2026-10-06): train **353 valid crash** (161/111/81) + 727 no-crash; val **88 valid crash** (43/28/17) + 178 no-crash (before the margin: 442 / 110; 111 crash windows removed).
  Across the whole pool 304 of 856 crash windows end before `time_of_alert`; over all 750 Nexar train positives, 47/189/397 windows at TTE 0.5/1.0/1.5 s end before the alert.

### 2c. Pipeline verification (local)
- `r1_bridge_test.py`: prompt ids identical to `Qwen3VLProcessor` for a 16×256×256 video at 7.5 fps (588 tokens, grid (8,16,16), 512 video tokens);
  logits identical to the official `Qwen3VLModel` forward on a tiny random model (max diff 0.0); KV-cache generation == full-forward argmax.
- Feature cache: Nexar cached P(collision) equals the recorded A1-compress256 pool scores (`outputs/e4_vjepa_reason/pool1761_scores/A1-compress256.jsonl`)
  within 0.003 on 6 windows. DADA 40 windows from part_aa: crash windows median P(collision) 0.88, pre-abnormal windows 0.09.
- Tiny-LM smoke runs (CPU): Phase 1 and Phase 2 training, per-epoch gaps, checkpoints and the gates script all run end to end.
- DADA part_aa holds 31 sequences (all accident type 8); ~20 are train/val with valid windows.

### 2d. Dataset audits (verified; samples in `outputs/dataset_review_2026-10/`)
| dataset | facts |
|---|---|
| **MM-AU** (HF `JeffreyChou/MM-AU`, CC BY-NC; README: academic use, contact lotvsmmau@gmail.com) | 11,730 videos = CAP 9,768 + DADA 1,962; `video_metadata.csv` has **no fps column**; DADA = 30 fps, 1584×660 PNG; CAP mixes sources (1,350 sequences of exactly 50 frames ≈ CCD 10 fps; others unknown). Text is a closed list: CAP 223 description variants / 109 causes / 107 measures (1,671 distinct text+cause pairs); DADA 114 / 100. `t_ai`→`t_co` median 45 frames (1.5 s) in DADA. Event sentences are 3–6 words in the samples |
| **ArA labels** (`ArA label.zip`, local `dataset/Q&A_labels/`) | one question per video ("What is the cause of the accident?") × 5 options: 11,730 × 5 = 58,650 "pairs"; correct option = the `causes` sentence; official split DADA 1,100/362/500, CAP 6,837/978/1,953 |
| **CAViAR** (`nec-labs-ma/CAViAR`) | 2,249 videos = 1,500 CCD (train, annotations not released) + 749 Nexar (released; all our train positives, 0 overlap with tests). Human-written but templated: Violation == Reason verbatim 744/749; 282/749 are near-misses ("None"); dense caption median 40 words, 333 repeat the cause sentence |
| **VRU-Accident** (HF `kyh9191/VRU-Accident`) | 1,000 pedestrian/cyclist crash videos (≈510 from MM-AU, 100 DoTA, rest manual), all re-encoded to 20 fps (one clip 32 % duplicate frames), 2–34 s; dense captions GPT-4o + human verification, 126–175 words; 7 captions duplicated across different videos; **no timing** |
| **BDD-X** (local CSV) | 6,999 rows with video (5,999 unique BDD100K ids), 26,538 human action+justification segments, median 6 s each, 93.5 % ≥ 2 s; videos only via BDD100K (30 fps, 720p); SafeAuto-BDDX HF mirror is 3 fps / 455×256 (unusable) |
| **TAU-106K** (github `cool-xuan/TABot`) | released val+test video annotations: 2,941 videos (2,711 dashcam, 230 CCTV); 1,279 dashcam accident clips with human captions (median 43 words, normalized timestamps inside the text, accident segment, per-object labels+boxes); 1,432 normal clips with templated text (85 distinct); no train annotations; videos via repo scripts. YouTube estimate (40 sampled): 88 % available, mean 53 MB at ≤720p → **~106 GB** for 2,290 sources (~57 GB segments only); Bilibili 1,056 not measured. 3 samples in `tau106k/` |
| **CrashChat** (HF `KDliang/CrashChat`) | 18,385 videos = MM-AU + 750 renumbered Nexar negatives + 6,313 D²-City; its text = MM-AU list (58 distinct descriptions; negatives answer "No accident, no cause."); CAP mp4s at 10 fps |
| **LLaVA-Video-178K** (0_30_s_youtube) | 79,346 GPT-4o captions, median 229 words; 4,448 road-related, 433 dashcam-like; 30 fps 640×360 |
| **V12 teacher text (Nexar)** | Gemini 3.6 Flash via OpenRouter, blind, ≤ 40 words, outcome words banned; logged cost **$0.0365/window** (900 calls) |

### 2f. Phase 1 — alignment (2026-10-06, RTX PRO 4500; bundle `outputs/r1_week1/pod_phase1_2026-10-06/`, HF `checkpoints/phase1/`)
Merger only (27 M), LM and encoder frozen; DADA crash windows with A1 P(collision) ≥ 0.5 (1,049 train / 334 val; 295 val windows with P < 0.5 evaluated separately);
answer `event; cause`; random init (2-epoch A/B: random wrong-video gap +0.233 [0.201, 0.266] vs Qwen-init +0.187 [0.163, 0.213]); 15 epochs, 990 steps, lr 1e-3, batch 16, ~1.5 h.
Definitions: loss = mean −ln p(correct word-piece) over the answer pieces (+ end token); wrong-video gap = loss with another clip's tokens − own, mean over windows (95 % bootstrap CI);
retrieval = own video picked among 10 (own + 9 others with different text) by text likelihood; cause choice = likelihood-ranked 5 ArA options (event given), chance 20 %.
| measure | A: flagged (334) | B: not flagged (295) | C: Nexar zero-shot (266) |
|---|---|---|---|
| wrong-video gap [CI] | +0.891 [0.786, 0.997] | +0.609 [0.504, 0.716] | +0.040 [−0.092, 0.162] |
| retrieval@1 of 10 | 45.5 % | 29.8 % | 10.5 % |
| cause choice real / blank | 65.9 % / 37.4 % | 59.0 % / 28.8 % | — |
| ALL windows written: exact event / cause / both | 34.7 % / 20.4 % / 12.0 % | 13.6 % / 8.5 % / 2.0 % | — |
| always-most-common-text baseline event / cause | 8.7 % / 14.1 % | 12.2 % / 9.5 % | — |
| BERTScore F1 (floor, baseline) | 0.464 (0.273, 0.268) | 0.376 (0.258, 0.245) | 0.085 (0.078) |
Over-fit: train loss 2.1 → 0.0005; DADA-val loss lowest at epoch 5 (0.538), 0.865 at epoch 15; wrong-video gap kept rising (selected epoch 15). The 50-window sample used first overstated
the model (event 46 % / cause 26 %). No-crash check: 97.0 % of 237 DADA and 91.0 % of 178 Nexar no-crash windows get a DADA crash phrase.

### 2g. Phase 2 — SFT (2026-10-06; bundle `outputs/r1_week1/pod_phase2_2026-10-06/`, HF `checkpoints/phase2/`)
From Phase-1 best; merger lr 2e-5 + LoRA r16 (33 M, lr 2e-4) on q,k,v,o,gate,up,down; 3,667 windows per epoch (DADA 1,870 crash + 717 no-crash, Nexar 353 + 727; 50/50 sampler);
answer `Collision: yes|no. Time to impact: .. Event: .. Cause: ..`; micro-batch 4, eff. batch 16, 1.4 h. Early stop (Nexar-val loss rose epochs 3, 4); kept epoch 3
(Nexar-val gap +0.103 [0.088, 0.117], DADA-val +0.142 [0.127, 0.156]). Train loss 0.69 → 0.22, validation loss flat.
Yes/no = P(" yes") / (P(" yes") + P(" no")) at the verdict word, all validation windows, threshold 0.5:
| set | n (crash / no-crash) | LLM AUC | A1 AUC | acc @0.5 | TP/FN/FP/TN | agreement with A1 @0.5 | Spearman vs A1 |
|---|---|---|---|---|---|---|---|
| Nexar val | 266 (88 / 178) | 0.964 | 0.975 | 88.0 % | 77/11/21/157 | 93.2 % | 0.95 |
| DADA val | 866 (629 / 237) | 0.735 | 0.692 | 73.1 % | 592/37/196/41 | 54.3 % | 0.90 |
LLM "yes" on DADA no-crash windows 83 % (A1: 24 %). Written answers: Nexar crash event BERTScore 0.387 (floor 0.359), time-to-impact bucket 26 % (always 0.5 s: 49 %), 97 distinct answers in 266;
DADA crash exact event / cause / both 23.7 % / 14.8 % / 6.0 % (most-common baseline 10.3 % / 11.9 %), BERTScore 0.500 (floor 0.312), cause choice 53.3 % vs blank 22.1 %, time-to-impact bucket 38.8 %.
Caveats: LLM reads the same tokens as A1's head (agreement is not independent evidence); A1 did NOT train on the Nexar val windows (they are A1's own 348-window val split of the 1,761 pool; verified 2026-10-06), but that split was used for A1's checkpoint choice and is enriched with mined A1 failures, so it is not representative of the Nexar test sets; DADA ground truth is one coarse list phrase
per video, so word-for-word match understates reasonable answers (user review, 2026-10-06).

### 2h. Data checks (2026-10-05/06)
HF repos verified: window counts equal the manifests; stored 256×256 frames reproduce the original encoder input exactly (max diff 0.0); Nexar P(collision) from stored frames = recorded A1 score (max diff 0.0000).
Crash-score AUC (A1, crash vs no-crash windows) is 0.975 on the Nexar windows but 0.698 on DADA (3,453 windows): the frozen crash head transfers poorly to DADA (different domain; annotation "abnormal start" often earlier than visible).
Visibility rule verified on all crash windows of both repos: 0 windows with time-to-alert < TTE (+ 0.27 s margin after 2026-10-06).

### 2i. Boxed-object teacher pilot (2026-10-08; `outputs/teacher_pilot_2026-10/`, website tab "Boxed-teacher pilot", plan `2026-10-08_Plan-BoxedTeacher-Pilot-rev3.md`)
Windows: Nexar train videos OUTSIDE the 1,761 pool. Set A = 9 crash videos × TTE 1.5/1.0/0.5 (time-to-alert ≥ 1.77 s, so all three valid) + 8 normal
videos × 3 windows cut with the test-set midpoint protocol (new folders `dataset/train/<vid>_hires_midtest*`) = 51 windows (set B = 51 more, cut and tracked,
attention only partly done, unused). Box: YOLOPv2 + Grounding-DINO (person/bicycle/motorcycle) + BoT-SORT tracks; target = track with the most crash-head
attention (A1-compress256 `temporal_processor.attention`, tubelets weighted to the end) in the last window, propagated by track ID to the earlier windows
(13 attention-top, 26 propagated, 12 own-top fallbacks). Median fraction of the head's real-token attention inside ANY tracked box: 0.14; the chosen track
gets a median 0.43 of the attention that falls on tracks. Teacher: Gemini 3.8 Flash via OpenRouter, 16 native-resolution boxed frames, prompt
`prompts/PROMPT_SEMSUP_V12BOX.py` (blind, fixed lists + 25–40-word explanation + collision yes/no), 16 calls in parallel.

| run | tier · order | total time | latency median / p95 | cost / window | note |
|---|---|---|---|---|---|
| R1 | Standard · explanation → verdict | 122 s | 23 / 51 s | $0.0155 | 33/51 calls had prompt-cache hits (identical requests to R2) |
| R2 | Flex · explanation → verdict | 148 s | 26 / 56 s | $0.0117 | served tier confirmed `flex` |
| R3 | Standard · verdict → explanation | 176 s | 26 / 67 s | $0.0241 | no cache hits = fair Standard price |
Full 4,446 windows extrapolated: Flex ≈ $52, Standard ≈ $107. Pilot total $2.6.
Blind teacher yes/no vs label: crash TTE 1.5 / 1.0 / 0.5 = 1–2 / 3–4 / 4–5 of 9 per run; normal 22–24 of 24 → strong NO bias (same as the August
blind-verdict teachers: recall 0.22 on val18). R1 vs R3 (order): same verdict 45/51, same class 50/51, explanation token-F1 0.63 → teacher order barely
matters. Across the 3 TTE of a crash video: same class 8/9, same position 6–7/9; "gap closing" on 71–79% of NORMAL windows (not discriminative);
box_ok 94–96%. Known text problem: explanations copy the list words ("A truck/bus travels beside right, moving same direction…").
User check of 00013 TTE 0.5: the car is pulling out of a parking spot without noticing ego (collision follows); all three runs said "stopped car".

### 2j. Teacher prompt study (2026-10-08..09; 51 pilot windows, then the pilot-to-flow design)
Same 51 set-A windows, Flex, `google/gemini-3.8-flash`, temperature 0.1, blind, free-text 25-45-word explanation, ending `risk_score` 0-100 + `collision` yes/no. Prompts in `prompts/PROMPT_TEACHER_STUDY.py`
(E1 V12 neutral, E2 + box, E3 re-balanced v6 + box, E4 debate; E5/E6 anchored on a correct sibling window, E7/E7-pre with hindsight frames). Outcome: E3 is the production blind prompt (the user judged its reasoning good),
it fails on about 28 % of windows (mostly crash windows called "no"); the failures are fixed by the anchored (E6) or hindsight (E7) second look, which is label-informed by design (read for text vs frames, not as accuracy).
Rescue test on 10 videos removed earlier: the E6 chain works when the first anchor is valid. Option (a) (E7 check on every anchored window) rejected as too expensive.

### 2k. Full teacher run (2026-10-10..11; `outputs/teacher_dataset_2026-10/`; flow doc `teacher_distillation/TEACHER_DATASET_FLOW.md` §6-9)
4,203 windows (1,401 videos). Boxes: 1,304 attention-top, 2,486 propagated, 325 own-top, 88 without a tracked object. E3 vs `label_vis`: crash kept 724 / 1,361 yes, normal 2,320 / 2,842 no; of 610 pre-alert windows E3 said no on 404.
Stage 2: E7-pre 206 · E6 412 · E7 280 · E6 chain 233 (final E6C 258 after the normal-window fix). Cost: E3 ≈ $44, stage 2 ≈ $13, predictions $6.4, total ≈ $63. 35 genuine crash outliers removed.
Hand review 2026-10-12: 30 windows rewritten (`edits.jsonl`), "check" list 19 → 0. Final export: **4,169 windows / 1,392 videos / label_vis 1,330 positive, 2,839 negative.**

### 2l. Run 1 + follow-ups on the Nexar reasoning dataset (2026-10-12; `outputs/r1_phase2r_2026-10-12/`, `outputs/r1_options_2026-10-12/`; HF `checkpoints/phase2r/`, `options_2026-10-12/`)
Phase 2 from the old Phase-1 checkpoint (DADA), projector lr 2e-5, LoRA r16 lr 2e-4, eff. batch 16, 50/50 sampler, stop on validation loss rise. Pod RTX PRO 4500, ~3 h + 4.3 h, ~$8. 612 val windows, thresholded numbers at P = 0.5.
| | Run 1 (ep 5) | Opt 1 (ep 3) | Opt 2 (cap) | Opt 3 (box words) | crash model |
|---|---|---|---|---|---|
| val loss | 0.980 | 0.942 | 0.927 | 0.928 | - |
| BERTScore (floor) | 0.398 (0.346) | 0.374 (0.337) | 0.381 (0.344) | 0.383 (0.339) | - |
| distinct expl. / top repeat | 259 / 147 | 148 / 104 | 130 / 73 | 148 / 56 | - |
| road-user match | 0.59 | 0.58 | 0.58 | 0.58 | - |
| wrong-video gap | +0.066 | +0.047 | +0.041 | +0.029 | - |
| AUC S_after / S_before | 0.813 / 0.909 | 0.814 / 0.905 | 0.733 / 0.914 | 0.826 / 0.882 | 0.919 |
S_teacher AUC 1.000 everywhere. Reasoning vs crash model per group (Run 1, McNemar p 0.045): reasoning right where crash wrong: normal 31, pre-alert 34; crash right where reasoning wrong: TTE0.5 23, 1.0 12, 1.5 7.
Fact agreement vs the teacher (outputs/r1_fact_eval_2026-10-13/summary.md): type / side / motion / gap = 0.56 / 0.49 / 0.37 / 0.54 (Run 1) vs majority baselines 0.54 / 0.39 / 0.34 / 0.55; only option 3 learns side (0.68).

### 2m. Session 1 - fresh Phase 1 on the Nexar texts + richer inputs (2026-10-13/14; `outputs/r1_levers_2026-10-13/`, HF `checkpoints/levers_2026-10-13/`; driver `r1_pod_levers.sh`)
Each run = new Phase 1 (projector only, random init, <= 4 epochs, lr 1e-3, still improving at epoch 4) then Phase 2 (<= 5 epochs), best = lowest validation loss; pod 100 GB disk, ~10 h, ~$7-8. Inputs: B3 last layer, B4 + crash-head attention per token, B5 + layer-11 tokens, B45 both (fuse layer starts as identity on the last-layer block).
Extras pass reproduced the crash-model P exactly (max diff 0.0000 on 4,169 windows).
| | Run 1 | B3 | B4 | B5 | B45 |
|---|---|---|---|---|---|
| distinct / top repeat | 259 / 147 | 536 / 6 | 571 / 6 | 566 / 6 | 574 / 4 |
| BERTScore (floor) | 0.398 (0.346) | 0.418 (0.341) | 0.408 | 0.413 | 0.412 |
| wrong-video gap | +0.066 | +0.178 | +0.178 | +0.190 | +0.185 |
| retrieval@10 | 0.40 | 0.52 | 0.49 | 0.51 | 0.56 |
| facts type / side / motion / gap | .56 / .49 / .37 / .54 | .58 / .73 / .42 / .51 | .59 / .68 / .39 / .51 | .57 / .65 / .44 / .57 | .56 / .70 / .43 / .51 |
| AUC S_after / S_before | .813 / .909 | .833 / .847 | .779 / .877 | .851 / .858 | .789 / .878 |
| bal. acc. S_before, threshold from 600 train windows | - | .767 | .813 | .787 | .799 |
Probe on the encoder tokens (`probe/fact_probe.json`, balanced acc.; chance): type .26 / .15 / .28 (.13), side .72 / .60 / .73 (.33), motion .36 / .29 / .35 (.17), gap .58 / .44 / .59 (.33) for last layer / layer 11 / attention-weighted.
Teacher type self-agreement across the 3 windows of one tracked object: 0.88 (measurement ceiling for the type fact).

### 2n. Phase 1 enlargement (2026-10-14..15; `outputs/r1_phase1aug_2026-10-14/`, HF `checkpoints/phase1aug_2026-10-14/`)
4x texts (3 rephrasings per window; Gemini 3.8 Flash writer + checks) and shifted windows; trained P1aug (checkpoint `P1aug/best.pt`, the fixed starting point of every later P2V). Superseded in detail by §2o; the 12,286 shifted-window copies' features were deleted (50 GB) and are not available.

### 2o. Phase 2 in three answer formats (2026-10-15..16; plans `2026-10-15_Plan-Phase2-Three-Formats.md`, `2026-10-16_Plan-Night-Run-Phase2-Three-Formats.md`; `outputs/r1_phase2f_2026-10-15/`)
Formats: reason_first (P2R), **verdict_first (P2V)**, facts_first. Recipe (P2V): from P1aug -> P1fixR (projector, `--init random --init-from P1aug/best.pt`, 4 ep, lr-merger 3e-4, patience 1) -> P2V (6 ep, patience 1, lr-merger 2e-5, lr-LoRA 2e-4, `--select val_loss`, `--drop-prealert --balance-tte --p-teacher 0.4 --val-texts none`, eff-bs 16, micro-bs 4, grad-ckpt); data = original 4,169-window manifest. Writing the reasoning before the verdict hurts ranking (S_after saturated); verdict-first gives the best decision score. Epoch 1 is always the best P2V epoch.

### 2p. P2V vs the crash model and BADAS on the test sets (2026-10-17..18; `outputs/r1_test_eval_2026-10-17/`, `r1_test_eval_public_2026-10-18/`, report `outputs/r1_p2v_report_2026-10-18/{report.md,metrics.json,figures,summary.md,RUNBOOK.md}`)
P2V = seed 0 of the 2026-10-15 run, decision score at "Collision:". Pooled test (private 677 + public 667 = 1,344), crash model = A1-compress256:
| | AP | AUC | Kaggle mAP |
|---|---|---|---|
| P2V (seed 0) | 0.926 | 0.928 | 0.928 |
| crash model | 0.911 | 0.913 | 0.916 |
| delta AP (paired bootstrap) | +0.015 [+0.005, +0.025] | | |
Per-TTE AP (pooled test) 0.5 / 1.0 / 1.5 s: P2V 0.948 / 0.936 / 0.900; BADAS-2.0 (paper) 0.943 / 0.957 / 0.921. Kaggle mAP = mean of per-TTE APs (test pairs crash TTE k with normals of the same horizon 1:1). On validation P2V does NOT beat the crash model (AUC 0.953 vs 0.957); train AUC 0.989 (over-fit). Threshold-dependent tables in the report are at P = 0.5 and at the train-sample threshold (P = 0.56 is stricter and lowers recall: an earlier claim that it would raise recall was wrong). Crash-model seed noise was about +-0.016 AP, so one P2V seed cannot be separated from luck -> seeds plan. `r1_test_refs_join.py`: 431 private windows with an agreed teacher reference; `r1_overlap_check.py`: no near-duplicates train/test. Text metrics (`r1_text_eval.py`) in the report.

### 2q. Extra TTE windows 0.75 / 1.25 s (2026-10-19; `outputs/teacher_extra_2026-10/`, plans `2026-10-19_Plan-Extra-TTE-Windows-Retrain{,-rev2}.md`)
874 windows cut (`r1_extra_windows.py`), teacher flow E3 -> E6 / E7 / E6C (siblings anchored), outcome sentences; flagged / check windows and 10 windows whose final verdict contradicted the label removed; 860 final (743 train: 460 at 0.75 s + 283 at 1.25 s; 117 val: 74 + 43). Rewordings: Gemini 3/window + one rewrite round -> automatic check -> Sonnet 5.5 judge -> Opus blind audit of 300 (250 accepted + 50 rejected). Gate: <= 1 % major errors among judge-accepted (met); minor-nuance 18.8 % (vs 7.2 % on the original data; wording of severity differs, not like-for-like); 13 versions with a contact word ("bumpers meet") removed (rule 8). Version counts: see PROJECT_STATE. Gemini cost within the $40 OpenRouter balance (Flex, parallel).

### 2r. Chain B = new data + gentler recipe, 1 seed (2026-10-19..20; plan rev 2; driver `r1_pod_chains.sh`; pod RTX PRO 4500 80 GB; HF `checkpoints/chains_2026-10-19/`)
Chain B only (chain A deliberately not run). Stages: P1aug-v2 12 fixed epochs (lr-merger 5e-4; training windows 4,300, 14,261 text versions) -> P1fixR-v2 4 fixed epochs (lr-merger 1.5e-4) -> P2V-v2 <= 6 epochs, patience 3 (merger 1e-5, LoRA 1e-4; 3,778 windows, 12,448 versions); top-3 epochs by val AUC (`r1_chain_select.py`); scoring evals val full / private / public; text evals + BERTScore for the best epoch. Total 7.7 h.
- Loss minima: P1aug-v2 val loss min at epoch 6 (1.103) then over-fits; P1fixR-v2 best epoch 1; P2V-v2 early-stopped after epoch 4. Lower learning rates did NOT delay over-fitting.
- Epoch 1 (the only epoch analysed so far): pooled-test AP 0.9204 / Kaggle mAP 0.9177; val core (524) AUC 0.942, val full (641, with new 0.75 / 1.25 s groups) AUC 0.938 -> below today's P2V; decision rule "chain B replaces P2V only with +0.016 AP and >= on val and both test halves" NOT met. Epochs 2-3 + text evals: on HF, analysis pending.
- Files on HF: `B/{P1aug,P1fixR,P2V}/{best.pt,train_log.jsonl}`, `B/top3.txt`, `eval/{B_ep01..03,P2V_old}/{val,private,public}/{metrics.json,predictions.jsonl,summary.md}`, `eval/B_best/{train600,val_text}/...`.

### 2s. Test-set teacher reasoning (2026-10-17..)
Private 677: pass 1 = E3 blind teacher (plan `2026-10-17_Plan-Test-Private-Addendum-Pass1-Only.md`); pass 2 anchored (`PROMPT_TEACHER_TEST_P2.py`) piloted on 20 windows, awaiting the user's review. Texts are for final evaluation only.

### 2t. Seeds plan (superseded: seeds now come LAST, after stage 3 and 3.1; `2026-10-20_Plan-Seeds-Of-Todays-P2V.md` kept as the template)
Stage 4 = 5 seeds of the FINAL recipe end to end (every stage, new seed), the test sets scored once (seeds + ensemble vs the crash model and BADAS-2.0, P2V as reference). Rules R1-R3 of that plan still apply.

### 2u. Stage 1 (2026-10-20; `outputs/r1_next_2026-10-20/summary.md`, HF `checkpoints/next_2026-10-20/`) - validation only
R2 = regularised Phase 2 (LoRA r8, dropout 0.2, wd 0.05, checks at 0.5..2 epochs, best by val AUC); D2 = relaxed rewordings (9,497 accepted + 2,585 nuance-only judge rejects); R1 skipped by the pre-set rule (R2 best at 1.5 epochs). One seed each.
| | val core AUC | val full AUC | 1.25 s AUC | train AUC | best | BERTScore (floor) | wrong-video gap | fact mean4 (motion / gap) |
|---|---|---|---|---|---|---|---|---|
| today's P2V | 0.953 | 0.951 | 0.931 | 0.989 | ep 1 | 0.425 (0.335) | +0.292 | 0.606 (0.591 / 0.641) |
| R2 | 0.951 | 0.948 | 0.923 | 0.990 | 1.5 ep | 0.425 (0.335) | +0.344 | 0.622 (0.595 / 0.696) |
| D2 (ep 3, val-loss rule) | 0.952 | 0.950 | 0.925 | 0.993 | ep 3 | 0.426 (0.338) | +0.298 | 0.620 (0.595 / 0.663) |
Decision gate (+0.005) not met. Text gate (+0.03 on motion / gap / mean4): R2 passes on gap (+0.055); D2 fails. Fusion on D2 (crash model + D2, val core AUC 0.9574 / 0.9524): fixed w 0.6 -> 0.9598 (+0.0024), learned gate 0.9562, "pick the more confident path" 0.9571, "pick the path more likely right" 0.9572, small MLP trained on the train sample 0.9427 (the models are over-fitted there), MLP 5-fold CV on val 0.9568: gate +0.005 not met, fusion CLOSED (`outputs/r1_next_2026-10-20/fusion_D2/`, `r1_fusion.py`, `r1_fusion_mlp.py`).

### 2v. Stage 2 #4: B4 (crash-head attention as projector input) vs control (2026-10-21; `outputs/r1_b4_2026-10-21/summary.md`, HF `checkpoints/b4_2026-10-21/`)
Both chains on D2's recipe, fresh Phase 1 (<= 10 ep, patience 2) -> P1fixR -> P2V; identical except `--input last+attn` vs `last`. One pod, ~8.5 h.
| | val core | val full | 1.0 / 1.5 s AUC | BERTScore | wrong-video gap | right clip of 10 | motion / gap / mean4 |
|---|---|---|---|---|---|---|---|
| B4 | 0.9534 | 0.9522 | 0.935 / 0.912 | 0.426 | +0.310 | 0.61 | 0.626 / 0.694 / 0.623 |
| control | **0.9592** | **0.9577** | 0.942 / 0.930 | 0.426 | +0.310 | 0.52 | 0.587 / 0.631 / 0.606 |
Decision gate (B4 >= control + 0.005) not met (-0.0058). Text gate nearly met (+0.039 motion, +0.063 gap) but B4's decision is just outside the -0.005 tolerance; B4 needs the crash head at inference -> not a final candidate. The control is the best validation decision so far (+0.006 over today's P2V).

### 2w. Single-stage recipe, 2 seeds (2026-10-21/22; `outputs/r1_base_2026-10-22/summary.md`, HF `checkpoints/base_2026-10-22/`)
ONE projector stage from random on P1fixR's data (pre-alert dropped, TTE-balanced, reasoning first; <= 10 ep, patience 2, lr 1e-3), then P2V as the control. One pod, two parallel seeds, ~6.5 h.
Val core / full AUC: seed 0 0.9515 / 0.9494, seed 1 0.9451 / 0.9461 (mean 0.9483 / 0.9478, spread 0.0064); control 0.9592 / 0.9577. BERTScore 0.415 / 0.418 vs 0.426; mean4 facts 0.594 / 0.596 vs 0.606. **Rule: difference 0.0109 > band max(0.005, 0.0064) -> the control (two-step Phase 1) is selected.** Dropping pre-alert windows inside Phase 1 did not help.

### 2x. Stage 3 (encoder LoRA trained by the LM loss only; code done, no result yet) - see PROJECT_STATE "Stage 3"
Measurements made while building it (real BADAS weights, local GPU): encoder-only tokens == cached tokens after fp16 rounding (0.0); stochastic depth 0.1 makes train-mode tokens random (relative change 0.73) unless zeroed; with it zeroed, train-mode tokens vs eval fp32: TF32 0.4 % (cos 1.0001), bf16 autocast 13 % (cos 0.982); encoder fwd + bwd peak 4.95 GB (micro-batch 2, checkpointing). Pod timings of the three precisions: pending (`r1_frames.py bench` on the pod).

### 2e. Literature facts used (full log: memory `article_reference_log.md`)
- V-JEPA 2 → LLM (Meta): projector-only stage 1, then full LLM training; 18 M (controlled) / 88.5 M pairs.
- VL-JEPA: frozen V-JEPA2 ViT-L + Llama-3.2-1B layers predicting text embeddings; separate decoder.
- GazeQwen: frozen V-JEPA 2.1 + 1–5 M resampler residuals into frozen Qwen2.5-VL-7B; ~6 k MCQ training items (host MLLM already sees video).
- VLM-AutoDrive: on Nexar data, zero-shot collision recall 0 % for Cosmos-Reason1-7B / NVILA-8B / Qwen2.5-VL-7B; after SFT F1 0.69 / 0.786.
- BADAS-2.0: BADAS-Reason = Qwen3-VL-4B QLoRA on 6,862 samples; teacher saw peak frame + boxes + the human description.
- Qwen3-VL-4B config: LM 36 layers, hidden 2560, interleaved M-RoPE; ViT 1024-d, depth 24, patch 16, temporal patch 2, merge 2, DeepStack layers 5/11/17.
- RunPod storage: volume $0.07/GB/month (billed when stopped), container disk $0.10/GB/month running, not billed stopped; no egress fees.
  HF private storage: 100 GB free.
