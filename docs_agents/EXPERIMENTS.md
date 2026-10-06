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
Caveats: LLM reads the same tokens as A1's head (agreement is not independent evidence); A1 may have trained on the Nexar val windows (unverified); DADA ground truth is one coarse list phrase
per video, so word-for-word match understates reasonable answers (user review, 2026-10-06).

### 2h. Data checks (2026-10-05/06)
HF repos verified: window counts equal the manifests; stored 256×256 frames reproduce the original encoder input exactly (max diff 0.0); Nexar P(collision) from stored frames = recorded A1 score (max diff 0.0000).
Crash-score AUC (A1, crash vs no-crash windows) is 0.975 on the Nexar windows but 0.698 on DADA (3,453 windows): the frozen crash head transfers poorly to DADA (different domain; annotation "abnormal start" often earlier than visible).
Visibility rule verified on all crash windows of both repos: 0 windows with time-to-alert < TTE (+ 0.27 s margin after 2026-10-06).

### 2e. Literature facts used (full log: memory `article_reference_log.md`)
- V-JEPA 2 → LLM (Meta): projector-only stage 1, then full LLM training; 18 M (controlled) / 88.5 M pairs.
- VL-JEPA: frozen V-JEPA2 ViT-L + Llama-3.2-1B layers predicting text embeddings; separate decoder.
- GazeQwen: frozen V-JEPA 2.1 + 1–5 M resampler residuals into frozen Qwen2.5-VL-7B; ~6 k MCQ training items (host MLLM already sees video).
- VLM-AutoDrive: on Nexar data, zero-shot collision recall 0 % for Cosmos-Reason1-7B / NVILA-8B / Qwen2.5-VL-7B; after SFT F1 0.69 / 0.786.
- BADAS-2.0: BADAS-Reason = Qwen3-VL-4B QLoRA on 6,862 samples; teacher saw peak frame + boxes + the human description.
- Qwen3-VL-4B config: LM 36 layers, hidden 2560, interleaved M-RoPE; ViT 1024-d, depth 24, patch 16, temporal patch 2, merge 2, DeepStack layers 5/11/17.
- RunPod storage: volume $0.07/GB/month (billed when stopped), container disk $0.10/GB/month running, not billed stopped; no egress fees.
  HF private storage: 100 GB free.
