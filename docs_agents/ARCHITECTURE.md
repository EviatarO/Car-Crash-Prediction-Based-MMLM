<!-- handoff-month: 2026-10 -->
# Architecture

Current-state map only. September detail (AA pipeline internals, AA/AA-H/look-ahead loss designs, semantic-supervision wiring, A1-failure
recovery, captioning pipelines, per-script API listings): `history/2026-09_ARCHITECTURE.md`. Block-by-block shapes: `ARCHITECTURE_BLOCKS.md`.

## 1. Crash-score path (frozen, the "fast path")
`16 JPEG frames (1280×720, stride-4 over ~60 source frames ≈ 2 s, ~7.5 fps)` → `preprocess_clip` → V-JEPA2 ViT-L encoder
(24 layers, LoRA on `query,key,value`) → **2048 real tokens** (8 tubelets × 16×16 patches × 1024) → BADAS `backbone.predictor` is fed those 2048
+ 512 mask tokens and returns 512 "future" tokens → `[2048 real, 512 predicted] = 2560 × 1024` into the crash head
(`temporal_processor` attentive probe, 12 learned queries → `pooled (1024)` → `classifier` → 2 logits) → `softmax(logits/T)[1]`.
- Crash head frozen in every shipped arm. Test-time temperature: A0's published scorer uses T=2; every trained arm's scorers use T=1
  (monotone: AP/AUC/CM@0.5 unaffected, Brier/ECE are — never compare calibration across the two).
- **Preprocessing:** `preprocess_clip(vjepa, paths, mode)` / `--preprocess {crop, compress256}`. `crop` (code default, historical) = shortest edge 292 then
  centre-crop 256×256 (keeps ~49 % of width). **`compress256` = full frame resized to 256×256 — use for everything new.** `img_size`/`norm_*` keys in
  `e4_stageA.yaml` are dead config. Token ↔ pixel (compress256): x·256/1280, y·256/720, then /16.
- **Open:** the concat order of the 2560 tokens (real-then-predicted) is assumed, not proven.
- Recipes: champion = A1-compress256; best-by-mean = same recipe with `Caption_Train4500_MidpointNeg_1761.jsonl` negatives. Commands in PROJECT_STATE.md.
  Pass `--lora-target-modules query,key,value` explicitly to reproduce them (108 adapters incl. 36 on the SSL predictor stack, 15.8 % of LoRA params,
  common-mode); the argparse default is now the encoder-only regex (72 adapters).

## 2. Reasoning path (the active direction; code exists from e4, plan is a draft)
**Existing code (e4 Stages B/C, June 2026 — runs on a pod, never locally):**
| file | role |
|---|---|
| `student_training/models/vjepa_reason.py` | `VJEPA2FeatureExtractor` (frozen BADAS; forward-pre-hook on `temporal_processor` → caches the 2560×1024 grid as fp16 + frozen score); `ResamplerProjector` (Perceiver-style: in_proj 1024→512, 64 learned queries cross-attend, optional self-attn when Q>1, FFN, Linear→`out_dim` 2560; ~6 M params; **batch-safe**); `PoolMLPProjector` (avg-pool + 2-layer MLP, weaker fallback); `StageBBridge` (LLM + projector; modes `none/zero/shuffle/mask`; visual tokens scattered via `embeds[vis_mask] = vis_tok.reshape(-1,H)` into `inputs_embeds`; optional `match_embed_norm`; greedy `generate`) |
| `student_training/scripts/e4_stageB_cache_features.py` | caches per-window `(P,1024)` fp16 grid + `cache_manifest_<split>.jsonl` (`assistant_target` copied from the teacher file). ~5.2 MB/window — 60 k windows ≈ 300 GB, beyond the pod volume; cache not kept locally |
| `e4_stageB_train_bridge.py`, `e4_stageB_eval_gate.py` | projector-only training (CE on the reason span; lr 1e-3, 40 ep, early stop) and the gate: Δ-PPL vs random projector and vs text-only, ΔCE with zeroed/shuffled tokens, hazard-lexicon gap |
| `e4_stageC_train_sft.py`, `e4_stageC_eval.py`, `e4_stageC_diagnose.py` | LoRA SFT on the LLM with frozen projector; full-JSON CE + optional **score-consistency anchor** `BCE(p_yes@verdict, BADAS score)`; per-clip generation eval; embedding-scale/chat-template/PPL diagnostics |
| `student_training/data/stageb_bridge_dataset.py` | example = hard-coded Qwen chat template + **64 `pad_token_id` placeholders** after the user header; target = `assistant_target` JSON `{verdict, reason}` (Stage B masks the verdict, Stage C supervises it) |
| `configs/e4_stageB.yaml`, `e4_stageC{,_v2,_v2_qwen3}.yaml` | Qwen3-4B-Instruct-2507 (strong gate) vs Qwen3.5-4B (weak gate); gate thresholds Δ-PPL ≥20 %, ΔCE-zero/shuffle ≥0.05 |
- **Caveat:** every e4 cache/projector was built under `crop` preprocessing and on the 2560-token grid (real+predicted). Reuse requires re-caching under
  `compress256` with the A1-compress256 trunk; a projector trained on cropped features is a distribution shift.
- Teacher text that exists: `dataset/teacher_labels/teacher_dataset_e3b.jsonl` (267 windows / 89 clips, causal hazard reasoning),
  `teacher_dataset_v11.jsonl` (100), V10 (leaky) / V12 (neutral, outcome words banned) / V13 (neutral, 96.9 % shared openers — **rejected by user**).
  Neutral corpora are the wrong target for explanations.

**Planned design (draft plan, not approved — see DECISIONS.md "Unresolved — reasoning path"):**
- Encoder = frozen **A1-compress256 trunk** with its exact preprocessing (faithfulness claim requires the same features that drive the score); stock BADAS-Open only as an ablation.
- `ResamplerProjector` (64 queries) kept; Stage 1 trains the projector only (LLM frozen); Stage 2 = LoRA on the LLM + score-consistency anchor + ~20 % Stage-1 replay.
- **New wrapper (not built):** encode K consecutive 2 s windows (K ≤ 10), resample each to N≈32 tokens with the shared projector, concatenate with a
  learned window-index embedding. Needed because many public captions describe 10–30 s clips while the encoder is trained on 2 s windows.
- **Source tag in the instruction** per training pair (e.g. "[MM-AU]") so style is conditioned on the prompt, not inferred from pixels; Nexar tag at SFT.
- LLM injection must be **native** for the chosen model (its own vision-token slots/position ids); e4 used generic `pad_token` placeholders (hypothesis, unverified, for the weak Qwen3.5 result).
- No detector boxes in the inference prompt; boxes only for training targets, grounding evaluation and one ablation arm.
- Output schema: CAViAR's question set (description → primary reason → at-fault → violated rule); BADAS-Reason's `{reasoning, action}` as the comparison format.
- **Reference baseline to beat (BADAS-Reason, arXiv 2604.05767 §6.2):** Qwen3-VL-4B-Instruct, QLoRA r16 (11.8 M/4.4 B trainable), peak-risk frame with attention bbox crop 256×256,
  6,862 samples, lr 1e-4, batch 16, 3 epochs. Qwen3-VL-4B = SigLIP2-Large + 2-layer MLP merger + Qwen3-4B LM (+ DeepStack injection into the first 3 LM layers).

## 3. Constraints / invariants (must stay true)
- **Splits by `video_id`, never by row**; a clip contributes up to 3 windows (1,761 windows = 1,107 clips). `clip_level_split(seed)`; `--split-seed` and `--init-seed` are separate (default `--seed`).
- Test sets: private `dataset/manifests/test_manifest_hires.jsonl` (677: 338 pos / 339 neg, group 0/1/2 = TTE 0.5/1.0/1.5 s, n=284/233/160) and public (667). **No training source may share a `video_id` with either** — assert in any new dataset class.
- Join caption/label files on `frames_dir` only. Windows: positives at TTE 0.5/1.0/1.5 s before `time_of_event`; negatives at midpoint-based offsets (`MID-10/-4/-8` historic; the midpoint-negatives recipe re-cuts them at the Nexar protocol).
- AP/AUC are threshold-free; CM/P/R/F1/Acc are reported at **threshold 0.5** and the threshold is always stated; no threshold calibration in reported results.
- Reporting rules: ≥3 seeds per arm, same fixed epoch for all seeds, mean ± sd, paired bootstrap against the same-seed twin (`paired_bootstrap_ab.py`), FPR at matched recall instead of FP at 0.5 across seeds, pool private+public (1,344). Seed variance (~0.014 AP) ≫ paired test noise (~0.003).
- Scoring determinism: `--deterministic` (default on) in `semsup_train.py` and `score_checkpoints_on_test.py`; the noise floor of the pre-2026-09-08 scorers was ΔAP 0.0009 (677/677 clips differ).
- I/O must use the concurrent `prefetch_clips` pipeline (the trunk is I/O-bound); captioning uses concurrent `_fetch_one` with `--concurrency` and writes `<out>.usage.jsonl` (real cost).
- Teacher text for **explanation** targets may know the outcome (unlike the retired semantic-supervision captions, which had to pass the leakage gate AUC < 0.75); use blind mode on negatives (V11 lesson: a GT block makes the teacher fabricate hazards on no-crash clips).
- Pods: results synced to local before `runpodctl stop`; outputs go to the container disk; checkpoints persist on volume `0hnvco2s4j` only if written there (it is at quota).

## 4. Files that matter
| path | purpose |
|---|---|
| `student_training/scripts/semsup_train.py` | trainer for every arm (crash-only is `--semantic-weight 0`); concurrent prefetch, `--preprocess`, `--split-seed/--init-seed`, `--deterministic`, prevalence-floor warning |
| `student_training/scripts/semsup_common.py` | `TrainableBadasWrapper` (LoRA wiring, hooks `_captured["patches"/"pooled"]`, `forward_clip`, `prefetch_clips`, `head_state_dict/load_head_state`), `load_training_examples`, `clip_level_split`, SigLIP helpers |
| `student_training/scripts/score_checkpoints_on_test.py` | loads BADAS once, swaps adapters; `--head-states`, `--temperature`, `--preprocess`, metrics via `metrics_core`; `NAME=NONE` = frozen baseline |
| `student_training/scripts/metrics_core.py` | `metrics_from_arrays` — the single metric function (also used by the website builders) |
| `student_training/scripts/paired_bootstrap_ab.py`, `stage_compare.py` | paired bootstrap (reads `ground_truth` or `gt_verdict` rows) and the pre-registered per-seed pass-rule comparison |
| `student_training/scripts/build_midpoint_negatives.py` | builds the midpoint-negatives caption/manifest file from raw mp4s (`--complete-horizons`) |
| `student_training/scripts/e4_stageA_badas_open_eval.py` | A0 scorer + `preprocess_clip`, `load_manifest`, `frame_paths_for` |
| `student_training/scripts/semsup_caption_promptbakeoff.py` | OpenRouter teacher captioning (all prompt versions v2–v13, `--provider-order`, `--token-cap`, concurrency, usage log) |
| `student_training/scripts/aa1_*.py`, `aa4_token_labels.py` | detection/tracking/ego-path/selection pipeline (YOLOPv2 → BoT-SORT → lane path → top-K threat); **reusable as grounding ground truth** for explanation evaluation. v1 (`aa1_detect_track_rank.py`, G-DINO) is superseded — do not extend |
| `outputs/aa1_pool1761/` | `_tracks_v2.json` + `_yolop.json` for all 1,107 pool videos (no threat files); `dataset/aa_token_labels/*.npz` per-window token labels; `outputs/aa1_v2_18clips/` full v2 pipeline on 36 clips |
| `student_training/models/lookahead.py` + `semsup_train.py --lookahead-*`, `--aux-mode/--aux-layer`, `aa_head_losses.py`, `--sem-pooled-weight`, `--per-layer-grads` | **closed/off-by-default research code** (look-ahead head, token-relevance aux, head-attention aux, pooled-tap semantic term, per-layer gradient probe). Do not enable without a new hypothesis; designs in `history/2026-09_ARCHITECTURE.md` |
| `outputs/a1_compress256/`, `outputs/overnight_2026-09-27/`, `outputs/confirm_seeds_2026-09-29/` | champion and midneg/fullpool runs (scores, bootstraps, summaries; adapters pod-only) |
| `website/` (`build_*_data.py`, `serve.py`, `index/dataset/experiments.html`) | local results site; builders pin EXPECTED confusion matrices and read metrics from `metrics_core`; see `WEBSITE.md`; `start_website_background.bat` to serve |
| `docs_agents/ARCHITECTURE_BLOCKS.md`, `CODE_GUIDE.md`, `NEXT_LORA_PLACEMENT.md`, `WEBSITE.md`, `DETECTION_GUIDED_SUMMARY_2026-09-13.md` | supporting docs outside the four handoff files |
| `dataset/BDD-X-Dataset/`, `dataset/Q&A_labels/data/{dada,cap}/*.xls` | local copies of BDD-X annotations and MM-AU accident-reason MCQ labels (short text; useful as anchor facts + timestamps, not as reasoning targets) |

## 5. Modified/added APIs worth knowing (signatures only)
- `preprocess_clip(vjepa, paths, mode="crop"|"compress256")`; `TrainableBadasWrapper(..., preprocess=...)`; `forward_clip(clip) -> (logits (1,2), patches (P,D))` (pooled via `badas._captured["pooled"]`).
- `paired_bootstrap_ab.load_scores(path) -> {video_id: (score, label01)}` (accepts both label conventions).
- `semsup_train.py` flags: `--preprocess`, `--split-seed`, `--init-seed`, `--deterministic/--no-deterministic`, `--head-init`, `--limit-random`, `--horizon-weights[-scope]`, `--lookahead-*`, `--aux-*`, `--sem-pooled-weight`, `--per-layer-grads`.
