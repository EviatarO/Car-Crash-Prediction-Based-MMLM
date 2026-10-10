<!-- handoff-month: 2026-10 -->
# Architecture

Current-state map only. September detail (AA pipeline internals, loss designs, semantic-supervision wiring, captioning pipelines):
`history/2026-09_ARCHITECTURE.md`. Block-by-block crash-path shapes: `ARCHITECTURE_BLOCKS.md`.

## 1. Crash-score path (frozen)
`16 JPEG frames (1280×720, every 4th of 30 fps = 7.5 fps over 2.0 s)` → `preprocess_clip(mode="compress256")` (full frame squashed to 256×256)
→ V-JEPA2 ViT-L (24 layers, A1 LoRA on `query,key,value`) → **2,048 real tokens** (8 tubelets × 16×16, index = t·256 + h·16 + w, 1024-d)
→ BADAS predictor adds 512 "future" tokens → `[2048 real, 512 predicted] = 2560×1024` → crash head (`temporal_processor` attentive probe →
`pooled` 1024 → `classifier` → 2 logits) → `softmax(logits/T)[1]` (T=1 for trained arms, T=2 for published A0).
- `crop` mode (code default) keeps only the centre ~49 % of width — never use for new work.

## 2. Reasoning path (week 1; code on branch `reasoning-path-vjepa2-llm`)
```
frozen encoder tokens (8×16×16×1024, cached once, fp16)  ──►  2×2 regroup (8,8,8, 4·1024)  ──►  Merger (Qwen3-VL merger STRUCTURE,
LN(1024) → concat 4096 → Linear 4096→4096 → GELU → Linear →2560, ~27 M, TRAINED from scratch or Qwen-init A/B)  ──►  512 visual tokens
──►  Qwen3-VL native video prompt: 8 × "<t seconds><|vision_start|>[64 × <|video_pad|>]<|vision_end|>" + question,
      3D M-RoPE ids from the model's own get_rope_index(video_grid_thw=(8,16,16)), DeepStack inputs = none
──►  Qwen3-VL-4B **language model only** (its ViT and merger weights are deleted) — frozen in Phase 1, LoRA in Phase 2  ──►  text
```
- **Why this shape:** Qwen3-VL-4B's own ViT has the same token geometry as V-JEPA2 ViT-L (1024-d, patch 16, temporal patch 2, depth 24),
  so its merger structure and native video-token interface (incl. 3D positions) apply 1:1; only the merger weights are new.
  The text-only LM + 64-query position-less resampler of June (e4) is superseded.
- **Phases:** Phase 1 = alignment (merger only, LM frozen) on DADA valid crash windows, target `"<event>; <cause>"`, question
  "Describe the motion and objects in this clip."; Phase 2 = SFT (merger + LoRA r16/α32 on q,k,v,o,gate,up,down) on DADA + Nexar,
  50/50 crash/no-crash sampler, question "Is a collision coming? Give: collision yes/no, time to impact, event, cause.", one schema for
  both classes: `Collision: yes|no. Time to impact: about X s | none in view. Event: … . Cause: … .` (Nexar uses V12 caption as Event and
  `Cause: not annotated` for both classes). Source tag `[MM-AU]` / `[Nexar]` prefixes every question.
- **Loss:** next-token CE on answer tokens only, mean per sample then per batch; vocab projection computed only at answer positions.
- **Selection / early stop:** validation **wrong-video gap** (loss with another clip's tokens − own), not val loss. Blank control = zero
  V-JEPA features into the merger.
- **Gates (`r1_eval_gates.py`):** wrong-video gap CI > 0, blank gap CI > 0, retrieval@1 of 10 ≥ 2.5× chance (candidates: different video
  and different answer text), distinct ≥ 90 % of generations, facts vs blank run (verdict, time-to-impact bucket, DADA event/cause exact,
  ArA 5-option cause choice by likelihood), plus agreement of the text verdict with the cached A1 P(collision) and caption-overlap metrics.
- **Precision:** encoder forward fp32 at caching, stored fp16 (asserted |x| < 1e4); merger fp32 master weights; LoRA fp32; LM bf16; grad-clip 1.0.

### Data layer
- **Windows:** 16 frames, stride 4 at 30 fps, ending at `collision − TTE` (TTE 0.5/1.0/1.5 s) for crash windows; DADA no-crash window ends
  0.5 s before the abnormal start `t_ai`. **Window visibility rule (same margin for every source since 2026-10-06):** a crash window is valid only if
  its end ≥ hazard start + 8 frames (0.27 s) (DADA `t_ai`; Nexar `time_of_alert`), i.e. time-to-alert ≥ TTE + 0.27 s; invalid windows are kept in
  manifests / `windows_excluded.jsonl` with `drop_reason` but never used. Verified on all crash windows of both HF repos.
- **Phase-1 subset (option b):** DADA crash windows with cached A1 P(collision) ≥ 0.5 train and select; the P < 0.5 windows are evaluated separately
  (`val_dada_lowp`, `g1 --max-p 0.5`).
- **Manifests:** `dataset/manifests/r1_mmau_dada_windows.jsonl` (official ArA split; 17 no-accident videos skipped) and
  `r1_nexar_v12_windows.jsonl` (1,761-window V12 pool; A1's `clip_level_split(val_frac=0.2, seed=0)`).
- **Planned HF repos (approved 2026-10-05, not created):** one private repo per dataset holding only encoder-ready frames:
  `README.md` (dataset card), `windows.jsonl` + `windows.csv` (window_id, video_id, time to alert/abnormal start, time to event/collision,
  TTE group, label, split, window_end_s, 16 source frame indices + fps, valid + drop_reason, reasoning text, r1 targets, ArA (DADA),
  A1 P(collision), preprocess string, shard), `shards/{split}-NNNN.tar` WebDataset with `<window_id>.npz` uint8 (16,256,256,3) lossless,
  frames produced by the V-JEPA2 processor's own resize. Repos: `eviatarO-org/nexar-windows`, `eviatarO-org/mmau-dada-windows`,
  `eviatarO-org/vjepa2-a1-features`, later `eviatarO-org/checkpoints`. Training pulls features only.

## 3. Constraints / invariants
- Splits by video, never by row. Nexar test sets (677/667) and DADA test split are untouched until evaluation; no training source may share
  a video with either Nexar test set.
- The encoder for the reasoning path is the frozen **A1-compress256** trunk with `compress256` (faithfulness: same tokens as the crash score).
- Crash and no-crash targets share one schema (style must not reveal the label). Targets contain ground-truth fields only (no invented
  chain-of-thought steps; MM-AU weather/light/scene/road codes are not used).
- Phase-1 training and its validation use valid crash windows only.
- AP/AUC threshold-free; any CM/P/R/F1/Acc reported at threshold 0.5 with the threshold stated.
- Pods: outputs to `/root`, results downloaded before `runpodctl stop`; durable data goes to HF private repos.

## 4. Files that matter
| path | purpose |
|---|---|
| `README.md` (branch root) | thesis goal (changed), architecture sketch, folder map in reading order |
| `student_training/models/r1_bridge.py` | `regroup_2x2`, `Merger`, `PromptBuilder`, `R1Bridge` (loss, generate, LoRA, save/load) |
| `student_training/scripts/r1_bridge_test.py` | 6 CPU unit tests incl. prompt-id equality with `Qwen3VLProcessor` and logit equality with the official forward; `tiny_qwen()` |
| `student_training/scripts/r1_common.py` | window/visibility rules, questions, `dada_targets`, `nexar_targets`, `parse_phase2`, `TAG` |
| `student_training/scripts/r1_build_manifests.py` | builds both manifests and asserts plan counts |
| `student_training/scripts/r1_download_dada_parts.py` | resumable ordered download of the 58 DADA parts with back-pressure (`--max-ahead`) |
| `student_training/scripts/r1_cache_features.py` | caches 2,048×1024 fp16 tokens + P(collision) per valid window; Nexar from `dataset/train`, DADA from tar parts (`ChainedStream`, waits for parts); resumable sink |
| `student_training/scripts/r1_data.py` | `Cache`, `load_items(source, phase, split)` (phase-1 Nexar items = V12 caption, eval only), `epoch_items` (50/50), `other_index` (wrong-video partner from another video) |
| `student_training/scripts/r1_train.py` | Phase 1/2 trainer, cosine LR with warmup, per-epoch `evaluate` (real/blank/wrong + bootstrap CI), early stop on wrong-video gap, `best.pt` |
| `student_training/scripts/r1_eval_gates.py` | gates → `gates.json`, `summary.md`, `generations.jsonl` |
| `student_training/scripts/r1_pod_run.sh`, `r1_pod_scan.sh` | pod driver (stages setup/tests/pull/get_p1/p1_ab/p1_full/g1/p1_complete/p2/g2/push_ckpt/bundle/check_stop/stop; code in `/root/r1`, outputs in `/root/r1_week1`) and read-only disk scan |
| `student_training/scripts/r1_build_window_repo.py` | manifest → 16×256×256 uint8 windows via the processor's own resize → WebDataset shards + `windows.jsonl/.csv` + card → private HF repo (resumable; DADA streamed from the tar parts) |
| `student_training/scripts/r1_data_review.py`, `r1_review_generations.py`, `r1_phase_report.py`, `r1_review_phase2.py` | local (no GPU) review: HF repo stats + contact sheets / sample export; Phase-1 review file; per-epoch curves + text metrics (token F1, ROUGE-L, BERTScore, embedding cosine / retrieval, list-class accuracy, floor + baseline); Phase-2 review of all validation outputs, yes/no by group |
| `student_training/scripts/r1_pilot_windows.py`, `r1_attention_boxes.py`, `r1_teacher_pilot.py`, `r1_teacher_pilot_report.py` | boxed-teacher pilot: window selection outside the 1,761 pool (normal windows cut with the test-set midpoint protocol); YOLOPv2 + Grounding-DINO + BoT-SORT tracks, A1 crash-head attention per track (hook on `temporal_processor.attention`), boxed frames + 4x4 grids; OpenRouter teacher runner with `service_tier` flex/standard, per-call latency/cost/served tier; report |
| `prompts/PROMPT_SEMSUP_V12BOX.py`, `prompts/PROMPT_TEACHER_STUDY.py` | pilot prompt (blind, closed lists, two answer orders); study prompts e1-e4 built from V12 / v6_balanced / v7.1 strings by asserted replacements, with `build_prompt(variant)`, `REQUIRED`, `audit()` (study file uncommitted) |
| `website/assets/pilot.js`, `website/build_pilot_data.py` | "Boxed-teacher pilot" tab of experiments.html: `createPilotView(deps)`; `pilot_data.js` generated from `outputs/teacher_pilot_2026-10/{windows,boxes}.jsonl` + `runs/*.jsonl` (+ large grids) |
| `outputs/r1_week1/RUNBOOK_pod.md`, `summary.md` | pod runbook (git-ignored folder) and implementation status |
| `student_training/scripts/r0_feature_probe.py` | linear probes on frozen tokens (token position; V12-derived side / type / colour / gap trend) |
| `student_training/scripts/dataset_sample_review.py` | 3 seeded (clip, text) samples per dataset → `outputs/dataset_review_2026-10/<dataset>/` (sources: dada, tau, caviar, mmau, vru, llava, bddx) |
| `student_training/scripts/build_reasoning_path_deck_2026-10.py` | October plan deck; asserts AP and per-TTE counts from `outputs/a1_compress256/scores/*` |
| `dataset/public_samples/` | downloaded public data: `mmau/` (annotation xls/xlsx), `mmau_ara/` (ArA CSVs), `qwen3vl_cfg/`, `tau/`, `caviar/`, `vru/`, `llava_video/` |
| `student_training/models/vjepa_reason.py`, `e4_stage*.py` | June e4 bridge (crop features, 64-query resampler, text-only Qwen3) — superseded prior art |
| `student_training/scripts/semsup_common.py` | `TrainableBadasWrapper` (`forward_clip` → (logits, 2560×1024 patches)), `load_training_examples`, `clip_level_split` |
| `student_training/scripts/aa1_token_probe.py` | `load_probe_model(cfg, "compress256", layers, adapter, "query,key,value")` — reused to load the frozen A1 encoder |
| `student_training/scripts/e4_stageA_badas_open_eval.py` | `preprocess_clip`, `load_badas` |

## 5. APIs added this month (signatures)
- `regroup_2x2(x: (B,2048,C)) -> (B,512,4C)`; `Merger(in_dim=1024, out_dim=2560).load_from_qwen(qwen.model.visual.merger)`.
- `PromptBuilder(tokenizer).encode(tag, question, answer=None) -> (ids, labels, mm_token_type_ids)`; `.collate(items) -> (ids, labels, mm, attention)`.
- `R1Bridge(qwen, tokenizer, init="random"|"qwen")`: `.enable_lora(r, alpha, dropout, targets)`, `.loss_per_sample(feats, batch, mode="real"|"blank"|"wrong", other)`,
  `.generate(feats(1,2048,1024), tag, question, max_new_tokens, mode)`, `.save(path)` / `.load(path)` (merger + LoRA).
- `r1_cache_features.preprocess_frames(vjepa, frames_rgb)` (in-memory twin of `preprocess_clip(compress256)`), `encode(badas, clip) -> (tokens fp16, p)`.
- `PromptBuilder.encode(..., end=True)`: `end=False` scores an answer prefix without `<|im_end|>` (verdict probability).
- `r1_data.load_items(source, phase, split, cache, min_p=None, max_p=None, nocrash_only=False)`: A1-score subset filters; `r1_train --p1-min-p 0.5`, `--stop-on-val-rise N`.
- `r1_eval_gates`: `verdict_probs(bridge, items, caches, dev)` → P(yes) per window = P(" yes")/(P(" yes")+P(" no")); `verdict_report(...)` (AUC/AP vs labels, confusion at 0.5,
  agreement / correlation with A1); `--n-gen 0` = write an answer for every window (`--n-gen-blank` blank-video controls); `--nocrash-only` hallucination check.
- `r1_cache_features.py --from-hf <repo> [--push-repo R] [--max-shards N]` (encode a window repo shard by shard), `--consolidate-from <repo|dir>` (merge feature chunks into the training cache).
- `r1_train.evaluate(bridge, items, caches, device) -> {loss_real/blank/wrong, gap_blank/wrong (+_ci, _frac_pos), n}`.
- Teacher pilot APIs: `r1_attention_boxes.score_tracks(r(8,16,16), tracks, idx, fps) -> (scores{tid}, inside, total)` (attention mass in a track's box cells via `aa4_token_labels.box_to_cells`, tubelets weighted proportional to t+1);
  `r1_teacher_pilot.call(client, model, messages, tier, timeout, retries, temperature) -> {text, usage, served_tier, wall_s, attempts, error}`; `PROMPT_TEACHER_STUDY.build_prompt(variant in e1|e2|e3|e4_tp|e4_tn)`.
