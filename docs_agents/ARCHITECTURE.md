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
  0.5 s before the abnormal start `t_ai`. **Window visibility rule:** a crash window is valid only if its end ≥ `t_ai` + 8 frames (DADA) /
  ≥ `time_of_alert` (Nexar); invalid windows are kept in manifests with `drop_reason` but never used.
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
| `student_training/scripts/r1_pod_run.sh`, `r1_pod_scan.sh` | pod driver (stages; outputs to `/root/r1_week1`) and read-only disk scan |
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
- `r1_train.evaluate(bridge, items, caches, device) -> {loss_real/blank/wrong, gap_blank/wrong (+_ci, _frac_pos), n}`.
