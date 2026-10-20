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
| `prompts/PROMPT_SEMSUP_V12BOX.py`, `prompts/PROMPT_TEACHER_STUDY.py` | pilot prompt (blind, closed lists, two answer orders); study prompts e1-e4 built from V12 / v6_balanced / v7.1 strings by asserted replacements, with `build_prompt(variant)`, `REQUIRED`, `audit()`  |
| `website/assets/pilot.js`, `website/build_pilot_data.py` | "Boxed-teacher pilot" tab of experiments.html: `createPilotView(deps)`; `pilot_data.js` generated from `outputs/teacher_pilot_2026-10/{windows,boxes}.jsonl` + `runs/*.jsonl` (+ large grids) |
| `student_training/scripts/r1_teacher_windows.py`, `r1_pod_cut_windows.py`, `r1_pod_get_videos.py`, `r1_teacher_pod.sh`, `r1_merge_pilot_boxes.py`, `r1_rescue_windows.py` | teacher run step A/B: manifest `windows.jsonl` (4,203 windows, `label`, `pre_alert`, `label_vis`, `frame_idx`), pod cutting + HF video download, pod driver (setup/hf_check/smoke/batches/push/verify/auto/stop; tracks with 6 procs then attention with 3), reuse of pilot boxes, rescue test |
| `student_training/scripts/r1_teacher_pilot.py` (runner, `--prompt e1..e7 --set A|B|R|full`, retries failed calls on resume), `r1_teacher_flow.py` (stage-2 routing → `flow_final.jsonl`, chain parallel over videos), `r1_add_prediction.py` (text-only short prediction → `runs/*P2.jsonl`), `r1_after_frames.py` (`ensure()`) | teacher calls (step C) |
| `prompts/PROMPT_TEACHER_STUDY.py` | E1-E7 prompts: `_e3_head(box)`, `build_anchor_prompt` (E5/E6), `build_hindsight_prompt(n_after, label, pre_alert=False)` (E7 / E7-pre), `PREDICTION`, `PRE_ALERT_NOTE`, `REQUIRED`, `audit()` (committed) |
| `website/build_teacher_data.py` → `website/teacher_data.js` (git-ignored) | builds the "Teacher reasoning" rows, `dataset_final.jsonl`, `check_windows.txt`, `removed_windows.json`; applies `outputs/teacher_dataset_2026-10/edits.jsonl` (new text wins, edited windows never dropped, `label_vis` from the edit) |
| `website/dataset.html` (tab `tch*`), `website/assets/pilot.js` | generalized view (`deps.prefix`, `pageSize`, `D.path_opts`, `D.status_opts`, `D.hide_cols`); notes in localStorage `ccp:review-notes:v1` with export/import JSON |
| `teacher_distillation/TEACHER_DATASET_FLOW.md`, `RUNBOOK_pod_stepB.md`; `reports/_scripts/_build_progress_report.py`, `reports/figures/*` | flow doc (results §6-9), pod runbook; progress report generator and architecture / routing figures (reports/ git-ignored) |
| `student_training/scripts/r1_reasoning.py` | decision scores: `logodds(bridge, items, prefixes, caches, dev)` = log P(" yes") - log P(" no") at the next piece after a prefix ending "Collision:"; `prefix_for(kind, item, expl)` (before / teacher / after / first); `epoch_decision_report` |
| `student_training/scripts/r1_eval_reasoning.py` | evaluation on the 612 val windows: loss gaps, retrieval@10, S_before / S_teacher / S_after, generations, train-chosen threshold, paired tests (`compute_metrics`), text numbers; `--before-ckpt`, `--old-ckpt` |
| `student_training/scripts/r1_review_reasoning.py` | PC-side BERTScore / ROUGE-L / BLEU-4 / token-F1 with chance floors -> `text_metrics.json`, `report.md` |
| `student_training/scripts/r1_fact_eval.py`, `r1_fact_probe.py` | rule-based 4-fact extractor (type / side / motion / gap) + agreement vs the teacher; small MLP probe per fact on pooled encoder tokens or on the projector output (`--projector-ckpt`) |
| `student_training/scripts/r1_shift_windows.py`, `r1_paraphrase_check.py` | shifted-window manifests (+ control set) with the visibility rule; paraphrase batch export / validator / merge (`paraphrases.jsonl`, rejects, flags) |
| `student_training/scripts/r1_pod_run.sh` (Run 1 stages), `r1_pod_options.sh`, `r1_pod_levers.sh`, `r1_pod_phase1aug.sh` | unattended pod drivers; each verifies the result bytes on HF before `runpodctl stop` |
| `student_training/scripts/r1_build_reasoning_manifest.py`, `r1_build_window_repo.py` (`--source nexar_reasoning`) | reasoning manifest (train / val by video; `gt.box_hint`) and the HF window repo `eviatarO-org/nexar-reasoning-windows` |
| `website/build_r2_data.py`, `website/assets/r2.js` | tab "Reasoning model - runs" (8 runs, comparison, probe, sortable window table); `r2_data.js` is generated and git-ignored |
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
- **Reasoning answer format** (`r1_common.py`): `q_reason(order)`, `reasoning_target(expl, pred, label_vis, order="reason_first"|"verdict_first")`, `parse_reasoning(text)`, `reasoning_group(label, tte, pre_alert)`; prompt lists Reasoning / Collision / Outcome in the answer's order.
- **Merger inputs** (`r1_bridge.py`): `INPUTS = {"last": (1,False), "last+attn": (1,True), "l11+last": (2,False), "l11+last+attn": (2,True)}`; `Merger(in_dim, out_dim, n_layers, attn)` - extended inputs add per-layer LayerNorm + a `fuse` Linear that STARTS as identity on the last-layer block (zero elsewhere); channel order [layer 11 | last | attention]; `R1Bridge(..., inp=...)`.
- **Cache** (`r1_data.py`): `Cache(dir, source, inp)` returns the channel-concatenated tokens (attention scaled to mean 1 per window, log1p); extra files `features_<src>_l11.npy`, `attn_<src>.npy`; sources `nexar_reasoning`, `nexar_reasoning_shifted`, `nexar_reasoning_control`.
- **Sampler** (`r1_data.py`): `epoch_items` draws k = min(neg, pos) from BOTH classes (negatives stratified by `group`), then `realize()` picks one (window version x text version) pair per original item (`attach_aug`, `load_texts`, `val_pairs` for the enlarged validation); the old DADA+Nexar path is bit-identical.
- **r1_train flags:** `--sources nexar_reasoning --input --order --select gap|val_loss --template-cap N --box-hint --aug-texts --aug-shifts --val-texts --val-shifts --p-teacher`; Phase 1 is allowed on the reasoning source.
- **r1_cache_features:** `load_model_extras()`, `encode_extras(...)`, `run_from_hf_extras` (`--extras-only`), `consolidate_extras` (`--with-extras`), `--extras` for local runs; frames root override `R1_TRAIN_DIR`.

## 6. Added 2026-10-15..20 (verdict-first model, extra TTE windows, chain B)
- **Model/answer (unchanged architecture):** frozen A1-compress256 tokens -> projector (Merger) -> Qwen3-VL-4B LM (+LoRA r16 in Phase 2). Headline format = **verdict-first (P2V)**: `Collision: yes|no. / Reasoning ... / Outcome ...`; decision score = log P(" yes") - log P(" no") at "Collision:" (no generation needed). Training stages: P1aug (projector only, LM frozen) -> P1fixR (projector, init from P1aug) -> P2V (projector + LoRA). Crash head not used by the final model; crash model and reasoning model stay two separate architectures.
- **Training invariants:** balanced epochs (equal crash / normal, TTE-balanced positives), `--drop-prealert`, text drawn teacher-with-p 0.4 else a rephrasing, validation = teacher text only (`--val-texts none`), `--select val_loss`; chain B adds `--fixed-epochs` and top-3 epochs by val AUC (`r1_chain_select.py`).
- **Data layer:** `R1_REASONING_MANIFEST` env overrides the manifest in `r1_data.py`, `r1_cache_features.py`, `r1_fact_cards.py` (`MAN`), paraphrase scripts (`R1_PARA_DIR`). `Cache` accepts comma-separated dirs (old + extra caches); `load_items(..., drop_extra_train)` drops rows flagged `extra:true` from train (`--no-extra-train` = control without the new windows). Eval groups include `tte0.75`, `tte1.25`; `stratified_sample` for `--sample`. Manifests: `r1_nexar_reasoning_{extra (860, extra:true), v2 (5,029), test, test_public}.jsonl`.
- **Pipeline for new windows** (`student_training/scripts/`): `r1_extra_windows.py` (cuts 874 windows to `dataset/train/<vid>_hires_tte075|tte125`, writes `windows_extra.jsonl`) -> `r1_extra_setup.py` (builds `outputs/teacher_extra_2026-10/` with new + sibling windows, tracks / attention / runs copied) -> `r1_extra_pipeline.sh` (unattended: `r1_teacher_flow.py --process-horizons 0.75,1.25`, outcome sentences, paraphrases via `r1_paraphrase_gemini.py` -> `r1_paraphrase_check.py` -> `r1_paraphrase_judge.py`, `r1_fact_cards.py`) -> `r1_extra_gates.py`, `r1_extra_merge.py` (dataset + manifest; removes `check` windows), `r1_extra_audit_report.py` (Opus audit gate).
- **Reports / evaluation:** `r1_test_report.py` (AP / AUC / Kaggle mAP, paired bootstrap), `r1_full_report.py` (full P2V report: splits train sample / val / private / public / pooled, CM per TTE at P = 0.5 and at the train-sample threshold, figures), `r1_text_eval.py`, `r1_test_refs_join.py` (431 agreed private references), `r1_overlap_check.py`, `r1_eval_reasoning.py` (`--sample --no-generate --no-extra-train`). Unit tests: `r1_reasoning_test.py` = 18 tests.
- **Pod drivers:** `r1_pod_chains.sh` (chain B: restore from HF, prep, smoke, background sync, 3 stages, scoring evals val full / private / public, text evals + BERTScore, verified push to HF `checkpoints/chains_2026-10-19/`, self-stop; stage markers `done.B_*` synced), `r1_hf_sync.py` (push / pull `<prefix>/live`; patterns `train_log.jsonl epoch_*.pt best.pt done.B_* top3.txt *.log chains.status`), `r1_pod_run.sh setup|check_stop`, `r1_pod_public.sh`, `r1_pod_trainsplit.sh`.
- **Website:** `website/build_aug_extra_data.py` -> `aug_extra_data.js` (860 windows, every version + status + reason + Opus verdict), `website/assets/aug.js` (`createAugView({data, prefix, groups, writers, esc})`), `assets/site.css` (`#viewAugX` rules), `dataset.html` (tabs `teacherx`, `augx`; loaders `showTeacherX`, `showAugX`), `build_teacher_data.py` options `--folder --horizons --js --var --dataset-key --no-dataset-files`, `assets/pilot.js` (`fmtTte`).
- **Plans** (`~/.claude/plans/CCP based BADAS/`): `2026-10-15_Plan-Phase2-Three-Formats`, `2026-10-16_Plan-Night-Run-Phase2-Three-Formats`, `2026-10-17_Plan-Test-Private-{Reasoning,Addendum-Pass1-Only,Pass2-Anchored}`, `2026-10-18_Plan-P2V-Report-and-Experiment-Backlog`, `2026-10-19_Plan-Extra-TTE-Windows-Retrain{,-rev2}`, `2026-10-20_Plan-Seeds-Of-Todays-P2V`.
