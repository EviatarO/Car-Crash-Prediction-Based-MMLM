# Architecture

## Semantic-supervision design

**Training:**
```
video (16f) → BADAS ViT-L trunk (frozen in A0/B1, LoRA in A1/B)
                    ↓
             patch grid (2560, 1024)   [confirmed at runtime]
                    ↓
      ┌─────────────┴──────────────┐
crash head (pooler+classifier,   Predictor (ResamplerProjector,
reused, frozen; LoRA never       num_queries=8, hidden=256) → mean-pool
touches it - see constraints)    over queries → 1024→256→Dt
      ↓                                ↓
  2 logits → P(collision)        predicted semantic embedding
                                        ↓ (loss: cosine OR infonce, see below)
                              caption → frozen SigLIP text encoder → target embedding (Dt=768)

Total loss (stage B) = crash_loss (CE) + semantic_weight * semantic_loss
```
**Semantic loss, two variants** (B1 only supports both via `--loss`; Stage B's trainer
(`semsup_train.py`) currently only implements cosine — InfoNCE there needs real batching
first, which the wrapper doesn't do, see Constraints):
- `cosine` (original, **proven degenerate** 2026-07-25): `1 - cos(pred, target)`. Minimizer
  for a video-blind predictor is `target_mean/‖target_mean‖` — the real B1 run beat that
  floor by only 0.53% of the available range.
- `infonce` (added 2026-07-25, B1 only so far): in-batch contrastive, CLIP/SigLIP-style,
  learnable temperature (init 0.07). Sibling-TTE rows of the same `video_id` masked out of
  the negative set. The collapse solution above scores at chance under this loss instead of
  getting a free ride — verified via a synthetic proof, see EXPERIMENTS.md.
**Inference (final target):** `video → BADAS ViT-L (LoRA) → crash head → P(collision)`.
Semantic branch (Predictor + SigLIP) is fully discarded — zero added inference cost. Language
is **train-only privileged information**; the precise framing is *cross-modal distillation
under the LUPI regime* (see EXPERIMENTS.md literature check).

## Constraints / assumptions
- BADAS-Open's backbone is a plain **V-JEPA2 ViT-L**, loaded via `nexar-ai/BADAS-Open`'s
  official `badas_loader.py` (which pulls `nexar-ai/nexight`). It is a **gated HF repo** —
  `hf auth login` is required on every new pod.
- **LoRA target modules = `query,key,value`**, found only under
  `backbone.encoder.layer.{0-23}.attention.*` — confirmed zero overlap with the crash head
  (`pooler.*`, `classifier.*`), so LoRA structurally cannot touch it. Applied LoRA is
  2,801,664 / 334,355,842 params (0.84%).
  Note: the same substring also matches 12 V-JEPA2 predictor-layer attention blocks — still
  undecided whether to restrict to encoder-only (see DECISIONS.md).
- Frames on disk are **HiRes (1280×720)** and that is correct/intended: BADAS's own
  `AutoVideoProcessor` squash-resizes to 224×224 + ImageNet norm inside `preprocess_clip()`.
  Feeding pre-squashed 256×256 frames would cause a lossy double-resize. A0's validated 0.853
  used the HiRes→224 single-resize path.
- SigLIP text embedding dim **Dt = 768** (`google/siglip-base-patch16-224`). Its tokenizer
  needs both `sentencepiece` and `protobuf`.
- Patch-grid / Predictor dtype mismatch: BADAS may run fp16, Predictor is fp32. Cast with
  `.to(dtype=torch.float32)` at that boundary — differentiable, so the stage-B gradient still
  reaches the LoRA-unfrozen trunk.
- `ResamplerProjector` **is batch-safe** (`B = patches.shape[0]`, `batch_first=True`, queries
  expanded per-batch) — verified, since it had only ever been called with batch size 1 before
  B1's caching rewrite. Its self-attention block is now **conditionally built**
  (`use_selfattn = num_queries > 1`): at `num_queries=1` it's mathematically a no-op (softmax
  over one key), so it's skipped entirely rather than carrying ~1M dead params. Unaffected for
  any caller using `num_queries>1` (e.g. the unrelated e4 Stage B bridge, `num_queries=64`).
- **The semantic predictor is sized `num_queries=8, hidden_dim=256, ffn_mult=2`** (~1.25M
  params) as of 2026-07-25 — was `num_queries=1, hidden_dim=512` (~5.13M params, larger than
  the ~2.8M LoRA trunk it's meant to gently steer). Multi-token output is **mean-pooled**
  over the query dimension before comparison to the single SigLIP target
  (`predictor(x).mean(dim=1)`, not `.squeeze(1)` — that assumed `num_queries=1`).
  `semsup_train.py`'s Stage B still processes one clip at a time (`TrainableBadasWrapper`
  is batch-size-1 in practice, called per-example), so InfoNCE's in-batch negatives are only
  wired up in B1 so far, which caches features and batches them.
- **`val_ap`/retrieval are now computed per CLIP, not per row** (fixed 2026-07-25, T-3): the
  51-row val split is only 17 independent clips (2-3 correlated TTE-window rows each,
  constant label per clip); treating rows as independent inflated the metric and ranked
  checkpoints in the OPPOSITE order from test_AP. `evaluate_crash_ap` (A1/B) and
  `clip_level_retrieval_acc` (B1) both pool a clip's rows (mean, renormalize for embeddings)
  before scoring. Row-level metrics are still reported alongside for continuity, not removed.
- Full Nexar train pool = 1500 clips (750/750 balanced), but only 89 have local
  frames+reasoning+captions. Stage-0 target ~4.5k windows; 267 exist.
- The **reasoning-generation route** (ReverseBERT decoder) is a paused, unrelated thread.
- **SigLIP's tokenizer hard-truncates at 64 tokens** (`tok.model_max_length == 64`,
  `siglip_text_embed()`'s `max_length=64`) — discovered 2026-07-27 while reviewing new caption
  prompts. The incumbent 267 captions measure 12-24 tokens (0% truncated); a representative
  70-120-word caption in an earlier draft prompt measured **128 tokens, 50% discarded**, always
  losing the outcome clause since it was written last. Any new captioning prompt MUST target
  well under 64 tokens with the important content stated first, not last.
- **Positive and negative clips use different windowing conventions** (discovered 2026-07-27
  building `semsup_sample_clips.py`'s preflight): positives are pre-extracted at
  `TTE_0.5/TTE_1.0/TTE_1.5` (seconds before the real event); negatives have no event to count
  down to, so they're pre-extracted at `MID/MID-4/MID-8` (offsets from the clip midpoint)
  instead. A "3 buckets per class" sampling scheme must use these two different bucket sets,
  not one shared TTE axis. Every (video_id, bucket) pair needing a frame window that isn't
  already extracted on disk is unreachable — no raw video exists in this repo to extract more.

## Files that matter
| Path | Purpose |
|---|---|
| `student_training/scripts/semsup_common.py` | Shared: caption↔frames_dir resolution (now reads only `teacher_dataset_e3b.jsonl` by default, raises on conflict), `TrainableBadasWrapper` (LoRA-capable, gradient-preserving patch tap), frozen SigLIP loader, `dry_run_modules()` |
| `student_training/scripts/semsup_b1_probe.py` | Stage B1: predictor-only probe. Caches frozen features once, early stopping, top-3 ckpts, collapse control, `--loss {cosine,infonce}`, clip-level retrieval |
| `student_training/scripts/semsup_train.py` | Unified A1/B LoRA trainer (`--semantic-weight 0.0` = A1, `>0` = B). `--seed`, per-clip val AP, `epoch_metrics.jsonl`, streamed test-scoring, per-clip error handling. Keeps top-3 ckpts by val_ap and scores **each** on the 677-clip test set |
| `student_training/scripts/metrics_core.py` | Pure numpy/sklearn metrics — importable on a pod without matplotlib/seaborn/pandas |
| `student_training/scripts/evaluate_metrics.py` | Local graph/report pipeline; delegates formulas to `metrics_core` instead of duplicating them (fixed 2026-07-25 — the two had diverged) |
| `student_training/scripts/e4_stageA_badas_open_eval.py` | Reused as-is: `load_badas()`, `preprocess_clip()`, `load_manifest()`, `frame_paths_for()`, A0 scorer |
| `student_training/models/vjepa_reason.py` | `ResamplerProjector` — semantic predictor now `num_queries=8` (was 1); self-attention conditionally built |
| `student_training/configs/e4_stageA.yaml` | BADAS config: `hf_repo`, preprocessing, acceptance band, `gt_field=event_occurs`, `frame_filename_pattern` |
| `outputs/semantic_captions/summary.md` | **Running experiment status doc — update on every stage run** (project convention) |
| `outputs/semantic_captions/Caption_Train_All_Clips.jsonl` / `.xlsx` | 267-row caption dataset (89 clips × ≤3 TTE); force-added to git |
| `outputs/semantic_captions/b1_metrics.json` | B1 real-run metrics, full per-epoch history |
| `RUNPOD_SEMANTIC_SUPERVISION.txt` | Pod runbook, all 4 stages |
| `reports/_scripts/report_helpers.py` | Shared python-docx styling for progress reports (`new_doc`, `title_page`, `h`, `body`, `bullet`, `warn`, `add_table`, `fig`, `side_by_side`) |
| `reports/_scripts/_build_progress_report.py` | Overwritten per `/progress-report` run |
| `reports/figures/semsup_architecture_2026-07-21.png` | High-level train-vs-inference block diagram (generated by `_build_arch_diagram.py`) — predictor sizing in the diagram now stale vs. the 2026-07-25 code change, not yet regenerated |
| `reports/project_reviews/2026-07-25_project_review.md` | Full `/project-review` findings (12 sections, severity-ranked). Gitignored — not tracked unless `git add -f`'d |
| `prompts/PROMPT_SEMSUP_V2.py` | **NEW (2026-07-27).** Single captioning prompt, JSON output `{caption_neutral, risk_clause, verdict, confidence}`. Replaces an earlier two-prompt draft that failed the 64-token SigLIP limit |
| `student_training/scripts/semsup_sample_clips.py` | **NEW.** Preflight + balanced distinct-video sampler for the prompt bake-off. Fails loudly (non-zero exit) rather than silently shrinking the target `--n` |
| `student_training/scripts/semsup_caption_qa.py` | **NEW.** Gate 0: token compliance, banned-word, duplicate-sentence, verdict-leakage checks; builds `arm_a/b/c.jsonl`. Also runs as a legacy self-test directly on `Caption_Train_All_Clips.jsonl` |
| `student_training/scripts/semsup_caption_geometry.py` | **NEW.** Gate 1: SigLIP-embedding geometry (anisotropy, mean pairwise cosine, effective rank, NN purity, centroid separation) per arm — free, CPU-only |
| `student_training/scripts/semsup_promptbakeoff_report.py` | **NEW.** Gate 2 collation: per-arm exact binomial test vs chance, paired bootstrap between arms on shared val clips, mechanical decision-rule application, writes `summary.md` |
| `outputs/semantic_captions/_build_promptbakeoff_xlsx.py` | **NEW.** Reviewable spreadsheet for the raw bake-off captions, reusing `_build_caption_xlsx.py`'s styling plus new banned-word/over-token QA colors |

## APIs / functions — semantic-supervision
- `TrainableBadasWrapper(stagea_cfg, lora_target_modules=None|[...], lora_r, lora_alpha, lora_dropout)`
  → `.forward(frame_paths: list) -> (logits (1,2), patches (P,D))`. Patches are **not**
  detached when `lora_target_modules` is set.
- `load_training_examples(limit=0, require_frames=True, captions_path=None) -> list[dict]` —
  keys `video_id`, `tte`, `frames_dir`, `frame_paths`, `caption`, `label`. **2026-07-27:** new
  `captions_path` param overrides the default `Caption_Train_All_Clips.jsonl` (used for the
  prompt-bakeoff `arm_a/b/c.jsonl` files). If a row already carries an explicit `frames_dir`
  field, it's used as-is instead of going through `build_frames_dir_index()` — needed because
  a fresh distinct-video sample draws from teacher_labels generations outside the default
  index's coverage. Rows without `frames_dir` (the original file) resolve exactly as before —
  regression-checked to still return 267/267 with `captions_path=None`.
- `clip_level_split(examples, val_frac=0.2, seed=0) -> (train, val)` — splits by unique
  `video_id`, so no clip leaks across train/val.
- `load_siglip(model_id, device) -> (model, tokenizer)`
- `siglip_text_embed(texts, model, tokenizer, device) -> (B, Dt)` — L2-normalized; handles
  several transformers output shapes defensively.
- `dry_run_modules(config_path, out_path)` — dumps `named_modules()`, no training. Run before
  ever changing `--lora-target-modules`.
- `build_frames_dir_index(label_files=None) -> dict` — default `label_files=["teacher_dataset_e3b.jsonl"]`
  (was: glob all 28 files in `dataset/teacher_labels/`). Raises `ValueError` on a genuine
  `(video_id, tte)` conflict instead of silent last-writer-wins. `_norm_tte(tte) -> str`
  normalizes numeric TTE keys (`1` and `1.0` now collide correctly).
- `semsup_b1_probe.py`: `infonce_loss(pred, tgt, vids_batch, log_tau)` — in-batch contrastive
  loss with sibling-TTE masking. `clip_level_retrieval_acc(P, T, vids_list) -> float` — pools
  rows per `video_id` before retrieval; returns NaN (not a crash) if fewer than 2 unique
  clips. `evaluate(X, Y, vids)` now returns a 5-tuple: `(loss, mean_cosine,
  retrieval_top1_acc, retrieval_top1_acc_sibling_ok, retrieval_top1_acc_clip)`.
- `semsup_train.py`: `evaluate_crash_ap(badas, examples, device)` now aggregates per clip
  (mean of a clip's row scores) before computing AP — signature unchanged, behavior changed.
- `semsup_b1_probe.py` (2026-07-27): new `--captions PATH` CLI flag, threaded into
  `load_training_examples`. `clip_level_retrieval_detail(P, T, vids_list) -> (clip_ids sorted,
  hit list 0/1)` — new function `clip_level_retrieval_acc` now calls internally; also saved
  into `b1_metrics.json` as `val_clip_ids`/`val_clip_hits` so a downstream report can do a
  PAIRED comparison between two separately-trained arms (aggregate accuracy alone can't support
  resampling clips together across runs).
- `semsup_caption_qa.py`: `token_report`, `banned_word_report`, `duplicate_report`,
  `verdict_leakage_report` — each prints and returns a dict; `BANNED_RE` and `MAX_TOKENS` are
  importable constants (reused by `_build_promptbakeoff_xlsx.py` rather than duplicated).
- `semsup_caption_geometry.py`: `anisotropy`, `mean_pairwise_cosine`, `effective_rank`,
  `nn_purity`, `centroid_separation` — all pure numpy on an `(n, Dt)` embedding matrix.
- `semsup_promptbakeoff_report.py`: `vs_chance_binomial(arm) -> dict` (exact binomial test,
  `scipy.stats.binomtest`), `paired_bootstrap_diff(arm_x, arm_y, n_boot, seed) -> dict`
  (aligns by `clip_id`, not list position), `decide(...)` — mechanical application of the
  pre-written decision table in DECISIONS.md.
- `metrics_core.metrics_from_arrays(y_true, y_score, groups=None, threshold=0.5, ece_bins=10) -> dict`
  — full E3 metric table: confusion matrix, accuracy, precision, `recall_sensitivity_tpr`,
  `specificity_tnr`, f1, `f1_optimal`, `optimal_threshold`, ap, `auc_roc`, brier, ece,
  `per_tte_ap`. NaN-prone fields emit `None` (JSON-safe).
- `metrics_core.expected_calibration_error(y_true, y_score, n_bins=10) -> float`

**Required workaround inside `semsup_train.py`** (do not remove):
```python
badas.nn_model.create_or_update_model_card = lambda *a, **k: None
```
`peft`'s `save_pretrained()` builds a model card before writing weights and assumes
`base_model.config` is dict-like; BADAS's `ModelArgs` isn't, so every checkpoint save crashed.

## Tooling / meta (user-level, affects all projects)
| Path | Purpose |
|---|---|
| `~/.claude/skills/handoff/SKILL.md` | Writes `docs_agents/` cold-start briefing + the freshness token the PreCompact gate checks |
| `~/.claude/skills/project-review/SKILL.md` | **NEW.** `/project-review` — whole-project ML+code audit; asks for scope, docs-first gate, 2 parallel agents, 12-section report to `reports/project_reviews/`. Deliberately NOT named `code-review` (built-in command owns that) |
| `~/.claude/hooks/precompact_gate.py` | Blocks compaction when handoff docs are stale (once per session, fails open) |
| `~/.claude/hooks/session_reload_docs.py` | Re-injects `docs_agents/` after a compaction (12k char cap) |
| `.claude/commands/progress-report.md` | Project-level `/progress-report` command |

**Hook design constraint:** the `cwd` field in Claude Code hook payloads is the app's *launch*
directory (e.g. `C:\Users\eviatar.ohayon`), **not** the project folder. Both hooks therefore
locate the project via a session-keyed pointer file
`~/.claude/hooks/.handoff_location_<session_id>.json` written by `/handoff`. Do not "simplify"
them back to trusting `cwd`.
