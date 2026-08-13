# Architecture

## Current design (e4_vjepa_reason)
Student = **BADAS-Open** (V-JEPA2 ViT-L backbone), LoRA-tuned on its trunk (`query,key,value`
projections, 72 encoder adapters by default). A **crash head** on top produces the collision
score used for AP/AUC. Optionally, a **train-only semantic-alignment branch** runs in parallel:
a small Predictor consumes the same visual features and is pulled toward a frozen SigLIP text
encoder's embedding of a teacher-written caption via either cosine or InfoNCE loss. The Predictor
and SigLIP text encoder are **fully discarded at inference** — the deployed model is
vision-only, same cost/latency as the crash-only variant. Block-by-block reference (shapes,
frozen status, equations) is in `docs_agents/ARCHITECTURE_BLOCKS.md`, matching the diagram at
`reports/figures/semsup_architecture_2026-07-21.png` (note: that diagram is stale in two spots —
it shows semantic weight `0.3×`, the actual default is `0.05`; and it labels the loss "meaning
match" which describes cosine, not the current InfoNCE default).

## Important constraints / invariants
- Inference path must never touch the Predictor or SigLIP text encoder — enforced by construction
  (they're just not called in `forward_clip`'s inference branch), not by a runtime flag.
- A1_1761's exact recipe (`query,key,value` legacy target modules, constant LR, dropout 0.05,
  seed=0, the 1,761-window enriched pool) is the **reference control** for every semantic-aux
  comparison. Any arm meant to isolate the semantic-loss effect must match this recipe exactly
  except for `--semantic-weight`/`--semantic-loss` — confirmed by checking the printed
  trainable-param count and adapters-by-stack breakdown at construction time.
- Any new caption corpus must pass the label-leakage gate (TF-IDF+LogisticRegression,
  GroupKFold-5 by `video_id`, target AUC<0.75) before being trusted for semantic-supervision
  training — a leaking corpus makes the auxiliary loss a redundant/noisy copy of the label
  rather than a semantic signal, and confounds any A-vs-B comparison.
- Training and captioning I/O must go through the concurrent pipelines below — do not revert to
  serial `for` loops over frame paths or API calls; the whole trunk is I/O/latency-bound, not
  compute-bound, at this problem scale.

### Pre-existing constraints (still true, carried forward)
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
- **Raw Nexar MP4s DO exist locally after all** (2026-07-28), in a sibling project folder not
  previously checked — this reverses the previous constraint for NEW extractions (the 500-clip
  bake-off set was extracted locally, no pod trip needed). The `dataset/train/`-resident
  extracted-frames constraint above still holds for anything reading pre-existing folders.
- **Positive windows use `train.csv`'s `time_of_event`** to place the TTE_0.5/1.0/1.5 windows;
  negative windows use the clip's own midpoint (MID/MID-4/MID-8) since no event exists to
  anchor to — same convention as the incumbent 267-caption set, now also used by
  `semsup_extract_promptbakeoff_frames.py` for the 500 new clips.
- **OpenRouter `preview`-tagged model aliases are not stable snapshots** (see PROJECT_STATE.md's
  gotchas) — `google/gemini-3.1-pro-preview`'s behavior changed between the historical v6
  baseline and a same-day rerun. Any comparison against a "current teacher" baseline must use
  a fresh same-day run, never a stored historical number, for any `preview`-tagged model.
- **The under-calling / recall problem is (evidence suggests) a calibration issue, not a
  perception issue.** Three different teacher candidates (Qwen3.7 Flash, GPT-5.6 Luna Pro,
  Qwen3-VL-235B-Thinking), two different prompts (v6 unmodified, and a from-scratch prompt with
  an explicit anti-under-calling instruction), all produce the identical confusion matrix on
  the 18-clip val set: TP=2, FP=0, TN=9, FN=7. On clip `00687`, the model's own caption
  correctly describes the hazard ("gray SUV merging into ego lane") while its verdict still says
  NO and its risk_clause says "normal merging traffic" — the perception is right, the decision
  layer discounts it. **Extended through 6 more prompt versions (V5-V9) and confirmed
  statistically inconclusive at this sample size**: every accuracy delta across all rounds sat
  inside every other's 95% CI (McNemar p>=0.125 throughout) - n=18 cannot distinguish these
  prompts. One robust result did survive: unmodified v6 on `google/gemini-3.6-flash` scored
  best on both axes (72.2% acc, 0 FP, best caption fidelity) while the *same prompt* on
  Qwen3-VL-235B-Thinking collapsed to 0/18 YES predictions - model-family x prompt interaction
  dominates prompt wording. This 18-clip screening approach is now superseded by inference-only
  failure-mining on the real ~4,500-window train pool (see PROJECT_STATE.md's
  "train4500-inference pipeline"), which has real statistical power and measures the frozen
  scorer directly instead of a caption-quality proxy.
- **Negative-window convention changed: `MID` moved from offset 0.0 (exact clip midpoint) to
  −10.0 (renamed `MID-10`)** — discovered 2026-08-01 via real A0 scoring on `train4500`'s
  chunk 0: the exact-midpoint window produced 42.8% error, 100% false positives, at 0.99+
  confidence, isolated to that one bucket (`build_train4500_manifest.py`'s `NEG_BUCKETS`,
  `semsup_extract_promptbakeoff_frames.py`'s matching constant). Root cause: `train.csv`'s
  label is clip-level, but a naturalistic ~40s clip's literal midpoint can look genuinely risky
  without ever becoming a collision — a real hard-negative the label can't express. Falls back
  to the pre-existing `T_FLOOR=2.0` mechanism for short clips (no new fallback logic needed).
  Any script that hardcodes the string `"MID"` as a bucket label is now wrong — check
  `build_caption_monitor.py`'s `_resolve_caption_bucket()` for the pattern (legacy
  `TN_MIDPOINT` captions at offset ~0 are deliberately left unresolved, not remapped to
  `MID-10`, since it's a genuinely different window).
- **`teacher_distillation/scripts/teacher_bakeoff.py` and `Teacher_dataset_distill_v11.py`
  both have a pre-existing broken top-level import** (`prompts/PROMPT_G2.py` and
  `prompts/templates.py` respectively no longer exist at those paths - the prompts they held
  now live under `prompts/old prompts/`, from an earlier reorganization). Discovered
  2026-07-28/29, unrelated to and not fixed by this thread's work. Any new script needing
  their helpers (image encoding, retry/backoff, JSON extraction) should copy the needed
  functions rather than import the module, until/unless someone deliberately fixes the
  reorg fallout - `semsup_caption_promptbakeoff.py` and `semsup_v6_control_rerun.py` both do
  this already.

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
**Semantic loss, two variants** (both now supported in B1 AND Stage B; the batching
objection below was resolved 2026-08-06 by precomputing a frozen caption bank —
`build_caption_bank`/`infonce_from_bank` — so batch-size-1 Stage B still gets N negatives):
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

## Concurrent I/O pipeline (training)
Root cause: reading+decoding 16 JPEGs per window costs ~1.17s (670ms raw read + 503ms
decode/resize) vs. an unmeasurably fast GPU forward pass — the trainer was I/O-bound, not
GPU-bound (verified via direct phase-by-phase profiling on the pod, not inferred from GPU
utilization graphs, which only showed the symptom: 24-33% utilization).

- `TrainableBadasWrapper.forward(frame_paths)` → now delegates to `forward_clip(clip)` after
  preprocessing (unchanged behavior/signature for any existing caller).
- `TrainableBadasWrapper.forward_clip(self, clip)` → runs the model on an already-decoded
  tensor; this is the actual GPU-compute entry point, separated out so it can be called from a
  pipeline that overlaps decode (CPU/IO) of window N+1 with compute (GPU) of window N.
- `TrainableBadasWrapper.prefetch_clips(self, examples, num_workers=8, prefetch=16,
  key="frame_paths")` → generator. `ThreadPoolExecutor`-based; maintains a futures dict keyed
  by index, keeps `prefetch` items in flight ahead of the current position, yields
  `(i, ex, clip_or_None, error_or_None)` strictly in submission order via `.pop(i).result()`.
  Works despite the GIL because file I/O and PIL JPEG decompression release it during their C-level
  work. Per-item errors are caught *inside* the worker function and yielded as a 4th-element
  exception rather than raised — raising inside a generator would kill the whole generator
  mid-stream; catching and yielding a sentinel lets iteration continue past a single bad clip.
  `num_workers<=0` falls back to a fully serial path (debugging escape hatch).
- Used in `semsup_train.py` for: the training loop, the merged `evaluate_val()` (combines what
  used to be two separate passes — `evaluate_crash_ap` + `evaluate_val_loss` — into one), and
  `score_checkpoint()` (test-set scoring; `records_wp` is precomputed once with a `frame_paths`
  key added via `frame_paths_for()` so the default `key="frame_paths"` works unchanged).
- Verified: isolated benchmark on the pod (workers=0 vs 8 vs 16) → 5.3× at 8 workers; live
  resumed A1-v2 run → 6.3× (epoch 1-2 pre-fix ~97 min avg/epoch → epoch 3 post-fix 15.5 min),
  GPU utilization 24-33% → 83-94%.

## Concurrent captioning pipeline
Same diagnosis applied to `semsup_caption_promptbakeoff.py`'s OpenRouter calls: latency-bound
(network round-trip per clip), not CPU-bound.
- `_fetch_one(row)` — worker-thread function, isolates exactly the network-bound part (missing-
  frame check, prompt build, image encode, `_call_model()` call). Returns a 4-tuple
  `(row, raw_text_or_None, error_or_None, usage_or_None)` — all 4 internal early-return points
  (including the 3 failure paths) were made consistent to this arity after an earlier bug where
  only the success path returned 4 elements.
- Main loop: `futures = [pool.submit(_fetch_one, row) for row in pending]`, consumed via
  `for idx, fut in enumerate(as_completed(futures), start=1)`. All downstream parsing/
  validation/row-building/file-writing stays serial in the main thread — only the network call
  itself is parallelized.
- New `--concurrency` CLI arg (default 4; used at 16 for the real V12 recaption run).
- **Usage/cost logging** (previously the `usage` field from every OpenRouter response was
  silently discarded): `usage_path = out_path.with_suffix(out_path.suffix + ".usage.jsonl")`,
  written alongside the main output; running totals accumulated per call
  (`total_prompt_tok`, `total_completion_tok`, `total_cost`); `_cost_str()` closure formats
  `"tok=N"` or `"tok=N cost=$X.XXX"` depending on whether the API returned a `cost` field.
  Printed at the 25-row progress cadence and in the final `DONE.` summary. This is now the
  source of truth for captioning cost — not the older, never-fully-verified doc estimates.
- Verified: serial 11.8s/clip → concurrency=16 ~1s/clip (~12×); cost-neutral (OpenRouter bills
  per-token, not per-request or wall-clock).

## LoRA target-module selection
`--lora-target-modules` accepts either the legacy comma-separated substring list (e.g.
`query,key,value` — matches 108 modules: 72 encoder + 36 V-JEPA2-predictor-stack) or a
`re:<regex>` prefix passed through untouched to `peft.LoraConfig(target_modules=<regex>)` (e.g.
`re:backbone\.encoder\.layer\.\d+\.attention\.(query|key|value)` — matches 72, encoder only).
`TrainableBadasWrapper`'s LoRA construction reports **adapters by stack**
(`{'backbone.encoder': N, 'backbone.predictor': M}`, via `Counter` on `lora_A.default` module
names) at construction time, with a printed NOTE if any adapter lands on `backbone.predictor` —
this makes the true scope of a run visible in the log without having to inspect the config file,
which is what caught the discrepancy between A1_1761 (108 modules, encoder+predictor) and
A1-v2's intended encoder-only design.

## V12 neutral captioning prompt (`prompts/PROMPT_SEMSUP_V12_NEUTRAL.py`)
Register-neutral redesign of the captioning prompt, built to close the label-leak found by the
`/project-review` audit. `build_prompt()` takes **no arguments** — there is no GT-informed vs.
blind branch at all (V10's leak came precisely from that branch: positives got a "this DOES end
in collision" framing, negatives didn't). Structure: `NEUTRALITY_BLOCK`, `ROLE_AUDIENCE`,
`INPUT_BLOCK`, `STEP123_BLOCK` (reused verbatim from V10), `STEP4_BLOCK` (primary-agent
identification), `GAP_TREND_BLOCK` (closed 4-way vocabulary:
`decreasing/increasing/constant/none_visible`, replacing V10's free-text `closing_dynamic`),
`CAPTION_RULES` (symmetric bans — outcome words, alarm words, reassurance words — applied
identically regardless of class), `_SCHEMA`. Drops `verdict`/`risk_score`/`confidence`/
`risk_clause` entirely (the model only describes, never judges). Field renames for register
hygiene (not a fabrication fix): `hazard_agent→primary_agent`, `hazard_motion→agent_motion`,
`hazard_position→agent_position`, `mechanism_visible→agent_visible`.

Wired into `semsup_caption_promptbakeoff.py` via `_v12_builder(gt_mode=None, is_positive=None)`
(adapter matching `TEMPLATE_BUILDERS`'s calling convention despite `build_prompt()` itself taking
no args), `V12_REQUIRED` tuple, `V12_GAP_TREND_VALUES` constant, a `validate_parsed()` branch
(hard-fails on missing keys / invalid `agent_visible` / invalid `gap_trend`; soft-notes if the
`gap_trend` word isn't found verbatim in the caption text), `prompt_tokens["v12"]=1050` for
`--dry-run` estimates, and a dedicated output-row writer (separate from v10/v10q since field
names differ) emitting `primary_agent`/`agent_motion`/`agent_position`/`gap_trend`/
`evidence_frames`/`agent_visible`.

Raw V12 output schema uses `caption_neutral` + `event_occurs` (0/1); `load_training_examples()`
expects `caption` + `gt_verdict` (YES/NO string) — a derived `_fortrain.jsonl` file with both
aliases added is what actually gets used for training/comparison scripts.

## Files that matter

| Path | Purpose |
|---|---|
| `student_training/scripts/semsup_train.py` | Main trainer. Crash-only or crash+semantic (cosine/InfoNCE) LoRA fine-tuning of BADAS-Open. Cosine/constant LR schedule, checkpointing, val/test scoring, all via the concurrent prefetch pipeline. |
| `student_training/scripts/semsup_common.py` | `TrainableBadasWrapper` — model construction, LoRA wiring (legacy list or regex target modules), `forward`/`forward_clip`/`prefetch_clips`. |
| `student_training/scripts/semsup_caption_promptbakeoff.py` | Captioning runner against OpenRouter — all prompt versions (v2-v12), concurrent fetch, usage/cost logging, validation per prompt family. |
| `prompts/PROMPT_SEMSUP_V12_NEUTRAL.py` | The V12 register-neutral prompt (no GT/blind branch). |
| `student_training/scripts/build_pool_from_manifest.py` | Wraps a Stage-A-scorer-schema manifest into the caption-training schema with a placeholder-caption tripwire, for crash-only (no semantic loss) runs against a manifest that has no captions yet (e.g. the full 4,446-window pool). |
| `student_training/scripts/sample_val_check_clips.py` | Draws a balanced, distinct-clip sample from a manifest excluding a given val manifest's video_ids — used to extend the n=18 leakage-judge check to n=100. |
| `student_training/scripts/plot_semsup_curves.py` | Reads `epoch_metrics.jsonl` (current trainer's schema), plots loss/val_AP/LR/train-val-gap curves with the selected epoch starred. (Distinct from the older, incompatible `plot_training_curves.py` built for the superseded InternVL3.5 pipeline — do not confuse the two.) |
| `teacher_distillation/scripts/score_val18_neutral.py` | Scores V12 vs V10 on the 18-clip val set (grounding/neutrality via calibrated `score_blob()`) and runs the leakage judge; writes Excel/summary.md. Contains `binom_ci()`. |
| `teacher_distillation/scripts/leakage_judge_100.py` | Combines 18 val + 82 sampled clips into n=100, runs the leakage judge with numeric IDs (not letter-capped), computes exact one-sided binomial p-value/CI (no scipy dependency, `math.comb` summation). |
| `docs_agents/ARCHITECTURE_BLOCKS.md` | Block-by-block reference (shapes/equations/frozen-status) for the architecture diagram. |

## APIs / functions (new or changed this segment, signatures only)

```python
# semsup_common.py
TrainableBadasWrapper.forward(self, frame_paths) -> ...          # unchanged signature, now delegates
TrainableBadasWrapper.forward_clip(self, clip) -> ...             # NEW: compute on pre-decoded tensor
TrainableBadasWrapper.prefetch_clips(self, examples, num_workers=8, prefetch=16, key="frame_paths")
    -> Iterator[tuple[int, dict, Tensor|None, Exception|None]]     # NEW: ordered concurrent prefetch

# semsup_train.py
evaluate_val(...) -> dict            # NEW: merged evaluate_crash_ap + evaluate_val_loss, single pass
score_checkpoint(...)                # rewired to use prefetch_clips + records_wp precompute
# new CLI args: --prefetch-workers (default 8), --prefetch-depth (default 16),
#   --lr-schedule {constant,cosine} (default constant), --warmup-frac (default 0.05),
#   --lora-dropout (default 0.05, now exposed)

# semsup_caption_promptbakeoff.py
_fetch_one(row) -> tuple[dict, str|None, Exception|None, dict|None]   # NEW: worker-thread network call
_v12_builder(gt_mode=None, is_positive=None) -> str                    # NEW: V12 adapter for TEMPLATE_BUILDERS
# new CLI arg: --concurrency (default 4)
# new output: <out_path>.usage.jsonl (per-call token/cost sidecar)

# prompts/PROMPT_SEMSUP_V12_NEUTRAL.py
build_prompt() -> str                 # NEW: no gt_mode/is_positive args, single neutral prompt
```

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

## APIs / functions — semantic-supervision (pre-existing, carried forward)
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
