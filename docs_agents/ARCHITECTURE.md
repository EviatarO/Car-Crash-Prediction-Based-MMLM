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

- **The three caption files use incompatible field conventions — join them ONLY on
  `frames_dir`.** `Caption_Train4500_Failures_587.jsonl` populates `horizon_label` and leaves
  `requested_time_to_event` null; `Caption_V12_Neutral_1761_fortrain.jsonl` does the reverse;
  `Caption_Train4500_Mixed_1761.jsonl` (V10) matches the failures convention. Joining on
  `(video_id, t_seconds)` additionally collides because one clip contributes up to 3 windows.
  `frames_dir` is unique per window and present in all of them.
- **A clip is not a window.** 1,761 windows come from 1,107 unique clips (578 clips give 1
  window, 404 give 2, 125 give 3). `clip_level_split` partitions by `video_id`, so 20% of
  clips (221) yields 348 windows, not 352. Any per-window 80/20 assumption is wrong and leaks.
- **V10 and V12 have different schemas.** V10 blind-mode rows carry `verdict`, `confidence`,
  `risk_score`; V10 gt-mode rows do not (the teacher was told the label); both carry
  `mechanism_visible`. **V12 dropped `verdict`/`risk_score`/`confidence`/`risk_clause`
  entirely**, so no teacher-prediction-vs-label check is possible on V12 alone — it must read
  the raw V10 files under `outputs/semantic_captions/failures587/`.
- **Conditional-formatting fills in `openpyxl` render from `bgColor`, not `fgColor`** — the
  reverse of a normal cell fill. Using `fgColor` in a `FormulaRule`/`CellIsRule` produces a
  rule that matches correctly but paints nothing in Excel.

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
- Frames on disk are **HiRes (1280×720)**. **Corrected 2026-09-14 (measured):** the V-JEPA2
  `AutoVideoProcessor` does NOT squash here — it resizes the shortest edge to 292 and
  **center-crops 256×256**, so every run to date (A0 0.853, A1 0.900, all B arms, SemTest-200,
  a1fail321) saw only source **x 321–953, y 42–674 (~49% of width)**. The earlier "squash to
  224×224" note described BADAS's released inference code, which this repo never implemented.
  `preprocess_clip(mode=...)` / `--preprocess {crop, compress256}` now makes it explicit; `crop`
  stays the default and is byte-identical to the historical path. `compress256` resizes the full
  frame to 256×256 (one token = 80×45 source px). Decision pending: AA.0 v2 plan.
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

## Crash-head unfreezing (`--unfreeze-head`, 2026-08-26) — new capability, confounded so far

Added specifically to test whether the frozen crash head is what stops any arm from
recalibrating after LoRA moves the trunk's features (see PROJECT_STATE.md's calibration
finding: every 1,761-pool arm's AP/AUC/class-separation is flat, but each arm's *optimal
decision threshold* drifts wildly — 0.812 for A1, 0.173 for B-v3 — consistent with a frozen
head unable to follow a shifted feature distribution, per Kumar et al. ICLR 2022).
`TrainableBadasWrapper.__init__` gains `unfreeze_module_substrings: list|None` — after LoRA
wrapping, sets `requires_grad_(True)` on every param whose name contains `temporal_processor`
or `classifier` (substring match survives peft's `base_model.model.` name prefixing). Exposes
`head_params`/`head_param_names`, and two new methods: `head_state_dict()` (returns just the
unfrozen head tensors, keyed by their peft-prefixed names — needed because peft's
`save_pretrained()` only ever persists the LoRA delta) and `load_head_state(path)` (loads them
back with `strict=False`). `semsup_train.py` wires this through `--unfreeze-head` +
`--head-lr-mult` (default 0.1): head params get their OWN optimizer param group at
`lr * head_lr_mult` (single flat `AdamW(trainable, lr=...)` became `AdamW(param_groups)`),
`_clip_grads()` gained a third clip budget for the head when `--clip-grad-per-group` is set,
and every epoch's checkpoint dir now also writes `head_state.pt` alongside `lora_adapter/`.

**⚠️ Confirmed NOT sufficient as configured**: at `--head-lr-mult 0.1` with the trunk's cosine
LR schedule applied to every param group uniformly (`LambdaLR`'s single `lr_lambda` scales all
groups by the same factor, so the head's already-10×-smaller LR also decays toward 0 by the
final epoch), 200 optimizer steps move the head's own weights by <0.05% relative magnitude —
see PROJECT_STATE.md's SemTest-200 section. A future run testing this hypothesis for real
needs a materially higher `--head-lr-mult` and/or a head-specific (non-cosine-decayed) schedule.

Two more additions for the SemTest-200 experiment, both in `semsup_train.py`:
- `--val-video-ids <file>`: newline-separated video_ids, overrides `clip_level_split` entirely
  with an explicit train/val partition. Needed because `clip_level_split` is neither
  label-stratified nor TTE-uniform, and SemTest-200's val set is deliberately stratified.
- `--dump-val-scores`: writes `val_scores_ep{NN}.jsonl` every epoch (per-window
  `{video_id, frames_dir, tte, label, score}`) inside `evaluate_val()`, which previously
  discarded per-window scores immediately after computing the clip-level AP. This is what makes
  per-epoch score-distribution/PR-curve analysis possible; off by default (extra I/O).

New script `score_semtest.py` (copied from `score_arms_on_pool1761.py`, generalized): scores an
arbitrary checkpoint on an arbitrary-size pool (drops the hardcoded 1,761-row expectation), and
adds `--head-state <path>` to load an `--unfreeze-head` run's head weights before scoring — a
checkpoint trained with `--unfreeze-head` is **not reproducible** without this, since the
crash-relevant weights live outside the LoRA adapter peft saves.

## SemTest-200 clip selection (`select_semtest200_recovery.py`, new)

Builds a small (200-window), one-window-per-video, deliberately-adversarial-to-A0 pool via a
**3-tier priority fill**, computed from a fresh A0 re-score of the full 4,446-window manifest
(`score_arms_on_pool1761.py`, reused unchanged — it only warns, not fails, when the pool isn't
1,761 rows) plus `dataset/train.xlsx`'s response-time column (col E = `time_of_event −
time_of_alert`, seconds; **null for every negative row**, so the RT-eligibility filter only
applies to positives: `response_time > TTE`).

Positive side (3 tiers, filled **tier-globally** across all 3 TTE buckets at once — filling
bucket-by-bucket starved TTE_1.5 because a single video can supply windows to more than one
TTE bucket and a naive per-bucket loop lets an earlier bucket consume shared videos):
1. **FN near-boundary** — GT=YES, RT-eligible, score∈[0.3,0.5) — take ALL of them.
2. **TP fill** — GT=YES, RT-eligible, score∈[0.5, `--tp-fill-max`) — lowest-score-first. The
   cap (default 0.85) is load-bearing: without it, the 639/491/290-clip mass at score≥0.85 per
   bucket absorbs every remaining slot and tier 3 can never fire.
3. **FN wide** — GT=YES, RT-eligible, score<0.3 — highest-score-first (closest to the tier-2
   boundary), only if tiers 1+2 together still can't fill quota.

Negative side (2 tiers, **100% FP by design, zero TN ever selected** — this is deliberate, per
the user's spec, not a bug): FP near-boundary [0.5,0.7) (all of them), then FP fill [0.7,1.0)
lowest-first.

`--exclude-frames-dir <file>`: excludes specific windows entirely (e.g. clips that failed a
caption-QC pass) and re-runs the same tier logic to backfill — used iteratively during
SemTest-200's caption-QC rounds. Val split: a fixed, deterministic 40-clip (20 TP-side/20
FP-side, evenly-strided by score-rank within each bucket) stratified split, written as `split`
in the output and as `val_vids.txt` for `--val-video-ids`.

`make_semtest200_shuffled.py` (new, small): the content-vs-presence control — permutes a
caption corpus's `caption` field WITHIN class (YES↔YES, NO↔NO) via a derangement (no row keeps
its own caption), seeded, so class label is preserved but content is fully scrambled.

## `PROMPT_SEMSUP_V13_CAUSAL.py` (new prompt, 2026-08-27) — causal-cue captioning

Same anti-leak machinery as V12 (`build_prompt()` takes no args, no GT/blind branch, closed
gap_trend vocabulary, symmetric outcome/alarm/reassurance/time bans) plus 5 new closed-vocab
fields targeting information NOT trivially recoverable from raw pixels: `lead_vehicle_lighting`
(brake_lights_on/indicator_left/indicator_right/**flashers_on**/none_visible — NOT
`hazards_on`: "hazard" is itself a banned outcome word, so an enum value the caption is
forbidden to say would silently never reach the SigLIP target), `ego_maneuver`, `road_geometry`,
`signal_state`, `occluded_or_peripheral`. Colour is banned from `caption_neutral`.
`caption_neutral` must be **42–52 words** (a floor as well as a ceiling — see PROJECT_STATE.md
for why a ceiling-only rule under-filled) and verbalize every populated field, not just
mention one. `validate_parsed()`'s new `v13` branch checks (all soft NOTEs, not hard failures):
closed-vocab membership per field, gap_trend word present verbatim, colour-word absence
(word-boundary regex — a naive substring scan false-positives on "tan" inside "dis**tan**ce"),
word-count against the 42–52 band, and per-field verbalization via a `_COVERAGE` keyword dict.

**⚠️ Known failure mode, not yet fixed**: this prompt's one worked example caused 96.9% of the
full 4,446-window run's captions to open with one of 3 near-identical phrases — see
PROJECT_STATE.md/EXPERIMENTS.md. A future prompt like this needs either no single canonical
worked example, or several structurally different ones, plus an explicit instruction against
copying the example's literal opening.

`semsup_caption_promptbakeoff.py` additions supporting this: `--provider-order <slug,...>`
(passes `extra_body={"provider": {"order": [...], "allow_fallbacks": False}}` to
`client.chat.completions.create` — pins a specific OpenRouter provider, since different
providers serving the identical model slug can be priced 2×+ apart, e.g. Vertex's 75%-off vs
AI Studio's 50%-off on `gemini-3.7-flash`; `allow_fallbacks=False` turns a routing fallback
into a loud failure instead of a silent overpay), `--token-cap <N>` (tokenizes
`caption_neutral` with the SigLIP tokenizer post-parse, stamps `caption_token_len`, reports —
does not enforce — rows exceeding the cap), and `DEFAULT_MODEL` fixed to
`"google/gemini-3.7-flash"` (was a stale `"google/gemini-3.1-pro-preview"` that got silently
hit for real this session — always pass `--model` explicitly and verify it printed correctly).

## Head-LR schedule fix + caption-bank widening (`semsup_train.py`, 2026-08-29)

`--head-lr-schedule {cosine,constant}` (default `cosine` = old behavior; `constant` keeps the
head's LR flat after warmup instead of decaying it alongside the trunk's shared cosine
schedule) — fixes the SemTest-200-v1 confound where `--unfreeze-head`'s already-small head LR
(0.1× the trunk's) was ALSO decayed by the trunk's cosine schedule, netting <0.05% relative
head movement over 200 steps. `head_lr` is now logged per epoch in `epoch_metrics.jsonl` as an
audit trail for this.

`--bank-captions <corpus>` widens the InfoNCE **train** negative bank with extra distractors
from a wider corpus while preserving each anchor's own `_bank_idx` position — must append after
the anchor's own-caption block, never replace it (replacing would break the anchor's own
positive-pair index). Used in the A1-failure-recovery run (below) to bank each arm against its
own full 1,761-row corpus; the shuffled arm banks against a freshly-shuffled 1,761 corpus, not
the unshuffled one — banking against the wrong corpus would silently break the
content-vs-presence control.

Also fixed (pre-existing, not introduced this session, found while running these tools):
unescaped `%` in `semsup_train.py`'s argparse help strings — adjacent string-literal
concatenation produced runtime content like `"...0.53%" + "of..."`, which argparse's own
`%`-style formatting then crashed on, breaking `--help` entirely. All now `%%`-escaped.

## A1-failure-recovery — 4-arm fine-tune starting from A1's own failures (2026-08-29)

Tests whether semantic supervision can recover the specific clips A1 (the current champion,
crash-only LoRA) gets wrong, and whether trying damages A1's 0.900 test AP. Design point
deliberately different from SemTest-200-v2 (above/below): starts from A1's own converged
weights and a **frozen** head (isolates "does semantic supervision damage an already-correct,
calibrated head" from SemTest-200-v2's "does an open head change the calibration story").

**Pool**: all 321 windows (240 unique videos) A1 scores wrong at threshold 0.5, mined from the
1,761-pool. A1's own AUC on this pool is exactly 0.0 **by construction** (every row starts on
the wrong side of the boundary) — expected, not a bug, and must be stated whenever this pool's
in-pool numbers are read. Split 260 train / 61 val by `video_id` (seed 0). Selection script:
`select_a1fail321.py`, writes `outputs/a1fail321/selection_a1fail321.jsonl` + per-arm caption
files (321 rows each, joined from the existing 1,761-pool V10/V12 corpora plus 72 freshly-
captioned clips where needed).

**4 arms**, all initialized from A1's own LoRA weights
(`/workspace/semsup/a1_1761/epoch_04/lora_adapter`, r=16/α=32/dropout=0.05, config verified via
`adapter_config.json` before loading), head frozen, predictor warm-started from B-v3's B1
checkpoint (`/workspace/semsup/b1_v2_100pct/predictor_b1.pt`, shared across all 3 semantic arms
to hold initialization constant and vary only the caption file): `a1cont` (crash-only control,
`--semantic-weight 0.0`), `v10` (leaky captions), `v12` (clean captions), `v12shuf` (v12
captions shuffled within class — content-vs-presence control, this project's cleanest B-shuffle
result to date). Config: `--lr 2e-5` (5× below A1's own 1e-4 — refining, not retraining from
scratch), `--lr-schedule cosine --warmup-frac 0.1`, `--epochs 10 --keep-top-k 10
--semantic-weight 0.2` (3 semantic arms), `--bank-captions` = each arm's own full 1,761-row
corpus. Driver `run_a1fail321_4arms.sh` runs the 4 arms strictly sequentially (concurrent BADAS
loads can crash each other — pre-existing documented gotcha).

**Results** (see EXPERIMENTS.md for the full numbers/tables):
1. In-pool val (61 clips): all 4 arms produce **bit-identical** per-clip predictions
   (fixed_FP=39, fixed_FN=0, still_wrong=22, acc@0.5=0.6393) whether there's no semantic branch,
   real captions, or scrambled captions. AP/AUC vary only by ~0.02 noise at this n.
2. Predictor health: v10/v12 retrieval@1 peaks 35-44% vs a 2.1% collapse control; v12shuf sits
   at ~0% for the entire run — cleanest real-vs-scrambled separation this project has produced,
   the opposite of the earlier SemTest-200 (pre-A1fail321) result where the predictor was
   collapsed at chance for every arm including real captions. Attributed to this run's wider
   InfoNCE bank (`--bank-captions`, full 1,761 rows vs 160 train-split captions before) plus the
   B-v3-B1 warm start.
3. Test set (677 clips), via `score_checkpoints_on_test.py` (new — see Files table): A1
   reproduced at 0.8995/0.9034 (matches its documented 0.900/0.904 to 3 decimals, validating the
   scorer), v12 epoch-10 at 0.8972/0.9027 — flat within noise. `acc@0.5`'s apparent +2.65pp for
   v12 is a calibration artifact (v12's mean test score sits at 0.488 vs A1's 0.660 — the whole
   distribution shifted down, landing near 0.5 by coincidence); at each arm's own optimal
   threshold the gap collapses to +0.004.

**Mechanism**: `grad_cos_mean` (crash-loss vs semantic-loss gradient cosine on shared LoRA
params, via the existing `--grad-cosine-every 8`) sits at −0.04 to +0.05, sign-flipping epoch to
epoch, in all 3 semantic arms — 10-100× above the pure-random-orthogonality floor for a ~2.8M-
param space (not literally independent) but far below what conflict would look like
(persistently negative cosine, `frac_neg`→1.0). Reading: the two objectives want mildly
overlapping but mostly orthogonal features — captions are a lossy function of the same 16 frames
the student already sees, so semantic supervision was never adding new information, only a
reorganization pressure the frozen crash head's fixed linear readout is largely blind to.

`score_checkpoints_on_test.py` (new script): loads BADAS once, swaps LoRA adapters between
checkpoints for speed. Uses `softmax(logits/temperature)[0,1]` (`--temperature`, default 1.0
— no `/2.0`), matching `semsup_train.py`'s own scorer, unlike `e4_stageA_badas_open_eval.py`'s
published A0-scorer convention (T=2.0) — confirmed via direct comparison (§ below) that
temperature does not affect AP/AUC or the confusion matrix at threshold 0.5 (dividing logits
by a constant is monotone, preserving the 0.5 crossing); it DOES move Brier/ECE, so never
compare calibration metrics across two runs at different `--temperature` (project review
2026-09-06 §4.5 — comparing A0's published T=2 Brier/ECE against every other arm's T=1 values
inverts the calibration ranking, even though AP/AUC/CM are bit-identical).

**⚠️ Corrected 2026-09-08 — the `/2.0` divisor does NOT explain the A1-vs-A1 scoring gap.**
An earlier version of this doc (and of `EXPERIMENTS.md`) attributed the difference between
`a1_1761/test_results_ep04.jsonl` and `a1fail321/test_scores/A1.jsonl` — the SAME checkpoint
scored on the SAME 677 clips by two different runs of this script — to the temperature
convention. That is wrong on inspection: both files come from divisor-free scorers
(`semsup_train.py:1199` and `score_checkpoints_on_test.py` are both bare
`softmax(logits)[0,1]`), and a real `/2.0` would force every score to exactly 0.5, which is
not what the two files show. The actual, measured gap: **677/677 clips differ** (mean |Δ|
0.0078, max 0.0973, 5 clips flip the 0.5 decision, ΔAP 0.0009 — the same size as the headline
V12-vs-v12shuf effect). Zero-centred (signed mean +0.00118, median +0.00001), consistent with
**unmanaged GPU-inference nondeterminism** (no `torch.use_deterministic_algorithms`/
`cudnn.deterministic` was set in either scorer at the time), not a systematic scoring-formula
difference. See `PROJECT_STATE.md`'s "Measurement integrity" note and the 2026-09-06 project
review §3.1 for the full derivation. `--deterministic` (default on) was added to
`semsup_train.py` 2026-09-08 to close this; `score_checkpoints_on_test.py` should get the same
flag before its next real GPU use.

## `select_a1fail321.py` (new) and `build_a1fail321_comparison.py` (new)
`select_a1fail321.py`: mines A0/A1's own threshold-0.5 failures from the 1,761-pool into the
321-window pool above, splits by `video_id` (seed 0), writes per-arm caption files joined from
the existing V10/V12 1,761-pool corpora. `build_a1fail321_comparison.py`: per-clip comparison
workbook across the 4 arms (`outputs/a1fail321/a1fail321_arm_comparison.xlsx`), modeled on
`build_pool1761_comparison.py`.

## Presentation deck (2026-08-29)
`reports/presentations/2026-08_a1-failure-recovery.pptx`, generator
`build_a1fail_presentation.py` — 6 slides, house style matching the existing `2026-08-22` deck
(reuses its palette/helper-function conventions, does not modify that file). Has a `verify()`
gate that re-derives every embedded number from the actual score/result files and asserts
against known-good values before writing — run it after ANY score-file change, never hand-edit
numbers into the deck. Also regenerates `make_arch_figures_2026-08-22.py`'s `fig_L3()` (now
parameterized by `lam`/`out_name` so a different `semantic_weight` can be drawn without
overwriting the original 0.05-weight figure other decks depend on) — produced
`reports/figures/arch_L3_training_a1fail_2026-08-29.png` (lambda=0.2 variant).

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
| `teacher_distillation/scripts/caption_leakage_gate.py` | Persisted (2026-08-16) TF-IDF+LogReg+GroupKFold label-leakage gate — was run ad-hoc before. Reproduced V10=0.9643/V12=0.7640 exactly. |
| `student_training/scripts/p3_delta_patches_vs_pooled.py` | Does the semantic gradient reach the classifier's own pooled representation, or land where the pooler discards it? Paired bootstrap CI (20 noise draws/clip) on `‖Δpooled‖/‖Δpatches‖`, real vs random. |
| `student_training/scripts/p1_stageA_gate.py` | Scores a Stage-A (semantic-only) checkpoint's encoder against the UNCHANGED frozen crash head on the 677-clip test set — no training. The cheap check before committing to Stage B. |
| `student_training/scripts/score_arms_on_pool1761.py` | Scores one checkpoint (or the frozen A0 baseline, if `--lora-adapter` is omitted) on all 1,761 training-pool windows, inference only. Emits per-window JSONL. Fills the gap that the semantic arms were only ever scored on the 677-clip test set. |
| `student_training/scripts/build_pool1761_comparison.py` | Merges the 6 arms' score files into the per-clip comparison workbook. Fails loudly if A0's re-score does not reproduce the 587 mined failures, or if the split is not 1,413/348. |
| `student_training/scripts/plot_pool1761_comparison.py` | 7 diagnostic figures, each rendered twice (`_all1761` and `_val348`). Reads the same score dir as the workbook so numbers cannot diverge. |
| `student_training/scripts/build_status_presentation_2026-08-22.py` | Builds the 14-slide dark-theme status deck. Asserts every confusion-matrix number against the per-clip result files before writing. |
| `student_training/scripts/make_arch_figures_2026-08-22.py` | The 3 architecture figures (idea / inference / training) for the deck, dark theme. |
| `student_training/scripts/make_dataset_figure_2026-08-22.py` | Dataset+captioning pipeline block diagram; counts read live from the caption files. |
| `student_training/scripts/make_semantic_positive_figure.py` | Retrieval-vs-chance + caption-scaling figure, dark theme. |
| `docs_agents/ARCHITECTURE_BLOCKS.md` | Block-by-block reference (shapes/equations/frozen-status) for the architecture diagram, including the pooled-tap addition and the per-layer gradient diagnostic. Written 2026-09-09 (was previously a dangling reference — see its own header for why). |
| `student_training/scripts/score_semtest.py` | Scores a checkpoint on an arbitrary-size pool (generalized from `score_arms_on_pool1761.py`); `--head-state` loads an `--unfreeze-head` run's head weights. |
| `student_training/scripts/select_semtest200_recovery.py` | 3-tier priority clip selection for SemTest-200 (FN-near/TP-fill/FN-wide positives, 100%-FP negatives), one window per video, `--exclude-frames-dir` for iterative QC rounds. |
| `student_training/scripts/make_semtest200_shuffled.py` | Content-vs-presence control: permutes a caption corpus within class (derangement, seeded). |
| `student_training/scripts/build_semtest200_comparison.py` | Per-clip SemTest-200 comparison workbook (per_clip/summary_vs_A0/summary_vs_vision/metrics sheets), modeled on `build_pool1761_comparison.py`. |
| `student_training/scripts/plot_semtest200_curves.py` | Loss-vs-epoch (2×2 grid, dashed selected-checkpoint line) + val_AP-vs-epoch overlay for the 4 SemTest-200 arms. |
| `student_training/scripts/add_vs_a1_summary_sheet.py` | Adds `summary_vs_A1` sheet to `pool1761_arm_comparison.xlsx` (A1-baseline block, `broken_FP`/`broken_FN` split, corrected `still_wrong`). |
| `prompts/PROMPT_SEMSUP_V13_CAUSAL.py` | Causal-cue captioning prompt (brake lights, ego maneuver, road geometry, signal state, occlusion) — see the section above for its known opener-template-collapse issue. |
| `student_training/scripts/select_a1fail321.py` | Mines A1's own threshold-0.5 failures (321/1,761 windows) into the A1-failure-recovery pool, splits 260/61 by `video_id` (seed 0), writes per-arm caption files. |
| `student_training/scripts/run_a1fail321_4arms.sh` | Driver: runs the 4 A1-failure-recovery arms strictly sequentially (concurrent BADAS loads can crash each other). |
| `student_training/scripts/build_a1fail321_comparison.py` | Per-clip comparison workbook across the 4 A1-failure-recovery arms, modeled on `build_pool1761_comparison.py`. |
| `student_training/scripts/score_checkpoints_on_test.py` | Loads BADAS once, swaps LoRA adapters between checkpoints, scores each on the 677-clip test set. `softmax(logits/temperature)[0,1]`, default temperature=1.0, no `/2.0` (see the A1-failure-recovery section for why this doesn't affect AP/AUC/CM@0.5 — it DOES affect Brier/ECE, see the 2026-09-06 project review §4.5). `--deterministic` (default on, 2026-09-08) pins inference numerics; `--head-states` reloads/hard-requires `head_state.pt` for `--unfreeze-head` checkpoints. |
| `student_training/scripts/paired_bootstrap_ab.py` | **Documented here 2026-09-09 — was previously undocumented despite being the statistical backbone of every A-vs-B comparison in this thread** (project review §6.3). Paired bootstrap (5000 resamples default) on two `test_results_epNN.jsonl`-shaped files, matched by `video_id`; reports ΔAP/ΔAUC point estimate + 95% CI + P(B beats A). Reads either label convention a test-set score file in this project might use — `ground_truth` (0/1, `semsup_train.py`'s own scorer) or `gt_verdict` (`"YES"/"NO"`, `score_checkpoints_on_test.py`'s schema) — fixed 2026-09-09 (it previously read only `ground_truth` and could not run at all on `outputs/a1fail321/test_scores/*.jsonl`). Used for the B_1761-parallel/B-v2/B-v3/P1 ΔAP-vs-A1_1761 CIs quoted throughout `DECISIONS.md`/`EXPERIMENTS.md`, and (2026-09-09) the full recovery-family pairwise matrix — see `EXPERIMENTS.md`'s "Bootstrap CIs" section. Note its `ap_a`/`ap_b` are recomputed from the per-clip dump, which is why e.g. `bootstrap_vs_a1_1761.json`'s `ap_a=0.8986` differs from the headline published `0.9000` — see `WEBSITE.md`'s "Two sources of truth" note for why the dump and the summary disagree on AP but not the confusion matrix. |
| `student_training/scripts/build_a1fail_presentation.py` | 6-slide deck for the A1-failure-recovery result, house style matching the `2026-08-22` deck; `verify()` re-derives every number from score files before writing. |
| `student_training/scripts/make_semtest200_folds.py` | Stratified 5-fold split of the SemTest-200-v2 pool by `video_id` + source tier; self-asserts an exact partition. |
| `student_training/scripts/aggregate_semtest200_cv.py` | Pools per-fold val_scores from SemTest-200-v2 into a full-pool readout. |
| `student_training/scripts/select_semtest200_easy.py` | Selects 100 easy A0-correct anchor clips to add to the original 200-clip SemTest-200 pool (addresses its 100%-adversarial/zero-TN composition). |
| `student_training/scripts/merge_semtest200_v2.py` | Merges the 200-clip pool + 100 easy anchors into the SemTest-200-v2 300-clip pool. |
| `student_training/scripts/merge_semtest200_v2_captions.py` | Joins caption corpora for the merged SemTest-200-v2 pool. |
| `student_training/scripts/plot_semtest200_cv_curves.py` | Mean±std-band loss curves across SemTest-200-v2 folds; shared y-axis; right axis color-keyed to its own series; `--mark-epoch`/`--init-note` for annotating a selected checkpoint. |
| `student_training/scripts/siglip_bottleneck_probe.py` | Measures how much crash-relevant signal survives text→SigLIP-embedding vs raw text; ran on V10/V12/V13 — SigLIP retains 86-96% of the text's own crash-AUC, ruling out the encoder as the bottleneck for prior negative results. |
| `student_training/scripts/build_midpoint_negatives.py` | (2026-09-27) Re-cuts the 1,761 pool's 905 negative windows to the Nexar test protocol: per video one fake event = midpoint + N(0, `--noise-std` 1.0s) (seeded per video_id), windows end 0.5/1.0/1.5s before it; group index preserved (MID-10→0, MID-4→1, MID-8→2); extracts frames from the raw mp4s into `dataset/train/<vid>_hires_midtest{05,10,15}/`; writes `outputs/semantic_captions/Caption_Train4500_MidpointNeg_1761.jsonl`. `--dry-run` plans only. |
| `student_training/scripts/pooled_eval.py` | (2026-09-27) Pools private 677 + public 667 per-clip scores (asserts counts and no overlap) and compares arms against a `--ref` with a clip-level paired bootstrap; prints AP, AUC, ΔAP + 95% CI, P(better), FPR at matched recall (`--fpr-at-tpr 0.85,0.90`). Usage: `--arm NAME=private.jsonl,public.jsonl` (repeatable). |
| `outputs/overnight_2026-09-27/{run_overnight.sh,watchdog.sh,score_controls_public.sh}` | Pod driver scripts for the overnight run and the controls' public scoring; end with `runpodctl stop pod <id>` (pod ids hard-coded — edit per pod). Pattern to reuse; per memory rule, future drivers must sync results locally BEFORE the stop call. |

**Seeds in `semsup_train.py`:** `--split-seed` fixes the train/val partition; `--init-seed` (defaults
to `--seed`) seeds `random` + `torch` + CUDA, which drives LoRA init (lora_A random, lora_B zero, so
step-0 output equals BADAS-Open for every seed), the per-epoch training-example shuffle, and LoRA
dropout (p=0.05) — all from ONE RNG, so init vs data-order effects cannot be separated by flags.

**Website builders (additions through 2026-09-29):** `build_experiments_data.py` has `OVN` path,
`public_block()` + `EXPECTED_CM_PUBLIC` (public 667 panel, pinned to per-clip dumps), mean-over-N
AP from `by_epoch`, `STAGE1`-style noise floor derived from AA-ctrl-seed1/2; EXPECTED_CM /
EXPECTED_CM_PUBLIC already contain the 6 overnight arms (midneg/fullpool seed0-2) but no ARMS entries
reference them yet. `build_compare_data.py` `TEST_SCORES` includes those 6 arms (val-selected epochs).
`assets/site.css` `.pickrow` now sets `background:transparent` + resets button borders (was UA gray).

## P1 — two-stage (semantic-pretrain → crash-finetune) training

All four joint-training attempts (B_1761-parallel, B-v2, B-v3, +12-epoch extension) lost to
A1_1761, with the gap *widening* as execution defects were fixed — evidence the failure isn't
routing/leakage but something about training both objectives jointly under a fixed λ. P1 tests
the alternative: converge the semantic objective fully first, then fine-tune on crash alone.

```
STAGE A (semantic only)                    STAGE B (crash only)
16 frames → ViT-L + LoRA ─┐                16 frames → ViT-L + LoRA(init from Stage A)
                          ↓                            ↓
                     Predictor                    crash head (FROZEN)
                          ↓                            ↓
            InfoNCE vs frozen SigLIP bank        CE vs GT label
       train: LoRA + Predictor + log τ          train: LoRA only (Predictor discarded)
```

`semsup_train.py` implements both stages via two new flags:
- `--crash-weight` (default 1.0): weight on the crash CE term in the optimized loss
  (`loss = crash_weight*crash_loss + semantic_weight*sem_loss`). `0.0` = Stage A — no crash
  gradient reaches the trunk at all. Crash loss is still computed and logged every epoch
  regardless (a free diagnostic of head-compatibility), just not optimized. Guards against
  both weights being 0.
- `--select-by {val_ap, retrieval}` (default `val_ap`): checkpoint-ranking/early-stop metric.
  Stage A **requires** `retrieval` — val_ap is uninformative when nothing optimizes it (the
  same bug class already fixed once in `semsup_b1_probe.py`).

Real result (2026-08-17): Stage A peaked at epoch 10 (retrieval@1 46× chance) then overfit;
the gate passed (small expected dip vs A0); Stage B **lost by the widest margin in the entire
thread** (ΔAP=+0.0716 vs A1_1761, 95% CI excludes zero) — see EXPERIMENTS.md for the full
numbers and the measured overfitting mechanism (warm-started LoRA + unchanged from-scratch LR).

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

# semsup_train.py (2026-08-17, P1 two-stage)
evaluate_val(..., full_bank=None, retrieval_tolerance=0.92)
    -> (val_ap, val_crash_loss, val_sem_loss, n_failed, retrieval_stats: dict)
    # retrieval_stats keys (empty dict unless predictor+val_bank both exist):
    #   retrieval_clip, collapse_control_clip, retrieval_clip_full1761,
    #   retrieval_clip_tolerant, n_retrieval_clips,
    #   embed_margin_mean, embed_max_q_mean, embed_std_s_mean, embed_std_p
    # BREAKING for existing callers: was a 4-tuple, now 5. Only call site
    # (inside main()) already updated.
_clip_grads / gradient-angle probe / --clip-grad-per-group   # unchanged, pre-existing
# new CLI args: --crash-weight (default 1.0), --select-by {val_ap,retrieval} (default val_ap),
#   --retrieval-tolerance (default 0.92)
# checkpoint-summary JSON fields renamed: "val_ap" -> "selection_metric"/"selection_value"
#   in metrics_ep*.json/test_summary.json's per-checkpoint entries (train_metrics.json keeps
#   "best_val_ap" for backward compat, populated only when select_by=='val_ap'). Nothing
#   downstream currently parses these fields programmatically - verified before the rename.

# semsup_b1_probe.py (2026-08-17)
clip_level_retrieval_detail(P, T, vids_list) -> (clip_ids: list, hits: list[int])  # LIFTED to
clip_level_retrieval_acc(P, T, vids_list) -> float                                 # module level
    # were nested inside main(), uncallable from outside. Pure move, no logic change.
    # semsup_train.py now imports clip_level_retrieval_acc directly.
```

## APIs / functions (added 2026-08-23/24, signatures only)

```python
# score_arms_on_pool1761.py  (CLI, runs on the pod)
#   --config --captions-path --arm-name --out  [--lora-adapter]
#   omit --lora-adapter  -> frozen A0 baseline, no adapter attached at all
#   emits one JSON row per window:
#     {arm, video_id, frames_dir, requested_time_to_event, gt_verdict, score}

# build_pool1761_comparison.py  (CLI, local)
#   --scores-dir --out
clip_level_split(video_ids, val_frac=0.2, seed=0) -> set   # replica of semsup_common's
    # partition, kept in sync deliberately: the train/val column is wrong if it drifts
#   Hard gates (SystemExit): row counts != 1761, missing arm, arm-name divergence between
#   CM_ROWS and EXPERIMENTS, A0 re-score not matching the 587 mined failures.

# plot_pool1761_comparison.py  (CLI, local)
#   --scores-dir --out-dir   -> 7 figures x {_all1761, _val348}

# build_status_presentation_2026-08-22.py
verify()   # asserts every CM cell against outputs/e4_vjepa_reason/*/test_results_*.jsonl
           # and that arm names match across the two slide tables; exits non-zero on mismatch
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

## APIs / functions — SemTest-200 / head-unfreeze / V13 (added 2026-08-26/28, signatures only)

```python
# semsup_common.py
TrainableBadasWrapper.__init__(..., unfreeze_module_substrings: list|None = None)
    # NEW param: requires_grad_(True) on any param whose name contains a listed substring
    # (e.g. "temporal_processor", "classifier"), post-LoRA-wrap. Exposes .head_params /
    # .head_param_names.
TrainableBadasWrapper.head_state_dict() -> dict[str, Tensor]   # NEW: just the unfrozen head
TrainableBadasWrapper.load_head_state(path)                     # NEW: strict=False reload

# semsup_train.py
# new CLI args: --unfreeze-head, --head-lr-mult (default 0.1), --val-video-ids <file>,
#   --dump-val-scores
# AdamW(trainable, lr=...) -> AdamW(param_groups)  # head gets its own {params, lr} group
_clip_grads(..., head_params=None)   # third clip-budget group when --clip-grad-per-group
evaluate_val(..., dump_scores_path=None)   # NEW: writes val_scores_ep{NN}.jsonl per call
# epoch_XX/ dir now also writes head_state.pt when --unfreeze-head is set

# score_semtest.py (new script, CLI)
#   --config --captions-path --arm-name --out  [--lora-adapter] [--head-state]
#   drops score_arms_on_pool1761.py's hardcoded 1761-row expectation

# select_semtest200_recovery.py (new script, CLI)
#   --a0-scores --manifest --train-xlsx --out-dir  [--exclude-frames-dir] [--tp-fill-max=0.85]

# semsup_caption_promptbakeoff.py
# new CLI args: --provider-order <slug,...> (extra_body provider pin, allow_fallbacks=False),
#   --token-cap <N> (SigLIP-tokenize caption_neutral post-parse, stamp caption_token_len)
_stamp_token_len(out_row, siglip_tok, cap) -> bool          # NEW helper
validate_parsed(..., prompt_key="v13")                       # NEW branch: word-count band +
                                                               # per-field coverage + colour scan
# DEFAULT_MODEL fixed: "google/gemini-3.1-pro-preview" -> "google/gemini-3.7-flash"

# prompts/PROMPT_SEMSUP_V13_CAUSAL.py
build_prompt() -> str    # no args, same contract as V12
```

## APIs / functions — head-LR schedule + caption-bank widening (2026-08-29, signatures only)

```python
# semsup_train.py
# new CLI args: --head-lr-schedule {cosine,constant} (default cosine), --bank-captions <corpus>
# epoch_metrics.jsonl now also logs head_lr per epoch

# score_checkpoints_on_test.py (new script, CLI)
#   --config --test-manifest --test-frames-root --checkpoints <name=path,...> --out-dir
#   loads BADAS once, swaps LoRA adapters between checkpoints; softmax(logits)[0,1], no /2.0
#   divisor (see ARCHITECTURE.md's A1-failure-recovery section for why this is safe for AP/AUC/CM@0.5)

# select_a1fail321.py (new script, CLI)
#   --a1-scores <pool1761 A1 scores> --manifest --train-xlsx --out-dir
#   mines A1's threshold-0.5 failures (321/1761), splits 260/61 by video_id seed 0
```

**Required workaround inside `semsup_train.py`** (do not remove):
```python
badas.nn_model.create_or_update_model_card = lambda *a, **k: None
```
`peft`'s `save_pretrained()` builds a model card before writing weights and assumes
`base_model.config` is dict-like; BADAS's `ModelArgs` isn't, so every checkpoint save crashed.

## 1.5s-TTE plan tooling + Stage 2 Look-Ahead head (2026-09-29/30)

**Look-Ahead head (Stage 2, off by default).** `z` = the pooled 1024-d vector the crash classifier
consumes (probe output). `z_hat = z + MLP(z)` (LayerNorm→1024→256→GELU→1024, last layer zero-init),
`logits = logits(z) + g * classifier(z_hat)` with `g` a learnable scalar init 0 → at step 0 the
model equals BADAS-Open exactly. Training-only target: FROZEN BADAS-Open vector of the same video's
window 0.5 s later (1.5→1.0, 1.0→0.5), loss `MSE/copy_scale` (1.0 = no better than copying), weight
`--lookahead-weight` set by the pilot's gradient-norm ratio (0.3·|g_crash|/|g_aux|). Frozen (not
live) targets are a deliberate stable-target choice (no second forward pass).

Constraints: the head is attached to the wrapper, so validation, in-training test scoring and
`score_checkpoints_on_test.py` all use it; checkpoints carry `epoch_NN/lookahead.pt`, and the scorer
detaches it for arms without one. Shuffled control keeps class + horizon step.

| path | purpose |
|---|---|
| `student_training/models/lookahead.py` | `LookAheadHead`, `partner_name`, `load_features`, `build_pair_targets` |
| `student_training/scripts/semsup_train.py` | new flags `--lookahead-{features,weight,hidden,shuffle,no-mix}`, `--horizon-weights[-scope]`, `--limit-random`; logs `aux_grad_cos_by_group` |
| `student_training/scripts/semsup_common.py` | wrapper: `_classifier`, `attach_lookahead()`, logit mixing in `forward_clip`; examples carry `horizon_label` |
| `student_training/scripts/score_checkpoints_on_test.py` | loads `lookahead.pt` next to an adapter |
| `student_training/scripts/stage_compare.py` | paired per-seed arm comparison + pre-registered pass rule |
| `student_training/scripts/tte_same_video_diag.py` | Stage 0a: link test clips of one video by frame matching |
| `student_training/scripts/lookahead_extract_features.py` | Stage 0b (pod): frozen pooled vectors + head params; `--names-file` |
| `student_training/scripts/lookahead_feasibility.py` | Stage 0b (local): held-out ridge, copy baseline, oracle head AP; `--balance` |
| `student_training/scripts/build_midpoint_negatives.py` | `--complete-horizons`: all 3 midpoint windows per negative video, captions untouched |
| `outputs/stage2_lookahead_2026-09-30/run_stage2.sh` | Stage 2 driver (needs `/root/lookahead_features.npz`) |
| `outputs/lookahead_0b_merged/features.npz` | frozen vectors for the 1,761-pool videos x 3 horizons (look-ahead targets) |

Signatures: `LookAheadHead(dim=1024, hidden=256, gate_frozen=False).predict(z) -> z_hat`;
`build_pair_targets(examples, feats, shuffle=False, seed=0) -> (targets: dict[frames_dir, vec],
copy_scale: float, n_missing)`; `TrainableBadasWrapper.attach_lookahead(head)`.

**Pod constraints (new):** network volume 0hnvco2s4j is at its 56 GB quota — write run outputs to
the container disk (/root) and pull before stop. Drivers export `RUNPOD_API_KEY`/`RUNPOD_POD_ID`
from `/proc/1/environ` so `runpodctl stop` works.

## Stage AA — detection pipeline v2b (2026-09-19; separate from the student model above)

Offline subsystem, no BADAS/LoRA: produces per-object kinematic signals from raw dashcam video for
the father plan's detection-guided auxiliary supervision. Child plan:
`~/.claude/plans/CCP based BADAS/2026-09-15_Child-Plan-AA1-v2b-YOLOPv2-lanes-tracking.md` (stale in
Stage 2/3 detail; this section is authoritative). Stages 0–3 built; Stage 4 (targets/masks) not.

```
raw MP4 at native fps (decode_span)
 -> YOLOPv2 (aa1_scene.YOLOPv2.infer): vehicle boxes + drivable + lane masks, cached per frame
    (boxes down to conf 0.10, masks RLE)                              -> <vid>_yolop.json
 -> BoT-SORT (buffer 90, activation/high-conf 0.30, low-conf 0.10..0.30 only continues tracks)
    -> stitch_fragments -> is_ego_hood -> extend_edge_tracks           -> <vid>_tracks_v2.json
 -> load_frame_masks: per frame decode masks, trace_path (old drivable path, only used by nothing
    downstream now), estimate_ego_path                                  (in memory)
 -> compute_track_scores: one causal pass -> per (track, t) score dict
 -> per window: candidate_pool -> select_top_k(5)                      -> <vid>_threat_v2.json
    overlay video + target curves                                       -> _overlay_threat.mp4, _target_curves.png/csv
```

**Ego path** (`aa1_lanes.estimate_ego_path`, writes `frame_masks[t]["ego_path"] = dict(coef, top, src)`,
evaluate with `path_x_at(ep, y)`; x = polyval(coef, (y−360)/360), clamped to rows ≥ `top`):
1. `static_lane_mask`: pixels lane-classified in ≥ 95% of the clip's frames are removed from every
   frame (a fixed object, not paint). 4 known-good clips have 0 such pixels.
2. `_calibrate_lanes`: fixed image-centre reference, `_nearest_lane_points` (per row, nearest lane run
   each side of the reference, runs ≤ 60 px wide), `_ransac_curve`; frames with both lines and a
   lane that widens toward the ego give the clip's `anchor` (x of the lane centre at the nearest
   fitted row) and a lane-width model `w(y)=p·y+q`. Needs ≥ 8 pair frames (`PAIR_MIN_FRAMES`) else
   anchor = 640, no model.
3. Per frame `_measure_path`: `_exclude_vehicles` (zero lane pixels inside detected vehicle boxes),
   row scan around the PREVIOUS smoothed path, `_ransac_curve` per side (min 6 inliers, span 40 px).
   Both lines → `lane_pair` (centre, width must be 0.6–1.5× model if a model exists); one line →
   `lane_single` (shifted half a lane width; needs the model); else none.
   `_fit_centreline` fits a quadratic (curvature cap 120, else straight) through the samples plus the
   anchor at row 719 (weight 5).
4. Smoothing: EMA (τ 0.25 s) on the 3 coefficients, not per row. A measurement whose mean |Δx| over the
   lower half exceeds `PATH_GATE_PX`=120 is rejected unless rejections last `PATH_RESYNC_S`=0.4 s.
5. The first accepted measurement must be `lane_pair`; earlier frames are `backfilled` from it. After
   `PATH_RESET_HOLD_S`=0.8 s with no measurement, retry against the anchor as reference (accepted only
   as `lane_pair`). With no lane evidence the path is held (`held`); a clip with no seed gets a vertical
   line at the anchor (`default`).
`src` ∈ lane_pair / lane_single / held / backfilled / default. The overlay draws the path down to `top`.

**Selection score** (`aa1_collision.py`; NOT a threat, NOT a target): `score = EMA_τ0.3s( closeness ×
side × (1+approach) )`, ×0.25 if shielded.
- closeness = min(box_h/360, 1). approach = clip(α, 0, 1), α = log-size slope over the last 1 s
  (`aa1_lanes.fit_alpha`; area / width / height by which box edges are clean).
- side: g = horizontal gap from the box's bottom edge to the ego path at that row, in box heights
  (perspective-free ruler); g_rate = slope of g over 1 s; g_eff = min(g, max(g + g_rate·1 s, 0));
  side = 1 for g_eff ≤ 0.3, falls linearly to 0.15 at 1.5.
- shielded: another box is ≥ 10 px lower (nearer) and overlaps ≥ 35% in x.
- candidate pool per frame/window: recent (present in ≥ half of the last 0.5 s), box height ≥ 25 px, only
  the 8 tallest. Top-5 by smoothed score; every top-5 member is drawn coloured regardless of score.

**Target-curve plot** (`render_target_curves`): 3×2, row 3 = selection score across both columns; the 4
proposed Stage 4 targets α, g, closing_rate (= −g_rate), lane (LEFT/EGO/RIGHT, 5-sample majority);
title says TP/TN; dashed lines + labels at the window ends; per-panel x-axis and legend; shows the 5
longest-lived tracks (not the selected top-5); x = seconds into the decoded span (crash clips 3 s,
normal clips 8 s with an unused gap).

**Windows** (`window_ends`): positives TTE 1.5/1.0/0.5 s before `time_of_event` (span event−3.5…−0.5 s);
negatives MID−10/−8/−4 s around the clip midpoint; ends floored at `T_FLOOR`=2.0 s.

### Constraints / invariants
- `YOLOPv2.infer` requires 1280×720 BGR (asserts). Cache timestamps are rounded to 4 dp — always look up
  with `round(t, 4)`.
- `rle_encode` must not emit a zero-length first run (round-trip-test before changing).
- BoT-SORT needs frames passed to `update()` (CMC); its low-confidence stage is hardcoded `> 0.1`.
- `aa1_stage2.OUT_DIR` is module-level: `aa1_run_set.py` sets it; anything else reading another folder
  must set it too.
- `load_frame_masks` mutates `lane` masks in place (static pixels removed); callers after it see the
  cleaned mask (the overlay draws the cleaned one).
- Shell pipelines (`python ... | grep`) report grep's exit code, not python's — check for tracebacks.

### Files that matter

| Path | Purpose |
|---|---|
| `student_training/scripts/aa1_scene.py` | YOLOPv2 wrapper (`YOLOPv2.infer`) |
| `student_training/scripts/aa1_yolop_cache.py` | Stage 0 cache (`LOW_CONF_THRES`=0.10), recall vs v1 G-DINO |
| `student_training/scripts/aa1_tracks.py` | `track_from_yolop`, `stitch_fragments`, `is_ego_hood`, `extend_edge_tracks` |
| `student_training/scripts/aa1_track_stage1.py` | Stage 1 driver → `_tracks_v2.json`; exports `VAL_E3A_IDS`, `PALETTE_HEX` |
| `student_training/scripts/aa1_lanes.py` | lane geometry: `rle_decode`, `trace_path`, `fit_alpha`, `estimate_ego_path`, `path_x_at`, `track_geometry_v2` (legacy, unused) |
| `student_training/scripts/aa1_stage2.py` | loaders only: `load_frame_masks`, `load_stitched_tracks` |
| `student_training/scripts/aa1_collision.py` | `compute_track_scores`, `candidate_pool`, `select_top_k`, `side_gap`, `lane_side` |
| `student_training/scripts/aa1_stage3.py` | Stage 3 driver: window top-5, `render_overlay`, `render_target_curves`, `run_clip`, `main` (18 dev clips) |
| `student_training/scripts/aa1_run_set.py` | runs Stages 0/1/3 on the gen18 set (`--set gen18 --stages …`); seeded sampler, writes `clip_list.json` |
| `student_training/scripts/aa1_detect_track_rank.py` | v1 baseline (do not extend); still supplies decode/window helpers |
| `third_party/yolopv2_ref/` | vendored YOLOPv2 utils; `weights/yolopv2.pt` gitignored |
| `outputs/aa1_v2_18clips/` | ALL AA.1 outputs, 36 clips (18 dev + 18 gen18): `_yolop.json`, `_tracks_v2.json`, `_threat_v2.json`, `_overlay_threat.mp4`, `_target_curves.png/csv`; plus `stage1_summary.json`, `yolop_vs_gdino_comparison.json`, `clip_list.json` |
| `outputs/aa1_smoke_18clips/` | v1 baseline outputs — keep untouched |

Removed 2026-09-19 and no longer generated: `_grid16_{lanes,threat,v2}.jpg`, `_path_check*.jpg`,
`_geometry_v2.json`, `_overlay_lanes.mp4`, `_overlay_v2.mp4`, `_timeline_v2.png`.

### APIs — signatures only
```python
# aa1_lanes.py
fit_alpha(track, t_window_end, ...) -> (alpha, alpha_mode, pts, low_samples, span) | None
static_lane_mask(frame_masks, ts) -> bool (720,1280)              # STATIC_LANE_FRAC = 0.95
_exclude_vehicles(lane, boxes) -> lane
_nearest_lane_points(lane, xref_fn, step=4, max_run=60) -> (left, right)  # [(row, x)]
_ransac_curve(pts, rng, iters=200, thr=6, min_inl=10, min_span=60) -> (coef, y_min, y_max) | None
_calibrate_lanes(frame_masks, ts, rng) -> (anchor, width_model | None, n_pair_frames)
_measure_path(fm, xref_fn, anchor, model, rng) -> (coef, top, src) | None
estimate_ego_path(frame_masks, seed=0) -> summary dict            # writes ego_path per frame
path_x_at(ego_path, y) -> float
# aa1_collision.py
compute_track_scores(tracks, frame_masks, fps) -> {tid: {t: dict(score, raw, closeness, side, approach,
                                                          alpha, g, g_rate, lane, shielded, box_h)}}
candidate_pool(boxes_now, tracks, t, fps) -> {tid: box};  select_top_k(pool, scores_now, k=5)
side_gap(box, ego_path) -> float;  lane_side(box, ego_path) -> "LEFT"|"EGO"|"RIGHT"
# aa1_tracks.py
extend_edge_tracks(tracks, cache) -> (tracks, [(tid, n_added, side)])   # continues edge-touching tracks with
                                                                        # low-conf boxes, <= 0.3 s gaps
# aa1_stage3.py
run_clip(video_id, out_dir) -> windows;  render_target_curves(video_id, tracks, scores, timestamps, win_ends, is_pos, out_dir)
# aa1_run_set.py
sample_clips(n_pos, n_neg, seed) -> dict;  main(--set gen18, --stages 0,1,3)
```

### v1 baseline — `aa1_detect_track_rank.py` (committed `812e884`)
G-DINO → BoT-SORT → calibrated virtual horizon corridor; threat = `max(alpha·in_path, rho·ttc_lat_inv)`.
**Rejected** (DECISIONS.md). Its decode/window helpers are reused.

## Stage AA token-relevance auxiliary loss (2026-09-19 to 2026-09-24) — negative result

Teaches the ViT-L's intermediate tokens *which detected car is relevant to a possible collision*
via a per-token BCE side-loss, alongside the normal crash CE loss, hoping to beat A1-compress256
with zero added inference cost. **Result: does not beat A1-compress256** — see EXPERIMENTS.md for
all 6 runs' numbers and the diagnostics explaining why. Full design history/rationale (including a
rigorous review that reshaped an external "spatiotemporal attention supervision" proposal into
this plan) lives in the child plan:
`~/.claude/plans/CCP based BADAS/2026-09-19_Child-Plan-AA-token-relevance-aux.md` — read its
"Status 2026-09-24" block first, it is the up-to-date summary; the rest of that file is design
history, superseded where the status block disagrees with it.

### Data flow
```
            MAIN PATH (training AND inference, unchanged from A1-compress256)
  16 frames -> compress256 -> patch embed -> 2048 tokens x 1024 -> enc layer[0..16] -> enc layer[17]
                                                                          |
       same tensor continues, untouched --------------------------------+
                                                                          v
                             enc layer[18..23] -> layernorm -> predictor (+512 future tokens) -> 2560
                                                                          v
                                                  crash head (FROZEN or unfrozen, see runs) -> P(crash)
                                                                          v
                                                        LOSS 1 = CE vs crash label

            AUX PATH (TRAINING ONLY -- absent at inference)
      forward hook reads layer[N]'s output (N=17 default, N=23 tried once)
                    |  2048 tokens x 1024
      LayerNorm(1024, no affine) + Linear(1024 -> 1)   [FROZEN after Phase 3 probe fit]
                    v  2048 logits -> sigmoid -> r_hat per token (8x16x16 grid)
        LOSS 2 = BCE-with-logits(r_hat, R)   R = 'occ' (hard, any vehicle) or 'rel' (soft, top-5
                                              selection score / 2), from AA.4 labels

  L = crash_weight*LOSS_1 + aux_weight(lambda)*LOSS_2
  LOSS 1 gradient -> every LoRA adapter.   LOSS 2 gradient -> encoder LoRA in layers 0..N only
  (confirmed via the per-layer gradient trace: zero gradient in layers N+1..23 and the predictor
  stack, by construction of where the aux graph ends).
```
The hook, aux head, and aux loss term are all absent from the inference path — same cost/latency
as A1-compress256. Token index <-> (tubelet 0-7, row 0-15, col 0-15) <-> a 16x16 patch of the
256x256 compressed frame pair; `x*256/1280, y*256/720` then `/16` maps a detection box to tokens.

### Why the aux head is a single shared Linear(1024,1), not per-position weights
Applied identically to every token (like a 1x1 conv), 1,025 weights total. A per-position head
(`Linear(2048*1024 -> 2048)`) would have ~2.1B weights and memorize the 1,761 windows rather than
learn a position-invariant "is this patch relevant" readout. The head is fit once in Phase 3
(`aa1_token_probe.py`, frozen trunk) and then **frozen** for the actual training runs — letting it
train alongside LoRA would let it absorb the aux loss on its own, exactly like the semantic-
alignment Predictor did in the earlier semantic-supervision thread (see `P1`/`B-v*` sections
above) — the point is to force the *trunk representation itself* to make relevance linearly
readable, not to build a better relevance head.

### Runs (full numbers, diagnostics: EXPERIMENTS.md)
Recipe = A1-compress256's exact config (LoRA r16/α32/dropout0.05 on `query,key,value`, lr 2e-4
constant, grad_accum 8, compress256, pool1761, seed 0) + the aux loss, unless noted:
- **AA-rel** / **AA-occ** — 8 epochs, frozen crash head, `aux_layer=17`, λ sized to a ~10-12%
  gradient-norm pull. Neither beats A1-compress256.
- **AA-rel-L23** — same but `aux_layer=23` (closer to the crash head's actual input, per the user's
  explicit "run layer 23 first" instruction after layer-17's non-propagation was found). Does not
  fix it either.
- **AA-ctrl/occ/rel-unfrozen-check3** — 3-epoch checks with `--unfreeze-head --head-lr-mult 1.0
  --head-lr-schedule constant` (testing whether a frozen crash head, unable to recalibrate to a
  LoRA-shifted feature distribution, was the bottleneck — see the pre-existing calibration finding
  in PROJECT_STATE.md's SemTest-200/A1-failure-recovery sections). None separates from its control.

### λ sizing: gradient-norm vs loss-value disagree (state this plainly in any write-up)
λ is sized so `λ*||g_aux|| / ||g_crash|| ~ 0.1` on shared LoRA layers, measured via a 1-epoch pilot
— gradient norm is what actually moves the weights, the defensible criterion. But the **weighted
loss VALUE** (`λ*aux_loss`) can exceed `crash_loss` for most of training under this sizing (AA-rel,
λ=3.6: 0.80 vs 0.58 at epoch 1, still 0.72 vs 0.21 at epoch 8) — a genuine, surprising tension
between the two natural ways to size a multi-task loss weight, worth flagging explicitly since
eyeballing the raw loss curves alone would give the wrong impression of which term dominates.

### New/changed code (signatures only)
```python
# semsup_common.py — TrainableBadasWrapper.__init__ gains:
aux_layer: int | None = None   # installs a forward hook on `...encoder.layer.{aux_layer}`,
                                # stores output (undetached) to self._captured["aux_tokens"]

# semsup_train.py — new
build_aux_head(device) -> nn.Sequential   # LayerNorm(1024, elementwise_affine=False) + Linear(1024,1)
balanced_bce_loss(logits, target) -> Tensor
load_aux_target(frames_dir, mode, labels_dir, device) -> Tensor | None   # cached npz loader
write_grad_trace(...) -> None    # per-epoch grad_trace.jsonl: per-layer crash/aux grad norms,
                                  # cosine, vanishing_ratio, leaked_into_later_layers,
                                  # lora_weight_change_norm
# new CLI: --aux-mode {none,occ,rel} --aux-layer 17 --aux-weight --aux-labels-dir --aux-head-init
# L = crash_weight*crash_loss + semantic_weight*sem_loss + aux_weight*aux_loss

# semsup_train.py — --head-init (new, fixes the --unfreeze-head resume bug, commit 1c56fe2)
--head-init <path to a previous epoch's head_state.pt>
# REQUIRED alongside --lora-init whenever resuming a run that has --unfreeze-head and is past
# epoch 1 - otherwise load_head_state() is never called on resume and the head silently resets
# to its frozen starting weights while LoRA continues, with no error.

# aa1_token_probe.py (new, Phase 3 — frozen-trunk relevance probe)
fit_probe(feat, target, steps=500, lr=1e-2, tag="") -> nn.Module   # full-batch GPU Adam; returns
                                                                    # the SAME build_aux_head weights
spearman_residual(...) -> float   # fits rel ~ static_geometry (box_h, path_gap) via LinearRegression
                                   # on TRAIN object tokens, correlates the probe's VAL score with
                                   # the residual -- isolates the non-static (motion/occlusion) part
--checkpoint <free-form label>    # was closed to {base, a1compress256}; now any label, paired with
                                   # --lora-adapter (required unless checkpoint == "base")
--n-windows N                     # seeded random subsample -- --limit alone is biased because
                                   # Caption_Train4500_Mixed_1761.jsonl is class-sorted
```

### Files that matter (Stage AA token-relevance)
| Path | Purpose |
|---|---|
| `student_training/scripts/aa4_token_labels.py` | AA.4: per-window `rel`/`occ`/`box_h`/`path_gap` token-label arrays (`dataset/aa_token_labels/*.npz`); `render_check()` overlay for spot-checking a window's labels against the frame. |
| `student_training/scripts/aa1_token_probe.py` | Phase 3: frozen/LoRA-loaded multi-layer relevance probe. Caches object + background tokens per window, fits `build_aux_head`, reports AUROC/Spearman vs a static-geometry baseline. Used both to select `aux_layer` before training and, post-hoc, to diagnose a trained checkpoint (does the aux signal survive at a later layer). |
| `student_training/scripts/plot_grad_trace.py` | 4-panel figure from a run's `grad_trace.jsonl` (per-layer gradient norm, cosine, vanishing ratio, leak check). |
| `student_training/scripts/aa1_run_set.py` (`pool1761` set) | AA.2: `--set pool1761 --stages 0,1` runs detection+tracking on the full 1,761-window training pool (1,107 unique videos), with 1s pre-roll so α/EMA are valid at tubelet 0 (`pool_video_ids()`, `kind="pool"`). |
| `outputs/aa_token_aux/` | All Stage AA token-relevance outputs: `probe_base/`, `probe_a1compress256/`, `diag_{base,AA-rel,AA-occ}_n400/` (post-hoc probes on trained checkpoints), `AA-{rel,occ}[-L23]/`, `AA-{ctrl,occ,rel}-unfrozen-check3/` (train_metrics.json, epoch_metrics.jsonl, grad_trace.jsonl, test_results_epNN.jsonl per run — large per-epoch LoRA adapters stay pod-only on `/workspace`, not synced locally). |
| `dataset/aa_token_labels/` | AA.4 output: one `.npz` per window (`rel`, `occ`, `box_h`, `path_gap` arrays), from `aa4_token_labels.py`. |

### Corrections to prior architecture entries, confirmed while building this
- **2048-vs-2560 token concat order, open since August, now CONFIRMED**: read directly from
  `nexight/src/train/video_training.py` — `torch.cat([present_features, future_features], dim=1)`,
  i.e. `[2048 real, 512 predicted]`, default `combination_method='concat'`. The earlier "not yet
  verified" note in the 2026-09-14 PROJECT_STATE entry is resolved.
- **BADAS crash head architecture, corrects an earlier wrong guess** (was assumed to be a
  single-learned-query attention pool): it is `nn.MultiheadAttention` (8 heads) self-attention over
  all 2560 tokens -> LayerNorm -> **plain mean pool** (no learned query) -> MLP classifier. A
  token's influence on the crash score is "how much every other token attends to it, averaged" —
  relevant when reasoning about why a layer-17-specific representation change might or might not
  reach the final prediction.

## Stage AA-H — head-attention/decision-gradient supervision (2026-09-24/25, IN PROGRESS)

Replaces Stage AA's side-probe design (above) after its negative result. Instead of grading an
intermediate ViT-L layer via a frozen side head, the aux loss reads the crash **head's own**
attention table or decision gradient directly, so its gradient reaches every LoRA layer (0-23)
plus the predictor by construction, not just layers 0..aux_layer. Full design, the literature
review that motivated it (RARE/FAX/GAIN/CAMAL), and the live status block:
`~/.claude/plans/CCP based BADAS/2026-09-24_Child-Plan-AA-H-head-attention-supervision.md` —
**read that file's status block for current numbers**, this section covers the architecture only.

### The new head-attention hook
`semsup_common.py`'s `TrainableBadasWrapper.__init__` gains `capture_head_attention: bool =
False`. BADAS's crash head (`temporal_processor`, an `AttentionProcessor`) computes
`attn_out, _ = self.attention(x, x, x)` internally and **discards** the attention weights (`_`)
before returning — so a forward hook on `temporal_processor` itself (the existing `pooled` tap)
cannot see them. The fix hooks the INNER `nn.MultiheadAttention` submodule
(`temporal_processor.attention`) directly: its own forward return value IS
`(attn_output, attn_output_weights)`, so a forward hook there captures the (1, 2560, 2560)
row-normalized table (`self._captured["head_attn"]`), undetached, with a full gradient path back
through the LoRA-unfrozen trunk. The pre-existing pre-hook `self._captured["patches"]` (the
head's raw, still-batched input, (1, 2560, 1024)) already gives the tensor variant C
(`gradcam`) needs — no new hook required for that. Verified on real BADAS: shape, row-sums=1,
inert when `capture_head_attention=False`, gradient reaches encoder layers 0 and 23 and the
predictor stack, none reaches the frozen head.

### Three loss variants (`aa_head_losses.py`, pure/model-free, unit-tested on synthetic tensors)
Token sets per window: **P** = relevant tokens (`R > 0` or the hindsight partner), **V** =
other-vehicle tokens, **B** = background. `token_sets_from_labels()` returns `None` (skip this
window's aux term) whenever P is empty, or `positives_only=True` and the clip is negative.
- **`rank_loss`** (RARE-style): `L = max(0, m - ln(rho_P/rho_V)) + max(0, m - ln(rho_P/rho_B))`,
  `rho_S` = mean attention received over token set S. A term is exactly 0 (no gradient) once its
  margin is met — cannot over-push.
- **`mass_loss`** (FAX-style): `L = -ln(sum_k p_k * R_hat_k)`, `p` = attention over the 2048 real
  tokens renormalized to sum to 1, `R_hat` = soft relevance / its max. Always positive-valued;
  guarded by the on/off lambda schedule and `attention_entropy()` (logged, not optimized) against
  collapsing all attention onto one patch.
- **`gradcam_loss`** (GAIN/CAMAL-style): `c_k = ReLU(sum_d (dCrashLogit/dX)_kd * X_kd)`,
  normalized to [0,1] by its own max; `L = mean(c over V∪B) - mean(c over P)`. **Always
  positives-only** regardless of `--aux-label` (shapes crash EVIDENCE, must never see a no-crash
  window). Computed via `torch.autograd.grad(crash_logit, X, create_graph=True)` — a cheap
  second-order term since `crash_logit` depends on `X` only through the small frozen head, not
  the full ViT-L stack.
  **Bug fixed 2026-09-25**: must differentiate w.r.t. `badas._captured["patches"]` (the raw,
  still-batched tensor), NOT the `patches` variable `forward_clip()`/`forward()` return
  (`patches[0]`, a slice created AFTER the unsliced tensor already fed the head — a sibling node
  in the autograd graph, not an ancestor of `crash_logit`; calling `autograd.grad` against it
  raises "not used in the graph"). `gradcam_loss` already squeezes a batch dim, so passing the
  raw (1,2560,1024) tensor is correct as-is.
- `lambda_schedule(epoch_frac, lam_max, warmup_frac=0.5, on_epochs=3.0)`: ramps 0->max over the
  first half of epoch 1, holds through epoch 3, then drops to exactly 0 (HASTE/REPA's early-stop
  finding — the aux gradient helps early and becomes a brake later; matches this project's own
  aux-vs-crash cosine drift in the original Stage AA runs).

### `semsup_train.py` wiring
`--aux-mode` gains `attn_rank`/`attn_mass`/`gradcam` (the old `occ`/`rel` side-probe modes are
untouched, byte-identical). New: `--aux-label {R_all,R_pos,partner_pos}` (which token set backs
P — see DECISIONS.md for why R is used only as WHERE to look, never as a crash signal),
`--aux-margin`, `--aux-schedule {constant,warm_on_off}` (default `warm_on_off`),
`--aux-warmup-frac`, `--aux-on-epochs`, `--aux-label-shuffle` (Stage 3 control: redirects each
window's label lookup to a different window's, via a fixed seeded derangement,
`build_aux_shuffle_map()`). `load_aux_h_arrays()` resolves a window's (P, V, B, soft-rel-tensor)
given the chosen label; `load_aux_h_arrays`/`token_sets_from_labels` treat a missing/empty label
as "skip this window's aux term", never as an error.

**Bug fixed 2026-09-25 (found LIVE on the pod, not caught by local testing)**: the
crash-vs-aux gradient-cosine probe block (`epoch_metrics.jsonl`'s `aux_grad_norm_crash`/
`aux_grad_norm_aux` — what the lambda pilot reads) was gated to `aux_head is not None`, which is
**always False** for the new AA-H modes (they have no trainable projector). Widened to
`(aux_head is not None or args.aux_mode in AA_H_MODES)`. This surfaced a SECOND bug: whenever a
window's aux term is skipped (e.g. `--aux-label R_pos` on a negative clip — roughly half the
pool), `aux_loss` is a disconnected `torch.tensor(0.0)` with no `grad_fn`; the probe tried to
differentiate through it anyway, crashed, and **permanently disabled itself for the rest of the
epoch** starting from the first such window. Fixed by requiring `aux_loss.requires_grad` before
the probe attempts anything (skips that window's diagnostic only, not a permanent failure). Real
consequence before the fix: R_pos's first pilot got `aux_grad_cos_n_sampled=0` all epoch and
silently fell back to an uncalibrated `lambda=1.0`, which then trained for a full 8 epochs
before being caught — see EXPERIMENTS.md.

### `aa_head_attention_diag.py` (new)
Stage-0 measurement script: for a given checkpoint (base BADAS-Open, or base + a LoRA adapter),
measures `rho_P/rho_V`, `rho_P/rho_B`, attention entropy, and the share of attention landing on
the 512 predictor ("future") tokens, over a sampled set of windows. Used to (a) set
`attn_rank`'s margin from a real measurement (`current ratio + ln(1.5)`, per the plan — **only
run at 19-window scale so far, not the planned 200**, see PROJECT_STATE.md's disclosed
approximation), (b) check post-hoc whether a trained checkpoint's attention on relevant cars
actually moved. Reuses `TrainableBadasWrapper(..., capture_head_attention=True)` and the same
train/val split convention as `aa1_token_probe.py`.

### `aa4_partner_labels.py` (new) — the hindsight crash-partner label (`--aux-label partner_pos`)
For each positive video, decodes a short extension past AA.2's cached span (which stops at
`t_event - 0.5s`, since BADAS's own windows never look closer) out to `t_event + 0.3s`, runs
YOLOPv2 detection on it, tracks it with a minimal greedy IoU tracker
(`track_extension()` — a fresh LOCAL track-id space, NOT the cached stitched tracks' ids),
**re-identifies** each local track against the cached tracks' last-known boxes at the boundary
(`reidentify_tracks()`, IoU-matched), and picks the partner as the re-identified track with the
largest box still visible near the very end of the extension (ties broken by proximity to the
last-known ego path). Writes a `partner` array into each of that video's EXISTING window `.npz`
files (`aa4_token_labels.py`'s own output — `rel`/`occ`/`box_h`/`path_gap` untouched), using the
SAME box-to-token machinery restricted to the partner tid.

**Bug fixed 2026-09-25**: `track_extension()`'s local tracks were matched against the
REQUESTED extension start time (`t_ext_start`) with a `1e-6` float tolerance to decide "did this
track start in the first extension frame" — but `decode_frames()` snaps to the nearest real
frame, landing up to ~1/fps (33ms) later than requested. This silently excluded EVERY local
track from re-identification (0/70 matches, no error) on the first real video tested. Fixed by
having `track_extension()` return `first_frame_tids` (an index-based flag: "did this track get a
box in frame index 0", set during tracking, never compared as a float) instead of comparing
timestamps after the fact. Verified on 4 real videos post-fix (17-32 re-identified tracks each).
At pool scale (543 positive videos, run locally, ~30 min): 498 found a partner, 88.6% of windows
got a usable label (>> the plan's 60% gate).

### Files that matter (Stage AA-H)
| Path | Purpose |
|---|---|
| `student_training/scripts/aa_head_losses.py` | The 3 loss formulas (rank/mass/gradcam), `token_sets_from_labels`, `mean_attention_received`, `attention_entropy`, `lambda_schedule`. Model-free, unit-tested (`test_aa_head_losses.py`, 28 synthetic checks). |
| `student_training/scripts/aa_head_attention_diag.py` | Per-checkpoint attention-mass diagnostic (margin measurement + post-hoc "did the mechanism move" check). |
| `student_training/scripts/aa4_partner_labels.py` | L3 hindsight crash-partner label generation — writes `partner` into existing `aa4_token_labels.py` `.npz` files. |
| `outputs/aa_head_attn/` | Stage 1 (`AA-ctrl-seed{1,2}/`) and Stage 2a (`AA-H-rank-{R_all,R_pos,partner_pos}/`, each with `pilot/`, `train/`, `scores_public/`) — same schema as `outputs/a1_compress256/` and `outputs/aa_token_aux/`. Large per-epoch LoRA adapters stay pod-only; only metadata synced locally. |
| `run_stage1.sh`, `run_stage2a_1_2.sh`, `run_stage2a_3plus.sh` (on the pod, `outputs/aa_head_attn/`) | Self-contained driver scripts: pilot -> compute lambda* (0.3x gradient-norm target) -> full 8-epoch run + private test scoring -> public scoring, chained across labels/seeds, nohup+disowned so they survive SSH disconnects. `run_stage2a_3plus.sh` additionally aborts a label's full run if its pilot gets 0 gradient samples (guards against a repeat of the lambda=1.0 bug). |
