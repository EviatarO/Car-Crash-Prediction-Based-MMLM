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
checkpoints for speed. Uses `softmax(logits)[0,1]` with **no `/2.0` divisor**, unlike
`e4_stageA_badas_open_eval.py`'s published-scorer convention — confirmed this does not affect
AP/AUC or the confusion matrix at threshold 0.5 (dividing logits by a constant is monotone,
preserving the 0.5 crossing); it would only matter for calibration metrics at other thresholds.

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
| `docs_agents/ARCHITECTURE_BLOCKS.md` | Block-by-block reference (shapes/equations/frozen-status) for the architecture diagram. Also covers the pooled-tap experiments and the gradient-angle diagnostic (§5b/7c). |
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
| `student_training/scripts/score_checkpoints_on_test.py` | Loads BADAS once, swaps LoRA adapters between checkpoints, scores each on the 677-clip test set. `softmax(logits)[0,1]`, no `/2.0` divisor (see the A1-failure-recovery section for why this doesn't affect AP/AUC/CM@0.5). |
| `student_training/scripts/build_a1fail_presentation.py` | 6-slide deck for the A1-failure-recovery result, house style matching the `2026-08-22` deck; `verify()` re-derives every number from score files before writing. |
| `student_training/scripts/make_semtest200_folds.py` | Stratified 5-fold split of the SemTest-200-v2 pool by `video_id` + source tier; self-asserts an exact partition. |
| `student_training/scripts/aggregate_semtest200_cv.py` | Pools per-fold val_scores from SemTest-200-v2 into a full-pool readout. |
| `student_training/scripts/select_semtest200_easy.py` | Selects 100 easy A0-correct anchor clips to add to the original 200-clip SemTest-200 pool (addresses its 100%-adversarial/zero-TN composition). |
| `student_training/scripts/merge_semtest200_v2.py` | Merges the 200-clip pool + 100 easy anchors into the SemTest-200-v2 300-clip pool. |
| `student_training/scripts/merge_semtest200_v2_captions.py` | Joins caption corpora for the merged SemTest-200-v2 pool. |
| `student_training/scripts/plot_semtest200_cv_curves.py` | Mean±std-band loss curves across SemTest-200-v2 folds; shared y-axis; right axis color-keyed to its own series; `--mark-epoch`/`--init-note` for annotating a selected checkpoint. |
| `student_training/scripts/siglip_bottleneck_probe.py` | Measures how much crash-relevant signal survives text→SigLIP-embedding vs raw text; ran on V10/V12/V13 — SigLIP retains 86-96% of the text's own crash-AUC, ruling out the encoder as the bottleneck for prior negative results. |

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
