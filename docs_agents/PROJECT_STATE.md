# Project State

## ⚠️ 2026-09-14 status update — NEW CHAMPION, read this first

**Every arm in this file (A0=0.853 through every semantic-supervision arm below) was
measured under a preprocessing bug: the model saw only the center ~49% of each frame's
width, not the full frame.** Fixed. The straightforward fix alone — no architecture change,
no new data — beats every semantic-supervision effort in this document's entire history.

**What happened.** `preprocess_clip()` (`e4_stageA_badas_open_eval.py`) has, since its first
working version (2026-06-24), passed the raw 1280×720 frame straight into V-JEPA2's
`AutoVideoProcessor`, which by default **resizes the shortest edge to 292 then center-crops
256×256** — keeping only source `x∈[321,953]` of a 1280-wide frame (~49% of width, 88% of
height). Docs across this repo (`ARCHITECTURE.md`, `StageA_scorer/StageA_summary.md`, the
scorer's own docstring) claimed the opposite — "squash-resize, no crop" — describing BADAS's
released *inference* code, which the project never actually ran; the *training* code path
(which this repo does use) crops. Discovered while investigating why A1 misses cut-ins: its
two worst-missed real crashes both have the collision partner physically outside the crop.

**The fix and the result.** Added `--preprocess {crop, compress256}` to `preprocess_clip`,
the model wrapper, `semsup_train.py`, and all three scorers (`crop` = default, byte-identical
to every historical run — verified). Retrained A1's *exact* recipe (same 1,761-window pool,
same split, same hyperparameters, only `--preprocess compress256`) →
**`A1-compress256`: test AP 0.9128 (private) / 0.9096 (public)**, beating A1's 0.900/0.908.
**A1-compress256 is now the champion**, and `compress256` is now the project default for all
future work.

**Evidence, cleanest first:**
| Comparison | ΔAP | 95% CI | Verdict |
|---|---|---|---|
| A0 (frozen, **zero training**), crop vs compress256, private (677) | +0.0532 | [0.0356, 0.0727] | **excludes zero** |
| A0 (frozen), crop vs compress256, public (667) | +0.0331 | [0.0133, 0.0547] | **excludes zero, replicated** |
| A1-compress256 vs A1-recorded, private | +0.0137 | [-0.0015, 0.0292] | crosses zero, P=96.1% |
| A1-compress256 vs A1-crop, pooled private+public (1344) | +0.0071 | [-0.0034, 0.0180] | crosses zero, P=90.5-94.9% |

The A0 comparison (same frozen weights, zero training, one variable changed) is the
load-bearing evidence — clean, single-variable, both splits replicate. The fine-tuned
comparisons are directionally consistent everywhere and never once favour crop, but are
individually underpowered at n<1400 for a ~0.7-1.4pp effect — expected, not concerning.

**A planned same-environment control run (`A1-crop-rerun`) was started then deliberately
cancelled** (pod was paused mid-run; on review, it didn't gate the decision — the plan's own
fallback rule already said "adopt compress256 anyway" in the tie case, and **G1** — re-scoring
A1's *existing* checkpoint on this pod, ΔAP=0.0009 vs its historical number — already showed
this pod's environment doesn't inflate results). Full reasoning in
`outputs/a1_compress256/summary.md`.

**A correctness bug was found and fixed in the same session**:
`score_checkpoints_on_test.py`'s `NAME=NONE` ("frozen baseline") only skipped loading a new
adapter — it never reset the *previous* one. Scoring a real adapter then `NONE` in one
process silently re-scored the real adapter under the baseline's name (caught because A0
came back bit-identical to A1 — impossible). Fixed by snapshotting the zero-init LoRA state
and restoring it for every `NONE` arm (mirrors the pre-existing head-restore pattern).
Verified in isolation before use. **Any historical use of this script scoring a real adapter
alongside `NAME=NONE` in the same invocation should be treated as suspect** — check whether
the baseline number in that run matches a known-good frozen baseline before trusting it.

**Also resolved, in passing: the 2560-vs-2048 token question, open since August.**
Measured directly on the loaded `badas_open.pth`: `backbone.encoder` outputs 2048 real
tokens (8×16×16, confirmed via flatten-order + tubelet-grouping probes); BADAS's own
`backbone.predictor` is then called with those 2048 + 512 appended mask-tokens (2560 in) and
returns 512 predicted "future" tokens with no fixed pixel location, concatenated back to
`[2048 real, 512 predicted] = 2560` for the crash head. Both historical numbers were
correct, for different modules. **Not yet verified: the concat order** (real-then-predicted
assumed, not proven) — needed before any token-index-based masking (Stage AA) is built.

**What does NOT change**: the semantic-supervision null result itself. The crop bug was
common-mode across A0, A1, and every B-arm/semantic-supervision comparison in this document
(same logic as the already-documented 108-vs-72-adapter waste) — it shifted all of them
together, so it cannot explain the A-vs-B gap those experiments measured. A future semantic-
supervision arm, if one is ever run again, should now be built on `compress256`.

**Not done in this pass**: public-test support in the website's experiments detail page
(A1-compress256's public number exists but isn't surfaced there — pre-existing gap, not
introduced now); the concat-order verification above; `docs_agents/NEXT_LORA_PLACEMENT.md`
and `ARCHITECTURE_BLOCKS.md` need the 2560 resolution folded in.

Full record: `outputs/a1_compress256/summary.md` (all numbers, all bootstrap JSONs, the
cancelled-rerun reasoning). Plans:
`~/.claude/plans/CCP based BADAS/2026-09-13_Father-Plan-Stage-AA-BB.md` (the detection-guided
supervision program this gate unblocks) and its child plan
`2026-09-14_Child-Plan-Gate-AA0-v2-crop-vs-compress256.md`.

**Pod**: reconnect with a fresh IP/port each session (ask the user) — the persistent volume
(`/workspace/MMLM_AI`) survives pauses/resumes with all checkpoints intact; the Python
environment does not (reinstall every time, see the standing pod-state section below).

## Stage AA.1 — detection pipeline smoke test, IN PROGRESS (2026-09-14, same day, later)

Father plan's Stage AA.1 (detection → tracking → geometry → threat ranking on the 18
`val_e3a` clips, before committing to the full 4,446-window pass). Built and iterating
**locally** (not the pod) — a local CUDA-enabled PyTorch was installed this session
specifically for this (`torch==2.11.0+cu128`, replacing a CPU-only build; local GPU is an
RTX 1000 Ada, 6GB, ~0.6-0.9s/frame for detection — this measured rate is what informs the
full-run cost estimate in the father plan's D3).

**Script**: `student_training/scripts/aa1_detect_track_rank.py` (new). Stack: Grounding DINO
via `transformers` (`IDEA-Research/grounding-dino-tiny` — chosen specifically to avoid the
original repo's custom CUDA-op compilation) → `trackers.BoTSORTTracker` (Roboflow) → a
virtual-corridor geometry fallback (no CLRerNet wired up yet — see father plan's decision
D-lane, still open). See ARCHITECTURE.md for the module's functions/constants.

**Tested on 1 of 18 clips so far (00687, the known cut-in case) — repeated diagnosis, not yet
run on the other 17.** Two real bugs found and fixed along the way (both now in the script,
not just noted):
1. G-DINO's `post_process_grounded_object_detection` returns multiple overlapping candidate
   boxes per real object with no built-in NMS — added `sv.Detections.with_nms(threshold=0.5,
   class_agnostic=True)` before tracking.
2. **The load-bearing fix**: `BoTSORTTracker`'s default `high_conf_det_threshold=0.6` only
   lets high-confidence detections spawn brand-new tracks; lower-confidence ones may only
   extend an existing track. The actual cut-in vehicle on clip 00687 scored 0.39-0.65 across
   ~36 consecutive frames (visually confirmed correct, stable boxing throughout) and never
   got a track under the default — set `high_conf_det_threshold=BOX_THRESHOLD` (0.30, matching
   our own detector's cutoff) and it immediately started tracking correctly, continuously,
   for the full 93-frame window.
   - **A red herring on the way there, now reverted**: raising
     `minimum_iou_threshold_first_assoc` to 0.55 (from BoTSORT's default 0.2) was tried first,
     produced more tracks (3→6) but not the right one, and was traced to not be the actual
     cause — reverted before landing on the real fix above. Don't re-try that path.

**Current live issue, unresolved — the reason to read this before continuing**: with both
fixes, the dangerous object IS now tracked correctly and continuously, but still does **not**
rank #1 by threat (scores 0.197, well below the top of the list). Diagnosed cause: the virtual
corridor (a fixed trapezoid, ~154px wide near frame-bottom at this resolution) is too narrow
for this scene (a wide intersection) — the object's box only clips 51px into it, and because
the box already straddles the corridor boundary throughout the fit window (never approaches it
from clearly outside), the `ttc_lat_inv` closing-rate signal never triggers. The only
surviving signal is looming (`alpha=0.197`), not enough to rank above other objects. **This is
a geometry/ranking gap, not a detection or tracking gap** — those are now confirmed working.

**Decision needed from the user, not yet made** — three ways forward, presented but not
chosen:
- (A) Widen/recalibrate the virtual corridor (cheap, still a crude proxy)
- (B) Add an independent size/growth/proximity threat signal, not routed through corridor
  overlap at all (cheap, targets exactly this failure mode)
- (C) Skip ahead to real lane detection (CLRerNet) now instead of iterating the fallback

**Next step when resuming**: get the user's choice of A/B/C, apply it, confirm 00687's
acceptance test passes (true collision partner ranks #1), THEN run the smoke test on the
remaining 17 `val_e3a` clips (only 1 of 18 done), and only then proceed to AA.2 (the full
4,446-window pass, on the pod, given the measured ~0.7s/frame local rate makes that pass
infeasible on this laptop GPU).

## ⚠️ 2026-09-09 status update (read this first — the rest of this file predates it)

A `/project-review`-style audit ran 2026-09-06 (`reports/project_reviews/
2026-09-06_project_review.md`, gitignored/untracked — not in git history), followed by a
two-session remediation (commits `fcb91fb` 2026-09-08, `3f59ccc` 2026-09-09; not pushed —
the user pushes themselves). The rest of this file was written before that and is stale in
detail (git state, uncommitted-file list, "Next step") but not in substance — the
scientific conclusions below (A1=0.900 banked, semantic supervision null) still stand,
now with better-measured uncertainty. What changed:

- **Measurement integrity**: `--deterministic` (default on) added to `semsup_train.py`
  and `score_checkpoints_on_test.py` — the review found the SAME A1 checkpoint scored
  twice disagreed on 677/677 clips (max|Δ|=0.097, ΔAP=0.0009, the same size as the
  V12-vs-v12shuf headline effect) with nothing pinning inference numerics.
- **The recovery-family ordering (V10 > V12 > v12shuf > a1cont) is not established.**
  `paired_bootstrap_ab.py` (fixed 2026-09-09 to read the a1fail321 arms' score-file
  schema) now gives every pairwise ΔAP a 95% CI — see `EXPERIMENTS.md`'s "Bootstrap CIs"
  section for the full matrix and why even the CI-excluding-zero pairs can't be trusted
  without a same-checkpoint noise-floor replicate. **The null itself is not threatened**
  — an effect at/below the noise floor is what a null looks like — only the specific
  ranking is unsupported.
- **A new, default-off research capability exists**: `--sem-pooled-weight` attaches a
  second semantic loss term to `pooled` (the crash head's actual classifier input)
  instead of only the patch grid every prior arm used. This is architecturally the
  reason "captions add nothing" and "the channel to the classifier is too narrow" have
  been indistinguishable across every null this project has collected — see
  `docs_agents/ARCHITECTURE_BLOCKS.md` §5 (new doc, also written 2026-09-09) for the
  full derivation. **Not yet run** — staged for the next pod session.
- **The 2560-vs-2048 patch-token question is mostly resolved, locally, without a pod**:
  `nexar-ai/BADAS-Open`'s HF source turned out to be readable without gating. Confirmed
  `img_size=224` is dead for the real scoring path and the actual resolution is 256×256
  (measured directly); derived that the stock config forces exactly 2048 tokens, still in
  tension with this project's own 2026-08-13 runtime measurement of 2560 — needs one more
  pod-side shape print to fully close. See `ARCHITECTURE_BLOCKS.md` §1-2 and
  `NEXT_LORA_PLACEMENT.md`.
- Several silent-wrong-number bugs fixed across the website builders and
  `paired_bootstrap_ab.py` (previously undocumented, and unable to read the
  a1fail321 score files at all) — see the commit messages for `fcb91fb`/`3f59ccc` for
  the full list, and `experiments.html`'s comparison-table wrong-clip bug (browser-
  verified fix) in particular.

**Do not trust this file's "Git state" or "Next step" sections below without checking
`git log`/`git status` directly** — they describe the state before the above.

## Goal
MSc thesis: collision anticipation on Nexar dashcam clips via Teacher→Student distillation.
Accepted shipped baseline: InternVL3.5-4B-Flash student, test AP=0.762 (677 clips). Active
thread: **semantic-supervision** — test whether a language-derived auxiliary loss (caption
embedding alignment) improves BADAS-Open's (V-JEPA2) crash-prediction representation, while
keeping inference vision-only (no added cost/latency). Central question: does the semantic-aux
loss beat crash-only LoRA (**A1_1761**, test_AP=0.900) beat the frozen baseline (**A0**,
test_AP=0.853)?

## Implementation status

**A1_1761 (crash-only control) remains the champion: test_AP=0.900, AUC=0.904.** Beats A0
(0.853) by +0.047 — a real, standalone, publishable result, banked regardless of what happens
with the semantic-supervision question.

**A1-failure-recovery run (2026-08-29), the most recent and most decisive semantic-aux result:
A1's 0.900 survives training on its own failures intact, and semantic supervision is now
proven to work end-to-end for the first time in this project — it still doesn't transfer.**
321 windows (240 clips) A1 itself gets wrong at threshold 0.5, from the 1,761-pool; 4 arms
(crash-only control + 3 semantic variants: leaky-V10 / clean-V12 / V12-shuffled-within-class)
all initialized from A1's own LoRA weights, head kept **frozen** (deliberate — the counterpart
test to SemTest-200-v2's open-head test, isolating "does semantic supervision damage an
already-converged, correctly-calibrated head" from "does an open head change the calibration
story"). Real test-set (677 clips) numbers, via a new scorer (`score_checkpoints_on_test.py`,
see ARCHITECTURE.md) that reproduced A1's own 0.900/0.904 to 3 decimals as a validation check:
A1=0.8995/0.9034, v12 epoch-10=0.8972/0.9027 — **flat, within noise.** The semantic branch
demonstrably works now (retrieval@1 35-44% vs a 2.1% collapse control for v10/v12; the shuffled
control sits at ~0% for the entire run — the cleanest real-vs-scrambled separation this project
has produced), so this is a clean, mechanistically-explained non-transfer, not an artifact of a
broken predictor. Mechanism: crash/semantic gradient cosine on shared LoRA params sits at
−0.04 to +0.05, sign-flipping epoch to epoch, in all 3 semantic arms — near-orthogonal, not
opposed; captions are a lossy function of the same 16 frames the student already sees, not new
information. Full numbers, mechanism detail, and the resolved separate-LoRA-weight-zones
question: EXPERIMENTS.md / DECISIONS.md.

Results website: see WEBSITE.md.

**Every semantic-supervision attempt at the 1,761-window scale has lost** (B_1761-parallel,
B-v2, B-v3, B-v3-ext12, P1 two-stage — see EXPERIMENTS.md for the full table). Three
diagnostics ruled out the obvious explanations: information does reach the classifier's pooled
representation (B1 probe, 22× chance), the LoRA gradient reaches it at least as well as random
noise (P3, paired bootstrap), and the two objectives' gradients are near-orthogonal, not
opposed (cos≈0). **The per-clip diagnostic then localised the damage**: semantic arms recover
false alarms far better than A1 (B-v3 60.0% vs A1 21.5%) but recover missed crashes far worse
(20.0% vs 56.7%) — McNemar on A0's 30 test-set misses gives B-v3's correct set as a *strict
subset* of A1's (p=0.0026).

**⚠️ CORRECTION (2026-08-27, load-bearing — the previous leading hypothesis was wrong):**
The originally-recorded mechanism hypothesis ("V12 de-leaking removed class discrimination,
pulling YES/NO embeddings together") is **refuted by measurement**. Rigorous calibration
analysis on the 1,761-pool val set (348 windows) shows:
- AP/AUC are flat across every arm (A1 0.877, B-v1 0.875, B-v2 0.867, B-v3 0.874, P1 0.858 —
  spread inside CI width at this n). Cohen's *d* class separation is **flat-to-higher** for
  B-v3 (1.352) vs A1 (1.505) — not narrower as the old hypothesis required.
- What actually moves is the **optimal decision threshold**: A1's own best threshold is 0.812
  (not 0.5!), B-v3's is 0.173. Accuracy at each arm's *own* threshold converges to a narrow
  0.773–0.782 band; accuracy at the shared 0.5 cut is what manufactures the appearance of a
  "negative bias".
- Re-deriving the fixed/broken accounting at each arm's *own* calibrated threshold instead of
  0.5 makes B-v3's "31 broken crashes" (vs A1) collapse to **6**, and net goes to ≈0 for every
  arm — those crashes were never un-learned, they were sitting below an arbitrary cut.
- **Root cause, consistent with Kumar et al. (ICLR 2022, "Fine-Tuning can Distort Pretrained
  Features")**: the crash head (`temporal_processor`+`classifier`) is frozen in every 1,761-
  pool arm. LoRA moves the trunk's feature distribution; the head's decision boundary — fit to
  the *original* distribution — can't follow, so the damage shows up as a **calibration
  offset**, not lost ranking. This is exactly what motivated `--unfreeze-head` and SemTest-200
  below.
- **`still_wrong` formula was also wrong project-wide** (undercounted total-wrong by omitting
  newly-`broken` clips) — fixed everywhere to `still_wrong = baseline_wrong − fixed_FP −
  fixed_FN + broken`. `broken` is now also split into `broken_FP`/`broken_FN`
  (`add_vs_a1_summary_sheet.py`) — on val, ~100% of every arm's breakage vs A1 is TP→FN, 0% is
  TN→FP, but per the calibration finding above this asymmetry is *also* a threshold artifact,
  not a real class bias — do not re-read it as "semantic arms are FN-biased" without re-
  checking at calibrated thresholds first.

## SemTest-200 — controlled, small-scale, head-unfrozen experiment (2026-08-26/27)

Built specifically to test the frozen-head hypothesis above: 200 hand-curated windows (160
train/40 val, **one window per video** — eliminates the 1,761-pool's multi-window-per-clip
confound), A0-baseline-referenced 3-tier selection (`select_semtest200_recovery.py`: tier 1 =
FN near-boundary [0.3,0.5) RT-eligible, tier 2 = TP fill [0.5,0.85) lowest-score-first, tier 3
= FN wide (<0.3) highest-score-first for positives; FP near-boundary [0.5,0.7) then FP fill
[0.7,1.0) lowest-first for negatives — **negatives are 100% FP by design, no TN at all**).
4 arms, identical config except caption file: `vision` (crash-only control), `v10` (leaky V10
captions), `v12` (clean V12), `v12shuf` (V12 captions permuted within class — content-vs-
presence control). All 4: LoRA q/k/v r16 + **`--unfreeze-head --head-lr-mult 0.1
--clip-grad-per-group`**, lr 1e-4 cosine, 10 epochs, seed 0, semantic-weight 0.2 (raised from
the historic 0.05 — measured rel-pull at 0.05 was only 5–9%).

**Result: a clean, real null — but confounded, and the confound is now diagnosed.**
- All 4 arms land at val AUC≈0.49–0.50 (chance), val AP 0.515–0.542 (spread inside noise at
  n=40). Train AUC 0.85–0.87 — pure memorization, zero transfer to held-out clips, **including
  the vision-only control**. Since even the floor arm shows no real generalization, this run
  cannot yet distinguish "semantic doesn't help" from "nothing generalizes at this scale/pool".
- Paired per-clip delta (Δ_arm − Δ_vision on val, sign test + Wilcoxon): v10/v12/v12shuf all
  land within noise of each other (means −0.0043/−0.0046/−0.0041, all p>0.4) — **v12 ≈ v12shuf
  is the cleanest evidence in the whole thread that caption content isn't reaching the score
  at this scale**, independent of the confound below.
- **⚠️ The confound (code review, 2026-08-27): `--unfreeze-head` moved the head by <0.05%
  relative magnitude over 200 steps.** `head_state.pt`'s total L2 norm agrees to 4 decimal
  places across all 4 differently-trained arms; the final classifier bias moved by ~1e-6.
  Mechanism: head LR = 1e-5 (0.1× an already-modest 1e-4), further cosine-decayed toward 0
  alongside the trunk's LR, clipped to grad-norm 1.0/step. **The head was unfrozen in name,
  not in practice** — this run cannot yet distinguish "head-open semantic still doesn't help"
  from "the head was never really open". Not yet fixed or re-tested.
- LoRA itself trained fine for comparison (144/144 `lora_B` tensors nonzero, mean norm
  0.114–0.118 — real signal, since `lora_B` is zero-init).
- Full outputs: `outputs/semtest200/` (selection, captions, per-arm training results,
  `scores/{A0,vision,v10,v12,v12shuf}.jsonl`, `semtest200_arm_comparison.xlsx`,
  `code_review_findings_2026-08-27.md`, figures).

### Architecture literature review (2026-08-27, full report in `code_review_findings_...md`)
**Verdict: abandon trunk-level SigLIP-InfoNCE alignment as an accuracy-lift mechanism; retarget
language supervision to post-hoc explanation.** Two independent, literature-grounded reasons
converge with the measured evidence:
1. CLIP-style contrastive alignment (LiT/SLIP) has only ever worked at hundreds of millions of
   pairs — 5–6 orders of magnitude above this thesis's corpus.
2. SigLIP is a static-image, bag-of-words-ish text encoder (ARO/Winoground) — documented to
   discard exactly the relational/motion semantics ("closing distance") this task needs, and
   the shuffled-caption control (real ≈ scrambled ≈ none) empirically confirms no signal is
   being extracted, not merely a mis-weighted one.
Full references in the doc. Recommendation: keep the vision-only LoRA result + this negative-
result methodology as the reportable contribution; if one bounded confirmatory run is wanted,
swap SigLIP → a video-text encoder (InternVideo2) as a single capped follow-up, not a new arc.

## V13 caption redesign — full 4,446-window pool captioned, FAILED its own go/no-go (2026-08-27/28)

Motivated by two measurements: (1) SigLIP's real limit is 64 tokens, not the ~40-word V12 rule
— existing corpora never truncate (max 43 tokens across 2,161 captions), ~3× headroom unused;
(2) caption length vs SigLIP distinctiveness correlation on the existing V12 corpus is **−0.0017
(zero)** — more words of the *same kind* of content don't separate better. `PROMPT_SEMSUP_V13_
CAUSAL.py` (new) keeps V12's anti-leak machinery (blind, closed vocab, symmetric bans) and adds
5 closed-vocab causal-cue fields targeting information NOT trivially visible from raw pixels:
`lead_vehicle_lighting`, `ego_maneuver`, `road_geometry`, `signal_state`,
`occluded_or_peripheral`. Colour is banned from `caption_neutral`.

**⚠️ Also hit for real (not just a documented risk): `semsup_caption_promptbakeoff.py`'s stale
`DEFAULT_MODEL` bug.** Omitting `--model` on a SemTest-200 caption run silently used
`google/gemini-3.1-pro-preview` instead of the intended teacher. **Fixed**: default is now
`google/gemini-3.7-flash` (the current teacher). 22 mis-captioned + 27 not-yet-captioned
SemTest-200 clips were regenerated on the correct model before training (see git-tracked
`semtest200_captions_review.xlsx` history).

Full pool (4,446 windows) captioned via `gemini-3.7-flash`, pinned to the **Google Vertex
provider** (`--provider-order google-vertex`, `allow_fallbacks: False`) for a **75%-off launch
discount confirmed live on OpenRouter through at least 27–28/08/2026** (real billing, not a
price-list estimate — verify via a tiny paid call before trusting any future discount claim).
Real cost: **$24.85**, wall time 65 min at concurrency 16. 4,446/4,446 rows, 0 duplicates,
2,223/2,223 class balance. Outputs: `outputs/semantic_captions/v13/{raw_v13_4446.jsonl,
Caption_V13_Causal_4446_fortrain.jsonl, Caption_V13_Causal_4446.xlsx, leakage_gate_v13.json}`.

**QC results:**
- Leakage gate: AUC=0.7774 (up from V12's 0.764 — **expected and not automatically bad**: V13
  deliberately adds genuinely predictive facts like brake lights; top n-grams read as real
  causal signal — "brake lights", "distance decreasing" — not register leak).
- **⚠️ Decisive distinctiveness check FAILED, and the run does not pass its own pre-registered
  go/no-go**: mean cross-caption SigLIP cosine **rose** to 0.7974 (worse than V12's 0.7010);
  mean distinctiveness **fell** to 0.2026 (from 0.3003), **−32.5%**.
- **Root cause diagnosed, and it's fixable**: **96.9% of all 4,446 captions open with one of 3
  near-identical phrases** ("Ego moves straight...", "Ego travels straight...", "Ego remains
  stopped..." — 73.5%/16.8%/6.6%). The prompt's one worked example + "verbalize ego_maneuver
  ALWAYS" instruction caused template collapse, not genuine content homogeneity. SigLIP's
  bag-of-words sensitivity (see literature review above) means a 73.5%-shared prefix dominates
  the embedding regardless of what differs afterward.
- A first prompt-iteration bug was already caught and fixed mid-flight: the initial "≤45 words"
  rule was a ceiling with no floor, and "include ≥1 causal cue" (not "all populated fields")
  produced a mean of only 26.7 words / 30.4 tokens on a 15-clip gate — under half the budget,
  with fields recorded but never reaching the caption text. Fixed to a **42–52 word band (floor
  AND ceiling)** + a hard requirement to verbalize every populated field, with validator-side
  soft checks for word-count and per-field keyword coverage (`_stamp_token_len`, the
  `_COVERAGE` dict in `validate_parsed`'s `v13` branch).

**OPEN DECISION, not yet made by the user — the single most important next step:**
Fix the opener-template-collapse (vary sentence structure / ban repeating the worked example's
literal opening / provide multiple differently-structured examples), re-gate on 15 clips, and
only then decide on a full re-run (~$25 more, ~$50 total) — **or** stop here and treat the
failed distinctiveness check as a completed, reportable negative result (which would now be a
*third* independent piece of evidence, alongside the 1,761-pool results and the literature
review, that this specific SigLIP-alignment mechanism doesn't work). **Do not restart the full
4,446-window run without first re-gating the fix on 15 clips** (same STOP-gate discipline as
every other stage in this thread).

## What's genuinely still untested
- **SemTest-200-v2 with a real head LR — DONE (2026-08-29).** `--head-lr-schedule constant`
  fix landed in `semsup_train.py` (see below); SemTest-200-v2 (4 arms, 300-clip pool =
  original 200 + 100 easy A0-correct anchors, addressing the second bullet below too) re-ran
  and reproduced the same qualitative null (v12≈v12shuf, no transfer) — superseded in
  relevance by the A1-failure-recovery run above, which answers the same question via a
  cleaner, mechanism-explained route at a different design point (frozen head, 321-clip pool):
  retrieval demonstrably works, gradients are near-orthogonal, no transfer to test AP.
- **Mix easy/A0-correct anchor clips into SemTest-200's train+val — DONE**, via SemTest-200-v2's
  +100-easy-anchor 300-clip pool (see above). Result: same qualitative null as SemTest-200-v1.
- **V13 opener-diversity fix + re-gate** (see above — still open, not touched this session).
- **Concept-head supervision (2026-08-29) — the new pre-registered next direction, not yet
  run.** Predict the V13 caption schema's closed-vocab fields directly (`gap_trend`,
  `lead_vehicle_lighting`, `ego_maneuver`, `road_geometry`, `signal_state`) via small
  classification heads on the pooled embedding, instead of matching a whole caption's SigLIP
  embedding via InfoNCE. Rationale: those targets demand the same visual evidence collision
  prediction needs (is the gap closing, are brake lights on), so their gradients should align
  with the crash gradient rather than sit near-orthogonal to it — unlike whole-caption
  retrieval, which rewards scene-identity fingerprinting. Falsifiable success criterion
  (pre-registered, uses the already-instrumented `grad_cos` probe): grad_cos persistently
  above +0.15, vs the ±0.05 sign-flipping measured for whole-caption InfoNCE across every arm
  this session. If it fails that bar, trunk-level language-alignment-for-accuracy is a closed
  direction with three independent negative results (1,761-pool history, literature review,
  A1-failure-recovery) — redirect language supervision to post-hoc explanation instead.
- **Retention probe, B-shuffle at 1,761 scale, B-rev, λ sweep** — still on the list from before,
  now secondary to the above given the literature review's verdict. Note: the B-shuffle-style
  content-vs-presence control HAS now been run, just at the 321-clip A1-failure-recovery scale
  rather than the full 1,761 pool (v12shuf arm) — see EXPERIMENTS.md.

## Known bugs / gotchas (this thread, still open)
- `semsup_caption_promptbakeoff.py`'s `DEFAULT_MODEL` staleness is now **fixed** (see V13
  section) — moved to the "fixed" list below is not yet done, kept here as a live reminder
  that any script invocation MUST pass `--model` explicitly and verify it printed the intended
  model before trusting a run's captions.
- The three original caption files (V10/V12/587-failures) still use incompatible field
  conventions — **always join on `frames_dir`** (unique per window, consistent across all).
- `evaluate_val()`'s 5-tuple return (added `retrieval_stats`) — the one call site is current;
  don't unpack it as a 4-tuple in any new script.

## Current RunPod pod state
- Working repo: `/workspace/MMLM_AI` on the **persistent network volume**
  (`mfs#euro.runpod.net:9421`) — survives across different pod instances/containers, confirmed
  again this session across 2 more reconnects.
- **Every reconnect is a fresh container — pod IP/port changes every time, ask the user for
  the current one, and reinstall packages from scratch**: `pip install --break-system-packages
  huggingface_hub transformers peft safetensors scikit-learn pyyaml pillow openpyxl
  opencv-python-headless einops timm albumentations psutil sentencepiece protobuf`. Restore HF
  auth: `mkdir -p /root/.cache/huggingface && cp /workspace/.cache/huggingface/token
  /root/.cache/huggingface/token`. If SSH refuses, the user pastes the **existing** public key
  (`~/.ssh/id_ed25519.pub` — never generate a new one) into the pod's web terminal's
  `~/.ssh/authorized_keys`.
- **Last known state: pod stopped** (confirmed unreachable end of this session; not needed
  again until a SemTest-200-v2 or further 1,761-scale training run).
- All prior checkpoints (`a1_1761`, `b_v2_1761`, `b_v3_1761[_ext12]`, `p1_stageA`/`p1_stageB`,
  `semtest200/{vision,v10,v12,v12shuf}`) remain on the volume regardless of pod state.

## Important commands
```bash
# SemTest-200 training (identical across all 4 arms except --captions-path/--semantic-weight)
HF_HOME=/root/.cache/huggingface python3 semsup_train.py \
    --config ../configs/e4_stageA.yaml --lora-target-modules query,key,value \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --unfreeze-head --head-lr-mult 0.1 --clip-grad-per-group \
    --lr 1e-4 --lr-schedule cosine --warmup-frac 0.05 --epochs 10 --keep-top-k 10 --seed 0 \
    --val-video-ids /workspace/semtest200_data/val_vids.txt \
    --captions-path /workspace/semtest200_data/Caption_semtest200_V12.jsonl \
    --semantic-weight 0.2 --semantic-loss infonce --infonce-tau-init 0.07 \
    --dump-val-scores --grad-cosine-every 8 --out-dir /workspace/semtest200/v12

# Score a SemTest-200 checkpoint (needs --head-state if the run used --unfreeze-head)
python3 score_semtest.py --config ../configs/e4_stageA.yaml \
    --captions-path <any-semtest200-caption-file> \
    --lora-adapter <out-dir>/epoch_XX/lora_adapter --head-state <out-dir>/epoch_XX/head_state.pt \
    --arm-name v12 --out outputs/semtest200/scores/v12.jsonl
    # omit --lora-adapter and --head-state for the frozen A0 baseline

# Rebuild SemTest-200 clip selection (3-tier, respects RT-eligibility, one window/video)
python3 select_semtest200_recovery.py --a0-scores <A0_full4446.jsonl> \
    --manifest ../../dataset/manifests/train4500_hires.jsonl \
    --train-xlsx ../../dataset/train.xlsx --out-dir ../../outputs/semtest200 \
    [--exclude-frames-dir <qc_excluded.txt>] [--tp-fill-max 0.85]

# Caption a new corpus (ALWAYS pass --model explicitly, verify it printed correctly)
python3 semsup_caption_promptbakeoff.py \
    --manifest <manifest.jsonl> --frames-root ../../dataset/train --out <out.jsonl> \
    --prompt v13 --model google/gemini-3.7-flash --provider-order google-vertex \
    --token-cap 58 --concurrency 16
    # --provider-order pins a specific OpenRouter provider (allow_fallbacks=False) - different
    # providers serving the SAME model slug can be 2x+ apart in price; verify via a tiny paid
    # call before trusting any discount claim

# Caption leakage gate (persisted, reuse for any new corpus)
python teacher_distillation/scripts/caption_leakage_gate.py \
    --captions <corpus.jsonl> --caption-field caption --label-field gt_verdict \
    --positive-value YES --out <out.json>

# Per-clip arm comparison workbook (1,761-pool, corrected still_wrong/broken_FP/broken_FN)
python student_training/scripts/build_pool1761_comparison.py \
    --scores-dir outputs/e4_vjepa_reason/pool1761_scores \
    --out outputs/e4_vjepa_reason/pool1761_arm_comparison.xlsx
python student_training/scripts/add_vs_a1_summary_sheet.py \
    --xlsx outputs/e4_vjepa_reason/pool1761_arm_comparison.xlsx

# SemTest-200 results workbook + curves (all local, no pod needed)
python student_training/scripts/build_semtest200_comparison.py
python student_training/scripts/plot_semtest200_curves.py
```

## Git state
Branch `main`. Local commits since `5b40076`: `f91b4a6` (P11 hypothesis + docs), then a run of
website commits this session (`a42ebfc`, `8503609`, `54a962c`) — **only `website/` was
committed/pushed this session**; ask the parent session for exact push status if unsure.
**All of the A1-failure-recovery + SemTest-200-v2 infrastructure work is still uncommitted**:
`semsup_train.py`/`semsup_common.py` (`--unfreeze-head`, `--val-video-ids`, `--dump-val-scores`,
param-group optimizer, plus this session's `--head-lr-schedule {cosine,constant}`,
`--bank-captions`, and an unrelated pre-existing argparse `%%`-escaping fix),
`semsup_caption_promptbakeoff.py` (v13 prompt wiring, `--provider-order`, `--token-cap`,
`DEFAULT_MODEL` fix), new prompt `prompts/PROMPT_SEMSUP_V13_CAUSAL.py`, and ~20 new
`student_training/scripts/*.py`/`*.sh` files from this session — selection/merge/scoring/
plotting/presentation scripts for SemTest-200-v2 and A1-failure-recovery (`select_a1fail321.py`,
`run_a1fail321_4arms.sh`, `build_a1fail321_comparison.py`, `build_a1fail_presentation.py`,
`score_checkpoints_on_test.py`, `score_semtest.py`, `select_semtest200_recovery.py`,
`select_semtest200_easy.py`, `merge_semtest200_v2.py`, `merge_semtest200_v2_captions.py`,
`make_semtest200_folds.py`, `make_semtest200_shuffled.py`, `aggregate_semtest200_cv.py`,
`build_semtest200_comparison.py`, `plot_semtest200_curves.py`, `plot_semtest200_cv_curves.py`,
`add_vs_a1_summary_sheet.py`, `siglip_bottleneck_probe.py`, `build_pool1761_comparison.py`,
`score_checkpoints_on_test.py` — see ARCHITECTURE.md's files table for what each does). User
pushes/commits themselves — do not `git push`; commit only if explicitly asked.

## Next step
**Decision point — do not guess, ask the user.** Immediate fork, unchanged from before this
session (not resolved yet):
1. **Fix V13's opener-template-collapse** (vary sentence structure, kill the copied worked-
   example opener), re-gate on 15 clips, decide on a full re-run from there. Or:
2. **Stop the V13 line here** — treat the failed distinctiveness check as a completed negative
   result reinforcing the literature review's verdict.

**Resolved this session**: SemTest-200-v2 with a real head LR has now been run (see above) —
still a null. The new highest-priority open direction is **concept-head supervision** (predict
V13's closed-vocab causal-cue fields directly via small classification heads, instead of
whole-caption InfoNCE retrieval) — pre-registered, falsifiable via the `grad_cos` probe
(threshold +0.15), not yet run. See "What's genuinely still untested" above and DECISIONS.md.

Also outstanding: commit the substantial uncommitted training/analysis-script work (see Git
state above) — not done yet, user has not asked for it.

## Known bugs / gotchas (all fixed — don't re-hit these)
- **`| tail -N` on a backgrounded command masks the real exit code.** Redirect straight to a
  file (`> log.txt 2>&1`) and check `$?` explicitly.
- **Two BADAS-loading processes running concurrently can silently crash one of them** (shared
  on-disk HF cache contention). Run sequentially, not in parallel, when both load BADAS.
- **`peft`'s `save_pretrained()` crashes on BADAS** unless
  `create_or_update_model_card = lambda *a, **k: None` is stubbed.
- **Test scoring used the LAST epoch, not the best one** — fixed via `set_peft_model_state_dict`
  reload before scoring.
- Raw `NaN` in JSON on degenerate runs — all NaN-prone fields now emit `null`.
- Windows console (cp1255) crashes on emoji `print()` — always `PYTHONIOENCODING=utf-8
  PYTHONUTF8=1`.
- RunPod `/workspace` has a per-user quota far below cluster-wide free space — verify per-pod.
- Never `pip install -U torch` on a provisioned RunPod image.
- BADAS needs `albumentations`, `opencv-python-headless`, `psutil`, `einops`, `timm`; SigLIP's
  tokenizer needs **`sentencepiece` AND `protobuf`** — install ALL of these on every fresh pod
  container, not just the ones a first error message names (confirmed twice more this session:
  a fresh container is missing every one of them, and they surface as a chain of one-at-a-time
  `ModuleNotFoundError`s if installed reactively instead of up front).
- `SiglipModel.get_text_features()` returns `BaseModelOutputWithPooling`, not a tensor — handle
  defensively (`siglip_text_embed()` already does).
- `mv` across filesystems can leave 0-byte stubs on quota failure — `cp` first, verify, then
  remove the source.
- Backgrounded output piped through `tail` shows empty until exit — use `python -u`.
- **OpenRouter `preview`-tagged model aliases are not stable snapshots** — never trust a
  historical baseline for a `preview` model without a same-day reproducibility check.
- **openpyxl conditional-formatting (dxf) fills render from `bgColor`, not `fgColor`** — using
  `fgColor` produces a rule that matches correctly but paints nothing in Excel. Confirmed via
  Excel COM screenshot; re-applied correctly in every new workbook script this session.
- **A naive substring scan for banned words false-positives on words containing the banned word
  as a substring** (e.g. scanning for "tan" hits inside "dis-TAN-ce"/"cons-TAN-t") — use
  word-boundary regex (`re.search(rf"\b{w}\b", text)`), not `w in text`. Caught on 15/15 of a
  V13 gate run with zero real hits; would have produced constant false-positive noise at scale.
- **A closed-vocabulary enum value that overlaps a globally-banned word list silently can't be
  verbalized** — V13's `lead_vehicle_lighting` enum originally included `hazards_on`, but
  "hazard" is on the caption's own banned-outcome-word list, so the model could never legally
  write that fact into the caption it was required to populate. Renamed to `flashers_on`.
  General lesson: any prompt with both a banned-word list AND a required-vocabulary list must
  check the two don't collide.
- **A caption-length prompt rule needs a FLOOR, not just a ceiling, or the model converges to
  the minimum.** "≤45 words" alone produced a measured mean of 26.7 words. A worked example
  also gets copied as a literal template far more than intended — one example produced a 73.5%-
  shared 3-word opener across 4,446 independently-generated captions. Any future dense-caption
  prompt needs an explicit word floor AND either no single canonical example or several
  differently-structured ones.
- **Tar writes correct file count but 0-byte content when a disk quota is hit mid-stream** —
  always additionally check `os.path.getsize(f) > 0` per file, not just directory listing.
- **`open(path, 'w')` on Windows writes `\r\n`** — file lists fed to `tar -T` need
  `open(path, 'w', newline='\n')`.
- **A single benchmark against a fresh `--out` file doesn't dedupe against earlier test runs**
  (resume-skip is per-output-file, not global) — use non-overlapping `--limit` ranges or a
  shared test file when benchmarking a resumable captioning run.
- **`semsup_train.py`'s argparse help strings crashed `--help` entirely** (pre-existing, found
  and fixed 2026-08-29): unescaped `%` from adjacent string-literal concatenation produced
  runtime content like `"...0.53%" + "of..."`, which argparse's own `%`-style formatting then
  choked on. All now `%%`-escaped.
- **`aggregate_semtest200_cv.py`'s `metrics()` read a `gt_verdict` key that doesn't exist** in
  `--dump-val-scores` output, which actually uses `label` (int 0/1) — fixed 2026-08-29.
