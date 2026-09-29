<!-- handoff-month: 2026-09 -->
# Project State

## ⚠️ 2026-09-30 — 1.5s-TTE plan in progress: Stage 0 passed, Stage 1 (loss weighting) fails, Stage 2 (Look-Ahead head) coded, NOT launched

**Read this block first.** Plan file: `~/.claude/plans/wiggly-bubbling-newell.md` (Q&A + stages).
Numbers for everything below: EXPERIMENTS.md "2026-09-29/30" entries.

- **Midpoint negatives, 5-seed confirmation (seeds 3,4 added, epoch 1 pre-registered):** pooled AP
  +0.0092 ± 0.0052 vs control, **5/5 seeds** (sign test p=0.03) → confirmed. Kaggle mAP +0.0037,
  4/5 → **not confirmed** (1.5s AP −0.007, 2/5). Claim wording: "raises pooled AP and cuts false
  alarms at matched recall; no reliable Kaggle mAP gain".
- **Diagnosis of the chart "TP drop" (A1 → A1-compress256 → midneg):** it is a score/threshold
  shift, not lost detection. At a matched false-alarm rate all arms find the same TP per TTE. Real
  weakness: at FPR 10% only ~50% of 1.5s crashes are caught by any arm.
- **Stage 0a (same-video test check): gate passes.** Of 1.5s crashes missed at FPR 10%, 94% are
  caught by the same video's later (1.0s or 0.5s) clip → evidence exists later in the video.
- **Stage 0b (frozen features, 1,761-pool videos, balanced pairs 1,086 pos / 1,128 neg): gate
  passes.** Predicting the +0.5 s vector beats "copy" for crashes (error ratio 0.71/0.73), equals
  copy for normal driving (1.05/1.00 — no invented danger). Frozen head AP on 1.5s windows: 0.769
  now, 0.794 on the predicted future, 0.894 on the true future (oracle ceiling).
- **Stage 1 (horizon-weighted crash loss, the cheap baseline Stage 2 must beat): weak, not significant.**
  Crash-only weights (seeds 0,1): no 1.5s gain, scores shift up (more FP). Symmetric weights
  (both classes by horizon, user's design), 3 seeds, epoch 1: 1.5s AP +0.0042 (per seed −0.0002 /
  +0.0093 / +0.0035, sd 0.0048, t≈1.5, n.s.), Kaggle +0.0019 (2/3), FPR@85 −0.003; epoch 2 +0.0043;
  epoch 3 ≈0. Passes the literal pre-registered rule but is inside seed noise → treat as the
  baseline bar (Stage 2 should also be compared against it). Results pulled to
  `outputs/stage1_horizon_weights_2026-09-29/sym/`.
- **Stage 2 (Look-Ahead head) — code done and committed, NOT run.** See ARCHITECTURE.md. Driver:
  `outputs/stage2_lookahead_2026-09-30/run_stage2.sh` (pilot → la-full + la-shuf per seed 0-2 →
  la-auxonly). **Before launch:** scp `outputs/lookahead_0b_merged/features.npz` to
  `/root/lookahead_features.npz` on the pod. Pass rule pre-registered in DECISIONS.md.
- **Pod / volume constraints (new):** network volume 0hnvco2s4j is AT ITS 56 GB QUOTA — writes to
  /workspace fail. All new run outputs go to the pod's container disk (/root, 40 GB) and are pulled
  before stop. `runpodctl` on pods needs `RUNPOD_API_KEY` exported from /proc/1/environ (drivers do
  this now). Pod capacity is often unavailable — retry loop over many GPU types.
- **Git:** main has local commits since a9da412 not yet pushed by the user (e2378ac … 9a170a7).
- **Next step:** fold Stage 1 symmetric seeds 1-2 into the table (`stage_compare.py`), then launch
  Stage 2 (~6.5 h, ~$4.7) — start with pilot + seed-0 arms and STOP to review the pilot's
  per-horizon gradient cosine before the rest.

## 2026-09-29 addendum — seed consistency understood; website partially updated

- **Midpoint-negatives seed spread is a calibration offset, not a detection difference.**
  midneg-seed2 epoch 1 (its val-selected checkpoint, the one on the website) misses 265/672
  crashes at threshold 0.5 vs 100-153 for most other checkpoints, yet has the 2nd-best AP of the
  9 (0.9193): every score is shifted down (median crash score 0.67 vs 0.86-0.97 elsewhere). Its own
  epoch 3 has the FEWEST misses of all 9 (79) at the same AP. 1.5s TTE is hit hardest because those
  positives score closest to 0.5. Full seed x epoch x TTE confusion-matrix table: EXPERIMENTS.md.
- **Presentation rule (agreed):** report every seed at the SAME epoch, fixed in advance (or by val),
  as mean ± sd across seeds, plus mean-over-epochs as robustness; never each seed's test-best epoch.
  Keep threshold 0.5 as the standard for CM/P/R/F1/Acc (always stated); AP/Kaggle mAP are
  threshold-free and unaffected. No threshold calibration in reported results.
- **Website state:** Cross-Experiment Comparison page now has midneg-seed{0,1,2} and
  fullpool-seed{0,1,2} (677-clip private set, each at its own val-selected epoch — mixed epochs,
  see TODO). Supports the 1.5s-failure review: filter Window=TTE 1.5s, verdict=wrong, play clip.
  Picker-list contrast bug fixed. **Deferred website TODOs:** (1) detail pages for the 6 overnight
  arms (EXPECTED_CM / EXPECTED_CM_PUBLIC entries already in build_experiments_data.py, ARMS entries
  not written); (2) landing "Test-set comparison" table: add experiment-date column + click-to-sort
  on every column (dates = result-file mtimes, listed in EXPERIMENTS.md); (3) point comparison arms
  at ONE shared epoch per family instead of per-seed val-selected; (4) optionally show per-TTE AP.
- **Git:** main clean, in sync with origin (a9da412 pushed).
- **Pods:** nothing running. Two stopped pods remain on the account (egx54zfwpmpasg
  "controls-public-score", maeqpipl372s77 "pull-results"); 5ivn7also2b9u9 was deleted. All data is
  on network volume 0hnvco2s4j (EU-RO-1). Pod resume often fails ("not enough free GPUs on the host")
  — creating a fresh pod on the volume with a broad `gpuTypeIds` list worked first try.
- **Next step:** pick the 1.5s-TTE work (DECISIONS.md "Unresolved (new, 2026-09-27)"), starting with
  the local diagnosis of midneg's 1.5s misses via the comparison page / pointing-game heatmaps.

## ⚠️ 2026-09-27 — Overnight run DONE: midpoint negatives confirmed a real gain; 1.5s TTE is the new priority

**Read this block first.** Two crash-only variants, 3 seeds x 3 epochs each, evaluated on the
pooled 1,344-clip set (private 677 + public 667), paired against matched-seed/epoch controls:

- **Midpoint negatives** (re-cut training negatives to match the Nexar test protocol — fake event
  at video midpoint + noise, 3 sub-windows 0.5/1.0/1.5s before it, same 564 videos/905 windows,
  only the cut point changes) — **mean AP +0.0082, 9/9 seed x epoch pairs positive.** Public-set
  FP at threshold 0.5 down 34% (156→104 avg), specificity +8pt. **Confirmed real, not noise.**
- **Full pool** (4,446 windows instead of the curated 1,761, unchanged negative sampling) —
  **mean AP −0.0091, 8/9 pairs negative.** More data with the OLD sampling does not help; the
  win above is specifically from fixing the sampling mismatch, not pool size.
- **New weak point found:** midpoint negatives IMPROVE 0.5s/1.0s TTE sharply but slightly HURT
  1.5s TTE (0.8852→0.8705). On overall AP the net gain is +0.008; on the officially-weighted
  Kaggle mAP (equal weight per horizon) it shrinks to +0.001, since the 1.5s loss cancels most of
  the gain. **1.5s TTE is now the weakest bucket for every arm tried (0.87-0.89 vs 0.91-0.94) and
  the highest-value next target** — see DECISIONS.md's "Unresolved (new, 2026-09-27)".
- **Current best result vs literature (pooled 1,344 clips):** ours (midpoint neg, mean of 9) =
  AP 0.916 / Kaggle mAP 0.912, vs BADAS-Open (paper) 0.86, BADAS-1.0 (40k videos, paper) AP 0.91 /
  mAP 0.925. We reproduce BADAS-Open almost exactly (0.861), which calibrates this comparison.
  **Not yet publication-ready as a headline number** — needs the epoch fixed in advance (not
  picked from test-informed prior runs) and confirmed on 2+ more fresh seeds; see DECISIONS.md.
- **Do NOT report the single best checkpoint (midneg-seed0-ep1, AP 0.9237)** — it's 1 of 9 draws,
  within ordinary seed noise. Report the mean with its spread.
- Full tables (CM, per-TTE, per-epoch-per-seed breakdown): EXPERIMENTS.md's "Overnight run
  2026-09-27" entry. Driver/results: `outputs/overnight_2026-09-27/`.
- **Pod state:** nothing running or billing. Lesson learned and saved to memory
  (`pod_results_download_before_stop.md`): future driver scripts must sync results to local BEFORE
  calling `runpodctl stop pod`, to avoid the resume-just-to-fetch-results cycle hit twice this run.
- **No architecture change anywhere this project** — LoRA merges into existing weight matrices at
  inference; every gain to date is preprocessing/data/sampling, not a new module.

## ⚠️ 2026-09-26 (later) — CORRECTION after pooled 1,344-clip re-analysis — read this first

Full detail: EXPERIMENTS.md "Re-analysis 2026-09-26". Short version:
- **`attn_rank` is a null result, not an FP regression** (at matched recall it equals its seed-0 twin
  A1-compress256; the FP gap in the block below was seed operating-point noise on half the test set).
  `attn_mass`: ep4 = calibration shift only; ep8 = genuinely worse (−0.030 AP) and val_ap selected it.
- **Evaluation rules from now on:** pool private+public (1,344 clips); paired bootstrap vs the
  same-seed twin; FP at matched recall; ≥3 training seeds (seed variance ~0.014 >> paired test noise
  ~0.003); don't trust clip-level val_ap rank-1 selection.
- **Why val_ap ≈ 0.95:** clip-averaged metric (+~0.03), in-distribution val, and a confirmed
  train/test negative-sampling mismatch (test negatives are midpoint-cut; ours avoid the midpoint).
- **Where we stand vs literature:** A1-compress256 = AP 0.911 / Kaggle mAP 0.916 on 1,344 clips ≈
  BADAS-1.0 (40k videos). BADAS-2.0 needed ~178k labeled videos + 2.25M SSL videos for mAP 0.940.
  Published small-data methods on this benchmark: 0.85-0.87.
- **Candidate next steps (none started):** midpoint-aligned hard negatives in training (cheap,
  crash-only); Stage K kinematic targets with truncation-aware looming; multi-seed protocol.

## ⚠️ 2026-09-26 status update — Stage AA-H CLOSED, diagnosis revised (FP claims below partly corrected above)

**User decision: close Stage AA-H (no `gradcam` run), reassess the diagnosis.** The diagnosis
that motivated this whole stage — "the crash head underperforms because it doesn't attend enough
to the relevant/threat object" — is **rejected**. Both loss families genuinely moved the head's
attention onto the labeled-relevant object (confirmed via `rho_P/V`), and both made FP *worse*,
with a dose-response: the more aggressively attention was forced (`attn_mass` >> `attn_rank`), the
worse FP got. That rules out attention misallocation as the bottleneck — if it were the real
problem, forcing more attention there should help, not reliably hurt more.

**Revised diagnosis (backed by data, see EXPERIMENTS.md's 2026-09-26 post-mortem):** the 67 public
negatives `attn_mass` turns into false positives have a **median control-model score of 0.212 vs
0.051 across all 333 negatives** — they're disproportionately the "close call" negatives the plain
classifier already found borderline, not random clips. Forcing attention onto the relevant object
amplifies the model's existing proximity-to-risk association instead of teaching it to distinguish
"closing in dangerously" from "nearby but not converging" — because the relevance/partner label
never encoded that distinction (it encodes *which* object matters, not *whether* its trajectory
implies a hit). No amount of attention/gradient-location supervision can teach a distinction the
training signal doesn't carry.

**Implication for what's next (not yet scoped as a plan):** the bottleneck looks like evidence
integration given the already-identified relevant object, not attention placement. This points
back toward directly supervising **kinematic evidence of collision** (closing rate, TTC,
trajectory convergence) — the deferred Stage 2 kinematic-target design (α, g, closing_rate, lane)
from before Stage AA existed — rather than more attention/gradient-shaping variants. See
DECISIONS.md's "Stage AA-H closed, diagnosis revised" entry for full reasoning.

**Nothing running or billing.** All Stage AA/AA-H pod work is done; data is safe on network volume
`0hnvco2s4j`. RunPod API key is on file for whenever the next stage needs pod time.

## ⚠️ 2026-09-26 status update (superseded by the closure block above, kept for detail) — Stage AA-H Stage 2b DONE: `attn_mass` is WORSE than `attn_rank`, both loss families now fail

**`attn_mass` + R_pos (the user's own follow-up hypothesis after Stage 2a) is done. It's a
stronger, more negative result than `attn_rank`: FP=79-121 (vs control's 54-62) at every
checkpoint tested, despite the best mean-over-8 AP (0.9020) of the whole investigation — AP alone
would call this the winner; FP at threshold 0.5 says it's the worst. `rho_P/V` explodes to 195.85
(vs `attn_rank`'s ~4.6 peak) and never unwinds after the aux loss switches off. Also exposed the
worst val_ap/rank-1 selection failure yet: val_ap picked the epoch with the *worst* test AP of all
8.** Full table and mechanism reasoning in EXPERIMENTS.md's Stage 2b entry and the child plan's
2026-09-26 status block.

**STOP — two independent loss families now agree "make the head attend more to the labeled-
relevant object" doesn't reduce false alarms. Awaiting user decision: try `gradcam` (the third
literature option, gradient-based rather than attention-based) or close Stage AA-H entirely.** See
DECISIONS.md's "AA-H next step" open question.

**Pod handling note:** the driver script's own `runpodctl stop pod` auto-stop worked correctly at
job end — but the pod was then deleted (not stopped) by a manual dashboard action before results
were pulled. No data lost (`/workspace` is on network volume `0hnvco2s4j`, independent of any
pod) — a fresh pod attached to the same volume retrieved everything. **New pod
(`h4uxgf5c3hu4dw`) has already been stopped via the RunPod API after syncing results — nothing
running or billing right now.**

## ⚠️ 2026-09-25 status update — Stage AA-H Stage 2a DONE: `attn_rank` fails the FP gate on all 3 labels

New direction after Stage AA's negative result (below): instead of a side probe on an
intermediate ViT-L layer, supervise what the crash **head's own attention** (or decision
gradient) reads. Full design, literature review (RARE/FAX/GAIN/CAMAL), and the current status
block: `~/.claude/plans/CCP based BADAS/2026-09-24_Child-Plan-AA-H-head-attention-supervision.md`
— **read that file's own status block first**, it is more current than this summary.

**Stage 0 (local) and Stage 1 (pod, noise floor) are DONE.** Stage 1 measured how much test AP
moves from LoRA-init randomness alone (2 seeds of A1-compress256's exact recipe, split held
fixed): **mean-over-8-checkpoints AP noise floor is tight (0.8910–0.8958, range 0.0048)**;
**rank-1 (single best-validated checkpoint) noise floor is much wider (0.8924–0.9128, range
0.0204)**. Rule adopted for judging every arm below: **lead with mean-over-8, treat rank-1 as
secondary** — rank-1 alone can't distinguish a real effect from a re-roll of the random init.

**Stage 2a (screening the `attn_rank` loss with 3 label choices, seed 0) — DONE, all 3 labels.
Verdict: does not clear the plan's screen gate; do not carry `attn_rank` into Stage 3.**

| Label | λ* | rank-1 test AP (private) | mean AP over 8 ckpts | public AP | public FP (n=667) |
|---|---|---|---|---|---|
| AA-ctrl-seed1 (control) | — | 0.8924 | 0.8910 | 0.9017 | 62 |
| AA-ctrl-seed2 (control) | — | 0.9111 | 0.8958 | 0.9087 | 54 |
| R_all (every clip) | 0.1645 | 0.9159 | 0.8991 | 0.9119 | 67 |
| R_pos (crash clips only) | 0.6965 | 0.9124 | **0.9018** | 0.9102 | 66 |
| partner_pos (hindsight crash-partner) | 0.5036 | 0.9130 | 0.8951 | 0.9092 | 66 |

**Why it fails:** the mean-over-8 AP lift is marginal at best (R_pos +0.006, R_all +0.003 over
the control range) and absent for `partner_pos` (0.8951, inside the control range) — the label
closest to the user's original FP concern gave the *least* lift. Worse, **all three arms have
more public-set false positives than either control seed** (66-67 vs 54-62) — the exact FP-growth
failure mode that motivated moving off Stage AA's side-probe design in the first place is still
present. The attention mechanism does move as designed (`rho_P/V`/`rho_P/B` both rise sharply
during the aux-on epochs 1-3 and partly persist after), but for `partner_pos` that movement is
entirely P-vs-background, not P-vs-other-vehicle — it never learns to prefer the true collision
partner over other nearby traffic, which is why it didn't help FP. val_ap stayed in the normal
0.94–0.95 range throughout for all three (no crash-task damage). Full numbers and reasoning in
EXPERIMENTS.md's Stage 2a entry and the child plan's 2026-09-25 status block.

**(Superseded by the 2026-09-26 block above — attn_mass has since been run and also failed.)**

**RunPod API key and auto-stop are now set up** (as of 2026-09-25/26) — a Read/Write API key is
on file, used via `runpodctl stop pod <id>` both from inside driver scripts and manually via the
RunPod REST API when needed. Pod IDs are found via `curl .../v1/pods` (list) since
`$RUNPOD_POD_ID` is consistently empty inside these containers. This removes the earlier "no
auto-stop" risk for future stages, though a stopped pod can still be **deleted** by a manual
dashboard action (happened once, 2026-09-25 — no data lost since `/workspace` is a network volume
independent of any pod, but costs a pod-recreation cycle to get results back). All persistent
data (frames, labels, captions, all `outputs/aa_head_attn/` results) lives on network volume
`0hnvco2s4j` (EU-RO-1) and survives any pod being stopped or deleted.

**Two real bugs found and fixed this session** (both would have silently corrupted a run — see
EXPERIMENTS.md and the child plan for full detail):
1. **gradcam's autograd-graph-ancestry bug** — differentiated against `patches[0]`
   (`forward_clip`'s post-hoc return slice, a sibling node) instead of
   `badas._captured["patches"]` (the actual ancestor tensor the crash head reads). PyTorch
   raised loudly; fixed by using the raw captured tensor for both the `torch.autograd.grad`
   call and `gradcam_loss`'s own `x` argument.
2. **the crash-vs-aux gradient-cosine probe's `aux_loss.requires_grad` gap** — when a window's
   aux term is skipped (e.g. `--aux-label R_pos`/`partner_pos` on a negative clip, or any
   missing label), `aux_loss` is a disconnected `torch.tensor(0.0)` with no `grad_fn`. The probe
   tried to differentiate through it anyway, crashed, and **permanently disabled itself for the
   rest of the epoch** on the very first such window — R_pos's first pilot attempt hit this
   almost immediately (`aux_grad_cos_n_sampled=0` all epoch) and silently fell back to an
   **uncalibrated λ=1.0**, which then ran a full 8-epoch training job on bad data before being
   caught by manually checking the sample count (not by any automated guard). **Killed that run,
   fixed by requiring `aux_loss.requires_grad` before the probe attempts anything**, verified
   both by an isolated control-flow test and live on the pod (R_pos's re-run pilot got
   `aux_grad_cos_n_sampled=80`, `partner_pos`'s got 74). The corrected driver script
   (`run_stage2a_3plus.sh`) also aborts a label's full run outright if a future pilot ever gets
   0 samples again, instead of silently using a fallback lambda.

**Known approximation, moot now:** `attn_rank`'s margin was set to **0.5** for all of Stage 2a
from a thin sample (19 windows, not the full planned 200-window Stage-0 diagnostic). Since
`attn_rank` failed the screen gate on FP regardless of label, re-measuring this margin more
precisely is not worth doing unless `attn_rank` gets revisited later.

**(Superseded — Stage 2b has since run, see the 2026-09-26 block above. Current STOP point is
whether to try `gradcam` or close Stage AA-H; nothing is running or billing.)**

## ⚠️ 2026-09-24 status update — Stage AA token-relevance aux loss: negative result, read this first

**Champion is still A1-compress256, test AP 0.9128 (private) / 0.9096 (public).** Stage AA tested
whether teaching the ViT-L's intermediate tokens *which detected car matters* (a per-token BCE aux
loss, alongside the crash CE loss) could lift it. **Six training runs, three independent fixes
(layer choice, head-unfreeze, corrected gradient diagnostic), none separated from its control.**
Full numbers, per-run diagnostics, and the code fixes: `EXPERIMENTS.md`'s "Stage AA token-relevance
aux loss" section. Rejected variants: `DECISIONS.md`. Architecture (hook point, loss wiring, data
flow): `ARCHITECTURE.md`'s matching section. The experiment's own child plan (superseded design
history + a full status block) is the source-of-truth working doc:
`~/.claude/plans/CCP based BADAS/2026-09-19_Child-Plan-AA-token-relevance-aux.md`.

**What was built and confirmed to work:**
- AA.2 detection + AA.4 token labels on the full 1,761-window pool (1,107 videos) — `rel` (soft,
  top-5 selection score / 2) and `occ` (hard, any tracked vehicle) per-token targets, 2048 tokens
  (8 tubelets × 16×16), `dataset/aa_token_labels/`.
- `aa1_token_probe.py` (Phase 3): frozen-trunk linear probe confirms the aux signal IS present and
  layer-17-specific after training (AA-rel's layer-17 rel-Spearman gain over base +0.098 vs
  AA-occ's control gain +0.054) — **but that gain does not survive to layer 23 or to the crash
  head's actual input** (+0.056 vs +0.049 at layer 23, nearly equal). This is the core finding:
  the aux loss teaches something real and localized, it just doesn't propagate to where the
  prediction is made.
- `semsup_train.py --aux-mode {occ,rel} --aux-layer --aux-weight --aux-labels-dir
  --aux-head-init`: per-token aux BCE added to the crash loss, λ sized via a gradient-norm pilot
  (target ~10-12% pull on shared LoRA layers — the defensible criterion; note the loss VALUE
  ratio disagrees, see EXPERIMENTS.md), full per-layer gradient/weight-change diagnostics
  (`grad_trace.jsonl`, `plot_grad_trace.py`).

**Two fixes landed this session that matter for ANY future run in this codebase, not just AA:**
- `semsup_train.py` (`1c56fe2`): `--unfreeze-head` + `--lora-init` resume previously reloaded
  LoRA correctly but silently reset the head to its frozen starting weights (`load_head_state()`
  was only ever called at scoring time). **Any past "check N epochs, then continue" workflow using
  `--unfreeze-head` should be treated as suspect** unless it used a single unbroken run. Fixed via
  new `--head-init`.
- `f3c064a`: the global crash-vs-aux gradient-norm diagnostic (`epoch_metrics.jsonl`'s
  `aux_grad_norm_crash`/`aux_grad_norm_aux`) silently never ran for any prior AA run (`null` for
  all of them) due to a `None`-filtering bug — the per-layer `grad_trace.jsonl` diagnostic was
  unaffected and is what the reported numbers above rely on.

**Website**: A1-compress256, AA-occ-unfrozen, AA-rel-unfrozen published (landing page, experiment
detail, cross-experiment comparison) — commit `6812927`, verified rendering in-browser, not yet
pushed. Known gap: `train_aux`/`aux_grad_cos` fields are in `experiments_data.js` but **no chart
draws them** (`experiments.html`'s loss chart only plots `train_total`/`train_sem`) — the aux-loss-
vs-epoch curve is not visible on the site even though the data exists. Also fixed in this pass: a
stale hardcoded `"11 arms · A0 → v12shuf"` picker-hint string in `experiments.html`, now derived
from the arm list so it can't go stale again.

**Not yet decided (open question for next session):** given layer choice (17 vs 23) and
head-unfreeze both failed to move AP, the remaining options are (a) a much larger λ concentrated
at layer 17 with a frozen head — where AA-rel already showed a real, if non-propagating, effect,
(b) an attention-mass diagnostic (designed, not built) — does the crash head's actual attention to
the relevant car shift under any condition, or (c) accept "no AP gain from this aux signal, via
three different fixes" as the reportable negative result itself. Layer 13-19 probe sweep
(deferred earlier by the user) is lower priority than these three.

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

## Stage AA.1 — detection pipeline v2b: Stages 0–3 built, generalization check run (2026-09-19)

Father plan Stage AA.1: detection → tracking → ego path → object selection (top-5 per window) on
the 18 held-out `val_e3a` clips (9 pos / 9 neg), plus a 18-clip fresh generalization set. Active
child plan: `~/.claude/plans/CCP based BADAS/2026-09-15_Child-Plan-AA1-v2b-YOLOPv2-lanes-tracking.md`
(its Stage 2/3 text is stale — the built design is in ARCHITECTURE.md). Runs locally (RTX 1000 Ada
6 GB, `torch 2.11.0+cu128`). STOP for user review after each stage.

**v1 = baseline, superseded, do not extend** (`aa1_detect_track_rank.py`, commit `812e884`,
outputs `outputs/aa1_smoke_18clips/`): G-DINO + BoT-SORT + horizon corridor. Still supplies
`decode_span`/`decode_frames`/`window_ends`/`load_event_row`/`recency_ok`.

| Stage | What | Status |
|---|---|---|
| 0 | YOLOPv2 per-frame cache (`aa1_yolop_cache.py`, cached down to conf 0.10) | done, 36 clips |
| 1 | BoT-SORT (3 s buffer) + stitching + hood filter + edge-track extension (`aa1_tracks.py`, `aa1_track_stage1.py`) | done, 36 clips |
| 2 | Now only shared loaders `aa1_stage2.load_frame_masks` (masks + ego path) / `load_stitched_tracks`; no driver | done |
| 3 | Selection score, per-window top-5, overlay, target curves (`aa1_collision.py`, `aa1_stage3.py`) | done, 36 clips; user is still reviewing overlays |
| 4 | AA.4 target definition + patch masks | **not started** — target dims proposed, user go-ahead pending |

After Stage 4: AA.2 (full training-pool detection, hours at YOLOPv2 speed) → AA.4 masks/targets →
AA.5 training (AA-1 aux fit, AA-2 LoRA, matched AA-ctrl), warm-started from A1-compress256.

**Settled design** (rejected alternatives in DECISIONS.md):
- K = 5 objects per window, flat weights `a_i = 1/K_clip`. The selection score only chooses WHICH
  5 objects get supervised; it is never a training target and never fit to the crash label.
- Same selection procedure for positives and negatives.
- Aux tap = `backbone.encoder` output (2048 tokens); masks index tokens 0–2047 only; the predictor's
  512 tokens are never masked; the crash head still reads all 2560. compress256 box→token map:
  x·256/1280, y·256/720, then /16.
- Proposed Stage 4 targets, all against the ego path: α (looming), g (side gap, box-heights),
  closing_rate = −ġ (+ = approaching the path), lane state (LEFT/EGO/RIGHT). Drop `s`, `s_rate`, ρ.
  Box height is NOT a target (near-readable from appearance).
- Contribution checks planned for AA-2: AA-1 gate first (can the frozen trunk's object tokens predict
  the targets vs a mean predictor), then controls — λ_aux=0 with the same warm start, shuffled
  targets, random masks — plus per-epoch val AP and aux-vs-crash gradient ratio.

**Open TODOs:**
- User to review the 36 overlays + new `target_curves.png` (3×2 design) and decide on Stage 4 targets.
- GT partner labels: dev clips are MY visual drafts (user has not confirmed): 00147 id5, 00283 id2
  (low confidence — id1 scores 0.39 vs 0.45; user asked which car is hit, unanswered), 00372 id6,
  00474 id0, 00493 id0, 00529 id3, 00687 id0, 00077 id1, 00319 id3 (car leaves the frame; low).
  gen18: confirmed-by-eye 00136 id0, 00486 id0, 00505 id9, 00583 id0, 00803 id1; NOT labelled: 00903,
  00932, 01035; 00195's crash car is never detected.
- Open ego-path failures (below). Detection misses like 00195 need a different fix (recall).
- Pedestrians/cyclists: YOLOPv2 detects vehicles only.
- Negative clips decode one ~8 s block; windows MID-10/-8/-4 leave a ~4 s unused gap (visible in the
  target-curve plots). Fix or shade before AA.2.
- User-stated target: first training results ~2026-09-22 — tight (Stage 4, AA.2, two training phases).

**Known issues (ego path, `aa1_lanes.estimate_ego_path`):**
- 00486, 00505: a spare tire on the ego vehicle itself is intermittently classified as lane paint by
  YOLOPv2 (not a tracked box; persistence 63–100%), so the path points at it. Model-level; unfixed.
- 01532: a crosswalk at the start produces a plausible "lane pair" that seeds the path wrong; the
  street after the turn only offers weak single-line data, so the path stays stuck (held).
- 01478: long `held` stretches after the lines are lost. 01035: user reported a mid-clip new line not
  updating; not reproduced on 12 sampled frames — needs frame timing.
- 00493/00529/01552-like scenes with no forward lane paint (dense traffic, crosswalk stripes only):
  path stays default/held. Not a bug.
- Fixed this session: 01075 (bad first seed), 00932 (headlight glare inside a van's box), 01552
  (0→204 lane-pair frames), 00077 (90/90 measured).
- 358→386 stitch merges not individually verified; 02104 short-clip windows empty; 01737 near-empty scene.

**Reporting:** professor report `reports/progress_reports/2026_09_15_progress_report.docx` (build:
`python reports/_scripts/_build_progress_report.py`).

**Commands (run from repo root):**
```
python student_training/scripts/aa1_stage3.py                      # 18 dev clips: scores + overlay + curves
python student_training/scripts/aa1_run_set.py --set gen18 --stages 0,1   # detect+track the gen18 set
python student_training/scripts/aa1_run_set.py --set gen18 --stages 3     # score gen18
```
Outputs: `outputs/aa1_v2_18clips/` (both sets merged, 36 clips).

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
# AA.1 v2b detection pipeline (run from student_training/scripts/, local GPU)
python aa1_scene.py                                   # YOLOPv2 sanity check on 00687
python aa1_yolop_cache.py --all --out-dir ../../outputs/aa1_v2_18clips   # Stage 0 cache + recall vs v1 G-DINO (~1 min)
python aa1_track_stage1.py                            # Stage 1: tracks_v2.json, overlay_v2.mp4, grid16_v2.jpg, timeline_v2.png
python aa1_stage2.py                                  # Stage 2: geometry_v2.json, overlay_lanes.mp4, grid16_lanes.jpg, target_curves.png/csv
# v1 baseline (G-DINO, ~0.7 s/frame; do not extend)
python aa1_detect_track_rank.py --all --out-dir ../../outputs/aa1_smoke_18clips [--from-cache]
# YOLOPv2 weights (official release, 156 MB, gitignored) if missing
curl -sL https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt -o third_party/yolopv2_ref/weights/yolopv2.pt
# Professor progress report (docx + figures)
python reports/_scripts/_build_progress_report.py

# Stage AA token-relevance aux loss (run from student_training/scripts/)
python aa1_run_set.py --set pool1761 --stages 0,1        # AA.2: detect+track the 1,761-pool (1,107 videos), local
python aa4_token_labels.py                                # AA.4: per-window rel/occ/box_h/path_gap npz labels
python aa1_token_probe.py --checkpoint base --n-windows 1761 \
    --aux-layer 17 --out outputs/aa_token_aux/probe_base   # Phase 3: frozen-trunk relevance probe (GPU, ~1-4s/window)
python aa1_token_probe.py --checkpoint <label> --lora-adapter <epoch_dir>/lora_adapter \
    --aux-layer 17 --out outputs/aa_token_aux/<label>       # same probe on a trained checkpoint (diagnostic)
python plot_grad_trace.py <run_dir>/grad_trace.jsonl <out.png>   # per-layer gradient-norm/cosine figure

# semsup_train.py aux-loss flags (add to any normal training invocation)
--aux-mode {none,occ,rel} --aux-layer 17 --aux-weight <lambda> \
    --aux-labels-dir dataset/aa_token_labels --aux-head-init <probe_head.pt>
# resuming a run that used --unfreeze-head, past epoch 1: MUST also pass --head-init
# (pointing at the last epoch's head_state.pt) alongside --lora-init/--optimizer-init,
# or the head silently resets to its frozen starting weights with no error.
```

## Git state
Branch `main`, HEAD `6812927` (website: add AA-occ-unfrozen and AA-rel-unfrozen arms) — **still
the HEAD, no new commits this session**. **5 commits ahead of last-known `origin/main`**
(`a0341a0`..`6812927`) — not pushed; `git fetch` has no push-key access in this environment, the
user pushes. **Uncommitted as of 2026-09-25** (not yet committed — ask the user before
committing, per standing instructions): `student_training/scripts/semsup_common.py`,
`semsup_train.py` (modified — the new Stage AA-H aux modes + both bug fixes), new files
`aa_head_losses.py`, `aa_head_attention_diag.py`, `aa4_partner_labels.py`,
`test_aa_head_losses.py`, plus `dataset/aa_token_labels/*.npz` (now carry a `partner` array for
positive-video windows) and the new `outputs/aa_head_attn/` directory (Stage 1 + Stage 2a
results, synced from the pod).

## Next step
**Active thread: Stage AA-H (2026-09-26 status block above) — Stages 2a AND 2b are both DONE and
analyzed, both failed (attn_rank and attn_mass). This is a STOP point awaiting user decision on
`gradcam` vs closing AA-H, not a resume-and-continue point.** Nothing is running or billing. This
supersedes the 2026-09-24 Stage AA (side-probe) status
block right below, which is a paused, separate design that Stage AA-H replaced after finding
the literature pattern (RARE/FAX/GAIN/CAMAL: supervise the classifier's own attention/gradient,
not a side probe). The old AA.1-era material further below is fully superseded, not a separate
open thread.

If instead resuming the earlier AA.1-era open items (lower priority, listed for completeness):
1. GT labels still unconfirmed: gen18 00903/00932/01035, dev 00283 (id1 vs id2).
2. Open ego-path failures (spare-tire glare, crosswalk seed) — two attempts already rejected
   (DECISIONS.md); only revisit with explicit user direction.

The pre-2026-09-13 V13-caption / concept-head fork remains deprioritized behind Stage AA and is
not being pursued.

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
