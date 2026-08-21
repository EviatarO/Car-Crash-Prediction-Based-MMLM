# Decisions

## Rejected options (with reasons — don't re-propose these)

### Architecture / method
- **Reuse the public ReverseBERT decoder as-is for reasoning generation** → domain-locked to
  emotional-speech captions (`emoact_prompts`), produced nonsense on crash text. Needed a
  crash-domain fine-tune before it was usable at all.
- **Full-unfreeze BADAS's ViT-L trunk for A1/B** → far too large for the available data
  (267 → 4.5k). LoRA-only on `query,key,value`.
- **Guess BADAS's LoRA-able layer names from docs/convention** → BADAS internals are only
  knowable at runtime; guessing risks LoRA silently attaching to zero parameters. Resolved
  empirically via `dry_run_modules()`.
- **Reuse a frozen InternVL-style `mlp1` projector as a video→embedding bridge** → geometry
  mismatch (InternViT per-frame spatial features vs. a spatiotemporal source). Train the small
  Predictor fresh instead.
- **Attach an LLM to BADAS's V-JEPA2 encoder for score+reasoning** → no published V-JEPA2→LLM
  projector exists (confirmed: even Nexar's own BADAS-Reason avoids it, using a separate
  Qwen3-VL-4B fed peak-risk frames rather than a fused encoder→LLM). Training one is a
  research project on its own.
- **Use a general VLM as the crash predictor** → Nexar already tested this and it lost clearly
  (BADAS-2.0 paper: Cosmos F1 0.817, Gemini F1 0.662 vs 0.964). Language belongs in the
  *training* signal or as post-hoc explanation, not in the prediction path.
- **VL-JEPA as the predict+reason route** → the latent→text step needs a domain-trained
  Y-decoder; the public stand-in was domain-locked and incoherent on driving text. Would be a
  large side project that still doesn't address the AP gap.
- **Download a pre-trained Predictor / reuse an off-the-shelf projector (BLIP-2 Q-Former,
  LLaVA MLP)** → all are trained on CLIP/SigLIP *image* features, geometrically incompatible
  with V-JEPA2 spatiotemporal patches. Nothing off-the-shelf speaks that space.
- **Pre-train the Predictor on external video-caption data before the SFT** → the Predictor
  (~6M params) is not the bottleneck: in B1 it trained fine (loss 0.48→0.11) and *overfit*
  rather than underfit, against a near-degenerate anisotropic target. More Predictor reps
  can't change what is or isn't present in the frozen BADAS features. Would also require
  video-caption (not image-caption) data to avoid mis-shaping the temporal axis.

### Reporting / methodology
- **Report B's ep8 (test_AP 0.8742) as "B's result"** → it was val-rank #2; promoting it
  because it scored best *on test* is selecting on the test set. The pre-registered rule
  (best val_ap) gives 0.8574. Rejected as metric gaming.
- **Trust `val_ap` for checkpoint selection at n=267 without fixing it** → the 51-row val
  split is only 17 independent clips, so val_ap saturated at 0.96–0.98 and ranked near-inverted
  vs test. **Resolved 2026-07-25 (T-3)**: fixed by aggregating per clip rather than switching
  to an arbitrary fixed-epoch rule — the root cause (treating correlated rows as independent
  samples) is now actually fixed, not just worked around.
- **Keep the cosine-regression semantic loss and attribute B1's null purely to n=267** →
  **rejected 2026-07-25 (A-1)**: verified the objective's own analytic minimizer explains
  99.47% of the achieved loss reduction, independent of data scale. InfoNCE added as the
  actual fix; the real re-run is still pending (see Unresolved).
- **Carve a larger val set out of the 677-clip Private test** → any clip used for selection
  can't also count in the reported test AP without leaking, and it would break comparability
  with A0's already-published 0.853. Only viable if done very deliberately.
- **Read A1-vs-B at n=267 as a directional result** → the gap (±0.01 depending on aggregation
  rule) is smaller than the within-method checkpoint spread. Underpowered; no claim survives.
  **Confirmed quantitatively 2026-07-25 (T-2)**: paired bootstrap CI (5000 resamples) on the
  pre-registered best-val comparison (A1 ep8 vs B ep1) = [−0.0239, +0.0030] AP, crosses zero,
  P(B>A1)=7.4%. See EXPERIMENTS.md.

### Meta / tooling
- **Auto-trigger `/handoff` at exactly "50% context usage"** → context-window percentage is
  not exposed to hooks or the model (confirmed against Claude Code docs). Settled for: a
  user-triggered `/handoff` skill plus a hard `PreCompact` gate that blocks compaction with
  stale docs, with a once-per-session safety valve so it can never wedge a session shut.
- **Statusline context-usage indicator** → user declined; not built.
- **Trust the `cwd` field in hook payloads to locate the project** → it is the app's *launch*
  directory, not the working project. Both hooks read a session-keyed pointer file instead.
- **Name the new review skill `/code-review`** → collides with the built-in `/code-review`
  (which reviews a branch diff and powers `/code-review ultra`). Named `/project-review`.
- **Build custom Agents for planning** → plan mode is a separate built-in mechanism; no custom
  agent is needed. Custom agents only pay off for a repeated, specialized sub-task.
- **Trust the exit code of a backgrounded command piped through `tail`** → `tail`'s exit code
  is reported, not the piped process's; this masked a real crash as "completed, exit 0" with
  an empty output directory. Redirect straight to a file and check `$?` explicitly instead.
- **Run two local smoke tests that both load BADAS-Open concurrently in the background** →
  observed to silently crash one of them (shared on-disk HF cache contention); the identical
  command succeeded when re-run alone. Run sequentially when both load BADAS.

### Prompt bake-off design (2026-07-27)
- **Two separate captioning prompts (Driving-Semantic vs Risk-Aware-Causal)** → rejected before
  writing any code. The two drafts were too similar to distinguish at n≈300 (paired bootstrap
  already showed n≈300 is barely powered even for a MUCH bigger intervention), and running them
  as two independent teacher passes would let the teacher describe the same scene differently
  each time, confounding the one variable actually worth isolating. Replaced with **one prompt,
  three arms built from its structured JSON output** — see EXPERIMENTS.md.
- **70-120 word captions in the original prompt draft** → rejected: measured at 128 SigLIP
  tokens against a hard 64-token truncation limit, discarding the outcome clause every time
  (it was written last). Capped at 40 words, outcome-relevant content first.
- **"Consistency over variety" as a blanket instruction** → rejected: it's exactly the
  mechanism that produced the 0.8547 anisotropy collapse risk if applied to *content*, not just
  *relational vocabulary*. Scoped instead: canonical terms only for relations/motion
  (`braking`, `merging`, ...); content (actor, direction, proximity) must be clip-specific,
  enforced by an automated duplicate-sentence check, not just prompt wording.
- **Trusting the prompt's "don't mention risk in caption_neutral" instruction alone** →
  rejected: VLMs editorialize about danger on dashcam footage regardless of instruction. Backed
  by an automated banned-word check + regenerate loop (`semsup_caption_qa.py`), because Arm A's
  entire validity depends on it actually being neutral.
- **Decision rule for the bake-off, written down 2026-07-27 before any real captions exist**
  (implemented mechanically in `semsup_promptbakeoff_report.py`'s `decide()`):
  - A beats C, B ≈ A → structure carries it, verdict adds nothing → scale Arm A. Strongest claim.
  - A beats C, B beats A → both channels contribute → scale Arm B, report the split.
  - A ≈ C, B beats A → the gain is the label, not the language → drop the structure claim, use
    label smoothing instead.
  - Neither beats → nothing works at this scale → deprioritize (see option 4 below).
  - REF (incumbent 267) beats both new arms → stop, diagnose the captioning, don't scale anything.

### Teacher-model bake-off (2026-07-28/29)
- **Compare a candidate teacher against the historical v6 baseline (83.3%/6.78)** → rejected:
  a same-day rerun of the identical unmodified v6 prompt scored 50.0%/4.61 on the same clips -
  the historical number is stale (likely model drift on the `google/gemini-3.1-pro-preview`
  `preview` alias). Any teacher comparison must use a fresh same-day baseline. See
  EXPERIMENTS.md's "Round 0."
- **Conclude a candidate teacher is "better" from verdict accuracy alone on this balanced
  18-clip set** → rejected: both Qwen3.7 Flash and GPT-5.6 Luna Pro scored 61.1% (vs Gemini's
  50.0%) purely by predicting "no collision" on 16/18 clips (recall 0.22, same as Gemini's
  worst case). On a 9-pos/9-neg set, extreme conservatism is rewarded by accuracy without
  reflecting genuine detection capability. Always report recall/precision alongside accuracy
  for this kind of screen.
- **Assume an explicit anti-under-calling prompt instruction fixes the conservative bias** →
  refuted empirically: `PROMPT_SEMSUP_V4_QWEN.py` explicitly instructs against defaulting to
  NO and provides worked examples calibrating the threshold, run against a third, different
  model (`qwen/qwen3-vl-235b-a22b-thinking`) - identical confusion matrix (TP=2, FP=0, TN=9,
  FN=7) resulted anyway. 3x-replicated across 2 prompts x 3 models. See EXPERIMENTS.md
  "Round 2" and the `00687` finding (model perceives the hazard correctly, decision layer
  overrides it) - the fix, if any, is more likely in the decision structure (binary
  verdict + hard gates) than in surrounding instruction text.
- **Import `teacher_bakeoff.py` / `Teacher_dataset_distill_v11.py` directly for their
  OpenRouter helper functions** → both have a pre-existing broken top-level import
  (`prompts/PROMPT_G2.py` / `prompts/templates.py`, moved during an earlier reorg, unrelated
  to this thread and not fixed here). New scripts copy the needed functions instead
  (`semsup_caption_promptbakeoff.py`, `semsup_v6_control_rerun.py`).
- **Continue iterating prompt versions (V5 onward) on the 18-clip screen** → rejected
  2026-08-01, after V5-V9 (6 more rounds) produced no result distinguishable from any other at
  n=18 (McNemar p≥0.125 throughout, every CI overlapping). This was knowable from V4 onward;
  the 18-clip screen was never powered to rank prompts, only to catch gross failures. Pivoted
  to scoring the real ~4,500-window train pool through the frozen A0 scorer instead — real
  statistical power, and it measures the actual scorer rather than a caption-quality proxy.
- **Snap `MID`'s legacy offset-0 captions (`TN_MIDPOINT` in the old 267-row set) onto the new
  `MID-10` bucket** → rejected: they're genuinely different windows (offset 0 vs −10), not the
  same bucket renamed. Left permanently unresolved by `build_caption_monitor.py`'s
  `_resolve_caption_bucket()` rather than silently misattributed.

### train4500-inference pipeline (2026-08-01)
- **Keep the negative `MID` bucket at offset 0.0 (exact clip midpoint)** → rejected: real
  scoring found 42.8% error, 100% false positives, at 0.99+ confidence, isolated to that one
  bucket. Moved to offset −10.0 (renamed `MID-10`). See EXPERIMENTS.md for the full before/
  after numbers and ARCHITECTURE.md for the diagnosis (clip-level label vs. a genuinely
  risky-looking literal midpoint).
- **Trust a directory's file COUNT as proof a transfer succeeded** → rejected 2026-08-01:
  `tar` writes the correct file name/count even when a disk quota kills the actual content
  write, leaving 0-byte files that pass a count-only check. Must additionally check file size.
- **Treat `df -h /workspace`'s free-space number as proof a write will succeed** → rejected:
  it reports the network filesystem's cluster-wide free space, not the account/volume-specific
  quota that actually governs writes. A live test write is the only reliable check.

### Mixed-pool training fix (2026-08-02 – 06)
- **Two separate prompts (V11: positive-only GT with `hazard_*` fields, negative-only blind
  with neutral field names) to fix negative-clip fabrication** → rejected 2026-08-04: the
  fabrication was caused by the GT-block instruction pressuring the model to nominate a hazard
  on a "no collision" clip, not by the shared `hazard_*` field naming. V10's existing blind
  mode had already produced the correct `agent: None` answer on the exact clip that motivated
  V11. Built, tested, measured worse (negative recall 0.38 vs 0.435), and deleted same day.
  **General lesson, saved to memory**: before building a new configuration to fix an observed
  failure, check whether an existing, cheaper configuration already avoids it, and identify the
  specific causal mechanism before designing the fix.
- **Train A1/B on only the 587 A0-failure windows** → rejected after real measurement:
  test_AP=0.333, AUC=0.163 (anti-correlated). Diagnosed as inversion (frozen head + adversarial-
  only training data), not plain overfitting — AUC below 0.5, not converging toward it, is the
  tell. Fixed by mixing in 1,174 A0-correct windows (2:1 correct:hard). See PROJECT_STATE.md /
  ARCHITECTURE.md for the full mechanism.
- **Warm-start B_1761's Predictor from the OLD `b1_infonce` checkpoint (trained on the 267-row
  pool)** → rejected 2026-08-06: that checkpoint's own training clips overlap 13/221 with the
  new 1,761-pool's val split (different pool, different split, coincidental clip reuse) — would
  contaminate val-based checkpoint selection. Also caption-style mismatch (that checkpoint
  learned on old rephrased-teacher-reasoning captions, not the new V10 vision-grounded style).
  Trained a fresh `b1_1761_infonce` on the exact pool/split B_1761 uses instead — zero overlap
  by construction, and it shows *stronger* signal at the new scale anyway (32×/24× chance vs.
  the old checkpoint's ~4×).
- **Reuse `--semantic-weight 0.3` (the value tuned for cosine) when switching to InfoNCE** →
  rejected: cosine's loss magnitude is ~0.13, InfoNCE's is ~log(N)≈5-7 — reusing 0.3 would make
  the semantic term ~20-40× more dominant than the crash loss, testing "does an overwhelming
  semantic loss wreck crash performance" instead of "does semantic supervision help." Recomputed
  λ=0.05 to land the semantic term at roughly a third of the crash-loss magnitude instead.

### I/O speed fix, A1-v2, and the V12 leakage fix (2026-08-08 – 11)
- **Adopt A1-v2's recipe bundle (full 4,446-window pool, cosine LR, encoder-only LoRA, dropout
  0.10) as the new standard** → rejected: underperformed A1_1761 on the real metric that
  matters (test_AP 0.868-0.888 best case vs A1_1761's 0.900). Root cause (pool distribution vs.
  recipe bundle) not isolated — deprioritized rather than chased further, since the core B-vs-A1
  question was the higher-value target. A1_1761's original recipe remains the reference control
  for all future comparisons.
- **Infer the training bottleneck's cause from GPU-utilization graphs alone** → rejected: a
  utilization graph shows the symptom (24-33% GPU busy), not the cause. Required direct
  phase-by-phase profiling (raw read / decode+resize / GPU forward, timed separately over 20
  real windows) before writing any fix — confirmed I/O+CPU-bound, not GPU-bound, with numbers
  (670ms/503ms/~0ms), not a guess.
- **Trust the $81-total captioning-cost figure in the planning doc as verified** → rejected
  2026-08-10, after the user pushed back with a real remembered paid amount (~$35 for the first
  round) that didn't reconcile. The doc figure was explicitly marked "not re-verified against
  actual billing" and had never actually been checked against OpenRouter's real per-call `usage`
  data (which was being silently discarded at the time). Root-caused: no hidden billed-failures
  explained the gap (only 1/897 failures was a real billed BAD-JSON; the rest were free 402
  pre-generation rejections) — the real, owned mistake was ~64 overlapping clips re-captioned
  across 4 separate concurrency-benchmark test runs (per-output-file resume-skip doesn't dedupe
  across different output files), wasting ~$2-3. Fixed the class of problem, not just this
  instance, by adding real per-call usage/cost logging (`<out>.usage.jsonl`) so all future cost
  claims are measured, not estimated.
- **Keep the V10 prompt's GT-informed/blind branch and try to patch the leak with tighter word
  bans** → rejected: the `/project-review` audit found the leak's dominant driver was the branch
  itself (two different prompt registers by class), not lexical slip-ups within one register —
  V10 already had an outcome-word ban and still leaked at AUC=0.9643. V12 removes the branch
  entirely (`build_prompt()` takes no `gt_mode`/`is_positive` args) rather than adding more bans
  to a structurally leaking design.
- **Treat V12's AUC=0.7640 (vs <0.75 target) as an outright fail and iterate further before
  training anything** → considered, but user chose (via AskUserQuestion) to accept the narrow
  miss and proceed to B-v2, given: (a) the residual signal was traced to genuine kinematic
  vocabulary (`braking`, `decreasing gap`), not lexical leakage (0/100 banned-word violations),
  so further prompt engineering has uncertain further payoff without hurting caption accuracy;
  (b) a 43% reduction in excess-over-chance signal is a real, reportable improvement regardless
  of whether B-v2's result is decisive.

### P3 correction and P1 two-stage (2026-08-16 – 17)
- **Report P3's "1.8×" ratio without a confidence interval** → rejected 2026-08-17, after the
  user pointed out the first pass (single noise draw per clip, no per-clip data saved) couldn't
  support the claim. Re-ran with 20 draws/clip and a paired bootstrap; same point estimate, now
  with a real CI [0.00143, 0.00163] excluding zero. The general lesson: a ratio between a real
  effect and a noise control needs the control's own variance estimated (multiple draws), not
  assumed from one sample.
- **Call P1 (two-stage training) "the last untested structural lever"** → rejected 2026-08-16,
  after the user pushed back. Several levers remain genuinely untested: B-rev (reverse
  projection direction), structured-field targets, the λ sweep, SigLIP's sigmoid loss, corpus
  scale-up, unfreezing the crash head. P1 was one candidate among several, not a final one.
- **P1 two-stage training, reusing A1_1761's LR unchanged for Stage B** → this IS what was run,
  and it lost by the widest margin in the thread (+0.0716 AP). Not rejected as a design in
  principle — the specific choice to reuse the from-scratch LR on a warm-started checkpoint is
  the likely proximate cause (train_val_gap more than doubled vs A1 under identical settings).
  Whether a Stage-B-specific LR fixes this is an open question (see below), not yet tested.

## Unresolved design questions

- **Does semantic supervision beat crash-only LoRA (A1) on the 1,761-window mixed pool?**
  **Resolved, repeatedly, always in the negative.** Five real GPU attempts, gap widening each
  time execution got cleaner, not narrowing:

  | Arm | ΔAP vs A1_1761 (paired bootstrap) |
  |---|---|
  | B_1761 parallel (V10, leaky corpus) | +0.0105, CI [0.0040, 0.0173] |
  | B-v2 (V12, clean corpus, cold-start predictor) | +0.0189, CI [0.0099, 0.0285] |
  | B-v3 (V12, warm-started + per-group clip) | +0.0218, CI [0.0117, 0.0325] |
  | P1 two-stage (Stage A pretrain → Stage B finetune) | **+0.0716, CI [0.0477, 0.0977]** |

  Three independent diagnoses now rule out routing/leakage explanations: caption leakage fixed
  (didn't help), the semantic gradient measurably reaches the classifier's own representation
  (P3, paired CI excludes zero) at least as well as random, and the crash/semantic gradients are
  near-orthogonal rather than opposed (gradient-angle probe). P1 (the most different design
  tried) is currently the *worst* result, with a specific measured mechanism (overfitting under
  a warm-started-but-unchanged LR, not unexplained forgetting) — see EXPERIMENTS.md. **Open:**
  whether a Stage-B-specific (lower) LR recovers P1's loss, and whether any of the untested
  levers (B-shuffle control, B-rev/reverse-projection, λ sweep, corpus scale-up) change the
  picture. Not yet run.

- **Why does the frozen A0 scorer do notably better on train (13.2% error, n=4,446) than on
  the known 677-clip test set (23.6% error)?** Confirmed real and stable at full scale
  2026-08-01 — held consistently across all 3 independently-sampled chunks (~13.3% each), not
  a chunk-0 fluke. Test is FP-dominated (130:30, 4.3:1); train is nearly balanced (318:269,
  1.2:1) — a genuine distributional difference between the pools. Pipeline mechanics checked
  and ruled out (byte-identical sequential-decode extraction verified). Not investigated
  further — open for whoever picks this up next.

- **Which scale-up path?** **Gate resolved 2026-07-25/26**: the B1-InfoNCE re-run (see
  EXPERIMENTS.md) found real, statistically-supported video↔caption signal at n=267
  (clip-level retrieval@1 4× chance, p=0.015) — the original null was the cosine objective's
  fault, not proof the data carries nothing. Scale-up is now well-motivated in principle.
  **But the choice is reframed, not just "how many captions":** the current 267 are
  text-to-text rephrasings of the teacher's `final_reasoning` that never saw a frame, so
  whatever signal InfoNCE just found is coming through *indirectly* (via the teacher's own
  reasoning text), not from fresh visual grounding. Proposed next step (not yet started):
  ~300 new clips from **distinct videos** (150 TP/150 TN, no sibling-TTE reuse — cleaner
  InfoNCE negatives than the current 267's ~89-video pool), captioned with **two prompt
  variants**, at least one of which does genuine frame-grounded captioning (not rephrase-only)
  — then re-run the InfoNCE check on that set before committing to the full 4.5k spend. This
  replaces the old three-option framing (full/intermediate/deprioritize): the informed choice
  is now "which prompt variant, and does frame-grounding actually add signal over the
  rephrase-only baseline," decided on ~300 clips before any 4.5k-scale spend.
- **How to fix checkpoint selection** → **Resolved 2026-07-25 (T-3)**: per-clip aggregation,
  not a fixed-epoch rule. See the "Trust val_ap" entry above.
- (Historical framing of "does semantic-aux beat crash-only," n=267/cosine era — **superseded**,
  see the new entry above this list: InfoNCE is now ported into Stage B and tested at n=1,761;
  the open question is now specifically "parallel vs sequential," not "is InfoNCE even wired up.")
- **LoRA target_modules (`query,key,value`) match both the 24 encoder layers AND the 12
  V-JEPA2 predictor-layer attention blocks** (same substring, both under `backbone`).
  **Resolved 2026-08-09/10**: a `re:<regex>` prefix on `--lora-target-modules` now supports
  encoder-only targeting (72 vs 108 adapters) as an alternative to the legacy comma-list. Not
  used as the new default — A1_1761's original 108-module recipe remains the reference control
  (see A1-v2 entry below); the regex option exists for whoever wants to isolate the effect.
- **Thesis framing if the result stays null**: report a well-controlled negative result as a
  contribution (defensible — no prior art tests this exact train-only-language /
  vision-only-inference regime in crash anticipation), or treat the literature gap as a signal
  to deprioritize. **Not decided.**
- **Are the un-backed-up pod checkpoints disposable?** `/workspace/Car-Crash-Prediction-Based-
  MMLM/outputs/checkpoints/`: `e2_lora_100clips` (4.4G), `e3a_lora_89clips` (832M),
  `e3b_lora_267clips` (704M) exist **only** on that volume — verified NOT on HF Hub (only
  `e3a-epoch7-lora` and `e3b-ep3-lora` are). Not deleted pending explicit confirmation.
- **Which teacher/prompt should caption the production set?** **Resolved 2026-08-02**: Gemini
  3.6 Flash + `PROMPT_SEMSUP_V10_GT`, hybrid mode (GT on positives, blind on negatives) — see
  EXPERIMENTS.md's bake-off. Used for all 1,761 captioned windows so far.
- **Is the teacher-model conservative bias fixable by prompt engineering, or does it need a
  different decision structure?** **Partially answered 2026-08-01**: 6 more prompt structures
  (V5-V9, spanning risk-score, kinematic decomposition, ego-frame separation, narrative
  structure, and deliberate minimalism) produced no result distinguishable from any other at
  n=18. What *did* move the outcome was the **model**, not the prompt structure — v6_balanced
  unmodified went from 0/18 YES (Qwen3-VL-235B) to 72.2% acc/0 FP (Gemini 3.6 Flash) with zero
  wording changes. Still open: whether that's really a decision-structure effect expressed
  differently per model family, or something else about Gemini 3.6 Flash specifically — not
  isolated.
- **Why does chunk 0 of train4500 score notably better (13.3% error) than A0's known test-set
  error (23.6%)?** New, open. Not the MID artifact (separately fixed and accounted for). Test
  is FP-dominated (4.3:1), chunk 0 is nearly balanced (1.1:1) — a real distributional
  difference between the two pools. Pipeline mechanics checked and ruled out as the cause
  (byte-identical extraction verified). Chunks 1-2, once scored, will show whether this holds
  at 3× the sample or was a chunk-0-specific artifact of which 500 videos landed there.
- **Is A1-v2's underperformance driven by the natural (non-enriched) pool distribution, the
  recipe bundle (cosine LR / dropout 0.10 / encoder-only LoRA), or both?** Not isolated —
  deprioritized in favor of the B-vs-A1 question, which is now resolved (see above). Still
  genuinely open if anyone revisits pool-scaling; should ablate separately rather than
  re-running the same bundled config.
- **Is V12's residual leak (AUC=0.7640) removable by further prompt work, or is it an inherent
  floor once captions are kinematically accurate?** Open. The diagnosis (genuine `braking`/
  `decreasing gap` vocabulary, not register violations) suggests a floor, but this wasn't tested
  by trying a V13 that also normalizes kinematic phrasing. Lower priority now that leakage is
  ruled out as B's failure mode (fixing it made the gap wider, not narrower).
- **Does a lower Stage-B learning rate recover P1's loss?** New, open. P1's Stage B reused
  A1_1761's from-scratch LR (2e-4) on LoRA weights already displaced by 12 epochs of semantic
  pretraining, and overfit roughly 2× faster (train_val_gap 0.870 vs A1's 0.370 by epoch 8).
  Untested whether a gentler Stage-B LR (e.g. 0.1-0.5×) closes some of the +0.0716 AP gap, or
  whether the underlying representation is simply not crash-compatible regardless of LR.
- **Does Stage B's final encoder still retain Stage A's semantic structure, or was it
  overwritten?** New, open — the "retention probe" scoped but not run. Needs a small script
  pairing Stage B's LoRA weights with Stage A's (frozen, discarded-at-inference) Predictor to
  measure retrieval@1 post-Stage-B. Would distinguish "semantics got erased" from "semantics
  survived but are irrelevant to crash prediction" as P1's failure mode.
- **B-shuffle control (captions permuted within class) — still not run**, despite being on the
  plan since W3. This is the cleanest available test of whether caption *content* (vs. caption
  *class*) does any work at all, and would strengthen any write-up of the negative result.
- **`semsup_b1_probe.py`'s checkpoint-selection bug (selected on cosine loss even under
  `--loss infonce`)** → **Fixed 2026-08-13**: now selects on `val_retrieval_top1_acc_clip`
  under InfoNCE, `val_loss` under cosine. The two retrieval helpers (`clip_level_retrieval_
  {detail,acc}`) were also lifted to module level 2026-08-17 so other scripts can import them
  (`semsup_train.py` now does, for P1's Stage-A selection).
