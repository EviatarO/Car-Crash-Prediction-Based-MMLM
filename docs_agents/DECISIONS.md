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

## Unresolved design questions

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
- **Does the semantic-aux loss beat crash-only LoRA on AP at a data scale where it could?**
  Still the central question, and now sharper: at n=267 with the (broken) cosine objective the
  answer was "not measurable" — B1 showed no video↔caption alignment above chance, and B added
  ~3.5× the checkpoint variance of A1 without a mean shift. T-2's bootstrap CI confirms that
  n=267 A1-vs-B gap is noise under the cosine objective (see EXPERIMENTS.md). **Still genuinely
  open**: B1-InfoNCE now shows real video↔caption alignment exists at n=267 (see EXPERIMENTS.md)
  — but Stage B (the actual crash-prediction LoRA run) still uses the cosine loss;
  `semsup_train.py` was never given the `--loss infonce` option (noted as A-5 in the 2026-07-25
  review, blocked on a batching prerequisite). So the open question is now: does porting
  InfoNCE into Stage B, and/or the frame-grounded caption experiment above, actually move crash
  AP — neither has been tested yet.
- **LoRA target_modules (`query,key,value`) match both the 24 encoder layers AND the 12
  V-JEPA2 predictor-layer attention blocks** (same substring, both under `backbone`). Not
  decided whether to restrict to encoder-only via a more specific path pattern.
- **Thesis framing if the result stays null**: report a well-controlled negative result as a
  contribution (defensible — no prior art tests this exact train-only-language /
  vision-only-inference regime in crash anticipation), or treat the literature gap as a signal
  to deprioritize. **Not decided.**
- **Are the un-backed-up pod checkpoints disposable?** `/workspace/Car-Crash-Prediction-Based-
  MMLM/outputs/checkpoints/`: `e2_lora_100clips` (4.4G), `e3a_lora_89clips` (832M),
  `e3b_lora_267clips` (704M) exist **only** on that volume — verified NOT on HF Hub (only
  `e3a-epoch7-lora` and `e3b-ep3-lora` are). Not deleted pending explicit confirmation.
