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

## Unresolved design questions

- **Which scale-up path?** **Partially resolved 2026-07-25**: before choosing (1) full
  267→~4.5k captioning, (2) intermediate ~500–1000, or (3) deprioritizing the thread — first
  run the free re-diagnostic: B1 with `--loss infonce` at the current n=267 (mechanism
  verified, just not yet executed for real). If it shows a real signal, scaling captions is
  well-motivated; if it's still at chance, the problem may be that text-derived (not
  vision-derived) captions carry no per-clip visual signal to learn (see EXPERIMENTS.md's
  documentation-consistency finding: the 267 captions never saw a frame), which scaling alone
  won't fix either. **The scale-up choice itself is still not decided** — this just adds a
  cheap, informative step before making it.
- **How to fix checkpoint selection** → **Resolved 2026-07-25 (T-3)**: per-clip aggregation,
  not a fixed-epoch rule. See the "Trust val_ap" entry above.
- **Does the semantic-aux loss beat crash-only LoRA on AP at a data scale where it could?**
  Still the central question, and now sharper: at n=267 with the (broken) cosine objective the
  answer was "not measurable" — B1 showed no video↔caption alignment above chance, and B added
  ~3.5× the checkpoint variance of A1 without a mean shift. **Whether that was the data or the
  objective is exactly what the pending InfoNCE re-run is designed to answer** — see the
  scale-up entry above and PROJECT_STATE.md's Next step. (T-2's bootstrap CI is now in hand and
  confirms the n=267 A1-vs-B gap is noise, not signal, under the current cosine objective — see
  EXPERIMENTS.md. It doesn't change what the InfoNCE re-run still needs to test.)
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
