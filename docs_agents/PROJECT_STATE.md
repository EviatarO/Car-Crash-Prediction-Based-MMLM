# Project State

## Goal
MSc thesis: collision anticipation on Nexar dashcam clips via Teacher→Student distillation.
Accepted shipped baseline: InternVL3.5-4B-Flash student, test AP=0.762 (677 clips). Active
thread: **semantic-supervision** — test whether a language-derived auxiliary loss (caption
embedding alignment) improves BADAS-Open's (V-JEPA2) crash-prediction representation, while
keeping inference vision-only (no added cost/latency). Central question: does the semantic-aux
loss (**B**) beat crash-only LoRA (**A1**, test_AP=0.900) beat the frozen baseline (**A0**,
test_AP=0.853)?

## Implementation status

| Stage | What | Status |
|---|---|---|
| A0 | Frozen BADAS-Open baseline, 677-clip Private test | **DONE**. AP=0.853, AUC=0.864. |
| A1_1761 | Crash-only LoRA control, V10 corpus (1,761 pool) | **DONE**, **this is the reference control**. test_AP=0.900, AUC=0.904 @ epoch 4. |
| B1_1761 (InfoNCE) | Predictor-only probe on the 1,761 pool | **DONE**. retrieval@1 32×/24× chance (row/clip). |
| B_1761 sequential | Continue A1 ep4 + InfoNCE | **DONE**, flat/declining. test_AP=0.897. Confounded design (see DECISIONS.md), not a real test. |
| B_1761 parallel | From-scratch, same seed as A1_1761, InfoNCE λ=0.05, **on the V10 (label-leaking) corpus** | **DONE**. test_AP=0.8901, AUC=0.8955. Paired bootstrap vs A1: ΔAP=−0.0105, 95% CI [−0.0173,−0.0040] — **B is significantly worse than A1, not noise.** Root cause found: see below. |
| `/project-review` audit | Full ML+code review of the semantic-supervision thread | **DONE** (2026-08-08). Found the corpus is label-collinear (caption text alone predicts the crash label at AUC=0.9643) — this is why B lost to A1. Report: `reports/project_reviews/2026-08-08_project_review.md` (gitignored). |
| A1-v2 | Full 4,446-window pool, encoder-only LoRA, cosine LR, dropout 0.10 | **DONE**, negative result. Best checkpoint test_AP=0.888 (< A1_1761's 0.900). Not pursued further — see DECISIONS.md. |
| GPU I/O speed fix | `TrainableBadasWrapper.prefetch_clips()` | **DONE**. 6.3× real speedup, verified. |
| Captioning speed fix | Concurrent OpenRouter calls + real usage logging | **DONE**. ~12× real speedup, verified, zero extra cost. |
| V12 neutral prompt | Register-neutral captioning (no GT/blind branch) | **DONE**, built and validated at n=18, n=100, and full corpus (n=1,761). |
| V12 full recaption | All 1,761 windows re-captioned with V12 | **DONE**. Real cost ~$64 total (verified portion: $32.82/900 clips). |
| Leakage gate (task 1.6) | Full-corpus TF-IDF+GroupKFold AUC, target <0.75 | **DONE, narrow miss**: V10 AUC=0.9643 → V12 AUC=0.7640 (43% reduction in excess-over-chance signal). User accepted the miss and approved proceeding. |
| **B-v2** | From-scratch, same seed/recipe as A1_1761, InfoNCE λ=0.05, **on the corrected V12 corpus** | **DONE — lost.** test_AP=0.8796 (epoch 2, best-by-val), AUC=0.8905. Paired bootstrap vs A1_1761: ΔAP=+0.0189 in A1's favour, 95% CI [0.0099, 0.0285], P(B>A1)=0.0%. **Wider** than B_1761-parallel's gap on the leaky corpus — so fixing the caption leak did NOT close it. Loses under every selection rule incl. best-on-test (0.8931). **But see "Why B is not yet a valid test" below — this run deviated from the plan in two ways.** |
| **B-v3** | B-v2 re-run with the two execution defects corrected | **NOT STARTED** — the actual next step. |

## Why B is not yet a valid test of the thesis question

Two defects, both found 2026-08-12, both present in **B_1761-parallel AND B-v2**:

1. **The Predictor was cold-started, contrary to the written plan.** `2026-07-07_Plan Semantic-
   Supervision...` line 113 fixes the run order `0 → A0 → B1 → A1 → B`; lines 122-129 define B1
   as "train ONLY the Predictor" against a frozen trunk, whose stated aim includes "**(2)
   warm-start the Predictor**"; line 141 specifies B uses "Predictor (**warm-started from B1**)".
   Both B runs recorded `predictor_init: null`. Consequence: the semantic gradient must pass
   *through* a randomly-initialized Predictor to reach the ViT-L, so early-epoch updates push the
   trunk in near-random directions. **B-v2's selected checkpoint was epoch 2** — formed exactly
   when the Predictor was noisiest. (Note: the plan warm-starts but does NOT freeze the
   Predictor — line 145 says "Train: ViT-L LoRA + Predictor".)
2. **The arms differ by more than λ.** `clip_grad_norm_(trainable, 1.0)` clips ONE global norm
   across all trainable params. In A1 that is LoRA alone (2.8M); in B it is LoRA + Predictor
   (1.25M) + `log_tau` sharing the same budget 1.0. Large early Predictor gradients inflate the
   global norm, and the resulting scale-down is applied to the LoRA's *crash* gradients too — B's
   trunk gets systematically smaller effective updates than A1's, for reasons unrelated to
   semantics. Whether it actually bites depends on whether the clip is active, which was not
   logged (now measurable via `grad_norm_crash`).

Neither defect is the semantic idea failing; both are execution. Correcting them is ~1 day.

## What actually happened this session (only the parts that matter going forward)

**1. B_1761 parallel confirmed a real negative result, but on a broken corpus.** The clean,
matched-init comparison (B vs A1_1761, same seed, only λ/loss differ) showed B losing to A1 by
a statistically real margin (paired bootstrap excludes zero). This looked like "semantic
supervision doesn't help" — but the `/project-review` audit found the actual cause: **the V10
caption corpus leaks the label.** Positives were captioned in GT-informed mode ("this DOES end
in a collision, explain the mechanism"), negatives in blind mode — two different prompt
registers producing two different vocabularies. A plain TF-IDF+logistic-regression classifier on
caption text alone, GroupKFold by `video_id`, predicts the crash label at **AUC=0.9643** — higher
than the vision model's own test AP. So B's underperformance couldn't be trusted as "semantic
supervision doesn't work" — it could just as easily be "a redundant, noisier copy of the label
hurts, unrelated to real semantics." **Any future work must verify a caption corpus doesn't leak
the label before training on it** (the TF-IDF+GroupKFold check above is cheap and should be a
standing QA gate).

**2. The GPU trainer was never I/O-optimized — direct profiling found the real bottleneck.**
A1-v2 initially ran at ~97 min/epoch on an RTX 5090 despite the GPU being idle 76%+ of the time.
Root-caused via direct profiling (not inference from utilization graphs): reading+decoding 16
JPEGs takes ~1.17s/window; the GPU forward pass itself is unmeasurably fast by comparison.
Fixed with `TrainableBadasWrapper.prefetch_clips()` — a `ThreadPoolExecutor`-based concurrent
prefetch pipeline (default 8 workers, depth 16) applied to all three hot loops (train, val,
test-scoring). Verified in isolation (5.3× in a controlled benchmark) and in the live resumed
run (6.3×, epoch 1-2 pre-fix ~97 min avg → epoch 3 post-fix 15.5 min). The algorithm itself was
unit-tested locally (order preservation, error handling, race-condition stress test) before
touching the pod, since an index desync here would silently corrupt training data.

**3. The same pattern applied to captioning — OpenRouter calls were serial, latency-bound, and
fixable the same way.** ~11.8s/clip serial → ~1s/clip at `--concurrency 16` (~12×), with **zero
extra cost** (OpenRouter bills by tokens processed, not concurrency or wall-clock — verified).
Real per-call token usage (previously silently discarded) is now logged to a
`<output>.usage.jsonl` sidecar, so cost is measured going forward, not estimated from a stale
doc figure.

**4. A1-v2 (bigger pool + new recipe) underperformed the original A1_1761 and was not pursued
further.** Full 4,446-window pool (natural 13.2% hard-example rate) + cosine LR + dropout 0.10
+ encoder-only LoRA gave test_AP 0.868-0.888, below A1_1761's 0.900. Most likely explanation
(not proven): A1_1761's 1,761-window pool is *enriched* to 33% hard examples, and that
enrichment may have been genuinely helping, not just working around the earlier
training-inversion bug. Untangling "pool distribution" from "recipe bundle" as separate causes
was explicitly deprioritized in favor of moving straight to the real research question (B vs A1
on a clean corpus) — see DECISIONS.md.

**5. V12 (register-neutral prompt) substantially reduced but did not eliminate the corpus
leak.** No GT/blind branch (single `build_prompt()`, no arguments), a closed four-way
`gap_trend` vocabulary (`decreasing/increasing/constant/none_visible`) replacing free-text
`closing_dynamic`, and symmetric alarm/reassurance word bans on top of V10's existing
outcome-word ban. Validated at increasing scale before committing to the full recaption:
n=18 → 66.7% leakage-judge accuracy (not significant, p≈0.12); n=100 (18 + 82 fresh sampled
clips) → 72.0%, p<0.0001 (real, not noise); full corpus (n=1,761) → **AUC 0.9643 → 0.7640**
(target was <0.75, missed by 0.014). A local TF-IDF diagnostic on the n=100 sample found the
*residual* signal comes from genuine kinematic vocabulary (`braking`, `decreasing gap`) that
correlates with the label because it's physically real — not from any banned word (0 lexical
violations in either the n=18 or n=100 samples). **User's call: accept the near-miss, proceed
to B-v2, report the residual leak honestly.**

## Current RunPod pod state
- Working repo: `/workspace/MMLM_AI` (persistent network volume — survives across different pod
  instances, confirmed again this session).
- **This session's pod IP/port changes on every reconnect** — ask the user for the current one.
  Every reconnect is a fresh CONTAINER: reinstall packages (`pip install --break-system-packages
  badas openpyxl pyyaml scikit-learn pandas pillow matplotlib seaborn huggingface_hub
  albumentations sentencepiece peft transformers protobuf`), restore the HF token
  (`mkdir -p /root/.cache/huggingface && cp /workspace/.cache/huggingface/token
  /root/.cache/huggingface/token`), and if SSH refuses, have the user paste the EXISTING public
  key (`~/.ssh/id_ed25519.pub` — never generate a new one) into the pod's web terminal's
  `~/.ssh/authorized_keys`.
- **This session's I/O bottleneck was pod-specific in magnitude** (this particular instance's
  network volume was slower than prior ones) but the underlying fix (concurrent prefetch) is
  now the default behavior regardless of pod, so future pods should be fast automatically.
- **B-v2 is running right now** — `/workspace/semsup/b_v2_1761/`, log
  `/workspace/semsup/b_v2_1761_train.log`. Check epoch progress before doing anything else.

## Known bugs / gotchas (this session)
- **`load_training_examples()` expects `caption` + `gt_verdict` fields; V12's raw output uses
  `caption_neutral` + `event_occurs`.** A converted file
  (`Caption_V12_Neutral_1761_fortrain.jsonl`) was built with both aliases added — use that for
  training, not the raw `Caption_V12_Neutral_1761.jsonl`.
- **Benchmark-testing concurrency settings against fresh/empty `--out` files re-captions
  overlapping rows every time** (resume-skip is per-output-file, not global) — wasted ~64
  duplicate API calls (~$2-3) this session. Use non-overlapping `--limit` ranges or a shared
  test file when benchmarking.
- **`semsup_b1_probe.py`'s `evaluate()` selects checkpoints on cosine loss even when
  `--loss infonce`** (found by `/project-review`, not yet fixed). The currently-used
  `predictor_b1_ep028.pt` checkpoint (from B1_1761) was selected by the wrong criterion — a
  worse checkpoint (epoch 43) scored higher on the metric that actually matters
  (`val_retrieval_top1_acc_clip`: 0.1267 vs the selected 0.1086). Low priority since B-v2 warm-
  starts the predictor from scratch, not from this checkpoint, but flagged for anyone reusing
  `semsup_b1_probe.py`.
- **A1-v2's val_ap is not comparable to A1_1761's.** A1-v2 trained on the natural
  13.2%-hard-example pool; A1_1761 trained on the 33%-enriched 1,761 pool. Only `test_AP` on
  the fixed 677-clip set is comparable across arms.
- **`--start-epoch` resumes restart the cosine LR schedule's warmup fresh** rather than
  continuing the original single-run trajectory (a known, accepted quirk, not a bug — see
  `semsup_train.py`'s scheduler construction comment).

## Important commands
```bash
# B-v2 (the real test) — RUNNING, for reference/resume
python -u semsup_train.py --config ../configs/e4_stageA.yaml \
    --lora-target-modules query,key,value \
    --captions-path ../../outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl \
    --semantic-weight 0.05 --semantic-loss infonce --infonce-tau-init 0.07 \
    --epochs 8 --grad-accum 8 --seed 0 --keep-top-k 8 \
    --out-dir /workspace/semsup/b_v2_1761 \
    --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
    --test-frames-root ../../dataset/test
# prefetch-workers/depth default to 8/16 (the speed fix) - no need to pass explicitly.
# NOTE: --lora-target-modules query,key,value (legacy comma form, NOT the encoder-only regex) -
# deliberately matches A1_1761 exactly so semantic_weight/loss is the ONLY difference.

# Resume an interrupted run
#   add: --lora-init <out_dir>/epoch_0N/lora_adapter \
#        --predictor-init <out_dir>/epoch_0N/predictor.pt   (semantic runs only) \
#        --optimizer-init <out_dir>/epoch_0N/optimizer.pt --start-epoch <N+1>

# Encoder-only LoRA (excludes the V-JEPA2 predictor stack) - use the regex form:
--lora-target-modules 're:backbone\.encoder\.layer\.\d+\.attention\.(query|key|value)'

# Re-caption a pool with the neutral V12 prompt
python student_training/scripts/semsup_caption_promptbakeoff.py \
  --manifest <manifest.jsonl> --frames-root dataset/train \
  --prompt v12 --model google/gemini-3.6-flash \
  --frame-size 0 --detail high --temperature 0.1 --concurrency 16 \
  --out <output.jsonl>
# Real cost/token totals print live and log to <output.jsonl>.usage.jsonl.

# Full-corpus label-leakage check (the decisive gate, target AUC < 0.75) -
# TfidfVectorizer(1,2-gram) + LogisticRegression + GroupKFold(5) by video_id
# on caption_neutral (or caption) vs the true label. See EXPERIMENTS.md for the exact snippet.

# Sample a balanced, distinct-clip validation set (for extending a leakage-judge check to n>18)
python student_training/scripts/sample_val_check_clips.py \
    --manifest dataset/manifests/train4500_hires.jsonl \
    --val-manifest dataset/manifests/val_e3a.jsonl \
    --n-per-class 41 --seed 0 --out dataset/manifests/val82_v12_check.jsonl

# B1 (predictor-only probe) - frozen trunk + frozen SigLIP, trains ONLY the Predictor.
# This is the plan's mandated pre-step for any Stage B run (see "Next step" below).
python -u semsup_b1_probe.py --config ../configs/e4_stageA.yaml \
    --epochs 100 --out-dir /workspace/semsup/b1

# Local CPU smoke test of a training run (mechanics only; --limit 8 hits the
# single-class val path, so val_ap is meaningless - this checks wiring, not results)
#   add: --epochs 2 --grad-accum 1 --limit 8 --test-limit 3

# Crash-vs-semantic gradient angle (2026-08-12): on by default for semantic runs,
# --grad-cosine-every 8. Diagnostic only - autograd.grad() does not touch .grad, so
# training is bit-identical with it on or off. Reported per epoch as
# grad_cos_mean / grad_cos_frac_neg / grad_norm_{crash,sem} in epoch_metrics.jsonl.
#   cos<0 and frac_neg>0.5  -> destructive interference (aux fights crash)
#   cos~0, small |g_sem|    -> aux is weak noise, not conflict
#   watch lambda*|g_sem|/|g_crash| GROW across epochs as the crash loss saturates
```

## Git state
Branch `main`, **HEAD = `6cc67c7`, matches `origin/main`** (0 ahead / 0 behind — a previous
session's commit was pushed). **Substantial uncommitted work from this session**, not yet
staged or committed:
- Modified: `semsup_train.py` (evaluate_val merge, prefetch integration, LR schedule,
  lora-dropout arg, NaN-safety, encoder-only regex support), `semsup_common.py` (`forward_clip`,
  `prefetch_clips`, regex LoRA target support), `semsup_caption_promptbakeoff.py` (concurrent
  fetch, usage logging, V12 touchpoints), `docs_agents/*.md`.
- New, untracked: `prompts/PROMPT_SEMSUP_V12_NEUTRAL.py`,
  `student_training/scripts/{build_pool_from_manifest.py, sample_val_check_clips.py,
  plot_semsup_curves.py}`, `teacher_distillation/scripts/{score_val18_neutral.py,
  leakage_judge_100.py}`, `docs_agents/ARCHITECTURE_BLOCKS.md`.
- Not committed (by convention, weights/large outputs go to HF Hub not git): checkpoints under
  `outputs/e4_vjepa_reason/{a1_v2_full,b_v2_1761}/`, the V12 caption JSONLs and usage sidecars.
- User pushes themselves — do not `git push`. Commit only if explicitly asked.

## Next step

**Check B-v2's progress first** — it was launched this session and may have finished, be
mid-run, or (less likely) have failed. `ssh` to the pod (get current IP/port from the user),
`tail /workspace/semsup/b_v2_1761_train.log`.

**If finished:** compare test_AP against A1_1761's 0.900 with a paired bootstrap on the 677
per-clip scores (same method used for the B_1761-parallel comparison). This is the real answer
to the thesis question, on a corpus now known to leak far less. Report honestly whichever way
it lands, including the residual-leak caveat (AUC=0.764, not fully clean).

**If B also loses:** report a well-controlled negative result — real signal is close to (if not
identical to) what B_1761-parallel showed on the leaky corpus, suggesting the auxiliary genuinely
doesn't help at this scale/design, not just an artifact of label leakage. Consider the
shuffled-caption control (W3 plan, not yet run) as the final confirming experiment before
concluding.

**If B wins:** report `B − A1` with a bootstrap CI, and be explicit that the residual 0.264
excess-AUC leak means the result should be treated cautiously, not as fully clean evidence of
semantic grounding.

**Secondary, lower priority, unchanged from before:**
- Why does A0 score better on the general train pool (13.2% error) than the 677-clip test set
  (23.6%)? Still unexplained.
- Whether A1-v2's underperformance was pool-distribution or recipe-driven — not isolated,
  deprioritized in favor of the B-v2 result.

## Known bugs / gotchas (all fixed — don't re-hit these)
- **`| tail -N` on a backgrounded command masks the real exit code.** A crashed Python process
  (concurrent BADAS-loading resource contention, below) reported "completed, exit 0" via the
  pipe — `tail`'s exit code, not the process's. An empty output directory was the actual
  tell. Redirect straight to a file (`> log.txt 2>&1`) and check `$?` explicitly instead of
  piping through `tail` for anything you need a trustworthy exit code from.
- **Two BADAS-loading processes running concurrently on this machine can silently crash one
  of them** (both hit the same on-disk HF cache at once) — confirmed by re-running the exact
  same command alone, which then succeeded normally. Run local smoke tests sequentially, not
  in parallel background jobs, if they both load BADAS.
- **`peft`'s `save_pretrained()` crashes on BADAS**: it auto-generates a model card *before*
  writing adapter weights, assuming `base_model.config` supports `in` (a HF
  `PretrainedConfig`). BADAS's V-JEPA2 uses a plain `ModelArgs` dataclass → `TypeError`, and
  no checkpoint is ever written. Fixed in `semsup_train.py` by stubbing
  `create_or_update_model_card`.
- **Test scoring used the LAST epoch, not the best one**: `semsup_train.py` tracked
  `best_epoch` but never reloaded that checkpoint before scoring. Fixed via
  `set_peft_model_state_dict`.
- **Raw `NaN` written into JSON** on degenerate (single-class val) runs — invalid per strict
  RFC-8259 even though Python's `json.load` accepts it. All NaN-prone fields now emit `null`.
- Windows console (cp1255) crashes on emoji `print()` inside the `badas` package → always set
  `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`.
- RunPod `/workspace` has a **per-user quota** far below the cluster-wide free space `df -h`
  reports. Verify per-pod; don't trust `df`.
- Never `pip install -U torch` on a provisioned RunPod image (desyncs torch/CUDA from driver).
- BADAS needs `albumentations`; SigLIP's tokenizer needs **`sentencepiece` AND `protobuf`**
  (protobuf was missing from the original runbook and broke a live run).
- `SiglipModel.get_text_features()` returns `BaseModelOutputWithPooling`, not a tensor —
  handled defensively in `siglip_text_embed()`.
- `mv` across filesystems (overlay→network volume) copies then deletes; if it fails on quota
  mid-way it can leave 0-byte stubs. Use `cp` first, verify, then remove the source.
- Backgrounded Bash piped through `tail` shows an empty file until exit (block buffering, not
  a hang) — use `python -u`.
- **OpenRouter `preview`-tagged model aliases are not stable over time.** `google/gemini-3.1-
  pro-preview` silently changed behavior between when v6's baseline was recorded and when it
  was rerun (83.3%/6.78 → 50.0%/4.61 on the identical prompt/clips/settings). Never trust a
  historical baseline for a `preview` model without a same-day reproducibility check first.
- **Thinking/reasoning models can return an empty, unparseable response if `max_tokens` is
  too small** (the internal reasoning consumes the budget before the JSON output). Qwen3.7
  Flash failed on 4/18 clips at `max_tokens=8192`; raising to 20000 fixed some but broke
  *different* clips on retry (provider-side flakiness, likely OpenRouter load-balancing the
  model slug across backend hosts, not a deterministic token-budget issue) — use `--resume`
  on `semsup_v6_control_rerun.py`/`semsup_caption_promptbakeoff.py` to retry only the gaps.
- **`tar` writes the correct file COUNT but 0-byte CONTENT when a disk quota is hit mid-stream**
  — it creates every archive entry regardless of whether the write actually succeeded. A
  verification check that only does `len(os.listdir(d)) == 16` will report a directory as
  complete even when every frame is empty. Always additionally check `os.path.getsize(f) > 0`
  per file. Caught this 2026-08-01 after a "0 missing/incomplete" check passed on a transfer
  that had actually corrupted 1,674/2,946 directories.
- **Python's default text-mode `open(path, 'w')` on Windows writes `\r\n`, not `\n`.** A file
  list built this way and fed to `tar -T filelist` makes every entry fail with `Cannot stat:
  No such file or directory` (the literal filename has a trailing `\r`), and — worse — `tar`
  still exits 0 if piped through another `tar` on the receiving end that successfully extracts
  an empty stream. Always write file lists with `open(path, 'w', newline='\n')` on Windows.
- **Chunking a mixed pos/neg pool by plain `sorted(video_id)` breaks class balance per chunk.**
  `dataset/train.csv`'s raw ids place **every** positive at ids 0–1039 and **every** negative at
  1040–2139 — a clean split, not interleaved. A naive sort puts whole classes in separate
  chunks. Fixed in `semsup_extract_promptbakeoff_frames.py`/`run_train4500_pipeline.py` by
  explicitly round-robin-interleaving the two classes before chunking.
- **`val_e3a.jsonl` has no `t_seconds` field.** It lives in `outputs/prompt_bakeoff/
  highres_test.jsonl` (6 of the 18 clips) + `v6_hires_full18.jsonl` (the other 12) — together
  exactly cover the 18 val clips and are the runs that used these same `_hires` frames, so
  they're authoritative. Other files in `outputs/prompt_bakeoff/` disagree on `t_seconds` for
  some of these clips (different experiment generations placed windows differently) — do not
  merge them blindly; fail loudly if a clip's `t_seconds` doesn't resolve from the two
  authoritative files rather than falling back to a disagreeing source.
