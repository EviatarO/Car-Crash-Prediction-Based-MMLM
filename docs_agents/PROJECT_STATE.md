# Project State

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

**Every semantic-supervision attempt has lost, six real GPU runs, gap widening as execution
got cleaner — this is itself the finding:**

| Arm | Design | test_AP | ΔAP vs A1_1761 (paired bootstrap, excludes zero every time) |
|---|---|---|---|
| B_1761 parallel | Joint, V10 (leaky) corpus | 0.8901 | +0.0105, CI [0.0040, 0.0173] |
| B-v2 | Joint, V12 (clean) corpus, cold-start predictor | 0.8796 | +0.0189, CI [0.0099, 0.0285] |
| B-v3 | Joint, V12, warm-started + per-group clip | 0.8768 | +0.0218, CI [0.0117, 0.0325] |
| B-v3 ext12 | Same, 12-epoch exploratory extension | 0.8655 | +0.0330, CI [0.0156, 0.0519] |
| **P1 two-stage** | **Semantic-pretrain → crash-finetune** | **0.8266** | **+0.0716, CI [0.0477, 0.0977]** |

P1 (the most structurally different design tried) is the **worst result of all**, and falls
**below the frozen A0 baseline (0.853)**.

**Three diagnostics ran this session and ruled out the explanations we were assuming:**
1. **Pooled-tap B1 probe** — caption info survives the classifier's 2560× pooling bottleneck at
   22× chance (vs 31× from the raw patch grid). Rules out "information can't reach the
   classifier."
2. **P3 (Δpatches vs Δpooled, paired bootstrap CI)** — the real A1→B weight difference reaches
   the pooled representation *at least as well as* a random perturbation of equal size (paired
   diff 95% CI [0.00143, 0.00163], excludes zero, real > random). Rules out "the gradient is
   routed around the pooler."
3. **Crash-vs-semantic gradient-angle probe** — cos(crash,sem) ≈ 0, drifting mildly negative
   over training (not strongly opposed). Rules out "the objectives actively fight."

**Conclusion so far: the semantic signal reaches the classifier-relevant representation; it
just doesn't help once there.** P1 tested a genuinely different mechanism (training order
instead of routing) and lost worse, with a **specific measured cause**, not a mystery: Stage
B's `train_val_gap` grew to more than double A1_1761's under an *identical* LR schedule
(0.870 vs 0.370 by epoch 8) — warm-starting LoRA from Stage A's already-adapted weights and
reusing A1's from-scratch learning rate overfits much faster.

**Caption corpus:** V12 (register-neutral, no GT/blind branch) substantially reduced label
leakage (TF-IDF+GroupKFold AUC 0.9643 → 0.7640, narrow miss on the <0.75 target, accepted).
**But fixing the leak did not close the AP gap — it widened it**, ruling out leakage as B's
primary failure mode.

## What's genuinely still untested (not "the last lever" — several remain)
- **Does a lower Stage-B learning rate recover P1's loss?** Not tried. P1 reused A1's
  from-scratch LR on a warm-started checkpoint; that specific choice is the leading suspect for
  the overfitting, not "two-stage training is inherently bad."
- **Retention probe** — does Stage B's final encoder still retain Stage A's semantic structure?
  Scoped, not built (needs pairing Stage B's LoRA with Stage A's discarded Predictor).
- **B-shuffle control** (captions permuted within class) — on the plan since W3, never run. The
  cleanest available test of whether caption *content* matters at all, independent of class.
- **B-rev** (reverse the projection direction — text into vision space instead of vision into
  text) — proposed, not implemented.
- **λ sweep** — never run. A standing reviewer objection to the joint-training results (though
  now secondary, since P1 has no λ at all and still lost worst).
- **Corpus scale-up** (full 4,446-window pool, ~$11 via Qwen3-VL vs ~$162 via Gemini) — not
  done. The B1 scaling curve (13×/19×/31× chance at 25/50/100% of captions, still rising) is
  the strongest positive evidence that more data could help, independent of the routing/timing
  questions above.

## Known bugs / gotchas (this thread, still open)
- **B-v3's local `test_summary.json`/`epoch_metrics.jsonl` were overwritten** when pulling a
  later 12-epoch extension's results into the same local folder. The correct 8-epoch files
  still exist on the pod's persistent volume (`/workspace/semsup/b_v3_1761/`, distinct from
  `..._ext12/`) — not yet re-pulled. Low priority, doesn't affect any reported number (those
  came from the bootstrap JSON, which survived intact).
- **`load_training_examples()` expects `caption`+`gt_verdict`; V12's raw output uses
  `caption_neutral`+`event_occurs`.** Use `Caption_V12_Neutral_1761_fortrain.jsonl` (aliases
  added) for training, never the raw `Caption_V12_Neutral_1761.jsonl`.
- **`evaluate_val()`'s return signature changed 2026-08-17**: 4-tuple → 5-tuple (added
  `retrieval_stats` dict). The one call site in `semsup_train.py` is already updated; any other
  script importing this function directly would break silently on unpacking. None currently do.

## Current RunPod pod state
- Working repo: `/workspace/MMLM_AI` on the **persistent network volume**
  (`mfs#euro.runpod.net:9421`) — survives across different pod instances/containers.
- **Pod IP/port changes on every reconnect** — ask the user for the current one. Every
  reconnect is a **fresh container**: reinstall packages (`pip install --break-system-packages
  badas openpyxl pyyaml scikit-learn pandas pillow matplotlib seaborn huggingface_hub
  albumentations sentencepiece peft transformers protobuf`), restore the HF token
  (`mkdir -p /root/.cache/huggingface && cp /workspace/.cache/huggingface/token
  /root/.cache/huggingface/token`), and if SSH refuses, have the user paste the **existing**
  public key (`~/.ssh/id_ed25519.pub` — never generate a new one) into the pod's web
  terminal's `~/.ssh/authorized_keys`.
- **Last known state (end of previous session): pod was idle, user asked about stopping it,
  confirmed safe (no running jobs, volume persists) — likely stopped.** Assume a fresh
  reconnect is needed.
- All P1 checkpoints (`/workspace/semsup/p1_stageA/`, `/workspace/semsup/p1_stageB/`) and
  earlier runs (`a1_1761`, `b_v2_1761`, `b_v3_1761`, `b_v3_1761_ext12`, `b1_tap_*`,
  `b1_v2_*pct`) remain on the volume regardless of pod state.

## Important commands
```bash
# P1 Stage A (semantic-only, select on retrieval — the run that already happened)
python -u semsup_train.py --config ../configs/e4_stageA.yaml \
    --lora-target-modules query,key,value \
    --captions-path ../../outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl \
    --crash-weight 0.0 --semantic-weight 1.0 --semantic-loss infonce --infonce-tau-init 0.07 \
    --select-by retrieval --epochs 12 --grad-accum 8 --seed 0 --keep-top-k 8 \
    --clip-grad-per-group --out-dir /workspace/semsup/p1_stageA

# Gate: score a Stage-A checkpoint's encoder + UNCHANGED frozen head, no training
python -u p1_stageA_gate.py --config ../configs/e4_stageA.yaml \
    --lora-adapter /workspace/semsup/p1_stageA/epoch_10/lora_adapter \
    --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
    --test-frames-root ../../dataset/test \
    --out /workspace/semsup/p1_stageA_gate_ep10.json

# P1 Stage B (crash-only, warm-started from Stage A) — try a LOWER lr here first if re-running
python -u semsup_train.py --config ../configs/e4_stageA.yaml \
    --lora-target-modules query,key,value \
    --captions-path ../../outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl \
    --semantic-weight 0.0 \
    --lora-init /workspace/semsup/p1_stageA/epoch_10/lora_adapter \
    --select-by val_ap --epochs 8 --grad-accum 8 --seed 0 --keep-top-k 8 \
    --out-dir /workspace/semsup/p1_stageB_v2 \
    --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
    --test-frames-root ../../dataset/test
    # add --lr <lower value> to test the overfitting hypothesis (default 2e-4, same as A1_1761)

# P3 diagnostic (does the semantic gradient reach the pooled representation?)
python -u p3_delta_patches_vs_pooled.py --config ../configs/e4_stageA.yaml \
    --a-lora /workspace/semsup/a1_1761/epoch_04/lora_adapter \
    --b-lora <any-B-checkpoint>/lora_adapter \
    --n-clips 40 --n-noise 20 --n-boot 5000 --seed 0 \
    --out /workspace/semsup/p3_result.json

# Caption label-leakage gate (persisted script, reuse for any new corpus)
python teacher_distillation/scripts/caption_leakage_gate.py \
    --captions <corpus.jsonl> --caption-field <field> --label-field <field> \
    --positive-value <value> --out <out.json>

# Paired bootstrap, any two arms' test_results_ep*.jsonl
python student_training/scripts/paired_bootstrap_ab.py \
    --a <a>/test_results_epNN.jsonl --a-name A --b <b>/test_results_epNN.jsonl --b-name B \
    --n-boot 5000 --seed 42 --out <out.json>

# Encoder-only LoRA (excludes the V-JEPA2 predictor stack) — regex form
--lora-target-modules 're:backbone\.encoder\.layer\.\d+\.attention\.(query|key|value)'

# Resume an interrupted run
#   add: --lora-init <out_dir>/epoch_0N/lora_adapter \
#        --predictor-init <out_dir>/epoch_0N/predictor.pt   (semantic runs only) \
#        --optimizer-init <out_dir>/epoch_0N/optimizer.pt --start-epoch <N+1>
```

## Git state
Branch `main`, **HEAD = `46ec1b3`, matches `origin/main`** (0 ahead / 0 behind — pushed).
Uncommitted right now: only `docs_agents/{ARCHITECTURE,DECISIONS,EXPERIMENTS,PROJECT_STATE}.md`
(this handoff's updates). User pushes themselves — do not `git push`. Commit only if
explicitly asked.

## Next step

**Decision point, not a default action**: five real negative results is a lot of evidence.
Before spending more GPU time, the open question is which (if any) of the untested items above
is worth pursuing vs. writing this up as a well-controlled, mechanistically-diagnosed negative
result. If continuing:

1. **Cheapest, most targeted**: re-run P1 Stage B with a reduced LR (see command above) — tests
   the specific mechanism found (overfitting under an unchanged LR), not a new hypothesis.
2. **B-shuffle control** — cheap, strengthens the write-up regardless of outcome.
3. **Retention probe** — cheap, would clarify whether P1's failure is "forgot the semantics" or
   "kept them but they're irrelevant."

**What's already publishable regardless:**
1. A1_1761 beats the published BADAS-Open baseline (0.900 vs 0.853, +0.047).
2. A rigorously controlled negative result for language-only-at-train-time supervision, with
   multiple diagnosed (not guessed) mechanisms — label leakage ruled out, routing ruled out,
   gradient opposition ruled out, overfitting-under-warm-start identified and measured.
3. The training-inversion failure mode (training only on mined failures against a frozen head
   collapses to AP 0.333 / AUC 0.163) — a separate, reusable lesson.

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
- **A single benchmark test against a fresh/empty `--out` file doesn't dedupe against earlier
  test runs** (resume-skip is per-output-file, not global) — wasted ~64 duplicate captioning
  API calls across 4 separate concurrency-benchmark runs. Use non-overlapping `--limit` ranges
  or a shared test file when benchmarking anything that writes a resumable output file.
