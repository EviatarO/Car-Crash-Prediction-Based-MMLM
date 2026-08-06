# Project State

## Goal
MSc thesis: collision anticipation on Nexar dashcam clips via Teacher→Student distillation.
Accepted shipped baseline: InternVL3.5-4B-Flash student, test AP=0.762 (677 clips). Current
active thread: **semantic-supervision** — test whether a language-derived auxiliary loss
(caption embedding alignment) improves BADAS-Open's (V-JEPA2) crash-prediction representation,
while keeping inference vision-only (no added cost/latency). Blocker: only 267/4,500 target
captions exist, and B (semantic-aux) vs A1 (crash-only) is inconclusive at that scale (see
below). **Current sub-thread (2026-08-01, active): score the entire ~4,500-window train pool
through the frozen A0 scorer (inference only, nothing trains) to find where BADAS-Open actually
fails, so caption budget can be spent where it matters instead of uniformly.**

## Implementation status (semantic-supervision route, stages A0→B1→A1→B)

**The full A0→B1→A1→B pipeline is now executed end-to-end at n=267.** Mechanism verified;
scientific question unresolved (all deltas are inside the noise at this data scale).

| Stage | What | Status |
|---|---|---|
| A0 | Frozen BADAS-Open baseline, 677-clip Private test | **DONE** (2026-06-24). AP=0.853, AUC=0.864 — within 0.86±0.03 target. |
| Module discovery | BADAS's real LoRA-able layer names | **DONE**. `query,key,value` under `backbone.encoder.layer.{0-23}`. Zero overlap with crash head. |
| Stage 0 | Caption dataset for the semantic target | **PARTIAL**. 267 rows (89 clips × ≤3 TTE), rephrased from existing teacher `final_reasoning` (not fresh vision-captioning). Target ~4.5k — not scaled. |
| B1 | Predictor-only probe (frozen BADAS + frozen SigLIP) | **DONE** (2026-07-21, real GPU). retrieval_top1_acc = 0.0196 = chance = collapse-control. No learned video↔caption alignment at this scale. |
| A1 | Crash-only LoRA fine-tune (control) | **DONE** (2026-07-23, real GPU). val-selected test_AP=0.8638 vs A0's 0.853 — flat. |
| B | Crash + semantic-aux LoRA (treatment) | **DONE** (2026-07-23, real GPU). val-selected test_AP=0.8574 — inconclusive vs A1; see EXPERIMENTS.md for why the sign flips by aggregation rule. |

**2026-07-25: `/project-review` (new user-level skill) audited the semantic-supervision
thread and found the null result may be an artifact of a broken loss function, not proof
data is too small.** Verified from the recorded metrics: B1's cosine-regression objective's
analytic minimizer for a video-blind (learns-nothing) predictor is `target_mean/‖target_mean‖`,
worth `1-0.8648=0.1352`. The real trained run reached `0.1345` — beat the degenerate solution
by **0.53% of the available range**, with retrieval@1 exactly at chance. Scaling captions
17× without fixing the objective would reproduce this null at far higher cost. Fixes below
implement and verify the remedy; **the actual diagnostic re-run at scale has not happened
yet** (see Next step).

## Current RunPod pod state
- Working repo: **`/workspace/MMLM_AI`** (persistent network volume, confirmed 2026-08-01 via
  `git remote -v` → `EviatarO/Car-Crash-Prediction-Based-MMLM`). NOT `/root` — that is wiped on
  every new/recreated container (confirmed again 2026-08-01: this session's pod had zero
  pip packages and no HF login despite `/workspace` holding weeks-old data).
- Frame data: `dataset/train` and `dataset/test` are **symlinks** into the pod's pre-existing
  `/workspace/data/train_HiRes` and `/workspace/data/test_HiRes`.
- **`/workspace` has an account/volume-level write quota that is NOT what `df -h` reports.**
  `df -h /workspace` shows the *cluster's* total free space (2.3P / 761T available as of
  2026-08-01) regardless of whether writes actually succeed. Confirmed by a live 500MB test
  write failing at 0 bytes while `df` still showed hundreds of TB free. **Always verify with a
  real test write, never trust `df` alone, before pushing a large transfer.**
- **Quota was raised by the user (+15GB) and confirmed fixed (2026-08-01)** via a live 2GB
  test write. **Pod is currently STOPPED again (normal end-of-session stop, not a blocker this
  time)** — only Jupyter was running, all pipeline work was already complete and pulled
  locally before stopping. SSH access: local `~/.ssh/config` has a `Host runpod-train4500`
  alias (**IP/port changes on every pod restart — even a Stop→Start got a new address once
  this session** — re-point `HostName`/`Port` after the pod comes back up, then re-test); the
  local public key needs re-adding to `~/.ssh/authorized_keys` if the pod comes back as a
  genuinely new container (happened once this session despite being a Stop, not a Terminate —
  don't assume `/root` state survives, verify with `python3 -c "import badas"` first).
- Deps must be reinstalled on every NEW/recreated container: `pip install --break-system-packages
  badas openpyxl pyyaml scikit-learn pandas pillow matplotlib seaborn huggingface_hub
  albumentations` (Python 3.12 here is PEP-668 "externally managed" — plain `pip install` refuses;
  `--break-system-packages` is fine, this is a disposable GPU container). `hf auth login` must be
  re-run (BADAS-Open + its `nexar-ai/nexight` dependency are both gated HF repos).
- **HF token cache location mismatch, will bite again on a new container**: `hf auth login`
  stores the token at `/workspace/.cache/huggingface/token` by default, but every python
  invocation in this project is prefixed `HF_HOME=/root/.cache/huggingface` (project convention,
  to keep large model downloads off `/workspace`'s quota). Result: `hf auth login` reports
  success, but any `HF_HOME=/root/...`-prefixed script still gets 401 Unauthorized. Fix each
  time: `mkdir -p /root/.cache/huggingface && cp /workspace/.cache/huggingface/token
  /root/.cache/huggingface/token`.

## Fixes applied from the 2026-07-25 review (all committed, all verified — see EXPERIMENTS.md)
- **A-1 (Critical)**: `semsup_b1_probe.py` gained `--loss {cosine,infonce}` (default stays
  `cosine`, nothing existing changed). InfoNCE = in-batch contrastive, sibling-TTE rows of
  the same `video_id` masked out of the negative set. Mechanically verified (3 synthetic
  tests + real end-to-end run) but **not yet re-run as the actual diagnostic** — that's the
  next step.
- **T-1 (Critical)**: `--seed` added to `semsup_train.py` (was completely absent — A1 and B
  had been confounded by different LoRA init/data order, not just `semantic_weight`).
- **A-2 (High)**: semantic predictor resized `num_queries=1→8, hidden_dim=512→256`
  (~5.13M→~1.25M params, now smaller than the ~2.8M LoRA trunk as the plan intended).
  `ResamplerProjector`'s self-attention block is now skipped entirely when `num_queries≤1`
  (it was mathematically a no-op there — softmax over one key).
- **T-3 (High)**: val AP/retrieval now aggregated **per clip** (mean-pool a clip's 2-3
  TTE-window rows into one point) instead of per row — 51 correlated rows were being treated
  as 51 independent samples when the real count is ~17 clips. Applies identically to A1 and B.
- **R1 (High)**: `build_frames_dir_index()` now reads only `teacher_dataset_e3b.jsonl` by
  default (verified: alone covers all 267 keys) instead of globbing 28 files, and **raises**
  on a genuine `(video_id,tte)` conflict instead of silent last-writer-wins.
- **Q1 (High)**: `evaluate_metrics.compute_metrics()` now delegates to
  `metrics_core.metrics_from_arrays` instead of an independent re-implementation that had
  quietly diverged (single-class AP: `0.0` vs `null`; different key names for the same field).
- **C2/C4/R2/R3 (High)**: per-epoch `epoch_metrics.jsonl` log; full run config recorded in
  `train_metrics.json`; test-set scoring streams+flushes per clip instead of buffering all
  677 and losing everything on a mid-run failure; per-clip try/except + `--min-examples`
  fail-fast guard against a silently-shrunk (partially-synced) dataset.

**C1 and T-2 done (2026-07-25)**: real per-clip A1/B test-score files pulled off the pod into
`outputs/semantic_captions/Pod_Run_Results/` (6 files, 677 rows each, verified against the test
manifest — 0 mismatches), and a paired bootstrap CI computed on the pre-registered A1-vs-B
comparison: 95% CI [−0.0239, +0.0030] AP, crosses zero, P(B>A1)=7.4% — confirms the existing
"no directional claim survives" call with an actual number. Full writeup + a caveat about
4-decimal score rounding in the archived files: EXPERIMENTS.md.

Still open from the review, not yet done: **C3** (`--resume` + optimizer-state checkpointing).
Full findings + severities: `reports/project_reviews/2026-07-25_project_review.md` (gitignored,
not tracked — `git add -f` it if you want it versioned).

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

## Important commands
```bash
# Local data-loading self-test (no GPU)
python student_training/scripts/semsup_common.py

# A1 (crash-only control) — real pod run
cd /workspace/MMLM_AI/student_training/scripts
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HOME=/root/.cache/huggingface
python -u semsup_train.py --config ../configs/e4_stageA.yaml \
    --lora-target-modules query,key,value \
    --semantic-weight 0.0 --epochs 8 --grad-accum 8 \
    --out-dir /workspace/semsup/a1 \
    --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
    --test-frames-root ../../dataset/test

# B (crash+semantic treatment) — add these two flags instead
#   --semantic-weight 0.3 --predictor-init /workspace/semsup/b1/predictor_b1.pt

# B1 (predictor-only probe)
python -u semsup_b1_probe.py --config ../configs/e4_stageA.yaml \
    --epochs 100 --out-dir /workspace/semsup/b1

# Local CPU smoke test of either (mechanics only; --limit 8 hits the single-class val path)
#   add: --epochs 2 --grad-accum 1 --limit 8 --test-limit 3

# Prompt bake-off harness (2026-07-27) - run on the POD for the real distinct-video ceiling
python student_training/scripts/semsup_sample_clips.py --n 300 --dry-run   # preflight only
python student_training/scripts/semsup_sample_clips.py --n 300            # writes the manifest
# then caption the manifest manually with prompts/PROMPT_SEMSUP_V2.py, one JSON row per clip
python student_training/scripts/semsup_caption_qa.py --input <raw_captions.jsonl> \
    --manifest dataset/manifests/semsup_promptbakeoff_300.jsonl   # Gate 0 + builds arm_a/b/c.jsonl
python student_training/scripts/semsup_caption_geometry.py \
    --inputs outputs/semantic_captions/promptbakeoff/arm_a.jsonl ... --labels A B C   # Gate 1, free/CPU
python student_training/scripts/semsup_b1_probe.py --config ../configs/e4_stageA.yaml \
    --loss infonce --captions outputs/semantic_captions/promptbakeoff/arm_a.jsonl \
    --out-dir /workspace/semsup/bakeoff_a   # Gate 2, repeat per arm
python student_training/scripts/semsup_promptbakeoff_report.py \
    --arm-a .../b1_metrics.json --arm-b ... --arm-c ...   # collates + applies the decision rule

# 500-clip extraction is DONE - re-run only if extending further:
python student_training/scripts/semsup_extract_promptbakeoff_frames.py   # local, needs the sibling-project MP4s

# Teacher-model bake-off tooling (2026-07-28/29) - any OpenRouter vision model, on the 18 val clips
python student_training/scripts/semsup_v6_control_rerun.py \
    --model <slug> --max-tokens 20000 --resume \
    --out outputs/prompt_bakeoff/semsup_val18/raw_<name>.jsonl   # runs UNMODIFIED v6 prompt
python student_training/scripts/semsup_caption_promptbakeoff.py \
    --manifest dataset/manifests/val_e3a.jsonl --prompt {v2,v3,v4} \
    --model <slug> --frame-size 0 --detail high --max-tokens 20000 \
    --out outputs/prompt_bakeoff/semsup_val18/raw_<name>.jsonl   # runs a semsup captioning prompt
```
# train4500-inference pipeline (2026-08-01) - score the real train pool, frozen scorer, no training
python student_training/scripts/build_train4500_manifest.py --dry-run   # preflight
python student_training/scripts/build_train4500_manifest.py            # -> dataset/manifests/train4500_hires.jsonl (4,446 rows)
python student_training/scripts/run_train4500_pipeline.py --chunk-size 500 --start-chunk N --stop-after-chunk M --workers 2
    # extracts chunks N..M locally (sequential-decode, ~1-2s/video); --workers 8 OOM'd this
    # machine (~2.8GB free RAM at the time) - default is now 2, checked safe
# then (on the pod, after transferring frames+chunk manifest):
HF_HOME=/root/.cache/huggingface python student_training/scripts/e4_stageA_badas_open_eval.py \
    --config student_training/configs/e4_stageA.yaml \
    --manifest dataset/manifests/train4500_chunks/chunk_NN.jsonl \
    --frames_root dataset/train --split Train \
    --output outputs/train4500_inference/scores_NN.jsonl
python student_training/scripts/evaluate_metrics.py --results <scores.jsonl> --out_dir <dir> --tag <name>
python student_training/scripts/mine_train_failures.py --scores <scores.jsonl> \
    --manifest <chunk.jsonl> --out-dir outputs/train4500_inference   # failure taxonomy + A0-gap checkpoint
python teacher_distillation/scripts/build_caption_monitor.py   # 4,500-row coverage+verdict grid

New-pod setup: `pip install -U transformers peft huggingface_hub pyyaml scikit-learn
albumentations sentencepiece protobuf` then `hf auth login`.

Full runbook: `RUNPOD_SEMANTIC_SUPERVISION.txt`. Running status doc:
`outputs/semantic_captions/summary.md` (auto-maintained per stage — keep updating it).

## Git state
Branch `main`, **HEAD = `4e41877`** ("Fix MID negative-bucket windowing bug found by real
chunk-0 scoring"), **in sync with `origin/main`** (user committed + pushed 2026-08-01). This
includes the MID→MID-10 fix (manifest builder, extractor, caption monitor) and the regenerated
`train4500_hires.jsonl`/`chunk_00.jsonl`/`teacher_dataset_train4500.jsonl`.

**Still uncommitted** (`git status -sb`, current):
- `RunPod/` (untracked dir) + 11 `D` deletions of `RUNPOD_*.txt` at repo root — **the user's own
  reorg**, verified byte-identical content (a plain move, not a real change). Not touched, not
  committed by any automated step — the user's call whether/when to commit it.
- `outputs/semantic_captions/Caption_Train_All_Clips.xlsx` — still shows modified, predates
  recent sessions entirely, never touched by any of this work. Left for the user to resolve.
- This handoff's own edits to `docs_agents/DECISIONS.md`, `EXPERIMENTS.md`, `PROJECT_STATE.md`.
- Note: `chunk_01.jsonl`/`chunk_02.jsonl` and the final scores/taxonomy/monitor outputs under
  `outputs/train4500_inference/` were generated after the last commit — verify with
  `git status` before assuming what's tracked.

**Only commit when the user asks** — per standing instructions, do not commit unprompted even
though the diff is large.

Note: `outputs/` and `reports/` are both **gitignored**; tracked files under them
(`summary.md`, `b1_metrics.json`, the caption JSONL/XLSX) were added with `git add -f`. Use
`-f` for new files there, and expect a harmless "paths are ignored" warning that makes
`git add` exit nonzero — it still stages, but it breaks `&&` chains.

## Next step

**B1-InfoNCE re-run (2026-07-25/26) found real signal**: clip-level retrieval@1 0.2353 vs
chance/control 0.0588 (n=17 clips, p=0.015) — the original B1 null was the cosine objective's
fault, not proof the data carries nothing. Still true, not touched this session.

**The 18-clip prompt/teacher bake-off thread (V2→V9, 7+ rounds, 2026-07-27 through 2026-08-01)
is SUPERSEDED, not abandoned mid-way.** Every accuracy delta across all 9 prompt versions and
5 models tested sat inside every other's 95% CI (McNemar p≥0.125 throughout) — n=18 cannot
distinguish these prompts from each other. One genuinely useful result survived: unmodified
`PROMPT_G_OPT_v6_balanced` on `google/gemini-3.6-flash` scored best on both prediction (72.2%
acc, 0 false positives) and caption fidelity of anything tried, while the *same unmodified
prompt* on `qwen/qwen3-vl-235b-a22b-thinking` collapsed to 0/18 YES predictions (verdict
echoed the prompt's own "prefer NO" language back near-verbatim) — model-family × prompt
interaction matters more than prompt wording alone. Full detail: EXPERIMENTS.md. **Decision
made 2026-08-01: stop iterating prompts on 18 clips (underpowered by construction); pivot to
measuring where the *real* teacher (the frozen A0 scorer, on ~4,500 real train windows) fails,
since that ties directly to the actual thesis metric instead of a caption-quality proxy.**

**train4500-inference pipeline: DONE (2026-08-01).** Goal: score every train window (2,223
pos/2,223 neg = 4,446, excluding val_e3a's 18 clips) through the frozen A0 scorer — nothing
trains — to find systematic failure patterns and decide caption-budget allocation.

**Result, all 4,446 windows scored and verified**: `AP=0.9535 AUC=0.9474 accuracy=86.8%
TP/FP/TN/FN=1954/318/1905/269 error=13.2% (587 total failures: 318 FP + 269 FN)`.

1. **Chunk 0 found + fixed a real bug** (see the MID-10 entry in ARCHITECTURE.md/EXPERIMENTS.md
   for full diagnosis) — offset-0 `MID` bucket had 42.8% error, 100% FP; moved to `MID-10`
   (offset −10s), now 14.0% error, in line with every other bucket.
2. **Per-chunk numbers held stable across all 3 independent 500-video chunks**
   (AP 0.9555/0.9513/0.9535, AUC 0.9504/0.9454/0.9462) — confirms the result is a real
   property of the train pool, not chunk-0 noise.
3. **Failure taxonomy is DIFFUSE** (13.9% spread across the 6 buckets, worst=TTE_1.5 at 19.6%,
   best=TTE_0.5 at 5.7%) → `mine_train_failures.py`'s own classifier recommends **uniform**
   caption allocation over failure-targeting. This is the final answer, confirmed at full
   scale, not just chunk 0.
4. **Coverage**: 213/4,446 windows already captioned (4.8%) per
   `outputs/train4500_inference/monitor_train4500_coverage.xlsx` (pulled locally + sent to
   user) — this is the real starting point for scaling toward the ~4,500 target.
5. **Still open, not explained**: train's 13.2% error vs A0's known 677-clip test error of
   23.6% — a real, stable gap (confirmed at n=4,446, not a fluke). Test is FP-dominated
   (130:30, 4.3:1); train is nearly balanced (318:269, ~1.2:1). Pipeline mechanics were checked
   and ruled out (byte-identical extraction verified). Not yet investigated further — see
   DECISIONS.md's unresolved questions.

**Next step, not yet started**: with uniform allocation now the settled answer, decide the
next caption-scaling batch size/teacher-prompt combo and start captioning toward the ~4,500
target (213 done). The train-vs-test gap above is a secondary open thread, lower priority.

**Separately still open, not yet started:** porting `--loss infonce` into `semsup_train.py`
(Stage B) — currently only exists in `semsup_b1_probe.py`. Blocked on a batching prerequisite
(A-5 in the 2026-07-25 review), independent of the train4500 thread above.

Always `git pull` on a fresh pod session before trusting it has the latest scripts — this
session's pod was one full commit stale on first connect.
