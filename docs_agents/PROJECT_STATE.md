# Project State

## Goal
MSc thesis: collision anticipation on Nexar dashcam clips via Teacher→Student distillation.
Accepted shipped baseline: InternVL3.5-4B-Flash student, test AP=0.762 (677 clips). Current
active thread: **semantic-supervision** — test whether a language-derived auxiliary loss
(caption embedding alignment) improves BADAS-Open's (V-JEPA2) crash-prediction representation,
while keeping inference vision-only (no added cost/latency).

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
- Working repo: **`/workspace/MMLM_AI`** (persistent network volume). NOT `/root` — that is
  wiped on every new pod. **Pod is currently stopped** (confirmed 2026-07-25, network volume
  survives stop — verified by successfully pulling files after a restart).
- Frame data: `dataset/train` and `dataset/test` are **symlinks** into the pod's pre-existing
  `/workspace/data/train_HiRes` (295 folders) and `/workspace/data/test_HiRes` (677). All
  267/267 needed training folders verified present — no manual data transfer is needed.
  (A 780MB `b1_bundle.tar` was built locally for this and turned out unnecessary.)
- Results live at `/workspace/semsup/{b1,a1,b}/` on the persistent volume.
- `/workspace` volume was enlarged 24GB → 30GB after hitting a hard per-user quota.
- Deps must be reinstalled on every NEW pod (pip installs live in the container, not
  `/workspace`), and `hf auth login` must be re-run (BADAS-Open is a gated repo).

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
```
New-pod setup: `pip install -U transformers peft huggingface_hub pyyaml scikit-learn
albumentations sentencepiece protobuf` then `hf auth login`.

Full runbook: `RUNPOD_SEMANTIC_SUPERVISION.txt`. Running status doc:
`outputs/semantic_captions/summary.md` (auto-maintained per stage — keep updating it).

## Git state
Branch `main`. **HEAD = `952082e` (prompt bake-off harness), 1 commit AHEAD of `origin/main` —
unpushed.** Everything before it (`bd0eac5` restore `b1_metrics.json`, `9dc7afa` real
B1-InfoNCE result, and the `b23b16b`-era chain) is pushed. Working tree otherwise clean except
`outputs/semantic_captions/Caption_Train_All_Clips.xlsx`, which shows modified but predates
this session entirely — not touched, left for the user to resolve.

Also new this session (untracked, not a code change): `~/.claude/skills/project-review/
SKILL.md` — user-level `/project-review` skill (deliberately not named `/code-review`,
which is a different built-in command). Works in any project.

Note: `outputs/` and `reports/` are both **gitignored**; tracked files under them
(`summary.md`, `b1_metrics.json`, the caption JSONL/XLSX) were added with `git add -f`. Use
`-f` for new files there, and expect a harmless "paths are ignored" warning that makes
`git add` exit nonzero — it still stages, but it breaks `&&` chains.

## Next step
**B1-InfoNCE re-run is DONE (2026-07-25/26) — real signal found.** Clip-level retrieval@1
0.2353 vs chance/control 0.0588 (n_clips=17, p=0.015). Confirms A-1's diagnosis: the original
B1 null was the cosine objective's fault, not proof the data carries nothing. Full numbers:
EXPERIMENTS.md.

**Prompt bake-off harness (Gates 0-2) is DONE and verified (2026-07-27)** — see
`~/.claude/plans/CCP based MMLM - Student/2026-07-27_Plan-Prompt-bakeoff-harness-semantic-captions-Gates-0-2.md`
and the new "Prompt bake-off harness" section in EXPERIMENTS.md. Captioning design changed
from the originally-proposed two separate prompts to **one prompt, three arms built from its
JSON output** (Arm A = neutral description, Arm B = +risk clause, Arm C = label-only control) —
a paired, not independent, comparison. Every gate is built and calibration-tested; nothing has
been run on real new captions yet (captioning itself is manual/out of scope of the harness).

**Immediate next step: run `semsup_sample_clips.py --n 300 --dry-run` on the POD**, not
locally. The preflight discovered that positive and negative clips use *different* windowing
conventions (`TTE_0.5/1.0/1.5` for positives vs `MID/MID-4/MID-8` for negatives, since
negatives have no event to count down to) — and that **locally the achievable pool is 89
videos / 267 rows, identical to what already exists** (42 pos + 47 neg, zero new distinct
videos beyond the incumbent set). The real ceiling depends entirely on the pod's 295
`train_HiRes` folders, not yet checked against this logic.

**After that:** caption whatever the pod's real ceiling turns out to be, using
`prompts/PROMPT_SEMSUP_V2.py` (single prompt, JSON output: `caption_neutral` / `risk_clause` /
`verdict` / `confidence`), run `semsup_caption_qa.py` (Gate 0) →
`semsup_caption_geometry.py` (Gate 1, free/CPU) → `semsup_b1_probe.py --loss infonce
--captions arm_X.jsonl` per arm (Gate 2) → `semsup_promptbakeoff_report.py`, which applies the
pre-written decision rule (DECISIONS.md) mechanically.

**Separately still open, not yet started:** porting `--loss infonce` into `semsup_train.py`
(Stage B) — it currently only exists in `semsup_b1_probe.py`. B1's positive result is about
video↔caption alignment only; nothing yet shows it moves actual crash AP. Noted as A-5 in the
2026-07-25 review, blocked on a batching prerequisite not yet built.

Pod is currently **stopped** (network volume persists; restart + `git pull` before any of the
above — the pod's checkout was found stale by one commit during the InfoNCE run, so always
`git pull` on a fresh pod session before trusting it has the latest scripts).

**Longer-standing, still a cost/scope decision, not a task** — four options, laid out in
DECISIONS.md:
1. Full caption scale-up 267 → ~4.5k (original plan; expensive; prior fresh-captioning
   attempts hit real friction). Now better-motivated only if the InfoNCE re-run above shows
   a real (non-chance) signal — otherwise scaling data won't fix a broken objective.
2. Intermediate scale-up (~500–1000 captions) as a cheaper first check.
3. ~~Fix checkpoint selection~~ — **done this session** (T-3).
4. Deprioritize this thread in favor of the e4 faithfulness work (no crash-domain precedent
   exists for train-only-text/vision-only-inference; the domain SOTA uses no language).
