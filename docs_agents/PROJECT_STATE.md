<!-- handoff-month: 2026-10 -->
# Project State

Live files hold the current month only. Full prior state: `history/2026-09_{PROJECT_STATE,ARCHITECTURE,EXPERIMENTS,DECISIONS}.md`
(Jun–30 Sep 2026, superset of the 2026-08 snapshots) and `history/2026-08_*.md`.

## Goal
MSc thesis: collision anticipation on Nexar dashcam clips (BADAS-Open / V-JEPA2 ViT-L + LoRA student).
**Two threads now:** (1) the crash-score path — **FROZEN** by user decision 2026-10-02; (2) a **reasoning
path** — frozen vision encoder → projector → LLM that explains a clip from the encoder's own latents
(BADAS-2.0 paper p.8 names this "intrinsic reasoning from model representations" as future work).
Thread (2) is in the planning stage; nothing has been trained or spent.

## Status as of 2026-10-03

### Crash-score path (frozen — do not reopen without the user)
- **Champion single checkpoint: A1-compress256 — test AP 0.9128 (private 677) / 0.9096 (public 667).**
  Recipe: LoRA r16/α32/dropout 0.05 on `query,key,value`, crash CE only, lr 2e-4 constant, grad_accum 8,
  1,761-window pool, crash head frozen, `--preprocess compress256`. Pass `--preprocess compress256`
  explicitly: the code default is still `crop` (byte-identical to every pre-2026-09-14 run).
- **Best recipe by mean: midpoint negatives** (`Caption_Train4500_MidpointNeg_1761.jsonl`: negatives re-cut at
  the Nexar test protocol's fake-event midpoint). Mean of 9 checkpoints pooled-1,344 AP 0.9158 vs control
  0.9077 (+0.0082, 9/9 paired). 5-seed confirmation, epoch 1: pooled AP +0.0092 ± 0.0052, 5/5 seeds
  (sign test p=0.03); Kaggle mAP +0.0037, 4/5 → **not confirmed**. Claim wording: "raises pooled AP and cuts
  false alarms at matched recall; no reliable Kaggle mAP gain". Never report the single best checkpoint.
- Context: BADAS-Open paper 0.86 (we reproduce 0.861), BADAS-1.0 AP 0.91 / mAP 0.925, BADAS-2.0 mAP 0.940.
- **Closed negative results** (reasons in DECISIONS.md): every semantic-supervision arm (B-v1/v2/v3, P1, SemTest-200,
  a1fail321 recovery family), Stage AA token-relevance aux loss, Stage AA-H head-attention supervision,
  loss-based look-ahead head (λ=15.3 and λ=1.0), horizon-weighted loss (n.s.). 1.5 s TTE remains the
  weakest bucket (0.87–0.89 AP vs 0.91–0.94); the freeze accepts that ceiling.
- Preprocessing bug history that matters: before 2026-09-14 every run saw only the centre ~49 % of frame width
  (V-JEPA2 processor resizes shortest edge to 292, centre-crops 256×256). Fixed by `compress256`.

### Reasoning path (planning)
- **Prior art in this repo (June 2026, e4 Stages B/C):** the encoder→`ResamplerProjector`→Qwen3-4B bridge exists
  and runs. Stage B (projector only) passed its gate (Δ-PPL −48 %, 267 windows). Stage C (LoRA SFT) learned the
  verdict (88.9 % agreement with the vision score) but free-form reasoning **collapsed to two verdict-keyed
  templates**; diagnosed as data starvation (89 scenes). Never run: Stage C ep5/ep7 generation eval
  (only epoch 1, the val-CE-best, was tested). Stage D (2026-07-01) dropped the projector route.
- **Draft plan (rev. 4+, NOT approved — ExitPlanMode rejected five times; the last interruption came right after the answers to the user's three open questions were appended, so they are unconfirmed):**
  `~/.claude/plans/CCP based BADAS/2026-10-02_Plan-Reasoning-Path-Public-Pretrain-DRAFT.md`
  (identical copy: `~/.claude/plans/distributed-plotting-spark.md`). Shape: Stage 1 = projector-only
  pre-alignment on public driving video+text (Wave 1 MM-AU + teacher captions on Nexar train windows; Wave 2
  BDD-X + WTS; Wave 3 only if the learning curve still rises); Stage 2 = Nexar SFT (CAViAR human reasoning on 749
  positives + teacher text on 750 negatives + MM-AU-anchored teacher reasoning); gates G0 (LLM bake-off + ep5/ep7
  eval), G1 (H0 learning curve), G2 (pre-SFT baselines incl. zeroed-vision floor), G3 (post-SFT).
- **Verified dataset facts (2026-10-02/03)** are in EXPERIMENTS.md "Data audits". Headline: CAViAR released
  human reasoning for **749 Nexar clips, all in our train positives, zero overlap with both test sets**.
- Unanswered user questions on the draft are listed in DECISIONS.md "Unresolved — reasoning path".

## Open TODOs
1. User decision on the draft plan (approve / modify). If approved: it is already copied to the topic folder
   as a DRAFT; rename without `-DRAFT` and keep the harness copy in sync. Stop after each stage (e4 gating rule).
2. Phase 0 (if approved): G0 LLM bake-off on the 267-window cache with native vision-token injection
   (Qwen3-4B-Instruct-2507 control, Qwen3-VL-4B's LM, Qwen3.5-4B re-test); the never-run e4 ep5/ep7 generation
   eval (checkpoints on HF `EviatarO/e4-stageC-qwen3`).
3. Access requests (none sent): MM-AU (HF `JeffreyChou/MM-AU` is public, but the GitHub README says academic-only +
   email — resolve before use), WTS Google Form, BDD100K registration, RoadSafe365 authors (no release link found).
4. Spend needing explicit approval (not authorised): ~$25 Nexar descriptive captions, ~$5–10 negatives,
   ~$60 MM-AU anchored reasoning; Wave 3 only if triggered.
5. Deferred website TODOs: detail pages for the 6 overnight arms (EXPECTED_CM entries exist, ARMS entries not
   written); landing table date column + click-to-sort; point comparison arms at one shared epoch per family.
6. Unresolved crash-path leftovers (low priority while frozen): fixed-epoch recipe chosen from val not test;
   seed-to-seed calibration jitter (stabiliser candidates in DECISIONS.md).

## Known bugs / gotchas (still relevant)
- **Windows console:** always `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`; `open(path,'w')` writes `\r\n` (use `newline='\n'` for `tar -T` lists).
- **Never** `pip install -U torch` on a RunPod image. Fresh container needs: `huggingface_hub transformers peft safetensors
  scikit-learn pyyaml pillow openpyxl opencv-python-headless einops timm albumentations psutil sentencepiece protobuf`
  (install all up front) and `hf auth login` / copy `/workspace/.cache/huggingface/token`. Prefix pod Python with `HF_HOME=/root/.cache/huggingface`.
- Two BADAS-loading processes at once can silently crash one — run sequentially. `peft.save_pretrained` needs the model-card stub
  (`badas.nn_model.create_or_update_model_card = lambda *a, **k: None`).
- **Resuming an `--unfreeze-head` run past epoch 1 requires `--head-init`** or the head silently resets.
- **`score_checkpoints_on_test.py` `NAME=NONE`** now resets the LoRA (fixed 2026-09-14); older multi-arm invocations mixing a real adapter with NONE are suspect.
- Join caption/label files on `frames_dir` only (a clip has up to 3 windows; `video_id` is not unique per row).
- `semsup_caption_promptbakeoff.py`: always pass `--model` explicitly and verify the printed model; OpenRouter `preview` aliases are not stable.
- Backgrounded commands piped through `tail` mask exit codes — redirect to a file and check `$?`.
- Legacy `.xls` (MM-AU labels) cannot be read with pandas here (no `xlrd`); Excel COM works read-only (see Important commands).
- RunPod: network volume `0hnvco2s4j` (EU-RO-1) is **at its 56 GB quota** — write run outputs to the container disk (`/root`, 40 GB) and
  **sync results to local before `runpodctl stop`** (memory rule). Pod capacity is often unavailable — create a fresh pod on the volume with a
  broad `gpuTypeIds` list. Pods need `RUNPOD_API_KEY` exported from `/proc/1/environ`. Every reconnect is a new IP/port; never generate a new SSH key
  (reuse `~/.ssh/id_ed25519.pub`).

## Pod state
Nothing running or billing as of 2026-09-30. Stopped pods remain: `egx54zfwpmpasg` ("controls-public-score"), `maeqpipl372s77` ("pull-results").
All checkpoints/data are on the network volume `/workspace/MMLM_AI` (adapters of the 2026-09-27/29 runs are pod-side; locally only `adapter_config.json`).

## Important commands (run from `student_training/scripts/` unless noted)
```bash
# Champion / midpoint-negatives recipe (pod). Swap $CAPTIONS for Caption_Train4500_MidpointNeg_1761.jsonl.
python3 -u semsup_train.py --config ../configs/e4_stageA.yaml \
  --lora-target-modules query,key,value --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --semantic-weight 0.0 --epochs 3 --lr 2e-4 --lr-schedule constant --grad-accum 8 \
  --val-frac 0.2 --captions-path $CAPTIONS --keep-top-k 3 --split-seed 0 --init-seed $SEED --seed $SEED \
  --preprocess compress256 --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl \
  --test-frames-root ../../dataset/test --out-dir $RUN_DIR/train
# Score a checkpoint on the PUBLIC set
python3 -u score_checkpoints_on_test.py --config ../configs/e4_stageA.yaml \
  --test-manifest /workspace/data/test_manifest_public_hires.jsonl --test-frames-root /workspace/data/test_public \
  --adapters NAME=$RUN_DIR/train/epoch_01/lora_adapter --preprocess compress256 --out-dir $RUN_DIR/scores_public
# Paired bootstrap between two score files (either label schema); paired per-seed comparison
python paired_bootstrap_ab.py --a A.jsonl --a-name A --b B.jsonl --b-name B --n-boot 5000 --seed 42 --out out.json
python stage_compare.py ...            # pre-registered pass-rule comparison (see its --help)
# Caption a corpus (teacher, OpenRouter)
python3 semsup_caption_promptbakeoff.py --manifest <m.jsonl> --frames-root ../../dataset/train --out <o.jsonl> \
  --prompt <key> --model google/gemini-3.7-flash --provider-order google-vertex --concurrency 16
# e4 reasoning-path caching + Stage C eval (pod; runbooks RunPod/RUNPOD_E4_STAGE{B,C}.txt)
python e4_stageB_cache_features.py --config ../configs/e4_stageC_v2_qwen3.yaml \
  --manifest ../../dataset/manifests/val_e3a.jsonl --frames_root /workspace/data/train_HiRes --split val
export HF_HOME=/root/.cache/huggingface E4_LLM_MODEL_ID=Qwen/Qwen3-4B-Instruct-2507 E4_CACHE_DIR=/root/e4_stageB/cache \
  E4_OUTPUT_DIR=/root/e4_stageC/out_qwen3 E4_PROJECTOR_CKPT=/root/e4_stageC/projector_qwen3.pt
```
```powershell
# Read a legacy .xls read-only (MM-AU label files) — Excel COM, ReadOnly=true, do not save
$xl = New-Object -ComObject Excel.Application; $wb = $xl.Workbooks.Open($path, 0, $true)
```
CAViAR release check (re-runnable): `curl -sL https://raw.githubusercontent.com/nec-labs-ma/CAViAR/main/data/test.json` (749 items,
`video_path` = zero-padded Nexar id) joined to `dataset/train.csv` (`id,target`) and the test manifests.

## Git state
Branch `main`, HEAD `bdbca1b` before this handoff, **8 commits ahead of `origin/main`** (not pushed — the user pushes; never push). Working tree
was clean; this handoff adds uncommitted changes to `docs_agents/{PROJECT_STATE,ARCHITECTURE,EXPERIMENTS,DECISIONS}.md`,
`docs_agents/history/2026-09_*.md` (new, verbatim September archive) and `.handoff_state.json`. Commit only if the user asks.

## Next step
Wait for the user's decision on the draft reasoning-path plan; incorporate their answers (cross-dataset generalisation, what the H0 curve
is, realism of the Wave-1-only week) before any build. Do not start pod work, spend, or data downloads without approval. If approved, begin
at Phase 0 (G0 bake-off + ep5/ep7 eval + access requests) and stop for review after each phase.
