<!-- handoff-month: 2026-10 -->
# Project State

Live files hold the current month only. Full prior state: `history/2026-09_{PROJECT_STATE,ARCHITECTURE,EXPERIMENTS,DECISIONS}.md`
(Jun–30 Sep 2026) and `history/2026-08_*.md`. Last refreshed 2026-10-22.

## Goal
MSc thesis on Nexar dashcam collision anticipation. The crash-score path is **FROZEN** (A1-compress256, AP 0.913 private / 0.910 public).
The thesis question: can ONE vision pass (frozen V-JEPA2/BADAS A1-compress256 encoder -> trained projector -> Qwen3-VL-4B + LoRA) give the alert AND its explanation,
approaching BADAS (Kaggle mAP: BADAS-1.0 0.925, BADAS-2.0 0.940, BADAS-2.0-Flash 0.941)? Decision score = log P(" yes") - log P(" no") at "Collision:" (verdict-first answer).
Work lives on branch `reasoning-path-vjepa2-llm` (root `README.md` = goal + folder map). Model names for the presentation: crash model "BADAS-Open (SFT)", reasoning model "BADAS-JEPA-Reason".

## Standing rules of the user (binding)
- Never print API keys / the HF token (copy by scp only). **Claude never pushes git** unless the user asks for that push (git push failed once: the user's `id_ed25519_private` key needs a passphrase; the user pushes). Commit only on request.
- **The USER creates pods**; Claude copies code, launches the driver, drivers push logs to HF and **STOP THE POD on any major failure or after the HF byte check** (user rule 2026-10-22). Hourly health check while the session is open (CronCreate, session-only; delete the job when a run ends). Never edit a bash driver while it runs (bash re-reads it: that killed a run). Confirm before any spend beyond the approved amount.
- **Validation only** for every experiment until the final seeds stage; test sets scored ONCE at the end (pre-alert windows removed from training and scores; object crop excluded; test texts never for training / selection).
- **No crash head at all** in training or at inference of the reasoning model (single path: encoder -> projector -> language model); crash model and reasoning model stay two separate architectures; fusion is closed.
- Plans: a new file per plan in `~/.claude/plans/CCP based BADAS/`; answer plan questions in chat too; short plain answers, flag < 80 % confidence, state the threshold of every thresholded metric, per-experiment `summary.md`. No Windows power-setting changes. Do not run heavy GPU jobs on the user's laptop (it made the PC crawl).
- Professor-facing documents: no internal run names. Presentation website: new site `website_pres/`; lab site `website/` keeps every experiment; promotion to the presentation site only when the user says so.

## Status as of 2026-10-22

### Crash path (frozen)
A1-compress256: pooled test AP 0.911 / AUC 0.913 / Kaggle mAP 0.916 (private 0.9127, public 0.9096). Ledger: EXPERIMENTS.md §1.

### Reasoning model: where we are (all numbers validation, one seed; core = 524 original windows, full = 641 incl. 117 new 0.75 / 1.25 s windows)
| model | val core AUC | val full AUC |
|---|---|---|
| today's P2V (2026-10-15; recipe not reproducible: its Phase-1 features were deleted) | 0.953 | 0.951 |
| chain B (new 0.75 / 1.25 s data, gentler recipe) | 0.942 | 0.938 |
| R2 / D2 (stage 1) | 0.951 / 0.952 | 0.948 / 0.950 |
| B4 (crash-head attention input) / **control** (D2 recipe, fresh two-step Phase 1) | 0.953 / **0.959** | 0.952 / **0.958** |
| single-stage recipe, seeds 0 / 1 | 0.952 / 0.945 | 0.949 / 0.946 |
| crash model | 0.957 | 0.956 |
- **Selected base = the control** (pre-registered rule: single-stage mean 0.948 vs control 0.959, difference > noise band 0.0064). Its P2V checkpoint: HF `checkpoints/b4_2026-10-21/CT/P2V/best.pt` (val loss 1.078).
- Over-fitting is the measured problem (train AUC ~0.99 vs val 0.95; best epoch always early). Nothing changed the decision by more than ~0.006 AUC; recipe-to-recipe noise is ~0.007.
- Text: B4's attention input helps motion / gap facts (+0.04 / +0.06 over the control) but not the decision (-0.006), and needs the crash head at inference -> not a final-model candidate. R2 passes the text gate on gap (+0.055). D2 does not.
- Fusion (crash model + reasoning model): fixed w +0.002 AUC, learned gate / selection rules / small MLP all <= that: CLOSED (gate +0.005 not met).

### Datasets (private HF org `eviatarO-org`)
Teacher dataset 4,169 windows (`outputs/teacher_dataset_2026-10/`), extra windows 860 at TTE 0.75 / 1.25 s (`outputs/teacher_extra_2026-10/`), rewordings (accepted 9,497 + nuance-only judge rejects 2,585 = `paraphrases_relaxed.jsonl`, 12,082 versions), manifests `dataset/manifests/r1_nexar_reasoning_{windows,extra,v2,test,test_public}.jsonl`. HF window repo `nexar-reasoning-windows` (uint8 16x256x256x3 frames of all 4,169 windows). Test sets: private 677 + public 667; 431 private windows have a teacher reference (pass 2 not run; the 20-window pass-2 pilot awaits the user's review).

### Stage 3 (IN PROGRESS: code done, two pod attempts failed early, not yet running)
Plan `~/.claude/plans/CCP based BADAS/2026-10-22_Plan-Stage3-Encoder-LoRA.md`: the V-JEPA2 encoder's LoRA (144 matrices, rank 16 q/k/v, 2.36 M params) trains with the language model's loss ONLY, warm start = the control. Two arms on two pods:
**J** (encoder LoRA + projector + LM LoRA; lr 5e-5 / 1e-5 / 5e-5) and **E** (encoder LoRA only; projector + LM LoRA frozen; the clean diagnostic). <= 6 epochs, early stop patience 2, unscored checkpoint every 0.25 epoch + local refinement around the winning epoch, epoch-0 parity check (must reproduce the control's val loss 1.078 +- 0.01).
Gates: an arm wins at val core AUC >= 0.9642 and val full >= 0.9577 (control + 0.005).
- Code written + unit-tested (23 tests pass; 21 Oct run of the whole suite): `r1_frames.py`, `r1_train.py` stage-3 options, `r1_pod_stage3.sh` (arm via `ARM=J|E`).
- **Pod attempts (22 Oct) both stopped within ~10 min**: pod 1 by the driver's own fail-safe (cause not read: likely the encoder-only parity check, because TF32 had been switched on globally; logs were lost to the 20-s age rule, now fixed), pod 2 because the driver file was overwritten while running. Both fixed; **no stage-3 result exists**.
- Findings that matter for stage 3: (1) the V-JEPA2 config has **stochastic depth 0.1 and dropouts: in train mode the tokens change by 73 % and are random** -> `OnlineEncoder` zeroes drop-path / dropout and runs the net in train mode only while gradients flow (needed for gradient checkpointing); (2) encoder precision: TF32 fp32 token change 0.4 % vs eval fp32, **bf16 autocast 13 %** (cos 0.982) -> fp32 required; `--enc-precision fp32|fp32_strict|bf16` (default fp32 = TF32 while training, plain fp32 for eval / re-encoding); the driver waits up to 60 min after the smoke test for `$OUT/PRECISION` (default fp32) and runs `r1_frames.py bench` (timings + token errors for all three).
- Encoder-only tokens == cached tokens exactly after fp16 rounding (checked on real BADAS weights); GPU normalisation == processor (4.8e-7); gradients reach all 144 LoRA matrices; peak 4.95 GB at micro-batch 2 with checkpointing on the 6 GB laptop GPU.

## Open TODOs (in order)
1. **Restart two pods** (32 GB GPU RTX PRO 4500, 100 GB disk, no network volume), send SSH lines -> I read the real port through the proxy line (`cat /proc/1/environ | grep RUNPOD_TCP_PORT_22`; the console port was wrong twice), copy code + token (scp), `r1_pod_run.sh setup`, run `ARM=J` / `ARM=E bash r1_pod_stage3.sh` (never edit it afterwards), report the smoke test + `bench_precision.json` to the user, write `$OUT/PRECISION` after the user decides, re-create the hourly cron. ETA ~6-10 h per arm (fp32 slower than bf16: measure), ~$10-18 for both.
2. After stage 3: report with gates (val core / full AUC, text, fact cards locally via `r1_fact_cards.py --source model`, token drift, fact probe before / after). Then 3.1 (motion / gap auxiliary loss on the winning base) as its own plan, then **stage 4: 5 seeds of the final recipe end to end, test sets once (seeds + ensemble vs crash model and BADAS-2.0; P2V as reference)**.
3. Presentation website: Home page built (`website_pres/`); other pages follow in later rounds; user reviews wording (hero subtitle "causal reasoning", problem / solution text).
4. Review the 20-window pass-2 pilot with the user; open: P2V threshold chosen on validation, regularised Phase 2 variants, controls, faithfulness test, pre-alert human review.
5. Housekeeping: RunPod network volume (63 GB, $4.41/mo) unverified; HF private storage near the 100 GB limit (do not push epoch checkpoints); unsent access requests (DRAMA, WTS, RoadSafe365, MM-AU authors).

## Known bugs / gotchas
- **A smoke test must use options valid for its epoch count** (`--eval-at` fractions above `--epochs` abort); DRYRUN only prints commands: rehearse every training / eval command on a tiny model (`--lm tiny --device cpu`) before a pod run.
- **Heredocs through the Bash tool turn backslash-n into real newlines**: write scripts with the Write / Edit tools; `.sh` files must be LF (`sed -i "s/\r$//"` on the pod); Python `write_text` makes CRLF.
- `r1_hf_sync.py` skips files younger than 20 s: the failure handler sleeps 25 s before pushing; patterns are per driver (`--subdirs`, `--patterns`).
- Evaluator `load_ckpt` now reads the LoRA rank from the checkpoint and uses alpha = 2r (a rank-8 checkpoint was rejected / would have been rescaled).
- Windows: memory-mapped files cannot be deleted inside temp dirs (clear `.maps`, `gc.collect()` before the `with` exits).
- `sleep` is blocked in the tool (use until-loops inside `ssh ... 'for i in ...; do ...; sleep 10; done'` with a timeout, or run_in_background); background `ssh` waits need `timeout` below 600 s.
- Per-TTE AP on val / train is not comparable to test (pairing 1:10 vs 1:1): use per-TTE AUC. Never pick a model on the test sets.
- Local GPU is a 6 GB laptop card: no Qwen3-VL-4B; the real encoder at micro-batch > 1 thrashes memory and freezes the PC.
- Crash-path gotchas: `history/2026-09_PROJECT_STATE.md`.

## Pod state
Nothing should be running. Stage-3 pods `8n0e7usyf48b0b` and `aap202i5ku6luw` stopped (22 Oct); earlier pods stopped themselves after the HF byte check. A new pod is created by the USER; Claude copies code with `tar ... | ssh ... tar xzf - --no-same-owner`, the HF token by scp.

## Important commands (from `MMLM_AI/`, scripts in `student_training/scripts/`)
```bash
python r1_reasoning_test.py                         # 23 CPU unit tests (~5 min, run at low priority)
python r1_train.py --phase 2 ... --online-encoder <adapter> --frames <dirs> --cache-dir <old cache dirs> --init-from <control best.pt> [--freeze-reader] --eval-start --expect-val-loss 1.078 --save-every 0.25 --refine-around 0.25,0.5 --enc-precision fp32
python r1_frames.py build|parity|bench|encode|drift ...      # frames store, parity checks, precision bench, re-encode with a trained encoder, token drift
ARM=J|E bash r1_pod_stage3.sh                       # stage-3 pod driver (DRYRUN=1 prints commands)
bash r1_pod_base.sh / r1_pod_b4.sh / r1_pod_next.sh # finished night runs (single-stage seeds / B4 + control / R2 + D2)
python r1_fusion.py --pred <val predictions> --out <dir>; python r1_fusion_mlp.py --val ... --train ...   # fusion (closed)
python r1_best_check.py <run dir> [--after 1.0]; python r1_relaxed_paraphrases.py ...
PYTHONUTF8=1 python website_pres/build_pres_data.py # presentation site data (asserts every number); serve with python website/serve.py (port 8765)
```
Older commands (teacher flow, caching, probes): scripts' docstrings and `history/`.

## Git state
Branch `reasoning-path-vjepa2-llm`; three commits on 2026-10-20 (`e2e3759`, `5a6ffc6`, `d873c74`); the push failed (passphrase) and may still be pending: check `git status -sb`. **Uncommitted since:** `r1_train.py` / `r1_eval_reasoning.py` options, `r1_fusion*.py`, `r1_best_check.py`, `r1_relaxed_paraphrases.py`, `r1_frames.py`, `r1_pod_{next,b4,base,stage3}.sh`, `r1_hf_sync.py`, `r1_reasoning_test.py`, `website_pres/`. Commit only when the user asks.

## Next step
Get two pods from the user and run stage 3 (TODO 1). Do not relaunch without reading `bench_precision.json` and the smoke result to the user.
