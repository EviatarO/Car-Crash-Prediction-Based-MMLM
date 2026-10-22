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
- Never print API keys / the HF token (copy by scp only). **Claude never pushes git** unless asked; Claude's push fails anyway (`github-private` -> `id_ed25519_private` is rejected / needs a passphrase): Claude commits on request, **the user pushes**.
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
| stage 3 encoder LoRA, arm J / arm E (best scored point after epoch 0) | 0.9593 / 0.9592 | not scored |
| crash model | 0.957 | 0.956 |
- **Selected base = the control** (pre-registered rule: single-stage mean 0.948 vs control 0.959, difference > noise band 0.0064). Its P2V checkpoint: HF `checkpoints/b4_2026-10-21/CT/P2V/best.pt` (val loss 1.078).
- Over-fitting is the measured problem (train AUC ~0.99 vs val 0.95; best epoch always early). Nothing changed the decision by more than ~0.006 AUC; recipe-to-recipe noise is ~0.007.
- Text: B4's attention input helps motion / gap facts (+0.04 / +0.06 over the control) but not the decision (-0.006), and needs the crash head at inference -> not a final-model candidate. R2 passes the text gate on gap (+0.055). D2 does not.
- Fusion (crash model + reasoning model): fixed w +0.002 AUC, learned gate / selection rules / small MLP all <= that: CLOSED (gate +0.005 not met).

### Datasets (private HF org `eviatarO-org`)
Teacher dataset 4,169 windows (`outputs/teacher_dataset_2026-10/`), extra windows 860 at TTE 0.75 / 1.25 s (`outputs/teacher_extra_2026-10/`), rewordings (accepted 9,497 + nuance-only judge rejects 2,585 = `paraphrases_relaxed.jsonl`, 12,082 versions), manifests `dataset/manifests/r1_nexar_reasoning_{windows,extra,v2,test,test_public}.jsonl`. HF window repo `nexar-reasoning-windows` (uint8 16x256x256x3 frames of all 4,169 windows). Test sets: private 677 + public 667; 431 private windows have a teacher reference (pass 2 not run; the 20-window pass-2 pilot awaits the user's review).

### Stage 3 (DONE 2026-10-22: NEGATIVE)
Plan `~/.claude/plans/CCP based BADAS/2026-10-22_Plan-Stage3-Encoder-LoRA.md`: the V-JEPA2 encoder's LoRA (144 matrices, rank 16 q/k/v, 2.36 M params) trained by the language model's loss ONLY, warm start = the control. Arms **J** (encoder LoRA + projector + LM LoRA) and **E** (encoder LoRA only). Gate: val core AUC >= 0.9642 and val full >= 0.9577.
- **No scored point beat epoch 0** (core AUC 0.9572-0.9593 vs start 0.9593); J over-fits (val loss 1.078 -> 1.153 / 1.189), E is flat (1.086). Early stop after epoch 2; best = epoch 0 = the unchanged encoder. The encoder did move (update ratio 0.015-0.027, token cosine 0.998). **The control stays the base.** Numbers: EXPERIMENTS.md "Stage 3"; files `outputs/stage3_2026-10-22/` (local only).
- Lasting facts for any online-encoder work: (1) V-JEPA2 config has stochastic depth 0.1 + dropouts (train-mode tokens change 73 %) -> `OnlineEncoder` zeroes them, train mode only while gradients flow; (2) precision: TF32 0.5 % token change vs eval fp32, bf16 15 % -> fp32 (TF32 while training); bf16 saves only ~1 s of ~8.4 s per optimizer step; (3) **the old feature cache is NOT bit-reproducible**: today's encoder differs from it by ~2 % relative (cos 0.9986) on pod and laptop GPUs; the functional check is the epoch-0 val loss (1.078 +- 0.01, reproduced: 1.0779, AUC 0.9593).
- Cost / pace: RTX PRO 4500 32 GB, micro-batch 2, peak 12.7 GB, ~8.4 s / optimizer step (142 steps / epoch, ~20 min / epoch).

## Open TODOs (in order)
1. **User decision pending:** stage 4 (5 end-to-end seeds of the control recipe, test sets ONCE: seeds + ensemble vs crash model and BADAS-2.0, P2V as reference) now, or 3.1 (motion / gap auxiliary loss) first as its own plan. Recommendation given: stage 4 first (decision AUC has been flat at ~0.959 across every change; 3.1 would mainly help text faithfulness).
2. Write the plan file + driver + dry run for the chosen stage; the user opens pods.
3. Presentation website: Home page built (`website_pres/`); other pages follow in later rounds; user reviews wording (hero subtitle "causal reasoning", problem / solution text).
4. Review the 20-window pass-2 pilot with the user; open: P2V threshold chosen on validation, regularised Phase 2 variants, controls, faithfulness test, pre-alert human review.
5. Housekeeping: RunPod network volume (63 GB, $4.41/mo) unverified; HF private storage near the 100 GB limit (do not push epoch checkpoints); HF commit rate limit is 128 commits / hour per repo (two pods syncing + file-by-file uploads hit it: upload folders in one commit); unsent access requests (DRAMA, WTS, RoadSafe365, MM-AU authors).

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
Nothing is running. Stage-3 pods `otx4rtbdiql3lt` (J) and `uslgw2gcv1sxe8` (E) stopped by Claude on 22 Oct after training ended (remaining driver stages skipped on the user's request); no cron job is active. When killing a driver by name over ssh, `pkill -f <name>` also matches the ssh shell's own command line: kill by PID. A new pod is created by the USER; Claude copies code with `tar ... | ssh ... tar xzf - --no-same-owner`, the HF token by scp.

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
Branch `reasoning-path-vjepa2-llm`, pushed (by the user) at `1f70adb` (stage-3 code, pod drivers, fusion / best-check / relaxed-paraphrase scripts, tests, docs, `website_pres/`). Uncommitted after that: this handoff refresh of `docs_agents/`. `outputs/` is git-ignored.

## Next step
Get the user's choice for TODO 1 (stage 4 vs 3.1), then write its plan file in `~/.claude/plans/CCP based BADAS/`.
