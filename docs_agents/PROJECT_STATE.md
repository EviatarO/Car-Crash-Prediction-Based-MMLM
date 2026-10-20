<!-- handoff-month: 2026-10 -->
# Project State

Live files hold the current month only. Full prior state: `history/2026-09_{PROJECT_STATE,ARCHITECTURE,EXPERIMENTS,DECISIONS}.md`
(Jun–30 Sep 2026) and `history/2026-08_*.md`. Last refreshed 2026-10-20.

## Goal
MSc thesis on Nexar dashcam collision anticipation. The crash-score path is **FROZEN** (A1-compress256, AP 0.913 private / 0.910 public).
The thesis question is now: can ONE vision pass (frozen V-JEPA2/BADAS A1-compress256 encoder -> trained projector -> Qwen3-VL-4B + LoRA)
give the alert AND its explanation, and can its decision approach BADAS (Kaggle mAP: BADAS-1.0 0.925, BADAS-2.0 0.940, BADAS-2.0-Flash 0.941)?
Decision score = log P(" yes") - log P(" no") at the label "Collision:". Work lives on branch `reasoning-path-vjepa2-llm` (root `README.md` = goal + folder map).

## Standing rules of the user (binding)
- Never print API keys / the HF token (copy by scp only). **Claude never pushes git** (user pushes, with `--tags`); commit only on request.
- **The USER creates pods**; Claude copies code, launches the driver, the driver self-stops only after HF byte verification. Confirm before any spend beyond an approved amount.
- No redesign of an approved plan while the user is away: define gates (major/minor + action) first. Answer plan questions in the terminal too.
- Test texts only for final evaluation, never training / selection. Pre-alert windows removed from training and from scores. Object crop excluded.
- Professor-facing documents: no internal run names. Short, plain answers; flag anything < 80 % confidence; state the threshold of every thresholded metric.
- Do NOT touch Windows power settings. Per-experiment `summary.md` is auto-maintained.

## Status as of 2026-10-20

### Crash-score path (frozen — do not reopen without the user)
Champion **A1-compress256**: test AP **0.9128** private (677) / **0.9096** public (667), pooled 0.9110 (AUC 0.913, Kaggle mAP 0.916). Recipe, ledger: EXPERIMENTS.md §1.

### Datasets (all private HF org `eviatarO-org`; local copies git-ignored)
- **Teacher dataset (original):** 4,169 windows / 1,392 videos (TTE 0.5/1.0/1.5 s + normal), `label_vis` 1,330 pos / 2,839 neg; flow E3 blind -> E6 / E7 / E6C second look; hand-edited 30 windows. `outputs/teacher_dataset_2026-10/dataset_final.jsonl`.
- **Extra TTE windows (2026-10-19, DONE):** 860 new crash windows at TTE **0.75 s (534) and 1.25 s (326)** (743 train / 117 val; 10 contradicting-verdict windows removed). Same teacher flow (flow step `--process-horizons 0.75,1.25`), outcome sentences, 13 contact-word versions removed after the Opus audit.
  `outputs/teacher_extra_2026-10/` (summary.md, RUNBOOK.md REV 2, `dataset_final.jsonl`, `paraphrases_judged_v2.jsonl`, `fact_cards_v2.jsonl`); manifests `dataset/manifests/r1_nexar_reasoning_{extra,v2,test,test_public}.jsonl` (v2 = 5,029 = 4,169 original + 860).
- **Rewordings** (3 per window, Gemini 3.8 Flash writer -> automatic check -> Sonnet 5.5 strict judge (8 rules) -> Opus blind audit): original data 4x; new windows: 4,416 unique versions (used 2,119, spare 331, judge-rejected 1,084, auto-rejected 865, removed 13, unjudged 4); accepted new versions per group train 0.75: 1,128 / 1.25: 722, val 165 / 104; 35 train windows have 0 versions. Judge rejects ~35 %. Opus audit minor-nuance rate 18.8 % (original data 7.2 %; not like-for-like: different severity wording). Validation uses the teacher text only (`--val-texts none`).
- **Test sets:** private 677 + public 667 = 1,344 windows; 431 private windows have an agreed teacher reference (E3 blind pass 1, pass 2 anchored not run). No near-duplicates between train and test (`r1_overlap_check.py`, all 1,344).
- **Website:** Dataset page tabs: Teacher reasoning (+ `teacherx` new windows), Reasoning dataset - augmented (+ `augx` new-window rewordings: accepted + rejected, status chips, reasons, no horizontal scroll), Teacher reasoning - test (private; review notes). Rebuild: `PYTHONUTF8=1 python website/build_aug_extra_data.py`, `build_teacher_data.py --folder ... --js ... --var ... --dataset-key ... --no-dataset-files`.

### Reasoning models (details + numbers: EXPERIMENTS.md §2o–2t)
- **Today's P2V (verdict-first, seed 0, trained 2026-10-15):** pooled-test AP 0.926 / AUC 0.928 / Kaggle mAP 0.928 vs crash model 0.911 / 0.913 / 0.916 (dAP +0.015 [+0.005, +0.025]); per-TTE AP 0.5/1.0/1.5 = 0.948 / 0.936 / 0.900 (BADAS-2.0: 0.943 / 0.957 / 0.921). **On validation P2V does NOT beat the crash model** (AUC 0.953 vs 0.957); train AUC 0.989 (over-fit); P2V's best epoch is always the first.
- **Chain B (new data + gentler recipe, 1 seed, finished 2026-10-20, pushed to HF `checkpoints/chains_2026-10-19/`, NOT yet downloaded / analysed locally):** epoch 1 pooled-test AP 0.9204, Kaggle mAP 0.9177, val core AUC 0.942, val full AUC 0.938 -> does not beat today's P2V (rule: needs +0.016 AP, >= on val and on both test halves). Epochs 2-3, `B_best` train600 / val_text evals are on HF; top-3 = epochs 1 2 3. Pod was unreachable at 2026-10-20 07:55 UTC (presumably self-stopped; the user should confirm in the RunPod console).
- Crash-head path is no longer needed in the final model (user agreed): two separate architectures (crash model / reasoning model) are kept.

### Test-set teacher reasoning
Private 677 pass 1 (E3 blind) done; **pass-2 pilot of 20 windows is waiting for the user's review** (full pass 2 not run). Website tab shows P2R / P2V verdicts + reasoning per test window, confusion matrices per TTE, "teacher wrong, model right" filters.

## Open TODOs (in order)
1. **Answer the user's pending question (style vs diversity of rewordings):** "are rewordings rejected mostly for slang / style rather than added information? Is a single uniform format better for training, or diversity?" First quantify from `outputs/teacher_extra_2026-10/paraphrase/{paraphrase_judge.jsonl, paraphrase_judge_rejects.jsonl, paraphrase_rejects.jsonl}` (judge problems by rule number; style vs fact changes; auto-check reasons such as near copy), then recommend. Do not change anything without the user; a test needs a training run.
2. **Download chain B from HF** (`curl.exe --ssl-no-revoke` or `snapshot_download(local_dir=...)` + size check), write the full report: B top-3 epochs vs today's P2V on val core, val full (0.75 / 1.25 s groups separately), private, public, pooled; loss curves of the three stages; text metrics; per-TTE AUC (val/train per-TTE AP is not comparable: crash TTE k vs ALL normals ~1:10; test pairs 1:1). Helper scripts: scratchpad `prelim_compare.py`, `show_logs.py` (copy to `student_training/scripts/` if kept). Confirm the pod stopped; per-experiment `summary.md`.
3. **Seeds plan** `~/.claude/plans/CCP based BADAS/2026-10-20_Plan-Seeds-Of-Todays-P2V.md`: user must decide (a) 4 new seeds (recommended, 5 total) vs 2, (b) reuse the current pod (restart in console) vs new pod (32 GB GPU, 80 GB disk). Then write `r1_pod_seeds.sh` (from `r1_pod_chains.sh`, two parallel streams, `--seed k` in P1fixR + P2V, scoring evals only) and `r1_seed_report.py` (rules R1-R3 fixed in the plan), dry-run, ask the user for a pod. ~3-4 h, ~$2-2.7.
4. Review the 20-window pass-2 pilot with the user; decide whether to run pass 2.
5. Open experiment backlog (`2026-10-18_Plan-P2V-Report-and-Experiment-Backlog.md`): P2V threshold chosen on validation (free), regularised Phase 2, score fusion with the crash model (free), controls (verdict-only, crash-head retrain), pre-alert human review, faithfulness test, ViT-L LoRA fine-tune of the encoder (user: "retrain the LoRA of the V-JEPA2"), motion/gap loss, extra window points.
6. **Remind the user:** Run 2 (verdict-first) was done as P2V; chain A (the original wider plan) was deliberately NOT run.
7. Housekeeping: RunPod network volume (63 GB, $4.41/mo) contents unverified; the user decided to skip and delete only in the console; HF private storage near 100 GB (do not push shifted-window features); `website/teacher_data.js` blob in git history (commit `084fb56`); unsent access requests (DRAMA, WTS, RoadSafe365, MM-AU authors); TAU / VRU replies pending.

## Known bugs / gotchas
- **CRLF on Windows:** Python `write_text` converts to CRLF -> bash on the pod fails (`$'do\r'`); write scripts with newline-preserving writes (`write_bytes`) or the Write / Edit tools; `.sh` must be LF (`sed -i s/\r$//`, `bash -n`). PowerShell single-quoted `\r\n` is literal.
- Heredocs with nested quotes / `\n` break through the Bash tool: write a scratch .py with the Write tool. `sleep` is blocked in the tool (use Monitor / background).
- **Pod:** container disk default 30 GB -> set 80 GB (150 GB if encoding shifted windows), NO network volume; outputs under `/root`; background `r1_hf_sync.py` loop pushes epochs every 10 min (pod-loss safety); restore from HF on a fresh pod; self-stop key from `/proc/1/environ`.
- Windows HF downloads: `curl.exe --ssl-no-revoke` + size check; HF cache symlinks fail (`WinError 1314`) -> `local_dir`. Always `PYTHONUTF8=1`.
- Don't launch > ~4 Claude subagents in parallel (session limit); use an API for bulk text. Never `pkill -f` a string in your own ssh command; long local jobs as a direct background command, not `nohup &`.
- OpenRouter `usage.cost` includes prompt-cache discounts; Flex tier confirmed via `service_tier`. Keep-awake tool lasts ~5 min after idle.
- Never `pip install -U torch` on a RunPod image; two BADAS-loading processes can crash one. Crash-path gotchas: `history/2026-09_PROJECT_STATE.md`.
- Per-TTE AP on val/train is not comparable to test (pairing 1:10 vs 1:1): use per-TTE AUC.

## Pod state
Nothing should be running. Last pod `wyqpkgbsvh5ysy` (RTX PRO 4500, 80 GB, no volume; chain B, 7.7 h actual vs 16-18 h estimated; SSH `root@213.173.104.92 -p 16860` refused on 2026-10-20 = stopped). Earlier pods stopped themselves after the HF byte check. A new pod is started by the USER (SSH lines `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`); Claude copies code with `tar ... | ssh ... tar xzf - --no-same-owner`, the HF token by scp, runs `r1_pod_run.sh setup`, then the driver with `setsid nohup`.

## Important commands (from `MMLM_AI/`)
```bash
cd student_training/scripts
python r1_reasoning_test.py                      # 18 CPU unit tests (all pass)
python r1_train.py --phase 1|2 --sources nexar_reasoning --cache-dir <c> --out-dir <o> --select val_loss|... [--aug-texts <p> --facts-cards <p> --drop-prealert --balance-tte --val-texts none --p-teacher 0.4 --order reason_first|verdict_first --fixed-epochs N --no-extra-train --seed k]
python r1_eval_reasoning.py --ckpt <best.pt> ... [--sample N --no-generate --no-extra-train]   # groups incl. tte0.75 / tte1.25
python r1_test_report.py ; python r1_full_report.py   # AP/AUC/Kaggle mAP + paired bootstrap; full P2V report -> outputs/r1_p2v_report_2026-10-18/
python r1_chain_select.py                        # top-3 epochs by val AUC
bash r1_pod_chains.sh                            # chain B driver (restore, prep, smoke, sync loop, P1aug-v2 12 ep -> P1fixR-v2 4 ep -> P2V-v2, evals, push, self-stop); r1_pod_public.sh / r1_pod_trainsplit.sh = earlier evals
python r1_extra_windows.py ; python r1_extra_setup.py ; bash r1_extra_pipeline.sh ; python r1_extra_gates.py ; python r1_extra_merge.py ; python r1_extra_audit_report.py
PYTHONUTF8=1 python ../../website/build_aug_extra_data.py
```
Older commands (teacher flow, caching, probes, week-1 gates, crash path): `history/2026-09_PROJECT_STATE.md` and the scripts' docstrings.

## Git state
Branch **`reasoning-path-vjepa2-llm`**, HEAD `cbf7523`; `origin/main` at `3463da6`; **not pushed**. 42 uncommitted entries: modified r1_* scripts and website (aug.js, pilot.js, site.css, testref.js, dataset.html, build_*), untracked new scripts (`r1_chain_select.py`, `r1_extra_*.py`, `r1_full_report.py`, `r1_hf_sync.py`, `r1_overlap_check.py`, `r1_pod_chains.sh`, `r1_pod_public.sh`, `r1_pod_trainsplit.sh`, `r1_test_refs_join.py`, `r1_test_report.py`, `r1_text_eval.py`), `website/aug_extra_data.js`, `website/build_aug_extra_data.py`, `website/teacher_extra_data.js`, `prompts/PROMPT_TEACHER_TEST_P2.py`, `ccp-test-review-notes-2026-10-17.json`. Commit only when the user asks; the user pushes. `dataset/`, `outputs/`, `reports/` are git-ignored; durable results are on HF (`checkpoints/{phase1aug_2026-10-14,chains_2026-10-19,...}`; the 2026-10-15 P2V run's outputs are local in `outputs/r1_phase2f_2026-10-15/`).
Progress reports: `reports/progress_reports/2026_10_11_*.docx`, `2026_10_14_*.docx` (generator `reports/_scripts/_build_progress_report.py`).

## Next step
Answer the rewordings style-vs-diversity question with data (TODO 1), then download + report chain B (TODO 2), then get the user's decisions for the seeds plan (TODO 3).
