<!-- handoff-month: 2026-10 -->
# Project State

Live files hold the current month only. Full prior state: `history/2026-09_{PROJECT_STATE,ARCHITECTURE,EXPERIMENTS,DECISIONS}.md`
(Jun–30 Sep 2026) and `history/2026-08_*.md`.

## Goal
MSc thesis on Nexar dashcam collision anticipation. **The thesis question changed (Oct 2026):** the crash-score path is
**FROZEN** (A1-compress256, AP 0.913 private / 0.910 public); the new question is whether a language model can **explain an
alert from the frozen V-JEPA2 encoder's own features** (frozen encoder → trained projector → Qwen3-VL-4B language model),
so explanation and score share one vision pass. BADAS-2.0 (p.8) names this latent-based reasoning as future work.
New work lives on git branch `reasoning-path-vjepa2-llm` (root `README.md` = goal + folder map).

## Status as of 2026-10-12

### Crash-score path (frozen — do not reopen without the user)
- Champion **A1-compress256**: test AP **0.9128** (private 677) / **0.9096** (public 667). Recipe: LoRA r16/α32/dropout 0.05 on
  `query,key,value`, crash CE only, lr 2e-4 constant, grad_accum 8, 1,761-window pool, head frozen, `--preprocess compress256`
  (pass it explicitly; code default is still `crop`). Adapter: `outputs/a1_compress256/train/epoch_02/lora_adapter` (local + volume).
- Midpoint-negatives recipe raises pooled AP (+0.0092 ± 0.0052, 5/5 seeds) but Kaggle mAP gain not confirmed. Never report the single best checkpoint.
- Versus published BADAS-Open (crop) 0.853 / 0.871: correct-at-0.5 private TTE0.5/1.0/1.5/neg = 135/104/69/209 → 129/101/56/284;
  most of the AP gain comes from preprocessing (A0-compress256, untrained, 0.907 / 0.904).

### Reasoning path — week 1 (run on a pod 2026-10-06: Phase 1 and Phase 2 done; gate 1 met, reasoning quality NOT yet)
- **Plans (`~/.claude/plans/CCP based BADAS/`):** `2026-10-05_Plan-Week1-GoNoGo-rev3.md` (design, gates), `2026-10-05_Plan-HF-Window-Repos-and-Pod.md` (HF data).
- **Data on HF, org `eviatarO-org` (private, READMEs at every level; rule in memory `hf-project-grouping`):** `nexar-windows` (1,346 valid windows after the
  8-frame margin; the 111 removed are in `windows_excluded.jsonl`), `mmau-dada-windows` (4,582), `vjepa2-a1-features` (encoder tokens + A1 score, Nexar 1,457
  incl. the 111, DADA train/val 3,453; DADA test not encoded), `checkpoints` (model repo: `phase1/`, `phase2/` with `best.pt`, `train_log.jsonl`, `epochs/`, `results/`).
- **Visibility rule is now identical for every source:** crash window kept only if time-to-alert ≥ TTE + 8 frames (0.27 s). Nexar valid crash 353 train / 88 val; DADA 1,870 / 629 / 800.
- **Phase 1 (option b):** merger only, 1,049 DADA crash windows with A1 P ≥ 0.5, random init, 15 epochs; checkpoint = epoch 15 (val loss was best at epoch 5 → over-fit;
  per-epoch checkpoints were not kept). Code tag `phase1-2026-10-06` (= ac40861).
- **Phase 2:** from Phase-1 best, merger + LoRA r16, 3,667 windows/epoch (50/50), stopped after epoch 4 by `--stop-on-val-rise 2`; kept epoch 3.
- **Results and numbers:** EXPERIMENTS.md §2f–2g; files `outputs/r1_week1/{summary.md, pod_phase1_2026-10-06/, pod_phase2_2026-10-06/}` (git-ignored; copy on HF `checkpoints/phase*/results/`).
- **Honest status:** the LM uses the V-JEPA2 tokens (wrong-/blank-video gaps > 0, cause choice 53 % vs blank 22 %), but written reasoning is weak: DADA exact event 24 %,
  Nexar event text ≈ floor, time-to-impact not learned, DADA no-crash false alarms 83 %. Decision on the next step is the user's (DECISIONS.md, open question 1).
- **Supporting results:** feature probe (`outputs/r0_feature_probe/n600/`), dataset reviews (`outputs/dataset_review_2026-10/`, incl. `training_windows_hf/`),
  data review (`outputs/r1_week1/data_review_2026-10-05/`), plan deck `reports/presentations/2026-10_reasoning-path-plan.pptx`.

### Nexar reasoning dataset (teacher run DONE 2026-10-10..12; hand-reviewed; ready for the student retrain)
- **Why:** Nexar has no reasoning text; the August captions carry what the encoder cannot (colour, model). Goal: explanation of the object the frozen crash model attends to, in words the V-JEPA2 tokens can support.
- **Dataset** (`outputs/teacher_dataset_2026-10/dataset_final.jsonl`, rebuilt by `website/build_teacher_data.py`): **4,169 windows, 1,392 videos** (3 windows per video: crash TTE 1.5 / 1.0 / 0.5 s, normal windows before a
  fake event at the midpoint), **`label_vis` 1,330 positive / 2,839 negative** (1,309 crash windows kept by the visibility rule, 610 pre-alert crash windows (label 1 → label_vis 0, 18 of them turned 1 by hindsight), 2,250 normal).
  Per window: Nexar `label`, `pre_alert`, `label_vis`, `tte`, 25-45-word explanation, 4-15-word outcome-only prediction, risk 0-100, path, status, `edited`. 35 genuine crash outliers removed (`removed_windows.json`).
- **Teacher flow** (full description + results: `teacher_distillation/TEACHER_DATASET_FLOW.md` §1-9; routing figure `reports/figures/teacher_prompt_routing_2026-10-10.png`): Gemini 3.8 Flash on OpenRouter Flex, boxed frames.
  Stage 1 E3 (blind) on every window + text-only short prediction; stage 2 only where the verdict differs from `label_vis`: E7-pre (hindsight, pre-alert windows), E6 (anchored on a correct sibling), E7 on TTE 0.5 when no anchor,
  then an E6 chain (E6C). Final paths: E3 3,027 · E6 412 · E6C 258 · E7 266 · E7-pre 206. Teacher spend ≈ $63 (OpenRouter balance ≈ $7.5 left); GPU step on a RunPod pod (boxes, tracks, attention), results on HF.
- **Hand review (2026-10-12):** the user commented on 28 windows on the website; 30 windows were rewritten (18 crash, 12 normal) in `outputs/teacher_dataset_2026-10/edits.jsonl` (new text + comment + original),
  applied by `build_teacher_data.py`; windows with an edit are never dropped (00512 TTE 1.5 reinstated); every edited crash window is `label_vis` 1. The "check" list (verdict ≠ label_vis) is empty. The user's notes file: `ccp-review-notes-2026-10-11.json` (repo root).
- **Website:** Dataset page → tab "Teacher reasoning" (4,169 rows, Review filter, notes with export/import; `website/teacher_data.js` is generated and git-ignored: rebuild with `PYTHONUTF8=1 python website/build_teacher_data.py`).
- **HF:** private repo `eviatarO-org/teacher-dataset-2026-10` (tracks.tar.gz, attn.tar.gz, windows.jsonl, logs; boxed frames are rebuilt locally in `outputs/teacher_dataset_2026-10/boxed/`).
- **Report:** `reports/progress_reports/2026_10_11_progress_report.docx` (2 pages, delivered; numbers updated to 4,169 / 1,330 / 2,839 on 2026-10-12 via `reports/_scripts/_build_progress_report.py`).

### Access requests
Sent by the user 2026-10-04: TAU-106K (train split), VRU-Accident (source-id mapping + collision times).
Not sent: DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email (lotvsmmau@gmail.com; draft in §7 of
`2026-10-05_Plan-Week1-GoNoGo-rev3.md`; asks for academic-use confirmation + per-video fps of CAP-DATA). BDD100K portal showed a certificate error 2026-10-04. Drafts:
`outputs/dataset_review_2026-10/access_requests.md`. Nexar HF repo `nexar-ai/nexar_collision_prediction` is gated and the local HF
account is not authorised (full clips unavailable locally; the pod downloaded the training videos from this dataset with its HF token for the teacher run).

## Open TODOs (in order)
1. **Student retrain on the new dataset** (needs a GPU pod; stop-gated, user approves the runbook): Phase 1 / Phase 2 with targets = explanation, outcome sentence, `label_vis` verdict; split by video; compare with the DADA+caption model on the same Nexar windows
   (BERTScore vs floor 0.359, retrieval, yes/no AUC). Open: whether to mix DADA back in.
2. **Text-only leakage test** (how much of `label_vis` the explanation alone reveals; E6 / E7 explanations were written knowing the outcome) and a hand review of a sample of the E3-kept windows.
3. Nexar test-set features (677 / 667 windows, GPU) for a representative final evaluation; DADA **test** split features (about 4 min on a GPU pod).
4. Housekeeping: `website/teacher_data.js` blob (4.5 MB) is still in git history (commit `084fb56`) - offer to rewrite if the branch is not pushed yet.
5. Optional / postponed by the user: per-word-piece probabilities, LLM judge on inference outputs only; Δpredictor/ΔLoRA look-back (memory `pending_delta_ratio_diagnostic`).
6. Unsent access requests (DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email); TAU / VRU replies pending.
7. User pushes branch `reasoning-path-vjepa2-llm` and tags: `git push origin reasoning-path-vjepa2-llm --tags`; Claude never pushes.

## Known bugs / gotchas
- **Windows console:** always `PYTHONIOENCODING=utf-8 PYTHONUTF8=1`. Bash heredocs with nested quotes fail in this tool — write a scratch .py and run it.
- **HF cache symlinks fail on Windows** (`WinError 1314`): use `snapshot_download(..., local_dir=...)`. Qwen3-VL tokenizer/processor files are in
  `dataset/public_samples/qwen3vl_cfg/`.
- **DADA parts:** the 58 `DADA2000.part_*` files are pieces of ONE tar.gz — read in order; a single part ends mid-member (`tarfile.ReadError`, use
  `--allow-truncated` for dry runs). HTTP streaming dropped mid-part once → the pod downloads parts with `hf_hub_download` + retry.
- DADA key = `(type, video)` (folder `DADA2000/<type>/<video:03d>/images/NNNN.png`, 1-based frames, 1584×660, 30 fps); every ArA row matches Sheet1 on it.
- Legacy `.xls` (MM-AU): no `xlrd`; converted read-only via Excel COM to `dataset/public_samples/mmau_ara/*.csv` and `mmau/cap_text_annotations_conv.xlsx`.
- TAU-106K `accident_segments` are strings (`[["0.62","0.73"]]`); `accident_objects` may be null.
- Never `pip install -U torch` on a RunPod image. Two BADAS-loading processes at once can crash one. `peft.save_pretrained` needs the model-card stub.
- Resuming an `--unfreeze-head` run past epoch 1 requires `--head-init`. `score_checkpoints_on_test.py NAME=NONE` resets the LoRA (fixed 2026-09-14).
- Join caption/label files on `frames_dir` only. Backgrounded commands piped through `tail` mask exit codes.
- RunPod: volume `0hnvco2s4j` (EU-RO-1) **at its 56 GB quota**; container disk is not billed when stopped (treat as wiped); outputs go to `/root`;
  results synced to local **before** `runpodctl stop`. Saved SSH hosts in `~/.ssh/config` are stale. Never generate a new SSH key.
- Local GPU is a 6 GB laptop card: BADAS encode ~15 s/window; cannot hold Qwen3-VL-4B. Running the detector/tracker and the A1 attention model at the same time (or two copies of a job) makes the PC crawl.
- **Long local jobs:** start them as the DIRECT background command (`run_in_background`), never `nohup ... &` inside it: the wrapper exits and the first copy can survive as an orphan (two copies once ran in parallel);
  find stray ones with PowerShell `Get-CimInstance Win32_Process` and stop them. Never `pkill -f` a string that also appears in your own ssh command line (it kills the shell).
- OpenRouter `usage.cost` shows prompt-cache discounts: identical concurrent requests (same frames + text) got cache hits (R1 33/51 calls), so cost per window is only fair on runs without hits.
  Served tier is reported in `response.service_tier` (`flex` confirmed; Standard shows `default` / `provisioned`).
- Website: `website/serve.py` normally runs on :8765 (preview_start fails with "port in use": just navigate to
  `http://localhost:8765/MMLM_For_Cars_Collision_Anticipation/MMLM_AI/website/experiments.html#pilot`). Rebuild `pilot_data.js` with `website/build_pilot_data.py` after any run changes.

## Pod state
Nothing running. The teacher-run pod (RTX PRO 4500, 30 GB disk, batches of 150 videos) stopped itself via `r1_teacher_pod.sh auto` only after the results were verified on HF (2026-10-10). Earlier pod `8tnse35hx8rnqe`
was stopped by Claude 2026-10-06 13:26 UTC after the bundle was downloaded. Claude can stop pods itself: key from the pod's `/proc/1/environ` (`bash r1_pod_run.sh check_stop` first, `stop <pod_id>` after the bundle is local;
memory `pod-self-stop`). The user's RunPod account has the PC's public key saved (new pods accept it). HF token for pods: `/workspace/.cache/huggingface/token`
if the old volume is attached, else the user pastes it in the pod's web terminal. Container disk must be ≥ 60 GB.

## Important commands (from `MMLM_AI/` unless noted)
```bash
python student_training/scripts/r1_bridge_test.py          # bridge unit tests (CPU, tiny random LM)
python student_training/scripts/r1_build_manifests.py      # rebuild manifests + assert counts (DADA 3,299; Nexar 353 / 88 with the 8-frame margin)
cd student_training/scripts
python r1_cache_features.py --source nexar --splits train,val --out-dir <dir> [--limit N]
python r1_cache_features.py --source dada --parts-dir <dir>/DADA-2000_chunks --n-parts 58 --delete-parts --out-dir <dir>
python r1_download_dada_parts.py --dir <dir> --max-ahead 6
python r1_train.py --phase 1 --cache-dir <c> --out-dir <o> --init random --epochs 15 --lr-merger 1e-3 --grad-ckpt
python r1_train.py --phase 2 --cache-dir <c> --out-dir <o> --init-from <phase1>/best.pt --epochs 8 --lr-merger 2e-5 --lr-lora 2e-4 --grad-ckpt
python r1_eval_gates.py --phase 2 --source nexar --ckpt <o>/best.pt --cache-dir <c> --out-dir <g>
#   smoke tests locally: add --lm tiny --device cpu --val-split train
bash r1_pod_run.sh <setup|tests|pull|get_p1|p1_ab|p1_full random|g1|p1_complete|p2|g2|push_ckpt|bundle|check_stop|stop POD_ID>   # on the pod (code in /root/r1)
python r1_review_phase2.py --run-dir <bundle dir>     # review file + curves + yes/no by group + text metrics (local, no GPU)
python r1_phase_report.py --run-dir <bundle dir> --phase 1 [--full]   # Phase-1 curves + text metrics
python r1_pilot_windows.py [--dry-run]                      # pilot window selection (+ cuts the missing normal windows)
python r1_attention_boxes.py --stage tracks|attn|boxes [--set A]   # detection+tracking, crash-head attention, boxed frames + grids
python r1_teacher_pilot.py --name R2 --set A --tier flex --order explanation_first --model google/gemini-3.8-flash --concurrency 16   # --flex-test = one call
python r1_teacher_pilot_report.py                            # outputs/teacher_pilot_2026-10/summary.md + review.xlsx
python ../../website/build_pilot_data.py                     # website tab data (pilot)
python r1_teacher_windows.py                                 # manifest of the 4,203 teacher windows (Step A)
bash r1_teacher_pod.sh <setup|hf_check|smoke|batches|push|verify|auto|stop>   # on the pod (GPU step B: tracks, attention, boxes -> HF)
python r1_teacher_pilot.py --prompt e3 --set full --tier flex --concurrency 32   # stage 1; stage 2: python r1_teacher_flow.py; short predictions: python r1_add_prediction.py
PYTHONUTF8=1 python ../../website/build_teacher_data.py      # teacher-dataset website tab + dataset_final.jsonl + check_windows.txt (applies edits.jsonl)
PYTHONUTF8=1 python ../../reports/_scripts/_build_progress_report.py   # progress report .docx
python dataset_sample_review.py --only <dada|tau|caviar|mmau|vru|llava|bddx>   # 3 seeded review samples per dataset
python r0_feature_probe.py --config ../configs/e4_stageA.yaml --lora-adapter ../../outputs/a1_compress256/train/epoch_02/lora_adapter --n-windows 600 --out-dir <dir>
python build_reasoning_path_deck_2026-10.py                # rebuilds the October deck (asserts numbers from score files)
```
Crash-path commands (champion recipe, public scoring, paired bootstrap) are unchanged: see `history/2026-09_PROJECT_STATE.md`.

## Git state
Branch **`reasoning-path-vjepa2-llm`**, HEAD = the commit after `0b1c8c0` that adds the hand-edit support + this handoff (see `git log`); `origin/main` at `3463da6`; **not pushed** (user pushes, with `--tags`).
Tags `phase1-2026-10-06` (= `ac40861`), `phase2-2026-10-06`. `dataset/`, `outputs/`, `reports/` are git-ignored (so `edits.jsonl` and the report are local only; results are also on HF:
`eviatarO-org/checkpoints`, `eviatarO-org/teacher-pilot-2026-10`, `eviatarO-org/teacher-dataset-2026-10`).

## Next step
Write the runbook + dry-run for the student retrain on `dataset_final.jsonl` (new targets, split by video, DADA mix decision), show it to the user, and stop for approval before starting any pod (memory `e4_stage_gating`).
