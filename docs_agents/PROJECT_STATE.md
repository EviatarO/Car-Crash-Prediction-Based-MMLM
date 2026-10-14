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

## Status as of 2026-10-14

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

### Reasoning model results (2026-10-12..14; full tables: `outputs/r1_phase2r_2026-10-12/summary.md`, `outputs/r1_options_2026-10-12/summary.md`, `outputs/r1_levers_2026-10-13/summary.md`; HF `checkpoints/{phase2r,options_2026-10-12,levers_2026-10-13}/`)
- **Setup:** frozen V-JEPA2 (A1) last-layer tokens (cached) -> trained projector -> Qwen3-VL-4B (+ LoRA r16 in Phase 2). Answer format for EVERY window: `Reasoning: ... / Collision: yes|no. / Outcome: ...`; prompt lists the three fields.
  Train 3,557 windows (1,188 videos), validation 612 windows (204 videos = the old validation videos); each epoch 50/50 on `label_vis`, negatives stratified (normal / pre-alert). Decision score = log P(" yes") - log P(" no") read at the label "Collision:" (`r1_reasoning.py`);
  S_before = reasoning cut out, S_after = after the model's own reasoning, S_teacher = after the teacher's reasoning (AUC 1.0: the teacher text determines the verdict).
- **Run 1** (Phase 1 from the OLD DADA checkpoint, then Phase 2): text barely above chance (BERTScore 0.398 vs floor 0.346), 259 distinct explanations (one generic sentence x147), road-user match 0.59; S_after AUC 0.813 (saturated, 97 % |log-odds| > 5), S_before 0.909, crash model 0.919.
- **Options 1-3** (epoch-3 checkpoint / template cap / box position as words in the prompt): none improved the text; lower validation loss did not mean better generation.
- **Session 1 (the big finding):** a FRESH Phase 1 (projector only, from random) on the Nexar reasoning texts instead of the DADA checkpoint removed the template collapse: B3 536 distinct explanations (top sentence x6), wrong-video gap +0.178 (was +0.066), retrieval@10 0.52 (0.40), BERTScore 0.418 vs floor 0.341,
  fact agreement side 0.73 (was 0.49; baseline 0.39). Extra inputs (B4 crash-head attention per token, B5 layer-11 tokens, B45 both) add little (+-0.02 on texts); road-user TYPE stays at the majority baseline (0.56-0.59 vs 0.54) in every run.
  Probe (`r1_fact_probe.py`, balanced acc.): type 0.26 last layer / 0.15 layer 11 / 0.28 attention-weighted (chance 0.13); side 0.72 / 0.60 / 0.73; motion 0.36; gap 0.58 -> the encoder tokens carry side/gap/motion partly but road-user type hardly (caveat: pooled probe, 600-window old probe agreed).
  Decision: no run reaches the crash model (AUC 0.919); S_before at epoch 3 / with a threshold chosen on training windows is competitive (B4 balanced acc 0.813 vs crash 0.824 at P=0.5); S_after is saturated everywhere.
- **Teacher texts are label-informed** (S_teacher AUC 1.0): the student's verdict after its own reasoning inherits any text error with near-certain confidence.

### Phase 1 enlargement (IN PROGRESS, 2026-10-14; plan `~/.claude/plans/CCP based BADAS/2026-10-14_Plan-Phase1-Enlarged-Dataset.md`)
- **Goal:** enlarge training x4 in both text and windows, train Phase 1 up to 15 epochs (it was still improving at its 4-epoch cap), judged on an enlarged validation (2,448 pairs).
- **Shifted windows (DONE locally):** `r1_shift_windows.py` -> `outputs/teacher_dataset_2026-10/windows_shifted.jsonl` + `dataset/manifests/r1_nexar_reasoning_shifted.jsonl` (12,286 copies: 10,485 train / 1,801 val; windows end 1/2/3 source frames earlier;
  dropped 153 by the visibility rule, 63 pre-alert windows raised to 1 by hindsight, 5 first-frame < 0) + 24-window control set (`r1_nexar_reasoning_control.jsonl`, `windows_control.jsonl`). Cutter verified: re-cut originals are byte-identical, shifted frames differ; shifted crash-model P median |diff| 0.0005.
  NOT yet cut / encoded on a pod.
- **Rephrased texts (PARTLY DONE):** `r1_paraphrase_check.py export` wrote 28 input batches of 150 windows (`outputs/teacher_dataset_2026-10/paraphrase_batches/`); Sonnet subagents wrote `paraphrase_out/batch_NN.jsonl` (3 versions of explanation + outcome per window).
  Complete: batches 00-07 (150 windows each); partial: batch_09_part1-3 (90), batch_12 (40, contains a stray `@@END@@` line, skipped by the checker). Missing: 08, 10, 11, 13-27 (those agents hit the Claude session limit). The prompt that worked (7.5 % rejects) = strong "write from scratch, start each version with a different element, no >6 consecutive words, keep the 4 facts, add nothing";
  pilots in `paraphrase_pilots/` (pilot 1 with the weaker prompt had 63 % rejects).
  Checker (`check`): hard rejects = length, outcome words in the explanation, a road-user class or left/right that the original lacks (compared as SETS), outcome polarity from the known verdict, near copy (token-F1 >= 0.8); flags (for the Opus verifier) = motion/gap change or addition, type/side not stated, new colour/object/place word, unclear outcome polarity.
  Current numbers on the 1,329 windows returned: 91 % of versions accepted, 26-30 % flagged.
- **Decided not done yet:** finish the remaining ~2,840 windows (user was asked: option A Gemini 3.8 Flash via OpenRouter text-only, about $2-5, ~30 min, matches the teacher style; option B more Claude agents ~4-5 M tokens over several limit windows), a rewrite pass for windows with < 3 accepted versions,
  an Opus verification of all flagged versions + a 10 % sample (stop and verify all if the sample error rate > 3 %), 30 hand-checked.
- **Pod driver ready, untested on a pod:** `r1_pod_phase1aug.sh` (needs >= 150 GB container disk; pulls the original cache from HF, downloads videos, cuts + encodes the shifted windows, checks the 24 control windows reproduce the cached crash-model P, trains P1-aug and P1-long in parallel, evaluates both + old B3 Phase 1 + projector-output fact probes, pushes to HF `checkpoints/phase1aug_2026-10-14/`, verifies, stops the pod).
  Expected 8-9 h, ~$6-8 (under 80 % confidence). Tiny-model dry run of both runs with augmentation passed locally.
- **Website tab "Reasoning dataset - augmented" (Dataset page) NOT built** (`website/build_aug_data.py` does not exist yet); the tab "Reasoning model - runs" (Experiments page, `#r2`, `website/build_r2_data.py`) shows 8 runs, comparison table, probe, sortable/searchable window table.

### Access requests
Sent by the user 2026-10-04: TAU-106K (train split), VRU-Accident (source-id mapping + collision times).
Not sent: DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email (lotvsmmau@gmail.com; draft in §7 of
`2026-10-05_Plan-Week1-GoNoGo-rev3.md`; asks for academic-use confirmation + per-video fps of CAP-DATA). BDD100K portal showed a certificate error 2026-10-04. Drafts:
`outputs/dataset_review_2026-10/access_requests.md`. Nexar HF repo `nexar-ai/nexar_collision_prediction` is gated and the local HF
account is not authorised (full clips unavailable locally; the pod downloaded the training videos from this dataset with its HF token for the teacher run).

## Open TODOs (in order)
1. **Finish the rephrased texts** (see Phase 1 enlargement): decide the writer for the remaining ~2,840 windows (user was asked; recommended Gemini 3.8 Flash via OpenRouter text-only, ~$2-5; OpenRouter balance ~$7.5), run `r1_paraphrase_check.py check`, rewrite versions rejected, Opus verification of flagged + 10 % sample, hand-check 30.
2. **Website tab "Reasoning dataset - augmented"** (teacher text + 3 rephrased versions tagged `teacher` / `rephrased_v1..v3`, 4 window versions per window, filters) - write `website/build_aug_data.py` (Sonnet subagent is fine).
3. **Commit** the uncommitted `r1_paraphrase_check.py` change (skips malformed JSONL lines), then run the pod (>= 150 GB container disk, no network volume, ask the user to start it; copy code + HF token like the earlier sessions; `r1_pod_phase1aug.sh`). STOP for the user's approval before the pod.
4. **After Phase 1 (P1-aug vs P1-long vs old B3 Phase 1):** table of fact agreement / BERTScore / distinct / retrieval / decision / projector-output probe, then the user chooses the continuation (ranked list in the plan section 5: Phase 2 on the new Phase 1; direct fact signal = structured fact fields or auxiliary fact loss; Run 2 verdict-first;
   short reasoning; attention/layer-11 input on the new base; longer Phase 2; hallucination + grouped-type metrics; test set).
5. **REMIND the user: Run 2 (verdict-first ablation) is pending** (memory `pending-run2-verdict-first`; `auto_r2` stage exists in `r1_pod_run.sh`, not run). The user expects it to score <= the crash model.
6. **Test-set teacher reasoning (private 677 windows, E3 blind) is ON HOLD** (plan `2026-10-12_Plan-Test-Private-Teacher-E3.md`; needs boxes via a pod + ~$8 OpenRouter + a top-up).
7. Nexar test-set features (677 / 667 windows) for a representative final evaluation; DADA test features; text-only leakage test on the explanations; hand review of E3-kept windows.
8. Housekeeping: `website/teacher_data.js` blob (4.5 MB) is still in git history (commit `084fb56`) - offer to rewrite if the branch is not pushed yet; HF private storage is near its 100 GB limit (do not push the shifted-window features, 50 GB).
9. Unsent access requests (DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email); TAU / VRU replies pending. User pushes branch `reasoning-path-vjepa2-llm` and tags (`git push origin reasoning-path-vjepa2-llm --tags`); Claude never pushes.

## Known bugs / gotchas
- **Claude session limit:** launching ~15 Sonnet subagents in parallel (each 200-280 k tokens per 150 windows) exhausted the session limit and killed most of them (2026-10-14); run at most ~4 at a time, or use an API for bulk text.
- **Heredoc mangling:** Python heredocs through the Bash tool turn backslash-n inside strings into real newlines -> write a script file with the Write tool, or use Edit.
- **.sh files must be LF** (git autocrlf=true converts on commit); the copy to a pod runs `sed -i s/\r$//` on them; check with `bash -n`.
- **RunPod container disk default is 30 GB** and a network volume restricts the datacenter (few GPUs free): set Container Disk (Edit Pod) to >= 100 GB (150 GB for the Phase 1 enlargement) and use NO network volume; the pod has its own `/proc/1/environ` RUNPOD_API_KEY for self-stop.
- **Paraphrase checker pitfalls (fixed):** the first-mentioned road user / side changes when clauses are reordered -> compare sets; synonym verbs ("runs into") break a collision-word heuristic -> use the known verdict; `\bhit\b` must be whole-word ("white" contains "hit").
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
Nothing running. Pods used 2026-10-12..14: `mmiy8p5k4kx45v` (Run 1, ~3 h), `jkbrbgb451mp2z` (options 1-3, ~4.3 h) and `hfc3vjmx1hzxit` (Session 1 levers, 100 GB disk, ~10 h) - all stopped themselves via `runpodctl stop` only after the HF byte check. A new pod is started by the USER in the RunPod console
(then send SSH lines `ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519`); Claude copies code with `tar ... | ssh ... tar xzf - --no-same-owner`, copies the HF token with scp (never print it), runs `r1_pod_run.sh setup`, then the driver with `setsid nohup`.
Features cache layout on a pod: `$CACHE/features_nexar_reasoning.npy` (+ `_l11.npy`, `attn_...npy` after the extras pass), `index_nexar_reasoning.jsonl`; `/dev/shm` (29 GB) can hold one 17.5 GB cache when the disk is small.

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
python r1_build_reasoning_manifest.py                     # dataset/manifests/r1_nexar_reasoning_windows.jsonl from outputs/teacher_dataset_2026-10/dataset_final.jsonl (train/val by video; box_hint)
python r1_build_window_repo.py --source nexar_reasoning --repo eviatarO-org/nexar-reasoning-windows --work-dir <d>   # frames -> HF window repo (done, 4,169 windows)
python r1_cache_features.py --source nexar_reasoning --from-hf eviatarO-org/nexar-reasoning-windows --splits train,val --push-repo eviatarO-org/vjepa2-a1-features --out-dir <d>   # [--extras-only for layer 11 + attention]
python r1_train.py --phase 1|2 --sources nexar_reasoning --input last|last+attn|l11+last|l11+last+attn --cache-dir <c> --out-dir <o> --select val_loss [--aug-texts <paraphrases.jsonl> --aug-shifts --val-texts <p> --val-shifts --p-teacher 0.4] [--order verdict_first]
python r1_eval_reasoning.py --ckpt <best.pt> --input <type> --cache-dir <c> --out-dir <o> [--before-ckpt <p1> --old-ckpt <old p2> --train-threshold 600]   # predictions.jsonl + metrics.json + summary.md
python r1_review_reasoning.py --run-dir <results dir>        # BERTScore / ROUGE / BLEU with chance floors (PC)
python r1_fact_eval.py --runs name=<predictions.jsonl> ...    # 4-fact agreement vs the teacher (type / side / motion / gap) + majority baseline
python r1_fact_probe.py --cache-dir <c> --out-dir <o> --inputs last,l11,attn [--projector-ckpt <best.pt> --proj-tag <t>]
python r1_shift_windows.py ; python r1_paraphrase_check.py export --batch 150 ; python r1_paraphrase_check.py check
python r1_reasoning_test.py                                   # CPU unit tests (round trips, sampler, merger inputs, cache layout, fact extractor, probe, augmentation)
bash r1_pod_run.sh <features_r|pull_r|get_old|smoke_r|p2r reason_first|verdict_first|g2r|push_r|verify_r|stop_r|auto_r|auto_r2>   # Run 1 / Run 2 stages on a pod
bash r1_pod_options.sh ; bash r1_pod_levers.sh ; bash r1_pod_phase1aug.sh                 # unattended pod drivers (verify on HF, then self-stop)
PYTHONUTF8=1 python ../../website/build_r2_data.py            # website tab "Reasoning model - runs"
```
Crash-path commands (champion recipe, public scoring, paired bootstrap) are unchanged: see `history/2026-09_PROJECT_STATE.md`.

## Git state
Branch **`reasoning-path-vjepa2-llm`**, HEAD `de29160` (Phase 1 enlargement code); `origin/main` at `3463da6`; **not pushed** (user pushes, with `--tags`). Uncommitted: `student_training/scripts/r1_paraphrase_check.py` (skips malformed JSONL lines).
Tags `phase1-2026-10-06` (= `ac40861`), `phase2-2026-10-06`. `dataset/`, `outputs/`, `reports/` are git-ignored (so `edits.jsonl`, paraphrases, manifests and the progress reports are local only; results are on HF: `eviatarO-org/checkpoints`
(phase1, phase2, phase2r, options_2026-10-12, levers_2026-10-13), `teacher-pilot-2026-10`, `teacher-dataset-2026-10`, `nexar-reasoning-windows`, `vjepa2-a1-features/nexar_reasoning/`).
Progress reports: `reports/progress_reports/2026_10_11_progress_report.docx`, `2026_10_14_progress_report.docx` (generator `reports/_scripts/_build_progress_report.py`, overwritten each run).

## Next step
Ask the user how to finish the rephrased texts (Gemini via OpenRouter recommended, or more Claude agents at <= 4 at a time), complete + verify them, build the augmented-dataset website tab, commit, then (after the user's go and a >= 150 GB pod) run `r1_pod_phase1aug.sh`. Remind about Run 2.
