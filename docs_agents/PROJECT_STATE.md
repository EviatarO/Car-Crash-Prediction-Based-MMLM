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

## Status as of 2026-10-06

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

### Access requests
Sent by the user 2026-10-04: TAU-106K (train split), VRU-Accident (source-id mapping + collision times).
Not sent: DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email (lotvsmmau@gmail.com; draft in §7 of
`2026-10-05_Plan-Week1-GoNoGo-rev3.md`; asks for academic-use confirmation + per-video fps of CAP-DATA). BDD100K portal showed a certificate error 2026-10-04. Drafts:
`outputs/dataset_review_2026-10/access_requests.md`. Nexar HF repo `nexar-ai/nexar_collision_prediction` is gated and the local HF
account is not authorised (full clips unavailable; 16-frame windows exist locally).

## Open TODOs (in order)
1. **User decides the next step** after reading `outputs/r1_week1/pod_phase2_2026-10-06/review/phase2_validation_outputs.md` (options in DECISIONS.md, open question 1).
2. Not done yet: encode the DADA **test** split features (≈4 min on a GPU pod) before any final evaluation; DADA test is untouched.
3. The reasoning path was evaluated only on the 1,761-pool val split (266 valid windows; enriched with mined A1 failures; A1 did not train on it). Not yet evaluated on the representative Nexar test sets (677 private / 667 public): needs their encoder features (GPU).
4. Optional evaluation additions the user postponed: per-word-piece probabilities (actor / action words), LLM judge on inference outputs only.
5. Unsent access requests (DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email) and TAU / VRU replies pending.
6. User pushes branch `reasoning-path-vjepa2-llm` and the tag: `git push origin reasoning-path-vjepa2-llm --tags`; Claude never pushes.

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
- Local GPU is a 6 GB laptop card: BADAS encode ~15 s/window; cannot hold Qwen3-VL-4B.

## Pod state
Nothing running. Last pod `8tnse35hx8rnqe` (RTX PRO 4500, 60 GB container disk, no volume needed) was stopped by Claude 2026-10-06 13:26 UTC after the bundle was
downloaded. Claude can stop pods itself: key from the pod's `/proc/1/environ` (`bash r1_pod_run.sh check_stop` first, `stop <pod_id>` after the bundle is local;
memory `pod-self-stop`). The user's RunPod account has the PC's public key saved (new pods accept it). HF token for pods: `/workspace/.cache/huggingface/token`
if the old volume is attached, else the user pastes it in the pod's web terminal. Container disk must be ≥ 60 GB.

## Important commands (from `MMLM_AI/` unless noted)
```bash
python student_training/scripts/r1_bridge_test.py          # bridge unit tests (CPU, tiny random LM)
python student_training/scripts/r1_build_manifests.py      # rebuild manifests + assert counts (3,299 / 442 / 110)
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
python dataset_sample_review.py --only <dada|tau|caviar|mmau|vru|llava|bddx>   # 3 seeded review samples per dataset
python r0_feature_probe.py --config ../configs/e4_stageA.yaml --lora-adapter ../../outputs/a1_compress256/train/epoch_02/lora_adapter --n-windows 600 --out-dir <dir>
python build_reasoning_path_deck_2026-10.py                # rebuilds the October deck (asserts numbers from score files)
```
Crash-path commands (champion recipe, public scoring, paired bootstrap) are unchanged: see `history/2026-09_PROJECT_STATE.md`.

## Git state
Branch **`reasoning-path-vjepa2-llm`**, HEAD `994897d` (+ this handoff commit); `origin/main` at `3463da6`; **not pushed** (user pushes, with `--tags`).
Tag `phase1-2026-10-06` = `ac40861` (code of the Phase-1 run). `dataset/`, `outputs/`, `reports/` are git-ignored.

## Next step
Wait for the user's decision on the next experiment after they have reviewed the Phase-2 review file. Nothing is running or scheduled.
