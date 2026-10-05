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

## Status as of 2026-10-05

### Crash-score path (frozen — do not reopen without the user)
- Champion **A1-compress256**: test AP **0.9128** (private 677) / **0.9096** (public 667). Recipe: LoRA r16/α32/dropout 0.05 on
  `query,key,value`, crash CE only, lr 2e-4 constant, grad_accum 8, 1,761-window pool, head frozen, `--preprocess compress256`
  (pass it explicitly; code default is still `crop`). Adapter: `outputs/a1_compress256/train/epoch_02/lora_adapter` (local + volume).
- Midpoint-negatives recipe raises pooled AP (+0.0092 ± 0.0052, 5/5 seeds) but Kaggle mAP gain not confirmed. Never report the single best checkpoint.
- Versus published BADAS-Open (crop) 0.853 / 0.871: correct-at-0.5 private TTE0.5/1.0/1.5/neg = 135/104/69/209 → 129/101/56/284;
  most of the AP gain comes from preprocessing (A0-compress256, untrained, 0.907 / 0.904).

### Reasoning path — week 1 (plan approved, code done, nothing run on a pod)
- **Approved plans (topic folder `~/.claude/plans/CCP based BADAS/`):** `2026-10-05_Plan-Week1-GoNoGo-rev3.md` (training design,
  gates, schedule) and `2026-10-05_Plan-HF-Window-Repos-and-Pod.md` (HF data repos + pod sessions; latest). Earlier context:
  `2026-10-03_Plan-Reasoning-Path-rev3.md`, `2026-10-04_Plan-Reasoning-Path-Deck.md`, `2026-10-04_Plan-Dataset-Status-rev4-TAU-VRU.md`.
- **Built and tested locally** (see ARCHITECTURE.md §2): `models/r1_bridge.py` (6 unit tests incl. logit-equivalence with the official
  Qwen3-VL forward), manifests (`dataset/manifests/r1_*_windows.jsonl`, counts asserted), feature cache (Nexar verified against
  recorded A1 scores; DADA from a local part), Phase 1/2 trainer and gates (smoke runs on a tiny random LM, CPU),
  pod driver `r1_pod_run.sh` + read-only `r1_pod_scan.sh`, runbook `outputs/r1_week1/RUNBOOK_pod.md`, status `outputs/r1_week1/summary.md`.
- **Not yet verified:** real Qwen3-VL-4B load/memory/speed, gradient checkpointing with LoRA, full DADA download + encode, any training number.
- **Data decisions:** coarse alignment (Phase 1) on MM-AU **DADA** part only (30 fps known); Phase 2 SFT on DADA + Nexar V12 pool;
  CAViAR dropped; window visibility rule applies everywhere (DECISIONS.md).
- **Supporting results:** feature probe (`outputs/r0_feature_probe/n600/`), dataset sample review (`outputs/dataset_review_2026-10/`),
  plan deck `reports/presentations/2026-10_reasoning-path-plan.pptx` (title + 5 slides; built by `build_reasoning_path_deck_2026-10.py`).

### Access requests
Sent by the user 2026-10-04: TAU-106K (train split), VRU-Accident (source-id mapping + collision times).
Not sent: DRAMA form, WTS form, RoadSafe365 email, MM-AU authors email (lotvsmmau@gmail.com; draft in §7 of
`2026-10-05_Plan-Week1-GoNoGo-rev3.md`; asks for academic-use confirmation + per-video fps of CAP-DATA). BDD100K portal showed a certificate error 2026-10-04. Drafts:
`outputs/dataset_review_2026-10/access_requests.md`. Nexar HF repo `nexar-ai/nexar_collision_prediction` is gated and the local HF
account is not authorised (full clips unavailable; 16-frame windows exist locally).

## Open TODOs (in order)
1. **User opens the data-prep pod** (recommended A40 48 GB, $0.49/h, container disk ≥ 150 GB, volume `0hnvco2s4j` attached) and sends
   `ssh root@<host> -p <port>`. Claude cannot start pods (no RunPod API key locally).
2. On the pod: `bash r1_pod_scan.sh` (read-only) → show the clean-up candidate list → delete **only user-approved items**; back up
   volume-only artifacts (old e2/e3a/e3b adapters) first.
3. Check HF write scope (local token is fine-grained; known only on first `create_repo`).
4. **Write `student_training/scripts/r1_build_window_repo.py`** (not written yet): manifest → 16 frames at 256×256 via the processor's own
   resize → WebDataset shards `<window_id>.npz` uint8 (16,256,256,3) + `windows.jsonl`/`.csv` + dataset card → `HfApi.upload_large_folder`.
   Private repos: `EviatarO/r1-nexar-windows` (1,457 windows, built on the PC), `EviatarO/r1-mmau-dada-windows` (4,582 windows, built on the pod),
   `EviatarO/r1-vjepa-a1-features` (derived tokens). Exactness check: P(collision) from stored frames vs originals within 0.004 on 20 windows.
5. Change `r1_cache_features.py` to read the HF window repos (`--from-hf`); add `scan/build_dada/features/push/pull` stages to `r1_pod_run.sh`.
6. Pod `smoke` stage (real LM, 50 steps) → stop and report memory / s/step / loss.
7. Training session (L40S, or A100 SXM if unavailable): `p1_ab → p1_full → g1 → p2 → g2 → bundle`; push `best.pt` to a private
   `r1-checkpoints` repo; download the bundle **before** stopping the pod.
8. **Next session, once Phase-1 results exist:** measure the hazard hallucination rate on no-crash validation windows (DADA val + 178 Nexar val
   no-crash); if high → re-weight Phase 2 toward no-crash or add no-crash windows to Phase 1 at ~20%.
9. User pushes branch `reasoning-path-vjepa2-llm`; Claude never pushes.

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
Nothing running (both saved hosts refused connections 2026-10-05). Stopped pods on record: `egx54zfwpmpasg`, `maeqpipl372s77`.

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
bash r1_pod_run.sh <setup|tests|smoke|cache_nexar|cache_dada|p1_ab|p1_full random|g1|p2|g2|bundle>   # on the pod
python dataset_sample_review.py --only <dada|tau|caviar|mmau|vru|llava|bddx>   # 3 seeded review samples per dataset
python r0_feature_probe.py --config ../configs/e4_stageA.yaml --lora-adapter ../../outputs/a1_compress256/train/epoch_02/lora_adapter --n-windows 600 --out-dir <dir>
python build_reasoning_path_deck_2026-10.py                # rebuilds the October deck (asserts numbers from score files)
```
Crash-path commands (champion recipe, public scoring, paired bootstrap) are unchanged: see `history/2026-09_PROJECT_STATE.md`.

## Git state
Branch **`reasoning-path-vjepa2-llm`** (created from `main` 2026-10-05), HEAD `6583923` = README + r1 pipeline; `origin/main` is at
`3463da6`, so the branch is 1 commit ahead and **not pushed** (user pushes). Uncommitted: `r1_pod_run.sh` (storage moved to `/root`),
`dataset_sample_review.py` (TAU source), new `r1_pod_scan.sh`, and this handoff's docs. `dataset/`, `outputs/`, `reports/` are git-ignored.

## Next step
Wait for the user to open the A40 pod and send the SSH command; then follow `2026-10-05_Plan-HF-Window-Repos-and-Pod.md` §3
(scan → user-approved clean-up → HF write check → write/run `r1_build_window_repo.py` → features → smoke) and stop for review after the smoke stage.
