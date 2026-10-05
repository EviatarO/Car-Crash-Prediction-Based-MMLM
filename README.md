# MMLM_AI — Explaining collision alerts from V-JEPA2 features

MSc thesis code. Task: anticipate collisions from dashcam video (Nexar dataset), and **explain each alert in
text from the same video features that produced the score**.

## Thesis question (changed, October 2026)

| | Earlier question (Jun–Sep 2026) | Current question (this branch) |
|---|---|---|
| Goal | Can training-time language supervision (teacher captions) raise crash-prediction AP? | Can a language model explain a crash alert **from the frozen encoder's own features**, so explanation and score share one vision pass? |
| Outcome | Crash path reached AP **0.913 (private) / 0.910 (public)** (BADAS-Open + LoRA, full-frame input, `A1-compress256`). Every semantic-supervision arm lost or tied. | Crash path is **frozen**. New work = the reasoning path below. Nexar names this ("latent-based reasoning") as future work in the BADAS-2.0 paper. |

## Architecture of the new work (the "reasoning path")

```
16 frames (2 s) ─► compress256 (full frame → 256×256) ─► frozen V-JEPA2 ViT-L + A1 LoRA ─► 2,048 tokens × 1024
                                                              │
        prediction path (unchanged) ◄─────────────────────────┤ → attentive probe → classifier → P(collision)
                                                              ▼
              2×2 regroup → Merger (Qwen3-VL projector structure, TRAINED) → 512 tokens × 2560
                                                              ▼
              Qwen3-VL-4B language model (native video tokens, 3D M-RoPE)  frozen → LoRA in phase 2 → text
```

* **Phase 1 (alignment):** only the projector trains; data = MM-AU DADA crash windows (event + cause labels).
* **Phase 2 (SFT):** projector + LoRA on the language model; data = DADA + Nexar windows, 50/50 crash / no-crash.
* **Window visibility rule:** a crash window is used only if the hazard has already started inside it
  (DADA: window end ≥ `t_ai` + 8 frames; Nexar: window end ≥ `time_of_alert`).
* **Success is measured by held-out tests, not by loss:** wrong-video gap, blank-video gap, retrieval@1-of-10,
  output diversity, fact accuracy vs ground truth (verdict, time to impact, event, cause, ArA multiple choice).

**Status:** code written and unit-tested locally; nothing trained yet. See `outputs/r1_week1/RUNBOOK_pod.md`
(git-ignored folder) and `docs_agents/PROJECT_STATE.md`.

## Folder map (in the order you would read it)

| Path | What it is |
|---|---|
| `docs_agents/` | Cold-start briefing: `PROJECT_STATE.md` (read first), `ARCHITECTURE.md`, `EXPERIMENTS.md` (results ledger), `DECISIONS.md` (rejected options), `CODE_GUIDE.md`. Older months in `docs_agents/history/`. |
| `student_training/models/` | Models. **`r1_bridge.py`** = the reasoning-path bridge (merger + Qwen3-VL prompt + loss + generation). `vjepa_reason.py` = the June e4 attempt (superseded). `lookahead.py` = closed experiment. |
| `student_training/scripts/r1_*` | Week-1 reasoning-path pipeline, in run order: `r1_common.py` (window + visibility rules, target texts) → `r1_build_manifests.py` → `r1_download_dada_parts.py` → `r1_cache_features.py` → `r1_train.py` (+ `r1_data.py`) → `r1_eval_gates.py`; `r1_bridge_test.py` (unit tests); `r1_pod_run.sh` (RunPod driver). |
| `student_training/scripts/r0_feature_probe.py` | Probe: which facts (position, agent side, gap trend) the frozen tokens hold. |
| `student_training/scripts/dataset_sample_review.py` | Pulls 3 seeded (clip, text) samples per candidate dataset for manual review. |
| `student_training/scripts/build_reasoning_path_deck_2026-10.py` | Builds the October plan deck (`reports/presentations/`). |
| `student_training/scripts/semsup_*.py`, `score_*.py`, `aa*.py`, `e4_*.py` | Crash-path training / scoring (champion A1-compress256), the closed semantic-supervision and attention-supervision experiments, and the e4 reasoning stages from June. |
| `student_training/configs/` | YAML configs (`e4_stageA.yaml` is used to load BADAS-Open). |
| `teacher_distillation/`, `prompts/` | Earlier teacher pipeline and the teacher prompt versions (V12 = the neutral window caption prompt used for Nexar text). |
| `website/` | Local results website (see `website/README.md`). |
| `RunPod/`, `RUNPOD_*.sh` | Notes and scripts from earlier pod sessions. |
| `third_party/` | Reference code (YOLOPv2). |
| `dataset/`, `outputs/`, `reports/` | **Git-ignored** (large / generated). `dataset/manifests/r1_*` = the week-1 manifests, `dataset/public_samples/` = downloaded public data, `outputs/dataset_review_2026-10/` = dataset sample review, `outputs/r1_week1/` = runbook and results, `reports/presentations/` = decks. |

## Datasets in use

* **Nexar** collision prediction (train clips only; test sets untouched).
* **MM-AU / DADA-2000 part** (HF `JeffreyChou/MM-AU`, CC BY-NC 4.0, academic use): crash windows with timing + human labels; official ArA train/val/test split.
* Candidates under review / requested: BDD-X, TAU-106K, VRU-Accident, DRAMA, WTS. CAViAR was dropped (templated text).

## Reproducing the local checks

```bash
python student_training/scripts/r1_bridge_test.py        # bridge unit tests (CPU, tiny random LM)
python student_training/scripts/r1_build_manifests.py    # builds + asserts the window counts
```
Training and feature caching run on a RunPod GPU (`outputs/r1_week1/RUNBOOK_pod.md`).
