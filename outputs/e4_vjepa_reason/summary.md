# Experiment e4_vjepa_reason — summary

**Initiative:** Dynamics-grounded, faithful collision reasoning (unified V-JEPA2-native VLM).
Plan: `~/.claude/plans/CCP based BADAS/2026-06-23_Thesis-Upgrade-...md`.

**Thesis framing:** AP target = match BADAS-Open (~0.86, up from our 0.762); headline
contribution = faithfulness (reasoning provably tied to the score), with score↔reason
coupling as the secondary AP-lift result.

## Stages
| Stage | Status | Output |
|---|---|---|
| A — scorer reproduce + freeze | **COMPLETE ✓ PASS** (2026-06-24) | StageA_scorer/ |
| B — V-JEPA2→LLM bridge | **COMPLETE ✓ PASS** (2026-06-24) | e4_StageB_bridge/ |
| C — reasoning SFT (LoRA, Qwen3.5-4B) | **CODE READY, dry-run validated** (awaiting RunPod) | e4_StageC_reason_sft/ |
| D — AP-lift (ensemble + fusion head) | not started | — |
| E — faithfulness | not started | — |
| F — unfreeze ablation | not started | — |

## Stage A — BADAS-Open reproduction (COMPLETE)

### Results (2026-06-24)
| | Private (677) | Public (667) |
|---|---|---|
| **AP** | **0.853** | **0.871** |
| **AUC** | 0.864 | 0.872 |
| F1 @0.5 | 0.794 | 0.783 |
| Recall @0.5 | 0.911 | 0.907 |
| Accuracy @0.5 | 0.764 | 0.748 |
| AP 95% CI | [0.815, 0.889] | [0.836, 0.904] |

**Replication:** Public AP (0.871) inside Private CI [0.815, 0.889] → REPLICATES ✓  
**Per-group AP flat:** 0.5s/1.0s/1.5s all in 0.856–0.890 — no degradation further from event.  
**Acceptance:** PASS (|0.853−0.86|=0.007 ≤ 0.03 Private; |0.871−0.86|=0.011 ≤ 0.03 Public) ✓

**New baseline: ~0.86 AP** (up from InternVL student 0.762). Stage B may proceed.

## Stage B — V-JEPA2→projector→Qwen3-4B bridge (COMPLETE ✓ PASS, 2026-06-24)

Trained ONLY the ResamplerProjector (64 queries, 6.08M params) on cached features; V-JEPA2 + Qwen3-4B frozen.
- **① Δ-PPL (gate):** trained **10.19** vs random **20.19** / text-only **19.99** → **49.5% / 49.0%** gain (≥20% ✓). Best @ ep3, early-stopped.
- **② Ablation:** ΔCE(zero)=**0.685** ✓ (uses visual tokens); ΔCE(shuffle)≈**0.001** (not clip-specific — generic hazard/safe prior).
- **③ Gens:** 44.7 words, 100% parseable, 0% repetition; **clean pos/neg semantic separation** (t=1→"closing rapidly, high collision probability"; t=0→"constant speed, no risk"). Hazard-lexicon gap saturated (both 1.0) → uninformative metric.
- **⑤ Per-TTE (frozen):** AP 0.979/0.942/0.888 (0.5/1.0/1.5s) — monotonic anticipation curve; scorer anchor intact.
- **Tap point:** `temporal_processor` pre-hook (patch grid (2560,1024)); cache+out on `/root` (workspace quota).
- **Carry-forward:** templated/generic reasoning (ΔCE-shuffle≈0) — Stage C LoRA → content-specific; Stage E → faithfulness.

Detail: `e4_StageB_bridge/StageB_summary.md`. Artifacts on pod `/root/e4_stageB/`.

### Key discoveries during Stage A (RunPod)
- `VJEPAModel` is NOT an `nn.Module` — the actual nn.Module is `vjepa.model`.
- `img_size=224` (confirmed from `badas_loader.py`), NOT 256 as the paper implies.
- Preprocessing: squash resize (cv2) + ImageNet norm (confirmed from `preprocessing.py`).
- Scoring: `softmax(logits / 2.0)[1]` — temperature T=2.0 applied before softmax.
- Speed: ~1.8 clips/s on RTX Pro 4500; 677+667 clips in ~13 min total.

### Setup
- Eval: `student_training/scripts/e4_stageA_badas_open_eval.py`
- Config: `student_training/configs/e4_stageA.yaml` (img_size=224 confirmed)
- Frames: `/workspace/data/test_HiRes/` (Private) + `/workspace/data/test_public/` (Public)
- Manifests: `test_manifest_hires.jsonl` (Private) + `/workspace/data/test_manifest_public_hires.jsonl`
- Acceptance: PASS if |AP − 0.86| ≤ 0.03 both splits + Public inside Private CI

### Outputs in `outputs/e4_vjepa_reason/StageA_scorer/`
- `badas_open_private.jsonl` / `badas_open_public.jsonl` — per-clip raw scores (677+667)
- `metrics_private/` / `metrics_public/` — metrics.json + PR/ROC/CM/score-dist PNGs
- `private_vs_public_comparison.json` — bootstrap AP CIs + replication verdict
- `badas_open_StageA_scores.xlsx` — combined 1344-row Excel (1016 green / 328 red @0.5)

_Last updated: 2026-06-24 (Stage A complete, both splits PASS, replicates)._
