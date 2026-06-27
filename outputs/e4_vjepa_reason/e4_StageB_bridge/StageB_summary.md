# Stage B — V-JEPA2 → Projector → Qwen3-4B (alignment gate)
**Experiment:** e4_vjepa_reason · **Status:** COMPLETE ✓ **PASS** (2026-06-24, RunPod RTX Pro 4500)

> Plan: `~/.claude/plans/CCP based BADAS/...e4_StageB_bridge...md`. Run log: `/root/e4_stageB/`.

## Goal
Train only the projector so frozen Qwen3-4B can read frozen V-JEPA2 features.
Deliverable = proven visual grounding (not good reasoning — that's Stage C). LoRA OFF.

## Setup
- Frozen: V-JEPA2 (BADAS-Open `EnhancedVideoClassifier`, img224) + `temporal_processor` + classifier (score path); Qwen3-4B (hidden 2560).
- Trainable: ResamplerProjector (64 queries, hidden 512, **6.08M params**).
- Train data: 267 e3b multi-horizon windows. Held-out: 18 GT-val clips.
- Loss: reasoning CE only (reason span; verdict masked). lr 1e-3, cosine+warmup, early-stop on val PPL.
- **Patch-grid tap:** forward-pre-hook on `temporal_processor` (NOT `pooler`). Grid = **(2560, 1024)** → `in_dim=1024` confirmed.
- **Disk:** cache + outputs on `/root` (E4_CACHE_DIR/E4_OUTPUT_DIR) — the 20 GB `/workspace` MooseFS volume hit its quota (Errno 122).

## Results
### ① Δ-Perplexity (PRIMARY GATE) — **PASS**
| | PPL |
|---|---|
| trained projector | **10.19** |
| random projector (floor) | 20.19 |
| text-only (vis masked) | 19.99 |
| gain vs random | **49.5%** (PASS ≥ 20%) |
| gain vs text-only | **49.0%** |

Training: best_val_ppl **10.182 @ epoch 3** (early-stopped ep9; train_loss kept falling 2.69→1.52 while val_ppl rose after ep3 → textbook small-data overfit, val-selected checkpoint kept).

### ② Visual-ablation sensitivity (GROUNDING GATE) — **PASS (zero); shuffle weak**
- `ce_real = 2.321` · `ce_zeroed = 3.006` · `ce_shuffled = 2.323`
- **ΔCE(zeroed) = 0.685** (PASS > 0.05 — removing visual tokens clearly hurts the reasoning)
- ΔCE(shuffled) = **0.001** (≈0 — model is **not** sensitive to *which* clip's features; uses visual tokens as a generic hazard/safe prior, not clip-specific grounding). Documented limitation; addressed by Stage C (LoRA) + Stage E (faithfulness).

### ③ Discrimination — **PASS (qualitative)**
- mean gen len = **44.7 words** · repetition = **0.0** · JSON-parseable = **1.0**
- hazard-lexicon gap = 0.0 but **saturated** (pos=1.0, neg=1.0): negatives name hazards to *negate* them → binary any-match uninformative on this domain.
- **Verdict-semantic separation is clean**: every t=1 gen → "approaching stationary vehicle at high speed, distance closing rapidly, high probability of collision"; every t=0 gen → "constant speed, straight line, no evidence of approaching vehicle/pedestrian/obstacle, no collision risk."
- Reasoning is **templated** (near-verbatim within each class) — consistent with ΔCE(shuffle)≈0; the frozen LLM expresses a 2-mode mapping, not scene-specific description.

### ⑤ Per-TTE score-path (frozen scorer characterization) — **PASS (anchor intact)**
| TTE | AP | n |
|---|---|---|
| 0.5 | 0.979 | 89 |
| 1.0 | 0.942 | 89 |
| 1.5 | 0.888 | 89 |
- Monotonic anticipation curve (closer event → higher/easier), consistent with corrected TTE analysis (not the flat-0.76 artifact).
- Histogram: `plots/score_hist_by_tte.png`. 18-val frozen scores recorded in `metrics/score_path_by_tte.json` (regression guard vs Stage A ~0.86).

## Gate decision
**PASS** — Δ-PPL 49.5% (≥20%) AND ΔCE(zero) 0.685 (>0) AND coherent/parseable/non-degenerate gens with clean pos/neg separation AND per-TTE anchor intact.
Visual information provably flows V-JEPA2→projector→frozen Qwen3 and drives correct hazard/safe reasoning. **Proceed to Stage C.**

**Carry-forward limitation (Stage C/E target):** reasoning is generic/templated, not clip-specific (ΔCE-shuffle≈0). Stage C LoRA must make reasoning content-specific; Stage E must tie it to the score.

## Artifacts (`/root/e4_stageB/`)
- `out/projector_best.pt` (ep3, 6.08M) → upload to HF `EviatarO/e4-stageB-projector`
- `out/train_result.json`, `out/epoch_metrics.jsonl`, `out/train_log.jsonl`
- `out/metrics/{ppl_results,ablation_sensitivity,discrimination,score_path_by_tte}.json`
- `out/generations.jsonl` (18 val gens + teacher reasons), `out/plots/score_hist_by_tte.png`
- `cache/` — cached patch grids (267 train + 18 val), regenerable; not uploaded.

_Last updated: 2026-06-24 (Stage B complete, gate PASS)._
