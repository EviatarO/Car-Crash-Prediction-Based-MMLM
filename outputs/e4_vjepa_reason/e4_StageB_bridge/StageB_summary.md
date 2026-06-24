# Stage B — V-JEPA2 → Projector → Qwen3-4B (alignment gate)
**Experiment:** e4_vjepa_reason · **Status:** CODE READY, not yet run (awaiting RunPod)

> Filled in after the RunPod run. Plan: `~/.claude/plans/CCP based BADAS/...e4_StageB_bridge...md`.

## Goal
Train only the projector so frozen Qwen3-4B can read frozen V-JEPA2 features.
Deliverable = proven visual grounding (not good reasoning — that's Stage C). LoRA OFF.

## Setup
- Frozen: V-JEPA2 (BADAS-Open, img224) + attentive probe + BADAS classifier (score path); Qwen3-4B.
- Trainable: ResamplerProjector (64 queries, hidden 512, ~6M params).
- Train data: 267 e3b multi-horizon windows. Held-out: 18 GT-val clips.
- Loss: reasoning CE only (reason span; verdict masked). lr 1e-3, cosine+warmup, early-stop on val PPL.

## Results (FILL AFTER RUN)
### ① Δ-Perplexity (PRIMARY GATE)
| | PPL |
|---|---|
| trained projector | — |
| random projector (floor) | — |
| text-only (vis masked) | — |
| gain vs random | —% (PASS ≥ 20%) |

### ② Visual-ablation sensitivity (GROUNDING GATE)
- ΔCE(zeroed) = — (PASS > 0) · ΔCE(shuffled) = —

### ③ Discrimination
- hazard gap (pos−neg) = — · mean gen len = — · repetition = — · JSON-parseable = —

### ⑤ Per-TTE score-path (frozen scorer characterization)
| TTE | AP | AUC | n |
|---|---|---|---|
| 0.5 | — | — | — |
| 1.0 | — | — | — |
| 1.5 | — | — | — |
- Histogram: `plots/score_hist_by_tte.png`. Regression guard (18-val == Stage A): —

## Gate decision
PASS / FAIL — (Δ-PPL ≥20% AND ΔCE>0 AND hazard gap clear AND AP intact). Proceed to Stage C iff PASS.
