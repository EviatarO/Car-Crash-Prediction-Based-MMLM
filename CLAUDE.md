# Project: Car-Crash Collision Anticipation via MLLM Distillation

## Mission (the box you operate in)
MSc thesis. Explainable collision anticipation for dashcam video (Nexar dataset)
via Teacher→Student distillation. **Active architecture (since the 2026-06-23
pivot, `e4_vjepa_reason`): a BADAS-Open (V-JEPA2 ViT-L) student**, LoRA-tuned on
its trunk, with an optional train-only semantic-alignment branch (Predictor +
frozen SigLIP text encoder, supervised by teacher captions) that is fully
discarded at inference — zero added cost. **This is NOT InternVL3.5** — see
"Historical" below. Full design: `docs_agents/ARCHITECTURE.md`.

**Primary success metric: test AP, threshold-free** — trust AP/AUC over
F1/accuracy on these small, imbalanced sets, and say so when reporting. Live
question: does semantic-aux (**B**) beat crash-only LoRA (**A1**) beat the frozen
baseline (**A0**, test AP=0.853/AUC=0.864, 677 clips)?

## Current state (update as it moves)
- Live sub-thread status: `docs_agents/PROJECT_STATE.md`.
- **Historical / superseded**: an earlier architecture used an **InternVL3.5-4B-Flash**
  student (LoRA r=16/α=32 + ScoreHead) and reached test AP=0.762/AUC=0.784 (Epoch 7,
  `EviatarO/e3a-epoch7-lora`; val AP=0.913 on 18 GT clips). Real, shipped result —
  but not the current architecture. Do not describe the active student as InternVL3.5.

## Plan routing
New plans go under `C:\Users\eviatar.ohayon\.claude\plans\`, in the subfolder matching
the plan's **topic**, not the chat it was written in (e.g. BADAS/e4/semantic-supervision
work → `CCP based BADAS\`; classic InternVL3.5 student work → `CCP based MMLM - Student\`).
Title = relevant description + date, one new file per plan (never overwrite).

## Environment rules (RunPod)
- **Always** prefix python with `HF_HOME=/root/.cache/huggingface`. The 20 GB
  `/workspace` volume overflows otherwise (the local disk has ~3.7 TB).
- Checkpoints/weights/datasets too big for git → HF Hub, not `git add`.
- Frames: HiRes = native 1280×720, sequential naming `frame_00001..16.jpg`.

## When reporting metrics
Explain the confusion matrix and why F1/precision/accuracy can disagree with the
ranking metrics at threshold 0.5 — the student's score distribution sits low
(optimal threshold ≈ 0.32 on test), so AP/AUC are the honest headline.
