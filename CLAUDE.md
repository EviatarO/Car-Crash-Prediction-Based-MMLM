# Project: Car-Crash Collision Anticipation via MLLM Distillation

## Mission (the box you operate in)
MSc thesis. Teacher→Student knowledge distillation for dashcam collision
anticipation on the Nexar dataset. A strong VLM teacher produces reasoned labels;
a compact **InternVL3.5-4B-Flash student** (LoRA r=16/α=32 + ScoreHead) is trained
to predict P(collision) ∈ [0,1] *before* the event.

**Primary success metric: test AP > 0.67** (threshold-free; trust AP/AUC over
F1/accuracy on these small, imbalanced sets — and say so when reporting).

## Current state (update as it moves)
- Best checkpoint: **Epoch 7** (`step_000077`) — backed up to HF Hub
  `EviatarO/e3a-epoch7-lora`.
- Test result (677 clips): **AP=0.762, AUC=0.784** — target met.
- Val result (18 GT clips): AP=0.913.

## This chat = "CCP based MMLM - Student"
New plans → `C:\Users\eviatar.ohayon\.claude\plans\CCP based MMLM - Student\`

## Environment rules (RunPod)
- **Always** prefix python with `HF_HOME=/root/.cache/huggingface`. The 20 GB
  `/workspace` volume overflows otherwise (the local disk has ~3.7 TB).
- Checkpoints/weights/datasets too big for git → HF Hub, not `git add`.
- Frames: HiRes = native 1280×720, sequential naming `frame_00001..16.jpg`.

## When reporting metrics
Explain the confusion matrix and why F1/precision/accuracy can disagree with the
ranking metrics at threshold 0.5 — the student's score distribution sits low
(optimal threshold ≈ 0.32 on test), so AP/AUC are the honest headline.
