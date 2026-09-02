#!/bin/bash
# A1-failure recovery run: 4 arms (a1cont/v10/v12/v12shuf), 1 fold, starting from A1's
# own LoRA weights (epoch_04, val_ap=0.9143 - A1's best, and the checkpoint
# score_arms_on_pool1761.py used to build pool1761_arm_comparison.xlsx).
#
# Head stays FROZEN (no --unfreeze-head): A1 itself trained with a frozen head, and
# SemTest-200-v2 measured head@1e-4 on ~241 examples producing catastrophic overfit
# (80% of val scores pinned past 0.99/0.01, 18/39 hard clips confidently WRONG by
# epoch 10). With 260 examples here that failure mode is near-certain, and unfreezing
# would also change two variables (head + starting weights) at once.
#
# --lr 2e-5 is 5x below A1's own 1e-4: we are refining an already-converged model, not
# training from scratch - full LR would overwrite A1's representation in the first few
# steps. --lr-schedule cosine (not constant) per the v2 lesson: a non-annealing LR kept
# hammering the model to the last step and drove the confident-wrong behavior there.
#
# Runs STRICTLY SEQUENTIALLY (never run BADAS-loading processes concurrently - see
# PROJECT_STATE.md's known gotchas). No --test-manifest here by design - test scoring
# against the 677-clip set is deferred to Stage 2, after these loss curves are reviewed.
set -e
cd /workspace/MMLM_AI/student_training/scripts
export HF_HOME=/root/.cache/huggingface
BASE=/workspace/MMLM_AI/outputs/a1fail321
A1_CKPT=/workspace/semsup/a1_1761/epoch_04/lora_adapter

echo "[wait] for any running semsup_train.py to exit..."
while pgrep -f "semsup_train.py" > /dev/null; do sleep 5; done
echo "[wait] clear, starting a1cont"

# Predictor warm-start (2026-08-29 fix): B-v3 (the 1761-pool precedent this run is
# modeled on) warm-started its predictor from a B1 probe rather than cold-starting -
# the first launch of this driver missed that and ran v10/v12/v12shuf cold (those
# results are preserved as v10_coldstart/v12_coldstart, not discarded). No V10-specific
# B1 checkpoint exists on the pod, so all three semantic arms share this SAME
# V12-trained checkpoint - holding initialization constant and varying only the
# caption file, matching the "identical config except captions" principle used
# throughout this project's arm comparisons (SemTest-200 included). a1cont has no
# predictor (semantic_weight=0) and is unaffected - its earlier result stands as-is.
PREDICTOR_INIT=/workspace/semsup/b1_v2_100pct/predictor_b1.pt

run_arm() {
  arm=$1; captions=$2; semweight=$3; bank=$4
  echo "=== $arm fold_01 starting $(date) ==="
  extra=""
  if [ -n "$bank" ]; then
    extra="--bank-captions $bank --semantic-loss infonce --infonce-tau-init 0.07 --grad-cosine-every 8 --predictor-init $PREDICTOR_INIT"
  fi
  python3 -u semsup_train.py \
    --config ../configs/e4_stageA.yaml --lora-target-modules query,key,value \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --lora-init "$A1_CKPT" \
    --captions-path "$captions" \
    --lr 2e-5 --lr-schedule cosine --warmup-frac 0.1 --epochs 10 --keep-top-k 10 --seed 0 \
    --val-video-ids $BASE/val_vids.txt \
    --semantic-weight $semweight $extra \
    --dump-val-scores --out-dir $BASE/results/$arm/fold_01 \
    > /tmp/a1fail_${arm}_fold01.log 2>&1
  echo "=== $arm fold_01 finished $(date), exit=$? ==="
  df -h /workspace | tail -1
}

run_arm v10 "$BASE/Caption_a1fail321_V10.jsonl" 0.2 \
  /workspace/MMLM_AI/outputs/semantic_captions/Caption_Train4500_Mixed_1761.jsonl

run_arm v12 "$BASE/Caption_a1fail321_V12.jsonl" 0.2 \
  /workspace/MMLM_AI/outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl

run_arm v12shuf "$BASE/Caption_a1fail321_V12_shuffled.jsonl" 0.2 \
  /workspace/MMLM_AI/outputs/semantic_captions/Caption_V12_Neutral_1761_shuffled_fortrain.jsonl

echo "ALL_A1FAIL321_ARMS_DONE $(date)"
