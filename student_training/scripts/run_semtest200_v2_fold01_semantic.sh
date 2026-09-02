#!/bin/bash
# Sequential fold-1 launcher for v10/v12/v12shuf on the pod. Run AFTER vision's
# fold-1 has finished (never run BADAS-loading processes concurrently - see
# PROJECT_STATE.md's known gotchas). Waits for any still-running semsup_train.py
# to exit first, then runs the 3 semantic arms strictly one at a time.
set -e
cd /workspace/MMLM_AI/student_training/scripts
export HF_HOME=/root/.cache/huggingface
BASE=/workspace/MMLM_AI/outputs/semtest200_v2

echo "[wait] for any running semsup_train.py to exit..."
while pgrep -f "semsup_train.py" > /dev/null; do sleep 5; done
echo "[wait] clear, starting v10"

run_arm() {
  arm=$1; captions=$2; bank=$3
  echo "=== $arm fold_01 starting $(date) ==="
  python3 -u semsup_train.py \
    --config ../configs/e4_stageA.yaml --lora-target-modules query,key,value \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --captions-path "$captions" --bank-captions "$bank" \
    --unfreeze-head --head-lr-mult 1.0 --head-lr-schedule constant --clip-grad-per-group \
    --lr 1e-4 --lr-schedule cosine --warmup-frac 0.05 --epochs 10 --keep-top-k 1 --seed 0 \
    --val-video-ids $BASE/folds/fold_01_val_vids.txt \
    --semantic-weight 0.2 --semantic-loss infonce --infonce-tau-init 0.07 \
    --grad-cosine-every 8 \
    --dump-val-scores --out-dir $BASE/results/$arm/fold_01 \
    > /tmp/${arm}_fold01.log 2>&1
  echo "=== $arm fold_01 finished $(date), exit=$? ==="
}

run_arm v10 "$BASE/Caption_semtest200_V10.jsonl" \
  /workspace/MMLM_AI/outputs/semantic_captions/Caption_Train4500_Mixed_1761.jsonl

run_arm v12 "$BASE/Caption_semtest200_V12.jsonl" \
  /workspace/MMLM_AI/outputs/semantic_captions/Caption_V12_Neutral_1761_fortrain.jsonl

run_arm v12shuf "$BASE/Caption_semtest200_V12_shuffled.jsonl" \
  /workspace/MMLM_AI/outputs/semantic_captions/Caption_V12_Neutral_1761_shuffled_fortrain.jsonl

echo "ALL_FOLD1_SEMANTIC_ARMS_DONE $(date)"
