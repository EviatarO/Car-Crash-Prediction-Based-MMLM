#!/bin/bash
set -x
cd /workspace/MMLM_AI/student_training/scripts
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HOME=/workspace/.cache/huggingface
RUN_NAME=AA-H-mass-R_pos
RUN_DIR=/workspace/MMLM_AI/outputs/aa_head_attn/$RUN_NAME
mkdir -p $RUN_DIR/logs

echo "=== [$RUN_NAME] training starting at $(date -u) ===" | tee -a $RUN_DIR/logs/driver.log
python3 -u semsup_train.py --config ../configs/e4_stageA.yaml \
  --lora-target-modules query,key,value --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --semantic-weight 0.0 --epochs 8 --lr 2e-4 --lr-schedule constant --grad-accum 8 \
  --val-frac 0.2 --captions-path ../../outputs/semantic_captions/Caption_Train4500_Mixed_1761.jsonl \
  --keep-top-k 8 --split-seed 0 --init-seed 0 --seed 0 --preprocess compress256 \
  --aux-mode attn_mass --aux-label R_pos --aux-weight 0.5518 \
  --aux-schedule warm_on_off --aux-on-epochs 3 --aux-warmup-frac 0.5 \
  --test-manifest ../../dataset/manifests/test_manifest_hires.jsonl --test-frames-root ../../dataset/test \
  --out-dir $RUN_DIR/train \
  > $RUN_DIR/logs/train.log 2>&1
TRAIN_EXIT=$?
echo "=== [$RUN_NAME] training exit=$TRAIN_EXIT at $(date -u) ===" | tee -a $RUN_DIR/logs/driver.log
if [ $TRAIN_EXIT -ne 0 ]; then
  echo "=== [$RUN_NAME] FAILED - skipping public scoring, not stopping pod (needs inspection) ===" | tee -a $RUN_DIR/logs/driver.log
  exit 1
fi

E=$(python3 -c "import json;print('%02d'%json.load(open('$RUN_DIR/train/train_metrics.json'))['best_epoch'])")
echo "=== [$RUN_NAME] best_epoch=$E - scoring public ===" | tee -a $RUN_DIR/logs/driver.log
mkdir -p $RUN_DIR/scores_public
python3 -u score_checkpoints_on_test.py --config ../configs/e4_stageA.yaml \
  --test-manifest /workspace/data/test_manifest_public_hires.jsonl --test-frames-root /workspace/data/test_public \
  --adapters ${RUN_NAME}=$RUN_DIR/train/epoch_$E/lora_adapter \
  --preprocess compress256 --out-dir $RUN_DIR/scores_public \
  > $RUN_DIR/logs/public_score.log 2>&1
echo "=== [$RUN_NAME] public scoring exit=$? at $(date -u) ===" | tee -a $RUN_DIR/logs/driver.log

touch /workspace/MMLM_AI/outputs/aa_head_attn/STAGE2B_MASS_RPOS_DONE.marker
echo "=== [$RUN_NAME] ALL DONE at $(date -u), stopping pod ===" | tee -a $RUN_DIR/logs/driver.log
runpodctl stop pod zimrtinaa091cn >> $RUN_DIR/logs/driver.log 2>&1
