#!/bin/bash
# run_tte_all_offsets.sh
# Waits for each per-offset zip to land, then runs all 4 inferences.
# Runs all 5 offsets + val + train readback automatically.
# Usage: bash run_tte_all_offsets.sh
set -e
cd /workspace/Car-Crash-Prediction-Based-MMLM
export HF_HOME=/root/.cache/huggingface

OFFSETS="tte30 tte25 tte20 tte15 tte10"
PARTS=outputs/e3b_student_267clips_tte/tte_curve/parts
mkdir -p "$PARTS"
mkdir -p outputs/e3b_student_267clips_tte/metrics/e3b_val_step000099
mkdir -p outputs/e3b_student_267clips_tte/metrics/e3b_train_readback_step000099

E3A_CKPT=outputs/checkpoints/e3a-epoch7-lora
E3B_CKPT=outputs/checkpoints/e3b-ep3-lora/step_000099

# ------------------------------------------------------------------ #
run_inference () {  # $1=tag  $2=ckpt  $3=half  $4=suffix
  OUT=${PARTS}/${1}_tte_curve_${3}_${4}.jsonl
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 142 ]; then
    echo "  [skip] ${1} ${3} ${4} already complete"
    return
  fi
  python student_training/scripts/trained_eval.py \
    --checkpoint "$2" \
    --manifest   dataset/manifests/test_tte_curve_${3}_${4}_manifest.jsonl \
    --frames_root /workspace/data/test_tte_curve/${3} \
    --output      "$OUT" \
    --config      student_training/configs/train_lora.yaml
}

run_offset () {  # $1=suffix (e.g. tte30)
  local S=$1
  ZIP=/workspace/data/test_tte_curve_${S}.zip

  # If frames already extracted, skip zip wait entirely
  PRIV_COUNT=$(ls /workspace/data/test_tte_curve/private/ 2>/dev/null | grep "_${S}$" | wc -l)
  if [ "$PRIV_COUNT" -ge 142 ]; then
    echo "[$S] Frames already on disk — skipping zip wait and unzip."
  else

  # Wait for zip to exist AND stabilize in size (upload complete)
  WAITED=0
  while true; do
    if [ ! -f "$ZIP" ]; then
      echo "  [wait] $ZIP not here yet — sleeping 60s (waited ${WAITED}s)"
      sleep 60; WAITED=$((WAITED + 60))
    else
      SIZE1=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
      sleep 15
      SIZE2=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
      if [ "$SIZE1" -eq "$SIZE2" ] && [ "$SIZE1" -gt 800000000 ]; then
        echo "  [ready] $ZIP stable at $(( SIZE1 / 1024 / 1024 )) MB"
        break
      else
        echo "  [wait] $ZIP still uploading (${SIZE1} -> ${SIZE2} bytes) — sleeping 30s"
        sleep 30; WAITED=$((WAITED + 45))
      fi
    fi
    if [ $WAITED -ge 10800 ]; then echo "ERROR: timed out waiting for $ZIP"; exit 1; fi
  done

  # Validate zip before extracting
    echo "[$S] Validating $ZIP ..."
    if ! unzip -tq "$ZIP" > /dev/null 2>&1; then
      echo "  [error] $ZIP failed validation — deleting and re-waiting for a good copy"
      rm -f "$ZIP"
      # re-enter wait loop by recursing
      run_offset "$S"
      return
    fi
    echo "[$S] Unzipping $ZIP ..."
    unzip -oq "$ZIP" -d /workspace/data/
    echo "[$S] Unzip done."
  fi  # end of zip-wait+unzip block

  echo "[$S] Running 4 inference jobs..."
  for HALF in private public; do
    echo "=== e3a $HALF $S ==="
    run_inference e3a "$E3A_CKPT" "$HALF" "$S"
    echo "=== e3b $HALF $S ==="
    run_inference e3b "$E3B_CKPT" "$HALF" "$S"
  done
  echo "[$S] DONE"
}
# ------------------------------------------------------------------ #

# --- Main loop over all offsets ---
for S in $OFFSETS; do
  run_offset "$S"
done

# --- Val (18 clips, ~1 min) ---
VAL_OUT=outputs/e3b_student_267clips_tte/metrics/e3b_val_step000099/e3b_val_step000099.jsonl
if [ -f "$VAL_OUT" ] && [ "$(wc -l < "$VAL_OUT")" -ge 18 ]; then
  echo "[skip] val already complete"
else
  echo "=== e3b val ==="
  python student_training/scripts/trained_eval.py \
    --checkpoint "$E3B_CKPT" \
    --manifest   dataset/manifests/val_e3a.jsonl \
    --frames_root /workspace/data/train_HiRes \
    --output      "$VAL_OUT" \
    --config      student_training/configs/train_lora.yaml
fi

# --- Train readback (267 clips, ~14 min) ---
TRAIN_OUT=outputs/e3b_student_267clips_tte/metrics/e3b_train_readback_step000099/e3b_train_readback_step000099.jsonl
if [ -f "$TRAIN_OUT" ] && [ "$(wc -l < "$TRAIN_OUT")" -ge 267 ]; then
  echo "[skip] train readback already complete"
else
  echo "=== e3b train readback ==="
  python student_training/scripts/trained_eval.py \
    --checkpoint "$E3B_CKPT" \
    --manifest   dataset/teacher_labels/teacher_dataset_e3b.jsonl \
    --frames_root /workspace/data/train_HiRes \
    --output      "$TRAIN_OUT" \
    --config      student_training/configs/train_lora.yaml
fi

# --- Concatenate parts -> combined curve JSONLs ---
echo "Concatenating parts..."
cd outputs/e3b_student_267clips_tte/tte_curve
cat parts/e3a_tte_curve_private_tte*.jsonl > e3a_tte_curve_private_epoch07.jsonl
cat parts/e3a_tte_curve_public_tte*.jsonl  > e3a_tte_curve_public_epoch07.jsonl
cat parts/e3b_tte_curve_private_tte*.jsonl > e3b_tte_curve_private_step000099.jsonl
cat parts/e3b_tte_curve_public_tte*.jsonl  > e3b_tte_curve_public_step000099.jsonl
echo "Line counts (each should be 710):"
wc -l *.jsonl
cd /workspace/Car-Crash-Prediction-Based-MMLM

echo ""
echo "======================================================"
echo "ALL DONE. Upload results with:"
echo "  HF_HOME=/root/.cache/huggingface python -c \""
echo "from huggingface_hub import HfApi; from pathlib import Path"
echo "api = HfApi()"
echo "files = list(Path('outputs/e3b_student_267clips_tte/tte_curve').glob('*.jsonl')) + \\"
echo "        list(Path('outputs/e3b_student_267clips_tte/metrics/e3b_val_step000099').glob('*.jsonl')) + \\"
echo "        list(Path('outputs/e3b_student_267clips_tte/metrics/e3b_train_readback_step000099').glob('*.jsonl'))"
echo "for f in files:"
echo "    api.upload_file(path_or_fileobj=str(f), path_in_repo=f'results/tte_curve/{f.name}', repo_id='EviatarO/e3b-ep3-lora', repo_type='model')"
echo "    print('Uploaded', f.name)"
echo "\""
echo "======================================================"
