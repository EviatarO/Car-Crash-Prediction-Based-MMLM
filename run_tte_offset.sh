#!/bin/bash
# run_tte_offset.sh  —  run all 4 inferences for ONE TTE offset as its zip lands.
# Usage:  bash run_tte_offset.sh tte30
# Unzips the offset's frames, runs e3a+e3b x private+public (4 x 142 clips ~ 28 min),
# writes per-offset part files. Concatenate at the end (see runbook).
set -e
S=$1
if [ -z "$S" ]; then echo "Usage: bash run_tte_offset.sh tte30"; exit 1; fi
cd /workspace/Car-Crash-Prediction-Based-MMLM
export HF_HOME=/root/.cache/huggingface

ZIP=/workspace/data/test_tte_curve_${S}.zip
PRIV_COUNT=$(ls /workspace/data/test_tte_curve/private/ 2>/dev/null | grep "_${S}$" | wc -l)
if [ "$PRIV_COUNT" -ge 142 ]; then
    echo "Frames already extracted for $S (found $PRIV_COUNT private dirs) — skipping unzip."
else
    if [ ! -f "$ZIP" ]; then echo "ZIP not here yet: $ZIP"; exit 1; fi
    echo "Unzipping $ZIP ..."
    unzip -oq "$ZIP" -d /workspace/data/
fi

PARTS=outputs/e3b_student_267clips_tte/tte_curve/parts
mkdir -p "$PARTS"

run () {  # $1=tag(e3a|e3b) $2=ckpt $3=half
  OUT=${PARTS}/${1}_tte_curve_${3}_${S}.jsonl
  if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -ge 142 ]; then
    echo "  [skip] $OUT already complete ($(wc -l < "$OUT") lines)"
    return
  fi
  python student_training/scripts/trained_eval.py \
    --checkpoint "$2" \
    --manifest dataset/manifests/test_tte_curve_${3}_${S}_manifest.jsonl \
    --frames_root /workspace/data/test_tte_curve/${3} \
    --output "$OUT" \
    --config student_training/configs/train_lora.yaml
}

for HALF in private public; do
  echo "=== e3a $HALF $S ==="
  run e3a outputs/checkpoints/e3a-epoch7-lora           "$HALF"
  echo "=== e3b $HALF $S ==="
  run e3b outputs/checkpoints/e3b-ep3-lora/step_000099  "$HALF"
done

echo "OFFSET $S DONE  ->  $PARTS/{e3a,e3b}_tte_curve_{private,public}_${S}.jsonl"
