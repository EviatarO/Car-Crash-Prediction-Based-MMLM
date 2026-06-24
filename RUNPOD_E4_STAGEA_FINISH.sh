#!/bin/bash
# =============================================================================
#  RUNPOD_E4_STAGEA_FINISH.sh
#  Run in a SECOND terminal while Private scoring runs in the first.
#  Waits for Private to complete, then:
#    1. Runs Public scoring (667 clips)
#    2. evaluate_metrics.py x2
#    3. compare_private_public.py
#    4. e4_stageA_build_excel.py
#    5. Stops the pod (saves credits)
#
#  Usage:
#    export RUNPOD_API_KEY="rpa_XXXX"
#    bash RUNPOD_E4_STAGEA_FINISH.sh
# =============================================================================
set -euo pipefail

export HF_HOME=/root/.cache/huggingface
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-YOUR_API_KEY_HERE}"

REPO=/workspace/Car-Crash-Prediction-Based-MMLM
STAGE_DIR=$REPO/outputs/e4_vjepa_reason/StageA_scorer
PRIVATE_JSONL=$STAGE_DIR/badas_open_private.jsonl
PUBLIC_JSONL=$STAGE_DIR/badas_open_public.jsonl
TOTAL_PRIVATE=677
TOTAL_PUBLIC=667

cd $REPO

# ── STEP 1: wait for Private to finish ───────────────────────────────────────
echo "=== Waiting for Private scoring to complete (target: $TOTAL_PRIVATE clips) ==="
while true; do
    N=$(wc -l < "$PRIVATE_JSONL" 2>/dev/null || echo 0)
    echo "  $(date +%H:%M:%S)  Private: $N/$TOTAL_PRIVATE clips written..."
    if [ "$N" -ge "$TOTAL_PRIVATE" ]; then
        echo "  Private DONE."
        break
    fi
    sleep 30
done

# ── STEP 2: Public 667 clips ─────────────────────────────────────────────────
echo ""
echo "=== Scoring PUBLIC split ($TOTAL_PUBLIC clips) ==="
HF_HOME=/root/.cache/huggingface python student_training/scripts/e4_stageA_badas_open_eval.py \
    --config      student_training/configs/e4_stageA.yaml \
    --manifest    /workspace/data/test_manifest_public_hires.jsonl \
    --frames_root /workspace/data/test_public \
    --split       Public \
    --output      $PUBLIC_JSONL
echo "[$(date +%H:%M:%S)] Public scoring DONE."

# ── STEP 3: metrics per split ─────────────────────────────────────────────────
echo ""
echo "=== evaluate_metrics.py — Private ==="
python student_training/scripts/evaluate_metrics.py \
    --results $PRIVATE_JSONL \
    --out_dir $STAGE_DIR/metrics_private

echo "=== evaluate_metrics.py — Public ==="
python student_training/scripts/evaluate_metrics.py \
    --results $PUBLIC_JSONL \
    --out_dir $STAGE_DIR/metrics_public

# ── STEP 4: Private vs Public comparison ─────────────────────────────────────
echo ""
echo "=== compare_private_public.py ==="
python student_training/scripts/compare_private_public.py \
    --private $PRIVATE_JSONL \
    --public  $PUBLIC_JSONL \
    --out     $STAGE_DIR/private_vs_public_comparison.json

# ── STEP 5: combined Excel ────────────────────────────────────────────────────
echo ""
echo "=== e4_stageA_build_excel.py ==="
python student_training/scripts/e4_stageA_build_excel.py \
    --private  $PRIVATE_JSONL \
    --public   $PUBLIC_JSONL \
    --out_xlsx $STAGE_DIR/badas_open_StageA_scores.xlsx

# ── DONE ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " ALL DONE — $(date)"
echo " Results in: $STAGE_DIR"
ls -lh $STAGE_DIR/
echo ""
echo " Stopping pod $RUNPOD_POD_ID in 60 seconds (Ctrl-C to cancel)..."
echo "======================================================================"
sleep 60

curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" \
    | python -c "import sys,json; r=json.load(sys.stdin); print('Pod stop:', r)"
