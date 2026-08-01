#!/bin/bash
# =============================================================================
#  RUNPOD_TRAIN4500_STAGEA.sh
#  train4500-inference plan (~/.claude/plans/but-if-b-a1-it-woolly-metcalfe.md)
#  Paste the CONTENTS of this file into the RunPod terminal (not as a file).
#
#  UNLIKE RUNPOD_E4_STAGEA_RUN.sh, this does NOT self-terminate the pod: the
#  workflow is chunked (~500 videos/chunk, driven from the LOCAL machine by
#  run_train4500_pipeline.py) and is meant to be re-run as each new chunk
#  arrives via rsync, not run once end-to-end. Stop the pod manually (see the
#  commented block at the bottom) once you're done with a session.
#
#  Prerequisite (done from your LOCAL machine, once per chunk):
#    1. python student_training/scripts/run_train4500_pipeline.py \
#           --chunk-size 500 --stop-after-chunk N --workers 8
#       (extracts chunk(s) 0..N locally, writes
#        dataset/manifests/train4500_chunks/chunk_00.jsonl .. chunk_0N.jsonl,
#        and prints the exact rsync commands - copy them from its output,
#        they include your pod's actual IP/port which this script can't know)
#    2. rsync -avz --progress 'dataset/train/' root@<POD_IP>:$REPO/dataset/train/ -e 'ssh -p <PORT>'
#    3. rsync -avz --progress 'dataset/manifests/train4500_chunks/' \
#           root@<POD_IP>:$REPO/dataset/manifests/train4500_chunks/ -e 'ssh -p <PORT>'
# =============================================================================
set -euo pipefail

export HF_HOME=/root/.cache/huggingface
REPO=/workspace/MMLM_AI
STAGE_DIR=$REPO/outputs/train4500_inference
CHUNK_DIR=$REPO/dataset/manifests/train4500_chunks

mkdir -p "$STAGE_DIR"

echo "======================================================================"
echo " train4500 Stage A (inference-only, no training) — $(date)"
echo " Pod ID: ${RUNPOD_POD_ID:-unknown}"
echo "======================================================================"

# ── STEP 0: verify code + deps ────────────────────────────────────────────
cd "$REPO"
git pull
pip install -q badas openpyxl pyyaml scikit-learn pandas pillow matplotlib seaborn
echo "[$(date +%H:%M:%S)] Code + deps up to date."

# ── STEP 1: score every chunk manifest found that isn't already scored ────
N_SCORED=0
N_SKIPPED=0
for chunk_manifest in "$CHUNK_DIR"/chunk_*.jsonl; do
    [ -e "$chunk_manifest" ] || continue
    chunk_name=$(basename "$chunk_manifest" .jsonl)          # e.g. chunk_00
    out_file="$STAGE_DIR/scores_${chunk_name#chunk_}.jsonl"   # e.g. scores_00.jsonl
    n_rows=$(wc -l < "$chunk_manifest")

    if [ -f "$out_file" ] && [ "$(wc -l < "$out_file")" -eq "$n_rows" ]; then
        echo "  [SKIP] $chunk_name already scored ($n_rows/$n_rows rows) -> $out_file"
        N_SKIPPED=$((N_SKIPPED+1))
        continue
    fi

    echo ""
    echo "=== Scoring $chunk_name ($n_rows rows) ==="
    python student_training/scripts/e4_stageA_badas_open_eval.py \
        --config      student_training/configs/e4_stageA.yaml \
        --manifest    "$chunk_manifest" \
        --frames_root dataset/train \
        --split       Train \
        --output      "$out_file"
    echo "[$(date +%H:%M:%S)] $chunk_name DONE."
    N_SCORED=$((N_SCORED+1))
done
echo ""
echo "Chunks scored this run: $N_SCORED  |  already-done skipped: $N_SKIPPED"

if [ "$N_SCORED" -eq 0 ] && [ "$N_SKIPPED" -eq 0 ]; then
    echo "No chunk manifests found in $CHUNK_DIR - nothing to do. Push a chunk from local first."
    exit 0
fi

# ── STEP 2: concatenate all scored chunks + run metrics/mining/monitor ────
echo ""
echo "=== Aggregating all scores_*.jsonl -> scores_all.jsonl ==="
cat "$STAGE_DIR"/scores_*.jsonl > "$STAGE_DIR/scores_all.jsonl"
N_TOTAL=$(wc -l < "$STAGE_DIR/scores_all.jsonl")
echo "  $N_TOTAL total scored rows so far."

echo ""
echo "=== evaluate_metrics.py ==="
python student_training/scripts/evaluate_metrics.py \
    --results "$STAGE_DIR/scores_all.jsonl" \
    --out_dir "$STAGE_DIR/metrics" \
    --tag "train4500 (partial - $N_TOTAL/4446 scored)"

echo ""
echo "=== mine_train_failures.py ==="
python student_training/scripts/mine_train_failures.py \
    --scores "$STAGE_DIR/scores_all.jsonl" \
    --manifest dataset/manifests/train4500_hires.jsonl \
    --out-dir "$STAGE_DIR"

echo ""
echo "=== build_caption_monitor.py ==="
python teacher_distillation/scripts/build_caption_monitor.py \
    --scores "$STAGE_DIR/scores_all.jsonl" \
    --out "$STAGE_DIR/monitor_train4500_coverage.xlsx"

echo ""
echo "======================================================================"
echo " DONE — $(date). $N_TOTAL/4446 windows scored so far."
echo " Outputs in: $STAGE_DIR"
echo " Review the CHECKPOINT line printed by mine_train_failures.py above:"
echo "   compare this run's error_rate against A0's known test error_rate"
echo "   (23.6%, TP/FP/TN/FN=308/130/209/30) - a gap > 5pp means stop and"
echo "   diagnose (manifest/preprocessing bug), not a real finding."
echo "======================================================================"

# ── Manual pod stop (uncomment when you're done for the session) ──────────
# export RUNPOD_API_KEY="YOUR_API_KEY_HERE"
# curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
#     -H "Content-Type: application/json" \
#     -d "{\"query\": \"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" \
#     | python -c "import sys,json; r=json.load(sys.stdin); print('Pod stop response:', r)"
