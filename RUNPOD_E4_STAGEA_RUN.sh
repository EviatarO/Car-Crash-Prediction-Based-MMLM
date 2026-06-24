#!/bin/bash
# =============================================================================
#  RUNPOD_E4_STAGEA_RUN.sh
#  e4_vjepa_reason — Stage A full run (BADAS-Open scorer reproduction)
#  Paste the CONTENTS of this file into the RunPod terminal (not as a file).
#  Or: chmod +x RUNPOD_E4_STAGEA_RUN.sh && nohup bash RUNPOD_E4_STAGEA_RUN.sh > e4_stageA_master.log 2>&1 &
#
#  Self-terminates the pod when all steps are done (saves credits).
#  Requires: RUNPOD_API_KEY env var set below (or export before running).
# =============================================================================
set -euo pipefail

# ── CONFIG — set your API key here ──────────────────────────────────────────
# Get it from: https://www.runpod.io/console/user/settings (API Keys tab)
export RUNPOD_API_KEY="${RUNPOD_API_KEY:-YOUR_API_KEY_HERE}"
# ─────────────────────────────────────────────────────────────────────────────

export HF_HOME=/root/.cache/huggingface
REPO=/workspace/Car-Crash-Prediction-Based-MMLM
STAGE_DIR=$REPO/outputs/e4_vjepa_reason/StageA_scorer

echo "======================================================================"
echo " e4 Stage A — $(date)"
echo " Pod ID: $RUNPOD_POD_ID"
echo "======================================================================"

# ── STEP 0: verify frames are on the volume ──────────────────────────────────
echo ""
echo "=== Verifying frame directories ==="
N_PRIV=$(ls $REPO/dataset/test/ 2>/dev/null | grep "_hires" | wc -l)
N_PUB=$(ls $REPO/dataset/test_public/ 2>/dev/null | grep "_hires" | wc -l)
echo "  Private _hires dirs : $N_PRIV  (expect 677)"
echo "  Public  _hires dirs : $N_PUB  (expect 667)"

if [ "$N_PRIV" -lt 677 ] || [ "$N_PUB" -lt 667 ]; then
    echo ""
    echo "ERROR: Frame directories missing or incomplete."
    echo "Upload from your LOCAL machine before re-running:"
    echo ""
    echo "  # Private (677 clips):"
    echo "  rsync -avz --progress 'LOCAL_PATH/dataset/test/' \\"
    echo "      root@<POD_IP>:$REPO/dataset/test/ -e 'ssh -p <PORT>'"
    echo ""
    echo "  # Public (667 clips):"
    echo "  rsync -avz --progress 'LOCAL_PATH/dataset/test_public/' \\"
    echo "      root@<POD_IP>:$REPO/dataset/test_public/ -e 'ssh -p <PORT>'"
    exit 1
fi
echo "  Frames OK. Proceeding."

# ── STEP 1: pull latest code ─────────────────────────────────────────────────
cd $REPO
git pull
echo "[$(date +%H:%M:%S)] Code up to date."

# ── STEP 2: install dependencies (badas pulls V-JEPA2 ViT-L weights) ─────────
pip install -q badas openpyxl pyyaml scikit-learn pandas pillow
echo "[$(date +%H:%M:%S)] Dependencies installed."

# ── STEP 3: verify BADAS preprocessing (resolves 224-vs-256 + normalization) ─
echo ""
echo "=== BADAS preprocessing source ==="
python -c "
import badas, inspect, sys
fn = getattr(badas, 'preprocess_video', None) or getattr(badas, 'build_transform', None)
if fn:
    print(inspect.getsource(fn))
else:
    print('WARNING: preprocess_video / build_transform not found in badas module.')
    print('Attributes:', [a for a in dir(badas) if not a.startswith('_')])
"
echo "[$(date +%H:%M:%S)] Preprocessing verified — check output above."

# ── STEP 4: PRIVATE 677 clips ────────────────────────────────────────────────
echo ""
echo "=== Scoring PRIVATE split (677 clips) ==="
python student_training/scripts/e4_stageA_badas_open_eval.py \
    --config      student_training/configs/e4_stageA.yaml \
    --manifest    dataset/manifests/test_manifest_hires.jsonl \
    --frames_root dataset/test \
    --split       Private \
    --output      $STAGE_DIR/badas_open_private.jsonl
echo "[$(date +%H:%M:%S)] Private scoring DONE."

# ── STEP 5: PUBLIC 667 clips ─────────────────────────────────────────────────
echo ""
echo "=== Scoring PUBLIC split (667 clips) ==="
python student_training/scripts/e4_stageA_badas_open_eval.py \
    --config      student_training/configs/e4_stageA.yaml \
    --manifest    dataset/manifests/test_manifest_public_hires.jsonl \
    --frames_root dataset/test_public \
    --split       Public \
    --output      $STAGE_DIR/badas_open_public.jsonl
echo "[$(date +%H:%M:%S)] Public scoring DONE."

# ── STEP 6: evaluate_metrics.py per split ────────────────────────────────────
echo ""
echo "=== Metrics: Private ==="
python student_training/scripts/evaluate_metrics.py \
    --results $STAGE_DIR/badas_open_private.jsonl \
    --out_dir $STAGE_DIR/metrics_private \
    --tag "BADAS-Open Private"

echo "=== Metrics: Public ==="
python student_training/scripts/evaluate_metrics.py \
    --results $STAGE_DIR/badas_open_public.jsonl \
    --out_dir $STAGE_DIR/metrics_public \
    --tag "BADAS-Open Public"

echo "[$(date +%H:%M:%S)] Metrics done."

# ── STEP 7: Private-vs-Public comparison (bootstrap AP CI) ───────────────────
python student_training/scripts/compare_private_public.py \
    --private $STAGE_DIR/badas_open_private.jsonl \
    --public  $STAGE_DIR/badas_open_public.jsonl \
    --out     $STAGE_DIR/private_vs_public_comparison.json
echo "[$(date +%H:%M:%S)] Comparison done."

# ── STEP 8: combined Excel ────────────────────────────────────────────────────
python student_training/scripts/e4_stageA_build_excel.py \
    --private  $STAGE_DIR/badas_open_private.jsonl \
    --public   $STAGE_DIR/badas_open_public.jsonl \
    --out_xlsx $STAGE_DIR/badas_open_StageA_scores.xlsx
echo "[$(date +%H:%M:%S)] Excel written."

# ── DONE ──────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " ALL STEPS COMPLETE — $(date)"
echo " Outputs in: $STAGE_DIR"
echo " Stopping pod $RUNPOD_POD_ID in 60 seconds (Ctrl-C to cancel)..."
echo "======================================================================"
sleep 60

# Stop the pod via RunPod GraphQL API
curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" \
    | python -c "import sys,json; r=json.load(sys.stdin); print('Pod stop response:', r)"
