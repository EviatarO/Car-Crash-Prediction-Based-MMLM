#!/bin/bash
# Week-1 reasoning-path run on a RunPod pod (plan 2026-10-05_Plan-Week1-GoNoGo-rev3).
# Usage:  bash r1_pod_run.sh <stage> [arg]
#   setup | tests | smoke | cache_nexar | cache_dada | p1_ab | p1_full <random|qwen> | g1 | p2 | g2 | bundle
# Each stage logs to $OUT/logs/<stage>.log and writes $OUT/logs/<stage>.done on success (check $?, no pipes).
# Pod: one 48 GB card (L40S / A6000 class), same datacenter as the network volume. NEVER `pip install -U torch`.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 HF_HOME=/root/.cache/huggingface
ROOT=/workspace/MMLM_AI
OUT=$ROOT/outputs/r1_week1
SC=$ROOT/student_training/scripts
CACHE=$OUT/cache
DADA_PARTS=/workspace/dada_parts
mkdir -p $OUT/logs $CACHE
cd $SC
STAGE=${1:?stage}; ARG=${2:-}
LOG=$OUT/logs/$STAGE${ARG:+_$ARG}.log

run() {   # run <cmd...>  -> logs, records exit, marks done
  echo "=== [$STAGE $ARG] start $(date -u) ===" | tee -a $LOG
  "$@" >> $LOG 2>&1
  EX=$?
  echo "=== [$STAGE $ARG] exit=$EX $(date -u) ===" | tee -a $LOG
  [ $EX -eq 0 ] && touch ${LOG%.log}.done
  return $EX
}

case $STAGE in
setup)
  pip install --break-system-packages huggingface_hub transformers peft safetensors scikit-learn pandas \
      pyyaml pillow openpyxl opencv-python-headless einops timm sentencepiece protobuf >> $LOG 2>&1
  mkdir -p /root/.cache/huggingface && cp /workspace/.cache/huggingface/token /root/.cache/huggingface/token
  pip freeze > $OUT/pod_env.txt
  python3 -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))" | tee -a $LOG
  df -h /workspace | tee -a $LOG
  # data checks: Nexar window frames and the two manifests must exist on the volume
  ls $ROOT/dataset/train | wc -l | tee -a $LOG        # expect >= 4446 window folders
  wc -l $ROOT/dataset/manifests/r1_*_windows.jsonl | tee -a $LOG
  ls $ROOT/outputs/a1_compress256/train/epoch_02/lora_adapter | tee -a $LOG
  ;;
tests)
  run python3 -u r1_bridge_test.py
  ;;
smoke)   # 64 Nexar windows -> 50 real-LM optimizer steps -> gates on 8 generations. Prints memory + s/step.
  run python3 -u r1_cache_features.py --source nexar --splits train,val --limit 64 --out-dir $OUT/cache_smoke || exit 1
  run python3 -u r1_train.py --phase 2 --cache-dir $OUT/cache_smoke --out-dir $OUT/smoke_train --val-split train \
        --epochs 1 --eff-bs 16 --micro-bs 4 --max-steps 50 --grad-ckpt --lr-merger 1e-3 --lr-lora 2e-4 || exit 1
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a $LOG
  run python3 -u r1_eval_gates.py --phase 2 --source nexar --split train --ckpt $OUT/smoke_train/best.pt \
        --cache-dir $OUT/cache_smoke --out-dir $OUT/smoke_gates --n-gen 8
  ;;
cache_nexar)
  run python3 -u r1_cache_features.py --source nexar --splits train,val --out-dir $CACHE
  ;;
cache_dada)   # downloader runs ahead in the background (back-pressure), the cache script consumes + deletes parts
  nohup python3 -u r1_download_dada_parts.py --dir $DADA_PARTS --max-ahead 6 > $OUT/logs/dada_download.log 2>&1 &
  run python3 -u r1_cache_features.py --source dada --splits train,val --parts-dir $DADA_PARTS/DADA-2000_chunks \
        --n-parts 58 --delete-parts --out-dir $CACHE
  ;;
p1_ab)    # init A/B: 2 epochs each, random vs Qwen's merger weights; compare the DADA-val wrong-video gap
  for I in random qwen; do
    run python3 -u r1_train.py --phase 1 --cache-dir $CACHE --out-dir $OUT/phase1_ab_$I --init $I \
          --epochs 2 --patience 9 --eff-bs 16 --micro-bs 8 --lr-merger 1e-3 --grad-ckpt
  done
  python3 - <<'PY' | tee -a $LOG
import json
for i in ("random","qwen"):
    rows=[json.loads(l) for l in open(f"/workspace/MMLM_AI/outputs/r1_week1/phase1_ab_{i}/train_log.jsonl")]
    r=rows[-1]["val_dada"]; print(i, "val real loss %.3f  wrong-video gap %+.3f %s"%(r["loss_real"],r["gap_wrong"],r["gap_wrong_ci"]))
PY
  ;;
p1_full)  # ARG = winning init (random | qwen)
  run python3 -u r1_train.py --phase 1 --cache-dir $CACHE --out-dir $OUT/phase1 --init ${ARG:-random} \
        --epochs 15 --patience 3 --eff-bs 16 --micro-bs 8 --lr-merger 1e-3 --grad-ckpt
  ;;
g1)       # phase-1 gates: DADA val (valid crash windows only) + zero-shot on Nexar val (V12 description)
  run python3 -u r1_eval_gates.py --phase 1 --source dada  --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_dada
  run python3 -u r1_eval_gates.py --phase 1 --source nexar --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_nexar_zeroshot
  ;;
p2)
  run python3 -u r1_train.py --phase 2 --cache-dir $CACHE --out-dir $OUT/phase2 --init-from $OUT/phase1/best.pt \
        --epochs 8 --patience 2 --eff-bs 16 --micro-bs 4 --lr-merger 2e-5 --lr-lora 2e-4 --grad-ckpt
  ;;
g2)
  run python3 -u r1_eval_gates.py --phase 2 --source nexar --ckpt $OUT/phase2/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase2_nexar
  run python3 -u r1_eval_gates.py --phase 2 --source dada  --ckpt $OUT/phase2/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase2_dada
  ;;
bundle)   # results bundle defined upfront: logs, jsons, summaries, generations + best.pt of each phase (no caches)
  cd $OUT
  tar czf /workspace/r1_week1_results.tar.gz --exclude='cache*' --exclude='epoch_*.pt' --exclude='*.npy' \
      logs phase1_ab_* phase1 phase2 gates_* smoke_* pod_env.txt 2>/dev/null
  ls -la /workspace/r1_week1_results.tar.gz | tee -a $LOG
  echo "NEXT: download /workspace/r1_week1_results.tar.gz to local BEFORE 'runpodctl stop \$RUNPOD_POD_ID'"
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
