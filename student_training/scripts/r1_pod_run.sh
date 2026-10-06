#!/bin/bash
# Week-1 reasoning-path run on a RunPod pod (plans 2026-10-05_Plan-Week1-GoNoGo-rev3 and 2026-10-05_Plan-HF-Window-Repos-and-Pod).
# Usage:  bash r1_pod_run.sh <stage> [arg]
#   setup | tests | build_dada | features_nexar | features_dada | smoke | pull | p1_ab | p1_full <random|qwen> | g1 | get_p1 | p1_complete | p2 | g2 | push_ckpt | bundle | check_stop | stop <pod_id>
# Each stage logs to $OUT/logs/<stage>.log and writes $OUT/logs/<stage>.done on success (check $?, no pipes).
# Data lives on private HF repos (eviatarO-org/nexar-windows, mmau-dada-windows, vjepa2-a1-features, eviatarO-org/checkpoints);
# the pod holds only working files under /root (container disk). Code is copied to $ROOT (default /root/r1) from the PC.
# NEVER `pip install -U torch`; the venv re-uses the image's torch (--system-site-packages).
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
ROOT=${ROOT:-/root/r1}
OUT=${OUT:-/root/r1_week1}
PY=${PY:-/root/venv/bin/python}
SC=$ROOT/student_training/scripts
CACHE=$OUT/cache
DADA_PARTS=${DADA_PARTS:-/root/dada_parts}
for TOKEN_FILE in ${TOKEN_FILE:-} /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do   # never printed
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
R_NEX=eviatarO-org/nexar-windows; R_DADA=eviatarO-org/mmau-dada-windows; R_FEAT=eviatarO-org/vjepa2-a1-features
R_CKPT=eviatarO-org/checkpoints
mkdir -p $OUT/logs $CACHE
STAGE=${1:?stage}; ARG=${2:-}
LOG=$OUT/logs/$STAGE${ARG:+_$ARG}.log
cd $SC 2>/dev/null || true

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
  python3 -m venv --system-site-packages /root/venv
  /root/venv/bin/pip install -q "transformers==5.8.0" "peft==0.19.1" "huggingface_hub>=1.0,<2" opencv-python-headless albumentations pyyaml \
      pandas accelerate safetensors einops timm scikit-learn sentencepiece protobuf >> $LOG 2>&1
  $PY -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.get_device_name(0))" | tee -a $LOG
  $PY -m pip freeze > $OUT/pod_env.txt
  df -h /root | tee -a $LOG
  ;;
tests)
  run $PY -u r1_bridge_test.py
  ;;
build_dada)   # MM-AU DADA tar parts -> windows -> private HF repo (downloader runs ahead with back-pressure; parts deleted after use)
  nohup $PY -u r1_download_dada_parts.py --dir $DADA_PARTS --max-ahead 3 > $OUT/logs/dada_download.log 2>&1 &
  run $PY -u r1_build_window_repo.py --source dada --repo $R_DADA --parts-dir $DADA_PARTS/DADA-2000_chunks --delete-parts \
        --threads 24 --shard-size 400 --work-dir /root/r1_window_repos/dada
  ;;
features_nexar)   # encode the Nexar window repo shard by shard and push the chunks to the features repo
  run $PY -u r1_cache_features.py --source nexar --from-hf $R_NEX --splits train,val --push-repo $R_FEAT --out-dir $OUT/feat
  ;;
features_dada)
  run $PY -u r1_cache_features.py --source dada --from-hf $R_DADA --splits train,val --push-repo $R_FEAT --out-dir $OUT/feat
  ;;
smoke)   # first Nexar shard (400 windows) -> 50 real-LM optimizer steps -> 8 generations. Prints memory + s/step.
  run $PY -u r1_cache_features.py --source nexar --from-hf $R_NEX --splits train --max-shards 1 --out-dir $OUT/feat_smoke || exit 1
  run $PY -u r1_cache_features.py --source nexar --consolidate-from $OUT/feat_smoke/chunks --out-dir $OUT/cache_smoke || exit 1
  run $PY -u r1_train.py --phase 2 --cache-dir $OUT/cache_smoke --out-dir $OUT/smoke_train --val-split train \
        --epochs 1 --eff-bs 16 --micro-bs 4 --max-steps 50 --grad-ckpt --lr-merger 1e-3 --lr-lora 2e-4 || exit 1
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a $LOG
  run $PY -u r1_eval_gates.py --phase 2 --source nexar --split train --ckpt $OUT/smoke_train/best.pt \
        --cache-dir $OUT/cache_smoke --out-dir $OUT/smoke_gates --n-gen 8
  ;;
pull)    # training pod: merge the feature chunks of the features repo into the caches r1_data.Cache reads
  for S in nexar dada; do
    run $PY -u r1_cache_features.py --source $S --consolidate-from $R_FEAT --out-dir $CACHE
  done
  ;;
p1_ab)    # init A/B: 2 epochs each, random vs Qwen's merger weights; compare the DADA-val wrong-video gap
  for I in random qwen; do
    run $PY -u r1_train.py --phase 1 --cache-dir $CACHE --out-dir $OUT/phase1_ab_$I --init $I \
          --epochs 2 --patience 9 --eff-bs 16 --micro-bs 8 --lr-merger 1e-3 --grad-ckpt
  done
  $PY - <<'PY' | tee -a $LOG
import json, os
out = os.environ.get('OUT', '/root/r1_week1')
for i in ("random", "qwen"):
    rows = [json.loads(l) for l in open(f"{out}/phase1_ab_{i}/train_log.jsonl")]
    r = rows[-1]["val_dada"]; print(i, "val real loss %.3f  wrong-video gap %+.3f %s" % (r["loss_real"], r["gap_wrong"], r["gap_wrong_ci"]))
PY
  ;;
p1_full)  # ARG = winning init (random | qwen)
  run $PY -u r1_train.py --phase 1 --cache-dir $CACHE --out-dir $OUT/phase1 --init ${ARG:-random} \
        --epochs 15 --patience 3 --eff-bs 16 --micro-bs 8 --lr-merger 1e-3 --grad-ckpt
  ;;
g1)       # phase-1 gates: DADA val (valid crash windows only) + zero-shot on Nexar val (V12 description)
  # option b (2026-10-06): Phase 1 trains/selects on DADA crash windows with A1 P(collision) >= 0.5 (r1_train --p1-min-p 0.5 default);
  # the gates run on that group AND, separately, on the windows the encoder did not flag (P < 0.5)
  run $PY -u r1_eval_gates.py --phase 1 --source dada  --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_dada --min-p 0.5
  run $PY -u r1_eval_gates.py --phase 1 --source dada  --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_dada_lowp --max-p 0.5
  run $PY -u r1_eval_gates.py --phase 1 --source nexar --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_nexar_zeroshot
  ;;
get_p1)   # Phase-1 checkpoint from HF (a new pod does not have it)
  mkdir -p $OUT/phase1
  run $PY -c "from huggingface_hub import hf_hub_download as d; print(d('$R_CKPT', 'phase1/best.pt', local_dir='$OUT'))"
  ;;
p1_complete)   # finish the Phase-1 evaluation (2026-10-06): written answers for EVERY validation window of groups A/B/C
               # + what the Phase-1 model writes for no-crash windows (hallucination check). Same checkpoint, no training.
  run $PY -u r1_eval_gates.py --phase 1 --source dada  --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_full_dada --min-p 0.5 --n-gen 0 --n-gen-blank 20
  run $PY -u r1_eval_gates.py --phase 1 --source dada  --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_full_dada_lowp --max-p 0.5 --n-gen 0 --n-gen-blank 20
  run $PY -u r1_eval_gates.py --phase 1 --source nexar --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_full_nexar_zeroshot --n-gen 0 --n-gen-blank 20
  run $PY -u r1_eval_gates.py --phase 1 --source dada  --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_nocrash_dada --nocrash-only
  run $PY -u r1_eval_gates.py --phase 1 --source nexar --ckpt $OUT/phase1/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase1_nocrash_nexar --nocrash-only
  ;;
p2)   # selection = Nexar-val wrong-video gap (pre-registered); also stop when the Nexar-val loss rises 2 epochs in a row
      # (Phase-1 lesson: the gap kept rising while the model over-fitted). Every epoch_XX.pt is kept and pushed to HF.
  run $PY -u r1_train.py --phase 2 --cache-dir $CACHE --out-dir $OUT/phase2 --init-from $OUT/phase1/best.pt \
        --epochs 8 --patience 2 --stop-on-val-rise 2 --eff-bs 16 --micro-bs 4 --lr-merger 2e-5 --lr-lora 2e-4 --grad-ckpt
  ;;
g2)   # all validation windows: verdict probability P(yes) for every window + a written answer for every window
  run $PY -u r1_eval_gates.py --phase 2 --source nexar --ckpt $OUT/phase2/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase2_nexar --n-gen 0 --n-gen-blank 20
  run $PY -u r1_eval_gates.py --phase 2 --source dada  --ckpt $OUT/phase2/best.pt --cache-dir $CACHE --out-dir $OUT/gates_phase2_dada --n-gen 0 --n-gen-blank 20
  ;;
push_ckpt)   # best.pt of both phases -> private HF repo (so nothing durable is only on the pod)
  $PY - <<'PY' | tee -a $LOG
import os
from huggingface_hub import HfApi
a = HfApi(); repo = "eviatarO-org/checkpoints"; out = os.environ.get('OUT', '/root/r1_week1')
a.create_repo(repo, repo_type="model", private=True, exist_ok=True)
if "README.md" not in a.list_repo_files(repo, repo_type="model"):
    card = """---
license: other
---
# checkpoints — reasoning-path training checkpoints (CCP V-JEPA2 Reasoning)

Private. Trained weights of the bridge between the frozen V-JEPA2 encoder (A1-compress256) and the frozen Qwen3-VL-4B language model.
Each `<phase>/best.pt` is a torch dict {"merger": projector state_dict (Qwen3-VL merger shape: LN 1024 -> 2x2 merge 4096 -> 4096 -> GELU -> 2560),
"lora": LoRA weights on the language model (Phase 2 only)}. Load with `R1Bridge.load(path)` (`student_training/models/r1_bridge.py`).
Selection: validation wrong-video gap (not validation loss).

* `phase1/` alignment, merger only, on DADA crash windows the encoder flags (A1 P >= 0.5), target "event; cause".
* `phase2/` SFT, merger + LoRA r16, DADA + Nexar, target "Collision / Time to impact / Event / Cause".

Inputs: [eviatarO-org/vjepa2-a1-features](https://huggingface.co/datasets/eviatarO-org/vjepa2-a1-features).
Code: GitHub EviatarO/Car-Crash-Prediction-Based-MMLM, branch reasoning-path-vjepa2-llm (`r1_train.py`, `r1_eval_gates.py`).
"""
    a.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo, repo_type="model")
import glob
have = set(a.list_repo_files(repo, repo_type="model"))
for ph in ("phase1", "phase2"):
    for f in [f"{out}/{ph}/best.pt", f"{out}/{ph}/train_log.jsonl"] + sorted(glob.glob(f"{out}/{ph}/epoch_*.pt")):
        name = os.path.basename(f)
        dst = f"{ph}/epochs/{name}" if name.startswith("epoch_") else f"{ph}/{name}"
        if os.path.exists(f) and not (ph == "phase1" and dst in have):
            a.upload_file(path_or_fileobj=f, path_in_repo=dst, repo_id=repo, repo_type="model"); print("pushed", dst)
    if os.path.exists(f"{out}/{ph}/best.pt") and f"{ph}/epochs/README.md" not in have and glob.glob(f"{out}/{ph}/epoch_*.pt"):
        a.upload_file(path_or_fileobj=(f"# {ph}/epochs/\n\nThe checkpoint saved after every epoch (`epoch_XX.pt`, same format as "
                                       f"`../best.pt`), kept so any epoch can be re-evaluated. `../train_log.jsonl` has the per-epoch "
                                       f"validation numbers.\n").encode(), path_in_repo=f"{ph}/epochs/README.md", repo_id=repo, repo_type="model")
PY
  [ ${PIPESTATUS[0]} -eq 0 ] && touch ${LOG%.log}.done
  ;;
bundle)   # results bundle defined upfront: logs, jsons, summaries, generations + best.pt of each phase (no caches)
  cd $OUT
  tar czf /root/r1_week1_results.tar.gz --exclude='cache*' --exclude='feat*' --exclude='epoch_*.pt' --exclude='*.npy' \
      logs phase1_ab_* phase1 phase2 gates_* smoke_* pod_env.txt 2>/dev/null
  ls -la /root/r1_week1_results.tar.gz | tee -a $LOG
  echo "NEXT: download /root/r1_week1_results.tar.gz to local BEFORE 'runpodctl stop \$RUNPOD_POD_ID'"
  ;;
check_stop)   # can this pod stop itself? (RunPod injects RUNPOD_API_KEY into the container's start environment; value never printed)
  if tr '\0' '\n' < /proc/1/environ | grep -q '^RUNPOD_API_KEY='; then echo "RUNPOD_API_KEY: present"; else echo "RUNPOD_API_KEY: MISSING"; fi
  command -v runpodctl >/dev/null && echo "runpodctl: $(command -v runpodctl)" || echo "runpodctl: MISSING"
  tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_POD_ID=' | sed 's/^/pod id env: /' || true
  ;;
stop)   # ARG = pod id (from the console, or the prefix of the ssh.runpod.io user name). Only after the bundle was DOWNLOADED.
  [ -f /root/r1_week1_results.tar.gz ] || { echo "no results bundle: run 'bundle' and download it first"; exit 3; }
  export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs)
  POD=${ARG:-${RUNPOD_POD_ID:-}}
  [ -n "$POD" ] || { echo "pod id unknown: pass it as the 2nd argument"; exit 4; }
  echo "stopping pod $POD at $(date -u)" | tee -a $LOG
  runpodctl stop pod "$POD" 2>&1 | tee -a $LOG
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
