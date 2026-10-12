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

R_NEXR=eviatarO-org/nexar-reasoning-windows
VARIANT=${ARG:-reason_first}; OD=$OUT/phase2r; ED=$OUT/eval_phase2r; HFDIR=phase2r
if [ "$VARIANT" = verdict_first ]; then OD=$OUT/phase2r_vfirst; ED=$OUT/eval_phase2r_vfirst; HFDIR=phase2r_vfirst; fi
TRAIN_R="$PY -u r1_train.py --phase 2 --sources nexar_reasoning --order $VARIANT --init-from $OUT/phase1/best.pt --epochs 8 --patience 2 --stop-on-val-rise 2 --eff-bs 16 --micro-bs 4 --lr-merger 2e-5 --lr-lora 2e-4 --grad-ckpt"
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

# ---------------------------------------------------------------------------------------------------------------------------
# Nexar reasoning dataset (plan 2026-10-12_Plan-Phase2-Reasoning-Retrain). Run 1 = reason_first (this session), Run 2 = verdict_first (later).
#   bash r1_pod_run.sh setup ; get_p1 ; get_old ; smoke_r ; auto_r      (auto_r = features_r pull_r p2r g2r push_r verify_r then stop; pod stays up on any failure)
features_r)   # encode the 4,169 windows of the HF window repo; chunks go to vjepa2-a1-features/nexar_reasoning/. Also checks the encoding against the recorded crash-model P.
  run $PY -u r1_cache_features.py --source nexar_reasoning --from-hf $R_NEXR --splits train,val --push-repo $R_FEAT --out-dir $OUT/feat
  ;;
pull_r)       # merge the chunks into the cache r1_data.Cache reads (expect 3,557 train + 612 val)
  run $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE
  ;;
get_old)      # old Phase-2 checkpoint, only for the same-window comparison
  mkdir -p $OUT/old_phase2
  run $PY -c "from huggingface_hub import hf_hub_download as d; print(d('$R_CKPT', 'phase2/best.pt', local_dir='$OUT/old_phase2'))"
  ;;
smoke_r)      # first window shard -> 50 optimizer steps on the real LM -> 16 evaluated windows. Prints memory + s/step.
  run $PY -u r1_cache_features.py --source nexar_reasoning --from-hf $R_NEXR --splits train --max-shards 1 --out-dir $OUT/feat_smoke || exit 1
  run $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $OUT/feat_smoke/chunks --out-dir $OUT/cache_smoke || exit 1
  run $PY -u r1_train.py --phase 2 --sources nexar_reasoning --order reason_first --cache-dir $OUT/cache_smoke --out-dir $OUT/smoke_train_r --val-split train \
        --init-from $OUT/phase1/best.pt --epochs 1 --eff-bs 16 --micro-bs 4 --max-steps 50 --grad-ckpt --lr-merger 2e-5 --lr-lora 2e-4 || exit 1
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv | tee -a $LOG
  run $PY -u r1_eval_reasoning.py --ckpt $OUT/smoke_train_r/best.pt --cache-dir $OUT/cache_smoke --out-dir $OUT/smoke_eval_r --split train --limit 16 --max-new 120 --n-blank 2
  ;;
p2r)          # ARG = reason_first (Run 1) | verdict_first (Run 2). Selection = val wrong-video gap; stop when the val loss rises 2 epochs in a row.
  run $TRAIN_R --cache-dir $CACHE --out-dir $OD
  ;;
g2r)          # evaluation on all 612 validation windows (+ Phase-1 'before' row and the old Phase-2 model on the same windows for Run 1)
  EXTRA=""; [ "$VARIANT" = reason_first ] && EXTRA="--before-ckpt $OUT/phase1/best.pt --old-ckpt $OUT/old_phase2/phase2/best.pt"
  run $PY -u r1_eval_reasoning.py --ckpt $OD/best.pt --order $VARIANT --cache-dir $CACHE --out-dir $ED $EXTRA
  ;;
push_r)       # checkpoints + train log + evaluation files -> private HF model repo (nothing durable is only on the pod)
  VARIANT=$VARIANT OD=$OD ED=$ED HFDIR=$HFDIR $PY - <<'PY' | tee -a $LOG
import glob, os
from huggingface_hub import HfApi
a = HfApi(); repo = "eviatarO-org/checkpoints"; hd = os.environ["HFDIR"]; od, ed = os.environ["OD"], os.environ["ED"]
a.create_repo(repo, repo_type="model", private=True, exist_ok=True)
files = [(f"{od}/best.pt", f"{hd}/best.pt"), (f"{od}/train_log.jsonl", f"{hd}/train_log.jsonl")] + \
        [(f, f"{hd}/epochs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{od}/epoch_*.pt"))] + \
        [(f, f"{hd}/results/{os.path.basename(f)}") for f in sorted(glob.glob(f"{ed}/*")) if os.path.isfile(f)]
for src, dst in files:
    if os.path.exists(src):
        a.upload_file(path_or_fileobj=src, path_in_repo=dst, repo_id=repo, repo_type="model"); print("pushed", dst)
card = (f"# {hd}/\\n\\nPhase-2 fine training on the Nexar reasoning dataset ({os.environ['VARIANT']}): projector + LoRA r16 on Qwen3-VL-4B, started from phase1/best.pt, "
        "features from vjepa2-a1-features/nexar_reasoning. `best.pt` = {merger, lora}; `epochs/` = every epoch; `results/` = predictions.jsonl (one row per validation window), "
        "metrics.json, summary.md. Data: eviatarO-org/nexar-reasoning-windows. Plan: 2026-10-12_Plan-Phase2-Reasoning-Retrain.\\n")
a.upload_file(path_or_fileobj=card.encode(), path_in_repo=f"{hd}/README.md", repo_id=repo, repo_type="model")
PY
  [ ${PIPESTATUS[0]} -eq 0 ] && touch ${LOG%.log}.done
  ;;
verify_r)     # the same bytes are on HF (best.pt, train_log, predictions, metrics) - only then may the pod stop
  OD=$OD ED=$ED HFDIR=$HFDIR $PY - <<'PY' | tee -a $LOG
import os, sys
from huggingface_hub import HfApi
hd, od, ed = os.environ["HFDIR"], os.environ["OD"], os.environ["ED"]
need = [(f"{od}/best.pt", f"{hd}/best.pt"), (f"{od}/train_log.jsonl", f"{hd}/train_log.jsonl"),
        (f"{ed}/predictions.jsonl", f"{hd}/results/predictions.jsonl"), (f"{ed}/metrics.json", f"{hd}/results/metrics.json")]
remote = {x.path: x.size for x in HfApi().get_paths_info("eviatarO-org/checkpoints", [d for _, d in need], repo_type="model")}
ok = True
for s, d in need:
    print(d, "local", os.path.getsize(s) if os.path.exists(s) else None, "HF", remote.get(d)); ok &= os.path.exists(s) and remote.get(d) == os.path.getsize(s)
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
  [ ${PIPESTATUS[0]} -eq 0 ] && touch ${LOG%.log}.done
  ;;
stop_r)       # only after verify_r
  [ -f $OUT/logs/verify_r${ARG:+_$ARG}.done ] || { echo "results not verified on HF: run verify_r first"; exit 3; }
  export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs)
  [ -n "${RUNPOD_POD_ID:-}" ] || { echo "pod id unknown"; exit 4; }
  echo "stopping pod $RUNPOD_POD_ID at $(date -u)" | tee -a $LOG
  runpodctl stop pod "$RUNPOD_POD_ID" 2>&1 | tee -a $LOG
  ;;
auto_r)       # the whole run, unattended. ARG = reason_first (default) | verdict_first. Run 2 (auto_r verdict_first) needs the cache from Run 1's pod - run features_r/pull_r again if this is a new pod.
  AUTO=$OUT/logs/auto_r${ARG:+_$ARG}.status; echo "start $(date -u)" > $AUTO
  step() { echo "[auto_r] $1 ${2:-} $(date -u)" | tee -a $AUTO; bash $0 $1 ${2:-} >> $OUT/logs/auto_r.out 2>&1 || { echo "[auto_r] FAILED at $1: pod left running" | tee -a $AUTO; exit 1; }; }
  [ -f $CACHE/index_nexar_reasoning.jsonl ] || { step features_r; step pull_r; }
  [ -f $OUT/phase1/best.pt ] || step get_p1
  if [ "$VARIANT" = reason_first ] && [ ! -f $OUT/old_phase2/phase2/best.pt ]; then step get_old; fi
  step p2r $VARIANT; step g2r $VARIANT; step push_r $VARIANT; step verify_r $VARIANT
  echo "[auto_r] verified on HF; stopping the pod $(date -u)" | tee -a $AUTO
  bash $0 stop_r $VARIANT >> $OUT/logs/auto_r.out 2>&1; echo "[auto_r] stop command returned $? $(date -u)" | tee -a $AUTO
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
