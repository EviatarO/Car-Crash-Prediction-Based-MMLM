#!/bin/bash
# Step B of the full teacher run (plan 2026-10-10_Plan-Teacher-Full-Run.md): tracks + A1 attention for 4,203 windows on a RunPod pod.
# The pod does the GPU work only: it writes tracks/<vid>.json and attn/<frames_dir>.json (small). The PC draws the boxes and runs the teacher.
# Usage:  bash r1_teacher_pod.sh <stage> [arg]
#   setup | hf_check | smoke | batches [videos_per_batch] | videos | cut | tracks_attn [n_procs] | push | verify | check_stop | stop <pod_id> | auto
# Layout on the pod: code /root/r1, work dir /root/teacher_dataset (windows.jsonl in, tracks/ attn/ out), videos /root/nexar/train, frames /root/frames.
# Videos: Hugging Face dataset nexar-ai/nexar_collision_prediction (gated 'auto'; the HF account of the token must have accepted the terms).
# Credentials (never printed): HF -> /root/.cache/huggingface/token.
# `auto` runs batches -> push -> verify and stops the pod ONLY if the results are verified on HF.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
ROOT=${ROOT:-/root/r1}
WORK=${WORK:-/root/teacher_dataset}
VIDEOS=${NEXAR_VIDEOS:-/root/nexar/train}
FRAMES=${R1_TRAIN_DIR:-/root/frames}
PY=${PY:-/root/venv/bin/python}
SC=$ROOT/student_training/scripts
REPO=eviatarO-org/teacher-dataset-2026-10
export NEXAR_VIDEOS=$VIDEOS R1_TRAIN_DIR=$FRAMES R1_PILOT_OUT=$WORK
for TOKEN_FILE in ${TOKEN_FILE:-} /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
NPROC=${NPROC:-3}   # tracking / attention processes per batch (GPU has room for several)
mkdir -p $WORK/logs $WORK/batches $VIDEOS $FRAMES
STAGE=${1:?stage}; ARG=${2:-}
LOG=$WORK/logs/$STAGE${ARG:+_$ARG}.log
cd $SC 2>/dev/null || true

run() { echo "=== [$STAGE $ARG] start $(date -u) ===" | tee -a $LOG; "$@" >> $LOG 2>&1; EX=$?
        echo "=== [$STAGE $ARG] exit=$EX $(date -u) ===" | tee -a $LOG; [ $EX -eq 0 ] && touch ${LOG%.log}.done; return $EX; }

do_batch() {   # do_batch <windows file of one batch> <tag>
  local F=$1 T=$2 K T0=$(date +%s)
  [ -f $WORK/logs/batch_$T.done ] && { echo "[batch $T] already done"; return 0; }
  $PY -u r1_pod_get_videos.py --windows $F --out $VIDEOS --threads 16 > $WORK/logs/batch_${T}_videos.log 2>&1 || { echo "[batch $T] video download failed"; return 1; }
  local T1=$(date +%s)
  for K in 0 1 2 3; do $PY -u r1_pod_cut_windows.py --windows $F --shard $K/4 > $WORK/logs/batch_${T}_cut_$K.log 2>&1 & done; wait
  [ "$(grep -h "\[cut\] done" $WORK/logs/batch_${T}_cut_?.log | grep -c " 0 incomplete")" -eq 4 ] || { echo "[batch $T] cut failed"; return 1; }
  local T2=$(date +%s)
  # two phases, NOT overlapped: the A1 model of the attention step needs ~6 GB per process and OOMs next to tracking processes
  for K in $(seq 0 $((NPROC-1))); do
    $PY -u r1_attention_boxes.py --stage tracks --windows-file $F --shard $K/$NPROC > $WORK/logs/batch_${T}_tracks_$K.log 2>&1 &
  done; wait
  local NA=${NPROC_ATTN:-3}
  for K in $(seq 0 $((NA-1))); do
    $PY -u r1_attention_boxes.py --stage attn --windows-file $F --shard $K/$NA > $WORK/logs/batch_${T}_attn_$K.log 2>&1 &
  done; wait
  local T3=$(date +%s)
  $PY - $F <<'PY' || { echo "[batch $T] outputs incomplete"; return 1; }
import json, os, sys
w = os.environ.get("WORK", "/root/teacher_dataset")
rows = [json.loads(l) for l in open(sys.argv[1])]
mt = [v for v in {r["video_id"] for r in rows} if not os.path.exists(f"{w}/tracks/{v}.json")]
ma = [r["frames_dir"] for r in rows if not os.path.exists(f"{w}/attn/{r['frames_dir']}.json")]
print(f"batch outputs: {len(rows)} windows, missing tracks {len(mt)}, missing attn {len(ma)}")
sys.exit(1 if (mt or ma) else 0)
PY
  $PY - $F <<'PY'
import json, os, shutil, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
fr = os.environ.get("R1_TRAIN_DIR", "/root/frames"); vd = os.environ.get("NEXAR_VIDEOS", "/root/nexar/train")
for r in rows: shutil.rmtree(f"{fr}/{r['frames_dir']}", ignore_errors=True)
for v in {r["video_id"] for r in rows}:
    try: os.remove(f"{vd}/{v}.mp4")
    except FileNotFoundError: pass
PY
  touch $WORK/logs/batch_$T.done
  echo "[batch $T] $(grep -c "" $F) windows | download $((T1-T0))s, cut $((T2-T1))s, tracks+attention $((T3-T2))s, total $(( $(date +%s)-T0 ))s" | tee -a $WORK/logs/batch_times.log
}

case $STAGE in
setup)   # same pins as the week-1 pods and the local environment (transformers 5.8.0, peft 0.19.1); torch comes from the image
  python3 -m venv --system-site-packages /root/venv
  /root/venv/bin/pip install -q "transformers==5.8.0" "peft==0.19.1" "huggingface_hub>=1.0,<2" opencv-python-headless pyyaml pandas accelerate \
      safetensors einops timm scikit-learn scipy supervision trackers albumentations >> $LOG 2>&1
  $PY -c "import torch,transformers,cv2,trackers;print(torch.__version__,transformers.__version__,cv2.__version__,torch.cuda.get_device_name(0))" | tee -a $LOG
  $PY -m pip freeze > $WORK/pod_env.txt
  df -h /root | tee -a $LOG
  ls $ROOT/third_party/yolopv2_ref/weights/ | tee -a $LOG
  ls $ROOT/outputs/a1_compress256/train/epoch_02/lora_adapter | tee -a $LOG
  wc -l $WORK/windows.jsonl | tee -a $LOG
  ;;
hf_check)   # read-only: HF token accepted by the gated Nexar dataset? all needed videos listed?
  run $PY -u r1_pod_get_videos.py --windows $WORK/windows.jsonl --out $VIDEOS --check
  ;;
videos)   # the 1,401 needed train videos from the HF dataset, 16 threads; resumable, sizes verified
  run $PY -u r1_pod_get_videos.py --windows $WORK/windows.jsonl --out $VIDEOS --threads 16 || exit 1
  echo "videos present: $(ls $VIDEOS/*.mp4 2>/dev/null | wc -l)" | tee -a $LOG
  du -sh $VIDEOS | tee -a $LOG
  ;;
cut)      # the 4,203 windows at native resolution (JPEG q95), 4 processes
  for K in 0 1 2 3; do
    ($PY -u r1_pod_cut_windows.py --windows $WORK/windows.jsonl --shard $K/4 > $WORK/logs/cut_$K.log 2>&1; echo "exit=$?" >> $WORK/logs/cut_$K.log) &
  done
  wait
  tail -n 2 $WORK/logs/cut_?.log | tee -a $LOG
  echo "window folders: $(ls $FRAMES | wc -l) (expect 4203)" | tee -a $LOG
  du -sh $FRAMES | tee -a $LOG
  ;;
smoke)    # first 3 videos through the whole batch procedure; prints the time per step. READ before running everything.
  $PY - <<'PY'
import json, os
w = os.environ.get("WORK", "/root/teacher_dataset")
rows = [json.loads(l) for l in open(f"{w}/windows.jsonl")]
vids = sorted({r["video_id"] for r in rows})[:3]
open(f"{w}/batches/smoke.jsonl", "w").write("".join(json.dumps(r) + chr(10) for r in rows if r["video_id"] in vids))
PY
  do_batch $WORK/batches/smoke.jsonl smoke
  ;;
batches)   # ARG = videos per batch (default 150): download -> cut -> tracks + attention -> delete videos and frames; resumable per batch
  $PY - "${ARG:-150}" <<'PY'
import json, os, sys
w = os.environ.get("WORK", "/root/teacher_dataset"); n = int(sys.argv[1])
rows = [json.loads(l) for l in open(f"{w}/windows.jsonl")]
vids = sorted({r["video_id"] for r in rows})
for b in range(0, len(vids), n):
    vs = set(vids[b:b + n])
    open(f"{w}/batches/b_{b // n:03d}.jsonl", "w").write("".join(json.dumps(r) + chr(10) for r in rows if r["video_id"] in vs))
print(len(vids), "videos in", -(-len(vids) // n), "batches of", n)
PY
  for F in $WORK/batches/b_*.jsonl; do
    T=$(basename $F .jsonl)
    do_batch $F $T || { echo "BATCH $T FAILED" | tee -a $LOG; exit 1; }
    echo "[batches] $T done $(date -u) | tracks $(ls $WORK/tracks | wc -l) of 1401 | attn $(ls $WORK/attn | wc -l) of 4203 | disk $(df -h /root | tail -1 | awk '{print $4}') free" | tee -a $LOG
  done
  ;;
tracks_attn)   # ARG = number of parallel processes (default 3). Each process: tracks then attention for its share of the videos. Resumable.
  N=${ARG:-3}
  for K in $(seq 0 $((N-1))); do
    ( $PY -u r1_attention_boxes.py --stage tracks --shard $K/$N > $WORK/logs/tracks_$K.log 2>&1 &&
      $PY -u r1_attention_boxes.py --stage attn --shard $K/$N > $WORK/logs/attn_$K.log 2>&1; echo "exit=$?" >> $WORK/logs/attn_$K.log ) &
  done
  wait
  echo "tracks: $(ls $WORK/tracks | wc -l) of 1401 | attn: $(ls $WORK/attn | wc -l) of 4203" | tee -a $LOG
  ;;
push)     # small results (tracks, attn, windows, logs) -> private HF dataset repo; nothing durable stays only on the pod
  cd $WORK && tar czf /root/tracks.tar.gz tracks && tar czf /root/attn.tar.gz attn
  $PY - <<'PY' | tee -a $LOG
import os
from huggingface_hub import HfApi
a = HfApi(); repo = "eviatarO-org/teacher-dataset-2026-10"; w = os.environ.get("WORK", "/root/teacher_dataset")
a.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
readme = """---
license: other
---
# teacher-dataset-2026-10 — tracks and A1 attention for the teacher reasoning dataset (CCP V-JEPA2 Reasoning)

Private. Output of the GPU step (Step B) of the full teacher run: 4,203 windows from 1,401 Nexar training videos
(`windows.jsonl`, written by `r1_teacher_windows.py`). The window frames are NOT here; they are cut from the Nexar videos.

* `tracks.tar.gz` — per video `tracks/<vid>.json`: YOLOPv2 (vehicles) + Grounding-DINO-tiny (person / bicycle / motorcycle) detections tracked with BoT-SORT
  (stitched, ego hood removed); `tracks[id].points = [t, [x1,y1,x2,y2], conf]`.
* `attn.tar.gz` — per window `attn/<frames_dir>.json`: A1-compress256 crash score and the crash-head attention mass of every track.
* `windows.jsonl` — the window manifest (label, label_vis, pre_alert, frame_idx, ...).
* `logs/` — pod logs.

Code: GitHub EviatarO/Car-Crash-Prediction-Based-MMLM, branch reasoning-path-vjepa2-llm (`r1_attention_boxes.py`, `r1_teacher_pod.sh`).
"""
a.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=repo, repo_type="dataset")
for f, dst in ((f"/root/tracks.tar.gz", "tracks.tar.gz"), (f"/root/attn.tar.gz", "attn.tar.gz"), (f"{w}/windows.jsonl", "windows.jsonl")):
    a.upload_file(path_or_fileobj=f, path_in_repo=dst, repo_id=repo, repo_type="dataset"); print("pushed", dst)
a.upload_folder(repo_id=repo, repo_type="dataset", folder_path=f"{w}/logs", path_in_repo="logs", allow_patterns=["*.log", "*.done"])
a.upload_file(path_or_fileobj=b"# logs/\n\nPod logs of Step B (setup, videos, cut, tracks, attention).\n", path_in_repo="logs/README.md", repo_id=repo, repo_type="dataset")
print(len(a.list_repo_files(repo, repo_type="dataset")), "files on HF; private:", a.dataset_info(repo).private)
PY
  [ ${PIPESTATUS[0]} -eq 0 ] && touch ${LOG%.log}.done
  ;;
verify)   # archives complete (1401 tracks, 4203 attention files) AND the same bytes are on HF
  $PY - <<'PY' | tee -a $LOG
import os, sys, tarfile
from huggingface_hub import HfApi
w = os.environ.get("WORK", "/root/teacher_dataset"); repo = "eviatarO-org/teacher-dataset-2026-10"
ok = True
for f, n in (("/root/tracks.tar.gz", 1401), ("/root/attn.tar.gz", 4203)):
    with tarfile.open(f) as t:
        cnt = sum(1 for m in t.getmembers() if m.isfile() and m.name.endswith(".json"))
    print(f, "json files:", cnt, "expected", n); ok &= cnt == n
remote = {x.path: x.size for x in HfApi().get_paths_info(repo, ["tracks.tar.gz", "attn.tar.gz", "windows.jsonl"], repo_type="dataset")}
for f, dst in (("/root/tracks.tar.gz", "tracks.tar.gz"), ("/root/attn.tar.gz", "attn.tar.gz"), (f"{w}/windows.jsonl", "windows.jsonl")):
    print(dst, "local", os.path.getsize(f), "HF", remote.get(dst)); ok &= remote.get(dst) == os.path.getsize(f)
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
  [ ${PIPESTATUS[0]} -eq 0 ] && touch ${LOG%.log}.done
  ;;
auto)     # the whole run, unattended. Stops the pod only when `verify` passed. On any failure the pod stays up (nothing is lost, resumable).
  AUTO=$WORK/logs/auto.status; echo "start $(date -u)" > $AUTO
  step() { echo "[auto] $1 $(date -u)" | tee -a $AUTO; bash $0 $1 ${2:-} >> $WORK/logs/auto.out 2>&1 || { echo "[auto] FAILED at $1: pod left running" | tee -a $AUTO; exit 1; }; }
  step batches; step push; step verify
  echo "[auto] verified on HF; stopping the pod $(date -u)" | tee -a $AUTO
  bash $0 stop >> $WORK/logs/auto.out 2>&1; echo "[auto] stop command returned $? $(date -u)" | tee -a $AUTO
  ;;
check_stop)
  if tr '\0' '\n' < /proc/1/environ | grep -q '^RUNPOD_API_KEY='; then echo "RUNPOD_API_KEY: present"; else echo "RUNPOD_API_KEY: MISSING"; fi
  command -v runpodctl >/dev/null && echo "runpodctl: $(command -v runpodctl)" || echo "runpodctl: MISSING"
  tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_POD_ID=' | sed 's/^/pod id env: /' || true
  ;;
stop)   # only after push AND verify (results are byte-identical on HF; the PC pulls them from there)
  [ -f $WORK/logs/verify.done ] || { echo "results not verified on HF: run 'push' then 'verify' first"; exit 3; }
  export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs)
  POD=${ARG:-${RUNPOD_POD_ID:-}}
  [ -n "$POD" ] || { echo "pod id unknown: pass it as the 2nd argument"; exit 4; }
  echo "stopping pod $POD at $(date -u)" | tee -a $LOG
  runpodctl stop pod "$POD" 2>&1 | tee -a $LOG
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
