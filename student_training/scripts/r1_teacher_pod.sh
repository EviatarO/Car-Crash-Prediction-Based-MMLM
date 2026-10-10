#!/bin/bash
# Step B of the full teacher run (plan 2026-10-10_Plan-Teacher-Full-Run.md): tracks + A1 attention for 4,203 windows on a RunPod pod.
# The pod does the GPU work only: it writes tracks/<vid>.json and attn/<frames_dir>.json (small). The PC draws the boxes and runs the teacher.
# Usage:  bash r1_teacher_pod.sh <stage> [arg]
#   setup | kaggle_check | videos | cut | smoke | tracks_attn [n_procs] | push | check_stop | stop <pod_id>
# Layout on the pod: code /root/r1, work dir /root/teacher_dataset (windows.jsonl in, tracks/ attn/ out), videos /root/nexar/train, frames /root/frames.
# Credentials (never printed): Kaggle -> KAGGLE_USERNAME + KAGGLE_KEY (pod env) or ~/.kaggle/kaggle.json; HF -> /root/.cache/huggingface/token.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
ROOT=${ROOT:-/root/r1}
WORK=${WORK:-/root/teacher_dataset}
VIDEOS=${NEXAR_VIDEOS:-/root/nexar/train}
FRAMES=${R1_TRAIN_DIR:-/root/frames}
PY=${PY:-/root/venv/bin/python}
SC=$ROOT/student_training/scripts
COMP=${NEXAR_COMP:-nexar-collision-prediction}
PREFIX=${NEXAR_PREFIX:-train/}
REPO=eviatarO-org/teacher-dataset-2026-10
export NEXAR_VIDEOS=$VIDEOS R1_TRAIN_DIR=$FRAMES R1_PILOT_OUT=$WORK
for TOKEN_FILE in ${TOKEN_FILE:-} /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
mkdir -p $WORK/logs $VIDEOS $FRAMES
STAGE=${1:?stage}; ARG=${2:-}
LOG=$WORK/logs/$STAGE${ARG:+_$ARG}.log
cd $SC 2>/dev/null || true

run() { echo "=== [$STAGE $ARG] start $(date -u) ===" | tee -a $LOG; "$@" >> $LOG 2>&1; EX=$?
        echo "=== [$STAGE $ARG] exit=$EX $(date -u) ===" | tee -a $LOG; [ $EX -eq 0 ] && touch ${LOG%.log}.done; return $EX; }

case $STAGE in
setup)   # same pins as the week-1 pods and the local environment (transformers 5.8.0, peft 0.19.1); torch comes from the image
  python3 -m venv --system-site-packages /root/venv
  /root/venv/bin/pip install -q "transformers==5.8.0" "peft==0.19.1" "huggingface_hub>=1.0,<2" opencv-python-headless pyyaml pandas accelerate \
      safetensors einops timm scikit-learn scipy supervision trackers kaggle >> $LOG 2>&1
  $PY -c "import torch,transformers,cv2,trackers;print(torch.__version__,transformers.__version__,cv2.__version__,torch.cuda.get_device_name(0))" | tee -a $LOG
  $PY -m pip freeze > $WORK/pod_env.txt
  df -h /root | tee -a $LOG
  ls $ROOT/third_party/yolopv2_ref/weights/ | tee -a $LOG
  ls $ROOT/outputs/a1_compress256/train/epoch_02/lora_adapter | tee -a $LOG
  wc -l $WORK/windows.jsonl | tee -a $LOG
  ;;
kaggle_check)   # read-only: is the token accepted, what is the competition's file layout?
  /root/venv/bin/kaggle competitions list -s nexar 2>&1 | head -8 | tee -a $LOG
  /root/venv/bin/kaggle competitions files -c $COMP 2>&1 | head -12 | tee -a $LOG
  ;;
videos)   # the 1,401 needed train videos, one file per call, 8 in parallel; resumable (existing non-empty files are skipped)
  $PY - <<'PY' > $WORK/need_videos.txt
import json, os
w = os.environ.get("WORK", "/root/teacher_dataset")
print("\n".join(sorted({json.loads(l)["video_id"] for l in open(f"{w}/windows.jsonl")})))
PY
  wc -l $WORK/need_videos.txt | tee -a $LOG
  export COMP PREFIX VIDEOS
  cat $WORK/need_videos.txt | xargs -P 8 -I{} bash -c '
    f=$VIDEOS/{}.mp4
    [ -s "$f" ] && exit 0
    /root/venv/bin/kaggle competitions download -c $COMP -f ${PREFIX}{}.mp4 -p $VIDEOS --quiet >/dev/null 2>&1
    [ -s "$VIDEOS/{}.mp4.zip" ] && (cd $VIDEOS && unzip -qo {}.mp4.zip && rm -f {}.mp4.zip)
    [ -s "$f" ] || echo "MISSING {}" ' >> $LOG 2>&1
  echo "videos present: $(ls $VIDEOS/*.mp4 2>/dev/null | wc -l) of $(wc -l < $WORK/need_videos.txt); missing: $(grep -c MISSING $LOG)" | tee -a $LOG
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
smoke)    # 3 videos end to end: tracks + attention; prints seconds per video / window. READ before running everything.
  START=$(date +%s)
  run $PY -u r1_attention_boxes.py --stage tracks --limit-videos 3 || exit 1
  MID=$(date +%s)
  run $PY -u r1_attention_boxes.py --stage attn --limit-videos 3 || exit 1
  END=$(date +%s)
  echo "tracks: $((MID-START)) s for 3 videos | attention: $((END-MID)) s for $(ls $WORK/attn | wc -l) windows" | tee -a $LOG
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv | tee -a $LOG
  ;;
tracks_attn)   # ARG = number of parallel processes (default 3). Each process: tracks then attention for its share of the videos. Resumable.
  N=${ARG:-3}
  for K in $(seq 0 $((N-1))); do
    ( $PY -u r1_attention_boxes.py --stage tracks --shard $K/$N > $WORK/logs/tracks_$K.log 2>&1 &&
      $PY -u r1_attention_boxes.py --stage attn --shard $K/$N > $WORK/logs/attn_$K.log 2>&1; echo "exit=$?" >> $WORK/logs/attn_$K.log ) &
  done
  wait
  echo "tracks: $(ls $WORK/tracks | wc -l) of $(wc -l < $WORK/need_videos.txt) | attn: $(ls $WORK/attn | wc -l) of 4203" | tee -a $LOG
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
check_stop)
  if tr '\0' '\n' < /proc/1/environ | grep -q '^RUNPOD_API_KEY='; then echo "RUNPOD_API_KEY: present"; else echo "RUNPOD_API_KEY: MISSING"; fi
  command -v runpodctl >/dev/null && echo "runpodctl: $(command -v runpodctl)" || echo "runpodctl: MISSING"
  tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_POD_ID=' | sed 's/^/pod id env: /' || true
  ;;
stop)   # only after the push finished AND the PC downloaded tracks/attn from HF
  [ -f $WORK/logs/push.done ] || { echo "push has not finished: run 'push' first"; exit 3; }
  export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs)
  POD=${ARG:-${RUNPOD_POD_ID:-}}
  [ -n "$POD" ] || { echo "pod id unknown: pass it as the 2nd argument"; exit 4; }
  echo "stopping pod $POD at $(date -u)" | tee -a $LOG
  runpodctl stop pod "$POD" 2>&1 | tee -a $LOG
  ;;
*) echo "unknown stage $STAGE"; exit 2 ;;
esac
