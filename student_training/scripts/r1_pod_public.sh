#!/bin/bash
# Public-test session (2026-10-18): decision-only evaluation of P2V (verdict-first) on the 667 Nexar PUBLIC test windows, the half never used so far.
# A cut-down r1_pod_test.sh: no red boxes, no teacher. Unattended; stops the pod ONLY after every result file is verified byte-identical on HF.
#   0  checks; download the 667 public test videos (HF nexar-ai/nexar_collision_prediction, test-public/); extract the 16 frames of each window
#      (extract_highres_frames_test.py: the same code that made the private frames)
#   1  encode the 667 windows with the frozen V-JEPA2 / A1 encoder -> features cache; SANITY: the crash model must reproduce its known public-test
#      AP 0.9096 (+-0.01), otherwise frames / windows / adapter are wrong and the pod is left running
#   2  P2V (HF checkpoints/phase2f_2026-10-15/P2V/best.pt): a 20-window smoke, then all 667 windows (--no-ref: decision scores + generated answers)
#   3  push the results to HF eviatarO-org/checkpoints/test_eval_public_2026-10-18/, verify sizes, stop the pod. The pod is left running on any failure.
# Expects in /root/r1 (copied from the PC): code, dataset/manifests/{r1_nexar_reasoning_test_public.jsonl,test_manifest_public.jsonl},
# outputs/teacher_test_2026-10/windows_public.jsonl, outputs/a1_compress256/train/epoch_02/lora_adapter.
#   setsid nohup bash r1_pod_public.sh > /root/public_test/logs/ppub.nohup 2>&1 < /dev/null &        status: /root/public_test/logs/ppub.status
# DRYRUN=1 bash r1_pod_public.sh   prints the commands without running them (no GPU, no network).
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts
WORK=${WORK:-/root/public_test}; VIDEOS=/root/nexar/test_public; FRAMES=/root/test_frames_public; CACHE=$WORK/cache; EV=$WORK/eval
R_CKPT=eviatarO-org/checkpoints; HFDIR=test_eval_public_2026-10-18; N=667; KNOWN_AP=0.9096
export R1_TRAIN_DIR=$FRAMES WORK CACHE R_CKPT HFDIR KNOWN_AP
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
run() { if [ "$D" = 1 ]; then echo "  + $*"; else eval "$@"; fi; }
mkdir -p $WORK/logs; [ "$D" = 1 ] || mkdir -p $VIDEOS $FRAMES $CACHE $EV; cd $SC
ST=$WORK/logs/ppub.status; echo "start $(date -u)" > $ST
say() { echo "[ppub] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; [ "$D" = 1 ] || exit 1; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
W=$RR/outputs/teacher_test_2026-10/windows_public.jsonl; MAN=$RR/dataset/manifests/r1_nexar_reasoning_test_public.jsonl; TM=$RR/dataset/manifests/test_manifest_public.jsonl
for f in $W $MAN $TM $RR/outputs/a1_compress256/train/epoch_02/lora_adapter; do [ -e $f ] || fail "missing $f"; done
[ "$(wc -l < $W)" -eq $N ] || fail "windows_public.jsonl has $(wc -l < $W) rows, expected $N"
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 20 ] || fail "container disk has only ${FREE_GB} GB free (need >= 20)"

# 0 ---- videos + frames
run "$PY -c 'import cv2' 2>/dev/null || /root/venv/bin/pip install -q opencv-python-headless >> $WORK/logs/pip.log 2>&1"
say "download the $N public test videos (prefix test-public/)"
run "$PY -u r1_pod_get_videos.py --windows $W --out $VIDEOS --prefix test-public/ --threads 16 > $WORK/logs/videos.log 2>&1" || fail "videos (see logs/videos.log; if the repo folder is not test-public/, fix --prefix)"
[ "$D" = 1 ] || [ "$(ls $VIDEOS/*.mp4 | wc -l)" -ge $N ] || fail "videos: $(ls $VIDEOS/*.mp4 | wc -l) of $N"
say "extract the 16 frames per window"
run "$PY -u extract_highres_frames_test.py --manifest_in $TM --dst_root $FRAMES --src_videos $VIDEOS --manifest_out $WORK/test_manifest_public_hires.jsonl --failures_json $WORK/extract_failures.json > $WORK/logs/extract.log 2>&1" || fail extract
[ "$D" = 1 ] || [ "$(find $FRAMES -name 'frame_*.jpg' | wc -l)" -eq $((N * 16)) ] || fail "frames: $(find $FRAMES -name 'frame_*.jpg' | wc -l) of $((N * 16))"
say "frames ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# 1 ---- encode + crash-model sanity
say "encode the $N windows"
run "$PY -u r1_cache_features.py --source nexar_reasoning_test_public --splits test --out-dir $CACHE > $WORK/logs/encode.log 2>&1" || fail encode
[ "$D" = 1 ] || [ "$(wc -l < $CACHE/index_nexar_reasoning_test_public.jsonl)" -eq $N ] || fail "encode: $(wc -l < $CACHE/index_nexar_reasoning_test_public.jsonl) of $N"
if [ "$D" = 1 ]; then echo "  + crash-model sanity: AP on the $N windows within 0.01 of $KNOWN_AP"; else
$PY - <<'PY' >> $ST 2>&1 || fail crash_model_sanity
import json, os
from sklearn.metrics import average_precision_score, roc_auc_score
c, known = os.environ["CACHE"], float(os.environ["KNOWN_AP"])
idx = [json.loads(l) for l in open(f"{c}/index_nexar_reasoning_test_public.jsonl")]
man = {m["id"]: m for m in map(json.loads, open("/root/r1/dataset/manifests/r1_nexar_reasoning_test_public.jsonl"))}
y = [man[r["id"]]["label"] for r in idx]; p = [r["p_collision"] for r in idx]
ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
print(f"[ppub] crash model on the {len(idx)} public test windows: AP {ap:.4f}  AUC {auc:.4f}  (known: AP {known} on the public test)")
assert abs(ap - known) < 0.01, "the crash model does not reproduce its known public-test AP: windows / frames / adapter mismatch"
PY
fi
say "crash-model sanity passed"

# 2 ---- P2V
mkdir -p $WORK/ckpt/P2V
run "$PY -c \"from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','phase2f_2026-10-15/P2V/best.pt',local_dir='$WORK/_hf'),'$WORK/ckpt/P2V/best.pt')\"" || fail get_P2V
EVC="$PY -u r1_eval_reasoning.py --source nexar_reasoning_test_public --no-ref --cache-dir $CACHE --max-new 160 --ckpt $WORK/ckpt/P2V/best.pt --order verdict_first"
say "evaluation smoke (20 windows)"
run "$EVC --limit 20 --out-dir $EV/smoke > $WORK/logs/eval_smoke.log 2>&1" || fail eval_smoke
say "evaluation of P2V on the $N public windows"
run "$EVC --out-dir $EV/P2V > $WORK/logs/eval_P2V.log 2>&1" || fail eval_P2V
say "evaluation done: $(grep -h -a '^| a1_p' $EV/P2V/summary.md 2>/dev/null | cut -c1-80)"

# 3 ---- push + verify + stop
if [ "$D" = 1 ]; then echo "  + push $EV/P2V/*, logs, status, windows to $R_CKPT/$HFDIR; verify sizes; stop the pod"; exit 0; fi
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); w = os.environ["WORK"]; rc, base = os.environ["R_CKPT"], os.environ["HFDIR"]
files = [(f, f"P2V/results/{os.path.basename(f)}") for f in sorted(glob.glob(f"{w}/eval/P2V/*")) if os.path.isfile(f)]
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{w}/logs/*.log"))]
files += [(f"{w}/logs/ppub.status", "ppub.status"), (f"{os.environ['CACHE']}/index_nexar_reasoning_test_public.jsonl", "index_nexar_reasoning_test_public.jsonl")]
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=rc, repo_type="model")
rem = {x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files], repo_type="model")}
ok = all(rem.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files)
for s, d in files:
    print(d, os.path.getsize(s), rem.get(f"{base}/{d}"))
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
