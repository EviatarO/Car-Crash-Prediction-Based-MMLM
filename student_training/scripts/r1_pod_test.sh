#!/bin/bash
# Private-test session (plan 2026-10-17_Plan-Test-Private-Reasoning, step 2). Unattended; stops the pod ONLY after every result file is verified byte-identical on HF.
#   0  checks; download the 677 private test videos (HF nexar-ai/nexar_collision_prediction, test-private/)
#   1  red boxes: tracks (YOLOPv2 + Grounding-DINO + BoT-SORT) then crash-model attention, 3 processes each (resumable); first a 2-video smoke
#   2  encode the 677 windows with the frozen V-JEPA2 / A1 encoder -> features cache; SANITY: the crash model's AUC / AP on these windows must reproduce its
#      known private-test result (AP ~ 0.913)
#   3  decision-only evaluation of P2V (verdict-first) and P2R (reason-first) from HF checkpoints/phase2f_2026-10-15/ (no reference text: --no-ref);
#      a 20-window smoke first
#   4  push tracks / attention / windows to HF dataset eviatarO-org/teacher-test-2026-10 (private) and the evaluation results to
#      eviatarO-org/checkpoints/test_eval_2026-10-17/, verify sizes, stop the pod. The pod is left running on any failure.
# Expects in /root/r1 (copied from the PC): code, dataset/manifests/r1_nexar_reasoning_test.jsonl, outputs/teacher_test_2026-10/windows.jsonl,
# outputs/a1_compress256/train/epoch_02/lora_adapter; and the 677 frame folders in /root/test_frames/<id>_hires/ (16 jpg each).
#   setsid nohup bash r1_pod_test.sh > /root/teacher_test/logs/ptest.nohup 2>&1 < /dev/null &        status: /root/teacher_test/logs/ptest.status
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-/root/venv/bin/python}; RR=/root/r1; SC=$RR/student_training/scripts
WORK=/root/teacher_test; VIDEOS=/root/nexar/test; FRAMES=/root/test_frames; CACHE=$WORK/cache; EV=$WORK/eval
R_CKPT=eviatarO-org/checkpoints; R_TEST=eviatarO-org/teacher-test-2026-10; HFDIR=test_eval_2026-10-17
export R1_PILOT_OUT=$WORK NEXAR_VIDEOS=$VIDEOS R1_TRAIN_DIR=$FRAMES
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
mkdir -p $WORK/logs $VIDEOS $CACHE $EV; cd $SC
ST=$WORK/logs/ptest.status; echo "start $(date -u)" > $ST
say() { echo "[ptest] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; exit 1; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
W=$WORK/windows.jsonl; cp $RR/outputs/teacher_test_2026-10/windows.jsonl $W
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 20 ] || fail "container disk has only ${FREE_GB} GB free (need >= 20)"
[ "$(ls -d $FRAMES/*_hires 2>/dev/null | wc -l)" -eq 677 ] || fail "expected 677 frame folders in $FRAMES, found $(ls -d $FRAMES/*_hires 2>/dev/null | wc -l)"

# 0 ---- videos
say "tracking dependencies (YOLOPv2 code is in /root/r1/third_party; trackers / supervision / matplotlib are not in the standard setup)"
$PY -c "import trackers, supervision, matplotlib, scipy" 2>/dev/null || /root/venv/bin/pip install -q scipy supervision trackers matplotlib albumentations >> $WORK/logs/pip.log 2>&1
[ -f $RR/third_party/yolopv2_ref/weights/yolopv2.pt ] || fail "third_party/yolopv2_ref/weights/yolopv2.pt missing (copy third_party to /root/r1)"
say "download the 677 private test videos"
$PY -u r1_pod_get_videos.py --windows $W --out $VIDEOS --prefix test-private/ --threads 16 > $WORK/logs/videos.log 2>&1 || fail videos
[ "$(ls $VIDEOS/*.mp4 | wc -l)" -ge 677 ] || fail "videos: $(ls $VIDEOS/*.mp4 | wc -l) of 677"
say "videos ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# 1 ---- boxes (two phases, NOT overlapped: the A1 model of the attention step needs ~6 GB per process and OOMs next to tracking processes)
say "boxes smoke: 2 videos"
$PY -u r1_attention_boxes.py --stage tracks --windows-file $W --limit-videos 2 > $WORK/logs/smoke_tracks.log 2>&1 || fail smoke_tracks
$PY -u r1_attention_boxes.py --stage attn --windows-file $W --limit-videos 2 > $WORK/logs/smoke_attn.log 2>&1 || fail smoke_attn
[ "$(ls $WORK/tracks | wc -l)" -eq 2 ] && [ "$(ls $WORK/attn | wc -l)" -eq 2 ] || fail "smoke outputs: tracks $(ls $WORK/tracks | wc -l), attn $(ls $WORK/attn | wc -l)"
say "tracks (3 processes)"
for K in 0 1 2; do $PY -u r1_attention_boxes.py --stage tracks --windows-file $W --shard $K/3 > $WORK/logs/tracks_$K.log 2>&1 & done; wait
say "attention (3 processes)"
for K in 0 1 2; do $PY -u r1_attention_boxes.py --stage attn --windows-file $W --shard $K/3 > $WORK/logs/attn_$K.log 2>&1 & done; wait
NT=$(ls $WORK/tracks | wc -l); NA=$(ls $WORK/attn | wc -l)
say "tracks $NT of 677 | attention $NA of 677"
[ "$NT" -eq 677 ] && [ "$NA" -eq 677 ] || fail "boxes incomplete: tracks $NT, attention $NA"

# 2 ---- encode for the reasoning model + the crash-model sanity check
say "encode the 677 windows"
$PY -u r1_cache_features.py --source nexar_reasoning_test --splits test --out-dir $CACHE > $WORK/logs/encode.log 2>&1 || fail encode
[ "$(wc -l < $CACHE/index_nexar_reasoning_test.jsonl)" -eq 677 ] || fail "encode: $(wc -l < $CACHE/index_nexar_reasoning_test.jsonl) of 677"
$PY - <<'PY' >> $ST 2>&1 || fail crash_model_sanity
import json, os
from sklearn.metrics import average_precision_score, roc_auc_score
c = os.environ.get("CACHE", "/root/teacher_test/cache")
idx = [json.loads(l) for l in open(f"{c}/index_nexar_reasoning_test.jsonl")]
man = {m["id"]: m for m in map(json.loads, open("/root/r1/dataset/manifests/r1_nexar_reasoning_test.jsonl"))}
y = [man[r["id"]]["label"] for r in idx]; p = [r["p_collision"] for r in idx]
ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
print(f"[ptest] crash model on the 677 private test windows: AP {ap:.4f}  AUC {auc:.4f}  (known: AP 0.9128 on the private test)")
assert abs(ap - 0.9128) < 0.02, "the crash model does not reproduce its known private-test AP: windows / frames / adapter mismatch"
PY
say "crash-model sanity passed"

# 3 ---- decision-only evaluation of the finalists
mkdir -p $WORK/ckpt/P2V $WORK/ckpt/P2R
for N in P2V P2R; do
  $PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','phase2f_2026-10-15/$N/best.pt',local_dir='$WORK/_hf'),'$WORK/ckpt/$N/best.pt')" || fail get_$N
done
EVC="$PY -u r1_eval_reasoning.py --source nexar_reasoning_test --no-ref --cache-dir $CACHE --max-new 160"
say "evaluation smoke (20 windows)"
$EVC --ckpt $WORK/ckpt/P2V/best.pt --order verdict_first --limit 20 --out-dir $EV/smoke > $WORK/logs/eval_smoke.log 2>&1 || fail eval_smoke
say "evaluation of P2V and P2R (parallel)"
$EVC --ckpt $WORK/ckpt/P2V/best.pt --order verdict_first --out-dir $EV/P2V > $WORK/logs/eval_P2V.log 2>&1 & A=$!
$EVC --ckpt $WORK/ckpt/P2R/best.pt --order reason_first --out-dir $EV/P2R > $WORK/logs/eval_P2R.log 2>&1 & B=$!
wait $A || fail eval_P2V; wait $B || fail eval_P2R
say "evaluations done: $(grep -h -a '^| a1_p' $EV/P2V/summary.md | cut -c1-80)"

# 4 ---- push + verify + stop
cd $WORK && tar czf /root/tracks.tar.gz tracks && tar czf /root/attn.tar.gz attn && cd $SC
export WORK R_TEST R_CKPT HFDIR
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); w = os.environ["WORK"]; rt, rc, base = os.environ["R_TEST"], os.environ["R_CKPT"], os.environ["HFDIR"]
a.create_repo(rt, repo_type="dataset", private=True, exist_ok=True)
readme = """---
license: other
---
# teacher-test-2026-10 - boxes inputs for the Nexar PRIVATE test windows (CCP V-JEPA2 Reasoning)

Private. 677 private-test windows (one per video, ending at its TTE 0.5 / 1.0 / 1.5 s).
* `tracks.tar.gz` - per video `tracks/<vid>.json`: YOLOPv2 + Grounding-DINO-tiny detections tracked with BoT-SORT (stitched, ego hood removed).
* `attn.tar.gz` - per window `attn/<id>_hires.json`: the crash model's score and the crash-head attention mass of every track.
* `windows.jsonl` - the window manifest (label = Nexar `event_occurs`, TTE, frame indices).
The window frames are not here (they are the Nexar test frames, extracted locally). Test texts are used ONLY for final evaluation.
Code: branch reasoning-path-vjepa2-llm (`r1_test_windows.py`, `r1_attention_boxes.py`, `r1_pod_test.sh`).
"""
files_t = [("/root/tracks.tar.gz", "tracks.tar.gz"), ("/root/attn.tar.gz", "attn.tar.gz"), (f"{w}/windows.jsonl", "windows.jsonl")]
a.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md", repo_id=rt, repo_type="dataset")
files_c = []
for n in ("P2V", "P2R"):
    files_c += [(f, f"{n}/results/{os.path.basename(f)}") for f in sorted(glob.glob(f"{w}/eval/{n}/*")) if os.path.isfile(f)]
files_c += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{w}/logs/*.log"))]
files_c += [(f"{w}/logs/ptest.status", "ptest.status")]
for s, d in files_t:
    a.upload_file(path_or_fileobj=s, path_in_repo=d, repo_id=rt, repo_type="dataset")
for s, d in files_c:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=rc, repo_type="model")
rem_t = {x.path: x.size for x in a.get_paths_info(rt, [d for _, d in files_t], repo_type="dataset")}
rem_c = {x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files_c], repo_type="model")}
ok = all(rem_t.get(d) == os.path.getsize(s) for s, d in files_t) and all(rem_c.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files_c)
for s, d in files_t + files_c:
    print(d, os.path.getsize(s))
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
