#!/bin/bash
# Phase 1 on the enlarged Nexar reasoning dataset (plan 2026-10-14_Plan-Phase1-Enlarged-Dataset + Text-Judge-Addendum: 3 arms). Unattended; stops the pod ONLY after every result
# file is verified byte-identical on HF. Needs a pod with >= 150 GB container disk (no network volume needed).
#   P1-aug  : projector only, from random; each epoch draws one random (window version x text version) pair per original window
#             (window = original or its 1/2/3-frame shifted copy, text = teacher's [p 0.4] or one of 3 rephrased versions); up to 15 epochs, early stop
#   P1-long : the control, original windows + teacher texts only, same epochs; BOTH validated on the same enlarged validation pairs
# Then both Phase-1 checkpoints and the old B3 Phase 1 (HF levers_2026-10-13/B3/p1) are evaluated the same way (generation with the frozen language model),
# plus a fact probe on the PROJECTOR OUTPUT.
# Expects in /root/r1 (copied from the PC): code, dataset/manifests/{r1_nexar_reasoning_windows,r1_nexar_reasoning_shifted,r1_nexar_reasoning_control}.jsonl,
# outputs/teacher_dataset_2026-10/{paraphrases_judged.jsonl,windows_shifted.jsonl,windows_control.jsonl}.
#   setsid nohup bash r1_pod_phase1aug.sh > /root/r1_week1/logs/p1aug.nohup 2>&1 < /dev/null &      status: $OUT/logs/p1aug.status
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
OUT=${OUT:-/root/r1_week1}; PY=${PY:-/root/venv/bin/python}; SC=/root/r1/student_training/scripts; RR=/root/r1
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; HFDIR=phase1aug_2026-10-14
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
cd $SC; mkdir -p $OUT/logs
ST=$OUT/logs/p1aug.status; echo "start $(date -u)" > $ST
say() { echo "[p1aug] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; exit 1; }
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ -f $OUT/cache/index_nexar_reasoning_shifted.jsonl ] || [ "${FREE_GB:-0}" -ge 140 ] || fail "container disk has only ${FREE_GB} GB free (need >= 140)"   # only before the caches exist
export CACHE=$OUT/cache NEXAR_VIDEOS=/root/nexar/train R1_TRAIN_DIR=/root/frames
mkdir -p $CACHE $NEXAR_VIDEOS $R1_TRAIN_DIR
W=$RR/outputs/teacher_dataset_2026-10; PARA=$W/paraphrases_judged.jsonl     # copied from the PC later; training waits for it + $W/AUDIT_OK
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }

# 0 ---- original cache (last-layer tokens) from HF
[ -f $CACHE/index_nexar_reasoning.jsonl ] || { say "pull original cache"; $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE > $OUT/logs/pull_r.log 2>&1 || fail pull_r; }

# 1 ---- shifted + control windows: videos -> frames -> encode -> delete frames/videos
if [ ! -f $CACHE/index_nexar_reasoning_shifted.jsonl ] || [ "$(wc -l < $CACHE/index_nexar_reasoning_shifted.jsonl)" -lt 12000 ]; then
  say "videos (HF Nexar train set)"
  cat $W/windows_shifted.jsonl $W/windows_control.jsonl > $OUT/windows_all_new.jsonl
  $PY -u r1_pod_get_videos.py --windows $OUT/windows_all_new.jsonl --out $NEXAR_VIDEOS --threads 16 > $OUT/logs/videos.log 2>&1 || fail videos
  say "cut frames (6 processes)"
  for k in 0 1 2 3 4 5; do $PY -u r1_pod_cut_windows.py --windows $OUT/windows_all_new.jsonl --shard $k/6 > $OUT/logs/cut_$k.log 2>&1 & done
  wait
  EXPECT=$(wc -l < $OUT/windows_all_new.jsonl); GOT=$(ls $R1_TRAIN_DIR | wc -l)
  [ "$GOT" -eq "$EXPECT" ] || fail "cut: $GOT folders, expected $EXPECT"
  say "frames cut: $GOT folders; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"
  say "encode the 24 control windows (their crash-model P must equal the cached one)"
  $PY -u r1_cache_features.py --source nexar_reasoning_control --splits train,val --out-dir $OUT/cache_control > $OUT/logs/encode_control.log 2>&1 || fail encode_control
  $PY - <<'PY' >> $ST 2>&1 || fail control_mismatch
import json, os
c = os.environ["CACHE"]; o = os.environ.get("OUT", "/root/r1_week1")
new = {r["id"][:-2]: r["p_collision"] for r in map(json.loads, open(f"{o}/cache_control/index_nexar_reasoning_control.jsonl"))}
old = {r["id"]: r["p_collision"] for r in map(json.loads, open(f"{c}/index_nexar_reasoning.jsonl"))}
d = max(abs(new[k] - old[k]) for k in new)
print(f"[p1aug] control: {len(new)} re-cut windows, max |P new - P cached| = {d:.5f}")
assert d <= 0.004, "re-cut windows do not reproduce the cached crash-model P"
PY
  say "encode shifted windows (12,286)"
  $PY -u r1_cache_features.py --source nexar_reasoning_shifted --splits train,val --out-dir $CACHE > $OUT/logs/encode_shifted.log 2>&1 || fail encode_shifted
  [ "$(wc -l < $CACHE/index_nexar_reasoning_shifted.jsonl)" -ge 12000 ] || fail "shifted cache incomplete"
  rm -rf $R1_TRAIN_DIR $NEXAR_VIDEOS; say "shifted cache ready; frames and videos deleted; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"
fi

# 1b ---- wait for the judged texts (copied from the PC); give up after 5 h -> stop the pod
if [ ! -s $PARA ]; then
  say "waiting for paraphrases_judged.jsonl"
  for i in $(seq 1 300); do [ -s $PARA ] && break; sleep 60; done
  [ -s $PARA ] || { say "no judged texts after 5 h - stopping the pod (no training)"; stop_pod; exit 1; }
fi
say "texts: $(wc -l < $PARA) judged versions"

# 2 ---- three Phase-1 arms, all validated on the SAME pairs (original + shifted windows, teacher's texts only: --val-texts none)
#   P1long  : control, original windows + teacher texts
#   P1shift : + frame-shifted windows (no rephrased texts)  -- the main arm (the rephrased texts failed the strict Opus audit: 0 major, 7.2 % minor nuances)
#   P1aug   : + shifted windows + judged rephrased texts (p_teacher 0.4) -- labelled extra arm; used only if the user accepts the minor nuances
COMMON="$PY -u r1_train.py --phase 1 --sources nexar_reasoning --cache-dir $CACHE --init random --epochs 15 --patience 2 --select val_loss --eff-bs 16 --micro-bs 4 --lr-merger 1e-3 --grad-ckpt --val-texts none --val-shifts"
EV="$PY -u r1_eval_reasoning.py --order reason_first --cache-dir $CACHE --max-ret 200 --n-blank 10 --train-threshold 600"
say "train P1long + P1shift (parallel)"
$COMMON --out-dir $OUT/P1long > $OUT/logs/P1long_train.log 2>&1 & A=$!
$COMMON --out-dir $OUT/P1shift --aug-shifts > $OUT/logs/P1shift_train.log 2>&1 & B=$!
wait $A || fail P1long_train; wait $B || fail P1shift_train; say "P1long + P1shift trained"
say "evaluate P1long + P1shift (parallel)"
$EV --ckpt $OUT/P1long/best.pt --out-dir $OUT/P1long/eval > $OUT/logs/P1long_eval.log 2>&1 & A=$!
$EV --ckpt $OUT/P1shift/best.pt --out-dir $OUT/P1shift/eval > $OUT/logs/P1shift_eval.log 2>&1 & B=$!
wait $A || fail P1long_eval; wait $B || fail P1shift_eval
say "train P1aug (texts) alone, then evaluate it + the old coarse-stage checkpoint (B3) in parallel"
ARMS="P1long P1shift B3_p1"
mkdir -p $OUT/B3_p1; $PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','levers_2026-10-13/B3/p1/best.pt',local_dir='$OUT/_hf'),'$OUT/B3_p1/best.pt')" || fail get_b3_p1
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','phase2r/results/predictions.jsonl',local_dir='$OUT/_hf'),'$OUT/run1_predictions.jsonl')" || say "run1 predictions not fetched"
if $COMMON --out-dir $OUT/P1aug --aug-texts $PARA --aug-shifts --p-teacher 0.4 > $OUT/logs/P1aug_train.log 2>&1; then
  say "P1aug trained"
  $EV --ckpt $OUT/P1aug/best.pt --out-dir $OUT/P1aug/eval > $OUT/logs/P1aug_eval.log 2>&1 & A=$!
  ARMS="P1long P1shift P1aug B3_p1"
else
  say "P1aug training FAILED (continuing without it)"; A=""
fi
$EV --ckpt $OUT/B3_p1/best.pt --out-dir $OUT/B3_p1/eval > $OUT/logs/B3p1_eval.log 2>&1 || fail B3p1_eval
[ -n "$A" ] && { wait $A || say "P1aug eval FAILED (continuing)"; }
say "projector-output fact probes"
for N in $ARMS; do
  $PY -u r1_fact_probe.py --cache-dir $CACHE --out-dir $OUT/probe_$N --projector-ckpt $OUT/$N/best.pt --proj-tag $N --runs run1=$OUT/run1_predictions.jsonl > $OUT/logs/probe_$N.log 2>&1 || say "probe $N failed (continuing)"
done
export ARMS

# 4 ---- push + verify + stop
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
out = os.environ.get("OUT", "/root/r1_week1"); a = HfApi(); repo = "eviatarO-org/checkpoints"; base = "phase1aug_2026-10-14"
files = []
for n in os.environ["ARMS"].split():
    files += [(f"{out}/{n}/best.pt", f"{n}/best.pt")]
    files += [(f"{out}/{n}/train_log.jsonl", f"{n}/train_log.jsonl")] if os.path.exists(f"{out}/{n}/train_log.jsonl") else []
    files += [(f, f"{n}/results/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/{n}/eval/*")) if os.path.isfile(f)]
    files += [(f"{out}/probe_{n}/fact_probe.json", f"{n}/probe/fact_probe.json")] if os.path.exists(f"{out}/probe_{n}/fact_probe.json") else []
files += [(f"{out}/logs/p1aug.status", "p1aug.status")] + [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/logs/*_train.log"))]
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=repo, repo_type="model")
remote = {x.path: x.size for x in a.get_paths_info(repo, [f"{base}/{d}" for _, d in files], repo_type="model")}
ok = all(remote.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files)
for s, d in files:
    print(d, os.path.getsize(s), remote.get(f"{base}/{d}"))
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
