#!/bin/bash
# Phase 2 with three answer formats (plan 2026-10-15_Plan-Phase2-Three-Formats). Unattended; stops the pod ONLY after every result file is verified
# byte-identical on HF. Needs ~40 GB free container disk (the 18 GB features cache + checkpoints), no network volume.
#   0  pull the features cache and the P1aug checkpoint from HF; smoke tests of the real model (4 steps, 8 generated answers)
#   1  Phase-1 continuation from P1aug, <= 4 epochs, in parallel (pre-alert windows dropped, positives balanced over the TTE groups):
#        P1fixR (answer: Reasoning -> Collision -> Outcome) and P1fixF (Facts -> Reasoning -> Collision -> Outcome); each is evaluated (Phase-1 gate numbers)
#   2  Phase 2 (projector + LoRA, <= 6 epochs): P2R reason-first and P2V verdict-first (Run 2) from P1fixR in parallel, then P2F facts-first from P1fixF
#   3  push best.pt / train_log / eval results / logs to HF checkpoints/phase2f_2026-10-15/, verify sizes, stop the pod
# Everything is validated on the same windows (524 val windows without pre-alert, the teacher's text). A failed run does not stop the others;
# the pod is left running only if a smoke test or the HF verification fails.
# Expects in /root/r1 (copied from the PC): code, dataset/manifests/r1_nexar_reasoning_windows.jsonl,
#   outputs/teacher_dataset_2026-10/{paraphrases_judged.jsonl,fact_cards_teacher.jsonl}.
#   setsid nohup bash r1_pod_phase2f.sh > /root/r1_week1/logs/p2f.nohup 2>&1 < /dev/null &      status: $OUT/logs/p2f.status
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
OUT=${OUT:-/root/r1_week1}; PY=${PY:-/root/venv/bin/python}; SC=/root/r1/student_training/scripts; RR=/root/r1
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; HFDIR=phase2f_2026-10-15
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
cd $SC; mkdir -p $OUT/logs
ST=$OUT/logs/p2f.status; echo "start $(date -u)" > $ST
say() { echo "[p2f] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; exit 1; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
export CACHE=$OUT/cache; mkdir -p $CACHE
W=$RR/outputs/teacher_dataset_2026-10; PARA=$W/paraphrases_judged.jsonl; CARDS=$W/fact_cards_teacher.jsonl
[ -s $PARA ] && [ -s $CARDS ] || fail "paraphrases_judged.jsonl / fact_cards_teacher.jsonl missing"
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ -f $CACHE/index_nexar_reasoning.jsonl ] || [ "${FREE_GB:-0}" -ge 40 ] || fail "container disk has only ${FREE_GB} GB free (need >= 40)"

DATA="--sources nexar_reasoning --cache-dir $CACHE --facts-cards $CARDS --drop-prealert --balance-tte --aug-texts $PARA --val-texts none --p-teacher 0.4 --select val_loss --eff-bs 16 --micro-bs 4 --grad-ckpt"
P1="$PY -u r1_train.py --phase 1 --init random --init-from $OUT/P1aug/best.pt --epochs 4 --patience 1 --lr-merger 3e-4 $DATA"
P2="$PY -u r1_train.py --phase 2 --epochs 6 --patience 1 --lr-merger 2e-5 --lr-lora 2e-4 $DATA"
EV="$PY -u r1_eval_reasoning.py --facts-cards $CARDS --drop-prealert --cache-dir $CACHE --max-ret 100 --n-blank 5 --train-threshold 600"

# 0 ---- features cache + P1aug from HF
[ -f $CACHE/index_nexar_reasoning.jsonl ] || { say "pull features cache"; $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE > $OUT/logs/pull_cache.log 2>&1 || fail pull_cache; }
mkdir -p $OUT/P1aug
[ -f $OUT/P1aug/best.pt ] || $PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','phase1aug_2026-10-14/P1aug/best.pt',local_dir='$OUT/_hf'),'$OUT/P1aug/best.pt')" || fail get_p1aug
say "cache + P1aug ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

say "smoke tests (real model: 4 training steps per format, 8 generated answers)"
$P2 --order facts_first --init-from $OUT/P1aug/best.pt --max-steps 4 --epochs 1 --out-dir $OUT/smoke_F > $OUT/logs/smoke_F.log 2>&1 || fail smoke_train_F
$P2 --order verdict_first --init-from $OUT/P1aug/best.pt --max-steps 4 --epochs 1 --out-dir $OUT/smoke_V > $OUT/logs/smoke_V.log 2>&1 || fail smoke_train_V
$EV --ckpt $OUT/smoke_F/best.pt --order facts_first --limit 8 --max-new 120 --max-ret 4 --n-blank 1 --train-threshold 0 --out-dir $OUT/smoke_F/eval > $OUT/logs/smoke_eval_F.log 2>&1 || fail smoke_eval_F
say "smoke tests passed"

# 1 ---- Phase-1 continuation (gate numbers: missed crashes per TTE vs P1aug's 14 / 25 / 39 %)
run_p1() { N=$1; O=$2
  $P1 --order $O --out-dir $OUT/$N > $OUT/logs/${N}_train.log 2>&1 || return 1
  $EV --ckpt $OUT/$N/best.pt --order $O --out-dir $OUT/$N/eval > $OUT/logs/${N}_eval.log 2>&1 || return 1
}
say "Phase-1 continuation P1fixR + P1fixF (parallel)"
run_p1 P1fixR reason_first & A=$!
run_p1 P1fixF facts_first & B=$!
wait $A || say "P1fixR FAILED (its Phase-2 runs are skipped)"
wait $B || say "P1fixF FAILED (the facts-first Phase-2 run is skipped)"
say "Phase-1 gate numbers: $(grep -h -a 'missed crashes' $OUT/P1fixR/eval/summary.md $OUT/P1fixF/eval/summary.md 2>/dev/null | cut -c1-300 | tr '\n' ' ')"

# 2 ---- Phase 2
run_p2() { N=$1; O=$2; START=$3
  $P2 --order $O --init-from $OUT/$START/best.pt --out-dir $OUT/$N > $OUT/logs/${N}_train.log 2>&1 || return 1
  $EV --ckpt $OUT/$N/best.pt --order $O --before-ckpt $OUT/$START/best.pt --out-dir $OUT/$N/eval > $OUT/logs/${N}_eval.log 2>&1 || return 1
}
if [ -f $OUT/P1fixR/best.pt ]; then
  say "Phase 2: P2R (reason-first) + P2V (verdict-first) in parallel"
  run_p2 P2R reason_first P1fixR & A=$!
  run_p2 P2V verdict_first P1fixR & B=$!
  wait $A || say "P2R FAILED"; wait $B || say "P2V FAILED"
fi
if [ -f $OUT/P1fixF/best.pt ]; then
  say "Phase 2: P2F (facts-first)"
  run_p2 P2F facts_first P1fixF || say "P2F FAILED"
fi
ARMS=""; for N in P1fixR P1fixF P2R P2V P2F; do [ -f $OUT/$N/best.pt ] && ARMS="$ARMS $N"; done
[ -n "$ARMS" ] || fail "no run produced a checkpoint"
export ARMS HFDIR
say "arms with results:$ARMS"

# 3 ---- push + verify + stop
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
out = os.environ.get("OUT", "/root/r1_week1"); a = HfApi(); repo = "eviatarO-org/checkpoints"; base = os.environ["HFDIR"]
files = []
for n in os.environ["ARMS"].split():
    files += [(f"{out}/{n}/best.pt", f"{n}/best.pt")]
    files += [(f"{out}/{n}/train_log.jsonl", f"{n}/train_log.jsonl")] if os.path.exists(f"{out}/{n}/train_log.jsonl") else []
    files += [(f, f"{n}/results/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/{n}/eval/*")) if os.path.isfile(f)]
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/logs/*_train.log")) + sorted(glob.glob(f"{out}/logs/*_eval.log")) + sorted(glob.glob(f"{out}/logs/smoke_*.log"))]
files += [(f"{out}/logs/p2f.status", "p2f.status")]
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
