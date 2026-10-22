#!/bin/bash
# Step 1 of plan 2026-10-22_Plan-Single-Stage-and-Stage3-Encoder-LoRA: the SINGLE-STAGE recipe, 2 seeds in parallel on ONE pod (the control's second seed was skipped by the user).
#   S0 / S1 (--seed 0 / 1):  ONE projector stage from a random start on P1fixR's data (pre-alert dropped, TTE-balanced positives, reasoning first), <= 10 epochs,
#                            patience 2, lr 1e-3, best by validation loss  ->  P2V exactly as the 21-Oct control (verdict first, projector + LoRA, <= 6 epochs, patience 1,
#                            lr 2e-5 / 2e-4, best by validation loss; checks at 0.5 / 0.75 / 1.25 / 1.5 epochs logged only). Relaxed rewordings, p_teacher 0.4. Input: last-layer tokens.
# Evaluations per run, VALIDATION ONLY: scoring on val full (641), scoring on the 600-window train sample, text on the 524 core windows + BERTScore. Fact cards run locally afterwards.
# User rule 2026-10-22: on ANY major failure the driver pushes the logs + status to HF and STOPS the pod (no idle pod); after success it stops after the HF byte check.
# Needs a 32 GB GPU and >= 50 GB free disk (100 GB recommended), NO network volume. Expects in /root/r1 (copied from the PC): code,
# dataset/manifests/{r1_nexar_reasoning_windows,r1_nexar_reasoning_extra,r1_nexar_reasoning_v2}.jsonl, dataset/public_samples/qwen3vl_cfg, outputs/a1_compress256/train/epoch_02/lora_adapter,
# outputs/teacher_dataset_2026-10/{paraphrases_relaxed,fact_cards_teacher}.jsonl, outputs/teacher_extra_2026-10/{windows_new,fact_cards_v2}.jsonl.
#   setsid nohup bash r1_pod_base.sh > /root/r1_base/logs/base.nohup 2>&1 < /dev/null &            status: /root/r1_base/logs/base.status
# DRYRUN=1 prints the commands without running them (no GPU, no network).
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts; OUT=${OUT:-/root/r1_base}
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; HFDIR=base_2026-10-22
CACHE=$OUT/cache; CACHEX=$OUT/cache_extra; CACHES="$CACHE,$CACHEX"
VID=/root/nexar; FR=/root/frames_extra
MAN=$RR/dataset/manifests; W=$RR/outputs/teacher_dataset_2026-10; W2=$RR/outputs/teacher_extra_2026-10
PARAX=$W/paraphrases_relaxed.jsonl; CARDS=$W/fact_cards_teacher.jsonl; CARDS2=$W2/fact_cards_v2.jsonl
NEWWIN=$W2/windows_new.jsonl; NEWVAL=$OUT/windows_new_val.jsonl; V2=$MAN/r1_nexar_reasoning_v2.jsonl; EXTRA=$MAN/r1_nexar_reasoning_extra.jsonl
export OUT CACHE CACHEX HFDIR R_CKPT
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
mkdir -p $OUT/logs; cd $SC
ST=$OUT/logs/base.status; echo "start $(date -u)" >> $ST
SYNC="--subdirs S0,S1,logs,. --patterns train_log.jsonl,best.pt,done.*,*.log,base.status"
say() { echo "[base] $* $(date -u)" | tee -a $ST; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
fail() {                                                         # user rule: never leave a failed pod idle
  say "FAILED (major): $* - pushing the logs to HF, then stopping the pod"
  [ "$D" = 1 ] && return 0
  $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/failed $SYNC >> $OUT/logs/sync.log 2>&1 || true
  stop_pod; exit 1; }
stage() { n=$1; lg=$2; shift 2
  if [ "$D" = 1 ]; then echo "  + [$n] $*"; return 0; fi
  [ -f "$OUT/done.$n" ] && { say "skip $n (done)"; return 0; }
  say "start $n"; "$@" > "$lg" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch "$OUT/done.$n"; say "done $n"; else say "FAILED $n (rc=$rc), log $lg"; fi; return $rc; }

# 0 ---- restore + checks
if [ "$D" = 1 ]; then echo "  + [restore] $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live"; else
  $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live >> $OUT/logs/restore.log 2>&1 || say "nothing to restore from HF (fresh start)"
  say "restore: $(ls $OUT/done.* 2>/dev/null | wc -l) finished stages found"; fi
for f in $NEWWIN $PARAX $CARDS $CARDS2 $V2 $EXTRA $MAN/r1_nexar_reasoning_windows.jsonl $RR/outputs/a1_compress256/train/epoch_02/lora_adapter $RR/dataset/public_samples/qwen3vl_cfg; do
  [ -e $f ] || fail "missing $f"; done
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$D" = 1 ] || [ -f $CACHE/index_nexar_reasoning.jsonl ] || [ "${FREE_GB:-0}" -ge 50 ] || fail "container disk has only ${FREE_GB} GB free (need >= 50)"
stage pip $OUT/logs/pip.log bash -c "$PY -c 'import bert_score' 2>/dev/null || /root/venv/bin/pip install -q bert-score" || fail pip

# 1 ---- the features cache (original manifest order) + the 117 new validation windows (val full)
mkdir -p $CACHE $CACHEX $VID
stage cache_old $OUT/logs/pull_cache.log env -u R1_REASONING_MANIFEST $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE || fail pull_cache
stage mk_val_windows $OUT/logs/mk_val_windows.log $PY -c "
import json
rows=[json.loads(l) for l in open('$NEWWIN') if l.strip()]
v=[r for r in rows if r['split']=='val']
open('$NEWVAL','w').write(''.join(json.dumps(r)+'\n' for r in v)); print(len(v),'validation windows of',len(rows))" || fail mk_val_windows
stage new_videos $OUT/logs/new_videos.log $PY -u r1_pod_get_videos.py --windows $NEWVAL --out $VID/train --prefix train/ --threads 16 || fail new_videos
stage new_frames $OUT/logs/new_frames.log env NEXAR_VIDEOS=$VID/train R1_TRAIN_DIR=$FR $PY -u r1_pod_cut_windows.py --windows $NEWVAL || fail new_frames
stage new_encode $OUT/logs/new_encode.log env R1_TRAIN_DIR=$FR R1_REASONING_MANIFEST=$EXTRA $PY -u r1_cache_features.py --source nexar_reasoning --splits val --out-dir $CACHEX || fail new_encode
if [ "$D" != 1 ]; then
  NXV=$($PY -c "import json; print(sum(1 for l in open('$EXTRA') if l.strip() and json.loads(l)['split']=='val'))")
  [ "$(wc -l < $CACHEX/index_nexar_reasoning.jsonl)" -eq "$NXV" ] || fail "extra cache has $(wc -l < $CACHEX/index_nexar_reasoning.jsonl) rows, expected $NXV"
  $PY - <<'PY' >> $ST 2>&1 || fail new_windows_sanity
import json, os
idx = [json.loads(l) for l in open(os.environ["CACHEX"] + "/index_nexar_reasoning.jsonl")]
p = [r["p_collision"] for r in idx]
print(f"[base] crash model on the {len(p)} new validation crash windows: mean P {sum(p) / len(p):.3f} (stage 1: 0.884)")
assert abs(sum(p) / len(p) - 0.884) < 0.01, "the encoder does not reproduce stage 1 on the new windows"
PY
fi
rm -rf $VID/train $FR 2>/dev/null
say "data ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

DATA="--sources nexar_reasoning --cache-dir $CACHE --facts-cards $CARDS --aug-texts $PARAX --val-texts none --p-teacher 0.4 --eff-bs 16 --micro-bs 4 --grad-ckpt"
P1S="$PY -u r1_train.py --phase 1 --init random --epochs 10 --patience 2 --select val_loss --lr-merger 1e-3 $DATA --drop-prealert --balance-tte --order reason_first"
P2="$PY -u r1_train.py --phase 2 --epochs 6 --patience 1 --select val_loss --lr-merger 2e-5 --lr-lora 2e-4 $DATA --drop-prealert --balance-tte --order verdict_first --eval-at 0.5,0.75,1.25,1.5 --checks-log-only"
EVO="$PY -u r1_eval_reasoning.py --facts-cards $CARDS --drop-prealert --cache-dir $CACHE --order verdict_first"
EVS="env R1_REASONING_MANIFEST=$V2 $PY -u r1_eval_reasoning.py --order verdict_first --facts-cards $CARDS2 --cache-dir $CACHES --drop-prealert"

# 2 ---- smoke tests (real model): the single stage, P2V with one mid-epoch check, a scoring pass on val full
stage smoke_p1 $OUT/logs/smoke_p1.log $P1S --seed 1 --epochs 1 --max-steps 4 --out-dir $OUT/smoke_P1 || fail smoke_p1
stage smoke_p2 $OUT/logs/smoke_p2.log $P2 --seed 1 --init-from $OUT/smoke_P1/best.pt --epochs 1 --max-steps 4 --eval-at 0.01 --out-dir $OUT/smoke_P2 || fail smoke_p2
stage smoke_eval $OUT/logs/smoke_eval.log $EVS --ckpt $OUT/smoke_P2/best.pt --split val --no-generate --limit 20 --max-ret 4 --out-dir $OUT/smoke_eval || fail smoke_eval
say "smoke tests passed"
if [ "$D" = 1 ]; then echo "  + [sync] $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 (background)"; else
  nohup $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 >> $OUT/logs/sync.log 2>&1 & SYNC_PID=$!; fi

# 3 ---- one seed: single stage -> P2V -> evaluations (validation only)
run() { N=$1; SEED=$2
  stage ${N}_p1 $OUT/logs/${N}_p1.log $P1S --seed $SEED --out-dir $OUT/$N/P1 || return 1
  stage ${N}_p2v $OUT/logs/${N}_p2v.log $P2 --seed $SEED --init-from $OUT/$N/P1/best.pt --out-dir $OUT/$N/P2V || return 1
  stage ev_${N}_val $OUT/logs/ev_${N}_val.log $EVS --ckpt $OUT/$N/P2V/best.pt --split val --no-generate --max-ret 50 --out-dir $OUT/eval/$N/val
  stage ev_${N}_train $OUT/logs/ev_${N}_train.log $EVO --ckpt $OUT/$N/P2V/best.pt --split train --sample 600 --sample-seed 0 --no-generate --max-ret 50 --out-dir $OUT/eval/$N/train600
  stage text_${N} $OUT/logs/text_${N}.log $EVO --ckpt $OUT/$N/P2V/best.pt --split val --max-ret 100 --n-blank 5 --train-threshold 600 --out-dir $OUT/eval/$N/val_text
  stage bert_${N} $OUT/logs/bert_${N}.log $PY -u r1_review_reasoning.py --run-dir $OUT/eval/$N/val_text
}
say "single-stage seeds S0 and S1 start in parallel"
run S0 0 & PA=$!
run S1 1 & PB=$!
wait $PA; RA=$?; wait $PB; RB=$?
[ $RA -eq 0 ] || say "seed S0 FAILED somewhere (see the logs)"
[ $RB -eq 0 ] || say "seed S1 FAILED somewhere (see the logs)"
[ $RA -eq 0 ] || [ $RB -eq 0 ] || fail "both seeds failed"

# 4 ---- push + verify + stop
[ "$D" = 1 ] || { kill $SYNC_PID 2>/dev/null; $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC >> $OUT/logs/sync.log 2>&1; }
if [ "$D" = 1 ]; then echo "  + push train logs, best.pt, results, logs to $R_CKPT/$HFDIR; verify sizes; stop the pod"; exit 0; fi
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); out = os.environ["OUT"]; rc, base = os.environ["R_CKPT"], os.environ["HFDIR"]
files = []
for ch in ("S0", "S1"):
    for st in ("P1", "P2V"):
        for f in ("train_log.jsonl", "best.pt"):
            p = f"{out}/{ch}/{st}/{f}"
            if os.path.exists(p):
                files.append((p, f"{ch}/{st}/{f}"))
for root, _, names in os.walk(f"{out}/eval"):
    for n in names:
        if n in ("predictions.jsonl", "metrics.json", "summary.md", "text_metrics.json", "report.md"):
            p = os.path.join(root, n); files.append((p, "eval/" + os.path.relpath(p, f"{out}/eval").replace(os.sep, "/")))
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/logs/*.log"))] + [(f"{out}/logs/base.status", "base.status")]
assert any(d.endswith("P2V/best.pt") for _, d in files), "no seed produced a P2V checkpoint"
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=rc, repo_type="model")
rem = {x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files], repo_type="model")}
ok = all(rem.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files)
print(len(files), "files;", "VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
