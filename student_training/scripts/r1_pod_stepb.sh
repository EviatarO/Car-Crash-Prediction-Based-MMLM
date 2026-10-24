#!/bin/bash
# Step B of plan 2026-10-24_Plan-StepB-P2V-Recipe-Seeds: four NEW end-to-end seeds of the P2V recipe (2026-10-14/15: r1_pod_phase1aug.sh -> r1_pod_phase2f.sh), then the same
# validation and test evaluations as stage 4 (r1_pod_stage4.sh), so the P2V recipe can be compared seed-for-seed with the final recipe. Two pods:
#   POD=1  seeds 1, 2        POD=2  seeds 3, 4        (P2V itself = seed 0 of this recipe, already scored on everything; it is NOT retrained and NOT counted in the decision)
# Per seed: P1aug (projector from random; original + frame-shifted windows; teacher text p 0.4 else a JUDGED rewording; <= 15 ep, patience 2; validation incl. the shifted copies)
#   -> P1fixR (projector from P1aug; original windows; pre-alert dropped; TTE-balanced; reason-first; <= 4 ep, patience 1, lr 3e-4)
#   -> P2V (projector + LoRA from P1fixR; verdict-first; <= 6 ep, patience 1, lr 2e-5 / 2e-4); best epoch of every stage = lowest validation loss.
# Data: the original cache from HF; the 12,286 frame-shifted windows rebuilt from the Nexar videos (their old cache was deleted), checked first on 24 control windows
#   re-cut at shift 0 (crash-model P must equal the cached one); the 117 extra validation windows (val full); both test halves with the crash-model sanity check.
# Test references (private 431, public 460) are shipped with the code. Any major failure: logs pushed to HF, pod STOPPED. Success: final push + size check, then the pod stops itself.
# Needs a 32 GB GPU, 150 GB container disk, NO network volume. Expects in /root/r1 (copied from the PC): code, dataset/manifests/{r1_nexar_reasoning_windows,
# r1_nexar_reasoning_shifted,r1_nexar_reasoning_control,r1_nexar_reasoning_extra,r1_nexar_reasoning_v2,r1_nexar_reasoning_test,r1_nexar_reasoning_test_public,
# r1_nexar_reasoning_test_refs,r1_nexar_reasoning_test_public_refs,test_manifest_private,test_manifest_public}.jsonl, dataset/public_samples/qwen3vl_cfg,
# outputs/a1_compress256/train/epoch_02/lora_adapter, outputs/teacher_dataset_2026-10/{paraphrases_judged,fact_cards_teacher,windows_shifted,windows_control}.jsonl,
# outputs/teacher_extra_2026-10/{windows_new,fact_cards_v2}.jsonl, outputs/teacher_test_2026-10/{windows,windows_public}.jsonl.
#   POD=1 setsid nohup bash r1_pod_stepb.sh > /root/r1_sb/logs/sb.nohup 2>&1 < /dev/null &        status: /root/r1_sb/logs/sb.status        DRYRUN=1 prints the commands only.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
POD=${POD:-1}; [ "$POD" = 1 ] || [ "$POD" = 2 ] || { echo "POD must be 1 or 2"; exit 2; }
if [ "$POD" = 1 ]; then SEEDS=${SEEDS:-"1 2"}; else SEEDS=${SEEDS:-"3 4"}; fi
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts; OUT=${OUT:-/root/r1_sb}
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints
TOP=stepb_2026-10-24; HFDIR=$TOP/pod$POD
CACHE=$OUT/cache; CACHEX=$OUT/cache_extra; CACHES="$CACHE,$CACHEX"; CTP=$OUT/cache_test_private; CTU=$OUT/cache_test_public
VID=/root/nexar; FR=/root/frames_extra; FS=/root/frames_shifted; TV=/root/nexar/test; TF=/root/test_frames
MAN=$RR/dataset/manifests; W=$RR/outputs/teacher_dataset_2026-10; W2=$RR/outputs/teacher_extra_2026-10; TW=$RR/outputs/teacher_test_2026-10
PARA=$W/paraphrases_judged.jsonl; CARDS=$W/fact_cards_teacher.jsonl; CARDS2=$W2/fact_cards_v2.jsonl
NEWWIN=$W2/windows_new.jsonl; NEWVAL=$OUT/windows_new_val.jsonl; V2=$MAN/r1_nexar_reasoning_v2.jsonl; EXTRA=$MAN/r1_nexar_reasoning_extra.jsonl
REFP=$MAN/r1_nexar_reasoning_test_refs.jsonl; REFU=$MAN/r1_nexar_reasoning_test_public_refs.jsonl
export OUT CACHE CACHEX CTP CTU HFDIR R_CKPT TOP POD MAN
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
mkdir -p $OUT/logs; cd $SC
ST=$OUT/logs/sb.status; echo "start $(date -u) pod $POD seeds $SEEDS" >> $ST
SDIRS=$(for k in $SEEDS; do printf 'S%s,' $k; done)
SYNC="--subdirs ${SDIRS}eval,logs,. --patterns train_log.jsonl,best.pt,predictions.jsonl,metrics.json,summary.md,text_metrics.json,report.md,done.*,*.log,sb.status"
say() { echo "[sb-$POD] $* $(date -u)" | tee -a $ST; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
fail() {                                                         # user rule: never leave a failed pod idle
  say "FAILED (major): $* - pushing the logs to HF, then stopping the pod"
  [ "$D" = 1 ] && return 0
  sleep 25                                                          # r1_hf_sync skips files younger than 20 s (the status line just written)
  $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/failed $SYNC >> $OUT/logs/sync.log 2>&1 || true
  stop_pod; exit 1; }
stage() { n=$1; lg=$2; shift 2
  if [ "$D" = 1 ]; then echo "  + [$n] $*"; return 0; fi
  [ -f "$OUT/done.$n" ] && { say "skip $n (done)"; return 0; }
  say "start $n"; "$@" > "$lg" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch "$OUT/done.$n"; say "done $n"; else say "FAILED $n (rc=$rc), log $lg"; fi; return $rc; }

# 0 ---- restore (a fresh pod after a lost one) + checks
if [ "$D" = 1 ]; then echo "  + [restore] $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live"; else
  $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live >> $OUT/logs/restore.log 2>&1 || say "nothing to restore from HF (fresh start)"
  say "restore: $(ls $OUT/done.* 2>/dev/null | wc -l) finished stages found"
  # a fresh container has none of the data the data markers stand for: keep a marker only when its output came back from HF (training: best.pt; evaluations: metrics.json)
  if [ ! -f $CACHE/index_nexar_reasoning.jsonl ]; then
    for m in $OUT/done.*; do [ -e "$m" ] || continue; n=${m##*/done.}; keep=0
      case $n in
        S*_p1aug) [ -f $OUT/${n%_p1aug}/P1aug/best.pt ] && keep=1 ;;
        S*_p1fixr) [ -f $OUT/${n%_p1fixr}/P1fixR/best.pt ] && keep=1 ;;
        S*_p2v) [ -f $OUT/${n%_p2v}/P2V/best.pt ] && keep=1 ;;
        ev_*|text_*|bert_*|gen_*|ground_*) d=${n#*_}; t=${n%%_*}; s=${d##*_}; h=${d%_*}
          case $t in ev) [ -f $OUT/eval/$s/$h/metrics.json ] && keep=1 ;; text) [ -f $OUT/eval/$d/val_text/metrics.json ] && keep=1 ;;
                     bert) [ -f $OUT/eval/$d/val_text/text_metrics.json ] && keep=1 ;;
                     gen) [ -f $OUT/eval/$s/test_$h/metrics.json ] && keep=1 ;; ground) [ -f $OUT/eval/$s/ground_$h/metrics.json ] && keep=1 ;; esac ;;
      esac
      [ $keep = 1 ] || rm -f "$m"; done
    say "fresh container: data stages will be rebuilt ($(ls $OUT/done.* 2>/dev/null | wc -l) stage markers kept)"
  fi; fi
for f in $NEWWIN $PARA $CARDS $CARDS2 $V2 $EXTRA $W/windows_shifted.jsonl $W/windows_control.jsonl $MAN/r1_nexar_reasoning_windows.jsonl \
         $MAN/r1_nexar_reasoning_shifted.jsonl $MAN/r1_nexar_reasoning_control.jsonl $MAN/r1_nexar_reasoning_test.jsonl $MAN/r1_nexar_reasoning_test_public.jsonl $REFP $REFU \
         $MAN/test_manifest_private.jsonl $MAN/test_manifest_public.jsonl $TW/windows.jsonl $TW/windows_public.jsonl \
         $RR/outputs/a1_compress256/train/epoch_02/lora_adapter $RR/dataset/public_samples/qwen3vl_cfg; do
  [ -e $f ] || fail "missing $f"; done
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$D" = 1 ] || [ -f $CACHE/index_nexar_reasoning_shifted.jsonl ] || [ "${FREE_GB:-0}" -ge 140 ] || fail "container disk has only ${FREE_GB} GB free (need >= 140; 150 recommended)"
stage pip $OUT/logs/pip.log bash -c "$PY -c 'import bert_score, cv2' 2>/dev/null || /root/venv/bin/pip install -q bert-score opencv-python-headless" || fail pip

# 1 ---- data: original cache, the frame-shifted windows (rebuilt), the 117 extra validation windows, both test halves (+ the sanity checks)
mkdir -p $CACHE $CACHEX $CTP $CTU $VID $TV $TF $OUT/cache_control
stage cache_old $OUT/logs/pull_cache.log env -u R1_REASONING_MANIFEST $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE || fail pull_cache
# shifted windows (r1_pod_phase1aug.sh step 1): videos -> 16 frames per window (6 processes) -> 24 control windows re-cut at shift 0 must reproduce the cached crash-model P -> encode
stage shift_windows $OUT/logs/shift_windows.log bash -c "cat $W/windows_shifted.jsonl $W/windows_control.jsonl > $OUT/windows_shift_all.jsonl && wc -l < $OUT/windows_shift_all.jsonl" || fail shift_windows
stage shift_videos $OUT/logs/shift_videos.log $PY -u r1_pod_get_videos.py --windows $OUT/windows_shift_all.jsonl --out $VID/train --prefix train/ --threads 16 || fail shift_videos
cut_all() { for k in 0 1 2 3 4 5; do env NEXAR_VIDEOS=$VID/train R1_TRAIN_DIR=$FS $PY -u r1_pod_cut_windows.py --windows $OUT/windows_shift_all.jsonl --shard $k/6 > $OUT/logs/shift_cut_$k.log 2>&1 & done; wait
  E=$(wc -l < $OUT/windows_shift_all.jsonl); G=$(ls $FS | wc -l); echo "cut $G folders of $E"; [ "$G" -eq "$E" ]; }
stage shift_cut $OUT/logs/shift_cut.log cut_all || fail "shift_cut (folders missing; see logs/shift_cut_*.log)"
stage shift_control $OUT/logs/shift_control.log env R1_TRAIN_DIR=$FS $PY -u r1_cache_features.py --source nexar_reasoning_control --splits train,val --out-dir $OUT/cache_control || fail shift_control
if [ "$D" != 1 ]; then
  $PY - <<'PY' >> $ST 2>&1 || fail shift_control_mismatch
import json, os
c, o = os.environ["CACHE"], os.environ["OUT"]
new = {r["id"][:-2]: r["p_collision"] for r in map(json.loads, open(f"{o}/cache_control/index_nexar_reasoning_control.jsonl"))}
old = {r["id"]: r["p_collision"] for r in map(json.loads, open(f"{c}/index_nexar_reasoning.jsonl"))}
d = max(abs(new[k] - old[k]) for k in new)
print(f"[sb] control: {len(new)} windows re-cut at shift 0, max |P new - P cached| = {d:.5f} (limit 0.004)")
assert len(new) == 24 and d <= 0.004, "re-cut windows do not reproduce the cached crash-model P"
PY
fi
stage shift_encode $OUT/logs/shift_encode.log env R1_TRAIN_DIR=$FS $PY -u r1_cache_features.py --source nexar_reasoning_shifted --splits train,val --out-dir $CACHE || fail shift_encode
[ "$D" = 1 ] || [ "$(wc -l < $CACHE/index_nexar_reasoning_shifted.jsonl)" -ge 12000 ] || fail "shifted cache incomplete: $(wc -l < $CACHE/index_nexar_reasoning_shifted.jsonl) rows"
[ "$D" = 1 ] || { rm -rf $FS $VID/train; say "shifted cache ready ($(wc -l < $CACHE/index_nexar_reasoning_shifted.jsonl) windows); frames + videos deleted; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"; }
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
print(f"[sb] crash model on the {len(p)} new validation crash windows: mean P {sum(p) / len(p):.3f} (stage 1: 0.884)")
assert abs(sum(p) / len(p) - 0.884) < 0.01, "the encoder does not reproduce stage 1 on the new windows"
PY
fi
[ "$D" = 1 ] || rm -rf $VID/train $FR 2>/dev/null

# test half H: videos -> 16 frames per window -> features cache -> the crash model must reproduce its known AP (frames / windows / adapter check)
test_data() { H=$1; PREFIX=$2; WIN=$3; TMAN=$4; SRC=$5; CDIR=$6; KNOWN=$7; N=$8
  stage videos_$H $OUT/logs/videos_$H.log $PY -u r1_pod_get_videos.py --windows $WIN --out $TV/$H --prefix $PREFIX --threads 16 || fail videos_$H
  [ "$D" = 1 ] || [ "$(ls $TV/$H/*.mp4 | wc -l)" -ge $N ] || fail "videos $H: $(ls $TV/$H/*.mp4 | wc -l) of $N"
  stage frames_$H $OUT/logs/frames_$H.log $PY -u extract_highres_frames_test.py --manifest_in $TMAN --dst_root $TF/$H --src_videos $TV/$H \
        --manifest_out $OUT/test_manifest_${H}_hires.jsonl --failures_json $OUT/extract_failures_$H.json || fail frames_$H
  [ "$D" = 1 ] || [ "$(find $TF/$H -name 'frame_*.jpg' | wc -l)" -eq $((N * 16)) ] || fail "frames $H: $(find $TF/$H -name 'frame_*.jpg' | wc -l) of $((N * 16))"
  stage encode_$H $OUT/logs/encode_$H.log env R1_TRAIN_DIR=$TF/$H $PY -u r1_cache_features.py --source $SRC --splits test --out-dir $CDIR || fail encode_$H
  [ "$D" = 1 ] && { echo "  + crash-model sanity on $H: AP within 0.01 of $KNOWN"; return 0; }
  [ "$(wc -l < $CDIR/index_$SRC.jsonl)" -eq $N ] || fail "encode $H: $(wc -l < $CDIR/index_$SRC.jsonl) of $N"
  H=$H SRC=$SRC CDIR=$CDIR KNOWN=$KNOWN $PY - <<'PY' >> $ST 2>&1 || fail crash_model_sanity_$H
import json, os
from sklearn.metrics import average_precision_score, roc_auc_score
h, src, c, known = os.environ["H"], os.environ["SRC"], os.environ["CDIR"], float(os.environ["KNOWN"])
idx = [json.loads(l) for l in open(f"{c}/index_{src}.jsonl")]
man = {m["id"]: m for m in map(json.loads, open(f"{os.environ['MAN']}/r1_nexar_reasoning_{'test' if h == 'private' else 'test_public'}.jsonl"))}
y = [man[r["id"]]["label"] for r in idx]; p = [r["p_collision"] for r in idx]
ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
print(f"[sb] crash model on the {len(idx)} {h} test windows: AP {ap:.4f}  AUC {auc:.4f}  (known AP {known})")
assert abs(ap - known) < 0.01, f"the crash model does not reproduce its known {h}-test AP: windows / frames / adapter mismatch"
PY
  say "test data $H ready (crash-model sanity passed)"; }
test_data private test-private/ $TW/windows.jsonl $MAN/test_manifest_private.jsonl nexar_reasoning_test $CTP 0.9127 677
test_data public test-public/ $TW/windows_public.jsonl $MAN/test_manifest_public.jsonl nexar_reasoning_test_public $CTU 0.9096 667
[ "$D" = 1 ] || rm -rf $TV $TF
say "data ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# 2 ---- the P2V recipe's commands (r1_pod_phase1aug.sh COMMON + P1aug flags; r1_pod_phase2f.sh DATA / P1 / P2), + --seed; the evaluations of stage 4
P1A="$PY -u r1_train.py --phase 1 --sources nexar_reasoning --cache-dir $CACHE --init random --epochs 15 --patience 2 --select val_loss --eff-bs 16 --micro-bs 4 --lr-merger 1e-3 --grad-ckpt --val-texts none --val-shifts --aug-texts $PARA --aug-shifts --p-teacher 0.4"
DATA="--sources nexar_reasoning --cache-dir $CACHE --facts-cards $CARDS --drop-prealert --balance-tte --aug-texts $PARA --val-texts none --p-teacher 0.4 --select val_loss --eff-bs 16 --micro-bs 4 --grad-ckpt"
P1F="$PY -u r1_train.py --phase 1 --init random --epochs 4 --patience 1 --lr-merger 3e-4 $DATA --order reason_first"
P2="$PY -u r1_train.py --phase 2 --epochs 6 --patience 1 --lr-merger 2e-5 --lr-lora 2e-4 $DATA --order verdict_first"
EVO="$PY -u r1_eval_reasoning.py --facts-cards $CARDS --drop-prealert --cache-dir $CACHE --order verdict_first"
EVS="env R1_REASONING_MANIFEST=$V2 $PY -u r1_eval_reasoning.py --order verdict_first --facts-cards $CARDS2 --cache-dir $CACHES --drop-prealert"
EVT="$PY -u r1_eval_reasoning.py --order verdict_first --max-new 160"
stage smoke_p1aug $OUT/logs/smoke_p1aug.log $P1A --seed 9 --epochs 1 --max-steps 4 --out-dir $OUT/smoke_P1aug || fail smoke_p1aug
stage smoke_p1fixr $OUT/logs/smoke_p1fixr.log $P1F --seed 9 --init-from $OUT/smoke_P1aug/best.pt --epochs 1 --max-steps 4 --out-dir $OUT/smoke_P1fixR || fail smoke_p1fixr
stage smoke_p2 $OUT/logs/smoke_p2.log $P2 --seed 9 --init-from $OUT/smoke_P1fixR/best.pt --epochs 1 --max-steps 4 --out-dir $OUT/smoke_P2 || fail smoke_p2
stage smoke_eval $OUT/logs/smoke_eval.log $EVS --ckpt $OUT/smoke_P2/best.pt --split val --no-generate --limit 20 --max-ret 4 --out-dir $OUT/smoke_eval || fail smoke_eval
stage smoke_gen $OUT/logs/smoke_gen.log $EVT --source nexar_reasoning_test --no-ref --cache-dir $CTP --ckpt $OUT/smoke_P2/best.pt --limit 8 --out-dir $OUT/smoke_gen || fail smoke_gen
stage smoke_ground $OUT/logs/smoke_ground.log env R1_TEST_PUBLIC_MANIFEST=$REFU $EVT --source nexar_reasoning_test_public --cache-dir $CTU --ckpt $OUT/smoke_P2/best.pt --no-generate --limit 8 --max-ret 4 --out-dir $OUT/smoke_ground || fail smoke_ground
say "smoke tests passed"
if [ "$D" = 1 ]; then echo "  + [sync] $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 (background, one commit per pass)"; else
  nohup $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 >> $OUT/logs/sync.log 2>&1 & SYNC_PID=$!; fi

# 3 ---- one seed: P1aug -> P1fixR -> P2V (all with --seed k) -> validation evaluations (as stage 4)
chain() { N=S$1; K=$1
  stage ${N}_p1aug $OUT/logs/${N}_p1aug.log $P1A --seed $K --out-dir $OUT/$N/P1aug || return 1
  stage ${N}_p1fixr $OUT/logs/${N}_p1fixr.log $P1F --seed $K --init-from $OUT/$N/P1aug/best.pt --out-dir $OUT/$N/P1fixR || return 1
  stage ${N}_p2v $OUT/logs/${N}_p2v.log $P2 --seed $K --init-from $OUT/$N/P1fixR/best.pt --out-dir $OUT/$N/P2V || return 1
  stage ev_val_$N $OUT/logs/ev_val_$N.log $EVS --ckpt $OUT/$N/P2V/best.pt --split val --no-generate --max-ret 50 --out-dir $OUT/eval/$N/val
  stage ev_train600_$N $OUT/logs/ev_train600_$N.log $EVO --ckpt $OUT/$N/P2V/best.pt --split train --sample 600 --sample-seed 0 --no-generate --max-ret 50 --out-dir $OUT/eval/$N/train600
  stage text_$N $OUT/logs/text_$N.log $EVO --ckpt $OUT/$N/P2V/best.pt --split val --max-ret 100 --n-blank 5 --train-threshold 600 --out-dir $OUT/eval/$N/val_text
  stage bert_$N $OUT/logs/bert_$N.log $PY -u r1_review_reasoning.py --run-dir $OUT/eval/$N/val_text
}
PIDS=""
for k in $SEEDS; do chain $k & PIDS="$PIDS $!"; done
say "chains of seeds $SEEDS started in parallel"
NOK=0; for p in $PIDS; do wait $p && NOK=$((NOK + 1)) || say "a chain FAILED (minor; see the logs); its finished parts are kept"; done
[ "$D" = 1 ] || [ $NOK -gt 0 ] || fail "no chain finished"

# 4 ---- test, once: every seed on both halves (references shipped with the code)
test_seed() { N=S$1; CK=$OUT/$N/P2V/best.pt
  [ "$D" = 1 ] || [ -f $CK ] || { say "no checkpoint for $N: its test pass is skipped"; return 1; }
  stage gen_private_$N $OUT/logs/gen_private_$N.log $EVT --source nexar_reasoning_test --no-ref --cache-dir $CTP --ckpt $CK --out-dir $OUT/eval/$N/test_private
  stage gen_public_$N $OUT/logs/gen_public_$N.log $EVT --source nexar_reasoning_test_public --no-ref --cache-dir $CTU --ckpt $CK --out-dir $OUT/eval/$N/test_public
  stage ground_private_$N $OUT/logs/ground_private_$N.log env R1_TEST_MANIFEST=$REFP $EVT --source nexar_reasoning_test --cache-dir $CTP --ckpt $CK --no-generate --max-ret 50 --out-dir $OUT/eval/$N/ground_private
  stage ground_public_$N $OUT/logs/ground_public_$N.log env R1_TEST_PUBLIC_MANIFEST=$REFU $EVT --source nexar_reasoning_test_public --cache-dir $CTU --ckpt $CK --no-generate --max-ret 50 --out-dir $OUT/eval/$N/ground_public
}
PIDS=""
for k in $SEEDS; do test_seed $k & PIDS="$PIDS $!"; done
for p in $PIDS; do wait $p || true; done
say "test pass done"

# 5 ---- final push (one commit) + size check of every result file, then stop
[ "$D" = 1 ] && { echo "  + final push of train logs, best.pt of every stage, eval/, logs to $R_CKPT/$HFDIR/live (one commit); verify sizes; stop the pod"; exit 0; }
kill $SYNC_PID 2>/dev/null; sleep 25
for i in 1 2 3; do $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC >> $OUT/logs/sync.log 2>&1; grep -q "commit of .* failed" <(tail -1 $OUT/logs/sync.log) || break; sleep 120; done
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import fnmatch, os, sys
from huggingface_hub import HfApi
a = HfApi(); out, rc, base = os.environ["OUT"], os.environ["R_CKPT"], os.environ["HFDIR"] + "/live"
pats = ("train_log.jsonl", "best.pt", "predictions.jsonl", "metrics.json", "summary.md", "text_metrics.json", "report.md")
files = []
tops = {"eval"} | {d for d in os.listdir(out) if d.startswith("S") and os.path.isdir(f"{out}/{d}")}
for root, _, names in os.walk(out):
    rel = os.path.relpath(root, out).replace(os.sep, "/")
    if rel.split("/")[0] in tops:
        files += [(f"{root}/{n}", f"{rel}/{n}") for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
assert any(d.endswith("P2V/best.pt") for _, d in files), "no seed produced a P2V checkpoint"
rem = {}
for i in range(0, len(files), 200):
    rem.update({x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files[i:i + 200]], repo_type="model")})
bad = [d for s, d in files if rem.get(f"{base}/{d}") != os.path.getsize(s)]
print(f"[sb] final check: {len(files)} result files, {len(bad)} missing / different on HF", bad[:10]); sys.exit(1 if bad else 0)
PY
say "verified on HF; stopping the pod"
stop_pod
