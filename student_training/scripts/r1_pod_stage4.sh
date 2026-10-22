#!/bin/bash
# Stage 4 of plan 2026-10-22_Plan-Stage4-Final-Seeds-rev3: five end-to-end seeds of the final recipe (the 2026-10-21 "control" chain of r1_pod_b4.sh, --input last),
# then the test sets ONCE (decision + reasoning text). Two pods:
#   POD=1  seeds 1, 2 (+ the test pass of seed 0 = HF b4_2026-10-21/CT/P2V/best.pt, not retrained)        POD=2  seeds 3, 4 (+ the red boxes of the 667 public test windows)
# Per seed (validation only while training): P1 -> P1fixR -> P2V with --seed k, best epoch of every stage = lowest val loss on val core (524);
#   evaluations as the control: val full scoring (641), train-600 sample, text on val core + BERTScore.
# Test, per seed, both halves (private 677, public 667): decision scores + generated answers on every window (--no-ref), and the loss / gap / retrieval metrics on
#   the windows with a teacher pass-1 reference (R1_TEST_MANIFEST*: private refs shipped with the code, public refs made on the PC from pod 2's boxes and fetched from HF).
# Pod 2 runs the public red boxes (tracks + crash-model attention) BEFORE its chains (the GPU cannot hold both) and pushes them to HF dataset teacher-test-2026-10/public/;
#   the PC then boxes the frames, runs the teacher pass 1 (Gemini Flex) and uploads the references to checkpoints/stage4_2026-10-22/refs/.
# Any major failure: the logs are pushed to HF and the pod is STOPPED. Success: final push + size check of every result file, then the pod stops itself.
# Needs a 32 GB GPU, 100 GB container disk, NO network volume. Expects in /root/r1 (copied from the PC): code, dataset/manifests/{r1_nexar_reasoning_windows,
# r1_nexar_reasoning_extra,r1_nexar_reasoning_v2,r1_nexar_reasoning_test,r1_nexar_reasoning_test_public,r1_nexar_reasoning_test_refs,test_manifest_private,
# test_manifest_public}.jsonl, dataset/public_samples/qwen3vl_cfg, outputs/a1_compress256/train/epoch_02/lora_adapter, outputs/teacher_dataset_2026-10/
# {paraphrases_relaxed,fact_cards_teacher}.jsonl, outputs/teacher_extra_2026-10/{windows_new,fact_cards_v2}.jsonl, outputs/teacher_test_2026-10/{windows,windows_public}.jsonl;
# pod 2 also third_party/ (YOLOPv2 weights).
#   POD=1 setsid nohup bash r1_pod_stage4.sh > /root/r1_s4/logs/s4.nohup 2>&1 < /dev/null &        status: /root/r1_s4/logs/s4.status        DRYRUN=1 prints the commands only.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
POD=${POD:-1}; [ "$POD" = 1 ] || [ "$POD" = 2 ] || { echo "POD must be 1 or 2"; exit 2; }
if [ "$POD" = 1 ]; then SEEDS=${SEEDS:-"1 2"}; TEST0=1; else SEEDS=${SEEDS:-"3 4"}; TEST0=0; fi
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts; OUT=${OUT:-/root/r1_s4}
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; R_TEST=eviatarO-org/teacher-test-2026-10
TOP=stage4_2026-10-22; HFDIR=$TOP/pod$POD; S0_HF=b4_2026-10-21/CT/P2V/best.pt
CACHE=$OUT/cache; CACHEX=$OUT/cache_extra; CACHES="$CACHE,$CACHEX"; CTP=$OUT/cache_test_private; CTU=$OUT/cache_test_public; BOX=$OUT/boxes_public
VID=/root/nexar; FR=/root/frames_extra; TV=/root/nexar/test; TF=/root/test_frames
MAN=$RR/dataset/manifests; W=$RR/outputs/teacher_dataset_2026-10; W2=$RR/outputs/teacher_extra_2026-10; TW=$RR/outputs/teacher_test_2026-10
PARAX=$W/paraphrases_relaxed.jsonl; CARDS=$W/fact_cards_teacher.jsonl; CARDS2=$W2/fact_cards_v2.jsonl
NEWWIN=$W2/windows_new.jsonl; NEWVAL=$OUT/windows_new_val.jsonl; V2=$MAN/r1_nexar_reasoning_v2.jsonl; EXTRA=$MAN/r1_nexar_reasoning_extra.jsonl
REFP=$MAN/r1_nexar_reasoning_test_refs.jsonl; REFU=$OUT/refs/r1_nexar_reasoning_test_public_refs.jsonl
export OUT CACHE CACHEX CTP CTU HFDIR R_CKPT R_TEST BOX TOP POD MAN
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
mkdir -p $OUT/logs; cd $SC
ST=$OUT/logs/s4.status; echo "start $(date -u) pod $POD seeds $SEEDS" >> $ST
SDIRS=$(for k in $SEEDS; do printf 'S%s,' $k; done)
SYNC="--subdirs ${SDIRS}eval,logs,. --patterns train_log.jsonl,best.pt,predictions.jsonl,metrics.json,summary.md,text_metrics.json,report.md,done.*,*.log,s4.status"
say() { echo "[s4-$POD] $* $(date -u)" | tee -a $ST; }
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
        S*_p1) [ -f $OUT/${n%_p1}/P1/best.pt ] && keep=1 ;;
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
for f in $NEWWIN $PARAX $CARDS $CARDS2 $V2 $EXTRA $MAN/r1_nexar_reasoning_windows.jsonl $MAN/r1_nexar_reasoning_test.jsonl $MAN/r1_nexar_reasoning_test_public.jsonl $REFP \
         $MAN/test_manifest_private.jsonl $MAN/test_manifest_public.jsonl $TW/windows.jsonl $TW/windows_public.jsonl \
         $RR/outputs/a1_compress256/train/epoch_02/lora_adapter $RR/dataset/public_samples/qwen3vl_cfg; do
  [ -e $f ] || fail "missing $f"; done
[ "$POD" = 1 ] || [ -f $RR/third_party/yolopv2_ref/weights/yolopv2.pt ] || fail "third_party/yolopv2_ref/weights/yolopv2.pt missing (copy third_party to /root/r1)"
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$D" = 1 ] || [ -f $CACHE/index_nexar_reasoning.jsonl ] || [ "${FREE_GB:-0}" -ge 80 ] || fail "container disk has only ${FREE_GB} GB free (need >= 80; 100 recommended)"
stage pip $OUT/logs/pip.log bash -c "$PY -c 'import bert_score, cv2' 2>/dev/null || /root/venv/bin/pip install -q bert-score opencv-python-headless" || fail pip
[ "$POD" = 1 ] || stage pip_boxes $OUT/logs/pip_boxes.log bash -c "$PY -c 'import trackers, supervision, matplotlib, scipy' 2>/dev/null || /root/venv/bin/pip install -q scipy supervision trackers matplotlib albumentations" || fail pip_boxes

# 1 ---- data: training cache (original windows), the 117 new validation windows, both test halves (+ the crash-model sanity check on each)
mkdir -p $CACHE $CACHEX $CTP $CTU $VID $TV $TF $OUT/refs
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
print(f"[s4] crash model on the {len(p)} new validation crash windows: mean P {sum(p) / len(p):.3f} (stage 1: 0.884)")
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
print(f"[s4] crash model on the {len(idx)} {h} test windows: AP {ap:.4f}  AUC {auc:.4f}  (known AP {known})")
assert abs(ap - known) < 0.01, f"the crash model does not reproduce its known {h}-test AP: windows / frames / adapter mismatch"
PY
  say "test data $H ready (crash-model sanity passed)"; }
test_data private test-private/ $TW/windows.jsonl $MAN/test_manifest_private.jsonl nexar_reasoning_test $CTP 0.9127 677
test_data public test-public/ $TW/windows_public.jsonl $MAN/test_manifest_public.jsonl nexar_reasoning_test_public $CTU 0.9096 667
[ "$D" = 1 ] || [ "$POD" = 2 ] || rm -rf $TV $TF                     # pod 2 keeps the public videos + frames for the boxes
say "data ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# 2 ---- pod 2: red boxes of the public test windows (tracks, then the crash-model attention; not overlapped: the attention model OOMs next to the trackers)
if [ "$POD" = 2 ]; then
  BX="env R1_PILOT_OUT=$BOX NEXAR_VIDEOS=$TV/public R1_TRAIN_DIR=$TF/public $PY -u r1_attention_boxes.py --windows-file $TW/windows_public.jsonl"
  boxes_all() {
    $BX --stage tracks --limit-videos 2 && $BX --stage attn --limit-videos 2 || { echo "boxes smoke failed"; return 1; }
    for K in 0 1 2; do $BX --stage tracks --shard $K/3 > $OUT/logs/tracks_$K.log 2>&1 & done; wait
    for K in 0 1 2; do $BX --stage attn --shard $K/3 > $OUT/logs/attn_$K.log 2>&1 & done; wait
    NT=$(ls $BOX/tracks | wc -l); NA=$(ls $BOX/attn | wc -l); echo "tracks $NT of 667 | attention $NA of 667"
    [ "$NT" -eq 667 ] && [ "$NA" -eq 667 ]; }
  stage boxes_public $OUT/logs/boxes_public.log boxes_all || say "public boxes FAILED (minor): no public text references; the public decision scores are unaffected"
  if [ "$D" = 1 ]; then echo "  + push public tracks / attention / windows to $R_TEST/public/ (one commit), verify sizes"; elif [ -f $OUT/done.boxes_public ]; then
    mkdir -p $OUT/box_push && (cd $BOX && tar czf $OUT/box_push/tracks.tar.gz tracks && tar czf $OUT/box_push/attn.tar.gz attn) && cp $TW/windows_public.jsonl $OUT/box_push/windows.jsonl
    stage push_boxes $OUT/logs/push_boxes.log $PY - <<'PY' || say "push of the public boxes FAILED (minor): retry by hand"
import os
from huggingface_hub import HfApi
a, d, rt = HfApi(), os.environ["OUT"] + "/box_push", os.environ["R_TEST"]
open(f"{d}/README.md", "w").write("""# public/ - boxes inputs for the Nexar PUBLIC test windows (stage 4, 2026-10-22)

667 public-test windows (one per video, ending at its TTE 0.5 / 1.0 / 1.5 s); same content and code as the private files at the repo root:
* `tracks.tar.gz` - per video `tracks/<vid>.json` (YOLOPv2 + Grounding-DINO-tiny, BoT-SORT).
* `attn.tar.gz` - per window `attn/<id>_hires.json` (crash-model score + crash-head attention per track).
* `windows.jsonl` - the window manifest. Test texts are used ONLY for final evaluation.
""")
a.upload_folder(folder_path=d, path_in_repo="public", repo_id=rt, repo_type="dataset", commit_message="stage 4: public test boxes inputs")
names = sorted(os.listdir(d))
rem = {x.path: x.size for x in a.get_paths_info(rt, [f"public/{n}" for n in names], repo_type="dataset")}
bad = [n for n in names if rem.get(f"public/{n}") != os.path.getsize(f"{d}/{n}")]
print("public boxes on HF:", names, "mismatches:", bad); assert not bad
PY
  fi
  [ "$D" = 1 ] || rm -rf $TV $TF
fi

# 3 ---- training / evaluation commands (the 2026-10-21 control chain, + --seed) and the smoke tests
DATA="--sources nexar_reasoning --cache-dir $CACHE --facts-cards $CARDS --aug-texts $PARAX --val-texts none --p-teacher 0.4 --eff-bs 16 --micro-bs 4 --grad-ckpt --input last"
P1="$PY -u r1_train.py --phase 1 --init random --epochs 10 --patience 2 --select val_loss --lr-merger 1e-3 $DATA"
P1F="$PY -u r1_train.py --phase 1 --init random --epochs 4 --patience 1 --select val_loss --lr-merger 3e-4 $DATA --drop-prealert --balance-tte --order reason_first"
P2="$PY -u r1_train.py --phase 2 --epochs 6 --patience 1 --select val_loss --lr-merger 2e-5 --lr-lora 2e-4 $DATA --drop-prealert --balance-tte --order verdict_first --eval-at 0.5,0.75,1.25,1.5 --checks-log-only"
EVO="$PY -u r1_eval_reasoning.py --facts-cards $CARDS --drop-prealert --cache-dir $CACHE --order verdict_first"
EVS="env R1_REASONING_MANIFEST=$V2 $PY -u r1_eval_reasoning.py --order verdict_first --facts-cards $CARDS2 --cache-dir $CACHES --drop-prealert"
EVT="$PY -u r1_eval_reasoning.py --order verdict_first --max-new 160"
stage smoke_p1 $OUT/logs/smoke_p1.log $P1 --seed 9 --epochs 1 --max-steps 4 --out-dir $OUT/smoke_P1 || fail smoke_p1
stage smoke_p2 $OUT/logs/smoke_p2.log $P2 --seed 9 --init-from $OUT/smoke_P1/best.pt --epochs 1 --max-steps 4 --eval-at 0.01 --out-dir $OUT/smoke_P2 || fail smoke_p2
stage smoke_eval $OUT/logs/smoke_eval.log $EVS --ckpt $OUT/smoke_P2/best.pt --split val --no-generate --limit 20 --max-ret 4 --out-dir $OUT/smoke_eval || fail smoke_eval
stage smoke_gen $OUT/logs/smoke_gen.log $EVT --source nexar_reasoning_test --no-ref --cache-dir $CTP --ckpt $OUT/smoke_P2/best.pt --limit 8 --out-dir $OUT/smoke_gen || fail smoke_gen
stage smoke_ground $OUT/logs/smoke_ground.log env R1_TEST_MANIFEST=$REFP $EVT --source nexar_reasoning_test --cache-dir $CTP --ckpt $OUT/smoke_P2/best.pt --no-generate --limit 8 --max-ret 4 --out-dir $OUT/smoke_ground || fail smoke_ground
say "smoke tests passed"
if [ "$D" = 1 ]; then echo "  + [sync] $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 (background, one commit per pass)"; else
  nohup $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 >> $OUT/logs/sync.log 2>&1 & SYNC_PID=$!; fi

# 4 ---- one seed: P1 -> P1fixR -> P2V (all with --seed k) -> validation evaluations (as the control)
chain() { N=S$1; K=$1
  stage ${N}_p1 $OUT/logs/${N}_p1.log $P1 --seed $K --out-dir $OUT/$N/P1 || return 1
  stage ${N}_p1fixr $OUT/logs/${N}_p1fixr.log $P1F --seed $K --init-from $OUT/$N/P1/best.pt --out-dir $OUT/$N/P1fixR || return 1
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
if [ "$TEST0" = 1 ]; then
  mkdir -p $OUT/S0/P2V
  stage s0_ckpt $OUT/logs/s0_ckpt.log $PY -c "
from huggingface_hub import hf_hub_download as d; import shutil
shutil.copy(d('$R_CKPT','$S0_HF',local_dir='$OUT/_hf'),'$OUT/S0/P2V/best.pt')" || fail s0_ckpt
fi

# 5 ---- test, once: every seed on both halves; the public references come from the PC (wait up to 60 min, then go on without them)
if [ "$D" = 1 ]; then echo "  + [refs] wait <= 60 min for $R_CKPT/$TOP/refs/$(basename $REFU) -> $REFU"; elif [ ! -s $REFU ]; then
  say "waiting up to 60 min for the public references on HF"
  for i in $(seq 1 60); do
    $PY -c "
from huggingface_hub import hf_hub_download as d; import shutil
shutil.copy(d('$R_CKPT','$TOP/refs/$(basename $REFU)',local_dir='$OUT/_hf'),'$REFU')" >> $OUT/logs/refs.log 2>&1 && break
    sleep 60; done
  [ -s $REFU ] && say "public references: $(wc -l < $REFU) windows" || say "no public references (minor): the public text metrics come from the PC only"
fi
test_seed() { N=S$1; CK=$OUT/$N/P2V/best.pt
  [ "$D" = 1 ] || [ -f $CK ] || { say "no checkpoint for $N: its test pass is skipped"; return 1; }
  stage gen_private_$N $OUT/logs/gen_private_$N.log $EVT --source nexar_reasoning_test --no-ref --cache-dir $CTP --ckpt $CK --out-dir $OUT/eval/$N/test_private
  stage gen_public_$N $OUT/logs/gen_public_$N.log $EVT --source nexar_reasoning_test_public --no-ref --cache-dir $CTU --ckpt $CK --out-dir $OUT/eval/$N/test_public
  stage ground_private_$N $OUT/logs/ground_private_$N.log env R1_TEST_MANIFEST=$REFP $EVT --source nexar_reasoning_test --cache-dir $CTP --ckpt $CK --no-generate --max-ret 50 --out-dir $OUT/eval/$N/ground_private
  if [ "$D" = 1 ] || [ -s $REFU ]; then
    stage ground_public_$N $OUT/logs/ground_public_$N.log env R1_TEST_MANIFEST= R1_TEST_PUBLIC_MANIFEST=$REFU $EVT --source nexar_reasoning_test_public --cache-dir $CTU --ckpt $CK --no-generate --max-ret 50 --out-dir $OUT/eval/$N/ground_public
  fi
}
PIDS=""
for k in $SEEDS; do test_seed $k & PIDS="$PIDS $!"; done
for p in $PIDS; do wait $p || true; done
[ "$TEST0" = 1 ] && test_seed 0
say "test pass done"

# 6 ---- final push (one commit) + size check of every result file, then stop
[ "$D" = 1 ] && { echo "  + final push of train logs, best.pt of every stage, eval/, logs to $R_CKPT/$HFDIR/live (one commit); verify sizes; stop the pod"; exit 0; }
kill $SYNC_PID 2>/dev/null; sleep 25
for i in 1 2 3; do $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC >> $OUT/logs/sync.log 2>&1; grep -q "commit of .* failed" <(tail -1 $OUT/logs/sync.log) || break; sleep 120; done
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import fnmatch, os, sys
from huggingface_hub import HfApi
a = HfApi(); out, rc, base = os.environ["OUT"], os.environ["R_CKPT"], os.environ["HFDIR"] + "/live"
pats = ("train_log.jsonl", "best.pt", "predictions.jsonl", "metrics.json", "summary.md", "text_metrics.json", "report.md")
files = []
tops = {"eval"} | {d for d in os.listdir(out) if d.startswith("S") and d != "S0" and os.path.isdir(f"{out}/{d}")}   # S0 = seed 0's checkpoint, already on HF (b4_2026-10-21)
for root, _, names in os.walk(out):
    rel = os.path.relpath(root, out).replace(os.sep, "/")
    if rel.split("/")[0] in tops:
        files += [(f"{root}/{n}", f"{rel}/{n}") for n in names if any(fnmatch.fnmatch(n, p) for p in pats)]
assert any(d.endswith("P2V/best.pt") for _, d in files), "no seed produced a P2V checkpoint"
rem = {}
for i in range(0, len(files), 200):
    rem.update({x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files[i:i + 200]], repo_type="model")})
bad = [d for s, d in files if rem.get(f"{base}/{d}") != os.path.getsize(s)]
print(f"[s4] final check: {len(files)} result files, {len(bad)} missing / different on HF", bad[:10]); sys.exit(1 if bad else 0)
PY
say "verified on HF; stopping the pod"
stop_pod
