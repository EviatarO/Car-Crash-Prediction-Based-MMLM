#!/bin/bash
# Training chain B on the grown data (plan 2026-10-19_Plan-Extra-TTE-Windows-Retrain-rev2: chain B only, one seed; chain A dropped by the user). Unattended; stops the pod ONLY after every result file is verified
# byte-identical on HF; any major failure leaves the pod running.
#   chain B : today's windows + the extra windows (TTE 0.75 / 1.25 s); the new recipe; every epoch is saved; Phase 1 stages run all epochs, P2V-v2 stops early
#   after 3 epochs without a validation-loss improvement:
#     P1aug-v2  (projector only, random start, 12 epochs, merger lr 5e-4, teacher text + judged rewordings, p_teacher 0.4)
#     P1fixR-v2 (reason-first, pre-alert dropped, TTE-balanced positives, 4 epochs, merger lr 1.5e-4, from P1aug-v2 best val loss)
#     P2V-v2    (verdict-first, projector + LoRA, up to 6 epochs, patience 3, merger lr 1e-5, LoRA lr 1e-4, from P1fixR-v2 best val loss)
#   0 checks        1 old features cache (HF)     2 the 874 new windows: videos -> frames -> encode     3 the 1,344 test windows: videos -> frames -> encode + crash-model sanity
#   4 smoke tests (real model, 4 steps per stage, scoring pass)     5 chain B     6 top-3 epochs of P2V-v2 by VALIDATION AUC (val full), scoring-only
#   evaluations of them + of today's P2V on val full / private test / public test; text evaluations (val full, train sample 600, BERTScore) of the best epoch
#   7 push to HF checkpoints/chains_2026-10-19/, verify sizes, stop the pod.
# Needs a pod with a 32 GB GPU and >= 60 GB free container disk (80 GB recommended). Expects in /root/r1 (copied from the PC): code, dataset/manifests/{r1_nexar_reasoning_windows,
# r1_nexar_reasoning_extra,r1_nexar_reasoning_v2,r1_nexar_reasoning_test,r1_nexar_reasoning_test_public,test_manifest_private,test_manifest_public}.jsonl, dataset/public_samples/qwen3vl_cfg,
# outputs/a1_compress256/train/epoch_02/lora_adapter, outputs/teacher_extra_2026-10/{windows_new,paraphrases_judged_v2,fact_cards_v2}.jsonl.
#   setsid nohup bash r1_pod_chains.sh > /root/r1_chains/logs/chains.nohup 2>&1 < /dev/null &            status: /root/r1_chains/logs/chains.status
# DRYRUN=1 prints the commands without running them (no GPU, no network).
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts; OUT=${OUT:-/root/r1_chains}
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; HFDIR=chains_2026-10-19
CACHE=$OUT/cache; CACHEX=$OUT/cache_extra; CACHET=$OUT/cache_test; CACHES="$CACHE,$CACHEX"
VID=/root/nexar; FR=/root/frames_extra; FRT=/root/test_frames; FRP=/root/test_frames_public
MAN=$RR/dataset/manifests; W2=$RR/outputs/teacher_extra_2026-10
NEWWIN=$W2/windows_new.jsonl; PARA2=$W2/paraphrases_judged_v2.jsonl; CARDS2=$W2/fact_cards_v2.jsonl
V2=$MAN/r1_nexar_reasoning_v2.jsonl; EXTRA=$MAN/r1_nexar_reasoning_extra.jsonl
export OUT CACHE CACHEX CACHET HFDIR R_CKPT
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
mkdir -p $OUT/logs; cd $SC
ST=$OUT/logs/chains.status; echo "start $(date -u)" > $ST
say() { echo "[chains] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED (major): $* - pod left running"; [ "$D" = 1 ] || exit 1; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
# stage NAME LOG CMD... : skipped when done.NAME exists (resumable); output to LOG
stage() { n=$1; lg=$2; shift 2
  if [ "$D" = 1 ]; then echo "  + [$n] $*"; return 0; fi
  [ -f "$OUT/done.$n" ] && { say "skip $n (done)"; return 0; }
  say "start $n"; "$@" > "$lg" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch "$OUT/done.$n"; say "done $n"; else say "FAILED $n (rc=$rc), log $lg"; fi; return $rc; }
withV2() { R1_REASONING_MANIFEST=$V2 "$@"; }

# 0 ---- restore (a fresh pod after a lost one): finished training stages and their checkpoints come back from HF (the data preparation below is always redone)
if [ "$D" = 1 ]; then echo "  + [restore] $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live"; else
  $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live >> $OUT/logs/restore.log 2>&1 || say "restore from HF failed or nothing to restore (see logs/restore.log); continuing from scratch"
  say "restore: $(ls $OUT/done.B_* 2>/dev/null | wc -l) finished training stages found on HF"; fi

# 0 ---- checks
for f in $NEWWIN $PARA2 $CARDS2 $V2 $EXTRA $MAN/r1_nexar_reasoning_windows.jsonl $MAN/r1_nexar_reasoning_test.jsonl $MAN/r1_nexar_reasoning_test_public.jsonl \
         $MAN/test_manifest_private.jsonl $MAN/test_manifest_public.jsonl $RR/outputs/a1_compress256/train/epoch_02/lora_adapter $RR/dataset/public_samples/qwen3vl_cfg; do
  [ -e $f ] || fail "missing $f"; done
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$D" = 1 ] || [ "${FREE_GB:-0}" -ge 60 ] || fail "container disk has only ${FREE_GB} GB free (need >= 60; 80 recommended)"
[ "$D" = 1 ] || [ "$(wc -l < $NEWWIN)" -eq 874 ] || fail "windows_new.jsonl has $(wc -l < $NEWWIN) rows, expected 874"
stage pip $OUT/logs/pip.log bash -c "$PY -c 'import bert_score' 2>/dev/null || /root/venv/bin/pip install -q bert-score" || fail pip

# 1 ---- the old features cache (4,169 windows; the ORIGINAL manifest defines its row order, so the override is unset here)
mkdir -p $CACHE $CACHEX $CACHET $VID
stage cache_old $OUT/logs/pull_cache.log env -u R1_REASONING_MANIFEST $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE || fail pull_cache

# 2 ---- the 874 new windows: videos (HF train/) -> frames -> encode with the frozen V-JEPA2 / A1 encoder
stage new_videos $OUT/logs/new_videos.log $PY -u r1_pod_get_videos.py --windows $NEWWIN --out $VID/train --prefix train/ --threads 16 || fail new_videos
stage new_frames $OUT/logs/new_frames.log env NEXAR_VIDEOS=$VID/train R1_TRAIN_DIR=$FR $PY -u r1_pod_cut_windows.py --windows $NEWWIN || fail new_frames
stage new_encode $OUT/logs/new_encode.log env R1_TRAIN_DIR=$FR R1_REASONING_MANIFEST=$EXTRA $PY -u r1_cache_features.py --source nexar_reasoning --splits train,val --out-dir $CACHEX || fail new_encode
if [ "$D" != 1 ]; then
  NX=$(wc -l < $EXTRA)                      # the kept new windows (874 cut, the 4 the teacher flagged "hazard not visible" are not in the manifest)
  [ "$(wc -l < $CACHEX/index_nexar_reasoning.jsonl)" -eq "$NX" ] || fail "extra cache has $(wc -l < $CACHEX/index_nexar_reasoning.jsonl) rows, expected $NX"
  $PY - <<'PY' >> $ST 2>&1 || fail new_windows_sanity
import json, os
idx = [json.loads(l) for l in open(os.environ["CACHEX"] + "/index_nexar_reasoning.jsonl")]
p = [r["p_collision"] for r in idx]
print(f"[chains] crash model on the new crash windows: mean P {sum(p) / len(p):.3f}, share P >= 0.5: {sum(x >= 0.5 for x in p) / len(p):.3f} (the 0.5 s windows of the original set: mean about 0.9)")
assert sum(p) / len(p) > 0.6, "the crash model gives these crash windows a very low P: frames / windows / adapter mismatch"
PY
fi
rm -rf $VID/train $FR 2>/dev/null

# 3 ---- the 1,344 test windows (decision-only evaluations later) + the crash-model sanity (known AP 0.9127 private / 0.9096 public)
for H in private public; do
  if [ $H = private ]; then SRC=nexar_reasoning_test; PRE=test-private/; FRX=$FRT; TM=$MAN/test_manifest_private.jsonl; KNOWN=0.9127; else SRC=nexar_reasoning_test_public; PRE=test-public/; FRX=$FRP; TM=$MAN/test_manifest_public.jsonl; KNOWN=0.9096; fi
  stage test_${H}_videos $OUT/logs/test_${H}_videos.log $PY -u r1_pod_get_videos.py --windows $TM --out $VID/test_$H --prefix $PRE --threads 16 || fail test_${H}_videos
  stage test_${H}_frames $OUT/logs/test_${H}_frames.log $PY -u extract_highres_frames_test.py --manifest_in $TM --dst_root $FRX --src_videos $VID/test_$H --manifest_out $OUT/test_${H}_hires.jsonl --failures_json $OUT/test_${H}_fail.json || fail test_${H}_frames
  stage test_${H}_encode $OUT/logs/test_${H}_encode.log env R1_TRAIN_DIR=$FRX $PY -u r1_cache_features.py --source $SRC --splits test --out-dir $CACHET || fail test_${H}_encode
  if [ "$D" != 1 ]; then
    SRC=$SRC KNOWN=$KNOWN $PY - <<'PY' >> $ST 2>&1 || fail test_sanity
import json, os
from sklearn.metrics import average_precision_score
src, known = os.environ["SRC"], float(os.environ["KNOWN"])
idx = [json.loads(l) for l in open(f"{os.environ['CACHET']}/index_{src}.jsonl")]
man = {m["id"]: m for m in map(json.loads, open(f"/root/r1/dataset/manifests/r1_{src}.jsonl"))}
ap = average_precision_score([man[r["id"]]["label"] for r in idx], [r["p_collision"] for r in idx])
print(f"[chains] crash model on {src} ({len(idx)} windows): AP {ap:.4f} (known {known})")
assert abs(ap - known) < 0.01, "the crash model does not reproduce its known test AP"
PY
  fi
  rm -rf $VID/test_$H $FRX 2>/dev/null
done
say "data ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# 4 ---- smoke tests (real model)
export R1_REASONING_MANIFEST=$V2
DATA2="--sources nexar_reasoning --cache-dir $CACHES --facts-cards $CARDS2 --aug-texts $PARA2 --val-texts none --p-teacher 0.4 --select val_loss --eff-bs 16 --micro-bs 4 --grad-ckpt"
EV="$PY -u r1_eval_reasoning.py --order verdict_first --facts-cards $CARDS2 --cache-dir $CACHES --drop-prealert"
stage smoke_p1 $OUT/logs/smoke_p1.log $PY -u r1_train.py --phase 1 --init random --epochs 1 --max-steps 4 --lr-merger 5e-4 $DATA2 --fixed-epochs --out-dir $OUT/smoke_P1 || fail smoke_p1
stage smoke_p2 $OUT/logs/smoke_p2.log $PY -u r1_train.py --phase 2 --epochs 1 --max-steps 4 --lr-merger 1e-5 --lr-lora 1e-4 $DATA2 --drop-prealert --balance-tte --order verdict_first --init-from $OUT/smoke_P1/best.pt --out-dir $OUT/smoke_P2 || fail smoke_p2
stage smoke_eval $OUT/logs/smoke_eval.log $EV --ckpt $OUT/smoke_P2/best.pt --split val --no-generate --limit 20 --max-ret 4 --out-dir $OUT/smoke_eval || fail smoke_eval
say "smoke tests passed"
# background safety net: every new epoch checkpoint, log and stage marker goes to HF (private, size-verified) every 10 minutes; a lost pod then costs one stage at most
if [ "$D" = 1 ]; then echo "  + [sync] $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live --loop 600 (background)"; else
  nohup $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live --loop 600 >> $OUT/logs/sync.log 2>&1 & SYNC_PID=$!; fi

# 5 ---- chain B (plan rev 2: chain A dropped)
chain() { C=$1; X=$2
  stage ${C}_P1aug $OUT/logs/${C}_P1aug.log $PY -u r1_train.py --phase 1 --init random --epochs 12 --lr-merger 5e-4 $DATA2 --fixed-epochs $X --out-dir $OUT/$C/P1aug || return 1
  stage ${C}_P1fixR $OUT/logs/${C}_P1fixR.log $PY -u r1_train.py --phase 1 --init random --init-from $OUT/$C/P1aug/best.pt --epochs 4 --lr-merger 1.5e-4 $DATA2 --fixed-epochs $X --drop-prealert --balance-tte --order reason_first --out-dir $OUT/$C/P1fixR || return 1
  stage ${C}_P2V $OUT/logs/${C}_P2V.log $PY -u r1_train.py --phase 2 --epochs 6 --patience 3 --lr-merger 1e-5 --lr-lora 1e-4 $DATA2 $X --drop-prealert --balance-tte --order verdict_first --init-from $OUT/$C/P1fixR/best.pt --out-dir $OUT/$C/P2V || return 1
}
chain B "" || say "chain B FAILED (see its logs; the evaluations of today's P2V still run)"

# 6 ---- selection (validation only) and evaluations
stage old_p2v_ckpt $OUT/logs/old_p2v.log $PY -c "from huggingface_hub import hf_hub_download as d; import shutil, os; os.makedirs('$OUT/old/P2V', exist_ok=True); shutil.copy(d('$R_CKPT','phase2f_2026-10-15/P2V/best.pt',local_dir='$OUT/_hf'),'$OUT/old/P2V/best.pt')" || say "old P2V checkpoint not available (its evaluations are skipped)"
EVT="$PY -u r1_eval_reasoning.py --order verdict_first --no-ref --no-generate"
evals_of() { NAME=$1; CK=$2; NOEX=$3          # scoring-only: val full, private test, public test
  stage ev_${NAME}_val $OUT/logs/ev_${NAME}_val.log $EV --ckpt $CK --split val --no-generate --max-ret 50 --out-dir $OUT/eval/$NAME/val
  stage ev_${NAME}_private $OUT/logs/ev_${NAME}_private.log $EVT --source nexar_reasoning_test --cache-dir $CACHET --ckpt $CK --out-dir $OUT/eval/$NAME/private
  stage ev_${NAME}_public $OUT/logs/ev_${NAME}_public.log $EVT --source nexar_reasoning_test_public --cache-dir $CACHET --ckpt $CK --out-dir $OUT/eval/$NAME/public
}
[ -f $OUT/old/P2V/best.pt ] && evals_of P2V_old $OUT/old/P2V/best.pt ""
for C in B; do
  [ -f $OUT/$C/P2V/train_log.jsonl ] || { say "chain $C has no P2V log: no evaluations"; continue; }
  TOP=$($PY r1_chain_select.py $OUT/$C/P2V --k 3 2>> $OUT/logs/select_$C.log); say "chain $C: top-3 epochs by validation AUC: $TOP"
  echo "$TOP" > $OUT/$C/top3.txt
  for E in $TOP; do evals_of ${C}_ep$(printf %02d $E) $OUT/$C/P2V/epoch_$(printf %02d $E).pt ""; done
  BEST=$(echo $TOP | cut -d' ' -f1); CK=$OUT/$C/P2V/epoch_$(printf %02d $BEST).pt; NOEX=""; [ $C = A ] && NOEX="--no-extra-train"
  stage ${C}_text_val $OUT/logs/${C}_text_val.log $EV --ckpt $CK --split val --max-ret 100 --n-blank 5 --out-dir $OUT/eval/${C}_best/val_text
  stage ${C}_text_train $OUT/logs/${C}_text_train.log $EV --ckpt $CK --split train --sample 600 --sample-seed 0 $NOEX --max-ret 100 --n-blank 5 --out-dir $OUT/eval/${C}_best/train600
  for S in val_text train600; do stage ${C}_bert_$S $OUT/logs/${C}_bert_$S.log $PY -u r1_review_reasoning.py --run-dir $OUT/eval/${C}_best/$S; done
done

# 7 ---- push + verify + stop
[ "$D" = 1 ] || { kill $SYNC_PID 2>/dev/null; $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live >> $OUT/logs/sync.log 2>&1; }       # one last pass of the safety net
if [ "$D" = 1 ]; then echo "  + push logs, train logs, results, best.pt / top-3 epochs to $R_CKPT/$HFDIR; verify sizes; stop the pod"; exit 0; fi
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); out = os.environ["OUT"]; rc, base = os.environ["R_CKPT"], os.environ["HFDIR"]
files = []
for c in ("A", "B"):
    for st in ("P1aug", "P1fixR", "P2V"):
        for f in ("train_log.jsonl", "best.pt"):
            p = f"{out}/{c}/{st}/{f}"
            if os.path.exists(p):
                files.append((p, f"{c}/{st}/{f}"))
    top = f"{out}/{c}/top3.txt"
    if os.path.exists(top):
        files.append((top, f"{c}/top3.txt"))
        for e in open(top).read().split():
            p = f"{out}/{c}/P2V/epoch_{int(e):02d}.pt"
            if os.path.exists(p):
                files.append((p, f"{c}/P2V/epoch_{int(e):02d}.pt"))
for root, _, names in os.walk(f"{out}/eval"):
    for n in names:
        if n in ("predictions.jsonl", "metrics.json", "summary.md", "text_metrics.json", "report.md"):
            p = os.path.join(root, n); files.append((p, "eval/" + os.path.relpath(p, f"{out}/eval").replace(os.sep, "/")))
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/logs/*.log"))] + [(f"{out}/logs/chains.status", "chains.status")]
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=rc, repo_type="model")
rem = {x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files], repo_type="model")}
ok = all(rem.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files)
print(len(files), "files;", "VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
