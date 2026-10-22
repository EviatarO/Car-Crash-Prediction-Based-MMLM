#!/bin/bash
# Stage 3 of plan 2026-10-22_Plan-Stage3-Encoder-LoRA: the V-JEPA2 encoder's LoRA trains with the language model's loss ONLY (no crash head, no predictor), warm start = the control's P2V.
#   ARM=E  encoder LoRA only (projector + language-model LoRA frozen: the clean diagnostic)        ARM=J  encoder LoRA + projector + language-model LoRA (co-adaptation)
# One pod per arm (the memory does not allow two streams). Validation only; no test window is used. Steps:
#   data (old feature cache = index + parity reference; frames of the 4,169 windows from the HF window repo; the 117 new validation windows cut on the pod)
#   -> parity checks (GPU normalisation == processor; encoder-only tokens == cached tokens) -> smoke test (micro-batch 2, fallback 1)
#   -> training (epoch-0 parity check vs the control's validation loss 1.078; checkpoints every 0.25 epoch unscored; local refinement around the winning epoch)
#   -> re-encode all windows with the trained encoder -> the usual evaluations on the NEW cache (val full, train sample, text + BERTScore) + token drift + the fact probe before / after
#   -> push to HF + byte verification -> stop the pod.   ANY major failure: the logs are pushed to HF and the pod is STOPPED (user rule 2026-10-22).
# Needs a 32 GB GPU and >= 80 GB free disk (100 GB recommended), NO network volume. Expects in /root/r1 (copied from the PC): code,
# dataset/manifests/{r1_nexar_reasoning_windows,r1_nexar_reasoning_extra,r1_nexar_reasoning_v2}.jsonl, dataset/public_samples/qwen3vl_cfg, outputs/a1_compress256/train/epoch_02/lora_adapter,
# outputs/teacher_dataset_2026-10/{paraphrases_relaxed,fact_cards_teacher}.jsonl, outputs/teacher_extra_2026-10/{windows_new,fact_cards_v2}.jsonl.
#   ARM=J setsid nohup bash r1_pod_stage3.sh > /root/r1_s3/logs/s3.nohup 2>&1 < /dev/null &            status: /root/r1_s3/logs/s3.status        DRYRUN=1 prints the commands only.
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
ARM=${ARM:-J}; [ "$ARM" = J ] || [ "$ARM" = E ] || { echo "ARM must be J or E"; exit 2; }
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts; OUT=${OUT:-/root/r1_s3}
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; R_WIN=eviatarO-org/nexar-reasoning-windows; HFDIR=stage3_2026-10-22/$ARM
CTRL_HF=b4_2026-10-21/CT/P2V/best.pt; CTRL_LOSS=1.078
CACHE=$OUT/cache; CACHEX=$OUT/cache_extra; CACHES="$CACHE,$CACHEX"; NEWC=$OUT/cache_new; FRAMES=$OUT/frames; FRAMESX=$OUT/frames_extra
VID=/root/nexar; FR=/root/frames_extra
MAN=$RR/dataset/manifests; W=$RR/outputs/teacher_dataset_2026-10; W2=$RR/outputs/teacher_extra_2026-10
PARAX=$W/paraphrases_relaxed.jsonl; CARDS=$W/fact_cards_teacher.jsonl; CARDS2=$W2/fact_cards_v2.jsonl
NEWWIN=$W2/windows_new.jsonl; NEWVAL=$OUT/windows_new_val.jsonl; V2=$MAN/r1_nexar_reasoning_v2.jsonl; EXTRA=$MAN/r1_nexar_reasoning_extra.jsonl
A1=$RR/outputs/a1_compress256/train/epoch_02/lora_adapter; BASE=$OUT/base
export OUT CACHE CACHEX HFDIR R_CKPT
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
mkdir -p $OUT/logs; cd $SC
ST=$OUT/logs/s3.status; echo "start $(date -u) arm $ARM" >> $ST
SYNC="--subdirs $ARM,logs,. --patterns train_log.jsonl,monitor.jsonl,best.pt,adapter_*,done.*,*.log,s3.status"   # adapter_* = best_enc/ and ck_*/enc/ (encoder LoRA, ~10 MB each)
say() { echo "[s3-$ARM] $* $(date -u)" | tee -a $ST; }
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

# 0 ---- restore + checks
if [ "$D" = 1 ]; then echo "  + [restore] $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live"; else
  $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live >> $OUT/logs/restore.log 2>&1 || say "nothing to restore from HF (fresh start)"
  say "restore: $(ls $OUT/done.* 2>/dev/null | wc -l) finished stages found"
  # a fresh container has none of the data / caches the markers stand for (only logs, best.pt and the encoder adapters are synced): keep only done.train, and only if its outputs came back
  if [ ! -f $CACHE/index_nexar_reasoning.jsonl ]; then
    for m in $OUT/done.*; do [ -e "$m" ] || continue; [ "$(basename $m)" = done.train ] || rm -f "$m"; done
    { [ -f $OUT/$ARM/best.pt ] && [ -d $OUT/$ARM/best_enc ]; } || rm -f $OUT/done.train
    say "fresh container: data stages will be rebuilt ($(ls $OUT/done.* 2>/dev/null | wc -l) stage markers kept)"
  fi; fi
for f in $NEWWIN $PARAX $CARDS $CARDS2 $V2 $EXTRA $MAN/r1_nexar_reasoning_windows.jsonl $A1 $RR/dataset/public_samples/qwen3vl_cfg $SC/r1_frames.py; do
  [ -e $f ] || fail "missing $f"; done
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$D" = 1 ] || [ -f $CACHE/index_nexar_reasoning.jsonl ] || [ "${FREE_GB:-0}" -ge 80 ] || fail "container disk has only ${FREE_GB} GB free (need >= 80; 100 recommended)"
stage pip $OUT/logs/pip.log bash -c "$PY -c 'import bert_score' 2>/dev/null || /root/venv/bin/pip install -q bert-score" || fail pip

# 1 ---- data: old feature cache (index rows + parity reference), the control checkpoint, frames of every window, the 117 new validation windows
mkdir -p $CACHE $CACHEX $VID $BASE
stage cache_old $OUT/logs/pull_cache.log env -u R1_REASONING_MANIFEST $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE || fail pull_cache
stage control_ckpt $OUT/logs/control_ckpt.log $PY -c "
from huggingface_hub import hf_hub_download as d; import shutil
shutil.copy(d('$R_CKPT','$CTRL_HF',local_dir='$OUT/_hf'),'$BASE/best.pt')" || fail control_ckpt
stage frames $OUT/logs/frames.log $PY -u r1_frames.py build --repo $R_WIN --out $FRAMES || fail frames
stage mk_val_windows $OUT/logs/mk_val_windows.log $PY -c "
import json
rows=[json.loads(l) for l in open('$NEWWIN') if l.strip()]
v=[r for r in rows if r['split']=='val']
open('$NEWVAL','w').write(''.join(json.dumps(r)+'\n' for r in v)); print(len(v),'validation windows of',len(rows))" || fail mk_val_windows
stage new_videos $OUT/logs/new_videos.log $PY -u r1_pod_get_videos.py --windows $NEWVAL --out $VID/train --prefix train/ --threads 16 || fail new_videos
stage new_frames $OUT/logs/new_frames.log env NEXAR_VIDEOS=$VID/train R1_TRAIN_DIR=$FR $PY -u r1_pod_cut_windows.py --windows $NEWVAL || fail new_frames
stage new_encode $OUT/logs/new_encode.log env R1_TRAIN_DIR=$FR R1_REASONING_MANIFEST=$EXTRA $PY -u r1_cache_features.py --source nexar_reasoning --splits val --out-dir $CACHEX || fail new_encode
stage new_frames_u8 $OUT/logs/new_frames_u8.log $PY -u r1_frames.py build --frames-dirs $FR --manifest $EXTRA --split val --out $FRAMESX || fail new_frames_u8
if [ "$D" != 1 ]; then
  [ "$(wc -l < $CACHEX/index_nexar_reasoning.jsonl)" -eq "$(wc -l < $FRAMESX/index_nexar_reasoning.jsonl)" ] || fail "extra cache and extra frames differ in size"
  $PY - <<'PY' >> $ST 2>&1 || fail new_windows_sanity
import json, os
idx = [json.loads(l) for l in open(os.environ["CACHEX"] + "/index_nexar_reasoning.jsonl")]
p = [r["p_collision"] for r in idx]
print(f"[s3] crash model on the {len(p)} new validation crash windows: mean P {sum(p) / len(p):.3f} (stage 1: 0.884)")
assert abs(sum(p) / len(p) - 0.884) < 0.01, "the encoder does not reproduce stage 1 on the new windows"
PY
fi
[ "$D" = 1 ] || rm -rf $VID/train $FR 2>/dev/null
stage parity $OUT/logs/parity.log $PY -u r1_frames.py parity --frames $FRAMES --cache $CACHE --n 24 || fail parity_encoder_only_tokens_differ_from_cache
say "data ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"
stage bench_precision $OUT/logs/bench_precision.log $PY -u r1_frames.py bench --frames $FRAMES --n 8 --out $OUT/bench_precision.json

# 2 ---- training command (arm-specific) and the smoke test (micro-batch 2, fallback 1 on out-of-memory)
PREC=fp32                                                          # the smoke test runs fp32 (TF32); the training precision is read after it
if [ $ARM = E ]; then ARMF="--freeze-reader"; else ARMF="--lr-merger 1e-5 --lr-lora 5e-5"; fi
TRB="$PY -u r1_train.py --phase 2 --sources nexar_reasoning --online-encoder $A1 --frames $FRAMES,$FRAMESX --cache-dir $CACHES --init-from $BASE/best.pt --facts-cards $CARDS --aug-texts $PARAX --val-texts none --p-teacher 0.4 --drop-prealert --balance-tte --order verdict_first --eff-bs 16 --grad-ckpt --lr-encoder 5e-5 $ARMF --select val_loss --patience 2 --log-every 25"
MB=2
smoke() { stage smoke_mb$MB $OUT/logs/smoke_mb$MB.log $TRB --enc-precision $PREC --micro-bs $MB --epochs 1 --max-steps 4 --limit-val 24 --log-every 1 --eval-start --save-every 0.01 --refine-around 0.01 --out-dir $OUT/smoke_mb$MB; }
smoke || { if [ "$D" != 1 ] && grep -qi "out of memory" $OUT/logs/smoke_mb2.log; then say "out of memory at micro-batch 2: falling back to 1"; MB=1; smoke || fail smoke_mb1; else fail smoke; fi; }
say "smoke test passed (micro-batch $MB)"
# the user decides fp32 or bf16 from the bench + the smoke timings (written to $OUT/PRECISION by the operator); no answer within 60 min -> fp32 (the safe choice)
if [ "$D" != 1 ]; then
  say "smoke ok; waiting up to 60 min for the precision decision in $OUT/PRECISION (fp32 = TF32, fp32_strict or bf16); default fp32"
  for i in $(seq 1 180); do [ -s $OUT/PRECISION ] && break; sleep 20; done
  [ -s $OUT/PRECISION ] && PREC=$(grep -o -E 'fp32_strict|fp32|bf16' $OUT/PRECISION | head -1) || PREC=fp32
  [ "$PREC" = fp32 ] || [ "$PREC" = fp32_strict ] || [ "$PREC" = bf16 ] || PREC=fp32
  say "encoder precision for training: $PREC"
fi
if [ "$D" = 1 ]; then echo "  + [sync] $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 (background)"; else
  nohup $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC --loop 600 >> $OUT/logs/sync.log 2>&1 & SYNC_PID=$!; fi

# 3 ---- training: epoch-0 parity (must reproduce the control's validation loss), checkpoints every 0.25 epoch, early stop, local refinement around the winning epoch
stage train $OUT/logs/train.log $TRB --enc-precision $PREC --micro-bs $MB --epochs 6 --eval-start --expect-val-loss $CTRL_LOSS --expect-tol 0.01 --save-every 0.25 --refine-around 0.25,0.5 --out-dir $OUT/$ARM || fail training_failed_or_aborted
[ "$D" = 1 ] || [ -d $OUT/$ARM/best_enc ] || fail "no best_enc adapter was saved"

# 4 ---- re-encode every window with the trained encoder (encoder only, fp32, as the original caching); the crash-model P column is copied from the old index
stage reencode $OUT/logs/reencode.log $PY -u r1_frames.py encode --frames $FRAMES,$FRAMESX --adapter $OUT/$ARM/best_enc --p-from $CACHE,$CACHEX --out $NEWC --bs 4 || fail reencode
stage drift $OUT/logs/drift.log $PY -u r1_frames.py drift --old $CACHE --new $NEWC --split val --out $OUT/drift_val.json
stage probe_old $OUT/logs/probe_old.log $PY -u r1_fact_probe.py --cache-dir $CACHE --out-dir $OUT/probe_old --inputs last
stage probe_new $OUT/logs/probe_new.log $PY -u r1_fact_probe.py --cache-dir $NEWC --out-dir $OUT/probe_new --inputs last

# 5 ---- the usual evaluations on the NEW cache (validation only): the arm's projector + language-model LoRA = $OUT/$ARM/best.pt
CK=$OUT/$ARM/best.pt
EVO="$PY -u r1_eval_reasoning.py --facts-cards $CARDS --drop-prealert --cache-dir $NEWC --order verdict_first"
EVS="env R1_REASONING_MANIFEST=$V2 $PY -u r1_eval_reasoning.py --order verdict_first --facts-cards $CARDS2 --cache-dir $NEWC --drop-prealert"
stage ev_val $OUT/logs/ev_val.log $EVS --ckpt $CK --split val --no-generate --max-ret 50 --out-dir $OUT/eval/$ARM/val
stage ev_train $OUT/logs/ev_train.log $EVO --ckpt $CK --split train --sample 600 --sample-seed 0 --no-generate --max-ret 50 --out-dir $OUT/eval/$ARM/train600
stage text $OUT/logs/text.log $EVO --ckpt $CK --split val --max-ret 100 --n-blank 5 --train-threshold 600 --out-dir $OUT/eval/$ARM/val_text
stage bert $OUT/logs/bert.log $PY -u r1_review_reasoning.py --run-dir $OUT/eval/$ARM/val_text
[ "$D" = 1 ] || [ -f $OUT/eval/$ARM/val/predictions.jsonl ] || fail "the validation scoring produced no predictions"

# 6 ---- push + verify + stop
[ "$D" = 1 ] || { kill $SYNC_PID 2>/dev/null; $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live $SYNC >> $OUT/logs/sync.log 2>&1; }
if [ "$D" = 1 ]; then echo "  + push train logs, monitor, best.pt + best_enc, results, logs to $R_CKPT/$HFDIR; verify sizes; stop the pod"; exit 0; fi
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); out = os.environ["OUT"]; rc, base = os.environ["R_CKPT"], os.environ["HFDIR"]; arm = os.path.basename(base)
files = []
for f in ("train_log.jsonl", "monitor.jsonl", "best.pt"):
    p = f"{out}/{arm}/{f}"
    if os.path.exists(p):
        files.append((p, f))
for p in glob.glob(f"{out}/{arm}/best_enc/*"):
    files.append((p, "best_enc/" + os.path.basename(p)))
for f in ("drift_val.json",):
    if os.path.exists(f"{out}/{f}"):
        files.append((f"{out}/{f}", f))
for sub in ("eval", "probe_old", "probe_new"):
    for root, _, names in os.walk(f"{out}/{sub}"):
        for n in names:
            if n in ("predictions.jsonl", "metrics.json", "summary.md", "text_metrics.json", "report.md", "fact_probe.json"):
                p = os.path.join(root, n); files.append((p, sub + "/" + os.path.relpath(p, f"{out}/{sub}").replace(os.sep, "/")))
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/logs/*.log"))] + [(f"{out}/logs/s3.status", "s3.status")]
assert any(d == "best.pt" for _, d in files) and any(d.startswith("best_enc/") for _, d in files), "no trained checkpoint"
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=rc, repo_type="model")
rem = {x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files], repo_type="model")}
ok = all(rem.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files)
print(len(files), "files;", "VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
