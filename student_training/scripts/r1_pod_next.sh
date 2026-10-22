#!/bin/bash
# Stage 1 of plan 2026-10-20_Plan-Next-Experiments-After-ChainB: two arms in parallel on ONE pod, from today's base (original data, original manifest). Unattended; stops the pod ONLY after
# every result file is verified byte-identical on HF; any major failure leaves the pod running. Judged on VALIDATION only (no test window is downloaded, encoded or scored here).
#   stream A  R2   regularised Phase 2 (LoRA r 8, dropout 0.2, weight decay 0.05 on merger + LoRA) from today's P1fixR, validation checks at 0.5 / 0.75 / 1.0 / 1.25 / 1.5 / 2.0 epochs,
#                  best.pt = highest validation decision AUC, stop after 2 checks in a row without a new best.
#             R1   ONLY if R2's best check lies at or before epoch 1.0 (r1_best_check.py exits 3): today's Phase-2 settings with the same checks and rule.
#   stream B  D2   more diverse rewordings (the judge-accepted versions + the nuance-only rejects, r1_relaxed_paraphrases.py): P1fixR (from P1aug) then P2V with today's settings and today's
#                  rule (best epoch by validation loss); the checks at 0.5 / 0.75 / 1.25 / 1.5 are only logged and saved.
#   then, per arm: scoring on val full (641 = 524 core + the 117 new 0.75 / 1.25 s windows, decision only), the 600-window train sample (decision + losses, no text), and for the best checkpoint a
#   text evaluation on the 524 core validation windows (generation, same evaluator as today's P2V) + BERTScore. Fact cards (Gemini) are run LOCALLY afterwards (the key stays off the pod).
# Needs a pod with a 32 GB GPU and >= 50 GB free container disk (80 GB recommended), NO network volume. Expects in /root/r1 (copied from the PC): code, dataset/manifests/{r1_nexar_reasoning_windows,
# r1_nexar_reasoning_extra,r1_nexar_reasoning_v2}.jsonl, dataset/public_samples/qwen3vl_cfg, outputs/a1_compress256/train/epoch_02/lora_adapter,
# outputs/teacher_dataset_2026-10/{paraphrases_judged,paraphrases_relaxed,fact_cards_teacher}.jsonl, outputs/teacher_extra_2026-10/{windows_new,fact_cards_v2}.jsonl.
#   setsid nohup bash r1_pod_next.sh > /root/r1_next/logs/next.nohup 2>&1 < /dev/null &            status: /root/r1_next/logs/next.status
# DRYRUN=1 prints the commands without running them (no GPU, no network).
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts; OUT=${OUT:-/root/r1_next}
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; HFDIR=next_2026-10-20
CACHE=$OUT/cache; CACHEX=$OUT/cache_extra; CACHES="$CACHE,$CACHEX"
VID=/root/nexar; FR=/root/frames_extra
MAN=$RR/dataset/manifests; W=$RR/outputs/teacher_dataset_2026-10; W2=$RR/outputs/teacher_extra_2026-10
PARA=$W/paraphrases_judged.jsonl; PARAX=$W/paraphrases_relaxed.jsonl; CARDS=$W/fact_cards_teacher.jsonl; CARDS2=$W2/fact_cards_v2.jsonl
NEWWIN=$W2/windows_new.jsonl; NEWVAL=$OUT/windows_new_val.jsonl; V2=$MAN/r1_nexar_reasoning_v2.jsonl; EXTRA=$MAN/r1_nexar_reasoning_extra.jsonl
export OUT CACHE CACHEX HFDIR R_CKPT
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
mkdir -p $OUT/logs; cd $SC
ST=$OUT/logs/next.status; echo "start $(date -u)" > $ST
say() { echo "[next] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED (major): $* - pod left running"; [ "$D" = 1 ] || exit 1; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
# stage NAME LOG CMD... : skipped when done.NAME exists (resumable); output to LOG
stage() { n=$1; lg=$2; shift 2
  if [ "$D" = 1 ]; then echo "  + [$n] $*"; return 0; fi
  [ -f "$OUT/done.$n" ] && { say "skip $n (done)"; return 0; }
  say "start $n"; "$@" > "$lg" 2>&1; rc=$?
  if [ $rc -eq 0 ]; then touch "$OUT/done.$n"; say "done $n"; else say "FAILED $n (rc=$rc), log $lg"; fi; return $rc; }

# 0 ---- restore (a fresh pod after a lost one): finished stages, best.pt and train logs come back from HF (the data preparation below is always redone)
if [ "$D" = 1 ]; then echo "  + [restore] $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live"; else
  $PY r1_hf_sync.py pull --dir $OUT --repo $R_CKPT --prefix $HFDIR/live >> $OUT/logs/restore.log 2>&1 || say "restore from HF failed or nothing to restore (see logs/restore.log); continuing from scratch"
  say "restore: $(ls $OUT/done.* 2>/dev/null | wc -l) finished stages found on HF"; fi

# 0 ---- checks
for f in $NEWWIN $PARA $PARAX $CARDS $CARDS2 $V2 $EXTRA $MAN/r1_nexar_reasoning_windows.jsonl $RR/outputs/a1_compress256/train/epoch_02/lora_adapter $RR/dataset/public_samples/qwen3vl_cfg \
         $SC/r1_best_check.py $SC/r1_relaxed_paraphrases.py; do
  [ -e $f ] || fail "missing $f"; done
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "$D" = 1 ] || [ -f $CACHE/index_nexar_reasoning.jsonl ] || [ "${FREE_GB:-0}" -ge 50 ] || fail "container disk has only ${FREE_GB} GB free (need >= 50; 80 recommended)"      # a resumed run already holds the cache: skip
[ "$D" = 1 ] || [ "$(wc -l < $PARAX)" -gt "$(wc -l < $PARA)" ] || fail "paraphrases_relaxed.jsonl is not larger than paraphrases_judged.jsonl"
stage pip $OUT/logs/pip.log bash -c "$PY -c 'import bert_score' 2>/dev/null || /root/venv/bin/pip install -q bert-score" || fail pip

# 1 ---- the old features cache (4,169 windows; the ORIGINAL manifest defines its row order, so the manifest override is unset here)
mkdir -p $CACHE $CACHEX $VID $OUT/base/P1aug $OUT/base/P1fixR
stage cache_old $OUT/logs/pull_cache.log env -u R1_REASONING_MANIFEST $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE || fail pull_cache
stage base_ckpts $OUT/logs/base_ckpts.log $PY -c "
from huggingface_hub import hf_hub_download as d; import shutil
shutil.copy(d('$R_CKPT','phase1aug_2026-10-14/P1aug/best.pt',local_dir='$OUT/_hf'),'$OUT/base/P1aug/best.pt')
shutil.copy(d('$R_CKPT','phase2f_2026-10-15/P1fixR/best.pt',local_dir='$OUT/_hf'),'$OUT/base/P1fixR/best.pt')" || fail base_ckpts

# 2 ---- the 117 VALIDATION windows at TTE 0.75 / 1.25 s (the validation-full slices of the gate): videos (HF train/) -> frames -> encode with the frozen V-JEPA2 / A1 encoder
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
  [ "$(wc -l < $CACHEX/index_nexar_reasoning.jsonl)" -eq "$NXV" ] || fail "extra cache has $(wc -l < $CACHEX/index_nexar_reasoning.jsonl) rows, expected $NXV (validation rows of the extra manifest)"
  $PY - <<'PY' >> $ST 2>&1 || fail new_windows_sanity
import json, os
idx = [json.loads(l) for l in open(os.environ["CACHEX"] + "/index_nexar_reasoning.jsonl")]
p = [r["p_collision"] for r in idx]
print(f"[next] crash model on the {len(p)} new validation crash windows: mean P {sum(p) / len(p):.3f}, share P >= 0.5: {sum(x >= 0.5 for x in p) / len(p):.3f}")
assert sum(p) / len(p) > 0.6, "the crash model gives these crash windows a very low P: frames / windows / adapter mismatch"
PY
fi
rm -rf $VID/train $FR 2>/dev/null
say "data ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# training command pieces (the ORIGINAL manifest and cache: the same data as today's P2V; D2 swaps the rewordings file)
DATA="--sources nexar_reasoning --cache-dir $CACHE --facts-cards $CARDS --drop-prealert --balance-tte --aug-texts $PARA --val-texts none --p-teacher 0.4 --eff-bs 16 --micro-bs 4 --grad-ckpt"
DATAX="--sources nexar_reasoning --cache-dir $CACHE --facts-cards $CARDS --drop-prealert --balance-tte --aug-texts $PARAX --val-texts none --p-teacher 0.4 --eff-bs 16 --micro-bs 4 --grad-ckpt"
P2="$PY -u r1_train.py --phase 2 --epochs 6 --lr-merger 2e-5 --lr-lora 2e-4 --order verdict_first"
EVO="$PY -u r1_eval_reasoning.py --facts-cards $CARDS --drop-prealert --cache-dir $CACHE --order verdict_first"                       # original manifest: the 524 core validation windows
EVS="env R1_REASONING_MANIFEST=$V2 $PY -u r1_eval_reasoning.py --order verdict_first --facts-cards $CARDS2 --cache-dir $CACHES --drop-prealert"   # val full (641)

# 3 ---- smoke tests (real model): R2 flags + one mid-epoch check, D2 Phase 1 with the relaxed texts, a scoring pass on val full
stage smoke_r2 $OUT/logs/smoke_r2.log $P2 $DATA --init-from $OUT/base/P1fixR/best.pt --select auc --eval-at 0.01 --patience-checks 2 --lora-r 8 --lora-dropout 0.2 --weight-decay 0.05 --lora-weight-decay 0.05 --max-steps 4 --epochs 1 --out-dir $OUT/smoke_R2 || fail smoke_r2
stage smoke_d2 $OUT/logs/smoke_d2.log $PY -u r1_train.py --phase 1 --init random --init-from $OUT/base/P1aug/best.pt --epochs 1 --max-steps 4 --lr-merger 3e-4 $DATAX --order reason_first --select val_loss --out-dir $OUT/smoke_D2 || fail smoke_d2
stage smoke_eval $OUT/logs/smoke_eval.log $EVS --ckpt $OUT/smoke_R2/best.pt --split val --no-generate --limit 20 --max-ret 4 --out-dir $OUT/smoke_eval || fail smoke_eval
say "smoke tests passed"
# background safety net: train logs, best.pt, stage markers and logs go to HF (private, size-verified) every 10 minutes; epoch / check files are NOT synced (HF storage is nearly full)
if [ "$D" = 1 ]; then echo "  + [sync] $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live --subdirs R2,R1,D2P1,D2,logs,. --loop 600 (background)"; else
  nohup $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live --subdirs R2,R1,D2P1,D2,logs,. --patterns "train_log.jsonl,best.pt,best_auc.pt,best_check.txt,done.*,*.log,next.status" --loop 600 >> $OUT/logs/sync.log 2>&1 & SYNC_PID=$!; fi

# 4 ---- evaluations of one arm (validation only): scoring on val full, scoring on the train sample (600), text on the 524 core windows + BERTScore for the best checkpoint
evals_of() { N=$1; CK=$2
  stage ev_${N}_val $OUT/logs/ev_${N}_val.log $EVS --ckpt $CK --split val --no-generate --max-ret 50 --out-dir $OUT/eval/$N/val
  stage ev_${N}_train $OUT/logs/ev_${N}_train.log $EVO --ckpt $CK --split train --sample 600 --sample-seed 0 --no-generate --max-ret 50 --out-dir $OUT/eval/$N/train600
}
text_of() { N=$1; CK=$2
  stage text_${N} $OUT/logs/text_${N}.log $EVO --ckpt $CK --split val --max-ret 100 --n-blank 5 --train-threshold 600 --out-dir $OUT/eval/$N/val_text
  stage bert_${N} $OUT/logs/bert_${N}.log $PY -u r1_review_reasoning.py --run-dir $OUT/eval/$N/val_text
}

# 5 ---- stream A: R2, then R1 only if R2's best check lies at or before epoch 1.0
CHK="--eval-at 0.5,0.75,1.0,1.25,1.5,2.0"
streamA() {
  stage R2_p2v $OUT/logs/R2_p2v.log $P2 $DATA --init-from $OUT/base/P1fixR/best.pt --select auc $CHK --patience-checks 2 --lora-r 8 --lora-dropout 0.2 --weight-decay 0.05 --lora-weight-decay 0.05 --out-dir $OUT/R2 || return 1
  if [ "$D" = 1 ]; then echo "  + [R2 best check] $PY r1_best_check.py $OUT/R2 --after 1.0   (exit 0: best after epoch 1.0 -> skip R1; exit 3: at or before -> run R1)"; RC=3; else
    $PY r1_best_check.py $OUT/R2 > $OUT/R2/best_check.txt 2> $OUT/logs/R2_best_check.log; say "R2 best check: $(cat $OUT/R2/best_check.txt)"
    $PY r1_best_check.py $OUT/R2 --after 1.0 > /dev/null 2>&1; RC=$?; fi
  evals_of R2 $OUT/R2/best.pt
  text_of R2 $OUT/R2/best.pt
  if [ "$RC" = 3 ]; then
    say "R2's best check is at or before epoch 1.0: running R1 (today's settings with the same checks)"
    stage R1_p2v $OUT/logs/R1_p2v.log $P2 $DATA --init-from $OUT/base/P1fixR/best.pt --select auc $CHK --patience-checks 2 --out-dir $OUT/R1 || return 1
    [ "$D" = 1 ] || { $PY r1_best_check.py $OUT/R1 > $OUT/R1/best_check.txt 2> $OUT/logs/R1_best_check.log; say "R1 best check: $(cat $OUT/R1/best_check.txt)"; }
    evals_of R1 $OUT/R1/best.pt
    text_of R1 $OUT/R1/best.pt
  elif [ "$RC" = 0 ]; then say "R2's best check lies after epoch 1.0: R1 skipped (plan rule)"
  else say "r1_best_check.py returned $RC for R2: R1 skipped, look at logs/R2_best_check.log"; fi
}

# 6 ---- stream B: D2 (Phase-1 continuation from P1aug with the relaxed texts, then P2V with today's settings and rule)
streamB() {
  stage D2_p1 $OUT/logs/D2_p1.log $PY -u r1_train.py --phase 1 --init random --init-from $OUT/base/P1aug/best.pt --epochs 4 --patience 1 --lr-merger 3e-4 $DATAX --order reason_first --select val_loss --out-dir $OUT/D2P1 || return 1
  stage D2_p2v $OUT/logs/D2_p2v.log $P2 $DATAX --patience 1 --select val_loss --init-from $OUT/D2P1/best.pt --eval-at 0.5,0.75,1.25,1.5 --checks-log-only --out-dir $OUT/D2 || return 1
  if [ "$D" != 1 ]; then
    $PY r1_best_check.py $OUT/D2 > $OUT/D2/best_check.txt 2> $OUT/logs/D2_best_check.log; say "D2 best check by validation AUC: $(cat $OUT/D2/best_check.txt)"
    BCK=$(cut -d' ' -f1 $OUT/D2/best_check.txt); [ -f $OUT/D2/$BCK ] && cp $OUT/D2/$BCK $OUT/D2/best_auc.pt
  fi
  evals_of D2 $OUT/D2/best.pt                                        # the epoch-end rule, like today's P2V
  [ "$D" = 1 ] || cmp -s $OUT/D2/best.pt $OUT/D2/best_auc.pt || evals_of D2auc $OUT/D2/best_auc.pt    # the best check point by validation AUC, when it differs
  text_of D2 $OUT/D2/best.pt
}
say "streams A (R2 -> R1?) and B (D2) start in parallel"
streamA & PA=$!
streamB & PB=$!
wait $PA || say "stream A FAILED somewhere (see the logs); its finished parts are kept"
wait $PB || say "stream B FAILED somewhere (see the logs); its finished parts are kept"

# 7 ---- push + verify + stop
[ "$D" = 1 ] || { kill $SYNC_PID 2>/dev/null; $PY r1_hf_sync.py push --dir $OUT --repo $R_CKPT --prefix $HFDIR/live --subdirs R2,R1,D2P1,D2,logs,. --patterns "train_log.jsonl,best.pt,best_auc.pt,best_check.txt,done.*,*.log,next.status" >> $OUT/logs/sync.log 2>&1; }       # one last pass of the safety net
if [ "$D" = 1 ]; then echo "  + push logs, train logs, best.pt / best_auc.pt, results to $R_CKPT/$HFDIR; verify sizes; stop the pod"; exit 0; fi
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); out = os.environ["OUT"]; rc, base = os.environ["R_CKPT"], os.environ["HFDIR"]
files = []
for arm in ("R2", "R1", "D2P1", "D2"):
    for f in ("train_log.jsonl", "best.pt", "best_auc.pt", "best_check.txt"):
        p = f"{out}/{arm}/{f}"
        if os.path.exists(p):
            files.append((p, f"{arm}/{f}"))
for root, _, names in os.walk(f"{out}/eval"):
    for n in names:
        if n in ("predictions.jsonl", "metrics.json", "summary.md", "text_metrics.json", "report.md"):
            p = os.path.join(root, n); files.append((p, "eval/" + os.path.relpath(p, f"{out}/eval").replace(os.sep, "/")))
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/logs/*.log"))] + [(f"{out}/logs/next.status", "next.status")]
assert any(d.endswith("best.pt") for _, d in files), "no arm produced a checkpoint"
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=rc, repo_type="model")
rem = {x.path: x.size for x in a.get_paths_info(rc, [f"{base}/{d}" for _, d in files], repo_type="model")}
ok = all(rem.get(f"{base}/{d}") == os.path.getsize(s) for s, d in files)
print(len(files), "files;", "VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
stop_pod
