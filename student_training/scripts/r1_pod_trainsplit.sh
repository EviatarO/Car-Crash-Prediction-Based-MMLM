#!/bin/bash
# P2V report session (plan 2026-10-18, Part A). Unattended; stops the pod ONLY after every result file is verified byte-identical on HF.
#   0  checks; pull the 18 GB features cache (the training / validation windows) and the P2V + P1fixR checkpoints from HF
#   1  smoke: 20 stratified TRAIN windows, the full evaluation (loss / gaps / retrieval / generation)
#   2  P2V on a stratified sample of 600 TRAINING windows (pre-alert dropped, seed 0): decision scores, loss / perplexity, wrong- and blank-video gaps,
#      retrieval@10, generated answers      -> train600/
#   3  BERTScore / ROUGE-L / BLEU-4 / token-F1 (with floors) of the 600 train answers vs the teacher text (r1_review_reasoning.py, roberta-large) and of the
#      private-test "agreed" subset (431 windows whose pass-1 teacher verdict equals the label, r1_text_eval.py)   -> train600/, test_agreed/
#   4  push to HF eviatarO-org/checkpoints/p2v_report_2026-10-18/, verify sizes, stop the pod. The pod is left running on any failure.
# Needs >= 30 GB free container disk (cache 18 GB; a 40 GB pod is enough). Expects in /root/r1 (copied from the PC): code, dataset/manifests/r1_nexar_reasoning_windows.jsonl,
# outputs/teacher_dataset_2026-10/fact_cards_teacher.jsonl, outputs/r1_p2v_report_2026-10-18/test_agreed/predictions.jsonl, dataset/public_samples/qwen3vl_cfg.
#   setsid nohup bash r1_pod_trainsplit.sh > /root/p2v_report/logs/pts.nohup 2>&1 < /dev/null &        status: /root/p2v_report/logs/pts.status
# DRYRUN=1 bash r1_pod_trainsplit.sh   prints the commands without running them (no GPU, no network).
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
PY=${PY:-/root/venv/bin/python}; RR=${RR:-/root/r1}; SC=$RR/student_training/scripts
WORK=${WORK:-/root/p2v_report}; CACHE=$WORK/cache; CKPT=$WORK/ckpt
R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints; HFDIR=p2v_report_2026-10-18; N_TRAIN=600
export WORK CACHE R_CKPT HFDIR
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
D=${DRYRUN:-0}
run() { if [ "$D" = 1 ]; then echo "  + $*"; else eval "$@"; fi; }
mkdir -p $WORK/logs; [ "$D" = 1 ] || mkdir -p $CACHE $CKPT/P2V $CKPT/P1fixR; cd $SC
ST=$WORK/logs/pts.status; echo "start $(date -u)" > $ST
say() { echo "[pts] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; [ "$D" = 1 ] || exit 1; }
stop_pod() { export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs); runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"; }
CARDS=$RR/outputs/teacher_dataset_2026-10/fact_cards_teacher.jsonl; TEST=$RR/outputs/r1_p2v_report_2026-10-18/test_agreed
for f in $RR/dataset/manifests/r1_nexar_reasoning_windows.jsonl $CARDS $TEST/predictions.jsonl $RR/dataset/public_samples/qwen3vl_cfg; do [ -e $f ] || fail "missing $f"; done
[ "$D" = 1 ] || [ "$(wc -l < $TEST/predictions.jsonl)" -eq 431 ] || fail "test_agreed/predictions.jsonl has $(wc -l < $TEST/predictions.jsonl) rows, expected 431"
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ -f $CACHE/index_nexar_reasoning.jsonl ] || [ "${FREE_GB:-0}" -ge 30 ] || fail "container disk has only ${FREE_GB} GB free (need >= 30: the cache is 18 GB, checkpoints + venv about 5 GB)"

# 0 ---- features cache + checkpoints
run "$PY -c 'import bert_score' 2>/dev/null || /root/venv/bin/pip install -q bert-score >> $WORK/logs/pip.log 2>&1" || fail pip_bert_score
[ "$D" = 1 ] || [ -f $CACHE/index_nexar_reasoning.jsonl ] || { say "pull the features cache (18 GB)"; run "$PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE > $WORK/logs/pull_cache.log 2>&1" || fail pull_cache; }
[ "$D" = 1 ] && echo "  + (pull the features cache if absent)"
for N in P2V P1fixR; do
  run "$PY -c \"from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','phase2f_2026-10-15/$N/best.pt',local_dir='$WORK/_hf'),'$CKPT/$N/best.pt')\"" || fail get_$N
done
say "cache + checkpoints ready; disk: $(df -h /root | tail -1 | tr -s ' ' | cut -d' ' -f3,4)"

# 1 ---- smoke
EV="$PY -u r1_eval_reasoning.py --ckpt $CKPT/P2V/best.pt --before-ckpt $CKPT/P1fixR/best.pt --order verdict_first --split train --drop-prealert --facts-cards $CARDS --cache-dir $CACHE --sample-seed 0 --max-new 160"
say "smoke: 20 train windows"
run "$EV --sample 20 --max-ret 4 --n-blank 1 --out-dir $WORK/smoke > $WORK/logs/smoke.log 2>&1" || fail smoke

# 2 ---- the train sample
say "P2V on $N_TRAIN stratified training windows"
run "$EV --sample $N_TRAIN --max-ret 100 --n-blank 5 --out-dir $WORK/train600 > $WORK/logs/eval_train600.log 2>&1" || fail eval_train600
say "train sample done: $(grep -h -a '^| a1_p\|^| s_first' $WORK/train600/summary.md 2>/dev/null | cut -c1-90 | tr '\n' ' ')"

# 3 ---- text metrics with BERTScore
say "text metrics: train sample (r1_review_reasoning) and the private-test agreed subset (r1_text_eval)"
run "$PY -u r1_review_reasoning.py --run-dir $WORK/train600 > $WORK/logs/text_train600.log 2>&1" || fail text_train600
mkdir -p $WORK/test_agreed; [ "$D" = 1 ] || cp $TEST/predictions.jsonl $WORK/test_agreed/predictions.jsonl
run "$PY -u r1_text_eval.py --run-dir $WORK/test_agreed > $WORK/logs/text_test.log 2>&1" || fail text_test

# 4 ---- push + verify + stop
if [ "$D" = 1 ]; then echo "  + push train600/*, test_agreed/text_metrics.json, logs, status to $R_CKPT/$HFDIR; verify sizes; stop the pod"; exit 0; fi
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); w = os.environ["WORK"]; rc, base = os.environ["R_CKPT"], os.environ["HFDIR"]
files = [(f, f"train600/{os.path.basename(f)}") for f in sorted(glob.glob(f"{w}/train600/*")) if os.path.isfile(f)]
files += [(f"{w}/test_agreed/text_metrics.json", "test_agreed/text_metrics.json")]
files += [(f, f"logs/{os.path.basename(f)}") for f in sorted(glob.glob(f"{w}/logs/*.log"))]
files += [(f"{w}/logs/pts.status", "pts.status")]
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
