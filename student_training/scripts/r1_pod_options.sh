#!/bin/bash
# Three follow-up options after Run 1 (2026-10-12, user away; all on one pod, unattended, stops the pod only after HF verification):
#   opt1     : evaluate Run 1's epoch 3 (lowest validation loss) instead of epoch 5 (chosen by the wrong-video gap)      - no training
#   opt2_cap : retrain with a template cap (explanation openings seen > 5 times are kept with prob 5/count per epoch)      - template collapse
#   opt3_box : retrain with the position of the attended road user as words in the prompt (r1_common.box_hint; no class, no motion)
# Both retrains: Phase-1 start, same hyper-parameters as Run 1, best.pt = lowest validation loss, stop at the first epoch without improvement.
#   bash r1_pod_options.sh          (run with setsid nohup; status in $OUT/logs/options.status)
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
OUT=${OUT:-/root/r1_week1}; PY=${PY:-/root/venv/bin/python}; SC=/root/r1/student_training/scripts
export CACHE=${CACHE:-/dev/shm/r1_cache}
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
cd $SC; mkdir -p $OUT/logs
ST=$OUT/logs/options.status; echo "start $(date -u)" > $ST
say() { echo "[options] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; exit 1; }

# 0 ---- inputs (the RAM-disk cache is lost when the pod stops)
[ -f $CACHE/index_nexar_reasoning.jsonl ] || { say pull_r; bash r1_pod_run.sh pull_r || fail pull_r; }
[ -f $OUT/phase1/best.pt ] || { say get_p1; bash r1_pod_run.sh get_p1 || fail get_p1; }
if [ ! -f $OUT/phase2r/epoch_03.pt ]; then
  say get_epoch3
  $PY -c "from huggingface_hub import hf_hub_download as d; import shutil,os; os.makedirs('$OUT/phase2r',exist_ok=True); shutil.copy(d('eviatarO-org/checkpoints','phase2r/epochs/epoch_03.pt',local_dir='$OUT/_hf'),'$OUT/phase2r/epoch_03.pt')" || fail get_epoch3
fi

TR="$PY -u r1_train.py --phase 2 --sources nexar_reasoning --order reason_first --cache-dir $CACHE --init-from $OUT/phase1/best.pt
    --epochs 5 --patience 1 --select val_loss --eff-bs 16 --micro-bs 4 --lr-merger 2e-5 --lr-lora 2e-4 --grad-ckpt"
EV="$PY -u r1_eval_reasoning.py --order reason_first --cache-dir $CACHE --max-ret 200 --n-blank 10"

# A ---- both retrains in parallel (about 10 GB GPU each); opt1 evaluation joins when the GPU has room
say "train opt2_cap + opt3_box"
$TR --out-dir $OUT/opt2_cap --template-cap 5 > $OUT/logs/opt2_train.log 2>&1 & P2=$!
$TR --out-dir $OUT/opt3_box --box-hint      > $OUT/logs/opt3_train.log 2>&1 & P3=$!
sleep 300
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
P1=""
if [ "${FREE:-0}" -gt 14000 ]; then
  say "eval opt1 in parallel (free GPU ${FREE} MiB)"
  $EV --ckpt $OUT/phase2r/epoch_03.pt --out-dir $OUT/eval_opt1 > $OUT/logs/opt1_eval.log 2>&1 & P1=$!
else
  say "GPU free ${FREE} MiB: opt1 evaluation waits for the trainings"
fi
wait $P2 || fail opt2_train; say "opt2_cap trained"
wait $P3 || fail opt3_train; say "opt3_box trained"
if [ -z "$P1" ]; then
  $EV --ckpt $OUT/phase2r/epoch_03.pt --out-dir $OUT/eval_opt1 > $OUT/logs/opt1_eval.log 2>&1 & P1=$!
fi

# B ---- evaluations (two retrains in parallel, opt1 already running or started above)
$EV --ckpt $OUT/opt2_cap/best.pt --out-dir $OUT/eval_opt2_cap > $OUT/logs/opt2_eval.log 2>&1 & E2=$!
wait $P1 || fail opt1_eval; say "opt1 evaluated"
$EV --ckpt $OUT/opt3_box/best.pt --box-hint --out-dir $OUT/eval_opt3_box > $OUT/logs/opt3_eval.log 2>&1 & E3=$!
wait $E2 || fail opt2_eval; say "opt2_cap evaluated"
wait $E3 || fail opt3_eval; say "opt3_box evaluated"

# C ---- push + verify + stop
say push
$PY - <<'PY' >> $ST 2>&1 || fail push_verify
import glob, os, sys
from huggingface_hub import HfApi
a = HfApi(); repo = "eviatarO-org/checkpoints"; out = os.environ.get("OUT", "/root/r1_week1"); base = "options_2026-10-12"
plan = {"opt1_epoch3": [(f"{out}/eval_opt1/{f}", f"results/{f}") for f in ("predictions.jsonl", "metrics.json", "summary.md")],
        "opt2_cap": [(f"{out}/opt2_cap/best.pt", "best.pt"), (f"{out}/opt2_cap/train_log.jsonl", "train_log.jsonl")] +
                    [(f"{out}/eval_opt2_cap/{f}", f"results/{f}") for f in ("predictions.jsonl", "metrics.json", "summary.md")],
        "opt3_box": [(f"{out}/opt3_box/best.pt", "best.pt"), (f"{out}/opt3_box/train_log.jsonl", "train_log.jsonl")] +
                    [(f"{out}/eval_opt3_box/{f}", f"results/{f}") for f in ("predictions.jsonl", "metrics.json", "summary.md")]}
card = ("# options_2026-10-12/\n\nFollow-ups to phase2r (Run 1), plan 2026-10-12_Plan-Phase2-Reasoning-Retrain.\n\n"
        "* `opt1_epoch3/` - Run 1's epoch 3 (lowest validation loss; weights in ../phase2r/epochs/epoch_03.pt) evaluated on the 612 val windows.\n"
        "* `opt2_cap/` - retrain with a template cap (frequent explanation openings down-weighted).\n"
        "* `opt3_box/` - retrain with the position of the attended road user as words in the prompt.\n"
        "Each: `best.pt` (= lowest val loss), `train_log.jsonl`, `results/` (predictions.jsonl, metrics.json, summary.md).\n")
a.upload_file(path_or_fileobj=card.encode(), path_in_repo=f"{base}/README.md", repo_id=repo, repo_type="model")
need = []
for d, files in plan.items():
    for src, dst in files:
        if not os.path.exists(src):
            print("missing", src); sys.exit(1)
        a.upload_file(path_or_fileobj=src, path_in_repo=f"{base}/{d}/{dst}", repo_id=repo, repo_type="model"); need.append((src, f"{base}/{d}/{dst}"))
remote = {x.path: x.size for x in a.get_paths_info(repo, [d for _, d in need], repo_type="model")}
ok = all(remote.get(d) == os.path.getsize(s) for s, d in need)
for s, d in need:
    print(d, os.path.getsize(s), remote.get(d))
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs)
runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"
