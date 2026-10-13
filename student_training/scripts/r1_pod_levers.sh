#!/bin/bash
# Session 1 of plan 2026-10-13_Plan-Reasoning-Diagnose-and-Levers: richer inputs for the language-model bridge. Unattended; stops the pod ONLY after
# every result file is verified byte-identical on HF. Needs a pod with >= 100 GB container disk (3 feature caches of 17.5 GB each).
#   B3  : input 'last'           new coarse stage (control: same input, fresh Phase 1 on the Nexar reasoning texts)
#   B4  : input 'last+attn'      + the crash head's attention per token
#   B5  : input 'l11+last'       + layer-11 tokens
#   B45 : input 'l11+last+attn'  both
# Each run = Phase 1 (projector only, <= 2 epochs) -> Phase 2 (projector + LoRA, <= 5 epochs), best = lowest validation loss, stop at the first epoch
# without improvement; then r1_eval_reasoning (train-chosen threshold). Pairs run two at a time on the one GPU (about 10 GB each).
#   setsid nohup bash r1_pod_levers.sh > /root/r1_week1/logs/levers.nohup 2>&1 < /dev/null &      status: $OUT/logs/levers.status
set -u
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONUNBUFFERED=1
OUT=${OUT:-/root/r1_week1}; PY=${PY:-/root/venv/bin/python}; SC=/root/r1/student_training/scripts
R_NEXR=eviatarO-org/nexar-reasoning-windows; R_FEAT=eviatarO-org/vjepa2-a1-features; R_CKPT=eviatarO-org/checkpoints
for TOKEN_FILE in /root/.cache/huggingface/token /workspace/.cache/huggingface/token; do
  [ -f "$TOKEN_FILE" ] && export HF_TOKEN=$(cat "$TOKEN_FILE") && break
done
cd $SC; mkdir -p $OUT/logs $OUT/probe
ST=$OUT/logs/levers.status; echo "start $(date -u)" > $ST
say() { echo "[levers] $* $(date -u)" | tee -a $ST; }
fail() { say "FAILED: $* - pod left running"; exit 1; }
FREE_GB=$(df -BG --output=avail /root | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge 90 ] || fail "container disk has only ${FREE_GB} GB free (need >= 90)"
export CACHE=$OUT/cache; mkdir -p $CACHE

# 0 ---- inputs: last-layer cache, then the extras pass (layer 11 + attention) and its consolidation
say "pull_r (last-layer cache)"
[ -f $CACHE/index_nexar_reasoning.jsonl ] || $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --out-dir $CACHE > $OUT/logs/pull_r.log 2>&1 || fail pull_r
say "extras pass (encode again: layer 11 + attention; crash-model P must reproduce)"
$PY -u r1_cache_features.py --source nexar_reasoning --from-hf $R_NEXR --splits train,val --push-repo $R_FEAT --extras-only --out-dir $OUT/feat_x > $OUT/logs/extras.log 2>&1 || fail extras
say "pull extras"
[ -f $CACHE/attn_nexar_reasoning.npy ] || $PY -u r1_cache_features.py --source nexar_reasoning --consolidate-from $R_FEAT --with-extras --out-dir $CACHE > $OUT/logs/pull_x.log 2>&1 || fail pull_x
$PY - <<'PY' >> $ST 2>&1 || fail cache_counts
import json, numpy as np, os
c = os.environ["CACHE"]; n = sum(1 for _ in open(f"{c}/index_nexar_reasoning.jsonl"))
a = np.load(f"{c}/attn_nexar_reasoning.npy", mmap_mode="r"); l = np.load(f"{c}/features_nexar_reasoning_l11.npy", mmap_mode="r")
assert n == 4169 and a.shape == (4169, 2048) and l.shape[0] == 4169, (n, a.shape, l.shape)
assert float(np.abs(np.asarray(a[4000])).sum()) > 0 and float(np.abs(np.asarray(l[4000])).max()) > 0, "extras empty"
print("[levers] cache ok:", n, a.shape, l.shape)
PY

# 1 ---- probe: are the facts in the tokens? (Run 1's text agreement alongside)
say "fact probe"
$PY -c "from huggingface_hub import hf_hub_download as d; import shutil; shutil.copy(d('$R_CKPT','phase2r/results/predictions.jsonl',local_dir='$OUT/_hf'),'$OUT/run1_predictions.jsonl')" || say "run1 predictions not fetched"
$PY -u r1_fact_probe.py --cache-dir $CACHE --out-dir $OUT/probe --inputs last,l11,attn --runs run1=$OUT/run1_predictions.jsonl > $OUT/logs/probe.log 2>&1 || say "probe failed (continuing; see probe.log)"

# 2 ---- the four runs, two at a time
run() {   # run NAME INPUT
  N=$1; I=$2
  $PY -u r1_train.py --phase 1 --sources nexar_reasoning --input $I --cache-dir $CACHE --out-dir $OUT/$N/p1 --init random --epochs 2 --patience 1 \
      --select val_loss --eff-bs 16 --micro-bs 4 --lr-merger 1e-3 --grad-ckpt > $OUT/logs/${N}_p1.log 2>&1 || return 1
  $PY -u r1_train.py --phase 2 --sources nexar_reasoning --input $I --cache-dir $CACHE --out-dir $OUT/$N/p2 --init-from $OUT/$N/p1/best.pt --epochs 5 --patience 1 \
      --select val_loss --eff-bs 16 --micro-bs 4 --lr-merger 2e-5 --lr-lora 2e-4 --grad-ckpt > $OUT/logs/${N}_p2.log 2>&1 || return 1
  $PY -u r1_eval_reasoning.py --ckpt $OUT/$N/p2/best.pt --input $I --order reason_first --cache-dir $CACHE --out-dir $OUT/$N/eval \
      --max-ret 200 --n-blank 10 --train-threshold 600 > $OUT/logs/${N}_eval.log 2>&1 || return 1
}
push() {   # push NAME : checkpoints + logs + results to HF (best.pt of both phases)
  $PY - "$1" <<'PY' >> $ST 2>&1 || fail "push $1"
import glob, os, sys
from huggingface_hub import HfApi
n = sys.argv[1]; out = os.environ.get("OUT", "/root/r1_week1"); a = HfApi(); repo = "eviatarO-org/checkpoints"; base = f"levers_2026-10-13/{n}"
files = [(f"{out}/{n}/p1/best.pt", "p1/best.pt"), (f"{out}/{n}/p1/train_log.jsonl", "p1/train_log.jsonl"), (f"{out}/{n}/p2/best.pt", "p2/best.pt"),
        (f"{out}/{n}/p2/train_log.jsonl", "p2/train_log.jsonl")] + [(f, f"results/{os.path.basename(f)}") for f in sorted(glob.glob(f"{out}/{n}/eval/*")) if os.path.isfile(f)]
for s, d in files:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=repo, repo_type="model")
print("[levers] pushed", n, len(files), "files")
PY
}
say "pair 1: B3 (last) || B4 (last+attn)"
run B3 last & P1=$!
run B4 last+attn & P2=$!
wait $P1 || fail B3; wait $P2 || fail B4
push B3; push B4; say "pair 1 pushed"
say "pair 2: B5 (l11+last) || B45 (l11+last+attn)"
run B5 l11+last & P3=$!
run B45 l11+last+attn & P4=$!
wait $P3 || fail B5; wait $P4 || fail B45
push B5; push B45; say "pair 2 pushed"

# 3 ---- probe results + summaries -> HF, verify, stop
$PY - <<'PY' >> $ST 2>&1 || fail final_push_verify
import glob, os, sys
from huggingface_hub import HfApi
out = os.environ.get("OUT", "/root/r1_week1"); a = HfApi(); repo = "eviatarO-org/checkpoints"; base = "levers_2026-10-13"
extra = [(f"{out}/probe/fact_probe.json", "probe/fact_probe.json"), (f"{out}/logs/probe.log", "probe/probe.log"), (f"{out}/logs/levers.status", "levers.status")]
extra = [(s, d) for s, d in extra if os.path.exists(s)]
for s, d in extra:
    a.upload_file(path_or_fileobj=s, path_in_repo=f"{base}/{d}", repo_id=repo, repo_type="model")
need = [(s, f"{base}/{d}") for s, d in extra[:2] if os.path.exists(s)]
for n in ("B3", "B4", "B5", "B45"):
    need += [(f"{out}/{n}/p2/best.pt", f"{base}/{n}/p2/best.pt"), (f"{out}/{n}/eval/predictions.jsonl", f"{base}/{n}/results/predictions.jsonl"),
             (f"{out}/{n}/eval/metrics.json", f"{base}/{n}/results/metrics.json")]
remote = {x.path: x.size for x in a.get_paths_info(repo, [d for _, d in need], repo_type="model")}
ok = all(os.path.exists(s) and remote.get(d) == os.path.getsize(s) for s, d in need)
for s, d in need:
    print(d, os.path.getsize(s) if os.path.exists(s) else None, remote.get(d))
print("VERIFIED" if ok else "NOT VERIFIED"); sys.exit(0 if ok else 1)
PY
say "verified on HF; stopping the pod"
export $(tr '\0' '\n' < /proc/1/environ | grep -E '^(RUNPOD_API_KEY|RUNPOD_POD_ID)=' | xargs)
runpodctl stop pod "$RUNPOD_POD_ID" >> $ST 2>&1; say "stop returned $?"
