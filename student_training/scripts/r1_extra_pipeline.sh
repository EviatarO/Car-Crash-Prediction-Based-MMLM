#!/bin/bash
# Unattended teacher run on the 874 extra crash windows (TTE 0.75 / 1.25 s), plan 2026-10-19_Plan-Extra-TTE-Windows-Retrain, steps 2-4. Runs on the laptop.
# Everything is resumable (re-run the script: finished steps are skipped by their own checks). Every paid step is preceded by a budget check and followed by a gate
# (r1_extra_gates.py, defined in advance): a failed gate stops the run; nothing further is spent.
#   0 wait for the attention stage (r1_attention_boxes.py --stage attn, started separately)   1 red boxes + boxed frames   2 E3 blind first look (Gemini 3.8 Flash, Flex)
#   3 flow: second looks (E6 anchored on a correct sibling / E7 hindsight / E6C chain)         4 outcome sentences          5 dataset + manifests
#   6 rewordings: Gemini writer -> check -> Sonnet judge -> capped rewrite round -> audit sample (own folder paraphrase/)    7 fact cards    8 merged files for the training
# Status: outputs/teacher_extra_2026-10/pipeline.status (one line per step); details: pipeline.log.
#   bash r1_extra_pipeline.sh        (from student_training/scripts)
set -u
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
W=$(cygpath -m "$ROOT")/outputs/teacher_extra_2026-10          # mixed path (C:/...): valid in bash and in Windows python
D=$(cygpath -m "$ROOT")/outputs/teacher_dataset_2026-10
MANIFESTS=$(cygpath -m "$ROOT")/dataset/manifests
export R1_PILOT_OUT="$W"
MODEL=google/gemini-3.8-flash
ST="$W/pipeline.status"; L="$W/pipeline.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$ST"; }
stop() { say "STOPPED: $* (nothing further runs; details in pipeline.log)"; exit 2; }
run() { n=$1; shift; say "start $n"; "$@" >> "$L" 2>&1 || stop "$n exited with an error"; say "done $n"; }
gate() { r=$(python r1_extra_gates.py "$@" 2>&1 | tail -1); say "gate $*: $r"; case "$r" in PASS*) ;; *) stop "gate $* failed" ;; esac; }
N=874
say "pipeline started"

# 0 ---- attention
until [ "$(ls "$W/attn" | grep -cE '_tte(075|125)\.json$')" -ge $N ]; do sleep 120; done
say "attention complete ($N new windows)"

# 1 ---- boxes
if [ ! -s "$W/boxes.jsonl" ] || [ "$(grep -c '_tte075\|_tte125' "$W/boxes.jsonl")" -lt $N ]; then
  run boxes python r1_attention_boxes.py --stage boxes --windows-file "$W/windows.jsonl"
fi
gate boxes

# 2 ---- E3 (blind first look) on the new windows only (the existing ones are already in runs/E3.jsonl)
gate budget 14
run E3 python r1_teacher_pilot.py --name E3 --prompt e3 --set full --tier flex --model $MODEL --concurrency 32
run E3_retry python r1_teacher_pilot.py --name E3 --prompt e3 --set full --tier flex --model $MODEL --concurrency 32      # re-runs the failed calls
gate e3

# 3 ---- flow
gate budget 9
run flow python r1_teacher_flow.py --set ALL --model $MODEL --tier flex --concurrency 16 --process-horizons 0.75,1.25
gate flow

# 4 ---- outcome sentences (text only)
gate budget 4
for R in E3 E6 E6C E7; do run "outcome_$R" python r1_add_prediction.py --run $R --model $MODEL --tier flex --concurrency 16; done

# 5 ---- dataset + manifests
run dataset python r1_extra_merge.py dataset
run manifest python r1_extra_merge.py manifest

# 6 ---- rewordings (own folder; the original paraphrase files are never touched)
gate budget 7
export R1_PARA_DIR="$W/paraphrase" R1_REASONING_MANIFEST="$MANIFESTS/r1_nexar_reasoning_extra.jsonl"
mkdir -p "$R1_PARA_DIR"
run para_export python r1_paraphrase_check.py export
run para_writer python r1_paraphrase_gemini.py --concurrency 24
run para_pipeline bash r1_paraphrase_pipeline.sh
gate rewords

# 7 ---- fact cards of the new teacher texts
gate budget 1.5
run fact_cards python r1_fact_cards.py --source teacher --out "$W/fact_cards_extra.jsonl"

# 8 ---- merged files for the training (originals untouched)
run merge_cards bash -c "cat '$D/fact_cards_teacher.jsonl' '$W/fact_cards_extra.jsonl' > '$W/fact_cards_v2.jsonl'"
run merge_texts bash -c "cat '$D/paraphrases_judged.jsonl' '$W/paraphrase/paraphrases_judged.jsonl' > '$W/paraphrases_judged_v2.jsonl'"
say "ALL DONE: dataset_final.jsonl, manifests r1_nexar_reasoning_extra / _v2, paraphrases_judged_v2.jsonl, fact_cards_v2.jsonl; next: the Opus audit of paraphrase/paraphrase_audit_sample.jsonl"
