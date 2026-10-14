#!/bin/bash
# Unattended text-augmentation chain (plan addendum 2026-10-14): automatic check -> Sonnet judge -> ONE Gemini rewrite round for the windows with
# the fewest good versions, capped by the OpenRouter balance ($1 reserve; about $0.004 per window for writing + judging 3 versions) -> finalize -> audit sample.
set -u
export PYTHONUTF8=1 PYTHONUNBUFFERED=1
cd "$(dirname "$0")"
L=../../outputs/teacher_dataset_2026-10/pipeline.log
step() { echo "== $* $(date)" | tee -a $L; }
step check;    python r1_paraphrase_check.py check >> $L 2>&1 || exit 1
step judge;    python r1_paraphrase_judge.py judge --concurrency 16 >> $L 2>&1 || exit 1
python r1_paraphrase_judge.py judge --concurrency 8 >> $L 2>&1                 # unanswered ones
step finalize; python r1_paraphrase_judge.py finalize >> $L 2>&1 || exit 1
B=$(python r1_or_balance.py); N=$(python -c "print(max(0, int(($B - 1.0) / 0.0045)))")
step "balance \$$B -> rewrite up to $N windows"
if [ "$N" -gt 0 ]; then
  python r1_paraphrase_gemini.py --rewrite --limit $N --concurrency 16 >> $L 2>&1 || exit 1
  step check;    python r1_paraphrase_check.py check >> $L 2>&1 || exit 1
  B=$(python r1_or_balance.py); M=$(python -c "print(max(0, int(($B - 0.5) / 0.0011)))")
  step "judge rewrites (balance \$$B, up to $M versions)"
  python r1_paraphrase_judge.py judge --concurrency 16 --limit $M >> $L 2>&1 || exit 1
  step finalize; python r1_paraphrase_judge.py finalize >> $L 2>&1 || exit 1
fi
step sample;   python r1_paraphrase_judge.py sample --n 300 >> $L 2>&1 || exit 1
step DONE
