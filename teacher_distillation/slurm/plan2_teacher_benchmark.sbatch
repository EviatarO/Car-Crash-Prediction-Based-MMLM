#!/bin/bash
#SBATCH --job-name=mca_plan2
#SBATCH --partition=fullservers
#SBATCH --gres=gpu:0
#SBATCH --nodelist=srvr-1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/plan2_%j.out
#SBATCH --error=logs/plan2_%j.err
#SBATCH --open-mode=truncate
#SBATCH --signal=TERM@180

set -euo pipefail

ROOT_DIR="/home/eprojuser011/MMLM_CursorAI"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/outputs/benchmark"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mca

echo "[JOB-START] id=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} at $(date -Is)" 1>&2
trap 'echo "[JOB-SIGNAL] SIGTERM for job ${SLURM_JOB_ID} at $(date -Is)" 1>&2' TERM
on_exit(){ rc=$?; echo "[JOB-END]   id=${SLURM_JOB_ID} rc=${rc} at $(date -Is)" 1>&2; }
trap on_exit EXIT

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"

set -a
source "${ROOT_DIR}/.env"
set +a

PYTHON_BIN="${HOME}/miniconda3/envs/mca/bin/python"
if [ -x "${PYTHON_BIN}" ]; then
  PYTHON="${PYTHON_BIN}"
else
  PYTHON="python"
fi

${PYTHON} - <<'PY'
import json
import os
import sys
import time

log_path = "/home/eprojuser011/.cursor/debug.log"
timestamp = int(time.time() * 1000)

def write_log(message, data, hypothesis_id):
    payload = {
        "runId": os.environ.get("SLURM_JOB_ID", "manual"),
        "hypothesisId": hypothesis_id,
        "location": "plan2_teacher_benchmark.sbatch:deps",
        "message": message,
        "data": data,
        "timestamp": timestamp,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

write_log("pip_install_start", {"requirements": "requirements.txt"}, "H3")
PY

${PYTHON} -m pip install -r requirements.txt

${PYTHON} - <<'PY'
import json
import os
import sys
import time

log_path = "/home/eprojuser011/.cursor/debug.log"
timestamp = int(time.time() * 1000)

def write_log(message, data, hypothesis_id):
    payload = {
        "runId": os.environ.get("SLURM_JOB_ID", "manual"),
        "hypothesisId": hypothesis_id,
        "location": "plan2_teacher_benchmark.sbatch:preflight",
        "message": message,
        "data": data,
        "timestamp": timestamp,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

write_log(
    "python_env_info",
    {
        "python_exe": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_path_0": sys.path[0],
        "cwd": os.getcwd(),
    },
    "H1",
)

try:
    import dotenv  # noqa: F401
    write_log("dotenv_import_ok", {}, "H2")
except Exception as exc:
    write_log("dotenv_import_failed", {"error": str(exc)}, "H2")

try:
    import prompts  # noqa: F401
    write_log("prompts_import_ok", {}, "H3")
except Exception as exc:
    write_log("prompts_import_failed", {"error": str(exc)}, "H3")
PY

${PYTHON} -u scripts/teacher_benchmark.py \
  --manifest outputs/teacher_clips_blind.jsonl \
  --frames_root /home/eprojuser011/Data_Centric_using_3LC_and_MViT/src/projects/NexarChallenge/Nexar_DataSet/train_frames256 \
  --out_dir outputs/benchmark \
  --progress_jsonl outputs/benchmark/progress.jsonl

${PYTHON} -u scripts/benchmark_analysis.py \
  --results_dir outputs/benchmark \
  --manifest outputs/teacher_clips_manifest.jsonl \
  --out_dir outputs/benchmark
