#!/bin/bash
#SBATCH --job-name=mca_plan3_v2
#SBATCH --partition=fullservers
#SBATCH --gres=gpu:0
#SBATCH --nodelist=srvr-1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/plan3_v2_%j.out
#SBATCH --error=logs/plan3_v2_%j.err
#SBATCH --open-mode=truncate
#SBATCH --signal=TERM@180

set -euo pipefail

ROOT_DIR="/home/eprojuser011/MMLM_CursorAI"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/outputs/prompt_experiment"

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

${PYTHON} -u scripts/prompt_experiment_v2.py \
  --manifest outputs/teacher_clips_manifest.jsonl \
  --frames_root /home/eprojuser011/Data_Centric_using_3LC_and_MViT/src/projects/NexarChallenge/Nexar_DataSet/train_frames256 \
  --out_dir outputs/prompt_experiment \
  --n_05 5 \
  --n_10 5 \
  --n_15 5 \
  --n_tn 15
