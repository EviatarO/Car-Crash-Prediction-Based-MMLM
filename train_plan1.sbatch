#!/bin/bash
#SBATCH --job-name=mca_plan1
#SBATCH --partition=fullservers
#SBATCH --gres=gpu:nvidia_rtx_5000_ada_generation:1
#SBATCH --nodelist=srvr-1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=40:00:00
#SBATCH --nice=10000
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --open-mode=truncate
#SBATCH --signal=TERM@180

set -euo pipefail

# --- Prep dirs ---
ROOT_DIR="/home/eprojuser011/MMLM_CursorAI"
mkdir -p "${ROOT_DIR}/logs" "${ROOT_DIR}/outputs"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mca

# --- Job lifecycle banners (stderr) ---
echo "[JOB-START] id=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown} at $(date -Is)" 1>&2
trap 'echo "[JOB-SIGNAL] SIGTERM for job ${SLURM_JOB_ID} at $(date -Is)" 1>&2' TERM
on_exit(){ rc=$?; echo "[JOB-END]   id=${SLURM_JOB_ID} rc=${rc} at $(date -Is)" 1>&2; }
trap on_exit EXIT

# --- Prevent BLAS oversubscription ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- PyTorch CUDA allocator tweak ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- Run training (unbuffered) ---
cd "${ROOT_DIR}"
PYTHON_BIN="${HOME}/miniconda3/envs/mca/bin/python"
if [ -x "${PYTHON_BIN}" ]; then
  "${PYTHON_BIN}" -u train_sft.py --config configs/base.yaml
else
  if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    conda activate mca
  fi
  python -u train_sft.py --config configs/base.yaml
fi
