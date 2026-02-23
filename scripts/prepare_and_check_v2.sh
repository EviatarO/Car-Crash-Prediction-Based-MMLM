#!/bin/bash
# Run this once before: sbatch slurm/plan3_prompt_experiment_v2.sbatch
# Usage: bash MMLM_CursorAI/scripts/prepare_and_check_v2.sh

set -euo pipefail

ROOT_DIR="/home/eprojuser011/MMLM_CursorAI"
cd "${ROOT_DIR}"

# Ensure conda is available
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

# Create mca env if missing, then install deps
if ! conda env list | grep -q '^mca '; then
  echo "Creating conda env mca..."
  conda create -n mca python=3.10 -y
fi
conda activate mca

echo "Installing/updating Python dependencies..."
pip install -q python-dotenv openai openpyxl matplotlib pandas Pillow

# Quick import check (no API call)
echo "Checking script imports..."
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"
python -c "
from prompts.templates import PROMPT_B, PROMPT_D, PROMPT_F
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image
print('All imports OK.')
"

echo "Done. You can run: sbatch slurm/plan3_prompt_experiment_v2.sbatch"
