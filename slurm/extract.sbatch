#!/bin/bash
#SBATCH --job-name=phx-extract
#SBATCH -p gpu
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH -G 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH -o logs/extract-%A_%a.out
#SBATCH -e logs/extract-%A_%a.out

# Cache Phoenix image conditions, one array task per slide.
#   sbatch --array=0-11  slurm/extract.sbatch breast_1
#   sbatch --array=0-19  slurm/extract.sbatch lung_1

set -euo pipefail

COHORT="${1:?usage: sbatch --array=... extract.sbatch <cohort>}"
# Submit from the repository root, or set PHOENIX_TRAIN_ROOT and PYTHON.
PHX="${PHOENIX_TRAIN_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON="${PYTHON:-python}"
mkdir -p "$PHX/logs"

SLIDE=$($PYTHON -c "
import sys; sys.path.insert(0, '$PHX/finetune')
from common import slide_ids
print(slide_ids('$COHORT')[$SLURM_ARRAY_TASK_ID])
")

echo "[$(date)] $SLURM_JOB_ID task $SLURM_ARRAY_TASK_ID: $COHORT/$SLIDE on $(hostname)"
cd "$PHX/finetune"
srun $PYTHON extract_features.py --cohort "$COHORT" --slide "$SLIDE" --workers 9
echo "[$(date)] done $SLIDE"
