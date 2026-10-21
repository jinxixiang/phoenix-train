#!/bin/bash
#SBATCH --job-name=phx-train
#SBATCH -p gpu
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH -G 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH -o logs/train-%j.out
#SBATCH -e logs/train-%j.out

# Finetune Phoenix on one cohort, then evaluate the held-out slides.
#   sbatch slurm/train.sbatch breast_1
#   sbatch slurm/train.sbatch lung_1 40

set -euo pipefail

COHORT="${1:?usage: sbatch train.sbatch <cohort> [epochs]}"
EPOCHS="${2:-30}"
# Submit from the repository root, or set PHOENIX_TRAIN_ROOT and PYTHON.
PHX="${PHOENIX_TRAIN_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON="${PYTHON:-python}"
mkdir -p "$PHX/logs"

echo "[$(date)] $SLURM_JOB_ID train $COHORT for $EPOCHS epochs on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd "$PHX/finetune"
srun $PYTHON train.py --cohort "$COHORT" --epochs "$EPOCHS" --workers 9
srun $PYTHON evaluate.py --cohort "$COHORT"
echo "[$(date)] done $COHORT"
