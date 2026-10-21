#!/bin/bash
#SBATCH --job-name=phx-spatch
#SBATCH -p gpu
#SBATCH -C "GPU_MEM:48GB|GPU_MEM:80GB"
#SBATCH -G 1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH -o logs/spatch-%j.out
#SBATCH -e logs/spatch-%j.out

# Extract -> finetune -> evaluate, for one or more SPATCH cohorts in sequence.
# Each cohort is a single ~25k-cell slide (folds 0-3 train, fold 4 held out), so
# a cohort costs roughly 5 min extraction + 30 min training + 15 min sampling.
# Cohorts are packed into one job to keep the number of queued jobs small.
#
#   sbatch slurm/spatch.sbatch spatch_xenium_coad spatch_xenium_hcc
#
# Every stage is skipped when its output is already present, so a preempted and
# requeued job resumes rather than redoing work (training itself also resumes
# from last.ckpt).

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "usage: sbatch spatch.sbatch <cohort> [<cohort> ...]" >&2
    exit 2
fi

EPOCHS="${EPOCHS:-30}"
# Submit from the repository root, or set PHOENIX_TRAIN_ROOT and PYTHON.
PHX="${PHOENIX_TRAIN_ROOT:-$SLURM_SUBMIT_DIR}"
PYTHON="${PYTHON:-python}"
mkdir -p "$PHX/logs"
cd "$PHX/finetune"

echo "[$(date)] job $SLURM_JOB_ID on $(hostname), cohorts: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for COHORT in "$@"; do
    SLIDE=$($PYTHON -c "
import sys; sys.path.insert(0, '.')
from common import load_splits
print(load_splits('$COHORT')['val'][0])
")
    RUN="$PHX/outputs/runs/$COHORT"

    echo "[$(date)] === $COHORT ($SLIDE)"
    # File size is not a completeness test -- `feat` is pre-allocated, so a
    # killed extraction leaves ~18 GB of zeros behind. `features_complete`
    # checks the attributes and the last row instead.
    if $PYTHON -c "
import sys; sys.path.insert(0, '.')
from common import features_complete
sys.exit(0 if features_complete('$COHORT', '$SLIDE') else 1)
"; then
        echo "[$(date)] features complete, skipping extraction"
    else
        srun $PYTHON extract_features.py --cohort "$COHORT" --slide "$SLIDE" --workers 9
    fi

    if [ -s "$RUN/flow_model_finetuned.pth" ]; then
        echo "[$(date)] finetuned weights present, skipping training"
    else
        srun $PYTHON train.py --cohort "$COHORT" --epochs "$EPOCHS" --workers 9
    fi

    if [ -s "$RUN/pred_$SLIDE.npz" ] && [ -s "$RUN/eval.json" ]; then
        echo "[$(date)] predictions present, skipping evaluation"
    else
        srun $PYTHON evaluate.py --cohort "$COHORT"
    fi
    echo "[$(date)] done $COHORT"
done

echo "[$(date)] job $SLURM_JOB_ID complete"
