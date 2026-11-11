#!/bin/bash
# Hyperparameter / seed sweep as a Slurm job array. Each task trains one (model, seed) cell with
# member-grouped CV on the training split only and logs to the shared MLflow store.
#SBATCH --job-name=predictmed-sweep
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${PREDICTMED_HOME:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH=src
MODELS=(lightgbm catboost random_forest logistic mlp)
MODEL=${MODELS[$((SLURM_ARRAY_TASK_ID % 5))]}
SEED=$((SLURM_ARRAY_TASK_ID / 5))
python -m predictmed.sweep --model "$MODEL" --seed "$SEED"
