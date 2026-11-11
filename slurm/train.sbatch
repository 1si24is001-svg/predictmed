#!/bin/bash
# Training + governance as one reproducible Slurm job (CPU is sufficient for tabular models).
#SBATCH --job-name=predictmed-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${PREDICTMED_HOME:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH=src
echo "job=$SLURM_JOB_ID node=$(hostname) cpus=$SLURM_CPUS_PER_TASK commit=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
python -m predictmed.pipeline train
