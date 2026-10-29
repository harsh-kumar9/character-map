#!/bin/bash
# D-061 battery, CPU-only stages (item builds, clean re-runs, scoring) as a SLURM job instead of on ada's login node.
# Usage (from the clone on ada):  CM_MODEL=gemma-4-31b-it sbatch scripts/battery_cpu.sbatch cm.battery_code build
# First argument: the module; the rest is passed through.
#SBATCH --job-name=battery_cpu
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ada1/u/harsh/character-map/logs/%x_%j.log

set -euo pipefail
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$TMPDIR" /ada1/u/harsh/character-map/logs
pwd; hostname; date; echo "CM_MODEL=${CM_MODEL:?set CM_MODEL}"
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
cd /ada1/u/harsh/character-map
git log --oneline -1
MODULE="$1"; shift
python -u -m "$MODULE" "$@"
date; echo "[battery_cpu] done $MODULE"
