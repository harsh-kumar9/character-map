#!/bin/bash
# CPU-only runner for character-map analysis on mira/vega (no GPU). Usage: sbatch scripts/cm_cpu.sbatch <module> [args...]
# Harsh 2026-10-20: analysis runs through SLURM on mira/vega, not directly on ada.
#SBATCH --job-name=cm_cpu
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ada1/u/harsh/character-map/logs/%x_%j.log

set -euo pipefail
trap 'pkill -9 -P $$ 2>/dev/null || true' EXIT TERM INT
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache TMPDIR=/ada1/u/harsh/tmp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR" /ada1/u/harsh/character-map/logs
hostname; date
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
cd /ada1/u/harsh/character-map
git log --oneline -1
MOD="$1"; shift
python -u -m "$MOD" "$@"
