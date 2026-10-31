#!/bin/bash
# CPU runner for the D-064 amendment 1 ruling (d) recompute (the standardization floor). Meant to run from a SEPARATE
# clone on ada (for example /ada1/u/harsh/character-map-robust, with outputs -> the main checkout's outputs by symlink),
# so that the checkout a steering run imports is never touched: it runs the module from the directory it was submitted
# in. Usage, from the clone root (the log goes to <clone>/logs/):
#   CM_MODEL=gemma-4-31b-it sbatch --output=logs/%x_%j.log scripts/robust_floor.sbatch cm.coherence rep --workers 4 --sd-floor-q none,0.1,0.25,0.5
#   CM_MODEL=gemma-4-31b-it sbatch --output=logs/%x_%j.log scripts/robust_floor.sbatch cm.battery_onpolicy directions --sd-floor-q none,0.1,0.25,0.5
#   CM_MODEL=gemma-4-31b-it sbatch --output=logs/%x_%j.log --mem=120G scripts/robust_floor.sbatch cm.battery_onpolicy directions-all --sd-floor-q 0.5
#SBATCH --job-name=robust_floor
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
trap 'pkill -9 -P $$ 2>/dev/null || true' EXIT TERM INT
REPO="${SLURM_SUBMIT_DIR:?submit from the clone root}"
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface HF_HUB_OFFLINE=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache TMPDIR=/ada1/u/harsh/tmp
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p "$TMPDIR" "$REPO/logs"
hostname; date; echo "CM_MODEL=${CM_MODEL:?set CM_MODEL}"; echo "repo $REPO"
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
cd "$REPO"
git log --oneline -1
MOD="$1"; shift
python -u -m "$MOD" "$@"
date; echo "[robust_floor] done $MOD $*"
