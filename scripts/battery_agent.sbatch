#!/bin/bash
# D-061 amendment 2: multi-turn tool-use runner (cm.battery_agent) for one subject, one GPU, HF greedy.
# Usage (from the clone on ada):
#   CM_MODEL=gemma-4-31b-it sbatch scripts/battery_agent.sbatch gate --pair data/raw/DECEPEVAL/.../0201 --question "..."
# All arguments are passed through to cm.battery_agent.
#SBATCH --job-name=battery_agent
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/ada1/u/harsh/character-map/logs/%x_%j.log

set -euo pipefail
cleanup() {
  rc=$?
  pkill -9 -P $$ 2>/dev/null || true
  if [ "$rc" -ne 0 ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    for p in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$p" 2>/dev/null || true
    done
  fi
}
trap cleanup EXIT TERM INT
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export CUDA_CACHE_PATH=/ada1/u/harsh/.cache/nv/ComputeCache
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" /ada1/u/harsh/character-map/logs
pwd; hostname; date; echo "CM_MODEL=${CM_MODEL:?set CM_MODEL}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
nvidia-smi -i "$CUDA_VISIBLE_DEVICES"
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB"; exit 1; }
done
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
cd /ada1/u/harsh/character-map
git log --oneline -1
# MODULE picks the runner: cm.battery_agent (AR, default) or cm.battery_code (DC).
python -u -m "${MODULE:-cm.battery_agent}" "$@"
date; echo "[battery_agent] done $CM_MODEL ${MODULE:-cm.battery_agent}"
