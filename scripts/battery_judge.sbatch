#!/bin/bash
# D-061 battery judging for one subject with one judge (vLLM). Usage: CM_MODEL=<subject> sbatch scripts/battery_judge.sbatch gemma|qwen [cond]
#SBATCH --job-name=battery_judge
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
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
JUDGE="${1:?gemma|qwen}"; COND="${2:-baseline}"
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export CUDA_CACHE_PATH=/ada1/u/harsh/.cache/nv/ComputeCache FLASHINFER_WORKSPACE_BASE=/ada1/u/harsh/.cache
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor FLASHINFER_CACHE_DIR=/ada1/u/harsh/.cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$TMPDIR" /ada1/u/harsh/character-map/logs
pwd; hostname; date; echo "CM_MODEL=${CM_MODEL:?set CM_MODEL} judge=$JUDGE"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB"; exit 1; }
done
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
cd /ada1/u/harsh/character-map
git log --oneline -1
# MODULE picks the task set: cm.battery (seven single-turn tasks, default), cm.battery_agent (AR), cm.battery_code (DC).
python -u -m "${MODULE:-cm.battery}" judge --judge "$JUDGE" --cond "$COND"
date; echo "[battery_judge] done $CM_MODEL $JUDGE $COND"
