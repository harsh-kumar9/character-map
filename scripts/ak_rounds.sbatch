#!/bin/bash
# Group A/K builds that need several gemma-4 rounds (generate, then verify) in one queued job on one GPU, so a busy
# queue is waited on once. Each round: assemble (writes outputs if nothing is pending, else writes requests), pool the
# requests still missing a response, generate, unpool. Stops when a round has nothing to do, then validates and writes
# outputs/stimuli/_gen/<tag>/DONE. Resumable after preemption (generate skips answered requests).
# Usage (from the clone on ada):
#   sbatch scripts/ak_rounds.sbatch <tag> <modules,comma> <batches,comma> <concepts to validate,comma>
#   sbatch scripts/ak_rounds.sbatch a2fa a2 A2_sr,A2_fa A2
#SBATCH --job-name=ak_rounds
#SBATCH --partition=ashton
#SBATCH --qos=ashton
#SBATCH --exclude=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.log

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
TAG="${1:?tag}"; MODULES="${2:?modules}"; BATCHES="${3:?batches}"; CIDS="${4:?concepts}"
export HF_HOME=/ada1/u/harsh/.cache/huggingface HF_HUB_CACHE=/ada1/u/harsh/.cache/huggingface
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1
export XDG_CACHE_HOME=/ada1/u/harsh/.cache XDG_CONFIG_HOME=/ada1/u/harsh/.config TMPDIR=/ada1/u/harsh/tmp
export CUDA_CACHE_PATH=/ada1/u/harsh/.cache/nv/ComputeCache FLASHINFER_WORKSPACE_BASE=/ada1/u/harsh/.cache
export VLLM_CACHE_ROOT=/ada1/u/harsh/.cache/vllm TRITON_CACHE_DIR=/ada1/u/harsh/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=/ada1/u/harsh/.cache/torchinductor FLASHINFER_CACHE_DIR=/ada1/u/harsh/.cache/flashinfer
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$TMPDIR" logs
hostname; date
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "FATAL: no GPU binding"; exit 1; }
for USED in $(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits); do
  [ "$USED" -gt 8000 ] && { echo "FATAL: allocated GPU holds ${USED} MiB"; exit 1; }
done
set +u
eval "$(/ada1/u/harsh/miniconda3/bin/conda shell.bash hook)"
conda activate sote
set -u
GEN=outputs/stimuli/_gen
mkdir -p "$GEN/$TAG"
rm -f "$GEN/$TAG/DONE"
for ROUND in 1 2 3 4 5 6; do
  for M in ${MODULES//,/ }; do python -u -m "cm.stimuli.build.$M" assemble; done
  P="${TAG}_r${ROUND}"
  python -u -m cm.stimuli.build.ak_common pool-missing "$P" ${BATCHES//,/ }
  N=$(wc -l < "$GEN/$P/requests.jsonl")
  echo "[ak_rounds] round $ROUND: $N requests"
  [ "$N" -eq 0 ] && break
  python -u -m cm.stimuli.generate "$P" --shard 0 --nshards 1
  python -u -m cm.stimuli.build.ak_common unpool "$P"
done
python -u -m cm.stimuli.validate "$CIDS" | tee "$GEN/$TAG/validate.txt" || true
date > "$GEN/$TAG/DONE"
