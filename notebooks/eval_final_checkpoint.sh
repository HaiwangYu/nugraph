#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph/nugraph:/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph:${PYTHONPATH:-}"

PYTHON_EXE="${PYTHON_EXE:-/lus/eagle/projects/neutrinoGPU/abhat/conda/envs/nugraph-a-sophia/bin/python}"
DATA="${DATA:-/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/converted_labeled_samples_merged_350k.h5}"
LOGDIR="${LOGDIR:-/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph/notebooks/log}"
SHUFFLE_MODE="${SHUFFLE_MODE:-random}"
DATA_TAG="$(basename "${DATA}" .h5)"

RUN_NAME="${RUN_NAME:-N4_nw0_bs_4_lr3e4_nuhits_0_bf0p1_tf1p0_if4_hf256_nf64_intf32_nit10_shuffle_${SHUFFLE_MODE}_ledg0p01_epw0p15_lemb0p3_lcoh0_${DATA_TAG}_sophia}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
CKPT="${CKPT:-${LOGDIR}/${RUN_NAME}/checkpoints/${CKPT_NAME}}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
EDGE_THR="${EDGE_THR:-0.5}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-2}"
LIMIT_ARG=()

if [[ -n "${NUM_EVENTS:-}" && -z "${LIMIT:-}" ]]; then
  LIMIT=$(( (NUM_EVENTS + BATCH_SIZE - 1) / BATCH_SIZE ))
fi

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi

echo "Evaluating checkpoint: ${CKPT}"
echo "Data: ${DATA}"
echo "Split: ${SPLIT}"
if [[ -n "${LIMIT:-}" ]]; then
  echo "Limit: ${LIMIT} batches (~$(( LIMIT * BATCH_SIZE )) events at batch_size=${BATCH_SIZE})"
fi

"${PYTHON_EXE}" eval_semantic.py \
  --ckpt "${CKPT}" \
  --data-path "${DATA}" \
  --split "${SPLIT}" \
  --model nugraph4 \
  --in-features 4 \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --eval-edges \
  --eval-instances \
  --edge-thr "${EDGE_THR}" \
  --min-cluster-size "${MIN_CLUSTER_SIZE}" \
  "${LIMIT_ARG[@]}"
