#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph/nugraph:/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph:${PYTHONPATH:-}"

PYTHON_EXE="${PYTHON_EXE:-/lus/eagle/projects/neutrinoGPU/abhat/conda/envs/nugraph-a-sophia/bin/python}"
DATA="${DATA:-/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/converted_labeled_samples_1_geom_edges_full_25k.h5}"
LOGDIR="${LOGDIR:-/lus/eagle/projects/neutrinoGPU/abhat/sbnd/clustering/nugraph/notebooks/log}"
SHUFFLE_MODE="${SHUFFLE_MODE:-random}"
DATA_TAG="$(basename "${DATA}" .h5)"

RUN_NAME="${RUN_NAME:-N4_nw0_bs_4_lr3e4_nuhits_0_bf0p1_tf1p0_if4_hf256_nf64_intf32_nit10_shuffle_${SHUFFLE_MODE}_ledg0p01_epw1p0_lemb0p3_lcoh0_${DATA_TAG}_sophia}"
CKPT_NAME="${CKPT_NAME:-last.ckpt}"
CKPT="${CKPT:-${LOGDIR}/${RUN_NAME}/checkpoints/${CKPT_NAME}}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda}"
EDGE_THR="${EDGE_THR:-0.5}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-2}"
LIMIT_ARG=()
NU_THR_ARG=()
BETA_ARG=()
EVENT_ARGS=()
EDGE_ARGS=()
INSTANCE_ARGS=()
SEMANTIC_ARGS=()
MAX_EDGES_ARG=()

if [[ -n "${NUM_EVENTS:-}" && -z "${LIMIT:-}" ]]; then
  LIMIT=$(( (NUM_EVENTS + BATCH_SIZE - 1) / BATCH_SIZE ))
fi

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi

if [[ -n "${NU_THR:-}" ]]; then
  NU_THR_ARG=(--nu-thr "${NU_THR}")
fi

if [[ -n "${BETA:-}" ]]; then
  BETA_ARG=(--beta "${BETA}")
fi

if [[ "${SKIP_SEMANTICS:-0}" != "0" ]]; then
  SEMANTIC_ARGS=(--skip-semantics)
fi

if [[ -n "${MAX_EDGES:-}" ]]; then
  MAX_EDGES_ARG=(--max-edges "${MAX_EDGES}")
fi

EVENT_ARGS=(
  --event-min-pred-nu-hits "${EVENT_MIN_PRED_NU_HITS:-1}"
  --event-min-pred-nu-frac "${EVENT_MIN_PRED_NU_FRAC:-0.0}"
)

if [[ "${EVAL_EDGES:-1}" != "0" ]]; then
  EDGE_ARGS=(--eval-edges)
fi

if [[ "${EVAL_INSTANCES:-1}" != "0" ]]; then
  INSTANCE_ARGS=(--eval-instances)
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
  "${NU_THR_ARG[@]}" \
  "${BETA_ARG[@]}" \
  "${SEMANTIC_ARGS[@]}" \
  "${EVENT_ARGS[@]}" \
  "${EDGE_ARGS[@]}" \
  "${INSTANCE_ARGS[@]}" \
  --edge-thr "${EDGE_THR}" \
  --min-cluster-size "${MIN_CLUSTER_SIZE}" \
  "${MAX_EDGES_ARG[@]}" \
  "${LIMIT_ARG[@]}"
