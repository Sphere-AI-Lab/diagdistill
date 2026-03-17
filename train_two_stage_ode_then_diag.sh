#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")"

OUTPUT_ROOT="outputs"
STAGE1_CONFIG="configs/exp_stage1_all4_odeinit.yaml"
STAGE2_TEMPLATE="configs/exp_stage2_diag_from_stage1.yaml"
EXP_TS="$(date +%Y%m%d_%H%M%S)"
STAGE1_DIR="${OUTPUT_ROOT}/${EXP_TS}_stage1_all4_odeinit"
STAGE2_DIR="${OUTPUT_ROOT}/${EXP_TS}_stage2_diag"
MASTER_PORT_1="${MASTER_PORT_1:-$((20000 + RANDOM % 20000))}"
MASTER_PORT_2="${MASTER_PORT_2:-$((20000 + RANDOM % 20000))}"
export WANDB_MODE=disabled

echo "[Stage1] config=${STAGE1_CONFIG}"
echo "[Stage1] output=${STAGE1_DIR}"
mkdir -p "${STAGE1_DIR}" "${STAGE2_DIR}"
cp "${STAGE1_CONFIG}" "${STAGE1_DIR}/train_config_input.yaml"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=$(pwd)"
  echo "config=${STAGE1_CONFIG}"
  echo "master_port=${MASTER_PORT_1}"
} > "${STAGE1_DIR}/launch_info.txt"

torchrun \
  --nproc_per_node=8 \
  --master_port "${MASTER_PORT_1}" \
  train.py \
  --config_path "${STAGE1_CONFIG}" \
  --logdir "${STAGE1_DIR}" \
  --wandb-save-dir "${STAGE1_DIR}" \
  --disable-wandb 2>&1 | tee "${STAGE1_DIR}/train.log"

STAGE1_CKPT="${STAGE1_DIR}/checkpoint_model_001000/model.pt"
if [[ ! -f "${STAGE1_CKPT}" ]]; then
  latest_ckpt_dir="$(ls -1d "${STAGE1_DIR}"/checkpoint_model_* 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${latest_ckpt_dir}" ]]; then
    echo "No stage1 checkpoint found under ${STAGE1_DIR}" >&2
    exit 1
  fi
  STAGE1_CKPT="${latest_ckpt_dir}/model.pt"
fi

STAGE2_CONFIG_RUNTIME="${STAGE2_DIR}/train_config_runtime.yaml"
sed "s|__STAGE1_CKPT__|${STAGE1_CKPT}|g" "${STAGE2_TEMPLATE}" > "${STAGE2_CONFIG_RUNTIME}"
cp "${STAGE2_CONFIG_RUNTIME}" "${STAGE2_DIR}/train_config_input.yaml"
{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "pwd=$(pwd)"
  echo "config=${STAGE2_CONFIG_RUNTIME}"
  echo "master_port=${MASTER_PORT_2}"
  echo "stage1_ckpt=${STAGE1_CKPT}"
} > "${STAGE2_DIR}/launch_info.txt"

echo "[Stage2] config=${STAGE2_CONFIG_RUNTIME}"
echo "[Stage2] output=${STAGE2_DIR}"
echo "[Stage2] stage1_ckpt=${STAGE1_CKPT}"

torchrun \
  --nproc_per_node=8 \
  --master_port "${MASTER_PORT_2}" \
  train.py \
  --config_path "${STAGE2_CONFIG_RUNTIME}" \
  --logdir "${STAGE2_DIR}" \
  --wandb-save-dir "${STAGE2_DIR}" \
  --disable-wandb 2>&1 | tee "${STAGE2_DIR}/train.log"

echo "Done."
echo "Stage1 output: ${STAGE1_DIR}"
echo "Stage2 output: ${STAGE2_DIR}"
