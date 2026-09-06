#!/usr/bin/env bash
# Train TySAC on a chosen scenario. Continuous-action only.
#
# Usage:
#   bash scripts/train_tysac.sh <env> <scenario>
#
# Examples:
#   bash scripts/train_tysac.sh mamujoco Walker2d-v2
#   bash scripts/train_tysac.sh mamujoco HalfCheetah-2x3

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <env> <scenario>" >&2
    exit 1
fi
ENV="$1"
SCENARIO="$2"
SEED="${SEED:-1}"
EXP_NAME="tysac__${ENV}__${SCENARIO}__seed${SEED}"

python -m harl.train \
    --algo tysac \
    --env  "${ENV}" \
    --scenario "${SCENARIO}" \
    --algo-config harl/configs/algos_cfgs/tysac.yaml \
    --env-config  "harl/configs/envs_cfgs/${ENV}.yaml" \
    --seed "${SEED}" \
    --exp-name "${EXP_NAME}"
