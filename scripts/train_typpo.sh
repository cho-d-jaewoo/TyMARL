#!/usr/bin/env bash
# Train TyPPO on a chosen scenario.
#
# Usage:
#   bash scripts/train_typpo.sh <scenario>           # smacv2 default
#   bash scripts/train_typpo.sh <env> <scenario>     # explicit env
#
# Examples:
#   bash scripts/train_typpo.sh protoss_5_vs_5       # SMACv2 protoss (main)
#   bash scripts/train_typpo.sh smac MMM2            # original SMAC
#   SEED=3 bash scripts/train_typpo.sh terran_5_vs_5

set -euo pipefail

if [[ $# -eq 1 ]]; then
    ENV="smacv2"
    SCENARIO="$1"
elif [[ $# -eq 2 ]]; then
    ENV="$1"
    SCENARIO="$2"
else
    echo "Usage: $0 <scenario>  OR  $0 <env> <scenario>" >&2
    exit 1
fi

SEED="${SEED:-1}"
EXP_NAME="typpo__${ENV}__${SCENARIO}__seed${SEED}"

python -m harl.train \
    --algo typpo \
    --env  "${ENV}" \
    --scenario "${SCENARIO}" \
    --algo-config harl/configs/algos_cfgs/typpo.yaml \
    --env-config  "harl/configs/envs_cfgs/${ENV}.yaml" \
    --seed "${SEED}" \
    --exp-name "${EXP_NAME}"
