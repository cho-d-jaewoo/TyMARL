#!/usr/bin/env bash
# Run all relevant baselines on a (env, scenario) pair.
#
# Usage:
#   bash scripts/train_baselines.sh <env> <scenario>

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <env> <scenario>" >&2
    exit 1
fi
ENV="$1"
SCENARIO="$2"
SEED="${SEED:-1}"

case "${ENV}" in
    smac|smacv2)
        ALGOS=("mappo" "happo" "hatrpo" "haa2c" "qmix")
        ;;
    mamujoco|mpe|bidexhands)
        ALGOS=("mappo" "happo" "hatrpo" "haa2c" "hasac" "maddpg" "matd3")
        ;;
    *)
        echo "Unknown env '${ENV}'." >&2
        exit 1
        ;;
esac

for ALGO in "${ALGOS[@]}"; do
    EXP_NAME="${ALGO}__${ENV}__${SCENARIO}__seed${SEED}"
    echo "=== ${ALGO} on ${ENV}/${SCENARIO} (seed=${SEED}) ==="
    python -m harl.train \
        --algo "${ALGO}" \
        --env  "${ENV}" \
        --scenario "${SCENARIO}" \
        --algo-config "harl/configs/algos_cfgs/${ALGO}.yaml" \
        --env-config  "harl/configs/envs_cfgs/${ENV}.yaml" \
        --seed "${SEED}" \
        --exp-name "${EXP_NAME}" \
        || echo "[train_baselines] ${ALGO} failed; continuing."
done
