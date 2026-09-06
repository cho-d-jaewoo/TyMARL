#!/usr/bin/env bash
# Reproduce the main results in the paper.
#
# Layout:
#   1. SMACv2 stochastic-team scenarios (protoss/terran/zerg) -- primary
#      benchmark since the per-episode unit-type sampling is exactly what
#      our group-level sequential update was designed for.
#   2. SMAC MMM2 -- historic mixed-unit benchmark (static composition).
#   3. MAMuJoCo -- continuous-action benchmark (TySAC vs HASAC on body-part groups).

set -euo pipefail
SEEDS=(1 2 3 4 5)

run_ours() {
    local algo="$1" env="$2" scenario="$3"
    for s in "${SEEDS[@]}"; do
        SEED="${s}" bash "scripts/train_${algo}.sh" "${env}" "${scenario}" \
            || echo "[reproduce] ${algo} on ${env}/${scenario} seed=${s} failed; continuing."
    done
}

# 1. SMACv2 — main results table (3 races)
for SCEN in protoss_5_vs_5 terran_5_vs_5 zerg_5_vs_5; do
    run_ours typpo smacv2 "${SCEN}"
    SEED="${SEEDS[0]}" bash scripts/train_baselines.sh smacv2 "${SCEN}"
done

# 2. SMAC MMM2 — established mixed-unit benchmark
run_ours typpo smac MMM2
SEED="${SEEDS[0]}" bash scripts/train_baselines.sh smac MMM2

# 3. MAMuJoCo -- continuous-action setting (TySAC vs HASAC, body-part groups)
run_ours tysac mamujoco Walker2d-v2
SEED="${SEEDS[0]}" bash scripts/train_baselines.sh mamujoco Walker2d-v2

echo "[reproduce] done. next: scripts/extract_metrics.py, then scripts/plot_from_csv.py --all"
