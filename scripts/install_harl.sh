#!/usr/bin/env bash
# Clone the HARL upstream into third_party/HARL/ so that our HARL-family
# baseline wrappers (happo, hatrpo, haa2c, hasac, haddpg, hatd3, …) can
# resolve their `from harl.algorithms.actors.X import X` imports.
#
# We do this at install-time rather than vendoring HARL into the repo so the
# diff between TyMARL and HARL stays small and reviewable, and so users can
# pin a specific HARL commit without the diff polluting our repo's history.
#
# Usage:
#   bash scripts/install_harl.sh                 # default: latest main
#   HARL_PINNED_HASH=abc123 bash scripts/install_harl.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${REPO_ROOT}/third_party/HARL"
URL="https://github.com/PKU-MARL/HARL.git"
PINNED="${HARL_PINNED_HASH:-}"

mkdir -p "${REPO_ROOT}/third_party"

if [[ -d "${DEST}/.git" ]]; then
    echo "[install_harl] HARL already present at ${DEST}; pulling latest."
    git -C "${DEST}" fetch --quiet
else
    echo "[install_harl] Cloning HARL into ${DEST}."
    git clone --quiet "${URL}" "${DEST}"
fi

if [[ -n "${PINNED}" ]]; then
    echo "[install_harl] Checking out pinned commit ${PINNED}."
    git -C "${DEST}" checkout --quiet "${PINNED}"
else
    git -C "${DEST}" checkout --quiet main || git -C "${DEST}" checkout --quiet master
fi

echo "[install_harl] HARL is at: $(git -C "${DEST}" rev-parse --short HEAD)"
echo "[install_harl] Done. harl.train will add ${DEST} to sys.path automatically."
