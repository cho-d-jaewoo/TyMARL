"""
HADDPG — Heterogeneous-Agent reinforcement learning algorithm
(Zhong et al., JMLR 2024).

This file is a thin re-export from the HARL upstream (PKU-MARL/HARL). Run
`bash scripts/install_harl.sh` to clone the upstream into `third_party/HARL/`,
which is added to `sys.path` by `harl.train` so the import below resolves.

Compatible benchmarks (see docs/algorithm_compatibility.md)
-----------------------------------------------------------
SMAC ✅, SMACv2 ✅, mixed_smacv2 ✅, MAMuJoCo ✅, MPE ✅, Bi-DexHands ✅
"""

from __future__ import annotations

try:
    from harl.algorithms.actors.haddpg import HADDPG  # type: ignore[import-not-found]
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "HADDPG is provided by the upstream HARL repository. "
        "Run scripts/install_harl.sh to clone and patch it."
    ) from e

__all__ = ["HADDPG"]
