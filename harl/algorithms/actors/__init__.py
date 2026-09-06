"""
Algorithm-actor registry.

Each entry maps a CLI/config algorithm name to a string of the form
``"module:ClassName"``. The class is imported lazily by ``get_actor`` so
optional dependencies (StarCraft, MuJoCo, …) are not pulled in unless the
user actually selects an algorithm that needs them.

New algorithms register themselves here.
"""

from __future__ import annotations

from importlib import import_module
from typing import Type


# fmt: off
ALGO_REGISTRY = {
    # ----- Ours -----
    "typpo": "harl.algorithms.actors.typpo:TyPPO",
    "tysac": "harl.algorithms.actors.tysac:TySAC",
    # ----- Baselines (PPO family) -----
    "mappo":  "harl.algorithms.actors.mappo:MAPPO",
    "happo":  "harl.algorithms.actors.happo:HAPPO",
    "hatrpo": "harl.algorithms.actors.hatrpo:HATRPO",
    # ----- Baselines (off-policy) -----
    "hasac": "harl.algorithms.actors.hasac:HASAC",
    # ----- Baselines (value decomposition) -----
    "qmix":  "harl.algorithms.actors.qmix:QMIX",
}
# fmt: on


def get_actor(name: str) -> Type:
    """Resolve an algorithm name to its class. Lazy import."""
    name = name.lower()
    if name not in ALGO_REGISTRY:
        raise KeyError(
            f"Unknown algorithm '{name}'. Available: {sorted(ALGO_REGISTRY.keys())}"
        )
    module_path, cls_name = ALGO_REGISTRY[name].split(":")
    module = import_module(module_path)
    return getattr(module, cls_name)


__all__ = ["ALGO_REGISTRY", "get_actor"]
