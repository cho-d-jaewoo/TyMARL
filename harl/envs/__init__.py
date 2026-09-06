"""
Environment registry.

Each entry maps an env-name string (used at the CLI and in env configs) to a
factory thunk that returns a constructed env-wrapper class. Factories are
thunks so optional dependencies (StarCraft, MuJoCo, …) are imported only when
used. Per-env ``__init__.py`` files document algorithm compatibility and
recommended metrics.
"""

from __future__ import annotations

from typing import Callable, Dict


def _smac():
    try:
        from smac.env import StarCraft2Env  # type: ignore
        return StarCraft2Env
    except ImportError as e:  # pragma: no cover
        raise ImportError("SMAC: `pip install smac` (and install StarCraft II).") from e


def _smacv2():
    from harl.envs.smacv2 import SMACv2Env
    return SMACv2Env


def _mamujoco():
    """Multi-Agent MuJoCo wrapper. HARL upstream ships ``MujocoMulti``;
    this adapter conforms it to the TyMARL build_env interface."""
    try:
        from harl.envs.mamujoco.multiagent_mujoco.mujoco_multi import MujocoMulti  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Multi-Agent MuJoCo: bring it from HARL upstream and ensure mujoco / "
            "mujoco-py / gym are installed."
        ) from e

    class MAMuJoCoEnv(MujocoMulti):
        def __init__(self, scenario, seed=1, agent_conf="2x3", agent_obsk=1,
                     episode_length=1000, **kwargs):
            env_args = {
                "scenario": scenario,
                "agent_conf": agent_conf,
                "agent_obsk": agent_obsk,
                "episode_limit": episode_length,
                "seed": seed,
            }
            env_args.update(kwargs)
            super().__init__(env_args=env_args)

        def get_env_info(self):
            info = super().get_env_info() if hasattr(super(), "get_env_info") else {}
            if "n_agents" not in info:
                info["n_agents"] = getattr(self, "n_agents", 2)
            return info

    return MAMuJoCoEnv


def _mpe():
    """PettingZoo MPE adapter. Forwards TyMARL kwargs to the (rewritten)
    PettingZooMPEEnv, handling per-scenario kwarg name differences and
    routing seed through the wrapper's seed() method (PettingZoo takes
    seed at reset(), not __init__()).

    NOTE: get_env_info is intentionally NOT overridden here — the wrapper
    body's get_env_info already returns the keys TyMARL build_algo reads.
    """
    try:
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MPE: ensure pettingzoo (>=1.23) and supersuit are installed and "
            "harl/envs/pettingzoo_mpe/ is populated."
        ) from e

    class MPEEnv(PettingZooMPEEnv):
        # PettingZoo MPE: per-scenario kwarg names differ.
        _SCEN_KWARGS = {
            "simple_spread_v3":           ("N", "max_cycles", "continuous_actions"),
            "simple_speaker_listener_v4": ("max_cycles", "continuous_actions"),
            "simple_reference_v3":        ("max_cycles", "continuous_actions"),
            "simple_v3":                  ("max_cycles", "continuous_actions"),
        }

        def __init__(self, scenario, seed=1, num_agents=3, episode_length=25,
                     continuous_actions=True, **kwargs):
            allowed = self._SCEN_KWARGS.get(
                scenario, ("max_cycles", "continuous_actions")
            )
            scen_kwargs = {}
            if "N" in allowed:
                scen_kwargs["N"] = num_agents
            if "max_cycles" in allowed:
                scen_kwargs["max_cycles"] = episode_length
            if "continuous_actions" in allowed:
                scen_kwargs["continuous_actions"] = continuous_actions

            env_args = {"scenario": scenario, **scen_kwargs}
            env_args.update(kwargs)
            super().__init__(env_args)
            self.seed(seed)

    return MPEEnv


def _bidexhands():
    try:
        from harl.envs.bidexhands.bidex_env import BiDexEnv  # type: ignore
        return BiDexEnv
    except ImportError as e:  # pragma: no cover
        raise ImportError("Bi-DexterousHands needs Isaac Gym.") from e


ENV_REGISTRY: Dict[str, Callable[[], type]] = {
    "smac":       _smac,
    "smacv2":     _smacv2,
    "mamujoco":   _mamujoco,
    "mpe":        _mpe,
    "bidexhands": _bidexhands,
}


def get_env(name: str) -> type:
    name = name.lower()
    if name not in ENV_REGISTRY:
        raise KeyError(f"Unknown env '{name}'. Available: {sorted(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[name]()


__all__ = ["ENV_REGISTRY", "get_env"]
