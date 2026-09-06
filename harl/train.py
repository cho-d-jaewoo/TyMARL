"""
Unified training entry point.

Invoked by ``python -m harl.train`` (and indirectly by every script in
``scripts/``). Loads algorithm and environment configs, builds the appropriate
runner, and calls ``runner.run()``.

Example
-------
    python -m harl.train ^
        --algo typpo ^
        --env  smacv2 ^
        --scenario protoss_5_vs_5 ^
        --algo-config harl/configs/algos_cfgs/typpo.yaml ^
        --env-config  harl/configs/envs_cfgs/smacv2.yaml ^
        --seed 1 ^
        --exp-name typpo_protoss_seed1
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Per-process TEMP isolation -- MUST run before any pysc2 / smacv2 import.
#
# SMACv2 procedurally generates a new map for every episode and writes it to
# %TEMP%\StarCraft II\TempLaunchMap.SC2Map (a fixed path). When several SC2
# instances run concurrently (e.g. Slot A protoss/terran/zerg + Slot B), they
# all hit the same file and corrupt each other, surfacing as
#   pysc2 RequestError: ResponseCreateGame.Error.InvalidMapData:
#   'temporary map ...\TempLaunchMap.SC2Map has invalid data.'
# The fix is to give every Python process its own TEMP directory, which
# isolates the TempLaunchMap path. Done at import time so that the
# environment is set before pysc2 reads it.
# ---------------------------------------------------------------------------
import os as _os
import tempfile as _tempfile
import uuid as _uuid

def _isolate_temp_dir() -> None:
    base = _tempfile.gettempdir()
    pid = _os.getpid()
    suffix = _uuid.uuid4().hex[:8]
    isolated = _os.path.join(base, f"TyMARL_p{pid}_{suffix}")
    _os.makedirs(isolated, exist_ok=True)
    _os.environ["TEMP"] = isolated
    _os.environ["TMP"] = isolated
    print(f"[TyMARL] isolated TEMP={isolated}", flush=True)

_isolate_temp_dir()

import argparse
import importlib
import random
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import yaml


# Make HARL upstream importable when present at third_party/HARL/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_HARL_PATH = _REPO_ROOT / "third_party" / "HARL"
if _HARL_PATH.exists() and str(_HARL_PATH) not in sys.path:
    sys.path.insert(0, str(_HARL_PATH))


# Map algo name -> (module:class for algorithm, module:class for runner).
#
# All six algorithms route through OUR runners for fair comparison; only the
# training loop inside each algo class differs. MAPPO, HAPPO, HASAC, QMIX
# are independent standalone implementations -- not reductions of TyPPO/TySAC
# under specific group partitions. MAPPO and HAPPO differ from each other
# (and from TyPPO) not only in heterogeneity but also in update order and
# methodology, so it would be misleading to present them as boundary cases
# of a single framework. TyPPO/TySAC are positioned as practical algorithms
# that combine MAPPO's within-type parameter sharing with HAPPO's
# across-type sequential update.
ALGO_TABLE = {
    # ----- Ours -----
    "typpo": ("harl.algorithms.actors.typpo:TyPPO",
              "harl.runners.on_policy_group_runner:OnPolicyGroupRunner"),
    "tysac": ("harl.algorithms.actors.tysac:TySAC",
              "harl.runners.off_policy_group_runner:OffPolicyGroupRunner"),
    # ----- Baselines -----
    "mappo": ("harl.algorithms.actors.mappo:MAPPO",
              "harl.runners.on_policy_group_runner:OnPolicyGroupRunner"),
    "happo": ("harl.algorithms.actors.happo:HAPPO",
              "harl.runners.on_policy_group_runner:OnPolicyGroupRunner"),
    "hatrpo": ("harl.algorithms.actors.hatrpo:HATRPO",
               "harl.runners.on_policy_group_runner:OnPolicyGroupRunner"),
    "hasac": ("harl.algorithms.actors.hasac:HASAC",
              "harl.runners.off_policy_group_runner:OffPolicyGroupRunner"),
    "qmix":  ("harl.algorithms.actors.qmix:QMIX",
              "harl.runners.off_policy_group_runner:OffPolicyGroupRunner"),
}


# Algorithm × env hard-incompatibilities. SMACv2 / SMAC are discrete-action
# environments, while TySAC and HASAC operate in a continuous-action regime;
# pairing them produces a silent type mismatch. We refuse such combinations
# at the entry point rather than letting the runner crash mid-training.
INCOMPATIBLE_PAIRS = {
    ("tysac", "smacv2"),
    ("tysac", "smac"),
    ("hasac", "smacv2"),
    ("hasac", "smac"),
}


# Algorithms that are present in ALGO_TABLE because their actor file exists,
# but cannot actually be trained end-to-end in this repository because the
# matching runner is missing. We surface a clear error at entry rather than
# letting the runner crash with a cryptic KeyError mid-init.
UNSUPPORTED_ALGOS = {
    "qmix": (
        "QMIX requires a value-decomposition runner (epsilon-greedy "
        "exploration, mixer network, target update schedule) that is not "
        "currently implemented in this repository. The qmix.py actor file "
        "exists but the matching runner does not. To use QMIX as a baseline, "
        "install HARL upstream (https://github.com/PKU-MARL/HARL) and run "
        "via that codebase, or contribute a value-decomposition runner "
        "(harl/runners/value_decomp_group_runner.py) to this repository."
    ),
}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", required=True, choices=sorted(ALGO_TABLE.keys()))
    p.add_argument("--env", required=True)
    p.add_argument("--scenario", required=True)
    p.add_argument("--algo-config", required=True, type=Path)
    p.add_argument("--env-config", required=True, type=Path)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--exp-name", default="run")
    p.add_argument("--save-dir", default=None,
                   help="default: results/runs/<exp-name>")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-replay", action="store_true",
                   help="Save SC2 .SC2Replay files to results/replays/<exp-name>/. "
                        "SMACv2 only. Open with the StarCraft II client to watch.")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _import(spec: str):
    module_path, cls_name = spec.split(":")
    return getattr(importlib.import_module(module_path), cls_name)


# -----------------------------------------------------------------------------
# Env factory
# -----------------------------------------------------------------------------

def build_env(
    env_name: str,
    scenario: str,
    seed: int,
    env_cfg: Dict[str, Any] | None = None,
    replay_dir: str | None = None,
    replay_prefix: str | None = None,
) -> Tuple[Any, Any]:
    """Construct an env wrapper. Returns ``(env, group_assignment_or_lol)``.

    ``env_cfg`` is the parsed environment YAML; the relevant sub-dict
    ``env_cfg["env"]`` is forwarded to the wrapper as kwargs so that
    settings like ``n_units``, ``difficulty``, ``episode_limit``,
    ``reward_sparse``, etc., actually take effect (previously they were
    silently ignored).
    """
    env_kwargs: Dict[str, Any] = (env_cfg or {}).get("env", {}) if env_cfg else {}
    # Strip out keys that are CLI-controlled, not constructor kwargs.
    for k in ("name", "scenario"):
        env_kwargs.pop(k, None)

    if env_name == "smacv2":
        from harl.envs.smacv2 import SMACv2Env
        env = SMACv2Env(
            scenario=scenario, seed=seed,
            replay_dir=replay_dir, replay_prefix=replay_prefix,
            **env_kwargs,
        )
        env.reset()
        return env, env.groups                       # GroupAssignment

    if env_name == "smac":
        from harl.envs import get_env
        EnvCls = get_env("smac")
        env = EnvCls(map_name=scenario, seed=seed, **env_kwargs)
        info = env.get_env_info()
        groups = [[i] for i in range(info["n_agents"])]
        return env, groups

    if env_name == "mamujoco":
        from harl.envs import get_env
        EnvCls = get_env("mamujoco")
        env = EnvCls(scenario=scenario, seed=seed, **env_kwargs)
        info = env.get_env_info() if hasattr(env, "get_env_info") else {}
        n_agents = info.get("n_agents", 2)
        groups = [[i] for i in range(n_agents)]
        return env, groups

    if env_name == "mpe":
        from harl.envs import get_env
        EnvCls = get_env("mpe")
        env = EnvCls(scenario=scenario, seed=seed, **env_kwargs)
        info = env.get_env_info() if hasattr(env, "get_env_info") else {}
        n_agents = info.get("n_agents", 2)
        groups = [[i] for i in range(n_agents)]
        return env, groups

    raise ValueError(
        f"Unknown env '{env_name}'. Implemented here: smac, smacv2, mamujoco, mpe. "
        f"For bidexhands, add a build_env branch importing the "
        f"corresponding HARL wrapper from third_party/HARL/."
    )


# -----------------------------------------------------------------------------
# Group/agent inspection helper (handles GroupAssignment | list-of-lists)
# -----------------------------------------------------------------------------

def _n_agents_of(groups) -> int:
    if hasattr(groups, "n_agents"):
        return int(groups.n_agents)
    return sum(len(g) for g in groups)


# -----------------------------------------------------------------------------
# Algorithm factory
# -----------------------------------------------------------------------------

def build_algo(
    algo_name: str,
    AlgoCls: type,
    args: Dict[str, Any],
    env,
    groups,
    device: torch.device | str,
):
    """Hand the right kwargs to each algorithm class."""
    info = env.get_env_info() if hasattr(env, "get_env_info") else {}
    obs_dim = info.get("obs_shape", 64)
    n_agents = _n_agents_of(groups)
    state_dim = info.get("state_shape", obs_dim * n_agents)
    n_actions = info.get("n_actions", 5)
    # Continuous-action envs (e.g. MPE with continuous_actions=True) report
    # discrete=False and use action_dim. Discrete envs (SMAC family) report
    # discrete=True and use n_actions. PPO-family algos in this repo support
    # both via their `discrete` flag.
    env_discrete = bool(info.get("discrete", True))
    env_action_dim = info.get("action_dim", n_actions)

    # If the env reports dynamic groups (SMACv2 wrapper), propagate the
    # canonical-label list to TyPPO/TySAC so they build one actor per
    # canonical unit type, not per *current* episode partition.
    dynamic_kwargs: Dict[str, Any] = {}
    if info.get("dynamic_groups", False) and "canonical_types" in info:
        dynamic_kwargs["n_canonical_groups"] = info["n_canonical_types"]
        dynamic_kwargs["canonical_group_labels"] = info["canonical_types"]

    if algo_name == "typpo":
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=env_action_dim,
            groups=groups, discrete=env_discrete, device=device, **dynamic_kwargs,
        )
    if algo_name == "tysac":
        action_dim = info.get("action_dim", 1)
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
            groups=groups, device=device, **dynamic_kwargs,
        )
    if algo_name == "mappo":
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=env_action_dim,
            n_agents=n_agents, discrete=env_discrete, device=device,
        )
    if algo_name == "happo":
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=env_action_dim,
            n_agents=n_agents, discrete=env_discrete, device=device,
        )
    if algo_name == "hatrpo":
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim,
            action_dim=env_action_dim,
            n_agents=n_agents, discrete=env_discrete, device=device,
        )
    if algo_name == "hasac":
        action_dim = info.get("action_dim", 1)
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim, action_dim=action_dim,
            n_agents=n_agents, device=device,
        )
    if algo_name == "qmix":
        return AlgoCls(
            args=args, obs_dim=obs_dim, state_dim=state_dim, action_dim=n_actions,
            n_agents=n_agents, device=device,
        )
    raise ValueError(
        f"Unknown algo '{algo_name}'. Supported: typpo, tysac, mappo, happo, hasac, qmix."
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    args_ns = parse_args()

    # Hard incompatibility check — fail fast rather than silently produce
    # garbage by feeding continuous-action algos a discrete-action env.
    if (args_ns.algo, args_ns.env) in INCOMPATIBLE_PAIRS:
        raise SystemExit(
            f"[TyMARL] algo='{args_ns.algo}' is continuous-action only and "
            f"cannot be paired with the discrete-action env '{args_ns.env}'. "
            f"Use mamujoco for {args_ns.algo}, or pick typpo/mappo/happo/hatrpo "
            f"for {args_ns.env}."
        )

    # Refuse algorithms whose runner support is missing from this repo.
    if args_ns.algo in UNSUPPORTED_ALGOS:
        raise SystemExit(
            f"[TyMARL] algo='{args_ns.algo}' is not currently supported.\n"
            f"  {UNSUPPORTED_ALGOS[args_ns.algo]}"
        )

    set_seed(args_ns.seed)

    algo_cfg = load_yaml(args_ns.algo_config)
    env_cfg = load_yaml(args_ns.env_config)
    algo_cfg.setdefault("env", env_cfg.get("env", {}))
    algo_cfg["seed"] = args_ns.seed

    save_dir = args_ns.save_dir or f"results/runs/{args_ns.exp_name}"
    algo_cfg.setdefault("log", {})
    algo_cfg["log"].setdefault("save_dir", save_dir)

    print(f"[TyMARL] algo={args_ns.algo}  env={args_ns.env}  "
          f"scenario={args_ns.scenario}  seed={args_ns.seed}  device={args_ns.device}")

    replay_dir = None
    replay_prefix = None
    if args_ns.save_replay:
        if args_ns.env == "smacv2":
            replay_dir = f"results/replays/{args_ns.exp_name}"
            replay_prefix = f"{args_ns.algo}_{args_ns.scenario}_seed{args_ns.seed}"
            print(f"[TyMARL] Replays will be saved to {replay_dir}/")
        else:
            print("[TyMARL] --save-replay is supported only for smacv2; ignored.")

    envs, groups = build_env(
        args_ns.env, args_ns.scenario, args_ns.seed,
        env_cfg=env_cfg,
        replay_dir=replay_dir, replay_prefix=replay_prefix,
    )
    if hasattr(groups, "n_agents"):
        print(f"[TyMARL] {groups.n_agents} agents in {groups.n_groups} groups: {groups}")
    else:
        print(f"[TyMARL] groups: {groups}")

    AlgoCls = _import(ALGO_TABLE[args_ns.algo][0])
    RunnerCls = _import(ALGO_TABLE[args_ns.algo][1])

    algo = build_algo(args_ns.algo, AlgoCls, algo_cfg, envs, groups, args_ns.device)
    runner = RunnerCls(
        args=algo_cfg, envs=envs, eval_envs=envs, groups=groups, device=args_ns.device
    )
    runner.algo = algo
    runner.run()

    # Mark this run as completed. The bat-file launchers consult this file
    # when run with RESUME=1 to decide whether to skip a previously-finished
    # experiment. We deliberately write it AFTER ``runner.run`` returns so
    # crashed / Ctrl+C-interrupted runs do not produce a sentinel.
    try:
        done_path = Path(save_dir) / "done.flag"
        done_path.parent.mkdir(parents=True, exist_ok=True)
        done_path.write_text(
            f"algo={args_ns.algo}\nenv={args_ns.env}\nscenario={args_ns.scenario}\n"
            f"seed={args_ns.seed}\nexp_name={args_ns.exp_name}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[TyMARL] WARNING: could not write done.flag: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
