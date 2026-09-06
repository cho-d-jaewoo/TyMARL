"""
SMACv2 wrapper with dynamic, episode-level group assignment.

Why this wrapper exists
-----------------------
SMACv2 generates a fresh team composition every episode by sampling unit
types from a per-race distribution. This is precisely the setting where
HARL-style per-agent independent networks are awkward: the agent slot index
carries no semantic meaning across episodes, while the *unit type* of the
agent at that slot does. TyPPO and TySAC route every agent to the network
of its unit-type group, so when the composition changes between episodes,
the per-group networks see a consistent stream of within-type observations.

This wrapper implements that mapping. On every ``reset()``:

1. We read the actual unit types of the spawned ally team via SMACv2's
   ``capability_config["team_gen"]`` and the agent metadata.
2. We rebuild a ``GroupAssignment`` from the current unit types and store it
   in ``self.groups``.
3. ``info["groups_changed"]`` is set whenever the partition differs from the
   previous episode's, so the runner can broadcast the new partition to the
   algorithm via ``algo.set_groups(...)``.

Strict vs. dummy backend
------------------------
Importing the real ``smacv2`` package can fail for many reasons (StarCraft II
not installed, SC2 binary path misconfigured, capability-config format
changes between SMACv2 versions, map files missing, etc.). We previously
fell back silently to a synthetic ``_DummySMACv2Env`` that emits uniform
random rewards — this produced plausible-looking learning curves
(``ep_return_mean ~ 90``, ``win_rate ~ 0.27``) that had nothing to do with
the algorithm, only with the random reward distribution.

To prevent this, the default behaviour is now:

* If the real SMACv2 backend cannot be imported / constructed, the wrapper
  raises a ``RuntimeError`` so the failure is visible.
* The dummy backend is only used when the user explicitly opts in by
  setting ``TYMARL_ALLOW_DUMMY_SMACV2=1`` in the environment, or by
  passing ``allow_dummy=True`` at construction time. This is intended for
  CI / unit tests only.
* Whether the dummy backend is in use is exposed both through
  ``self._using_dummy`` and through ``get_env_info()["using_dummy"]`` so
  the runner can stamp it onto every log entry.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from harl.utils.group import GroupAssignment, assign_groups_by_type


# Standard SMACv2 unit-type pools (matches smacv2's capability presets).
SMACV2_UNIT_POOLS: Dict[str, List[str]] = {
    "protoss": ["stalker", "zealot", "colossus"],
    "terran":  ["marine", "marauder", "medivac"],
    "zerg":    ["zergling", "hydralisk", "baneling"],
}

# Fixed canonical order for one-hot encoding so the per-unit-type slot is
# stable across episodes regardless of which subset spawns this episode.
def _canonical_type_order(race: str) -> List[str]:
    return SMACV2_UNIT_POOLS.get(race, ["stalker", "zealot", "colossus"])


def _parse_scenario(scenario: str) -> Tuple[str, int | None]:
    """Parse 'protoss_5_vs_5' -> ('protoss', 5).

    Returns ``(race, n_units_from_name)``. ``n_units_from_name`` is ``None``
    when the scenario name has no ``_<N>_vs_<N>`` suffix (so the caller
    falls back to the constructor default).
    """
    parts = scenario.split("_")
    race = parts[0]
    n_units_from_name: int | None = None
    # Expected: 'race_N_vs_M' -- we use N (allies). M (enemies) is set
    # equal to N below for the standard symmetric setting.
    if len(parts) >= 4 and parts[2] == "vs":
        try:
            n_units_from_name = int(parts[1])
        except ValueError:
            n_units_from_name = None
    return race, n_units_from_name


class SMACv2Env:
    """SMACv2 with per-episode group assignment by unit type."""

    def __init__(
        self,
        scenario: str = "protoss_5_vs_5",
        seed: int = 0,
        n_units: int | None = None,
        difficulty: str = "7",
        episode_limit: int = 200,
        capability_config: Dict[str, Any] | None = None,
        replay_dir: str | None = None,
        replay_prefix: str | None = None,
        reward_sparse: bool | None = None,
        reward_scale: bool | None = None,
        reward_scale_rate: float | None = None,
        state_last_action: bool | None = None,
        obs_all_health: bool | None = None,
        allow_dummy: bool | None = None,
        **smac_kwargs: Any,
    ):
        # Parse scenario name. The unit count from the name takes precedence
        # over the constructor default when provided, so naming and reality
        # cannot drift apart.
        race, n_units_from_name = _parse_scenario(scenario)
        if race not in SMACV2_UNIT_POOLS:
            raise ValueError(
                f"Unknown SMACv2 race in scenario '{scenario}'. "
                f"Expected one of: {list(SMACV2_UNIT_POOLS.keys())}"
            )

        # Resolve n_units precedence: explicit kwarg > scenario name > default(5).
        if n_units is not None and n_units_from_name is not None and n_units != n_units_from_name:
            warnings.warn(
                f"scenario='{scenario}' implies n_units={n_units_from_name} but "
                f"n_units={n_units} was passed; using the explicit value "
                f"({n_units}). Rename the scenario for consistency.",
                stacklevel=2,
            )
        if n_units is None:
            n_units = n_units_from_name if n_units_from_name is not None else 5

        self.race = race
        self.scenario = scenario
        self.canonical_types = _canonical_type_order(race)
        self.n_canonical_types = len(self.canonical_types)
        self.seed = seed

        # Default capability config: SMACv2's "stochastic team gen" preset.
        if capability_config is None:
            capability_config = self._default_capability_config(race, n_units)
        self.capability_config = capability_config
        self.n_units = n_units
        self._raw_obs_dim: int | None = None

        # Replay setup -- when `replay_dir` is set, SMACv2 saves an .SC2Replay
        # file at the end of every episode that can be opened with the SC2
        # client to watch trained agents play.
        if replay_dir is not None:
            from pathlib import Path
            Path(replay_dir).mkdir(parents=True, exist_ok=True)
            smac_kwargs.setdefault("replay_dir", str(Path(replay_dir).absolute()))
            if replay_prefix is not None:
                smac_kwargs.setdefault("replay_prefix", replay_prefix)
        self._replay_dir = replay_dir
        self._replay_prefix = replay_prefix

        # Forward optional reward / state / obs flags to SMACv2 only when set.
        for key, value in {
            "reward_sparse": reward_sparse,
            "reward_scale": reward_scale,
            "reward_scale_rate": reward_scale_rate,
            "state_last_action": state_last_action,
            "obs_all_health": obs_all_health,
        }.items():
            if value is not None:
                smac_kwargs.setdefault(key, value)

        # Forward difficulty (always set; default "7" matches SMAC standard).
        # SC2's StarCraft2Env reads `difficulty` as the scripted-bot heuristic
        # level. Without this line the constructor argument was silently
        # dropped, so every run secretly used the SC2 default. Setting it
        # explicitly here guarantees the YAML setting actually takes effect.
        smac_kwargs.setdefault("difficulty", str(difficulty))

        # Determine whether the dummy fallback is permitted.
        if allow_dummy is None:
            allow_dummy = os.environ.get("TYMARL_ALLOW_DUMMY_SMACV2", "0") == "1"
        self._allow_dummy = bool(allow_dummy)

        # Hint: the SC2 init may fail if multiple SC2 instances launch in
        # quick succession on the same machine (port collision / pysc2
        # websocket handshake timing). We retry with exponential backoff
        # before giving up. ImportError (smacv2 not installed) is final --
        # no point retrying that.
        import time as _time
        self._env = None
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                # Official SMACv2 entry point for capability-config scenarios is
                # StarCraftCapabilityEnvWrapper, NOT StarCraft2Env. The wrapper
                # parses the capability_config and feeds the right pieces into
                # StarCraft2Env (e.g. translates team_gen into the SC2 unit-spawn
                # spec). Calling StarCraft2Env directly with a capability_config
                # leaves several internal lists empty, which causes
                # init_units() to fail with ``min() arg is an empty sequence``.
                from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper  # noqa: WPS433
                import inspect as _inspect

                # Pull in optional flags the example uses by default; users can
                # still override them via env_config YAML.
                _wrapper_defaults = dict(
                    debug=False,
                    conic_fov=False,
                    obs_own_pos=True,
                    use_unit_ranges=True,
                    min_attack_range=2,
                )
                for k, v in _wrapper_defaults.items():
                    smac_kwargs.setdefault(k, v)

                # Strip kwargs that aren't accepted by SC2 (defense in depth).
                for _strip in ("episode_limit", "scenario", "n_units"):
                    smac_kwargs.pop(_strip, None)

                # The wrapper forwards to StarCraft2Env, so accepted kwargs are
                # the union of the wrapper's signature and SC2's signature.
                _wrapper_sig = set(_inspect.signature(StarCraftCapabilityEnvWrapper.__init__).parameters.keys()) - {"self"}
                try:
                    from smacv2.env import StarCraft2Env as _SC2EnvForSig
                    _sc2_sig = set(_inspect.signature(_SC2EnvForSig.__init__).parameters.keys()) - {"self"}
                except Exception:
                    _sc2_sig = set()
                _allowed = _wrapper_sig | _sc2_sig
                _safe_kwargs = {k: v for k, v in smac_kwargs.items() if k in _allowed}
                _rejected = set(smac_kwargs) - set(_safe_kwargs)
                if _rejected and _attempt == 0:
                    warnings.warn(
                        f"SMACv2Env: ignoring unrecognised wrapper kwargs: "
                        f"{sorted(_rejected)}. Check env config YAML.",
                        stacklevel=2,
                    )
                print(
                    f"[SMACv2Env] init StarCraftCapabilityEnvWrapper("
                    f"map_name=10gen_{race}, seed={seed}, "
                    f"attempt={_attempt+1}/3, "
                    f"capability_config keys={sorted(capability_config.keys())}, "
                    f"safe_kwargs={sorted(_safe_kwargs)})",
                    flush=True,
                )
                self._env = StarCraftCapabilityEnvWrapper(
                    capability_config=capability_config,
                    map_name=f"10gen_{race}",
                    seed=seed,
                    **_safe_kwargs,
                )
                self._using_dummy = False
                last_exc = None
                break
            except ImportError as exc:
                last_exc = exc
                break  # smacv2 not installed -- retry won't help
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if _attempt < 2:
                    wait_s = 10 * (_attempt + 1)
                    print(
                        f"[SMACv2Env] launch attempt {_attempt+1}/3 failed "
                        f"({type(exc).__name__}: {exc}); waiting {wait_s}s "
                        f"before retry. This is normal when several SC2 "
                        f"instances launch concurrently.",
                        flush=True,
                    )
                    _time.sleep(wait_s)

        if self._env is None:
            if not self._allow_dummy:
                raise RuntimeError(
                    f"SMACv2 backend failed to initialise after 3 attempts. "
                    f"Last error: {last_exc!r}. "
                    f"Refusing to fall back to the synthetic dummy env, since "
                    f"that would produce meaningless metrics. Either fix the "
                    f"SMACv2 install, reduce concurrent SC2 instances, or "
                    f"set the env var TYMARL_ALLOW_DUMMY_SMACV2=1 "
                    f"(CI / smoke-test only)."
                ) from last_exc
            warnings.warn(
                f"SMACv2 init failed after retries ({last_exc!r}); falling "
                f"back to _DummySMACv2Env because TYMARL_ALLOW_DUMMY_SMACV2=1. "
                f"All training metrics will be SYNTHETIC and not reflect "
                f"real algorithm performance.",
                stacklevel=2,
            )
            self._env = _DummySMACv2Env(race, n_units, episode_limit, seed)
            self._using_dummy = True

        self._env_info_static = self._env.get_env_info()
        self.episode_limit = self._env_info_static["episode_limit"]
        self.n_actions = self._env_info_static["n_actions"]
        self.num_agents = self._env_info_static["n_agents"]
        self._raw_obs_dim = self._env_info_static["obs_shape"]
        self._raw_state_dim = self._env_info_static["state_shape"]

        # Initial groups (fresh assignment from the first reset).
        self.groups: GroupAssignment | None = None
        self._prev_groups_signature: Tuple[str, ...] | None = None

    # ------------------------------------------------------------------
    # SMACv2 capability_config preset
    # ------------------------------------------------------------------

    @staticmethod
    def _default_capability_config(race: str, n_units: int) -> Dict[str, Any]:
        """SMACv2 capability_config preset.

        Strategy: prefer the official YAML config that ships INSIDE the
        installed ``smacv2`` package (``smacv2/env/configs/*.yaml``). That
        guarantees the keys we hand SMACv2 are exactly the ones SMACv2 of
        the same install version expects to read -- no schema drift.

        We then override only the team size (``n_units`` / ``n_enemies``)
        in every place SMACv2 might read it (top level, team_gen,
        start_positions) so the user can change team size from CLI without
        editing the YAML.

        Falls back to a hand-rolled dict only when the YAML file cannot be
        found, which should not happen on a normal SMACv2 install.
        """
        cfg: Dict[str, Any] | None = None
        try:
            import smacv2  # noqa: WPS433
            import yaml as _yaml  # noqa: WPS433
            pkg_root = Path(smacv2.__file__).resolve().parent
            # Try a few known locations across SMACv2 versions.
            candidates = [
                pkg_root / "env" / "starcraft2" / "configs" / f"sc2_gen_{race}.yaml",
                pkg_root / "configs"            / f"sc2_gen_{race}.yaml",
                pkg_root / "env" / "configs"    / f"sc2_gen_{race}.yaml",
            ]
            for p in candidates:
                if p.exists():
                    raw = _yaml.safe_load(p.read_text(encoding="utf-8"))
                    # Some YAMLs nest the cap config under a key, some don't.
                    cfg = raw.get("env_args", raw).get("capability_config", raw)
                    print(f"[SMACv2Env] using shipped capability_config: {p}", flush=True)
                    break
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not load shipped SMACv2 capability_config "
                f"({exc!r}); falling back to hand-rolled config. This may "
                f"break on SMACv2 versions that expect different keys.",
                stacklevel=2,
            )

        if cfg is None:
            # Hand-rolled fallback. Tries to be permissive about which
            # SMACv2 version we're on by populating n_units / n_enemies in
            # every place we've seen them required across versions.
            if race == "protoss":
                unit_types, weights, exception = (
                    ["stalker", "zealot", "colossus"], [0.45, 0.45, 0.1], ["colossus"]
                )
            elif race == "terran":
                unit_types, weights, exception = (
                    ["marine", "marauder", "medivac"], [0.45, 0.45, 0.1], ["medivac"]
                )
            elif race == "zerg":
                unit_types, weights, exception = (
                    ["zergling", "hydralisk", "baneling"], [0.45, 0.45, 0.1], ["baneling"]
                )
            else:
                raise ValueError(f"Unknown race: {race}")
            cfg = {
                "n_units": n_units,
                "n_enemies": n_units,
                "team_gen": {
                    "dist_type": "weighted_teams",
                    "unit_types": unit_types,
                    "weights": weights,
                    "exception_unit_types": exception,
                    "observe": True,
                },
                "start_positions": {
                    "dist_type": "surrounded_and_reflect",
                    "p": 0.5,
                    "map_x": 32,
                    "map_y": 32,
                },
            }

        # Override team size in EVERY place SMACv2 might read it. We only
        # set the key if it already exists at that nesting level, so that
        # the shipped config's schema is preserved and we just update
        # values -- this avoids fighting SMACv2's own validation.
        if "n_units" in cfg:
            cfg["n_units"] = n_units
        if "n_enemies" in cfg:
            cfg["n_enemies"] = n_units
        if isinstance(cfg.get("team_gen"), dict):
            tg = cfg["team_gen"]
            if "n_units" in tg:
                tg["n_units"] = n_units
            if "n_enemies" in tg:
                tg["n_enemies"] = n_units
        if isinstance(cfg.get("start_positions"), dict):
            sp = cfg["start_positions"]
            if "n_units" in sp:
                sp["n_units"] = n_units
            if "n_enemies" in sp:
                sp["n_enemies"] = n_units
        # Defense-in-depth: if SMACv2 reads `n_units` / `n_enemies` from a
        # nested dict but our shipped YAML didn't list them there, we still
        # add them. SMACv2 will not complain about extra keys, only missing.
        if isinstance(cfg.get("team_gen"), dict):
            cfg["team_gen"].setdefault("n_units", n_units)
            cfg["team_gen"].setdefault("n_enemies", n_units)
        if isinstance(cfg.get("start_positions"), dict):
            cfg["start_positions"].setdefault("n_units", n_units)
            cfg["start_positions"].setdefault("n_enemies", n_units)
        cfg.setdefault("n_units", n_units)
        cfg.setdefault("n_enemies", n_units)

        return cfg

    # ------------------------------------------------------------------
    # Group assignment from current unit composition
    # ------------------------------------------------------------------

    def _read_current_unit_types(self) -> List[str]:
        """Inspect SMACv2 internals to recover ally agents' unit-type strings.

        SMACv2's ``StarCraft2Env`` stores ally agents in ``self.agents`` and
        their unit type as an attribute on each agent.  CRITICAL: that
        attribute holds the *raw StarCraft 2 unit type ID* (e.g. 74 for
        stalker, 73 for zealot, 4 for colossus), not an index into our
        canonical type list.  To recover the canonical type string we have
        to compare against the env's race-specific ID constants
        (``stalker_id``, ``zealot_id``, ``colossus_id``,
        ``marine_id``, ``marauder_id``, ``medivac_id``,
        ``zergling_id``, ``hydralisk_id``, ``baneling_id``).

        An earlier version of this method assumed the attribute was already
        a 0..K-1 canonical index and silently clamped large values to the
        last canonical type, which routed *every* unit to the last group
        regardless of its actual type. That broke the type-routing
        invariant of TyMARL even though training appeared to proceed.
        """
        try:
            # Reach the underlying StarCraft2Env through the
            # StarCraftCapabilityEnvWrapper. Both the wrapper and the
            # raw env expose ``agents`` and the per-race ID constants.
            inner = self._env
            for attr in ("env", "_env"):
                if hasattr(inner, attr) and hasattr(getattr(inner, attr), "agents"):
                    inner = getattr(inner, attr)
                    break

            agents = inner.agents
            if isinstance(agents, dict):
                agent_list = [agents[i] for i in sorted(agents.keys())]
            else:
                agent_list = list(agents)

            # Build a mapping: raw SC2 unit-type-id -> canonical type string,
            # restricted to this race's three units.
            id_to_canon: Dict[int, str] = {}
            for canon in self.canonical_types:
                attr_name = f"{canon}_id"
                if hasattr(inner, attr_name):
                    raw_id = getattr(inner, attr_name)
                    try:
                        id_to_canon[int(raw_id)] = canon
                    except (TypeError, ValueError):
                        pass

            # If the env didn't expose any of the expected *_id constants
            # we cannot recover types reliably; fall through to fallback.
            if not id_to_canon:
                raise RuntimeError(
                    f"SMACv2 env exposes none of the expected race-id "
                    f"constants for race={self.race} "
                    f"(canonical_types={self.canonical_types}). "
                    f"Did the SMACv2 internal API change?"
                )

            type_strs: List[str] = []
            for agent_id in range(self.num_agents):
                agent = agent_list[agent_id]
                # Try the standard attribute names that SMAC/SMACv2 use.
                raw: Any = None
                for attr in ("unit_type", "unit_type_id"):
                    if hasattr(agent, attr):
                        v = getattr(agent, attr)
                        if v is not None:
                            raw = v
                            break
                if raw is None:
                    raise RuntimeError(
                        f"agent {agent_id} has no unit_type / unit_type_id"
                    )
                raw_id = int(raw)
                if raw_id in id_to_canon:
                    type_strs.append(id_to_canon[raw_id])
                else:
                    # Unknown id -- this can happen if SMACv2 spawned an
                    # off-roster unit (shouldn't, but defensive). Fall
                    # back to the env's own get_unit_types_fallback if
                    # available, else assume the first canonical type.
                    if hasattr(inner, "get_unit_types_fallback"):
                        fb = inner.get_unit_types_fallback()
                        if agent_id < len(fb) and fb[agent_id] in self.canonical_types:
                            type_strs.append(fb[agent_id])
                            continue
                    warnings.warn(
                        f"agent {agent_id} has unknown unit_type id "
                        f"{raw_id}; expected one of {sorted(id_to_canon)}. "
                        f"Routing to canonical_types[0]={self.canonical_types[0]}.",
                        stacklevel=2,
                    )
                    type_strs.append(self.canonical_types[0])
            return type_strs
        except Exception as exc:
            # Fallback: dummy env path or unexpected SMACv2 internals.
            if hasattr(self._env, "get_unit_types_fallback"):
                fb = self._env.get_unit_types_fallback()
                # Validate that fallback returns canonical strings.
                if all(t in self.canonical_types for t in fb):
                    return fb
            warnings.warn(
                f"_read_current_unit_types failed ({type(exc).__name__}: "
                f"{exc}); returning all-{self.canonical_types[0]} as a "
                f"last-resort fallback. This will silently degrade "
                f"TyMARL to single-type training -- investigate!",
                stacklevel=2,
            )
            return [self.canonical_types[0]] * self.num_agents

    def _rebuild_groups(self) -> bool:
        """Refresh ``self.groups`` from the just-spawned team. Return True iff
        the partition changed compared to the previous episode."""
        types = self._read_current_unit_types()
        # CRITICAL: pass the race's canonical type ordering so that group IDs
        # are stable across episodes. Without this, group 0 could mean
        # "stalker" in one episode and "colossus" in the next, and a single
        # actor would receive gradients from inconsistent unit types --
        # which silently breaks the TyMARL invariant that one actor
        # represents one type.
        ga = assign_groups_by_type(types, canonical_order=self.canonical_types)
        signature = tuple(types)
        changed = (signature != self._prev_groups_signature)
        self._prev_groups_signature = signature
        self.groups = ga
        return changed

    # ------------------------------------------------------------------
    # Per-agent canonical-type ids (required by the runner / buffer)
    # ------------------------------------------------------------------

    def get_canonical_type_ids(self) -> np.ndarray:
        """Return ``np.ndarray[int]`` of shape ``(num_agents,)`` whose entry
        ``i`` is the canonical-type index of agent ``i`` *right now*.

        The runner stores this per-step so that, at training time, we can
        look at every (timestep, thread, agent) triple and route it to the
        actor whose canonical label matches the unit type at that triple --
        independent of any reset in between.
        """
        types = self._read_current_unit_types()
        return np.array(
            [self.canonical_types.index(t) for t in types],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # Per-agent alive mask
    # ------------------------------------------------------------------

    def get_alive_mask(self) -> np.ndarray:
        """Return ``(num_agents,)`` float mask: 1.0 if alive, 0.0 if dead.

        SMAC treats dead agents as still-stepping but with no available
        actions and zeroed observations. We surface the alive flag so the
        algorithm can mask their PPO contribution out of the loss.
        """
        try:
            agents = self._env.agents
            if isinstance(agents, dict):
                agent_list = [agents[i] for i in sorted(agents.keys())]
            else:
                agent_list = list(agents)
            alive = np.zeros(self.num_agents, dtype=np.float32)
            for i in range(self.num_agents):
                health_raw = getattr(agent_list[i], "health", 1.0)
                if health_raw is None:
                    alive[i] = 0.0
                else:
                    alive[i] = 1.0 if float(health_raw) > 0.0 else 0.0
            return alive
        except Exception:
            # Dummy backend or unexpected SMACv2 internals: assume alive.
            return np.ones(self.num_agents, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym-like API
    # ------------------------------------------------------------------

    def reset(self) -> Tuple[np.ndarray, np.ndarray, GroupAssignment]:
        # SMACv2 reset() can return: nothing (older), (obs, state) (newer),
        # or (obs, state, info). Discard whatever it returns and call
        # get_obs/get_state explicitly for stability.
        self._env.reset()
        groups_changed = self._rebuild_groups()
        obs = np.asarray(self._env.get_obs(), dtype=np.float32)
        state = np.asarray(self._env.get_state(), dtype=np.float32)
        # Augment obs with a stable, race-canonical unit-type one-hot so
        # parameter-shared baselines (MAPPO) can still distinguish types.
        obs = self._augment_obs(obs)
        # Stash group-changed flag on instance for the runner to read.
        self._last_groups_changed = groups_changed
        return obs, state, self.groups

    def step(self, joint_action) -> Tuple[np.ndarray, np.ndarray, float, bool, Dict]:
        # SMACv2 step() returns (reward, done, info) on most versions but
        # the info dict shape may vary. Be defensive about the return value.
        ret = self._env.step(joint_action)
        if len(ret) == 3:
            reward, done, info = ret
        elif len(ret) == 5:                                        # gym-style
            _, reward, done, _, info = ret
        else:
            raise RuntimeError(
                f"SMACv2.step() returned an unexpected tuple of length "
                f"{len(ret)}; expected 3 (reward, done, info) or 5 (gym-style)."
            )
        info = dict(info) if info is not None else {}
        info["groups_changed"] = bool(getattr(self, "_last_groups_changed", False))
        self._last_groups_changed = False

        obs = np.asarray(self._env.get_obs(), dtype=np.float32)
        state = np.asarray(self._env.get_state(), dtype=np.float32)
        return self._augment_obs(obs), state, float(reward), bool(done), info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Append a unit-type one-hot (canonical race order) per agent.

        This is *unit-type* one-hot, not group one-hot. The two are equivalent
        when groups are assigned by type, but using the canonical order keeps
        the augmentation stable across episodes regardless of which subset of
        types actually spawned.
        """
        eye = np.eye(self.n_canonical_types, dtype=np.float32)
        type_idx = np.array(
            [self.canonical_types.index(t) for t in self._read_current_unit_types()],
            dtype=np.int64,
        )
        type_oh = eye[type_idx]                                   # (N, n_types)
        return np.concatenate([obs, type_oh], axis=-1)

    def get_avail_actions(self) -> np.ndarray:
        return np.asarray(self._env.get_avail_actions(), dtype=np.float32)

    def get_env_info(self) -> Dict[str, Any]:
        info = dict(self._env_info_static)
        info["obs_shape"] = self._raw_obs_dim + self.n_canonical_types
        info["n_canonical_types"] = self.n_canonical_types
        info["canonical_types"] = list(self.canonical_types)
        info["dynamic_groups"] = True
        info["using_dummy"] = bool(self._using_dummy)
        return info

    def close(self) -> None:
        self._env.close()


# -----------------------------------------------------------------------------
# Dummy backend (CI; no SC2 install required)
# -----------------------------------------------------------------------------

class _DummyAgent:
    def __init__(self, unit_type_id: int):
        self.unit_type_id = unit_type_id
        self.health = 1.0


class _DummySMACv2Env:
    """Tiny synthetic SMACv2 surrogate for testing without StarCraft.

    .. WARNING:: This backend produces uniform-random rewards. The mean
    return and win rate it yields are properties of the random seed, NOT
    of the trained policy. It exists purely to allow shape / smoke tests
    to run in environments without SC2; the production SMACv2 wrapper now
    raises rather than falling back to it unless the user opts in via
    ``TYMARL_ALLOW_DUMMY_SMACV2=1``.
    """

    def __init__(self, race: str, n_units: int, episode_limit: int, seed: int):
        self._rng = np.random.default_rng(seed)
        self.race = race
        self.n_units = n_units
        self.n_agents = n_units
        self.n_actions = 6
        self._max_t = episode_limit
        self._t = 0
        # Dummy agents resampled each reset, mirroring SMACv2's stochastic teams.
        self.agents: List[_DummyAgent] = []
        self._resample_agents()

    def _resample_agents(self) -> None:
        self.agents = [
            _DummyAgent(unit_type_id=int(self._rng.integers(0, 3)))
            for _ in range(self.n_units)
        ]

    def reset(self):
        self._t = 0
        self._resample_agents()

    def get_env_info(self) -> Dict[str, int]:
        return {
            "obs_shape": 24,
            "state_shape": 32,
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "episode_limit": self._max_t,
        }

    def get_obs(self):
        return self._rng.standard_normal((self.n_agents, 24)).astype(np.float32)

    def get_state(self):
        return self._rng.standard_normal((32,)).astype(np.float32)

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions), dtype=np.float32)

    def step(self, joint_action):
        self._t += 1
        reward = float(self._rng.uniform(-0.1, 1.0))
        done = self._t >= self._max_t
        return reward, done, {"battle_won": done and reward > 0.7}

    def get_unit_types_fallback(self) -> List[str]:
        canon = SMACV2_UNIT_POOLS[self.race]
        return [canon[a.unit_type_id] for a in self.agents]

    def close(self):
        pass
