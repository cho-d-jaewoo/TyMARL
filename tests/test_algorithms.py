"""Tests for the TyPPO/TySAC training step.

Synthetic data only — no StarCraft, no MuJoCo. The goal is to catch shape
bugs and registration regressions, not to verify learning.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from harl.algorithms.actors import ALGO_REGISTRY, get_actor
from harl.algorithms.actors.typpo import TyPPO
from harl.algorithms.actors.tysac import TySAC
from harl.algorithms.actors.mappo import MAPPO
from harl.utils.buffers import OnPolicyBuffer, ReplayBuffer
from harl.utils.group import boundary_case_happo, boundary_case_mappo


# ---------------------------------------------------------------- fixtures

GROUPS = [[0, 1, 2], [3, 4], [5]]                      # canonical 3+2+1
N_AGENTS = sum(len(g) for g in GROUPS)
OBS_DIM = 8
STATE_DIM = 32
ACTION_DIM = 1
N_ACTIONS = 5
T = 16
B = 2


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fill_on_policy_buffer(buf: OnPolicyBuffer, n_actions: int) -> None:
    rng = np.random.default_rng(0)
    buf.obs[:] = rng.standard_normal(buf.obs.shape).astype(np.float32)
    buf.state[:] = rng.standard_normal(buf.state.shape).astype(np.float32)
    buf.actions[:] = rng.integers(0, n_actions, size=buf.actions.shape).astype(np.float32)
    buf.log_probs[:] = -np.log(n_actions).astype(np.float32)
    buf.rewards[:] = rng.standard_normal(buf.rewards.shape).astype(np.float32)
    buf.dones[:] = (rng.random(buf.dones.shape) < 0.05).astype(np.float32)
    buf.compute_gae(np.zeros(buf.n_threads, dtype=np.float32), gamma=0.99, gae_lambda=0.95)


# ---------------------------------------------------------------- registry

def test_registry_has_all_expected_algos():
    expected = {"typpo", "tysac", "mappo", "happo", "hatrpo", "hasac", "qmix"}
    assert expected <= set(ALGO_REGISTRY.keys())


def test_registry_resolves_our_algos():
    assert get_actor("typpo") is TyPPO
    assert get_actor("tysac") is TySAC
    assert get_actor("mappo") is MAPPO


def test_registry_raises_for_unknown():
    with pytest.raises(KeyError):
        get_actor("not_a_real_algo")


# ---------------------------------------------------------------- TyPPO

def test_typpo_one_update():
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1

    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=GROUPS, discrete=True, device="cpu",
    )

    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)

    info = algo.train(buf, buf.advantages)
    assert set(info["group_losses"].keys()) == {0, 1, 2}
    for losses in info["group_losses"].values():
        assert np.isfinite(losses["policy_loss"])
        assert np.isfinite(losses["value_loss"])
        assert np.isfinite(losses["entropy"])
        assert np.isfinite(losses["approx_kl"])


def test_typpo_select_actions_shape():
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=GROUPS, discrete=True, device="cpu",
    )
    obs = torch.randn(N_AGENTS, OBS_DIM)
    actions, log_probs = algo.select_actions(obs)
    assert actions.shape == (N_AGENTS,)
    assert log_probs.shape == (N_AGENTS,)


@pytest.mark.parametrize("groups", [
    boundary_case_mappo(6),                      # all-in-one partition (MAPPO topology)
    boundary_case_happo(6),                      # singletons (HAPPO topology)
    [[0, 1, 2], [3, 4], [5]],                    # 3+2+1 main
    ["a", "a", "a", "b", "b", "c"],              # equivalent specified by types
])
def test_typpo_boundary_cases_construct(groups):
    """Construction-only check: TyPPO must accept extreme partition shapes
    without errors. This does NOT verify any behavioural identity with the
    standalone MAPPO/HAPPO algorithms -- those are independent
    implementations with different training loops."""
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=groups, discrete=True, device="cpu",
    )
    assert algo.num_agents == 6


def test_mappo_standalone():
    """MAPPO is a standalone class with a single shared actor."""
    cfg = _load_cfg("harl/configs/algos_cfgs/mappo.yaml")
    algo = MAPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        n_agents=6, discrete=True, device="cpu",
    )
    # Single shared actor — not a ModuleList.
    import torch.nn as nn
    assert isinstance(algo.actor, nn.Module) and not isinstance(algo.actor, nn.ModuleList)
    assert algo.n_agents == 6


def test_happo_standalone():
    """HAPPO is a standalone class with one actor per agent."""
    from harl.algorithms.actors.happo import HAPPO
    cfg = _load_cfg("harl/configs/algos_cfgs/happo.yaml")
    algo = HAPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        n_agents=6, discrete=True, device="cpu",
    )
    # ModuleList with 6 actors.
    assert len(algo.actors) == 6
    assert algo.n_agents == 6


def test_mappo_one_update():
    """MAPPO train step runs and produces finite losses."""
    cfg = _load_cfg("harl/configs/algos_cfgs/mappo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    algo = MAPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        n_agents=N_AGENTS, discrete=True, device="cpu",
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)
    info = algo.train(buf, buf.advantages)
    losses = info["group_losses"][0]
    assert np.isfinite(losses["policy_loss"])
    assert np.isfinite(losses["value_loss"])
    assert np.isfinite(losses["entropy"])


def test_happo_one_update():
    """HAPPO train step: per-agent sequential update with cumulative M."""
    from harl.algorithms.actors.happo import HAPPO
    cfg = _load_cfg("harl/configs/algos_cfgs/happo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    algo = HAPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        n_agents=N_AGENTS, discrete=True, device="cpu",
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)
    info = algo.train(buf, buf.advantages)
    # One entry per agent, in any order.
    assert len(info["group_losses"]) == N_AGENTS
    for losses in info["group_losses"].values():
        assert np.isfinite(losses["policy_loss"])


def test_hatrpo_one_update():
    """HATRPO train step: TRPO-style per-agent sequential update with M."""
    from harl.algorithms.actors.hatrpo import HATRPO
    cfg = _load_cfg("harl/configs/algos_cfgs/hatrpo.yaml")
    cfg["trpo"]["cg_iters"] = 3              # speed test up
    cfg["trpo"]["line_search_steps"] = 3
    cfg["trpo"]["critic_epoch"] = 1
    cfg["trpo"]["mini_batch"] = 8
    algo = HATRPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        n_agents=N_AGENTS, discrete=True, device="cpu",
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)
    info = algo.train(buf, buf.advantages)
    assert len(info["group_losses"]) == N_AGENTS
    for losses in info["group_losses"].values():
        # policy_loss is finite even when the line search rejects (returns 0).
        assert np.isfinite(losses["policy_loss"])
        assert np.isfinite(losses["value_loss"])
        # KL is non-negative (or 0 if line search rejected).
        assert losses["approx_kl"] >= 0.0
        # Step size is non-negative; capped by trust region.
        assert losses["trpo_step_size"] >= 0.0


# ---------------------------------------------------------------- TySAC

def test_tysac_one_update():
    cfg = _load_cfg("harl/configs/algos_cfgs/tysac.yaml")
    cfg["sac"]["batch_size"] = 8
    cfg["sac"]["warmup_steps"] = 0

    algo = TySAC(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=ACTION_DIM,
        groups=GROUPS, device="cpu",
    )

    buf = ReplayBuffer(
        capacity=64, n_agents=N_AGENTS, obs_dim=OBS_DIM,
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
    )
    rng = np.random.default_rng(0)
    for _ in range(64):
        buf.push(
            obs=rng.standard_normal((N_AGENTS, OBS_DIM)).astype(np.float32),
            state=rng.standard_normal((STATE_DIM,)).astype(np.float32),
            actions=rng.standard_normal((N_AGENTS, ACTION_DIM)).astype(np.float32),
            reward=np.float32(rng.standard_normal()),
            next_obs=rng.standard_normal((N_AGENTS, OBS_DIM)).astype(np.float32),
            next_state=rng.standard_normal((STATE_DIM,)).astype(np.float32),
            done=np.float32(0.0),
        )

    info = algo.train(buf)
    assert set(info["group_losses"].keys()) == {0, 1, 2}
    assert np.isfinite(info["critic_loss"])
    assert len(info["alphas"]) == 3
    for losses in info["group_losses"].values():
        assert np.isfinite(losses["actor_loss"])
        assert np.isfinite(losses["alpha"])


def test_tysac_per_group_q_heads():
    """GroupQCritic should have one head per group."""
    cfg = _load_cfg("harl/configs/algos_cfgs/tysac.yaml")
    algo = TySAC(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=ACTION_DIM,
        groups=GROUPS, device="cpu",
    )
    assert algo.critic.n_groups == 3
    state = torch.randn(2, STATE_DIM)
    joint_a = torch.randn(2, N_AGENTS * ACTION_DIM)
    q1, q2 = algo.critic(state, joint_a)
    assert q1.shape == (2, 3)
    assert q2.shape == (2, 3)


def test_tysac_state_dict_round_trip():
    cfg = _load_cfg("harl/configs/algos_cfgs/tysac.yaml")
    algo = TySAC(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=ACTION_DIM,
        groups=GROUPS, device="cpu",
    )
    sd = algo.state_dict()
    algo2 = TySAC(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=ACTION_DIM,
        groups=GROUPS, device="cpu",
    )
    algo2.load_state_dict(sd)

    # Same params after load.
    for p1, p2 in zip(algo.actors[0].parameters(), algo2.actors[0].parameters()):
        assert torch.allclose(p1, p2)


# ---------------------------------------------------------------- dynamic groups

def test_typpo_dynamic_groups_set_groups():
    """In dynamic mode, set_groups should re-route a new partition to the
    pre-allocated actors keyed by canonical labels."""
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    canonical = ["stalker", "zealot", "colossus"]

    initial_types = ["stalker", "stalker", "zealot", "zealot", "colossus"]
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=initial_types, discrete=True, device="cpu",
        n_canonical_groups=3, canonical_group_labels=canonical,
    )
    assert algo.dynamic_groups
    assert len(algo.actors) == 3                # one per canonical type
    assert algo.num_agents == 5

    # Reassign with a different partition (no colossus this episode).
    algo.set_groups(["stalker", "stalker", "stalker", "zealot", "zealot"])
    assert algo.num_agents == 5
    active = algo._active_actor_indices()
    assert sorted(active) == [0, 1]            # stalker(0) + zealot(1)
    # Colossus actor (index 2) is not active.
    assert 2 not in active


def test_typpo_dynamic_groups_skip_empty_in_train():
    """Train step should silently skip canonical types that didn't spawn."""
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    canonical = ["stalker", "zealot", "colossus"]

    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=["stalker", "zealot", "zealot", "colossus", "colossus"],
        discrete=True, device="cpu",
        n_canonical_groups=3, canonical_group_labels=canonical,
    )
    # Now switch to a partition with no colossus at all.
    algo.set_groups(["stalker", "stalker", "zealot", "zealot", "zealot"])

    buf = OnPolicyBuffer(T, B, 5, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)

    info = algo.train(buf, buf.advantages)
    assert info["active_groups"] == 2         # colossus skipped
    assert 2 not in info["group_losses"]      # canonical-slot 2 = colossus


def test_typpo_dynamic_groups_reject_unknown_label():
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    canonical = ["stalker", "zealot", "colossus"]
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=["stalker", "zealot"], discrete=True, device="cpu",
        n_canonical_groups=3, canonical_group_labels=canonical,
    )
    with pytest.raises(ValueError, match="not in canonical"):
        algo.set_groups(["stalker", "marine"])  # marine is not protoss


def test_typpo_static_mode_rejects_n_groups_change():
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=GROUPS, discrete=True, device="cpu",
    )
    # Same n_groups OK.
    algo.set_groups([[0, 1], [2, 3, 4], [5]])
    # Different n_groups → reject.
    with pytest.raises(ValueError, match="static mode"):
        algo.set_groups([[0, 1, 2, 3, 4, 5]])


# ---------------------------------------------------------------- 
# TyPPO M-matrix propagation verification
# ---------------------------------------------------------------- 

def test_M_starts_at_one_in_typpo():
    """First group in TyPPO's seq order must see M=1 (no prior groups)."""
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    cfg["sequential_update"]["random_group_permutation"] = False
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=GROUPS, discrete=True, device="cpu",
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)

    seen_M = []
    orig = algo._update_group_actor
    def spy(buffer, M, adv, group_id, agent_ids):
        seen_M.append(M.detach().cpu().numpy().copy())
        return orig(buffer, M, adv, group_id, agent_ids)
    algo._update_group_actor = spy

    algo.train(buf, buf.advantages)
    assert np.allclose(seen_M[0], 1.0), "First group must see M=1"


def test_M_propagates_through_typpo_groups():
    """For TyPPO with K>1, M for groups updated AFTER the first must differ
    from 1 — i.e., the cumulative ratio of prior groups actually flows
    forward. Without this, TyPPO degenerates to running parameter-shared
    PPO independently on each type."""
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    cfg["sequential_update"]["random_group_permutation"] = False
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=GROUPS, discrete=True, device="cpu",   # 3 groups
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)

    seen_M = []
    orig = algo._update_group_actor
    def spy(buffer, M, adv, group_id, agent_ids):
        seen_M.append(M.detach().cpu().numpy().copy())
        return orig(buffer, M, adv, group_id, agent_ids)
    algo._update_group_actor = spy

    algo.train(buf, buf.advantages)

    assert len(seen_M) == 3
    assert np.allclose(seen_M[0], 1.0)
    assert not np.allclose(seen_M[1], 1.0)
    assert not np.allclose(seen_M[2], 1.0)


def test_happo_M_accumulates():
    """HAPPO standalone: each successive agent must see a non-trivial M."""
    from harl.algorithms.actors.happo import HAPPO
    cfg = _load_cfg("harl/configs/algos_cfgs/happo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    cfg["sequential_update"]["random_group_permutation"] = False
    algo = HAPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        n_agents=N_AGENTS, discrete=True, device="cpu",
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)

    seen_M = []
    orig = algo._update_agent_actor
    def spy(buffer, M, adv, agent_id):
        seen_M.append(M.detach().cpu().numpy().copy())
        return orig(buffer, M, adv, agent_id)
    algo._update_agent_actor = spy

    algo.train(buf, buf.advantages)
    assert len(seen_M) == N_AGENTS
    assert np.allclose(seen_M[0], 1.0)
    later = np.concatenate([m.reshape(-1) for m in seen_M[1:]])
    assert not np.allclose(later, 1.0), "HAPPO must accumulate cumulative ratios"


def test_advantage_normalisation_is_not_redundant():
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 1
    algo = TyPPO(
        args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
        groups=GROUPS, discrete=True, device="cpu",
    )
    buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, ACTION_DIM)
    _fill_on_policy_buffer(buf, N_ACTIONS)
    buf.advantages[:] = np.random.RandomState(0).normal(50.0, 10.0, size=buf.advantages.shape).astype(np.float32)
    info = algo.train(buf, buf.advantages)
    for losses in info["group_losses"].values():
        assert np.isfinite(losses["policy_loss"])
        assert np.isfinite(losses["value_loss"])
