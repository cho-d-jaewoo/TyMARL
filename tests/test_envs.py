"""Tests for ``harl.envs.smacv2``.

Uses the dummy SC2 backend so this runs anywhere — no StarCraft install
required. The dummy backend kicks in transparently when ``smacv2.env``
import fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from harl.envs.smacv2 import SMACv2Env, SMACV2_UNIT_POOLS
from harl.utils.group import GroupAssignment


@pytest.mark.parametrize("race", ["protoss", "terran", "zerg"])
def test_env_constructs_with_dummy_backend(race):
    env = SMACv2Env(scenario=f"{race}_5_vs_5", seed=0)
    assert env.race == race
    assert env.num_agents == 5
    assert env.canonical_types == SMACV2_UNIT_POOLS[race]


def test_reset_yields_a_group_assignment():
    env = SMACv2Env(scenario="protoss_5_vs_5", seed=0)
    obs, state, groups = env.reset()
    assert isinstance(groups, GroupAssignment)
    assert groups.n_agents == 5
    assert 1 <= groups.n_groups <= 3            # a subset of the 3-type pool


def test_obs_is_augmented_with_canonical_unit_type_one_hot():
    env = SMACv2Env(scenario="terran_5_vs_5", seed=1)
    obs, state, groups = env.reset()
    info = env.get_env_info()
    # post-augmentation obs_shape == raw_obs_shape + n_canonical_types
    assert info["obs_shape"] == env._raw_obs_dim + env.n_canonical_types
    assert obs.shape == (env.num_agents, info["obs_shape"])
    # The trailing 3 dims should be a one-hot per agent.
    onehot_block = obs[:, -env.n_canonical_types:]
    assert np.allclose(onehot_block.sum(axis=1), 1.0)


def test_groups_are_recomputed_each_reset():
    """SMACv2's stochastic team should give different partitions across resets."""
    env = SMACv2Env(scenario="zerg_5_vs_5", seed=0)
    seen_signatures = set()
    for _ in range(10):
        obs, state, groups = env.reset()
        # The ordered tuple of group labels uniquely identifies the partition.
        signature = (tuple(groups.group_labels), tuple(groups.group_sizes))
        seen_signatures.add(signature)
    # Dummy RNG occasionally produces the same composition twice; we want
    # to see at least 2 distinct partitions in 10 trials, which is virtually
    # certain when types are sampled uniformly.
    assert len(seen_signatures) >= 2, (
        f"Expected stochastic team composition; got constant: {seen_signatures}"
    )


def test_groups_changed_flag_propagates():
    """Each ``info`` from the post-reset step should report the change flag."""
    env = SMACv2Env(scenario="protoss_5_vs_5", seed=0)
    env.reset()
    joint = np.zeros(env.num_agents, dtype=np.int64)
    _, _, _, _, info = env.step(joint)
    assert "groups_changed" in info             # True on first step after reset


def test_episode_runs_to_done():
    env = SMACv2Env(scenario="terran_5_vs_5", seed=3)
    env.reset()
    done = False
    n = 0
    while not done and n < 1000:
        joint = np.zeros(env.num_agents, dtype=np.int64)
        _, _, _, done, _ = env.step(joint)
        n += 1
    assert done, "dummy backend should always terminate within episode_limit"


def test_env_info_announces_dynamic_groups():
    env = SMACv2Env(scenario="protoss_5_vs_5", seed=0)
    env.reset()
    info = env.get_env_info()
    assert info["dynamic_groups"] is True
    assert info["n_canonical_types"] == 3
    assert info["canonical_types"] == ["stalker", "zealot", "colossus"]
