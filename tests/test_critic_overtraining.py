"""Regression tests for the critic-overtraining bug.

The bug
-------
Earlier versions of HAPPO and TyPPO stepped ``critic_optim`` *inside* the
per-agent / per-group sequential update loop. With ``n_agents=5`` and
``ppo_epoch=5`` that meant the centralised critic was being updated 25
times per training iteration vs MAPPO's 5. The over-trained critic
produced noisy value estimates that fed back into the next iteration's
GAE, and was a major reason HAPPO under-performed MAPPO on SMACv2 in
our earlier benchmark.

The fix
-------
Both ``HAPPO.train`` and ``TyPPO.train`` now run a single critic-only
``_update_critic`` after the sequential actor loop is finished. This
brings the critic update count back to ``ppo_epoch`` per iteration,
matching MAPPO.

These tests count actual ``critic_optim.step`` calls during one
``algo.train(...)`` and assert the count is independent of n_agents /
n_groups.
"""

from __future__ import annotations

import numpy as np
import torch
import yaml

from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.typpo import TyPPO
from harl.utils.buffers import OnPolicyBuffer


# --- Test fixtures match those in test_algorithms.py ---
GROUPS = [[0, 1, 2], [3, 4], [5]]
N_AGENTS = sum(len(g) for g in GROUPS)
OBS_DIM = 8
STATE_DIM = 32
N_ACTIONS = 5
T = 16
B = 2


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fill_buffer(buf: OnPolicyBuffer, n_actions: int) -> None:
    rng = np.random.default_rng(0)
    for t in range(buf.size):
        for b in range(buf.n_threads):
            buf.insert(
                obs=rng.normal(size=(buf.n_threads, buf.n_agents, OBS_DIM)).astype(np.float32)[0:1],
                state=rng.normal(size=(buf.n_threads, STATE_DIM)).astype(np.float32)[0:1],
                next_state=rng.normal(size=(buf.n_threads, STATE_DIM)).astype(np.float32)[0:1],
                actions=rng.integers(0, n_actions, size=(buf.n_threads, buf.n_agents, 1)).astype(np.float32)[0:1],
                log_probs=rng.normal(size=(buf.n_threads, buf.n_agents)).astype(np.float32)[0:1],
                values=rng.normal(size=(buf.n_threads,)).astype(np.float32)[0:1],
                rewards=rng.normal(size=(buf.n_threads,)).astype(np.float32)[0:1],
                dones=np.zeros((buf.n_threads,), dtype=np.float32),
            )
    buf.compute_gae(np.zeros((buf.n_threads,), dtype=np.float32), gamma=0.99, gae_lambda=0.95)


class _CritStepCounter:
    """Wraps an optimiser, counts ``.step()`` calls."""
    def __init__(self, opt):
        self._opt = opt
        self.count = 0
    def __getattr__(self, name):
        return getattr(self._opt, name)
    def step(self, *a, **kw):
        self.count += 1
        return self._opt.step(*a, **kw)


def test_happo_critic_steps_independent_of_n_agents():
    """HAPPO's critic must be updated ``ppo_epoch * num_minibatches`` times
    per ``train()``, regardless of how many agents there are."""
    cfg = _load_cfg("harl/configs/algos_cfgs/happo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 3
    # Buffer holds T * B = 16 * 2 = 32 samples; mb_size = 8 -> 4 minibatches/epoch.
    n_minibatches_per_epoch = (T * B + cfg["ppo"]["mini_batch"] - 1) // cfg["ppo"]["mini_batch"]
    expected_steps = cfg["ppo"]["ppo_epoch"] * n_minibatches_per_epoch

    for n_agents in [2, 5, 8]:
        algo = HAPPO(
            args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
            n_agents=n_agents, discrete=True, device="cpu",
        )
        buf = OnPolicyBuffer(T, B, n_agents, OBS_DIM, STATE_DIM, 1)
        _fill_buffer(buf, N_ACTIONS)

        counter = _CritStepCounter(algo.critic_optim)
        algo.critic_optim = counter

        algo.train(buf, buf.advantages)

        assert counter.count == expected_steps, (
            f"HAPPO critic was stepped {counter.count} times with n_agents={n_agents}; "
            f"expected {expected_steps}. Critic over-training regression: the critic "
            f"step is leaking back inside the per-agent sequential loop."
        )


def test_typpo_critic_steps_independent_of_n_groups():
    """TyPPO's critic must be updated ``ppo_epoch * num_minibatches`` times
    per ``train()``, regardless of how many type-groups exist."""
    cfg = _load_cfg("harl/configs/algos_cfgs/typpo.yaml")
    cfg["ppo"]["mini_batch"] = 8
    cfg["ppo"]["ppo_epoch"] = 3
    n_minibatches_per_epoch = (T * B + cfg["ppo"]["mini_batch"] - 1) // cfg["ppo"]["mini_batch"]
    expected_steps = cfg["ppo"]["ppo_epoch"] * n_minibatches_per_epoch

    # Try several group partitions of the same N_AGENTS.
    for groups in [
        [list(range(N_AGENTS))],                  # 1 group
        [[0, 1, 2], [3, 4, 5]],                   # 2 groups
        [[0, 1, 2], [3, 4], [5]],                 # 3 groups
        [[0], [1], [2], [3], [4], [5]],           # 6 groups (HAPPO-like extreme)
    ]:
        algo = TyPPO(
            args=cfg, obs_dim=OBS_DIM, state_dim=STATE_DIM, action_dim=N_ACTIONS,
            groups=groups, discrete=True, device="cpu",
        )
        buf = OnPolicyBuffer(T, B, N_AGENTS, OBS_DIM, STATE_DIM, 1)
        _fill_buffer(buf, N_ACTIONS)

        counter = _CritStepCounter(algo.critic_optim)
        algo.critic_optim = counter

        algo.train(buf, buf.advantages)

        assert counter.count == expected_steps, (
            f"TyPPO critic was stepped {counter.count} times with "
            f"n_groups={len(groups)}; expected {expected_steps}. Critic over-training "
            f"regression: the critic step is leaking back inside the per-group loop."
        )
