"""
Replay buffers.

Two buffer types are provided:

- ``OnPolicyBuffer`` for TyPPO/HAPPO/MAPPO. Stores per-step
  (obs, action, log_prob, value, reward, done) and computes GAE advantages
  on demand. Shapes are agent-major: ``(T, n_threads, n_agents, *)``.

  In addition to the canonical PPO fields, the buffer also stores three
  optional per-step fields that are essential for SMAC-family environments:

  * ``available_actions``: per-(t, b, agent) boolean mask of legal actions.
    Required by SMACv2 because invalid actions otherwise contaminate the
    PPO ratio at both rollout time and update time.
  * ``active_masks``: per-(t, b, agent) float mask, 1.0 if the agent was
    alive at that step, 0.0 if dead. Used to weight the policy / entropy
    loss so dead agents do not corrupt the gradient.
  * ``actor_id_per_step``: per-(t, b, agent) int identifier of which
    actor / canonical type generated that action. This is what makes
    type-routed update correct under SMACv2's per-episode stochastic team
    composition: at training time the algorithm masks each actor's update
    to the (t, b, agent) triples that the *same* actor handled during the
    rollout, regardless of any reset that happened in between.

  All three fields are *optional*: if the env / runner does not populate
  them the buffer just leaves them as ``None`` and the algorithms fall
  back to the legacy "use the algorithm's current group assignment for
  the whole buffer" behaviour. The runner populates them whenever the
  underlying env exposes ``get_avail_actions``, ``get_alive_mask``, and
  ``get_canonical_type_ids`` respectively.

- ``ReplayBuffer`` for TySAC/HASAC/MADDPG. Stores transitions and yields
  random batches.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch


class OnPolicyBuffer:
    """Synchronous on-policy buffer with GAE."""

    def __init__(
        self,
        size: int,
        n_threads: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        shape = (size, n_threads, n_agents)
        self.obs = np.zeros((*shape, obs_dim), dtype=np.float32)
        self.state = np.zeros((size, n_threads, state_dim), dtype=np.float32)
        # next_state[t,b,:] = state observed *after* taking the action at
        # step t in thread b. Used for accurate GAE bootstrapping when the
        # rollout is cut short before terminal.
        self.next_state = np.zeros((size, n_threads, state_dim), dtype=np.float32)
        self.actions = np.zeros((*shape, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.values = np.zeros((size, n_threads), dtype=np.float32)
        self.rewards = np.zeros((size, n_threads), dtype=np.float32)
        self.dones = np.zeros((size, n_threads), dtype=np.float32)

        # Optional SMAC-family fields. Allocated lazily on first insert so
        # callers that never use them pay no memory cost.
        self.available_actions: np.ndarray | None = None
        self.active_masks: np.ndarray | None = None
        # Per-(t, b, agent) actor id. -1 is the sentinel for "no info".
        self.actor_id_per_step: np.ndarray | None = None

        # Computed by ``compute_gae``.
        self.advantages = np.zeros((size, n_threads), dtype=np.float32)
        self.returns = np.zeros((size, n_threads), dtype=np.float32)

        self.size = size
        self.n_threads = n_threads
        self.n_agents = n_agents
        self.step = 0
        self.device = device

    # ------------------------------------------------------------------
    # Insert / consume
    # ------------------------------------------------------------------

    def insert(self, **kwargs) -> None:
        # Strip optional fields that are None so lazy allocation below doesn't
        # try to derive shape from a 0-d ``np.asarray(None)``.
        for _opt_key in ("available_actions", "active_masks", "actor_id_per_step"):
            if _opt_key in kwargs and kwargs[_opt_key] is None:
                del kwargs[_opt_key]

        # Lazy allocation for optional SMAC-family fields, so the runner
        # can simply pass them in without any setup boilerplate.
        if "available_actions" in kwargs and self.available_actions is None:
            v = np.asarray(kwargs["available_actions"])
            n_actions = v.shape[-1]
            self.available_actions = np.zeros(
                (self.size, self.n_threads, self.n_agents, n_actions),
                dtype=np.float32,
            )
        if "active_masks" in kwargs and self.active_masks is None:
            self.active_masks = np.ones(
                (self.size, self.n_threads, self.n_agents),
                dtype=np.float32,
            )
        if "actor_id_per_step" in kwargs and self.actor_id_per_step is None:
            self.actor_id_per_step = -np.ones(
                (self.size, self.n_threads, self.n_agents),
                dtype=np.int64,
            )

        for k, v in kwargs.items():
            if v is None:
                continue
            getattr(self, k)[self.step] = v
        self.step = (self.step + 1) % self.size

    def compute_gae(self, last_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        adv = 0.0
        for t in reversed(range(self.size)):
            next_value = last_value if t == self.size - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[:] = self.advantages + self.values

    def get(
        self,
        agent_ids: Sequence[int],
        batch_size: int,
        M: "torch.Tensor | None" = None,
        adv: "torch.Tensor | None" = None,
    ):
        """Yield mini-batches restricted to a subset of agents (a group).

        Each yielded dict has obs/actions/log_probs sliced to ``agent_ids``
        and the centralised state/returns/values broadcast across the same
        (time, thread) index.

        Optional fields included when present in the buffer:
        - ``available_actions``: shape ``(B_mb, |G|, n_actions)``
        - ``active_masks``:      shape ``(B_mb, |G|)``
        - ``actor_id_per_step``: shape ``(B_mb, |G|)``

        Optional inputs:
        - ``M`` : cumulative scalar ratio of shape (T, B). The slice
          ``M[t, b]`` is included under key ``"M"``. Used by TyPPO to weight
          advantages by the cumulative ratio of already-updated groups.
        - ``adv`` : pre-normalised advantages of shape (T, B). The slice
          ``adv[t, b]`` is included under key ``"adv"``. Caller normalises
          once over the full iter to avoid double-normalisation that would
          cancel out M weighting.
        """
        n = self.size * self.n_threads
        idx = np.random.permutation(n)
        agent_ids_np = np.asarray(list(agent_ids), dtype=np.int64)
        for start in range(0, n, batch_size):
            sel = idx[start : start + batch_size]
            t, b = sel // self.n_threads, sel % self.n_threads
            out = {
                "obs":        torch.as_tensor(self.obs[t, b][:, agent_ids_np], device=self.device),
                "actions":    torch.as_tensor(self.actions[t, b][:, agent_ids_np], device=self.device),
                "log_probs":  torch.as_tensor(self.log_probs[t, b][:, agent_ids_np], device=self.device),
                "advantages": torch.as_tensor(self.advantages[t, b], device=self.device),
                "returns":    torch.as_tensor(self.returns[t, b], device=self.device),
                "values":     torch.as_tensor(self.values[t, b], device=self.device),
                "state":      torch.as_tensor(self.state[t, b], device=self.device),
            }
            if self.available_actions is not None:
                out["available_actions"] = torch.as_tensor(
                    self.available_actions[t, b][:, agent_ids_np], device=self.device
                )
            if self.active_masks is not None:
                out["active_masks"] = torch.as_tensor(
                    self.active_masks[t, b][:, agent_ids_np], device=self.device
                )
            if self.actor_id_per_step is not None:
                out["actor_id_per_step"] = torch.as_tensor(
                    self.actor_id_per_step[t, b][:, agent_ids_np], device=self.device
                )
            if M is not None:
                t_idx = torch.as_tensor(t, device=M.device, dtype=torch.long)
                b_idx = torch.as_tensor(b, device=M.device, dtype=torch.long)
                # Scalar M: shape (T, B) → indexed to (B_mb,)
                if M.dim() == 2:
                    out["M"] = M[t_idx, b_idx]
                elif M.dim() == 3:
                    # Backwards-compat: per-agent M (T, B, N) → reduce to scalar
                    # by taking the value at the FIRST agent of the group (all
                    # agents in a group share M under HAPPO Eq 10).
                    ag_idx = torch.as_tensor(list(agent_ids)[:1], device=M.device, dtype=torch.long)
                    out["M"] = M[t_idx, b_idx][:, ag_idx].squeeze(-1)
                else:
                    raise ValueError(f"M has unexpected ndim={M.dim()}")
            if adv is not None:
                t_idx = torch.as_tensor(t, device=adv.device, dtype=torch.long)
                b_idx = torch.as_tensor(b, device=adv.device, dtype=torch.long)
                out["adv"] = adv[t_idx, b_idx]
            yield out

    def size_per_agent(self) -> int:
        """Total number of (time, thread) samples in the buffer."""
        return int(self.size * self.n_threads)

    def after_update(self) -> None:
        self.step = 0


class ReplayBuffer:
    """Plain off-policy replay buffer with uniform sampling."""

    def __init__(
        self,
        capacity: int,
        n_agents: int,
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros_like(self.obs)
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros_like(self.state)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.ptr = 0
        self.full = False
        self.device = device

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr

    def push(self, obs, state, actions, reward, next_obs, next_state, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.state[i] = state
        self.actions[i] = actions
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_state[i] = next_state
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = np.random.randint(0, len(self), size=batch_size)
        return {
            "obs":        torch.as_tensor(self.obs[idx], device=self.device),
            "next_obs":   torch.as_tensor(self.next_obs[idx], device=self.device),
            "state":      torch.as_tensor(self.state[idx], device=self.device),
            "next_state": torch.as_tensor(self.next_state[idx], device=self.device),
            "actions":    torch.as_tensor(self.actions[idx], device=self.device),
            "rewards":    torch.as_tensor(self.rewards[idx], device=self.device).unsqueeze(-1),
            "dones":      torch.as_tensor(self.dones[idx], device=self.device).unsqueeze(-1),
        }
