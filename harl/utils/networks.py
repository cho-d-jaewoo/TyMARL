"""
Shared neural-network building blocks.

Architecture and initialisation choices match the MAPPO paper (Yu et al.
2022) and HAPPO paper (Kuba et al. 2022) so that our reproductions of
those baselines hit the published numbers:

* two hidden layers of width 64 (MAPPO/HAPPO standard for SMAC)
* ReLU activation
* Orthogonal weight init with gain=sqrt(2) for hidden layers,
  gain=0.01 for the policy output, gain=1.0 for the value output
* Bias init to zero
"""

from __future__ import annotations

from typing import Optional, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


def init_(layer: nn.Linear, gain: float = math.sqrt(2.0), bias: float = 0.0) -> nn.Linear:
    """In-place orthogonal init with bias=const. Matches MAPPO/HAPPO baselines."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(sizes: Sequence[int], activation: type = nn.ReLU, last_gain: float | None = None) -> nn.Sequential:
    """MLP with orthogonal init. Hidden layers use gain=sqrt(2); the final
    layer uses ``last_gain`` if provided (otherwise sqrt(2) too)."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        is_last = i == len(sizes) - 2
        gain = last_gain if (is_last and last_gain is not None) else math.sqrt(2.0)
        layers.append(init_(nn.Linear(sizes[i], sizes[i + 1]), gain=gain))
        if not is_last:
            layers.append(activation())
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# Actors
# -----------------------------------------------------------------------------

class CategoricalActor(nn.Module):
    """MLP actor with a Categorical output head (discrete actions).

    Matches MAPPO/HAPPO paper architecture: one hidden layer of width 64,
    ReLU activation, orthogonal init, gain=0.01 on the output head.

    Supports an optional boolean ``available_actions`` mask of shape
    ``(B, n_actions)`` that zeroes out logits for unavailable moves —
    needed for SMAC-family environments where dead/disabled agents have no
    legal actions.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        body_sizes = [obs_dim, *hidden]
        body_layers: list[nn.Module] = []
        for i in range(len(body_sizes) - 1):
            body_layers.append(init_(nn.Linear(body_sizes[i], body_sizes[i + 1]),
                                     gain=math.sqrt(2.0)))
            body_layers.append(nn.ReLU())
        self.body = nn.Sequential(*body_layers)
        self.head = init_(nn.Linear(hidden[-1], n_actions), gain=0.01)

    def _logits(
        self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = self.head(self.body(obs))
        if available_actions is not None:
            logits = logits.masked_fill(~available_actions.bool(), -1e10)
        return logits

    def forward(
        self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None
    ) -> Categorical:
        return Categorical(logits=self._logits(obs, available_actions))

    @torch.no_grad()
    def act(
        self, obs: torch.Tensor, available_actions: Optional[torch.Tensor] = None
    ):
        dist = self.forward(obs, available_actions)
        action = dist.sample()
        return action, dist.log_prob(action)


class GaussianActor(nn.Module):
    """MLP actor with a tanh-squashed diagonal-Gaussian output head."""

    LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0

    def __init__(self, obs_dim: int, action_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        body_sizes = [obs_dim, *hidden]
        body_layers: list[nn.Module] = []
        for i in range(len(body_sizes) - 1):
            body_layers.append(init_(nn.Linear(body_sizes[i], body_sizes[i + 1]),
                                     gain=math.sqrt(2.0)))
            body_layers.append(nn.ReLU())
        self.body = nn.Sequential(*body_layers)
        self.mu = init_(nn.Linear(hidden[-1], action_dim), gain=0.01)
        self.log_std = init_(nn.Linear(hidden[-1], action_dim), gain=0.01)

    def forward(self, obs: torch.Tensor) -> Normal:
        h = self.body(obs)
        log_std = self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(self.mu(h), log_std.exp())

    def sample(self, obs: torch.Tensor):
        """Reparametrised sample with tanh squashing.

        Returns
        -------
        a : (B, action_dim) tensor in (-1, 1)
        log_prob : (B, 1) tensor — total log-prob of the squashed action
        """
        dist = self.forward(obs)
        u = dist.rsample()
        a = torch.tanh(u)
        log_prob = dist.log_prob(u) - torch.log(1.0 - a.pow(2) + 1e-6)
        return a, log_prob.sum(-1, keepdim=True)


# -----------------------------------------------------------------------------
# Critics
# -----------------------------------------------------------------------------

class VCritic(nn.Module):
    """Centralised V-network (state -> scalar value); used by TyPPO."""

    def __init__(self, state_dim: int, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        sizes = [state_dim, *hidden, 1]
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            is_last = i == len(sizes) - 2
            gain = 1.0 if is_last else math.sqrt(2.0)
            layers.append(init_(nn.Linear(sizes[i], sizes[i + 1]), gain=gain))
            if not is_last:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class TwinQCritic(nn.Module):
    """Plain centralised twin-Q critic (state, joint-action) -> scalar.

    Provided for completeness; TySAC uses ``GroupQCritic`` instead.
    """

    def __init__(
        self, state_dim: int, joint_action_dim: int, hidden: Sequence[int] = (256, 256)
    ):
        super().__init__()
        in_dim = state_dim + joint_action_dim
        self.q1 = _mlp([in_dim, *hidden, 1], last_gain=1.0)
        self.q2 = _mlp([in_dim, *hidden, 1], last_gain=1.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# -----------------------------------------------------------------------------
# Value normaliser (PopArt-lite)
# -----------------------------------------------------------------------------

class ValueNormalizer:
    """Running mean/std normaliser for value targets.

    Used by TyPPO/MAPPO/HAPPO to stabilise critic learning per MAPPO paper
    Suggestion 1 ("Value Normalisation"). The critic regresses to
    *normalised* return targets; we denormalise on the inference path so
    GAE/value-clipping work in the original return scale.
    """

    def __init__(self, eps: float = 1e-6, beta: float = 0.99999):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0.0
        self.eps = eps
        self.beta = beta

    def update(self, x) -> None:
        import numpy as _np
        if isinstance(x, torch.Tensor):
            x = x.detach().reshape(-1).cpu().numpy()
        else:
            x = _np.asarray(x).reshape(-1)
        batch_mean = float(x.mean())
        batch_var = float(x.var())
        batch_count = int(x.size)
        if self.count == 0:
            self.mean = batch_mean
            self.var = max(batch_var, self.eps)
            self.count = float(batch_count)
            return
        delta = batch_mean - self.mean
        tot = self.count * self.beta + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count * self.beta
        m_b = batch_var * batch_count
        M2 = m_a + m_b + (delta ** 2) * (self.count * self.beta) * batch_count / tot
        self.var = max(M2 / tot, self.eps)
        self.count = tot

    @property
    def std(self) -> float:
        return float(self.var) ** 0.5 + self.eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean
