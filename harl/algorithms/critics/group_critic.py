"""
Centralised critics used by TyPPO and TySAC.

Like in HARL, the critic is global (takes the joint state) and is shared across
all agents. The actor is what is sharded by group.

Two critics live here:

- ``GroupCritic``  : V(s) — a single-head value network used by TyPPO.
- ``GroupQCritic`` : Q_k(s, a_joint) — twin Q with one head per group, used by
  TySAC. Per-group heads are an architectural ingredient of TySAC: the
  gradient flowing into group k's actor goes through head k only, mirroring the
  per-agent twin-critic structure of HASAC at the group granularity.

We keep the architectures intentionally small and standard so any measured
gain from TyM* is attributable to the actor side (the sequential update),
not to a fancier critic.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn


def _mlp(sizes: Sequence[int], activation=nn.ReLU, output_activation=nn.Identity) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# -----------------------------------------------------------------------------
# V critic
# -----------------------------------------------------------------------------

class GroupCritic(nn.Module):
    """V(s) — a single value head over the joint state.

    TyPPO uses GAE on this single V; per-agent advantages are obtained by
    broadcasting V to all agents of the team.
    """

    def __init__(self, state_dim: int, hidden_sizes: Sequence[int] = (256, 256)):
        super().__init__()
        self.net = _mlp([state_dim, *hidden_sizes, 1])

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


# -----------------------------------------------------------------------------
# Twin Q critic with per-group heads
# -----------------------------------------------------------------------------

class GroupQCritic(nn.Module):
    """Twin Q with per-group heads on a shared trunk.

    ``forward(state, joint_action)`` returns ``(q1, q2)``, each of shape
    ``(B, n_groups)``. Group k's actor is updated using head k's gradient
    only, so heads decouple cleanly across groups while the trunk shares
    representation learning.

    This is the only architectural addition we make to HASAC's critic; an
    ablation in the paper (Section 7.4) drops the per-group heads and confirms
    that the gain is mainly on the actor side, but the per-group head still
    helps marginally.
    """

    def __init__(
        self,
        state_dim: int,
        joint_action_dim: int,
        n_groups: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ):
        super().__init__()
        self.n_groups = n_groups

        def _make_branch():
            trunk = _mlp(
                [state_dim + joint_action_dim, *hidden_sizes],
                activation=nn.ReLU,
                output_activation=nn.ReLU,
            )
            heads = nn.ModuleList(
                [nn.Linear(hidden_sizes[-1], 1) for _ in range(n_groups)]
            )
            return trunk, heads

        self.trunk1, self.heads1 = _make_branch()
        self.trunk2, self.heads2 = _make_branch()

    def _forward_branch(
        self, trunk: nn.Sequential, heads: nn.ModuleList,
        state: torch.Tensor, joint_action: torch.Tensor,
    ) -> torch.Tensor:
        feat = trunk(torch.cat([state, joint_action], dim=-1))
        return torch.cat([h(feat) for h in heads], dim=-1)             # (B, K)

    def forward(
        self, state: torch.Tensor, joint_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q1 = self._forward_branch(self.trunk1, self.heads1, state, joint_action)
        q2 = self._forward_branch(self.trunk2, self.heads2, state, joint_action)
        return q1, q2
