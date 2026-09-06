"""
HASAC — Heterogeneous-Agent Soft Actor-Critic
        (Liu et al., NeurIPS 2024 / arXiv 2304.09870).

Standalone implementation: one actor per agent, sequential per-agent
update with a cumulative log-ratio multiplier (the off-policy analogue
of HAPPO's M). Twin Q with one head per agent.

Sits at the heterogeneous extreme of the SAC family in our paper —
TySAC (also in this repo) sits between HASAC and parameter-shared SAC by
sharing actor parameters within a type. HASAC is implemented separately
from TySAC for clarity, even though structurally TySAC at K=N is similar.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.utils.networks import GaussianActor
from harl.algorithms.critics.group_critic import GroupQCritic


class HASAC:
    """Heterogeneous-Agent SAC with per-agent actors and Q-heads."""

    def __init__(
        self,
        args: Dict[str, Any],
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        device: torch.device | str = "cpu",
    ):
        self.args = args
        self.device = torch.device(device)
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)

        # One actor + one target actor per agent.
        self.actors = nn.ModuleList(
            [GaussianActor(obs_dim, action_dim).to(self.device) for _ in range(self.n_agents)]
        )
        self.target_actors = nn.ModuleList(
            [copy.deepcopy(a) for a in self.actors]
        )
        for ta in self.target_actors:
            for p in ta.parameters():
                p.requires_grad = False

        # Twin Q-critic with one head per agent.
        self.critic = GroupQCritic(
            state_dim, n_agents * action_dim, n_groups=self.n_agents
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        opt = args["optim"]
        sac = args["sac"]

        self.actor_optims = [
            torch.optim.Adam(a.parameters(), lr=opt["actor_lr"]) for a in self.actors
        ]
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=opt["critic_lr"])

        self.gamma: float = float(sac["gamma"])
        self.tau: float = float(sac["tau"])
        self.batch_size: int = int(sac["batch_size"])

        # Per-agent temperature.
        self.auto_alpha: bool = bool(sac["auto_alpha"])
        init_log_alpha = float(np.log(sac["alpha"]))
        if self.auto_alpha:
            self.log_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(init_log_alpha, device=self.device))
                for _ in range(self.n_agents)
            ])
            self.alpha_optims = [
                torch.optim.Adam([la], lr=opt["alpha_lr"]) for la in self.log_alphas
            ]
            target_entropy = sac["target_entropy"]
            if target_entropy == "auto":
                target_entropy = -float(action_dim)
            self.target_entropy = float(target_entropy)
            self.alpha_min = float(sac.get("alpha_min", 0.05))
        else:
            self.log_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(init_log_alpha, device=self.device),
                             requires_grad=False)
                for _ in range(self.n_agents)
            ])
            self.alpha_optims = []
            self.alpha_min = float(sac.get("alpha_min", 0.05))

        seq = args.get("sequential_update", {})
        self.random_agent_permutation: bool = bool(seq.get("random_group_permutation", True))
        self._rng = np.random.default_rng(int(args.get("seed", 0)))

    def alpha(self, agent_id: int) -> torch.Tensor:
        return self.log_alphas[agent_id].exp()

    def _agent_order(self) -> np.ndarray:
        order = np.arange(self.n_agents, dtype=np.int64)
        if self.random_agent_permutation:
            self._rng.shuffle(order)
        return order

    def _sample_joint_action(self, obs: torch.Tensor, use_target: bool):
        B = obs.shape[0]
        joint = torch.zeros(B, self.n_agents, self.action_dim, device=self.device)
        log_probs: Dict[int, torch.Tensor] = {}
        actor_pool = self.target_actors if use_target else self.actors
        for i in range(self.n_agents):
            obs_i = obs[:, i]
            a, lp = actor_pool[i].sample(obs_i)
            joint[:, i] = a
            log_probs[i] = lp
        return joint.reshape(B, -1), log_probs

    @torch.no_grad()
    def _polyak(self, src: nn.Module, tgt: nn.Module) -> None:
        for p_src, p_tgt in zip(src.parameters(), tgt.parameters()):
            p_tgt.data.mul_(1.0 - self.tau)
            p_tgt.data.add_(self.tau * p_src.data)

    # ------------------------------------------------------------- public API

    def train(self, buffer) -> Dict[str, Any]:
        info: Dict[str, Any] = {"group_losses": {}}
        batch = buffer.sample(self.batch_size)

        info["critic_loss"] = self._update_critic(batch)

        B = batch["obs"].shape[0]
        log_M = torch.zeros(B, self.n_agents, device=self.device)

        for aid in self._agent_order():
            losses, log_M = self._update_agent(batch, int(aid), log_M)
            info["group_losses"][int(aid)] = losses

        # Polyak.
        self._polyak(self.critic, self.critic_target)
        for a, t in zip(self.actors, self.target_actors):
            self._polyak(a, t)

        info["num_groups"] = self.n_agents
        info["active_groups"] = self.n_agents
        info["alphas"] = [float(self.alpha(k).detach()) for k in range(self.n_agents)]
        return info

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            next_action, next_lp = self._sample_joint_action(batch["next_obs"], use_target=True)
            q1_t, q2_t = self.critic_target(batch["next_state"], next_action)
            q_t = torch.min(q1_t, q2_t)                         # (B, N)
            entropy_bonus = torch.zeros_like(q_t)
            for i in range(self.n_agents):
                entropy_bonus[:, i:i + 1] = self.alpha(i).detach() * next_lp[i]
            target = batch["rewards"] + self.gamma * (1 - batch["dones"]) * (q_t - entropy_bonus)

        joint_action = batch["actions"].reshape(batch["actions"].shape[0], -1)
        q1, q2 = self.critic(batch["state"], joint_action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optim.step()
        return float(loss.detach())

    def _update_agent(
        self,
        batch: Dict[str, torch.Tensor],
        agent_id: int,
        log_M: torch.Tensor,
    ):
        actor = self.actors[agent_id]
        actor_optim = self.actor_optims[agent_id]
        B = batch["obs"].shape[0]

        with torch.no_grad():
            joint_other_flat, _ = self._sample_joint_action(batch["obs"], use_target=False)
            joint_other = joint_other_flat.reshape(B, self.n_agents, self.action_dim)

        obs_i = batch["obs"][:, agent_id]
        a, lp = actor.sample(obs_i)
        # lp is (B, 1) from GaussianActor.sample.

        joint = joint_other.clone()
        joint[:, agent_id] = a
        joint_flat = joint.reshape(B, -1)

        q1, q2 = self.critic(batch["state"], joint_flat)
        q = torch.min(q1, q2)[:, agent_id:agent_id + 1]

        weight = log_M[:, agent_id:agent_id + 1].exp().detach()
        actor_loss = (self.alpha(agent_id).detach() * lp - weight * q).mean()

        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        # Auto-tuned alpha (per-agent log-prob compared to per-agent target).
        alpha_loss_val = 0.0
        if self.auto_alpha:
            alpha_loss = -(
                self.log_alphas[agent_id]
                * (lp.detach() + self.target_entropy)
            ).mean()
            self.alpha_optims[agent_id].zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optims[agent_id].step()
            with torch.no_grad():
                self.log_alphas[agent_id].clamp_(min=float(np.log(self.alpha_min)))
            alpha_loss_val = float(alpha_loss.detach())

        # log_M update via target actor for this agent.
        with torch.no_grad():
            tgt_actor = self.target_actors[agent_id]
            _, lp_target = tgt_actor.sample(obs_i)
            ratio_log = (lp.detach().squeeze(-1) - lp_target.squeeze(-1))
            log_M = log_M.clone()
            log_M[:, agent_id] = log_M[:, agent_id] + ratio_log

        return {
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": alpha_loss_val,
            "alpha":      float(self.alpha(agent_id).detach()),
        }, log_M

    # --------------------------------------------------------------- inference

    @torch.no_grad()
    def select_actions(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = (obs.dim() == 2)
        if squeeze:
            obs = obs.unsqueeze(0)
        n_threads, N = obs.shape[:2]
        actions = torch.zeros(n_threads, N, self.action_dim, device=self.device)
        for i in range(self.n_agents):
            obs_i = obs[:, i]
            a, _ = self.actors[i].sample(obs_i)
            actions[:, i] = a
        if squeeze:
            actions = actions.squeeze(0)
        return actions

    def set_groups(self, groups) -> None:
        return None

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "actors":         [a.state_dict() for a in self.actors],
            "target_actors":  [a.state_dict() for a in self.target_actors],
            "critic":         self.critic.state_dict(),
            "critic_target":  self.critic_target.state_dict(),
            "log_alphas":     [la.detach().cpu() for la in self.log_alphas],
        }

    def load_state_dict(self, sd: Dict) -> None:
        for a, w in zip(self.actors, sd["actors"]):
            a.load_state_dict(w)
        for a, w in zip(self.target_actors, sd["target_actors"]):
            a.load_state_dict(w)
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        for la, val in zip(self.log_alphas, sd["log_alphas"]):
            la.data = val.to(self.device)


__all__ = ["HASAC"]
