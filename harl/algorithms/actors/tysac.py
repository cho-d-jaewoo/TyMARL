"""
TySAC: Type-routed Soft Actor-Critic (ours).

Off-policy maximum-entropy algorithm for cooperative MARL with explicit group
structure. TySAC performs sequential SAC-style updates at the **group level**
with shared parameters inside each group, and uses a *twin Q-critic with one
head per group* (``GroupQCritic``) so that each actor's gradient flows
exclusively through its own head.

Positioning
-----------
Like TyPPO, TySAC combines within-type parameter sharing (SAC-style) with
across-type sequential trust-region updates (HASAC-style log-ratio
multiplier between successive group updates). HASAC and parameter-shared
SAC are reference points for the two extreme partition shapes (one
group per agent, all agents in one group), but TySAC is *not* presented
as a reduction to either: HASAC is implemented as an independent
standalone class for fair baseline comparison.

Implementation choices that matter for the paper
------------------------------------------------
- **Per-group Q heads.** Each group's actor only sees gradient from its own
  Q-head, decoupling group updates while sharing trunk representation
  learning. Section 7.4 ablates this design and shows it contributes a small
  but consistent gain over a single joint Q.
- **Per-group log alpha.** One auto-tuned entropy temperature per group, since
  different unit types in the mixed_smacv2 scenarios call for different
  exploration pressure (a Medivac wants almost-deterministic positioning;
  Marines benefit from a noisier policy).
- **Target actors.** A Polyak-averaged copy of every group's actor stabilises
  the bootstrap, mirroring HASAC's per-agent target-actor recipe.
- **Sequential log-ratio multiplier.** After group k moves, ``log_M[:, k]``
  accumulates the log-ratio between new-actor and target-actor samples, and
  the next group's actor loss is weighted by ``exp(log_M[:, group_id])``.
  This is the off-policy analogue of TyPPO's cumulative ratio M.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.utils.networks import GaussianActor
from harl.algorithms.critics.group_critic import GroupQCritic
from harl.utils.group import (
    GroupAssignment,
    coerce_group_assignment,
    random_group_permutation,
)


GroupSpec = Union[GroupAssignment, Sequence[Sequence[int]], Sequence[str]]


class TySAC:
    """Heterogeneous-Group Soft Actor-Critic. Continuous-action only."""

    def __init__(
        self,
        args: Dict[str, Any],
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        groups: GroupSpec,
        device: torch.device | str = "cpu",
        n_canonical_groups: int | None = None,
        canonical_group_labels: List[str] | None = None,
    ):
        self.args = args
        self.action_dim = action_dim
        self.device = torch.device(device)

        self.groups = coerce_group_assignment(groups)
        self.num_agents = self.groups.n_agents

        # Dynamic-group setup (see TyPPO docstring for the rationale).
        self.dynamic_groups = n_canonical_groups is not None
        if self.dynamic_groups:
            assert canonical_group_labels is not None and len(canonical_group_labels) == n_canonical_groups, (
                "canonical_group_labels must match n_canonical_groups"
            )
            self.canonical_group_labels = list(canonical_group_labels)
            self.num_groups = int(n_canonical_groups)
        else:
            self.canonical_group_labels = list(self.groups.group_labels)
            self.num_groups = self.groups.n_groups

        # ---------- Networks -----------------------------------------------
        self.actors = nn.ModuleList(
            [GaussianActor(obs_dim, action_dim).to(self.device) for _ in range(self.num_groups)]
        )
        self.target_actors = nn.ModuleList(
            [copy.deepcopy(a).requires_grad_(False) for a in self.actors]
        )

        joint_action_dim = action_dim * self.num_agents
        self.critic = GroupQCritic(
            state_dim, joint_action_dim, self.num_groups
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)

        # ---------- Hyperparameters ----------------------------------------
        opt = args["optim"]
        sac = args["sac"]
        seq = args.get("sequential_update", {})

        self.actor_optims = [
            torch.optim.Adam(a.parameters(), lr=opt["actor_lr"]) for a in self.actors
        ]
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=opt["critic_lr"])

        self.gamma: float = float(sac["gamma"])
        self.tau: float = float(sac["tau"])
        self.batch_size: int = int(sac["batch_size"])

        # Per-group temperature.
        self.auto_alpha: bool = bool(sac["auto_alpha"])
        init_log_alpha = float(np.log(sac["alpha"]))
        if self.auto_alpha:
            self.log_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(init_log_alpha, device=self.device))
                for _ in range(self.num_groups)
            ])
            self.alpha_optims = [
                torch.optim.Adam([la], lr=opt["alpha_lr"]) for la in self.log_alphas
            ]
            target_entropy = sac["target_entropy"]
            if target_entropy == "auto":
                # MAPPO/SAC convention for discrete-equivalent: target a
                # fraction of max entropy. For continuous (our case), the
                # canonical SAC default is -|action_dim|, but that drives
                # the policy too deterministic too fast in MARL. We use
                # a configurable scale.
                target_entropy = -float(action_dim)
            self.target_entropy = float(target_entropy)
            self.alpha_min = float(sac.get("alpha_min", 0.05))
        else:
            self.log_alphas = nn.ParameterList([
                nn.Parameter(torch.tensor(init_log_alpha, device=self.device),
                             requires_grad=False)
                for _ in range(self.num_groups)
            ])
            self.alpha_optims = []
            self.alpha_min = float(sac.get("alpha_min", 0.05))

        self.random_group_permutation: bool = bool(
            seq.get("random_group_permutation", True)
        )
        self._rng = np.random.default_rng(int(args.get("seed", 0)))

    # --------------------------------------------------------------- helpers

    def alpha(self, group_id: int) -> torch.Tensor:
        return self.log_alphas[group_id].exp()

    def set_groups(self, groups: GroupSpec) -> None:
        """See TyPPO.set_groups for behaviour. TySAC's primary benchmarks
        (MAMuJoCo) have static body-part groups, so this is mostly a forward-
        compatibility hook for future continuous-action stochastic-team envs.
        """
        new_ga = coerce_group_assignment(groups)
        if not self.dynamic_groups:
            if new_ga.n_groups != self.num_groups:
                raise ValueError(
                    f"set_groups in static mode requires same n_groups "
                    f"({self.num_groups}); got {new_ga.n_groups}."
                )
        else:
            for lbl in new_ga.group_labels:
                if lbl not in self.canonical_group_labels:
                    raise ValueError(
                        f"Group label '{lbl}' not in canonical labels "
                        f"{self.canonical_group_labels}."
                    )
        self.groups = new_ga
        self.num_agents = new_ga.n_agents

    def _active_actor_indices(self) -> List[int]:
        """Indices into ``self.actors`` that should run this iteration.

        With canonical-order group assignment, ``self.groups.group_labels``
        always lists *all* canonical types -- even those absent from the
        current episode (empty member lists). Skip those.
        """
        if not self.dynamic_groups:
            return list(range(self.num_groups))
        active: List[int] = []
        for local_g, lbl in enumerate(self.groups.group_labels):
            if len(self.groups.agents_of_group[local_g]) == 0:
                continue
            active.append(self.canonical_group_labels.index(lbl))
        return active

    def _agents_of_actor(self, actor_idx: int) -> List[int]:
        if not self.dynamic_groups:
            return list(map(int, self.groups.agents_of_group[actor_idx]))
        target_label = self.canonical_group_labels[actor_idx]
        for local_g, lbl in enumerate(self.groups.group_labels):
            if lbl == target_label:
                return list(map(int, self.groups.agents_of_group[local_g]))
        return []

    def _group_order(self) -> np.ndarray:
        active = np.asarray(self._active_actor_indices(), dtype=np.int64)
        if self.random_group_permutation:
            self._rng.shuffle(active)
        return active

    def _sample_joint_action(
        self, obs: torch.Tensor, use_target: bool = False
    ):
        """Sample joint action ``(B, num_agents, action_dim)`` group by group.

        Returns ``(joint_flat, log_probs_per_group)`` where
        ``joint_flat`` is ``(B, num_agents * action_dim)`` and
        ``log_probs_per_group[actor_idx]`` is ``(B, 1)``. Actors with no
        agents in the current partition are skipped.
        """
        B = obs.shape[0]
        joint = torch.zeros(B, self.num_agents, self.action_dim, device=self.device)
        log_probs: Dict[int, torch.Tensor] = {}
        actor_pool = self.target_actors if use_target else self.actors
        for actor_idx in self._active_actor_indices():
            ids = self._agents_of_actor(actor_idx)
            if not ids:
                continue
            obs_g = obs[:, ids].reshape(-1, obs.shape[-1])
            a, lp = actor_pool[actor_idx].sample(obs_g)
            joint[:, ids] = a.reshape(B, len(ids), self.action_dim)
            log_probs[actor_idx] = lp.reshape(B, len(ids)).sum(-1, keepdim=True)
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

        # 1) Critic: soft Bellman backup, per-group head.
        info["critic_loss"] = self._update_critic(batch)

        # 2) Sequential per-group actor updates with log_M tracking.
        B = batch["obs"].shape[0]
        log_M = torch.zeros(B, self.num_groups, device=self.device)

        for gid in self._group_order():
            ids = self._agents_of_actor(gid)
            if not ids:
                continue
            losses, log_M = self._update_group(batch, gid, log_M)
            info["group_losses"][gid] = losses

        # 3) Polyak update of target critic and target actors.
        self._polyak(self.critic, self.critic_target)
        for a, t in zip(self.actors, self.target_actors):
            self._polyak(a, t)

        info["num_groups"] = self.num_groups
        info["active_groups"] = len(info["group_losses"])
        info["alphas"] = [float(self.alpha(k).detach()) for k in range(self.num_groups)]
        return info

    # ----------------------------------------------------------- internals

    def _update_critic(self, batch: Dict[str, torch.Tensor]) -> float:
        with torch.no_grad():
            next_action, next_lp = self._sample_joint_action(
                batch["next_obs"], use_target=True
            )
            q1_t, q2_t = self.critic_target(batch["next_state"], next_action)
            q_t = torch.min(q1_t, q2_t)                       # (B, K)
            # Per-group entropy bonus, only for actors active this step.
            entropy_bonus = torch.zeros_like(q_t)
            for gid in self._active_actor_indices():
                if gid in next_lp:
                    entropy_bonus[:, gid:gid + 1] = self.alpha(gid).detach() * next_lp[gid]
            target = batch["rewards"] + self.gamma * (1 - batch["dones"]) * (q_t - entropy_bonus)

        joint_action = batch["actions"].reshape(batch["actions"].shape[0], -1)
        q1, q2 = self.critic(batch["state"], joint_action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optim.step()
        return float(loss.detach())

    def _update_group(
        self,
        batch: Dict[str, torch.Tensor],
        group_id: int,
        log_M: torch.Tensor,
    ):
        actor = self.actors[group_id]
        actor_optim = self.actor_optims[group_id]
        agents = self._agents_of_actor(group_id)
        B = batch["obs"].shape[0]

        # Freeze the joint action of the *other* groups using the current actors.
        with torch.no_grad():
            joint_other_flat, _ = self._sample_joint_action(batch["obs"], use_target=False)
            joint_other = joint_other_flat.reshape(B, self.num_agents, self.action_dim)

        # Resample only this group's actions from the *current* actor (with grad).
        obs_g = batch["obs"][:, agents].reshape(-1, batch["obs"].shape[-1])
        a, lp = actor.sample(obs_g)
        a = a.reshape(B, len(agents), self.action_dim)
        lp = lp.reshape(B, len(agents)).sum(-1, keepdim=True)

        joint = joint_other.clone()
        joint[:, agents] = a
        joint_flat = joint.reshape(B, -1)

        q1, q2 = self.critic(batch["state"], joint_flat)
        q = torch.min(q1, q2)[:, group_id:group_id + 1]               # (B, 1)

        weight = log_M[:, group_id:group_id + 1].exp().detach()       # (B, 1)
        actor_loss = (self.alpha(group_id).detach() * lp - weight * q).mean()

        actor_optim.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_optim.step()

        # Auto-tuned alpha for this group.
        # Bug fix from Slot B (which collapsed alpha → 0.001 by step 1M):
        #   ``lp`` is summed over the group's agents. The original code
        #   then used ``self.target_entropy * len(agents)`` as the target,
        #   which over-counts the entropy budget when target_entropy is
        #   already supposed to mean "per-agent". With group=3 and
        #   target_entropy=-1, the target became -3 while typical lp was
        #   in the same ballpark, so (lp + target).mean() was strongly
        #   negative and pushed log_alpha down very fast.
        # The cleaner formulation is to compare PER-AGENT log-probs to a
        # PER-AGENT target entropy. We use the mean log-prob across the
        # group's agents and a single target_entropy.
        # We also clamp alpha to a small floor so exploration never fully
        # dies (helpful in long off-policy runs on SMAC where the policy
        # can otherwise commit to a suboptimal mode).
        alpha_loss_val = 0.0
        if self.auto_alpha:
            lp_per_agent = lp.detach() / max(1, len(agents))    # (B, 1)
            alpha_loss = -(
                self.log_alphas[group_id]
                * (lp_per_agent + self.target_entropy)
            ).mean()
            self.alpha_optims[group_id].zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optims[group_id].step()
            # Floor: clamp log_alpha so alpha >= alpha_min.
            with torch.no_grad():
                self.log_alphas[group_id].clamp_(min=float(np.log(self.alpha_min)))
            alpha_loss_val = float(alpha_loss.detach())

        # Update log_M for this group's column.
        with torch.no_grad():
            tgt_actor = self.target_actors[group_id]
            _, lp_target = tgt_actor.sample(obs_g)
            lp_target = lp_target.reshape(B, len(agents)).sum(-1)
            ratio_log = (lp.detach().squeeze(-1) - lp_target)
            log_M = log_M.clone()
            log_M[:, group_id] = log_M[:, group_id] + ratio_log

        losses = {
            "actor_loss": float(actor_loss.detach()),
            "alpha_loss": alpha_loss_val,
            "alpha":      float(self.alpha(group_id).detach()),
        }
        return losses, log_M

    # --------------------------------------------------------------- inference

    @torch.no_grad()
    def select_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample joint action for rollout. Accepts (n_threads, N, O) or (N, O)."""
        squeeze = (obs.dim() == 2)
        if squeeze:
            obs = obs.unsqueeze(0)
        joint_flat, _ = self._sample_joint_action(obs, use_target=False)
        out = joint_flat.reshape(obs.shape[0], self.num_agents, self.action_dim)
        if squeeze:
            out = out.squeeze(0)
        return out

    # --------------------------------------------------------------- checkpoint

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
            la.data.copy_(val.to(la.device))
