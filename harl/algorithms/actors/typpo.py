"""
TyPPO: Type-routed PPO with group-level sequential updates (ours).

TyPPO sits between two well-known extremes in cooperative MARL:

* MAPPO (Yu et al., 2022): one shared actor across all agents, all agents
  updated simultaneously. No explicit heterogeneity.
* HAPPO (Kuba et al., 2022): one actor per agent, agents updated
  *sequentially* with a cumulative ratio M that flows previous-agent
  policy changes forward. Full per-agent heterogeneity.

TyPPO combines the two ideas at *type-level* granularity:

1. Identify agent types up front (e.g., unit types in SMACv2). Agents of
   the same type share an actor (MAPPO-style parameter sharing
   *within-type*).
2. Update one type at a time in randomly-permuted order, and propagate a
   cumulative importance ratio M between type updates so the next type
   optimises against an advantage that already accounts for what the
   prior types did (HAPPO-style sequential update *across-types*).

Routing under stochastic team composition
-----------------------------------------
Under SMACv2, a single rollout buffer can span multiple episodes with
*different* team compositions. To make sure the actor that originally
generated a transition is the same actor that receives that transition's
gradient at training time, we use the per-step ``actor_id_per_step`` that
the runner stamps onto every (timestep, thread, agent) triple in the
buffer. ``_update_group(actor_idx, ...)`` masks the buffer to exactly
those triples whose ``actor_id_per_step == actor_idx``.

If the buffer has no per-step actor ids (e.g. synthetic test buffers),
the algorithm falls back to using the current ``self.groups`` partition
for the entire buffer — this is the legacy behaviour and is correct
whenever the team composition is constant for the duration of the
rollout.

Implementation notes
--------------------
- M is a *scalar* tensor of shape (T, B). Within a type, every agent
  sees the same M (it depends only on previously-updated *types*, not on
  position within the current type).
- Advantages are normalised *once*, over the entire iter, before the
  sequential loop starts. Normalising again inside the loop would cancel
  out the M weighting.
- Per-batch infrastructure (value clipping + Huber + value normalisation)
  is shared with MAPPO and HAPPO via ``_ppo_common``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.utils.networks import (
    CategoricalActor,
    GaussianActor,
    VCritic,
    ValueNormalizer,
)
from harl.algorithms.actors._ppo_common import (
    ppo_value_loss,
)
from harl.utils.group import (
    GroupAssignment,
    coerce_group_assignment,
)


GroupSpec = Union[GroupAssignment, Sequence[Sequence[int]], Sequence[str]]


class TyPPO:
    """Heterogeneous-Group Proximal Optimization."""

    def __init__(
        self,
        args: Dict[str, Any],
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        groups: GroupSpec,
        discrete: bool = True,
        device: torch.device | str = "cpu",
        n_canonical_groups: int | None = None,
        canonical_group_labels: List[str] | None = None,
    ):
        self.args = args
        self.discrete = discrete
        self.device = torch.device(device)

        self.groups = coerce_group_assignment(groups)
        self.num_agents = self.groups.n_agents

        # Dynamic-group setup (SMACv2-style stochastic team composition).
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
        actor_cls = CategoricalActor if discrete else GaussianActor
        self.actors = nn.ModuleList(
            [actor_cls(obs_dim, action_dim).to(self.device) for _ in range(self.num_groups)]
        )
        self.critic = VCritic(state_dim).to(self.device)
        self.value_normalizer = ValueNormalizer()

        # ---------- Hyperparameters (HARL-style nested dict) ---------------
        ppo = args["ppo"]
        opt = args["optim"]
        seq = args.get("sequential_update", {})

        self.actor_optims = [
            torch.optim.Adam(a.parameters(), lr=opt["actor_lr"], eps=opt.get("optim_eps", 1e-5))
            for a in self.actors
        ]
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=opt["critic_lr"], eps=opt.get("optim_eps", 1e-5)
        )

        self.clip_param: float = float(ppo["clip_param"])
        self.ppo_epoch: int = int(ppo["ppo_epoch"])
        self.entropy_coef: float = float(ppo["entropy_coef"])
        self.value_loss_coef: float = float(ppo["value_loss_coef"])
        self.max_grad_norm: float = float(ppo["max_grad_norm"])
        self.huber_delta: float = float(ppo.get("huber_delta", 10.0))
        self.use_huber: bool = bool(ppo.get("use_huber_loss", True))
        self.use_clipped_value_loss: bool = bool(ppo.get("use_clipped_value_loss", True))
        self.use_value_norm: bool = bool(ppo.get("use_value_norm", True))
        self.num_mini_batch: int = int(ppo.get("num_mini_batch", 1))
        self.mini_batch: int = int(ppo.get("mini_batch", 0))   # 0 == use num_mini_batch
        self.target_kl: float = float(ppo.get("target_kl", 0.06))

        self.random_group_permutation: bool = bool(
            seq.get("random_group_permutation", True)
        )

        self._rng = np.random.default_rng(int(args.get("seed", 0)))

    # ------------------------------------------------------------------ utils

    def set_groups(self, groups: GroupSpec) -> None:
        """Update the agent-to-group mapping for the next rollout/episode."""
        new_ga = coerce_group_assignment(groups)
        if not self.dynamic_groups:
            if new_ga.n_groups != self.num_groups:
                raise ValueError(
                    f"set_groups in static mode requires same n_groups "
                    f"({self.num_groups}); got {new_ga.n_groups}. "
                    f"Use n_canonical_groups=... at construction for dynamic teams."
                )
        else:
            for lbl in new_ga.group_labels:
                if lbl not in self.canonical_group_labels:
                    raise ValueError(
                        f"Group label '{lbl}' not in canonical labels "
                        f"{self.canonical_group_labels}; cannot route."
                    )
        self.groups = new_ga
        self.num_agents = new_ga.n_agents

    def _active_actor_indices(self) -> List[int]:
        """Indices into ``self.actors`` that should run this iteration.

        With canonical-order group assignment, ``self.groups.group_labels``
        always lists *all* canonical types -- even those absent from the
        current episode (which therefore have empty member lists). We
        explicitly skip those: an actor with zero current agents has no
        on-policy data to learn from this episode.
        """
        if not self.dynamic_groups:
            return list(range(self.num_groups))
        active: List[int] = []
        for local_g, lbl in enumerate(self.groups.group_labels):
            if len(self.groups.agents_of_group[local_g]) == 0:
                continue  # type absent this episode -> skip
            active.append(self.canonical_group_labels.index(lbl))
        return active

    def _agents_of_actor(self, actor_idx: int) -> List[int]:
        """Agent ids handled by the actor at ``actor_idx`` *in self.groups*.

        Used as the legacy fallback when the buffer has no per-step actor
        ids. The correct (per-step) routing is in ``_update_group``.
        """
        if not self.dynamic_groups:
            return list(map(int, self.groups.agents_of_group[actor_idx]))
        target_label = self.canonical_group_labels[actor_idx]
        for local_g, lbl in enumerate(self.groups.group_labels):
            if lbl == target_label:
                return list(map(int, self.groups.agents_of_group[local_g]))
        return []

    def _group_order(self) -> np.ndarray:
        """Order in which to update actors. Includes ALL canonical actors,
        not just those active this episode, because the buffer may have
        transitions from earlier episodes where they were active."""
        if self.dynamic_groups:
            order = np.arange(self.num_groups, dtype=np.int64)
        else:
            order = np.asarray(self._active_actor_indices(), dtype=np.int64)
        if self.random_group_permutation:
            self._rng.shuffle(order)
        return order

    def _entropy_and_logprob(self, dist, actions: torch.Tensor):
        if self.discrete:
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)
        return log_prob, entropy

    # ---------------------------------------------------------- public API

    def train(self, buffer, advantages: np.ndarray) -> Dict[str, Any]:
        """One pass over the on-policy buffer with group-level sequential update."""
        info: Dict[str, Any] = {"group_losses": {}}

        # 1) Pre-compute normalised advantages (once, before the seq loop).
        adv_t = torch.as_tensor(advantages, device=self.device).float()    # (T, B)
        adv_mean = adv_t.mean()
        adv_std = adv_t.std() + 1e-8
        adv_t = (adv_t - adv_mean) / adv_std

        # 2) Initialise scalar cumulative ratio M(t,b).
        T_, B_ = adv_t.shape
        M = torch.ones(T_, B_, device=self.device)

        # 3) Update value normaliser on raw returns.
        if self.use_value_norm:
            self.value_normalizer.update(buffer.returns)

        # 4) Sequential per-actor PPO ACTOR update.
        for actor_idx in self._group_order():
            agent_ids: List[int] = self._agents_of_actor(actor_idx)
            # In dynamic mode, even if no agent of this type spawned in the
            # *current* GroupAssignment, the buffer may still have data from
            # a previous episode where one did. Skip only when there is
            # truly no data routed to this actor.
            has_data = self._actor_has_data_in_buffer(buffer, actor_idx, agent_ids)
            if not has_data:
                continue
            losses = self._update_group_actor(buffer, M, adv_t, actor_idx, agent_ids)
            info["group_losses"][int(actor_idx)] = losses
            # After actor a_m has moved, multiply M by its joint ratio.
            M = self._update_M_with_joint_ratio(buffer, M, actor_idx, agent_ids)

        # 5) Centralised critic update, ONCE per iteration after the seq
        #    actor loop. This avoids the N_groups x ppo_epoch over-training
        #    bug present in earlier versions where the critic was stepped
        #    inside the per-group loop.
        critic_info = self._update_critic(buffer)
        for gid in info["group_losses"]:
            info["group_losses"][gid]["value_loss"] = critic_info["value_loss"]
        info["critic_loss"] = critic_info["value_loss"]
        info["critic_n_steps"] = critic_info["n_steps"]

        info["num_groups"] = self.num_groups
        info["active_groups"] = len(info["group_losses"])
        info["group_sizes"] = self.groups.group_sizes
        return info

    # ----------------------------------------------------------- internals

    def _actor_has_data_in_buffer(
        self, buffer, actor_idx: int, agent_ids: Sequence[int]
    ) -> bool:
        """Return True if any (t, b, a) in the buffer was generated by ``actor_idx``."""
        if buffer.actor_id_per_step is not None:
            return bool((buffer.actor_id_per_step == actor_idx).any())
        # Legacy path: rely on the current ``self.groups``.
        return len(agent_ids) > 0

    def _update_group_actor(
        self,
        buffer,
        M: torch.Tensor,                 # (T, B) scalar cumulative ratio
        adv: torch.Tensor,               # (T, B) normalised advantages
        actor_idx: int,
        agent_ids: Sequence[int],
    ) -> Dict[str, float]:
        """PPO ACTOR update for one group, weighted by the cumulative scalar M.

        Routes by ``actor_id_per_step`` when the buffer has that field,
        otherwise falls back to ``agent_ids`` from ``self.groups``.

        The critic is NOT updated here; it is updated once per iteration in
        :meth:`_update_critic` to avoid the per-group critic over-training
        that destabilises value targets.
        """
        actor = self.actors[actor_idx]
        actor_optim = self.actor_optims[actor_idx]
        clip = self.clip_param

        policy_loss_acc = entropy_acc = 0.0
        approx_kl_acc = 0.0
        n_steps = 0
        early_stop = False

        # Compute mini-batch size.
        buffer_size = buffer.size_per_agent()
        if self.mini_batch > 0:
            mb_size = self.mini_batch
        else:
            mb_size = max(1, buffer_size // max(1, self.num_mini_batch))

        # Choose iteration agent_ids: when per-step actor ids are present,
        # we still need to pull SOME agents from the buffer to construct
        # batches; we pull ALL agents and filter inside the loop. When
        # per-step ids are absent, we pull only the static agent_ids.
        per_step_routing = buffer.actor_id_per_step is not None
        iter_agent_ids = list(range(buffer.n_agents)) if per_step_routing else list(agent_ids)
        if not iter_agent_ids:
            return {
                "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "early_stop": 0.0, "n_steps": 0,
            }

        for _ in range(self.ppo_epoch):
            for batch in buffer.get(iter_agent_ids, batch_size=mb_size, M=M, adv=adv):
                obs = batch["obs"]                  # (B_mb, |ids|, O)
                actions = batch["actions"]          # (B_mb, |ids|, 1) discrete
                old_lp = batch["log_probs"]         # (B_mb, |ids|)
                ret = batch["returns"]              # (B_mb,)
                state = batch["state"]              # (B_mb, S)
                old_values = batch["values"]        # (B_mb,)
                M_b = batch["M"]                    # (B_mb,) scalar cumulative ratio
                adv_b = batch["adv"]                # (B_mb,) normalised advantage

                avail = batch.get("available_actions")     # (B_mb, |ids|, n_actions)
                active = batch.get("active_masks")         # (B_mb, |ids|)
                aid    = batch.get("actor_id_per_step")    # (B_mb, |ids|) int

                B, G_ = obs.shape[:2]
                obs_flat = obs.reshape(B * G_, -1)
                if self.discrete:
                    act_flat = actions.reshape(B * G_, -1).squeeze(-1).long()
                else:
                    act_flat = actions.reshape(B * G_, -1)
                old_lp_flat = old_lp.reshape(B * G_)
                avail_flat = (
                    avail.reshape(B * G_, -1) if avail is not None and self.discrete else None
                )

                # ---- Sample mask: which (b, j) are routed to this actor? ----
                # Three contributions:
                #   (a) routed to this actor via per-step id (or agent membership)
                #   (b) the agent was alive (active_mask=1)
                # The PPO loss is computed only over samples where mask=True.
                if aid is not None:
                    routing_mask = (aid.long() == actor_idx)                       # (B, G_)
                else:
                    # Legacy: every agent in iter_agent_ids belongs to this actor.
                    routing_mask = torch.ones(B, G_, dtype=torch.bool, device=obs.device)
                if active is not None:
                    routing_mask = routing_mask & (active > 0.5)
                routing_flat = routing_mask.reshape(B * G_)
                n_eff = int(routing_flat.sum().item())
                if n_eff == 0:
                    continue

                # ---- Forward pass ----
                if self.discrete and avail_flat is not None:
                    dist = actor(obs_flat, available_actions=avail_flat)
                else:
                    dist = actor(obs_flat)
                new_lp, entropy = self._entropy_and_logprob(dist, act_flat)

                # Apply mask: zero-out contributions from non-routed / dead samples.
                mask_f = routing_flat.float()
                # Entropy: mean over routed/active samples only.
                entropy_loss = (entropy * mask_f).sum() / max(n_eff, 1)

                # *** Core TyPPO operation: A(s,a) * M(t,b) ***
                # adv_b has shape (B,); M_b has shape (B,). Broadcast both
                # across the agent-in-batch dimension.
                weighted_adv = (adv_b * M_b).unsqueeze(-1).expand(B, G_).reshape(B * G_)

                ratio = (new_lp - old_lp_flat).exp()
                surr1 = ratio * weighted_adv
                surr2 = ratio.clamp(1 - clip, 1 + clip) * weighted_adv
                # Sample-wise PPO loss, masked, mean over routed samples only.
                pl_per_sample = -torch.min(surr1, surr2)
                policy_loss = (pl_per_sample * mask_f).sum() / max(n_eff, 1)

                # ---- Backward + optimiser step (ACTOR ONLY) ----
                actor_optim.zero_grad(set_to_none=True)
                (policy_loss - self.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                actor_optim.step()

                with torch.no_grad():
                    diff = old_lp_flat - new_lp
                    approx_kl = (diff * mask_f).sum().item() / max(n_eff, 1)

                policy_loss_acc += float(policy_loss.detach())
                entropy_acc += float(entropy_loss.detach())
                approx_kl_acc += approx_kl
                n_steps += 1

                if approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break

            if early_stop:
                break

        n = max(n_steps, 1)
        return {
            "policy_loss": policy_loss_acc / n,
            "value_loss":  0.0,        # filled in by _update_critic in train()
            "entropy":     entropy_acc / n,
            "approx_kl":   approx_kl_acc / n,
            "early_stop":  float(early_stop),
            "n_steps":     n_steps,
        }

    def _update_critic(self, buffer) -> Dict[str, float]:
        """Centralised V-critic update, run ONCE per iteration after the
        sequential actor updates.

        Iterates ``ppo_epoch`` epochs over the buffer. The buffer's
        ``get([0], ...)`` interface is reused only to pull
        state/return/old_value tensors -- the per-agent slice is irrelevant
        for the critic, which only depends on the global state.
        """
        critic = self.critic
        critic_optim = self.critic_optim

        buffer_size = buffer.size_per_agent()
        if self.mini_batch > 0:
            mb_size = self.mini_batch
        else:
            mb_size = max(1, buffer_size // max(1, self.num_mini_batch))

        value_loss_acc = 0.0
        n_steps = 0
        ph_agent = [0]   # placeholder slice; we only use state/returns/values

        for _ in range(self.ppo_epoch):
            for batch in buffer.get(ph_agent, batch_size=mb_size):
                ret        = batch["returns"]        # (B,)
                state      = batch["state"]          # (B, S)
                old_values = batch["values"]         # (B,)

                values_raw = critic(state).squeeze(-1)
                if self.use_value_norm:
                    target = self.value_normalizer.normalize(ret).float()
                    old_pred = self.value_normalizer.normalize(old_values).float()
                else:
                    target = ret
                    old_pred = old_values

                value_loss = ppo_value_loss(
                    pred=values_raw, target=target, old_pred=old_pred,
                    clip_param=self.clip_param,
                    use_clipped_value_loss=self.use_clipped_value_loss,
                    use_huber=self.use_huber, huber_delta=self.huber_delta,
                )

                critic_optim.zero_grad(set_to_none=True)
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
                critic_optim.step()

                value_loss_acc += float(value_loss.detach())
                n_steps += 1

        n = max(n_steps, 1)
        return {"value_loss": value_loss_acc / n, "n_steps": n_steps}

    @torch.no_grad()
    def _update_M_with_joint_ratio(
        self,
        buffer,
        M: torch.Tensor,
        actor_idx: int,
        agent_ids: Sequence[int],
    ) -> torch.Tensor:
        """Multiply the scalar M(t,b) by the joint policy ratio of the just-updated actor.

        joint_ratio[t, b] = ∏_{(t,b,j) routed to actor_idx} π_new(a) / π_old(a)
                          = exp( ∑_{(t,b,j) routed} (new_lp - old_lp) )

        With per-step actor ids in the buffer, routing is exact even when
        the buffer spans multiple episodes with different team compositions.
        Without per-step ids, falls back to the static ``agent_ids``.
        """
        actor = self.actors[actor_idx]
        T_, B_ = M.shape

        if buffer.actor_id_per_step is not None:
            # Per-(t, b, agent) routing mask.
            mask_np = (buffer.actor_id_per_step == actor_idx)        # (T, B, N) bool
            if buffer.active_masks is not None:
                mask_np = mask_np & (buffer.active_masks > 0.5)

            obs_all = torch.as_tensor(buffer.obs, device=self.device).float()    # (T,B,N,O)
            act_all = torch.as_tensor(buffer.actions, device=self.device).float()
            lp_old_all = torch.as_tensor(buffer.log_probs, device=self.device).float()
            mask_t = torch.as_tensor(mask_np, device=self.device)
            avail_all = (
                torch.as_tensor(buffer.available_actions, device=self.device).float()
                if buffer.available_actions is not None else None
            )

            T, Bb, N = obs_all.shape[:3]
            obs_flat = obs_all.reshape(T * Bb * N, -1)
            if self.discrete:
                act_flat = act_all.reshape(T * Bb * N, -1).squeeze(-1).long()
            else:
                act_flat = act_all.reshape(T * Bb * N, -1)
            avail_flat = (
                avail_all.reshape(T * Bb * N, -1) if (avail_all is not None and self.discrete) else None
            )
            mask_flat = mask_t.reshape(T * Bb * N).float()

            if self.discrete and avail_flat is not None:
                dist = actor(obs_flat, available_actions=avail_flat)
            else:
                dist = actor(obs_flat)
            new_lp, _ = self._entropy_and_logprob(dist, act_flat)
            log_ratio = (new_lp - lp_old_all.reshape(T * Bb * N))

            # Sum log-ratios per (t, b), counting only routed/active samples.
            log_ratio = log_ratio * mask_flat
            log_ratio = log_ratio.reshape(T, Bb, N).sum(dim=-1)               # (T, B)
            log_ratio = log_ratio.clamp(-10.0, 10.0)
            return M * log_ratio.exp()

        # ---------- Legacy path: static agent_ids ----------
        if not agent_ids:
            return M
        n_g = len(agent_ids)
        flat_obs = buffer.obs[..., agent_ids, :].reshape(T_ * B_ * n_g, -1)
        flat_act = buffer.actions[..., agent_ids, :].reshape(T_ * B_ * n_g, -1)
        flat_lp_old = buffer.log_probs[..., agent_ids].reshape(T_ * B_ * n_g)
        flat_avail = None
        if buffer.available_actions is not None and self.discrete:
            flat_avail = torch.as_tensor(
                buffer.available_actions[..., agent_ids, :].reshape(T_ * B_ * n_g, -1),
                device=self.device,
            ).float()

        obs_t = torch.as_tensor(flat_obs, device=self.device).float()
        if self.discrete:
            act_t = torch.as_tensor(flat_act, device=self.device).squeeze(-1).long()
        else:
            act_t = torch.as_tensor(flat_act, device=self.device).float()
        lp_old_t = torch.as_tensor(flat_lp_old, device=self.device).float()

        if self.discrete and flat_avail is not None:
            dist = actor(obs_t, available_actions=flat_avail)
        else:
            dist = actor(obs_t)
        new_lp, _ = self._entropy_and_logprob(dist, act_t)
        log_ratio_per_agent = (new_lp - lp_old_t).reshape(T_, B_, n_g)
        joint_log_ratio = log_ratio_per_agent.sum(dim=-1).clamp(-10.0, 10.0)   # (T, B)
        return M * joint_log_ratio.exp()

    # ------------------------------------------------------------------
    # Inference (used by the runner during rollouts)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, available_actions: torch.Tensor | None = None
    ):
        """Sample joint action."""
        squeeze_threads = (obs.dim() == 2)
        if squeeze_threads:
            obs = obs.unsqueeze(0)
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0)
        n_threads = obs.shape[0]
        N = obs.shape[1]

        if self.discrete:
            actions = torch.zeros(n_threads, N, dtype=torch.long, device=self.device)
        else:
            actions = torch.zeros(n_threads, N, self.actors[0].mu.out_features,
                                  device=self.device)
        log_probs = torch.zeros(n_threads, N, device=self.device)

        for actor_idx in self._active_actor_indices():
            ids = self._agents_of_actor(actor_idx)
            if not ids:
                continue
            obs_k = obs[:, ids].reshape(n_threads * len(ids), -1)
            if self.discrete:
                avail_k = (available_actions[:, ids].reshape(n_threads * len(ids), -1)
                           if available_actions is not None else None)
                a, lp = self.actors[actor_idx].act(obs_k, avail_k)
                a = a.view(n_threads, len(ids))
                lp = lp.view(n_threads, len(ids))
                actions[:, ids] = a
            else:
                dist = self.actors[actor_idx](obs_k)
                a = dist.rsample()
                lp = dist.log_prob(a).sum(-1)
                a = a.view(n_threads, len(ids), -1)
                lp = lp.view(n_threads, len(ids))
                actions[:, ids] = a
            log_probs[:, ids] = lp

        if squeeze_threads:
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)
        return actions, log_probs

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        """Critic forward, denormalised. Used by the runner for GAE bootstrap."""
        v_norm = self.critic(state).squeeze(-1)
        if self.use_value_norm:
            return self.value_normalizer.denormalize(v_norm)
        return v_norm

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "actors": [a.state_dict() for a in self.actors],
            "critic": self.critic.state_dict(),
            "value_norm": {
                "mean": float(self.value_normalizer.mean),
                "var": float(self.value_normalizer.var),
                "count": float(self.value_normalizer.count),
            },
        }

    def load_state_dict(self, sd: Dict) -> None:
        for a, w in zip(self.actors, sd["actors"]):
            a.load_state_dict(w)
        self.critic.load_state_dict(sd["critic"])
        if "value_norm" in sd:
            self.value_normalizer.mean = sd["value_norm"]["mean"]
            self.value_normalizer.var = sd["value_norm"]["var"]
            self.value_normalizer.count = sd["value_norm"]["count"]
