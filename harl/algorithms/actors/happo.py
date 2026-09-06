"""
HAPPO -- Heterogeneous-Agent Proximal Policy Optimisation
       (Kuba et al., ICLR 2022, Algorithm 3).

Standalone implementation following the original paper's pseudocode. Sequential
per-agent update with cumulative ratio M:

    M_{i_1}(s, a)        = 1
    update agent i_m using objective         clip[ratio_{i_m}] * M_{i_{1:m}} * A(s,a)
    M_{i_{1:m+1}}(s, a)  = ratio_{i_m}(s, a) * M_{i_{1:m}}(s, a)

Architectural notes:
* One actor *network per agent* -- no parameter sharing.
* Centralised V-critic on the global state (CTDE, shared across agents).
* Shared per-batch infrastructure with MAPPO and TyPPO via ``_ppo_common``.

Critic update strategy (FIX 2026-05)
------------------------------------
The critic is updated in its OWN ppo_epoch loop AFTER all sequential actor
updates are finished. Earlier code stepped the critic optimiser inside the
per-agent actor loop, which (with N agents and ppo_epoch e=5) trained the
critic N*e=25 times per iteration vs MAPPO's e=5. That over-training caused
noisy value estimates that fed back into the next iteration's GAE -- a
likely contributor to HAPPO under-performing MAPPO on SMACv2 in our logs.
Decoupling the critic loop fixes this without changing the actor objective.

SMAC-family integration
-----------------------
``available_actions`` and ``active_masks`` are honoured when the buffer
provides them; dead agents are excluded from the loss and invalid actions
are masked out at both ratio and M-update time.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from harl.utils.networks import (
    CategoricalActor,
    GaussianActor,
    VCritic,
    ValueNormalizer,
)
from harl.algorithms.actors._ppo_common import (
    normalise_advantages,
    ppo_value_loss,
    hyperparams_from_args,
)


class HAPPO:
    """Heterogeneous-Agent PPO with one actor per agent, sequential update."""

    def __init__(
        self,
        args: Dict[str, Any],
        obs_dim: int,
        state_dim: int,
        action_dim: int,
        n_agents: int,
        discrete: bool = True,
        device: torch.device | str = "cpu",
    ):
        self.args = args
        self.discrete = discrete
        self.device = torch.device(device)
        self.n_agents = int(n_agents)

        # One actor per agent -- full agent-level heterogeneity.
        actor_cls = CategoricalActor if discrete else GaussianActor
        self.actors = nn.ModuleList(
            [actor_cls(obs_dim, action_dim).to(self.device) for _ in range(self.n_agents)]
        )
        self.critic = VCritic(state_dim).to(self.device)
        self.value_normalizer = ValueNormalizer()

        opt = args["optim"]
        self.actor_optims = [
            torch.optim.Adam(a.parameters(), lr=opt["actor_lr"], eps=opt.get("optim_eps", 1e-5))
            for a in self.actors
        ]
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=opt["critic_lr"], eps=opt.get("optim_eps", 1e-5)
        )

        for k, v in hyperparams_from_args(args).items():
            setattr(self, k, v)

        seq = args.get("sequential_update", {})
        self.random_agent_permutation: bool = bool(
            seq.get("random_group_permutation", True)
        )
        self._rng = np.random.default_rng(int(args.get("seed", 0)))

    # ------------------------------------------------------------------ utils

    def _agent_order(self) -> np.ndarray:
        order = np.arange(self.n_agents, dtype=np.int64)
        if self.random_agent_permutation:
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
        """One pass over the on-policy buffer following HAPPO Algorithm 3."""
        info: Dict[str, Any] = {"group_losses": {}}

        adv_t = torch.as_tensor(advantages, device=self.device).float()
        adv_t = normalise_advantages(adv_t)

        if self.use_value_norm:
            self.value_normalizer.update(buffer.returns)

        T_, B_ = adv_t.shape
        M = torch.ones(T_, B_, device=self.device)

        # ---- Sequential actor updates ----
        for agent_id in self._agent_order():
            losses = self._update_agent_actor(buffer, M, adv_t, int(agent_id))
            info["group_losses"][int(agent_id)] = losses
            M = self._update_M_with_ratio(buffer, M, int(agent_id))

        # ---- Critic-only update, ONCE per iteration ----
        critic_info = self._update_critic(buffer)
        # Attach the critic loss to every agent's loss dict so the existing
        # logger (which prints `vl=` per group) keeps working without
        # changes. Identical value across groups is correct because there is
        # only one shared critic.
        for gid in info["group_losses"]:
            info["group_losses"][gid]["value_loss"] = critic_info["value_loss"]
        info["critic_loss"] = critic_info["value_loss"]
        info["critic_n_steps"] = critic_info["n_steps"]

        info["num_groups"] = self.n_agents
        info["active_groups"] = self.n_agents
        info["group_sizes"] = [1] * self.n_agents
        return info

    # ----------------------------------------------------------- internals

    def _update_agent_actor(
        self,
        buffer,
        M: torch.Tensor,
        adv: torch.Tensor,
        agent_id: int,
    ) -> Dict[str, float]:
        """PPO actor update for one agent, weighted by the cumulative scalar M.

        This method updates ONLY the actor; the centralised critic is
        updated once per iteration in :meth:`_update_critic`.
        """
        actor = self.actors[agent_id]
        actor_optim = self.actor_optims[agent_id]
        clip = self.clip_param

        policy_loss_acc = entropy_acc = approx_kl_acc = 0.0
        n_steps = 0
        early_stop = False

        buffer_size = buffer.size_per_agent()
        if self.mini_batch > 0:
            mb_size = self.mini_batch
        else:
            mb_size = max(1, buffer_size // max(1, self.num_mini_batch))

        for _ in range(self.ppo_epoch):
            for batch in buffer.get([agent_id], batch_size=mb_size, M=M, adv=adv):
                obs        = batch["obs"]            # (B, 1, O)
                actions    = batch["actions"]        # (B, 1, 1)
                old_lp     = batch["log_probs"]      # (B, 1)
                M_b        = batch["M"]              # (B,) cumulative ratio
                adv_b      = batch["adv"]            # (B,) normalised advantage

                avail  = batch.get("available_actions")    # (B, 1, n_actions) or None
                active = batch.get("active_masks")         # (B, 1) or None

                B = obs.shape[0]
                obs_flat = obs.squeeze(1)
                if self.discrete:
                    act_flat = actions.squeeze(1).squeeze(-1).long()
                else:
                    act_flat = actions.squeeze(1)
                old_lp_flat = old_lp.squeeze(1)
                avail_flat = (
                    avail.squeeze(1) if (avail is not None and self.discrete) else None
                )

                if active is not None:
                    mask = (active.squeeze(1) > 0.5).float()
                else:
                    mask = torch.ones(B, device=obs.device)
                n_eff = float(mask.sum().item())
                if n_eff < 1.0:
                    continue

                if self.discrete and avail_flat is not None:
                    dist = actor(obs_flat, available_actions=avail_flat)
                else:
                    dist = actor(obs_flat)
                new_lp, entropy = self._entropy_and_logprob(dist, act_flat)

                entropy_loss = (entropy * mask).sum() / max(n_eff, 1.0)

                weighted_adv = adv_b * M_b   # (B,)
                ratio = (new_lp - old_lp_flat).exp()
                surr1 = ratio * weighted_adv
                surr2 = ratio.clamp(1 - clip, 1 + clip) * weighted_adv
                pl_per_sample = -torch.min(surr1, surr2)
                policy_loss = (pl_per_sample * mask).sum() / max(n_eff, 1.0)
                with torch.no_grad():
                    diff = old_lp_flat - new_lp
                    approx_kl = float(((diff * mask).sum() / max(n_eff, 1.0)).item())

                actor_optim.zero_grad(set_to_none=True)
                (policy_loss - self.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
                actor_optim.step()

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
        ``get([agent_id=0], ...)`` interface is reused only to pull
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
        ph_agent = [0]

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
    def _update_M_with_ratio(
        self, buffer, M: torch.Tensor, agent_id: int,
    ) -> torch.Tensor:
        """Multiply M(t,b) by the just-updated agent's policy ratio.

        Honours ``available_actions`` and ``active_masks`` when present:
        masked-out / dead samples contribute log_ratio = 0.
        """
        actor = self.actors[agent_id]
        T_, B_ = M.shape

        flat_obs = buffer.obs[..., agent_id, :].reshape(T_ * B_, -1)
        flat_act = buffer.actions[..., agent_id, :].reshape(T_ * B_, -1)
        flat_lp_old = buffer.log_probs[..., agent_id].reshape(T_ * B_)

        obs_t = torch.as_tensor(flat_obs, device=self.device).float()
        if self.discrete:
            act_t = torch.as_tensor(flat_act, device=self.device).squeeze(-1).long()
        else:
            act_t = torch.as_tensor(flat_act, device=self.device).float()
        lp_old_t = torch.as_tensor(flat_lp_old, device=self.device).float()

        avail_t = None
        if buffer.available_actions is not None and self.discrete:
            avail_t = torch.as_tensor(
                buffer.available_actions[..., agent_id, :].reshape(T_ * B_, -1),
                device=self.device,
            ).float()

        if self.discrete and avail_t is not None:
            dist = actor(obs_t, available_actions=avail_t)
        else:
            dist = actor(obs_t)
        new_lp, _ = self._entropy_and_logprob(dist, act_t)
        log_ratio = (new_lp - lp_old_t).reshape(T_, B_)

        # Mask out dead-agent contributions (log_ratio = 0 for those).
        if buffer.active_masks is not None:
            active_t = torch.as_tensor(
                buffer.active_masks[..., agent_id], device=self.device
            ).float()
            log_ratio = log_ratio * active_t

        log_ratio = log_ratio.clamp(-10.0, 10.0)
        return M * log_ratio.exp()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, available_actions: torch.Tensor | None = None
    ):
        """Sample joint action -- each agent uses its own actor."""
        squeeze_threads = (obs.dim() == 2)
        if squeeze_threads:
            obs = obs.unsqueeze(0)
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0)
        n_threads, N = obs.shape[:2]

        if self.discrete:
            actions = torch.zeros(n_threads, N, dtype=torch.long, device=self.device)
        else:
            actions = torch.zeros(n_threads, N, self.actors[0].mu.out_features,
                                  device=self.device)
        log_probs = torch.zeros(n_threads, N, device=self.device)

        for i in range(self.n_agents):
            obs_i = obs[:, i]
            if self.discrete:
                avail_i = available_actions[:, i] if available_actions is not None else None
                a, lp = self.actors[i].act(obs_i, avail_i)
            else:
                dist = self.actors[i](obs_i)
                a = dist.rsample()
                lp = dist.log_prob(a).sum(-1)
            actions[:, i] = a
            log_probs[:, i] = lp

        if squeeze_threads:
            actions = actions.squeeze(0); log_probs = log_probs.squeeze(0)
        return actions, log_probs

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        v = self.critic(state).squeeze(-1)
        if self.use_value_norm:
            return self.value_normalizer.denormalize(v)
        return v

    def set_groups(self, groups) -> None:
        return None

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


__all__ = ["HAPPO"]
