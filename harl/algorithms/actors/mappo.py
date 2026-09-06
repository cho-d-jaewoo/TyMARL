"""
MAPPO -- Multi-Agent PPO (Yu et al., NeurIPS 2022 Datasets and Benchmarks).

This is a standalone implementation following Algorithm 1 from the original
MAPPO paper. We share the per-batch infrastructure (value normaliser,
clipped value loss, advantage normalisation) with HAPPO and TyPPO via
``_ppo_common`` to keep the comparison apples-to-apples, but the *training
loop* is MAPPO-specific:

* Single shared actor network used by all agents (parameter sharing).
* Centralised V-critic on the global state (CTDE).
* All ``(time, thread, agent)`` triples are stacked into one big batch and
  the policy is updated *simultaneously* over the whole team.

SMAC-family integration
-----------------------
When the buffer carries optional SMAC-family fields, they are honoured:

* ``available_actions``: passed into the actor so invalid actions are
  masked out at both action sampling and ratio computation.
* ``active_masks``: dead agents are excluded from the policy / entropy
  loss via a sample-wise mask.
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


class MAPPO:
    """Multi-Agent PPO with parameter sharing and a centralised V-critic."""

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

        # Single shared actor — parameter sharing across all agents.
        actor_cls = CategoricalActor if discrete else GaussianActor
        self.actor = actor_cls(obs_dim, action_dim).to(self.device)
        self.critic = VCritic(state_dim).to(self.device)
        self.value_normalizer = ValueNormalizer()

        opt = args["optim"]
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=opt["actor_lr"], eps=opt.get("optim_eps", 1e-5)
        )
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=opt["critic_lr"], eps=opt.get("optim_eps", 1e-5)
        )

        for k, v in hyperparams_from_args(args).items():
            setattr(self, k, v)

    # ------------------------------------------------------------------ utils

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
        """One pass over the on-policy buffer with parameter-shared PPO."""
        info: Dict[str, Any] = {"group_losses": {}}

        adv_t = torch.as_tensor(advantages, device=self.device).float()
        adv_t = normalise_advantages(adv_t)

        if self.use_value_norm:
            self.value_normalizer.update(buffer.returns)

        all_agents = list(range(self.n_agents))

        buffer_size = buffer.size_per_agent()
        if self.mini_batch > 0:
            mb_size = self.mini_batch
        else:
            mb_size = max(1, buffer_size // max(1, self.num_mini_batch))

        policy_loss_acc = value_loss_acc = entropy_acc = approx_kl_acc = 0.0
        n_steps = 0
        early_stop = False

        for _ in range(self.ppo_epoch):
            for batch in buffer.get(all_agents, batch_size=mb_size, adv=adv_t):
                obs        = batch["obs"]            # (B, N, O)
                actions    = batch["actions"]        # (B, N, 1) discrete
                old_lp     = batch["log_probs"]      # (B, N)
                ret        = batch["returns"]        # (B,)
                state      = batch["state"]          # (B, S)
                old_values = batch["values"]         # (B,)
                adv_b      = batch["adv"]            # (B,) — already normalised

                avail  = batch.get("available_actions")    # (B, N, n_actions) or None
                active = batch.get("active_masks")         # (B, N) or None

                B, N = obs.shape[:2]
                obs_flat = obs.reshape(B * N, -1)
                if self.discrete:
                    act_flat = actions.reshape(B * N, -1).squeeze(-1).long()
                else:
                    act_flat = actions.reshape(B * N, -1)
                old_lp_flat = old_lp.reshape(B * N)
                avail_flat = (
                    avail.reshape(B * N, -1) if avail is not None and self.discrete else None
                )

                # Sample mask: only alive agents contribute. If active is None,
                # all samples count.
                if active is not None:
                    mask = (active > 0.5).reshape(B * N).float()
                else:
                    mask = torch.ones(B * N, device=obs.device)
                n_eff = float(mask.sum().item())
                if n_eff < 1.0:
                    continue

                # Forward pass.
                if self.discrete and avail_flat is not None:
                    dist = self.actor(obs_flat, available_actions=avail_flat)
                else:
                    dist = self.actor(obs_flat)
                new_lp, entropy = self._entropy_and_logprob(dist, act_flat)

                entropy_loss = (entropy * mask).sum() / max(n_eff, 1.0)

                # Same advantage for all agents on a given (t, b) -- broadcast.
                adv_per_pair = adv_b.unsqueeze(-1).expand(B, N).reshape(B * N)

                ratio = (new_lp - old_lp_flat).exp()
                surr1 = ratio * adv_per_pair
                surr2 = ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv_per_pair
                pl_per_sample = -torch.min(surr1, surr2)
                policy_loss = (pl_per_sample * mask).sum() / max(n_eff, 1.0)
                with torch.no_grad():
                    diff = old_lp_flat - new_lp
                    approx_kl = float(((diff * mask).sum() / max(n_eff, 1.0)).item())

                # Value loss (clipped + Huber, in normalised space if applicable).
                values_raw = self.critic(state).squeeze(-1)
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

                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                (policy_loss
                 + self.value_loss_coef * value_loss
                 - self.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.actor_optim.step()
                self.critic_optim.step()

                policy_loss_acc += float(policy_loss.detach())
                value_loss_acc += float(value_loss.detach())
                entropy_acc += float(entropy_loss.detach())
                approx_kl_acc += approx_kl
                n_steps += 1

                if approx_kl > 1.5 * self.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        n = max(n_steps, 1)
        info["group_losses"][0] = {
            "policy_loss": policy_loss_acc / n,
            "value_loss":  value_loss_acc / n,
            "entropy":     entropy_acc / n,
            "approx_kl":   approx_kl_acc / n,
            "early_stop":  float(early_stop),
            "n_steps":     n_steps,
        }
        info["num_groups"] = 1
        info["active_groups"] = 1
        info["group_sizes"] = [self.n_agents]
        return info

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(
        self, obs: torch.Tensor, available_actions: torch.Tensor | None = None
    ):
        """Sample joint action from the shared policy."""
        squeeze_threads = (obs.dim() == 2)
        if squeeze_threads:
            obs = obs.unsqueeze(0)
            if available_actions is not None:
                available_actions = available_actions.unsqueeze(0)
        n_threads, N = obs.shape[:2]

        obs_flat = obs.reshape(n_threads * N, -1)
        if self.discrete:
            avail_flat = (available_actions.reshape(n_threads * N, -1)
                          if available_actions is not None else None)
            a, lp = self.actor.act(obs_flat, avail_flat)
            a = a.view(n_threads, N)
            lp = lp.view(n_threads, N)
        else:
            dist = self.actor(obs_flat)
            a = dist.rsample()
            lp = dist.log_prob(a).sum(-1)
            a = a.view(n_threads, N, -1)
            lp = lp.view(n_threads, N)

        if squeeze_threads:
            a = a.squeeze(0); lp = lp.squeeze(0)
        return a, lp

    @torch.no_grad()
    def value(self, state: torch.Tensor) -> torch.Tensor:
        v = self.critic(state).squeeze(-1)
        if self.use_value_norm:
            return self.value_normalizer.denormalize(v)
        return v

    # ------------------------------------------------------------------
    # set_groups: ignored. MAPPO has no group structure; provided as a
    # no-op so the runner can call it uniformly across algorithms.
    # ------------------------------------------------------------------

    def set_groups(self, groups) -> None:
        return None

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "value_norm": {
                "mean": float(self.value_normalizer.mean),
                "var": float(self.value_normalizer.var),
                "count": float(self.value_normalizer.count),
            },
        }

    def load_state_dict(self, sd: Dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        if "value_norm" in sd:
            self.value_normalizer.mean = sd["value_norm"]["mean"]
            self.value_normalizer.var = sd["value_norm"]["var"]
            self.value_normalizer.count = sd["value_norm"]["count"]


__all__ = ["MAPPO"]
