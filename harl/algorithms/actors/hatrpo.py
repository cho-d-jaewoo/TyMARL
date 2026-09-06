"""
HATRPO -- Heterogeneous-Agent Trust Region Policy Optimisation
         (Kuba et al., ICLR 2022, Algorithm 2).

Standalone implementation. Sequential per-agent update with cumulative
ratio M, identical structure to HAPPO; the only difference is the
*per-agent* update: instead of clipped-surrogate PPO, HATRPO performs
a TRPO step:

  1. compute the policy gradient g of the (M-weighted) surrogate
  2. solve the linear system   F x = g    via conjugate gradient,
     where F is the Fisher information matrix of the current policy
     (Fisher-vector products are computed via Hessian-vector products
     of the mean KL divergence -- no explicit F construction)
  3. compute the maximal step size that satisfies the KL trust region
     beta = sqrt( 2 * max_kl / (x^T F x) )
  4. line-search by halving beta until both the surrogate improves
     and the KL constraint is satisfied

Critic update is the same MAPPO-style clipped value loss with optional
value normalisation (shared per-batch infra in ``_ppo_common``).

SMAC-family integration
-----------------------
``available_actions`` and ``active_masks`` are honoured if the buffer
provides them: invalid actions are masked out at action-distribution
time, dead agents are zero-weighted in the surrogate / KL.

References
----------
* Kuba, J. G., Chen, R., Wen, M., et al.,
  "Trust Region Policy Optimisation in Multi-Agent Reinforcement
  Learning." ICLR, 2022.
* Reference implementation: https://github.com/cyanrain7/TRPO-in-MARL
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
)


# ---------------------------------------------------------------------------
# Helpers for parameter / gradient flattening
# ---------------------------------------------------------------------------

def _flat_params(module: nn.Module) -> torch.Tensor:
    """Concatenate all of ``module``'s parameters into a single 1-D tensor."""
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def _set_flat_params(module: nn.Module, flat: torch.Tensor) -> None:
    """Inverse of ``_flat_params``: copy a 1-D tensor back into the module's
    parameters in the same order."""
    idx = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p.data))
        idx += n


def _flat_grad(
    loss: torch.Tensor,
    params,
    create_graph: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Return ``grad(loss, params)`` flattened. ``create_graph=True`` is
    needed when the gradient itself will be differentiated again (FVP)."""
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=retain_graph,
        allow_unused=True,
    )
    out = []
    for g, p in zip(grads, params):
        if g is None:
            out.append(torch.zeros_like(p).view(-1))
        else:
            out.append(g.contiguous().view(-1))
    return torch.cat(out)


def _conjugate_gradient(
    Fvp,                         # callable: vec -> Fisher-vector product
    b: torch.Tensor,             # right-hand side, shape (D,)
    n_iters: int = 10,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Solve ``F x = b`` for ``x`` using conjugate gradient.

    ``F`` is implicit -- only ``Fvp(v)`` matrix-vector products are used.
    Returns an approximate solution after ``n_iters`` CG steps or earlier
    if the residual drops below ``residual_tol``.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(n_iters):
        Fp = Fvp(p)
        alpha = rdotr / (torch.dot(p, Fp) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Fp
        new_rdotr = torch.dot(r, r)
        if float(new_rdotr) < residual_tol:
            break
        p = r + (new_rdotr / (rdotr + 1e-12)) * p
        rdotr = new_rdotr
    return x


# ---------------------------------------------------------------------------
# HATRPO
# ---------------------------------------------------------------------------

class HATRPO:
    """Heterogeneous-Agent TRPO with one actor per agent, sequential update.

    Architectural notes mirror ``HAPPO``:
    * One actor *network per agent* -- no parameter sharing.
    * Centralised V-critic on the global state (CTDE, shared across agents).
    """

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

        actor_cls = CategoricalActor if discrete else GaussianActor
        self.actors = nn.ModuleList(
            [actor_cls(obs_dim, action_dim).to(self.device) for _ in range(self.n_agents)]
        )
        self.critic = VCritic(state_dim).to(self.device)
        self.value_normalizer = ValueNormalizer()

        opt = args["optim"]
        # Note: actor optimizers are NOT used for the policy update in TRPO
        # (we set parameters directly via line search). Kept for API
        # uniformity and to allow future hybrid schemes.
        self.actor_optims = [
            torch.optim.Adam(a.parameters(), lr=opt["actor_lr"], eps=opt.get("optim_eps", 1e-5))
            for a in self.actors
        ]
        self.critic_optim = torch.optim.Adam(
            self.critic.parameters(), lr=opt["critic_lr"], eps=opt.get("optim_eps", 1e-5)
        )

        # ----- TRPO-specific hyperparameters --------------------------------
        trpo = args["trpo"]
        self.max_kl: float = float(trpo.get("max_kl", 0.01))
        self.cg_iters: int = int(trpo.get("cg_iters", 10))
        self.cg_damping: float = float(trpo.get("cg_damping", 0.1))
        self.line_search_steps: int = int(trpo.get("line_search_steps", 10))
        self.line_search_decay: float = float(trpo.get("line_search_decay", 0.5))
        self.entropy_coef: float = float(trpo.get("entropy_coef", 0.0))

        # Critic-update hyperparameters (PPO-style, since the critic loss
        # is the same regardless of how the actor moves).
        self.value_loss_coef: float = float(trpo.get("value_loss_coef", 1.0))
        self.max_grad_norm: float = float(trpo.get("max_grad_norm", 10.0))
        self.huber_delta: float = float(trpo.get("huber_delta", 10.0))
        self.use_huber: bool = bool(trpo.get("use_huber_loss", True))
        self.use_clipped_value_loss: bool = bool(trpo.get("use_clipped_value_loss", True))
        self.use_value_norm: bool = bool(trpo.get("use_value_norm", True))
        self.value_clip_param: float = float(trpo.get("clip_param", 0.2))
        self.critic_epoch: int = int(trpo.get("critic_epoch", trpo.get("ppo_epoch", 5)))
        self.num_mini_batch: int = int(trpo.get("num_mini_batch", 1))
        self.mini_batch: int = int(trpo.get("mini_batch", 0))

        seq = args.get("sequential_update", {})
        self.random_agent_permutation: bool = bool(
            seq.get("random_group_permutation", True)
        )
        self._rng = np.random.default_rng(int(args.get("seed", 0)))

    # ---------------------------------------------------------------- utils

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
        """One pass over the on-policy buffer with HATRPO sequential update."""
        info: Dict[str, Any] = {"group_losses": {}}

        adv_t = torch.as_tensor(advantages, device=self.device).float()
        adv_t = normalise_advantages(adv_t)

        if self.use_value_norm:
            self.value_normalizer.update(buffer.returns)

        T_, B_ = adv_t.shape
        M = torch.ones(T_, B_, device=self.device)

        for agent_id in self._agent_order():
            losses = self._update_agent(buffer, M, adv_t, int(agent_id))
            info["group_losses"][int(agent_id)] = losses
            M = self._update_M_with_ratio(buffer, M, int(agent_id))

        info["num_groups"] = self.n_agents
        info["active_groups"] = self.n_agents
        info["group_sizes"] = [1] * self.n_agents
        return info

    # ----------------------------------------------------------- internals

    def _gather_full_agent_batch(
        self, buffer, M: torch.Tensor, adv: torch.Tensor, agent_id: int,
    ) -> Dict[str, torch.Tensor]:
        """Collect every (t, b) sample for ``agent_id`` in one big tensor block.

        TRPO computes the natural gradient over the full batch (or a single
        large minibatch) rather than iterating over many small minibatches
        like PPO. This keeps the per-iteration FIM estimate stable.
        """
        T_, B_ = M.shape
        # Pull all (T*B) samples for this agent.
        obs_np = buffer.obs[..., agent_id, :].reshape(T_ * B_, -1)
        act_np = buffer.actions[..., agent_id, :].reshape(T_ * B_, -1)
        lp_old_np = buffer.log_probs[..., agent_id].reshape(T_ * B_)

        avail_np = None
        if buffer.available_actions is not None and self.discrete:
            avail_np = buffer.available_actions[..., agent_id, :].reshape(T_ * B_, -1)
        active_np = None
        if buffer.active_masks is not None:
            active_np = buffer.active_masks[..., agent_id].reshape(T_ * B_)

        return {
            "obs":     torch.as_tensor(obs_np, device=self.device).float(),
            "actions": (
                torch.as_tensor(act_np, device=self.device).squeeze(-1).long()
                if self.discrete
                else torch.as_tensor(act_np, device=self.device).float()
            ),
            "log_probs_old": torch.as_tensor(lp_old_np, device=self.device).float(),
            "M_flat":        M.reshape(T_ * B_),
            "adv_flat":      adv.reshape(T_ * B_),
            "available_actions": (
                torch.as_tensor(avail_np, device=self.device).float()
                if avail_np is not None else None
            ),
            "active_masks": (
                torch.as_tensor(active_np, device=self.device).float()
                if active_np is not None else None
            ),
        }

    def _surrogate(
        self, actor: nn.Module, batch: Dict[str, torch.Tensor],
    ):
        """Return (surrogate_loss_to_MAXIMIZE, mean_entropy, n_eff).

        surrogate = mean_over_active( (M * adv) * exp(new_lp - old_lp) )
        """
        if self.discrete and batch["available_actions"] is not None:
            dist = actor(batch["obs"], available_actions=batch["available_actions"])
        else:
            dist = actor(batch["obs"])
        new_lp, entropy = self._entropy_and_logprob(dist, batch["actions"])

        if batch["active_masks"] is not None:
            mask = (batch["active_masks"] > 0.5).float()
        else:
            mask = torch.ones_like(new_lp)
        n_eff = torch.clamp(mask.sum(), min=1.0)

        weighted_adv = batch["adv_flat"] * batch["M_flat"]
        ratio = (new_lp - batch["log_probs_old"]).exp()
        surr_per_sample = ratio * weighted_adv

        surrogate = (surr_per_sample * mask).sum() / n_eff
        ent = (entropy * mask).sum() / n_eff
        return surrogate, ent, n_eff

    def _kl_mean(
        self, actor: nn.Module, batch: Dict[str, torch.Tensor],
        old_dist_pkg,
    ) -> torch.Tensor:
        """Mean KL between the *old* (frozen-snapshot) policy and the
        current policy, restricted to active samples."""
        if self.discrete and batch["available_actions"] is not None:
            new_dist = actor(batch["obs"], available_actions=batch["available_actions"])
        else:
            new_dist = actor(batch["obs"])
        old_dist = old_dist_pkg[0]
        kl = torch.distributions.kl.kl_divergence(old_dist, new_dist)
        if kl.dim() > 1:
            kl = kl.sum(-1)
        if batch["active_masks"] is not None:
            mask = (batch["active_masks"] > 0.5).float()
            return (kl * mask).sum() / torch.clamp(mask.sum(), min=1.0)
        return kl.mean()

    def _capture_old_dist(
        self, actor: nn.Module, batch: Dict[str, torch.Tensor],
    ):
        """Capture the *current* (about-to-be-replaced) policy distribution,
        with parameters detached so subsequent autograd doesn't flow back."""
        with torch.no_grad():
            if self.discrete and batch["available_actions"] is not None:
                old_dist = actor(batch["obs"], available_actions=batch["available_actions"])
            else:
                old_dist = actor(batch["obs"])
            if isinstance(old_dist, torch.distributions.Categorical):
                old_dist = torch.distributions.Categorical(probs=old_dist.probs.detach())
            elif isinstance(old_dist, torch.distributions.Normal):
                old_dist = torch.distributions.Normal(
                    loc=old_dist.loc.detach(), scale=old_dist.scale.detach()
                )
        return (old_dist,)

    def _update_agent(
        self,
        buffer,
        M: torch.Tensor,
        adv: torch.Tensor,
        agent_id: int,
    ) -> Dict[str, float]:
        """TRPO step for one agent, weighted by the cumulative scalar M."""
        actor = self.actors[agent_id]
        params = list(actor.parameters())

        batch = self._gather_full_agent_batch(buffer, M, adv, agent_id)

        # 1) Compute the surrogate gradient g = grad(surrogate, params).
        old_dist_pkg = self._capture_old_dist(actor, batch)
        surrogate, ent_pre, _ = self._surrogate(actor, batch)
        # We optimise (surrogate + entropy_coef * entropy). HATRPO standard
        # uses entropy_coef=0; we expose it because zero-entropy regularisation
        # tends to collapse the policy on stochastic-team SMAC variants.
        objective = surrogate + self.entropy_coef * ent_pre

        # Maximise objective: gradient direction is +grad(objective).
        g = _flat_grad(objective, params, retain_graph=True)
        if (
            torch.isnan(g).any() or torch.isinf(g).any()
            or float(g.norm()) < 1e-12
        ):
            return {
                "policy_loss": 0.0, "value_loss": 0.0, "entropy": float(ent_pre),
                "approx_kl": 0.0, "trpo_step_size": 0.0, "ls_steps": 0.0,
                "n_steps": 0,
            }

        # 2) Build Fisher-vector product callable via Hessian of mean KL.
        def Fvp(v: torch.Tensor) -> torch.Tensor:
            kl_mean = self._kl_mean(actor, batch, old_dist_pkg)
            grads = _flat_grad(kl_mean, params, create_graph=True, retain_graph=True)
            gv = (grads * v).sum()
            hv = _flat_grad(gv, params, retain_graph=True)
            return hv + self.cg_damping * v

        # 3) Conjugate gradient: solve F * step_dir = g.
        step_dir = _conjugate_gradient(Fvp, g, n_iters=self.cg_iters)

        # 4) Maximal step size that satisfies the trust region.
        shs = 0.5 * float((step_dir * Fvp(step_dir)).sum())
        if shs <= 0 or not np.isfinite(shs):
            return {
                "policy_loss": 0.0, "value_loss": 0.0, "entropy": float(ent_pre),
                "approx_kl": 0.0, "trpo_step_size": 0.0, "ls_steps": 0.0,
                "n_steps": 0,
            }
        beta_max = float(np.sqrt(self.max_kl / shs))
        full_step = beta_max * step_dir

        # 5) Line search using the *expected vs actual improvement* ratio
        #    (standard TRPO criterion, see Schulman 2015 §6 and the
        #    cyanrain7/TRPO-in-MARL reference implementation). The
        #    expected first-order improvement of the objective is
        #    g . full_step; we accept the step if (a) the actual
        #    improvement is at least 10% of expected, (b) KL is within
        #    the trust region, and (c) the new objective is finite. This
        #    is much more permissive than a raw `> old + eps` test, which
        #    rejected nearly every step in our earlier runs because
        #    advantage normalisation drives the surrogate close to zero.
        expected_improve = float((g * full_step).sum())
        with torch.no_grad():
            old_params = _flat_params(actor)
            objective_old = float(objective.detach())

        ls_step = 0
        accepted = False
        beta = 1.0
        kl_after_f = 0.0
        actual_improve = 0.0
        for ls_step in range(self.line_search_steps):
            new_params = old_params + beta * full_step
            _set_flat_params(actor, new_params)
            with torch.no_grad():
                surrogate_new, ent_new, _ = self._surrogate(actor, batch)
                objective_new = float(surrogate_new + self.entropy_coef * ent_new)
                kl_after = self._kl_mean(actor, batch, old_dist_pkg)
                kl_after_f = float(kl_after)
            actual_improve = objective_new - objective_old
            ratio = actual_improve / (beta * expected_improve + 1e-12)
            kl_ok = kl_after_f <= self.max_kl
            finite = np.isfinite(objective_new)
            # Accept if expected improvement is positive AND we got at
            # least 10% of it AND KL is within bounds.
            if finite and kl_ok and expected_improve > 0 and ratio >= 0.1:
                accepted = True
                break
            beta *= self.line_search_decay
        if not accepted:
            _set_flat_params(actor, old_params)
            beta = 0.0
            kl_after_f = 0.0

        # 6) Critic update (PPO-style clipped value loss, multiple epochs).
        v_loss_acc = 0.0
        v_steps = 0
        buffer_size = buffer.size_per_agent()
        if self.mini_batch > 0:
            mb_size = self.mini_batch
        else:
            mb_size = max(1, buffer_size // max(1, self.num_mini_batch))

        for _ in range(self.critic_epoch):
            for v_batch in buffer.get([agent_id], batch_size=mb_size):
                state      = v_batch["state"]
                ret        = v_batch["returns"]
                old_values = v_batch["values"]
                values_raw = self.critic(state).squeeze(-1)
                if self.use_value_norm:
                    target = self.value_normalizer.normalize(ret).float()
                    old_pred = self.value_normalizer.normalize(old_values).float()
                else:
                    target = ret
                    old_pred = old_values
                value_loss = ppo_value_loss(
                    pred=values_raw, target=target, old_pred=old_pred,
                    clip_param=self.value_clip_param,
                    use_clipped_value_loss=self.use_clipped_value_loss,
                    use_huber=self.use_huber, huber_delta=self.huber_delta,
                )
                self.critic_optim.zero_grad(set_to_none=True)
                (self.value_loss_coef * value_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optim.step()
                v_loss_acc += float(value_loss.detach())
                v_steps += 1

        # 7) Reporting.
        with torch.no_grad():
            surrogate_post, ent_post, _ = self._surrogate(actor, batch)
        return {
            "policy_loss":    -float(surrogate_post),
            "value_loss":     v_loss_acc / max(v_steps, 1),
            "entropy":        float(ent_post),
            "approx_kl":      float(kl_after_f) if accepted else 0.0,
            "trpo_step_size": beta * beta_max if accepted else 0.0,
            "ls_steps":       float(ls_step + 1),
            "n_steps":        v_steps,
        }

    @torch.no_grad()
    def _update_M_with_ratio(
        self, buffer, M: torch.Tensor, agent_id: int,
    ) -> torch.Tensor:
        """Multiply M(t,b) by the just-updated agent's policy ratio.

        Same logic as HAPPO. Honours active_masks when present.
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

        if buffer.active_masks is not None:
            active_t = torch.as_tensor(
                buffer.active_masks[..., agent_id], device=self.device
            ).float()
            log_ratio = log_ratio * active_t

        log_ratio = log_ratio.clamp(-10.0, 10.0)
        return M * log_ratio.exp()

    # ------------------------------------------------------------------
    # Inference (used by the runner)
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
            actions = torch.zeros(
                n_threads, N, self.actors[0].mu.out_features, device=self.device,
            )
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
                "mean":  float(self.value_normalizer.mean),
                "var":   float(self.value_normalizer.var),
                "count": float(self.value_normalizer.count),
            },
        }

    def load_state_dict(self, sd: Dict) -> None:
        for a, w in zip(self.actors, sd["actors"]):
            a.load_state_dict(w)
        self.critic.load_state_dict(sd["critic"])
        if "value_norm" in sd:
            self.value_normalizer.mean  = sd["value_norm"]["mean"]
            self.value_normalizer.var   = sd["value_norm"]["var"]
            self.value_normalizer.count = sd["value_norm"]["count"]


__all__ = ["HATRPO"]
