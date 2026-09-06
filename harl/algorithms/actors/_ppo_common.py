"""
Shared helper functions used by MAPPO, HAPPO, and TyPPO.

Each algorithm has its OWN top-level training loop (sequential vs.
simultaneous updates, with vs. without cumulative ratio, etc.) — that is
where the algorithmic differences live and we keep them separate. But the
*per-batch* mechanics — value loss with clipping + Huber + normalisation,
advantage normalisation policy, KL early-stop bookkeeping — are
implementation details where having one well-tested version that all three
algorithms call into avoids drift bugs between baselines.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalise_advantages(adv: torch.Tensor) -> torch.Tensor:
    """Normalise advantages once over the entire iteration's data.

    Done before any sequential / mini-batch loop. Normalising again
    *inside* the loop after multiplying by a sequential weighting (TyPPO's
    or HAPPO's M) would cancel the weighting — that was the bug in our
    earlier code path.
    """
    mean = adv.mean()
    std = adv.std() + 1e-8
    return (adv - mean) / std


def ppo_value_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    old_pred: torch.Tensor,
    clip_param: float,
    use_clipped_value_loss: bool,
    use_huber: bool,
    huber_delta: float,
) -> torch.Tensor:
    """MAPPO-style value loss.

    With clipping (default): max((V - R)^2, (clip(V, V_old±ε) - R)^2)
    Optionally Huber-smoothed. Operates in *normalised* return space when
    a value normaliser is in use; the caller applies normalisation before
    passing in the tensors.
    """
    if use_clipped_value_loss:
        pred_clipped = old_pred + (pred - old_pred).clamp(-clip_param, clip_param)
        if use_huber:
            vl_unc = F.huber_loss(pred, target, delta=huber_delta, reduction="none")
            vl_cli = F.huber_loss(pred_clipped, target, delta=huber_delta, reduction="none")
        else:
            vl_unc = (pred - target).pow(2)
            vl_cli = (pred_clipped - target).pow(2)
        return torch.max(vl_unc, vl_cli).mean()
    else:
        if use_huber:
            return F.huber_loss(pred, target, delta=huber_delta)
        return F.mse_loss(pred, target)


def ppo_policy_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    weighted_advantage: torch.Tensor,
    clip_param: float,
):
    """Standard PPO clipped surrogate, with caller-supplied weighted advantages.

    `weighted_advantage` lets the caller plug in:
    - bare adv          -> standard PPO (used by MAPPO)
    - adv * M_cumulative -> sequential-update PPO (used by HAPPO and TyPPO)
    """
    ratio = (new_log_prob - old_log_prob).exp()
    surr1 = ratio * weighted_advantage
    surr2 = ratio.clamp(1 - clip_param, 1 + clip_param) * weighted_advantage
    policy_loss = -torch.min(surr1, surr2).mean()
    with torch.no_grad():
        approx_kl = (old_log_prob - new_log_prob).mean().item()
    return policy_loss, ratio, approx_kl


def hyperparams_from_args(args: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the PPO hyperparameters into a flat dict for class storage."""
    ppo = args["ppo"]
    return {
        "clip_param":              float(ppo["clip_param"]),
        "ppo_epoch":               int(ppo["ppo_epoch"]),
        "entropy_coef":            float(ppo["entropy_coef"]),
        "value_loss_coef":         float(ppo["value_loss_coef"]),
        "max_grad_norm":           float(ppo["max_grad_norm"]),
        "huber_delta":             float(ppo.get("huber_delta", 10.0)),
        "use_huber":               bool(ppo.get("use_huber_loss", True)),
        "use_clipped_value_loss":  bool(ppo.get("use_clipped_value_loss", True)),
        "use_value_norm":          bool(ppo.get("use_value_norm", True)),
        "num_mini_batch":          int(ppo.get("num_mini_batch", 1)),
        "mini_batch":              int(ppo.get("mini_batch", 0)),
        "target_kl":               float(ppo.get("target_kl", 0.06)),
    }
