"""
Off-policy runner for group-structured cooperative MARL.

Drives TySAC end-to-end. Mirrors HARL's ``off_policy_ha_runner.py`` but
performs sequential **group-level** updates rather than per-agent updates.
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from harl.utils.buffers import ReplayBuffer
from harl.utils.group import GroupAssignment


class OffPolicyGroupRunner:
    """Runner for TySAC-style off-policy training."""

    def __init__(
        self,
        args: Dict[str, Any],
        envs,
        eval_envs,
        groups,
        device: torch.device | str,
    ):
        self.args = args
        self.envs = envs
        self.eval_envs = eval_envs
        self.device = torch.device(device)

        if isinstance(groups, GroupAssignment):
            self.groups: List[List[int]] = groups.as_list_of_lists()
            self.group_assignment = groups
        else:
            self.groups = [list(map(int, g)) for g in groups]
            self.group_assignment = None
        self.num_agents = sum(len(g) for g in self.groups)

        self.algo = None
        self.buffer: ReplayBuffer | None = None

        sac = args["sac"]
        self.warmup: int = int(sac["warmup_steps"])
        self.update_per_train: int = int(sac.get("update_per_train", 1))
        self.batch_size: int = int(sac["batch_size"])
        self.buffer_size: int = int(sac["buffer_size"])
        self.num_env_steps: int = int(args["train"]["num_env_steps"])

        # n_rollout_threads is forced to 1; this runner does not VecEnv-batch.
        n_threads = int(args["train"].get("n_rollout_threads", 1))
        if n_threads != 1:
            raise NotImplementedError(
                "OffPolicyGroupRunner currently supports n_rollout_threads=1 "
                "only. Run multiple seeds in parallel instead."
            )

        log = args.get("log", {})
        self.log_interval: int = int(log.get("log_interval", 1000))
        self.save_interval: int = int(log.get("save_interval", 100_000))
        self.save_dir: Path = Path(log.get("save_dir", "results/runs/default"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Whether the synthetic SMACv2 dummy backend is in use. Off-policy
        # algorithms in this repo are continuous-action and TySAC/HASAC are
        # blocked from running on SMAC family in train.py -- so this flag
        # is virtually always False here, but we still surface it for
        # consistency with the on-policy runner.
        env_info = self.envs.get_env_info() if hasattr(self.envs, "get_env_info") else {}
        self._using_dummy_env: bool = bool(env_info.get("using_dummy", False))
        if self._using_dummy_env:
            print(
                "[runner] WARNING: running against a SYNTHETIC dummy env. "
                "All metrics will reflect random reward, NOT real algorithm "
                "performance.",
                flush=True,
            )

        self._returns: deque = deque(maxlen=100)
        self._wins: deque = deque(maxlen=100)
        self._t0 = time.time()

    # ---------------------------------------------------------------- main

    def run(self) -> None:
        self._lazy_init_buffer()

        if hasattr(self.envs, "reset"):
            obs, state, groups = self.envs.reset()
            self._maybe_update_groups(groups)
        else:
            obs, state = None, None
        ep_ret = 0.0

        for step in range(self.num_env_steps):
            if obs is None:
                break  # synthetic-buffer test path

            obs_t = torch.as_tensor(obs, device=self.device).float()
            if step < self.warmup:
                a = torch.empty(self.num_agents, self.algo.action_dim, device=self.device).uniform_(-1, 1)
            else:
                a = self.algo.select_actions(obs_t)

            obs_next, state_next, reward, done, info = self.envs.step(a.detach().cpu().numpy())
            self.buffer.push(
                obs=obs, state=state, actions=a.detach().cpu().numpy(),
                reward=reward, next_obs=obs_next, next_state=state_next, done=float(done),
            )
            ep_ret += reward
            obs, state = obs_next, state_next
            if done:
                self._returns.append(ep_ret)
                self._wins.append(float(info.get("battle_won", False)))
                ep_ret = 0.0
                obs, state, groups = self.envs.reset()
                self._maybe_update_groups(groups)

            if step >= self.warmup and len(self.buffer) >= self.batch_size:
                train_info: Dict[str, Any] = {}
                for _ in range(self.update_per_train):
                    train_info = self.algo.train(self.buffer)

                if step % self.log_interval == 0:
                    self._log(step, train_info)
                if step % self.save_interval == 0 and step > 0:
                    self._save(step)

    # ----------------------------------------------------------- internals

    def _maybe_update_groups(self, groups) -> None:
        """See OnPolicyGroupRunner._maybe_update_groups."""
        if groups is None or self.algo is None:
            return
        if not hasattr(self.algo, "set_groups"):
            return
        try:
            self.algo.set_groups(groups)
            self.num_agents = self.algo.num_agents
        except (ValueError, TypeError):
            pass

    def _lazy_init_buffer(self) -> None:
        info = self.envs.get_env_info() if hasattr(self.envs, "get_env_info") else {}
        obs_dim = info.get("obs_shape", 64)
        state_dim = info.get("state_shape", obs_dim * self.num_agents)
        action_dim = info.get("action_dim", self.algo.action_dim if self.algo else 1)
        self.buffer = ReplayBuffer(
            capacity=self.buffer_size,
            n_agents=self.num_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )

    def _log(self, step: int, info: Dict[str, Any]) -> None:
        elapsed = time.time() - self._t0
        payload: Dict[str, Any] = {
            "step": step,
            "elapsed_s": elapsed,
            "fps": int(step / max(elapsed, 1e-3)),
            "ep_return_mean": float(np.mean(self._returns)) if self._returns else float("nan"),
            "win_rate": float(np.mean(self._wins)) if self._wins else float("nan"),
            "critic_loss": info.get("critic_loss", float("nan")),
            "using_dummy": self._using_dummy_env,
        }
        for gid, losses in info.get("group_losses", {}).items():
            for k, v in losses.items():
                payload[f"g{gid}/{k}"] = v

        with open(self.save_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        msg = [f"[step {step}]", f"ret={payload['ep_return_mean']:.3f}",
               f"cl={payload['critic_loss']:.4f}"]
        if self._using_dummy_env:
            msg.append("[DUMMY]")
        for gid, losses in info.get("group_losses", {}).items():
            msg.append(f"g{gid}: al={losses['actor_loss']:+.4f} alpha={losses['alpha']:.3f}")
        print(" ".join(msg), flush=True)

    def _save(self, step: int) -> None:
        path = self.save_dir / f"ckpt_step{step}.pt"
        torch.save(self.algo.state_dict(), path)
