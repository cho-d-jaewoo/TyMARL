"""
On-policy runner for group-structured cooperative MARL.

Drives TyPPO end-to-end: collects rollouts, computes GAE on a centralised V,
calls the algorithm's group-level update, and handles JSONL logging +
checkpointing. Mirrors the structure of HARL's ``on_policy_ha_runner.py`` but
performs sequential **group-level** updates rather than per-agent updates.

What this runner makes sure to thread through to the buffer (and therefore
through to the algorithm at training time):

* ``available_actions`` — read every step from the env so SMACv2 invalid
  actions never enter the PPO ratio.
* ``active_masks`` — alive flags per agent so dead agents do not corrupt
  the policy / entropy loss.
* ``actor_id_per_step`` — the canonical-type id of each agent at the step
  the action was taken. This is what makes type-routed updates correct
  even when ``episode_length > episode_limit`` causes a buffer to span
  multiple SMACv2 episodes with different team compositions.
* ``next_state`` — for accurate GAE bootstrap when the rollout is cut
  short before terminal (used by ``_bootstrap_last_value``).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from harl.utils.buffers import OnPolicyBuffer
from harl.utils.group import GroupAssignment


class OnPolicyGroupRunner:
    """Runner for TyPPO-style on-policy training."""

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

        # Accept either a GroupAssignment or list-of-lists.
        if isinstance(groups, GroupAssignment):
            self.groups: List[List[int]] = groups.as_list_of_lists()
            self.group_assignment = groups
        else:
            self.groups = [list(map(int, g)) for g in groups]
            self.group_assignment = None
        self.num_agents = sum(len(g) for g in self.groups)

        self.algo = None      # populated by harl.train.main
        self.buffer: OnPolicyBuffer | None = None

        tr = args["train"]
        self.episode_length: int = int(tr["episode_length"])
        self.n_threads: int = int(tr.get("n_rollout_threads", 1))
        if self.n_threads != 1:
            raise NotImplementedError(
                "OnPolicyGroupRunner currently supports n_rollout_threads=1 "
                "only. The collection loop assumes a single non-VecEnv. "
                "Increase episode_length or run multiple seeds in parallel "
                "instead."
            )
        self.num_env_steps: int = int(tr["num_env_steps"])

        # Cross-check episode_length against env.episode_limit if exposed.
        # If episode_length > episode_limit, the buffer can hold multiple
        # episodes worth of transitions, which is supported (we track per-
        # step actor ids), but warn the user about the additional bookkeeping.
        env_info = self.envs.get_env_info() if hasattr(self.envs, "get_env_info") else {}
        env_ep_limit = int(env_info.get("episode_limit", 0))
        if env_ep_limit > 0 and self.episode_length > env_ep_limit:
            print(
                f"[runner] episode_length={self.episode_length} > "
                f"env.episode_limit={env_ep_limit}: a single rollout buffer "
                f"will span multiple episodes with potentially different "
                f"group compositions. This is handled correctly via the "
                f"per-step actor_id_per_step field, but the simpler matched "
                f"setting (episode_length == episode_limit) is recommended.",
                flush=True,
            )

        ppo = args["ppo"]
        self.gamma: float = float(ppo["gamma"])
        self.gae_lambda: float = float(ppo["gae_lambda"])

        log = args.get("log", {})
        self.log_interval: int = int(log.get("log_interval", 5))
        self.save_interval: int = int(log.get("save_interval", 100))
        self.save_dir: Path = Path(log.get("save_dir", "results/runs/default"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Whether we are running on the synthetic SMACv2 dummy backend.
        self._using_dummy_env: bool = bool(env_info.get("using_dummy", False))
        if self._using_dummy_env:
            print(
                "[runner] WARNING: running against a SYNTHETIC SMACv2 dummy "
                "env. All metrics produced will reflect random reward, NOT "
                "real algorithm performance. Each log entry is stamped with "
                "'using_dummy': true.",
                flush=True,
            )

        self._returns: deque = deque(maxlen=100)
        self._wins: deque = deque(maxlen=100)
        self._t0 = time.time()

        # Filled by _collect_rollouts at the very end of the loop. Used by
        # _bootstrap_last_value to compute V(s_T) for GAE.
        self._last_state_after_step: Optional[np.ndarray] = None
        self._last_step_was_terminal: bool = False

    # ----------------------------------------------------------------- main

    def run(self) -> None:
        self._lazy_init_buffer()
        steps_per_iter = self.episode_length * self.n_threads
        num_iters = max(self.num_env_steps // steps_per_iter, 1)

        for it in range(num_iters):
            self._collect_rollouts()
            last_value = self._bootstrap_last_value()
            self.buffer.compute_gae(last_value, self.gamma, self.gae_lambda)

            train_info = self.algo.train(self.buffer, self.buffer.advantages)

            if it % self.log_interval == 0:
                self._log(it, train_info)
            if it % self.save_interval == 0 and it > 0:
                self._save(it)
            self.buffer.after_update()

    # -------------------------------------------------------------- internals

    def _lazy_init_buffer(self) -> None:
        info = self.envs.get_env_info() if hasattr(self.envs, "get_env_info") else {}
        obs_dim = info.get("obs_shape", 64)
        state_dim = info.get("state_shape", obs_dim * self.num_agents)
        action_dim = 1                          # discrete: stored as int
        self.buffer = OnPolicyBuffer(
            size=self.episode_length,
            n_threads=self.n_threads,
            n_agents=self.num_agents,
            obs_dim=obs_dim,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device,
        )

    def _read_avail_actions(self) -> np.ndarray | None:
        """Return ``(n_agents, n_actions)`` avail mask if the env provides it."""
        if not hasattr(self.envs, "get_avail_actions"):
            return None
        return np.asarray(self.envs.get_avail_actions(), dtype=np.float32)

    def _read_alive_mask(self) -> np.ndarray | None:
        """Return ``(n_agents,)`` alive mask if the env provides it."""
        if not hasattr(self.envs, "get_alive_mask"):
            return None
        return np.asarray(self.envs.get_alive_mask(), dtype=np.float32)

    def _read_actor_ids(self) -> np.ndarray | None:
        """Return per-agent canonical actor ids if the env provides them."""
        if hasattr(self.envs, "get_canonical_type_ids"):
            return np.asarray(self.envs.get_canonical_type_ids(), dtype=np.int64)
        # Fallback: derive from current GroupAssignment exposed by the algo,
        # which is constant across the rollout in this branch (no dynamic
        # group tracking is needed because there are no resets that change
        # the partition).
        if self.algo is not None and hasattr(self.algo, "groups"):
            ga = self.algo.groups
            if ga is not None and hasattr(ga, "group_of_agent"):
                if getattr(self.algo, "dynamic_groups", False) and hasattr(self.algo, "canonical_group_labels"):
                    # Map local group ids → canonical actor ids.
                    canonical = list(self.algo.canonical_group_labels)
                    return np.array([
                        canonical.index(ga.group_labels[g])
                        for g in ga.group_of_agent
                    ], dtype=np.int64)
                return np.asarray(ga.group_of_agent, dtype=np.int64)
        return None

    @torch.no_grad()
    def _collect_rollouts(self) -> None:
        """Collect ``episode_length`` steps from a single (non-VecEnv) env.

        For envs that change their team composition between episodes
        (e.g. SMACv2), we forward the new GroupAssignment to the algorithm via
        ``algo.set_groups`` whenever the env signals a partition change, AND
        we store the per-step canonical actor ids so the algorithm can route
        each transition to the correct actor at training time regardless of
        any reset that happened during the rollout.
        """
        if not hasattr(self.envs, "step"):
            return  # tests can pre-fill the buffer instead

        ep_ret = 0.0
        obs, state, groups = self.envs.reset()
        self._maybe_update_groups(groups)

        for t in range(self.episode_length):
            obs_t = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)   # (1, N, O)
            state_t = torch.as_tensor(state, device=self.device).float().unsqueeze(0)

            # Read SMAC-family per-step fields BEFORE stepping the env.
            avail = self._read_avail_actions()
            alive = self._read_alive_mask()
            actor_ids = self._read_actor_ids()

            avail_t: torch.Tensor | None = None
            if avail is not None:
                avail_t = torch.as_tensor(avail, device=self.device).float().unsqueeze(0)

            # Sample joint action with avail mask if the env supplies one.
            actions, log_probs = self.algo.select_actions(obs_t, available_actions=avail_t)

            # Use algo.value() which auto-denormalises if a value normaliser is in use.
            if hasattr(self.algo, "value"):
                value = self.algo.value(state_t).detach().cpu().numpy().reshape(self.n_threads)
            else:
                value = self.algo.critic(state_t).detach().cpu().numpy().squeeze(-1).reshape(self.n_threads)

            # Branch on action type. Discrete (SMAC family): one int per
            # agent. Continuous (MPE with continuous_actions=True): a
            # length-action_dim float vector per agent.
            is_discrete = bool(getattr(self.algo, "discrete", True))
            if is_discrete:
                joint_action_np = (
                    actions.detach().cpu().numpy()
                    .reshape(self.num_agents).astype(np.int64)
                )
                joint_action_for_env = [int(a) for a in joint_action_np]
                actions_for_buffer = joint_action_np.reshape(
                    1, self.num_agents, 1
                ).astype(np.float32)
            else:
                joint_action_np = (
                    actions.detach().cpu().numpy()
                    .reshape(self.num_agents, -1).astype(np.float32)
                )
                joint_action_for_env = joint_action_np
                actions_for_buffer = joint_action_np.reshape(
                    1, self.num_agents, -1
                )

            obs_next, state_next, reward, done, info = self.envs.step(joint_action_for_env)
            ep_ret += reward

            # Insert into buffer.
            buffer_kwargs = dict(
                obs=obs[None, :, :],
                state=state[None, :],
                next_state=state_next[None, :],
                actions=actions_for_buffer,
                log_probs=log_probs.detach().cpu().numpy().reshape(1, self.num_agents),
                values=value,
                rewards=np.array([reward], dtype=np.float32),
                dones=np.array([float(done)], dtype=np.float32),
            )
            if avail is not None and getattr(avail, "ndim", 0) >= 2:
                buffer_kwargs["available_actions"] = avail[None, :, :]
            if alive is not None and getattr(alive, "ndim", 0) >= 1:
                buffer_kwargs["active_masks"] = alive[None, :]
            if actor_ids is not None and getattr(actor_ids, "ndim", 0) >= 1:
                buffer_kwargs["actor_id_per_step"] = actor_ids[None, :]
            self.buffer.insert(**buffer_kwargs)

            # Track the post-step state for accurate GAE bootstrap.
            self._last_state_after_step = np.asarray(state_next, dtype=np.float32)
            self._last_step_was_terminal = bool(done)

            obs, state = obs_next, state_next
            if done:
                self._returns.append(ep_ret)
                self._wins.append(float(info.get("battle_won", False)))
                ep_ret = 0.0
                obs, state, groups = self.envs.reset()
                self._maybe_update_groups(groups)

    def _maybe_update_groups(self, groups) -> None:
        """Forward a fresh GroupAssignment to the algorithm if it accepts one.

        Algorithms in dynamic-group mode (TyPPO/TySAC built with
        ``n_canonical_groups``) consume this to re-route agents to the actor
        whose canonical label matches their unit type. Algorithms without
        ``set_groups`` (HARL family, parameter-shared baselines) are simply
        skipped.
        """
        if groups is None or self.algo is None:
            return
        if not hasattr(self.algo, "set_groups"):
            return
        try:
            self.algo.set_groups(groups)
            # Different algorithm families spell the agent-count attribute
            # differently: TyPPO/TySAC expose ``num_agents``; HAPPO/MAPPO/HATRPO
            # expose ``n_agents``. Read whichever is present, fall back to
            # the count we computed from the group assignment.
            self.num_agents = getattr(
                self.algo, "num_agents",
                getattr(self.algo, "n_agents", self.num_agents),
            )
        except (ValueError, TypeError):
            # Static-mode algos with an n_groups mismatch arrive here; benign.
            pass

    @torch.no_grad()
    def _bootstrap_last_value(self) -> np.ndarray:
        """V(s_T) for GAE bootstrapping.

        Uses the post-step state recorded by ``_collect_rollouts``. If that
        step was terminal, the bootstrap value is unused anyway (multiplied
        by next_nonterminal=0 inside compute_gae), but we still pass V(s_T)
        rather than zero to keep the function well-defined.

        Falls back to zeros only when no critic is wired (tests).
        """
        if self.algo is None or self._last_state_after_step is None:
            return np.zeros((self.n_threads,), dtype=np.float32)
        s = (
            torch.as_tensor(self._last_state_after_step, device=self.device)
            .float()
            .unsqueeze(0)
        )
        if hasattr(self.algo, "value"):
            v = self.algo.value(s).detach().cpu().numpy().reshape(self.n_threads)
        else:
            v = self.algo.critic(s).detach().cpu().numpy().squeeze(-1).reshape(self.n_threads)
        return v.astype(np.float32)

    def _log(self, it: int, info: Dict[str, Any]) -> None:
        elapsed = time.time() - self._t0
        env_steps = (it + 1) * self.episode_length * self.n_threads
        payload: Dict[str, Any] = {
            "iter": it,
            "env_steps": env_steps,
            "elapsed_s": elapsed,
            "fps": int(env_steps / max(elapsed, 1e-3)),
            "ep_return_mean": float(np.mean(self._returns)) if self._returns else float("nan"),
            "win_rate": float(np.mean(self._wins)) if self._wins else float("nan"),
            "using_dummy": self._using_dummy_env,
        }
        # Per-group losses get flattened.
        for gid, losses in info.get("group_losses", {}).items():
            for k, v in losses.items():
                payload[f"g{gid}/{k}"] = v

        with open(self.save_dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        msg = [f"[iter {it}]", f"steps={env_steps}",
               f"ret={payload['ep_return_mean']:.3f}",
               f"win={payload['win_rate']:.3f}"]
        if self._using_dummy_env:
            msg.append("[DUMMY]")
        for gid, losses in info.get("group_losses", {}).items():
            msg.append(
                f"g{gid}: pl={losses['policy_loss']:+.4f} "
                f"vl={losses['value_loss']:.4f} ent={losses['entropy']:.3f}"
            )
        print(" ".join(msg), flush=True)

    def _save(self, it: int) -> None:
        path = self.save_dir / f"ckpt_iter{it}.pt"
        torch.save(self.algo.state_dict(), path)
