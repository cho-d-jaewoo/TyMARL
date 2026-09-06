"""
PettingZoo MPE wrapper, vendored from HARL upstream and rewritten to match
the SMACv2Env interface that TyMARL runners (on/off_policy_group_runner)
expect:

- ``reset()``  → (obs: ndarray, state: ndarray, groups | None)
- ``step(a)``  → (obs: ndarray, state: ndarray, reward: float, done: bool, info: dict)
- ``get_env_info()`` returns the keys TyMARL build_algo reads:
    obs_shape, state_shape, action_dim, n_actions, n_agents, discrete

Reward and done are scalars (cooperative MPE: agents share reward; episode
ends when all agents terminate or are truncated).
"""

import copy
import importlib
import logging

import numpy as np
import supersuit as ss

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def _unpack_reset(reset_result):
    """PettingZoo >=1.23 returns (obs_dict, info_dict); older returns obs_dict."""
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        return reset_result[0]
    return reset_result


class PettingZooMPEEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        del self.args["scenario"]
        self.discrete = True
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] is True
        ):
            self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26
        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        # Initial reset to populate agents/spaces. Handle both legacy and new
        # PettingZoo return signatures.
        _unpack_reset(self.env.reset())
        self.n_agents = self.env.num_agents
        self.agents = list(self.env.agents)
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        self._seed = 0

    # -------- TyMARL-facing API (matches SMACv2Env) -------------------------

    def reset(self):
        """Returns (obs, state, groups). 3rd element is None for static groups."""
        self._seed += 1
        self.cur_step = 0
        obs_dict = _unpack_reset(self.env.reset(seed=self._seed))
        # Refresh agent list (PettingZoo may rebuild it on reset).
        self.agents = list(self.env.agents)
        obs = np.asarray(self.unwrap(obs_dict), dtype=np.float32)
        state = np.asarray(self._global_state(obs_dict), dtype=np.float32)
        # Repeat per-agent state copies so shape matches SMACv2 (n_agents, state_dim).
        return obs, state, None

    def step(self, actions):
        """Returns (obs, state, reward_scalar, done_scalar, info_dict)."""
        actions = np.asarray(actions)
        if self.discrete:
            action_dict = self.wrap(actions.flatten())
        else:
            # Continuous: ensure agent-major (n_agents, action_dim).
            if actions.ndim == 1:
                actions = actions.reshape(self.n_agents, -1)
            action_dict = self.wrap(actions)

        obs, rew, term, trunc, info = self.env.step(action_dict)
        self.cur_step += 1
        if self.cur_step >= self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True

        # Cooperative MPE: agents share reward; episode ends when all done.
        total_reward = float(sum(rew[agent] for agent in self.agents))
        done = all(term[agent] or trunc[agent] for agent in self.agents)
        merged_info = {}
        for a in self.agents:
            for k, v in (info[a] or {}).items():
                merged_info[k] = v

        obs_arr = np.asarray(self.unwrap(obs), dtype=np.float32)
        state_arr = np.asarray(self._global_state(obs), dtype=np.float32)
        return obs_arr, state_arr, total_reward, bool(done), merged_info

    def get_env_info(self):
        """Provide shape info that TyMARL build_algo / runners read by these keys."""
        obs_shape = self.observation_space[0].shape
        obs_dim = int(np.prod(obs_shape))
        share_shape = self.share_observation_space[0].shape
        state_dim = int(np.prod(share_shape))
        a_space = self.action_space[0]
        if hasattr(a_space, "shape") and len(a_space.shape) > 0:
            action_dim = int(a_space.shape[0])  # continuous box
            n_actions = action_dim
        else:
            action_dim = int(a_space.n)         # discrete
            n_actions = int(a_space.n)
        return {
            "n_agents": self.n_agents,
            "obs_shape": obs_dim,
            "state_shape": state_dim,
            "action_dim": action_dim,
            "n_actions": n_actions,
            "discrete": self.discrete,
            "obs_space": self.observation_space,
            "share_obs_space": self.share_observation_space,
            "action_space": self.action_space,
        }

    # -------- internals ------------------------------------------------------

    def _global_state(self, obs_dict):
        """Get global state. Falls back to concatenated agent obs when state()
        isn't supported by the underlying scenario."""
        try:
            s = self.env.state()
            return np.asarray(s, dtype=np.float32)
        except Exception:
            return np.concatenate(
                [np.asarray(obs_dict[a], dtype=np.float32).flatten() for a in self.agents]
            )

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = int(seed)

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        out = []
        for agent in self.agents:
            out.append(d[agent])
        return out

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
