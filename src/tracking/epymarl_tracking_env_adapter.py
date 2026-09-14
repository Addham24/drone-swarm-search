"""
EPyMARL Environment Adapter for DSSE Dynamic Target Tracking
==============================================================
Wraps the PettingZoo-based DSSE tracking environment (with all wrappers)
into EPyMARL's MultiAgentEnv interface.

Observation structure (per agent, flattened):
    [positions_vector (22 floats) | matrix (25*25=625 floats)]
    Total obs_size = 22 + 625 = 647

State structure (global, for CTDE):
    Concatenation of all agent observations: n_agents * obs_size = 4 * 647 = 2588
"""

import os
import sys
import numpy as np

# Ensure src directory is on sys.path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tracking.tracking_env_creator import make_tracking_env


class DSSETrackingMultiAgentEnv:
    """
    EPyMARL-compatible MultiAgentEnv adapter for DSSE Dynamic Tracking.
    """

    def __init__(
        self,
        n_agents=4,
        grid_size=25,
        person_amount=1,
        timestep_limit=750,
        max_battery=125,
        fault_prob=0.0005,
        disable_self_healing=False,
        drift_speed=1.0,
        seed=None,
        **kwargs,
    ):
        self.n_agents = n_agents
        self.grid_size = grid_size
        self.episode_limit = timestep_limit
        self.disable_self_healing = disable_self_healing
        self.drift_speed = drift_speed

        is_self_heal = not disable_self_healing

        self.env = make_tracking_env(
            grid_size=grid_size,
            drone_amount=n_agents,
            person_amount=person_amount,
            person_initial_position=(grid_size // 2, grid_size // 2),
            timestep_limit=timestep_limit,
            max_battery=max_battery,
            fault_prob=fault_prob,
            is_self_heal=is_self_heal,
            drift_speed=drift_speed,
        )

        self.agents = self.env.possible_agents
        self._obs_vec_len = 22
        self._matrix_len = grid_size * grid_size
        self._single_obs_size = self._obs_vec_len + self._matrix_len

        self._current_obs = {}
        self.steps_count = 0

    def _flatten_obs(self, agent_obs):
        """Converts Tuple(pos_vector_22, matrix_25x25) -> flat array (647,)."""
        if agent_obs is None:
            return np.zeros(self._single_obs_size, dtype=np.float32)

        if isinstance(agent_obs, tuple) and len(agent_obs) == 2:
            pos_vec = agent_obs[0].astype(np.float32)
            matrix = agent_obs[1].astype(np.float32).flatten()
            return np.concatenate([pos_vec, matrix])

        return np.zeros(self._single_obs_size, dtype=np.float32)

    def reset(self, **kwargs):
        self.steps_count = 0
        obs_dict, infos = self.env.reset(**kwargs)
        self._current_obs = obs_dict
        return self.get_obs(), self.get_state()

    def step(self, actions):
        self.steps_count += 1
        
        if hasattr(actions, "tolist"):
            actions = actions.tolist()

        if isinstance(actions, (list, tuple)):
            action_dict = {
                agent: int(actions[i])
                for i, agent in enumerate(self.agents)
                if i < len(actions)
            }
        elif isinstance(actions, dict):
            action_dict = actions
        else:
            action_dict = {agent: int(actions) for agent in self.agents}

        obs_dict, rewards_dict, term_dict, trunc_dict, infos = self.env.step(action_dict)
        self._current_obs = obs_dict

        rewards = [rewards_dict.get(agent, 0.0) for agent in self.agents]
        reward = float(sum(rewards))

        all_term = all(term_dict.get(agent, False) for agent in self.agents)
        all_trunc = self.steps_count >= self.episode_limit
        terminated = all_term
        truncated = all_trunc

        info = {
            "target_found": any(r >= 1.0 for r in rewards_dict.values())
        }

        return self.get_obs(), reward, terminated, truncated, info

    def get_obs(self):
        obs_list = []
        for agent in self.agents:
            if agent in self._current_obs:
                obs_list.append(self._flatten_obs(self._current_obs[agent]))
            else:
                obs_list.append(np.zeros(self._single_obs_size, dtype=np.float32))
        return obs_list

    def get_obs_agent(self, agent_id):
        agent = self.agents[agent_id]
        if agent in self._current_obs:
            return self._flatten_obs(self._current_obs[agent])
        return np.zeros(self._single_obs_size, dtype=np.float32)

    def get_obs_size(self):
        return self._single_obs_size

    def get_state(self):
        return np.concatenate(self.get_obs()).astype(np.float32)

    def get_state_size(self):
        return self.n_agents * self._single_obs_size

    def get_avail_actions(self):
        return [[1] * 9 for _ in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * 9

    def get_total_actions(self):
        return 9

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def close(self):
        self.env.close()

    def render(self):
        pass

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
