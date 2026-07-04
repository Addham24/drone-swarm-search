"""
EPyMARL Environment Adapter for DSSE Coverage Drone Swarm
==========================================================
Wraps the PettingZoo-based DSSE environment (with all custom wrappers)
into EPyMARL's MultiAgentEnv interface.

EPyMARL expects:
  - Flat numpy observation vectors (not tuples)
  - List-indexed agents (not dict-keyed)
  - get_state() for centralized critic training (QMIX/QPLEX/COMA)
  - get_avail_actions() for action masking

This adapter handles:
  1. Flattening the Tuple(positions, matrix) obs into a single vector
  2. Converting dict-keyed PettingZoo API to list-indexed EPyMARL API
  3. Providing a global state (concatenation of all agent obs) for CTDE
  4. Managing agent attrition (dead agents get zeroed obs but remain in the list)
"""

import os
import numpy as np

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from global_reward_wrapper import GlobalRewardWrapper


class DSSEMultiAgentEnv:
    """
    EPyMARL-compatible MultiAgentEnv adapter for the DSSE Coverage environment.

    Observation structure (per agent, flattened):
        [positions_vector (22 floats) | coverage_matrix (25*25=625 floats)]
        Total obs_size = 22 + 625 = 647

    State structure (global, for centralized critic):
        Concatenation of all agent observations: n_agents * obs_size

    Action space: 9 discrete actions (8 directions + search/idle)
    """

    def __init__(
        self,
        # Environment parameters
        n_agents=4,
        grid_size=25,
        timestep_limit=750,
        # Battery parameters
        max_battery=125,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=0.0005,
        # Reward mixing
        mixing_alpha=0.5,
        # Self-healing compensation
        compensation_bonus=2.5,
        compensation_penalty=-0.5,
        compensation_horizon=100,
        # EPyMARL interface
        common_reward=True,
        reward_scalarisation="sum",
        seed=None,
        **kwargs,
    ):
        self.n_agents = n_agents
        self._grid_size = grid_size
        self.episode_limit = timestep_limit

        # Store params for env recreation
        self._env_params = {
            "n_agents": n_agents,
            "grid_size": grid_size,
            "timestep_limit": timestep_limit,
            "max_battery": max_battery,
            "depletion_rate": depletion_rate,
            "charge_rate": charge_rate,
            "fault_prob": fault_prob,
            "mixing_alpha": mixing_alpha,
            "compensation_bonus": compensation_bonus,
            "compensation_penalty": compensation_penalty,
            "compensation_horizon": compensation_horizon,
        }

        self.common_reward = common_reward
        self.reward_scalarisation = reward_scalarisation
        self._seed = seed

        # Build the wrapped PettingZoo environment
        self._env = self._build_env()

        # Cache agent ordering (PettingZoo uses string keys like "drone_0")
        self._agents = list(self._env.possible_agents)
        assert len(self._agents) == self.n_agents

        # Calculate observation dimensions
        sample_obs_space = self._env.observation_space(self._agents[0])
        self._positions_size = sample_obs_space[0].shape[0]  # e.g. 22
        self._matrix_size = int(np.prod(sample_obs_space[1].shape))  # e.g. 625
        self._obs_size = self._positions_size + self._matrix_size

        # Action space: 9 discrete actions
        self._n_actions = self._env.action_space(self._agents[0]).n

        # Internal state
        self._obs = None  # List of flat numpy arrays
        self._step_count = 0

    def _build_env(self):
        """Constructs the full DSSE environment with all wrappers."""
        p = self._env_params
        matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")

        env = CoverageDroneSwarmSearch(
            timestep_limit=p["timestep_limit"],
            drone_amount=p["n_agents"],
            prob_matrix_path=matrix_path,
        )
        env.reward_scheme = {
            "default": -0.1,
            "exceed_timestep": 0.0,
            "search_cell": 5.0,
            "done": 500.0,
            "reward_poc": 0.0,
        }

        env = AllPositionsWrapper(env)
        env = BatteryStationWrapper(
            env,
            max_battery=p["max_battery"],
            depletion_rate=p["depletion_rate"],
            charge_rate=p["charge_rate"],
            fault_prob=p["fault_prob"],
        )

        # Set compensation parameters
        env.COMPENSATION_BONUS = p["compensation_bonus"]
        env.COMPENSATION_PENALTY = p["compensation_penalty"]
        env.COMPENSATION_HORIZON = p["compensation_horizon"]

        env = GlobalRewardWrapper(env, mixing_alpha=p["mixing_alpha"])

        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
        env = RetainDronePosWrapper(env, positions)

        return env

    def _flatten_obs(self, obs_dict):
        """Convert PettingZoo dict of Tuple(positions, matrix) to list of flat vectors."""
        flat_obs = []
        for agent in self._agents:
            if agent in obs_dict:
                positions, matrix = obs_dict[agent]
                flat = np.concatenate([
                    positions.astype(np.float32).flatten(),
                    matrix.astype(np.float32).flatten(),
                ])
                flat_obs.append(flat)
            else:
                # Dead or removed agent gets zero observation
                flat_obs.append(np.zeros(self._obs_size, dtype=np.float32))
        return flat_obs

    def reset(self, seed=None, options=None):
        """Returns initial observations and info."""
        obs_dict, info_dict = self._env.reset()
        self._obs = self._flatten_obs(obs_dict)
        self._step_count = 0
        return self._obs, {}

    def step(self, actions):
        """
        Takes a list of integer actions (one per agent).
        Returns: obs_list, reward, terminated, truncated, info
        """
        self._step_count += 1

        # Convert list of actions to PettingZoo dict
        action_dict = {}
        for i, agent in enumerate(self._agents):
            action_dict[agent] = int(actions[i])

        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self._env.step(action_dict)

        # Flatten observations
        self._obs = self._flatten_obs(obs_dict)

        # Convert rewards to list
        rewards = [reward_dict.get(agent, 0.0) for agent in self._agents]

        # Handle reward aggregation for EPyMARL
        if self.common_reward:
            if self.reward_scalarisation == "sum":
                reward = float(sum(rewards))
            elif self.reward_scalarisation == "mean":
                reward = float(sum(rewards) / len(rewards))
            else:
                reward = float(sum(rewards))
        else:
            reward = rewards

        # Check termination
        terminated = all(term_dict.get(agent, False) for agent in self._agents)
        truncated = all(trunc_dict.get(agent, False) for agent in self._agents)

        # Extract coverage rate from info if available
        info = {}
        for agent in self._agents:
            agent_info = info_dict.get(agent, {})
            if isinstance(agent_info, dict) and "coverage_rate" in agent_info:
                info["coverage_rate"] = agent_info["coverage_rate"]
                break

        return self._obs, reward, terminated, truncated, info

    # ─── EPyMARL MultiAgentEnv Interface ─────────────────────────────────

    def get_obs(self):
        """Returns all agent observations in a list."""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id (integer index)."""
        return self._obs[agent_id]

    def get_obs_size(self):
        """Returns the flat observation size per agent."""
        return self._obs_size

    def get_state(self):
        """
        Returns the global state for centralized training (QMIX/QPLEX/COMA).
        Concatenation of all agent observations into a single flat vector.
        """
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """Returns the global state size."""
        return self.n_agents * self._obs_size

    def get_avail_actions(self):
        """Returns available actions for all agents as a list of lists."""
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """Returns available actions for a specific agent. All 9 actions always available."""
        return [1] * self._n_actions

    def get_total_actions(self):
        """Returns the total number of discrete actions per agent."""
        return self._n_actions

    def get_env_info(self):
        """Returns environment info dict required by EPyMARL."""
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit,
        }

    def render(self):
        pass

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._seed = seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}
