import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

class TrackingRewardWrapper(BaseParallelWrapper):
    """
    PettingZoo wrapper for DSSE DroneSwarmSearch.
    Provides dense reward shaping for tracking moving targets:
    1. Occupying high-probability cells: +5.0 * P(x, y)
    2. Moving toward higher-probability cells: +1.0
    3. Moving away from higher-probability cells: -0.5
    4. Physical target discovery jackpot: +500.0
    5. Step penalty: -0.1
    """

    def __init__(self, env, prob_scale=5.0, step_penalty=-0.1, discovery_bonus=500.0):
        super().__init__(env)
        self.prob_scale = prob_scale
        self.step_penalty = step_penalty
        self.discovery_bonus = discovery_bonus
        self.prev_prob = {}

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        self.prev_prob = {}
        for agent, o in obs.items():
            if isinstance(o, tuple) and len(o) == 2:
                matrix = o[1]
                pos_vec = o[0]
                drone_y, drone_x = int(pos_vec[0]), int(pos_vec[1])
                if 0 <= drone_y < matrix.shape[0] and 0 <= drone_x < matrix.shape[1]:
                    self.prev_prob[agent] = float(matrix[drone_y, drone_x])
                else:
                    self.prev_prob[agent] = 0.0
            else:
                self.prev_prob[agent] = 0.0
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        for agent in self.env.agents:
            if agent not in obs:
                continue

            agent_obs = obs[agent]
            # Expecting tuple(position/battery/crash vector, 25x25 matrix)
            if isinstance(agent_obs, tuple) and len(agent_obs) == 2:
                pos_vec = agent_obs[0]
                matrix = agent_obs[1]
                
                drone_y, drone_x = int(pos_vec[0]), int(pos_vec[1])
                
                # Check bounds
                if 0 <= drone_y < matrix.shape[0] and 0 <= drone_x < matrix.shape[1]:
                    curr_prob = float(matrix[drone_y, drone_x])
                else:
                    curr_prob = 0.0

                # 1. Probability proportional reward
                shaped_reward = self.prob_scale * curr_prob + self.step_penalty

                # 2. Gradient following bonus
                prev_p = self.prev_prob.get(agent, curr_prob)
                if curr_prob > prev_p + 1e-6:
                    shaped_reward += 1.0
                elif curr_prob < prev_p - 1e-6:
                    shaped_reward -= 0.5

                self.prev_prob[agent] = curr_prob

                # 3. Discovery Jackpot
                # DSSE native gives reward 1.0 on search_and_find
                if rewards.get(agent, 0.0) >= 1.0:
                    shaped_reward += self.discovery_bonus

                rewards[agent] = float(shaped_reward)

        return obs, rewards, terminations, truncations, infos
