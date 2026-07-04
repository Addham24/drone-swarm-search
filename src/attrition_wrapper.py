import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class AttritionWrapper(BaseParallelWrapper):
    """
    Simulates agent attrition (drone loss) during an episode.

    Dead drones are frozen in place — their actions are ignored, their
    positions remain unchanged, and their observations are zeroed out.
    This keeps the agent count fixed for RLlib compatibility.

    Args:
        env: The PettingZoo parallel environment to wrap.
        attrition_prob: Per-step probability that each alive drone dies (default 0.005).
        min_drones: Minimum number of drones that must stay alive (default 1).
    """

    def __init__(self, env, attrition_prob=0.001, min_drones=1):
        super().__init__(env)
        self.attrition_prob = attrition_prob
        self.min_drones = min_drones
        # Track which drones are alive — keyed by agent name
        self.alive = {}
        # Use a local random generator to ensure independent rolls per agent
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        # All drones start alive
        self.alive = {agent: True for agent in self.env.possible_agents}
        # Inject alive status into infos
        for agent in infos:
            infos[agent]["alive"] = True
            infos[agent]["attrition_event"] = False
        return obs, infos

    def step(self, actions):
        # --- Roll for attrition BEFORE stepping ---
        n_alive = sum(1 for v in self.alive.values() if v)
        new_deaths = []
        for agent in self.env.agents:
            if self.alive.get(agent, False) and n_alive > self.min_drones:
                if self.rng.random() < self.attrition_prob:
                    self.alive[agent] = False
                    n_alive -= 1
                    new_deaths.append(agent)

        # --- Override actions for dead drones (force SEARCH/IDLE = 8) ---
        filtered_actions = {}
        for agent in actions:
            if self.alive.get(agent, False):
                filtered_actions[agent] = actions[agent]
            else:
                # Action 8 = SEARCH/IDLE (stay in place) in DSSE
                filtered_actions[agent] = 8
        
        # Step the underlying environment
        obs, rewards, terminations, truncations, infos = self.env.step(filtered_actions)

        # --- Zero out observations for dead drones ---
        for agent in obs:
            if not self.alive.get(agent, False):
                positions, matrix = obs[agent]
                obs[agent] = (
                    np.zeros_like(positions),
                    np.zeros_like(matrix),
                )
                # Dead drones get zero reward
                rewards[agent] = 0.0

        # --- Inject status into infos ---
        for agent in infos:
            if isinstance(infos[agent], dict):
                infos[agent]["alive"] = self.alive.get(agent, False)
                infos[agent]["attrition_event"] = agent in new_deaths
            else:
                # If infos[agent] is not a dict (e.g. after termination), skip
                pass

        return obs, rewards, terminations, truncations, infos
