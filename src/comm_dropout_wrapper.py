import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class CommDropoutWrapper(BaseParallelWrapper):
    """
    Simulates communication dropout between drones.

    When a drone is in dropout, the positions of all OTHER drones in its
    observation vector are zeroed out. The drone can still see its own
    position and the full probability matrix.

    Works with AllPositionsWrapper which produces observations of the form:
        (positions_array, probability_matrix)
    where positions_array = [y_self, x_self, y_other1, x_other1, ...].

    Args:
        env: The PettingZoo parallel environment to wrap.
        dropout_prob: Per-step probability a drone enters dropout (default 0.1).
        dropout_duration: How many steps a dropout lasts (default 5).
    """

    def __init__(self, env, dropout_prob=0.02, dropout_duration=5):
        super().__init__(env)
        self.dropout_prob = dropout_prob
        self.dropout_duration = dropout_duration
        # Tracks remaining dropout steps for each agent (0 = normal comms)
        self.dropout_remaining = {}
        # Use a local random generator to ensure independent rolls per agent
        self.rng = np.random.RandomState()

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        # All comms start normal
        self.dropout_remaining = {agent: 0 for agent in self.env.possible_agents}
        # Inject comms status into infos
        for agent in infos:
            infos[agent]["comm_dropout"] = False
        return obs, infos

    def step(self, actions):
        # Step the underlying environment first
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # --- Roll for new dropouts and tick existing ones ---
        new_dropouts = []
        restored_agents = []
        for agent in self.env.possible_agents:
            if self.dropout_remaining.get(agent, 0) > 0:
                # Currently in dropout — tick down
                self.dropout_remaining[agent] -= 1
                if self.dropout_remaining[agent] == 0:
                    restored_agents.append(agent)
            else:
                # Roll for a new dropout
                if self.rng.random() < self.dropout_prob:
                    self.dropout_remaining[agent] = self.dropout_duration
                    new_dropouts.append(agent)

        # --- Mask teammate positions for drones in dropout ---
        for agent in obs:
            if self.dropout_remaining.get(agent, 0) > 0:
                positions, matrix = obs[agent]
                # positions layout from AllPositionsWrapper:
                # [y_self, x_self, y_other1, x_other1, y_other2, x_other2, ...]
                # Keep first 2 values (own position), zero out the rest
                masked_positions = positions.copy()
                masked_positions[2:] = 0
                obs[agent] = (masked_positions, matrix)

        # --- Inject comms status into infos ---
        for agent in infos:
            if isinstance(infos[agent], dict):
                is_in_dropout = self.dropout_remaining.get(agent, 0) > 0
                infos[agent]["comm_dropout"] = is_in_dropout
                infos[agent]["comm_dropout_new"] = agent in new_dropouts
                infos[agent]["comm_restored"] = agent in restored_agents
                infos[agent]["comm_dropout_remaining"] = self.dropout_remaining.get(agent, 0)

        return obs, rewards, terminations, truncations, infos
