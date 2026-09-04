"""
Obstacle Airspace Wrapper for PettingZoo Parallel Environments
================================================================
Implements physical obstacle collision mechanics for DSSE drone swarm environments.

Obstacle Physics:
  - Impassable Movement Blocking: Drones cannot enter obstacle cells.
    Attempts to move into obstacle cell (y', x') result in position retention (y, x).
  - Collision Penalty: -2.0 reward for attempting to enter an obstacle.
  - Observation Encoding: Obstacle cells are encoded as -1.0 in the probability matrix,
    signaling the CNN actor that the cell is an impassable wall.
  - Adjusted Coverage Rate: Coverage is computed over accessible free cells only:
    Coverage(%) = searched_free_cells / (total_cells - n_obstacles) * 100

Visual Rendering (Pygame):
  - Obstacle cells rendered as Dark Charcoal blocks (RGB: [40, 40, 40]).
"""

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class ObstacleAirspaceWrapper(BaseParallelWrapper):
    """
    Wraps a DSSE CoverageDroneSwarmSearch PettingZoo parallel environment
    with physical obstacle collision mechanics.
    
    Args:
        env: The PettingZoo parallel environment to wrap (after AllPositionsWrapper).
        obstacle_mask: numpy array of shape (H, W) where 1 = obstacle, 0 = free.
        collision_penalty: Reward penalty for attempting to move into an obstacle cell.
    """
    
    OBSTACLE_OBS_VALUE = -1.0  # Value used to encode obstacles in observation matrix
    CHARCOAL_RGB = (40, 40, 40)  # Pygame rendering color for obstacles
    
    def __init__(self, env, obstacle_mask, collision_penalty=-2.0):
        super().__init__(env)
        self.obstacle_mask = obstacle_mask.copy()
        self.collision_penalty = collision_penalty
        self.n_obstacles = int(self.obstacle_mask.sum())
        
        # Compute accessible free cells for adjusted coverage
        grid_h, grid_w = self.obstacle_mask.shape
        self.total_cells = grid_h * grid_w
        self.free_cells = self.total_cells - self.n_obstacles
        
        # Build set of obstacle coordinates for fast lookup: (x, y) format matching DSSE
        self.obstacle_coords = set()
        for y in range(grid_h):
            for x in range(grid_w):
                if self.obstacle_mask[y, x] == 1:
                    self.obstacle_coords.add((x, y))
        
        # Track which free cells have been searched for adjusted coverage
        self.searched_free_cells = set()
    
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.searched_free_cells = set()
        
        # Remove obstacle cells from DSSE's not_seen_states so they don't count toward coverage
        base_env = self._get_base_env()
        if hasattr(base_env, 'not_seen_states'):
            base_env.not_seen_states -= self.obstacle_coords
        if hasattr(base_env, 'all_states'):
            base_env.all_states -= self.obstacle_coords
        
        # Patch Pygame renderer if render_mode="human" is enabled
        if hasattr(base_env, 'pygame_renderer') and base_env.pygame_renderer is not None:
            self._patch_pygame_renderer(base_env.pygame_renderer)

        # Encode obstacles in observation matrices
        obs = self._encode_obstacles_in_obs(obs)
        
        # Inject adjusted coverage into infos
        for agent in infos:
            if isinstance(infos[agent], dict):
                infos[agent] = dict(infos[agent])
                infos[agent]["coverage_rate"] = 0.0
                infos[agent]["n_obstacles"] = self.n_obstacles
                infos[agent]["free_cells"] = self.free_cells
        
        return obs, infos

    def _patch_pygame_renderer(self, renderer):
        """Monkey-patch Pygame renderer so obstacle building cells draw in Dark Graphite (40, 40, 45)."""
        obstacle_mask = self.obstacle_mask
        grid_h, grid_w = obstacle_mask.shape

        def custom_draw():
            import pygame
            renderer.clock.tick(renderer.FPS)
            renderer.screen.fill((0, 0, 0))
            matrix = renderer.probability_matrix.get_matrix()
            max_matrix = matrix.max() if matrix.max() != 0.0 else 1.0

            for counter_x, x in enumerate(np.arange(10, renderer.window_size + 10, renderer.block_size)):
                for counter_y, y in enumerate(np.arange(10, renderer.window_size + 10, renderer.block_size)):
                    if counter_x < grid_w and counter_y < grid_h:
                        rectangle = renderer.get_position_rectangle((counter_x, counter_y))
                        if counter_x == 0 and counter_y == 0:
                            # Charging Base Station at (0,0) - Permanent Bright Emerald Green!
                            color = (40, 220, 100)
                        elif obstacle_mask[counter_y, counter_x] == 1:
                            # Dark Graphite / Charcoal for building obstacles!
                            color = (40, 40, 45)
                        else:
                            prob = matrix[counter_y][counter_x]
                            norm_prob = prob / max_matrix
                            color = renderer.compute_cell_color(norm_prob)
                        
                        pygame.draw.rect(renderer.screen, color, rectangle)
                        if renderer.render_grid:
                            pygame.draw.rect(renderer.screen, (0, 0, 0), rectangle, 2)

        renderer.draw = custom_draw
    
    def step(self, actions):
        # Intercept actions: block movement into obstacle cells
        base_env = self._get_base_env()
        blocked_agents = set()
        
        if hasattr(base_env, 'agents_positions'):
            for i, agent in enumerate(self.env.possible_agents):
                if agent not in actions:
                    continue
                if i < len(base_env.agents_positions):
                    current_pos = base_env.agents_positions[i]  # (x, y) in DSSE
                    new_pos = base_env.move_drone(current_pos, actions[agent])
                    
                    if new_pos in self.obstacle_coords:
                        # Block: force action to SEARCH (8) = stay in place
                        actions[agent] = 8
                        blocked_agents.add(agent)
        
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Apply collision penalties for blocked agents
        for agent in blocked_agents:
            if agent in rewards:
                rewards[agent] += self.collision_penalty
        
        # Track searched free cells for adjusted coverage
        if hasattr(base_env, 'seen_states'):
            self.searched_free_cells = base_env.seen_states - self.obstacle_coords
        
        # Compute adjusted coverage rate
        adjusted_coverage = len(self.searched_free_cells) / self.free_cells if self.free_cells > 0 else 0.0
        
        # Inject adjusted coverage into infos
        for agent in infos:
            if isinstance(infos[agent], dict):
                infos[agent] = dict(infos[agent])
                infos[agent]["coverage_rate"] = adjusted_coverage
                infos[agent]["n_obstacles"] = self.n_obstacles
                infos[agent]["free_cells"] = self.free_cells
                if agent in blocked_agents:
                    infos[agent]["obstacle_collision"] = True
        
        # Encode obstacles in observation matrices
        obs = self._encode_obstacles_in_obs(obs)
        
        return obs, rewards, terminations, truncations, infos
    
    def _encode_obstacles_in_obs(self, obs):
        """Encode obstacle cells as -1.0 in the probability matrix observations."""
        for agent in obs:
            if isinstance(obs[agent], tuple) and len(obs[agent]) == 2:
                positions, matrix = obs[agent]
                # Clone matrix to avoid modifying shared references
                matrix = matrix.copy()
                matrix[self.obstacle_mask == 1] = self.OBSTACLE_OBS_VALUE
                obs[agent] = (positions, matrix)
        return obs
    
    def _get_base_env(self):
        """Walk the wrapper chain to get the underlying DSSE environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'unwrapped'):
            return env.unwrapped
        return env
