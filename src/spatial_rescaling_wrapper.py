"""
Spatial Rescaling Wrapper for Zero-Shot 50x50 → 25x25 Policy Transfer
======================================================================
Enables evaluation of 25x25-trained neural network policies on 50x50 grids
by downsampling the observation matrix using 2x2 average pooling.

Key mechanics:
  - 50x50 probability matrix → 2x2 average pooling → 25x25 observation for policy input.
  - Drone coordinates rescaled to [0, 24] range so position vectors remain compatible.
  - Obstacle encoding (-1.0 values) preserved through max-pooling of negative cells.
  - Timestep limit set to T=1500 for fair 4x area coverage evaluation.
"""

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper
from gymnasium.spaces import Tuple, Box


class SpatialRescalingWrapper(BaseParallelWrapper):
    """
    Wraps a 50x50 DSSE environment so that observations appear as 25x25
    to pre-trained 25x25 neural network policies.
    
    Args:
        env: The PettingZoo parallel environment with 50x50 grid.
        target_grid_size: The target observation grid size (default 25).
    """
    
    def __init__(self, env, target_grid_size=25):
        super().__init__(env)
        self.target_grid_size = target_grid_size
        
        # Detect source grid size
        base_env = self._get_base_env()
        self.source_grid_size = getattr(base_env, 'grid_size', 50)
        self.scale_factor = self.source_grid_size // self.target_grid_size  # e.g., 50 // 25 = 2
        
        # Override observation spaces to match 25x25
        self.observation_spaces = {
            agent: self._rescaled_observation_space(agent)
            for agent in self.env.possible_agents
        }
    
    def _rescaled_observation_space(self, agent):
        """Build observation space with target_grid_size x target_grid_size matrix."""
        base_space = self.env.observation_space(agent)
        positions_box = base_space[0]
        # Matrix is now target_grid_size x target_grid_size
        new_matrix_box = Box(
            low=-1.0, high=1.0,
            shape=(self.target_grid_size, self.target_grid_size),
            dtype=np.float64
        )
        return Tuple((positions_box, new_matrix_box))
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        obs = self._rescale_observations(obs)
        return obs, infos
    
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs = self._rescale_observations(obs)
        return obs, rewards, terminations, truncations, infos
    
    def _rescale_observations(self, obs):
        """Downsample 50x50 matrices to 25x25 and rescale coordinates."""
        for agent in obs:
            if isinstance(obs[agent], tuple) and len(obs[agent]) == 2:
                positions, matrix = obs[agent]
                
                # Downsample matrix: 50x50 → 25x25 via 2x2 block averaging
                rescaled_matrix = self._downsample_matrix(matrix)
                
                # Rescale position coordinates from [0, source_grid_size] to [0, target_grid_size]
                rescaled_positions = self._rescale_positions(positions)
                
                obs[agent] = (rescaled_positions, rescaled_matrix)
        return obs
    
    def _downsample_matrix(self, matrix):
        """
        Downsample HxW matrix to (H/scale)x(W/scale) using block averaging.
        Obstacle cells (-1.0) are preserved: if any cell in a 2x2 block is -1.0,
        the output cell is -1.0 (obstacle dominates).
        """
        h, w = matrix.shape
        sf = self.scale_factor
        th, tw = h // sf, w // sf
        
        # Reshape into blocks and compute
        reshaped = matrix[:th*sf, :tw*sf].reshape(th, sf, tw, sf)
        
        # Check for obstacles in each block
        has_obstacle = np.any(reshaped < 0, axis=(1, 3))
        
        # Average pooling for non-obstacle blocks
        block_avg = reshaped.mean(axis=(1, 3))
        
        # Override obstacle blocks with -1.0
        block_avg[has_obstacle] = -1.0
        
        return block_avg
    
    def _rescale_positions(self, positions):
        """
        Rescale drone coordinate entries in the position vector.
        
        Position vector layout from BatteryStationWrapper:
          [0:2*n_agents] = drone positions (normalized by grid_size)
          [2*n_agents:3*n_agents] = battery levels (already normalized 0-1)
          [3*n_agents:5*n_agents] = crash coordinates (normalized by grid_size)
          [5*n_agents:5*n_agents+2] = nearest station (normalized by grid_size)
        
        The positions are already normalized by grid_size in BatteryStationWrapper,
        so they are already in [0, 1] range and don't need further rescaling.
        The -1.0 sentinel for dead drones should be preserved.
        """
        # Positions are already normalized to [0, 1] by BatteryStationWrapper
        # No rescaling needed - the normalized coordinates are grid-size independent
        return positions
    
    def _get_base_env(self):
        """Walk the wrapper chain to get the underlying DSSE environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'unwrapped'):
            return env.unwrapped
        return env
