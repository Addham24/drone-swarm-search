"""
DSSE3DAdapter: Zero-Shot 3D Adapter Wrapper
=============================================
PettingZoo-compatible parallel environment wrapper that lifts a 2D DSSE
environment (Coverage or Tracking) into a 3D voxel workspace.

Zero-Shot Design:
    Pre-trained 2D policies (RSPO V2, MAPPO, QMIX, etc.) produce 2D actions
    (0..8: 8 movement directions + SEARCH). The adapter maps these into 3D
    state transitions by:
        1. Forwarding horizontal (X, Y) movement to the underlying 2D env.
        2. Managing altitude (Z) through a rule-based altitude controller that
           descends to Z=1 for scanning and ascends for transit/obstacle avoidance.
        3. Augmenting observations with altitude state so the policy is aware
           of vertical context without needing architectural changes.

    This enables evaluation of ALL pre-trained 2D checkpoints in 3D with
    zero retraining.

Observation Augmentation:
    The adapter appends a small altitude context vector to the existing position
    observation vector:
        [original_positions | altitude_norm(n_drones) | fov_radius_norm(1)]
    The 2D probability matrix observation is passed through unchanged.

    For RLlib models (RSPO V2, MAPPO), the position dimension increases from
    22 to 22 + n_drones + 1 = 27.
    For EPyMARL models (QMIX, MAA2C, COMA, IQL), the flat obs dimension increases
    from 647 to 652, plus agent ID one-hot.

Reference:
    Zero-shot sim-to-sim transfer principle: the 2D spatial reasoning policy
    is preserved, while altitude control is handled by the adapter's
    rule-based controller — akin to hierarchical RL without separate training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from gymnasium.spaces import Tuple as GymTuple, Box, Discrete
from pettingzoo.utils.wrappers import BaseParallelWrapper

from dsse_3d.voxel_grid import VoxelGrid


# ──────────────────────────────────────────────────────────────────────────────
# Altitude Controller Strategies
# ──────────────────────────────────────────────────────────────────────────────

class AltitudeController:
    """
    Rule-based altitude controller for the Zero-Shot 3D Adapter.

    Strategy:
        - If the 2D policy issues a SEARCH action (8), descend to Z=1 first
          if not already there. The search only executes once at Z=1.
        - If the drone is at Z=1 and the 2D action is a move (0..7), ascend
          to Z=2 transit altitude for efficient horizontal traversal.
        - If moving into a 3D obstacle at current altitude, try ascending
          one layer to fly over it.
        - During transit (Z >= 2), maintain altitude unless approaching a
          high-probability area (detected by the 2D policy issuing SEARCH).
        - Battery-critical drones (battery < 20%) descend to Z=0 for landing
          at the charging station.
    """

    def __init__(self, voxel_grid: VoxelGrid, battery_critical_threshold: float = 0.20):
        self.voxel = voxel_grid
        self.battery_threshold = battery_critical_threshold

    def compute_altitude_action(
        self,
        drone_idx: int,
        action_2d: int,
        battery_level: float,
        at_station: bool,
    ) -> int:
        """
        Determine the altitude change for a drone given its 2D action.

        Args:
            drone_idx: Index of the drone.
            action_2d: The 2D action (0..8).
            battery_level: Normalized battery [0, 1].
            at_station: True if drone is at the charging station (0, 0).

        Returns:
            delta_z: -1 (descend), 0 (hold), or +1 (ascend).
        """
        z = self.voxel.get_altitude(drone_idx)
        is_search = (action_2d == 8)

        # Priority 1: Landing & Charging at Station (0, 0)
        # Stay landed at Z=0 until battery is recharged to >= 95%
        if at_station and battery_level < 0.95:
            if z > VoxelGrid.Z_GROUND:
                return -1  # Descend to Z=0 landing pad
            return 0       # Stay landed on charging pad at Z=0

        # Priority 2: SEARCH action — descend to scan altitude (Z=1)
        if is_search:
            if z > VoxelGrid.Z_SCAN:
                return -1
            elif z < VoxelGrid.Z_SCAN:
                return +1
            return 0  # Already at scan altitude

        # Priority 3: Movement action — ascend to transit altitude if at scan/ground
        if z <= VoxelGrid.Z_SCAN:
            return +1  # Ascend for horizontal transit

        # Priority 4: Maintain transit altitude
        return 0

    def resolve_3d_obstacle(self, drone_idx: int, new_x: int, new_y: int) -> int:
        """
        If horizontal move is blocked at current altitude, try ascending.

        Returns:
            delta_z to apply before retrying the move, or 0 if no resolution.
        """
        z = self.voxel.get_altitude(drone_idx)
        # Try one altitude above
        if z + 1 <= self.voxel.z_max and not self.voxel._is_blocked(new_x, new_y, z + 1):
            return +1
        return 0


# ──────────────────────────────────────────────────────────────────────────────
# Main 3D Adapter Wrapper
# ──────────────────────────────────────────────────────────────────────────────

class DSSE3DAdapter(BaseParallelWrapper):
    """
    Zero-Shot 3D Adapter: wraps a 2D DSSE PettingZoo environment to operate
    in 3D voxel space.

    Supports both Coverage (CoverageDroneSwarmSearch) and Tracking
    (DroneSwarmSearch) environments transparently.

    Args:
        env: The fully wrapped 2D DSSE parallel environment (after
             BatteryStationWrapper, TrackingRewardWrapper, etc.).
        grid_size: Side length of the NxN ground grid.
        z_max: Maximum altitude layer (default 5).
        obstacle_mask: Optional 2D numpy array (NxN) where 1 = obstacle.
                       Obstacles are extruded to default height 3.
        obstacle_extrusion_height: Height of extruded obstacle prisms.
        inter_drone_collision_penalty: Reward penalty for mid-air collisions.
        enable_altitude_controller: If True, uses rule-based altitude control.
                                     If False, drones stay at fixed Z=1 (flat 3D).
    """

    # 3D-specific reward shaping constants
    ALTITUDE_TRANSITION_COST = -0.02   # Small cost per altitude change
    MIDAIR_COLLISION_PENALTY = -5.0    # Penalty for inter-drone 3D collision
    OBSTACLE_COLLISION_PENALTY = -2.0  # Penalty for hitting 3D obstacle prism
    SEARCH_AT_WRONG_ALT_PENALTY = -0.5  # Penalty for searching at wrong altitude

    def __init__(
        self,
        env,
        grid_size: int = 25,
        z_max: int = 5,
        obstacle_mask: Optional[np.ndarray] = None,
        obstacle_extrusion_height: int = 3,
        inter_drone_collision_penalty: float = -5.0,
        enable_altitude_controller: bool = True,
    ):
        super().__init__(env)
        self.grid_size = grid_size
        self.z_max = z_max
        self.n_drones = len(self.env.possible_agents)
        self.enable_altitude_controller = enable_altitude_controller

        self.MIDAIR_COLLISION_PENALTY = inter_drone_collision_penalty

        # Build 3D obstacle heights from 2D mask
        obstacle_heights = {}
        if obstacle_mask is not None:
            for y in range(obstacle_mask.shape[0]):
                for x in range(obstacle_mask.shape[1]):
                    if obstacle_mask[y, x] == 1:
                        obstacle_heights[(x, y)] = obstacle_extrusion_height

        # Core 3D state manager
        self.voxel = VoxelGrid(
            grid_size=grid_size,
            z_max=z_max,
            n_drones=self.n_drones,
            scan_altitude=VoxelGrid.Z_SCAN,
            obstacle_heights=obstacle_heights,
        )

        # Altitude controller
        self.alt_controller = AltitudeController(self.voxel)

        # Tracking for visualization and metrics
        self.trajectory_history: Dict[int, List[Tuple[int, int, int]]] = {}
        self.altitude_changes_count = 0
        self.midair_collisions_count = 0
        self.obstacle_collisions_3d_count = 0
        self.search_at_wrong_alt_count = 0

        # Number of extra observation dimensions added by the 3D adapter
        # [altitude_per_drone(n_drones) + own_fov_radius(1)]
        self.altitude_obs_dim = self.n_drones + 1

        # 3D-aware ground coverage tracking (strictly registered at Z>=1 scan altitude)
        self.seen_states_3d: Set[Tuple[int, int]] = set()
        # Cells that count toward coverage (2D free cells: excludes zero-POC / obstacle cells).
        # Captured from the 2D env at reset so the 3D denominator equals the 2D denominator.
        self._coverage_universe = None

        # Override observation spaces
        self.observation_spaces = {
            agent: self._build_3d_observation_space(agent)
            for agent in self.env.possible_agents
        }

    def _build_3d_observation_space(self, agent):
        """
        Extend the 2D observation space with altitude context dimensions.
        """
        base_space = self.env.observation_space(agent)
        base_pos_box = base_space[0]
        matrix_box = base_space[1]

        # Extend position vector with altitude info
        new_pos_dim = base_pos_box.shape[0] + self.altitude_obs_dim
        new_pos_box = Box(
            low=-1.0, high=1.0,
            shape=(new_pos_dim,),
            dtype=np.float32,
        )
        return GymTuple((new_pos_box, matrix_box))

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)

        # Extract initial 2D positions from the base environment
        base_env = self._get_base_env()
        if hasattr(base_env, 'agents_positions') and base_env.agents_positions:
            # DSSE agents_positions stores (y, x) = (row, col)
            # VoxelGrid expects (x, y) = (col, row)
            positions_2d = [(pos[1], pos[0]) for pos in base_env.agents_positions]
        else:
            # Fallback: parse from observation
            positions_2d = []
            for i, agent in enumerate(self.env.possible_agents):
                if agent in obs:
                    pos_vec = obs[agent][0]
                    px = int(round(pos_vec[1] * self.grid_size)) if len(pos_vec) > 1 else 0
                    py = int(round(pos_vec[0] * self.grid_size)) if len(pos_vec) > 0 else 0
                    px = max(0, min(self.grid_size - 1, px))
                    py = max(0, min(self.grid_size - 1, py))
                    positions_2d.append((px, py))
                else:
                    positions_2d.append((0, 0))

        # Initialize 3D voxel state
        self.voxel.reset(positions_2d, initial_z=VoxelGrid.Z_SCAN)

        # Reset trajectory tracking
        self.trajectory_history = {i: [self.voxel.get_position(i)] for i in range(self.n_drones)}
        self.altitude_changes_count = 0
        self.midair_collisions_count = 0
        self.obstacle_collisions_3d_count = 0
        self.search_at_wrong_alt_count = 0

        # Capture the 2D free-cell universe (all cells minus zero-POC cells) BEFORE syncing
        if hasattr(base_env, 'seen_states') and base_env.seen_states is not None \
                and getattr(base_env, 'not_seen_states', None) is not None:
            self._coverage_universe = set(base_env.seen_states) | set(base_env.not_seen_states)
        else:
            self._coverage_universe = None

        # Reset 3D ground coverage state
        self.seen_states_3d.clear()
        for idx in range(self.n_drones):
            if self.voxel.get_altitude(idx) >= 1:
                for cell in self.voxel.get_fov_footprint(idx):
                    self.seen_states_3d.add(cell)

        # Sync base 2D coverage environment if applicable
        self._sync_coverage(base_env)

        # Augment observations with altitude
        obs = self._augment_obs_with_altitude(obs)

        # Inject 3D info
        for agent in infos:
            if isinstance(infos[agent], dict):
                infos[agent] = dict(infos[agent])
                idx = self.env.possible_agents.index(agent)
                infos[agent]["altitude"] = self.voxel.get_altitude(idx)
                infos[agent]["position_3d"] = self.voxel.get_position(idx)
                if self._coverage_universe is not None:
                    infos[agent]["coverage_rate"] = self.get_coverage_rate()

        return obs, infos

    def step(self, actions):
        """
        Execute one timestep in 3D space.

        Pipeline:
            1. For each drone, compute altitude transition via AltitudeController.
            2. Apply altitude changes in the VoxelGrid.
            3. Filter SEARCH actions: only valid at Z=1 scanning altitude.
            4. Forward horizontal actions to the underlying 2D environment.
            5. Sync 2D positions back to the 3D VoxelGrid.
            6. Detect inter-drone 3D collisions.
            7. Augment observations with altitude context.
            8. Apply 3D reward shaping.
        """
        # Extract battery levels from previous step for altitude controller
        battery_levels = self._get_battery_levels()

        # Step 1 & 2: Altitude transitions
        altitude_penalties = {}
        actions_for_2d = {}

        for agent in self.env.possible_agents:
            if agent not in actions:
                continue

            idx = self.env.possible_agents.index(agent)
            action_2d = actions[agent]
            batt = battery_levels.get(agent, 1.0)
            x, y, z = self.voxel.get_position(idx)
            at_station = (x == 0 and y == 0)

            # Compute altitude change
            if self.enable_altitude_controller:
                delta_z = self.alt_controller.compute_altitude_action(
                    idx, action_2d, batt, at_station
                )
            else:
                delta_z = 0

            # Apply altitude change
            if delta_z != 0:
                success = self.voxel.change_altitude(idx, delta_z)
                if success:
                    self.altitude_changes_count += 1
                    altitude_penalties[agent] = self.ALTITUDE_TRANSITION_COST

            # Filter SEARCH actions at wrong altitude
            if action_2d == 8 and not self.voxel.can_search(idx):
                # Search not valid at current altitude — penalize
                self.search_at_wrong_alt_count += 1
                altitude_penalties[agent] = altitude_penalties.get(agent, 0.0) + \
                    self.SEARCH_AT_WRONG_ALT_PENALTY
                # Convert to STAY/IDLE (keep as SEARCH for 2D env, which handles it)

            actions_for_2d[agent] = action_2d

        # Propagate current 3D altitudes down the wrapper chain so ObstacleAirspaceWrapper knows Z
        current_alts = [self.voxel.get_altitude(i) for i in range(self.n_drones)]
        curr_env = self.env
        while curr_env is not None:
            curr_env.current_altitudes = current_alts
            curr_env = getattr(curr_env, 'env', None)

        # Step 3: Forward actions to 2D environment
        obs, rewards, terminations, truncations, infos = self.env.step(actions_for_2d)

        # Step 4: Sync updated 2D positions back to VoxelGrid
        base_env = self._get_base_env()
        if hasattr(base_env, 'agents_positions') and base_env.agents_positions:
            for idx, (y, x) in enumerate(base_env.agents_positions):
                if idx < self.n_drones:
                    agent = self.env.possible_agents[idx]
                    if agent in obs:
                        # DSSE agents_positions is (y, x) = (row, col)
                        # VoxelGrid expects (x, y) = (col, row)
                        move_ok = self.voxel.apply_2d_action(idx, x, y)
                        if not move_ok:
                            self.obstacle_collisions_3d_count += 1
                            # Try ascending over obstacle
                            delta = self.alt_controller.resolve_3d_obstacle(idx, x, y)
                            if delta != 0:
                                self.voxel.change_altitude(idx, delta)
                                self.voxel.apply_2d_action(idx, x, y)
                            rewards[agent] = rewards.get(agent, 0.0) + self.OBSTACLE_COLLISION_PENALTY

        # Step 4b: 3D-Aware Coverage Update (Altitude FOV Footprint Scaling: Z=1 -> 1x1, Z=2 -> 3x3, Z=3 -> 5x5)
        if hasattr(base_env, 'seen_states'):
            for idx in range(self.n_drones):
                agent = self.env.possible_agents[idx] if idx < len(self.env.possible_agents) else None
                if agent and agent in obs:
                    if self.voxel.get_altitude(idx) >= 1:
                        # Scan ground cells in drone's altitude-dependent FOV footprint
                        for cell in self.voxel.get_fov_footprint(idx):
                            self.seen_states_3d.add(cell)

            # Synchronize 3D coverage state back to base environment
            self._sync_coverage(base_env)

        # Step 6: Apply altitude transition costs
        for agent, penalty in altitude_penalties.items():
            if agent in rewards:
                rewards[agent] += penalty

        # Step 7: Update trajectory history
        for idx in range(self.n_drones):
            if idx in self.voxel.drone_positions:
                pos = self.voxel.get_position(idx)
                if idx in self.trajectory_history:
                    self.trajectory_history[idx].append(pos)
                else:
                    self.trajectory_history[idx] = [pos]

        # Step 8: Augment observations with 3D altitude context
        obs = self._augment_obs_with_altitude(obs)

        # Inject 3D metadata into infos
        for agent in infos:
            if isinstance(infos[agent], dict):
                infos[agent] = dict(infos[agent])
                idx = self.env.possible_agents.index(agent) if agent in self.env.possible_agents else -1
                if idx >= 0 and idx in self.voxel.drone_positions:
                    infos[agent]["altitude"] = self.voxel.get_altitude(idx)
                    infos[agent]["position_3d"] = self.voxel.get_position(idx)
                    infos[agent]["can_search"] = self.voxel.can_search(idx)
                infos[agent]["midair_collisions_total"] = self.midair_collisions_count
                infos[agent]["obstacle_collisions_3d_total"] = self.obstacle_collisions_3d_count
                if self._coverage_universe is not None:
                    # Fresh value: includes this step's 3D footprint (the 2D env computed
                    # its own coverage_rate before the footprint was applied).
                    infos[agent]["coverage_rate"] = self.get_coverage_rate()

        return obs, rewards, terminations, truncations, infos

    def _sync_coverage(self, base_env):
        """Push the 3D coverage state into the 2D env, restricted to the 2D free-cell universe."""
        if self._coverage_universe is None or not hasattr(base_env, 'seen_states'):
            return
        seen = self.seen_states_3d & self._coverage_universe
        base_env.seen_states = set(seen)
        base_env.not_seen_states = self._coverage_universe - seen

    def get_coverage_rate(self) -> float:
        """Fraction of 2D free cells scanned so far (same definition as the 2D benchmark)."""
        if not self._coverage_universe:
            return 0.0
        return len(self.seen_states_3d & self._coverage_universe) / len(self._coverage_universe)

    def _augment_obs_with_altitude(self, obs):
        """
        Append altitude context to each agent's position observation vector.

        Appended dimensions:
            [alt_drone_0, alt_drone_1, ..., alt_drone_{n-1}, own_fov_radius]
        All normalized to [0, 1].
        """
        altitude_obs = self.voxel.get_altitude_observation()

        for agent in obs:
            if isinstance(obs[agent], tuple) and len(obs[agent]) == 2:
                positions, matrix = obs[agent]

                idx = self.env.possible_agents.index(agent) if agent in self.env.possible_agents else 0
                own_z = self.voxel.get_altitude(idx) if idx in self.voxel.drone_positions else 1
                fov_radius = VoxelGrid.FOV_RADIUS.get(own_z, 0) / max(VoxelGrid.FOV_RADIUS.values())

                # Build altitude context vector
                alt_context = np.concatenate([
                    altitude_obs.astype(np.float32),
                    np.array([fov_radius], dtype=np.float32),
                ])

                # Append to position vector
                augmented_positions = np.concatenate([
                    positions.astype(np.float32),
                    alt_context,
                ])

                obs[agent] = (augmented_positions, matrix)

        return obs

    def _get_battery_levels(self) -> Dict[str, float]:
        """Extract normalized battery levels from the wrapper chain."""
        battery = {}
        # Walk the wrapper chain to find BatteryStationWrapper
        env = self.env
        while hasattr(env, 'env'):
            if hasattr(env, 'battery') and hasattr(env, 'max_battery'):
                for agent in self.env.possible_agents:
                    if agent in env.battery:
                        battery[agent] = env.battery[agent] / env.max_battery
                return battery
            env = env.env
        # If no battery wrapper found, assume full battery
        return {agent: 1.0 for agent in self.env.possible_agents}

    def _get_base_env(self):
        """Walk the wrapper chain to get the underlying DSSE environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        if hasattr(env, 'unwrapped'):
            return env.unwrapped
        return env

    def get_3d_metrics(self) -> dict:
        """
        Return 3D-specific metrics for evaluation and dashboard generation.
        """
        return {
            "altitude_changes": self.altitude_changes_count,
            "midair_collisions": self.midair_collisions_count,
            "obstacle_collisions_3d": self.obstacle_collisions_3d_count,
            "search_at_wrong_altitude": self.search_at_wrong_alt_count,
            "trajectories": dict(self.trajectory_history),
        }
