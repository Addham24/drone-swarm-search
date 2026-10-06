"""
DSSE-3D Environment Creators
==============================
Factory functions for constructing 3D-wrapped DSSE environments for both
Coverage and Tracking benchmarks, maintaining full compatibility with the
existing 2D wrapper stacks.

These functions mirror the 2D counterparts:
    - src/tracking/tracking_env_creator.py  -> make_tracking_env_3d()
    - src/train_rspo_v2_vanilla.py          -> make_coverage_env_3d()

The 3D adapter is always the OUTERMOST wrapper, applied after all 2D wrappers
are in place. This ensures all 2D logic (battery, rewards, obstacles, drift)
operates identically to the 2D benchmarks.

Wrapper Stack (outermost → innermost):
    DSSE3DAdapter → RetainDronePosWrapper → [ObstacleAirspaceWrapper] →
    TrackingRewardWrapper → RandomDriftWrapper → BatteryStationWrapper →
    AllPositionsWrapper → DroneSwarmSearch / CoverageDroneSwarmSearch
"""

import os
import sys
import numpy as np
from typing import Optional

# Ensure src is on path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsse_3d.adapter import DSSE3DAdapter


def make_tracking_env_3d(
    grid_size: int = 25,
    drone_amount: int = 4,
    person_amount: int = 1,
    person_initial_position: tuple = (12, 12),
    timestep_limit: int = 750,
    max_battery: int = 125,
    fault_prob: float = 0.0005,
    is_self_heal: bool = False,
    drift_speed: float = 1.0,
    positions: Optional[list] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    z_max: int = 5,
    obstacle_extrusion_height: int = 3,
    enable_altitude_controller: bool = True,
):
    """
    Construct a 3D dynamic target tracking environment.

    This is the 3D counterpart of tracking.tracking_env_creator.make_tracking_env().
    The 2D environment is built identically, then wrapped with DSSE3DAdapter.

    Args:
        grid_size: Side length of the NxN grid (25 or 50).
        drone_amount: Number of drones.
        person_amount: Number of targets (always 1 for tracking).
        person_initial_position: Starting (y, x) of the target.
        timestep_limit: Maximum episode length.
        max_battery: Max battery capacity per drone.
        fault_prob: Random fault injection probability per step per drone.
        is_self_heal: If True, enable teammate compensation rewards.
        drift_speed: Target drift speed.
        positions: Drone spawn positions (default: top-left cluster).
        obstacle_mask: Optional 2D obstacle array.
        z_max: Maximum altitude layer.
        obstacle_extrusion_height: Height of extruded 3D obstacle blocks.
        enable_altitude_controller: If True, rule-based altitude management.

    Returns:
        A DSSE3DAdapter-wrapped tracking environment.
    """
    from tracking.tracking_env_creator import make_tracking_env

    # Build the standard 2D tracking environment with full wrapper stack
    env_2d = make_tracking_env(
        grid_size=grid_size,
        drone_amount=drone_amount,
        person_amount=person_amount,
        person_initial_position=person_initial_position,
        timestep_limit=timestep_limit,
        max_battery=max_battery,
        fault_prob=fault_prob,
        is_self_heal=is_self_heal,
        drift_speed=drift_speed,
        positions=positions,
        obstacle_mask=obstacle_mask,
    )

    # Wrap with 3D adapter (outermost layer)
    env_3d = DSSE3DAdapter(
        env_2d,
        grid_size=grid_size,
        z_max=z_max,
        obstacle_mask=obstacle_mask,
        obstacle_extrusion_height=obstacle_extrusion_height,
        enable_altitude_controller=enable_altitude_controller,
    )

    return env_3d


def make_coverage_env_3d(
    grid_size: int = 25,
    drone_amount: int = 4,
    timestep_limit: int = 750,
    max_battery: int = 125,
    fault_prob: float = 0.0005,
    is_self_heal: bool = False,
    positions: Optional[list] = None,
    obstacle_mask: Optional[np.ndarray] = None,
    z_max: int = 5,
    obstacle_extrusion_height: int = 3,
    enable_altitude_controller: bool = True,
    prob_matrix_path: Optional[str] = None,
):
    """
    Construct a 3D swarm area coverage environment.

    This is the 3D counterpart of the coverage env creators in
    train_rspo_v2_vanilla.py and evaluate_all_models.py.

    Args:
        grid_size: Side length of the NxN grid.
        drone_amount: Number of drones.
        timestep_limit: Maximum episode length.
        max_battery: Max battery capacity per drone.
        fault_prob: Random fault injection probability.
        is_self_heal: If True, enable compensation rewards.
        positions: Drone spawn positions.
        obstacle_mask: Optional 2D obstacle array.
        z_max: Maximum altitude layer.
        obstacle_extrusion_height: Height of 3D obstacle blocks.
        enable_altitude_controller: If True, rule-based altitude control.
        prob_matrix_path: Path to probability matrix .npy file.

    Returns:
        A DSSE3DAdapter-wrapped coverage environment.
    """
    from DSSE import CoverageDroneSwarmSearch
    from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
    from battery_station_wrapper import BatteryStationWrapper

    # Resolve probability matrix path
    if prob_matrix_path is None:
        prob_matrix_path = os.path.join(SRC_DIR, f"uniform_matrix_{grid_size}.npy")

    # Build the 2D coverage environment
    env = CoverageDroneSwarmSearch(
        timestep_limit=timestep_limit,
        drone_amount=drone_amount,
        prob_matrix_path=prob_matrix_path,
    )
    env.reward_scheme = {
        "default": -0.1,
        "exceed_timestep": 0.0,
        "search_cell": 5.0,
        "done": 500.0,
        "reward_poc": 0.0,
    }

    # Standard 2D wrapper stack
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(
        env,
        max_battery=max_battery,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=fault_prob,
    )

    if not is_self_heal:
        env.COMPENSATION_BONUS = 0.0
        env.COMPENSATION_PENALTY = 0.0
        env.COMPENSATION_HORIZON = 0
    else:
        env.COMPENSATION_BONUS = 2.5
        env.COMPENSATION_PENALTY = -0.5
        env.COMPENSATION_HORIZON = 100
        # 2D self-heal coverage evaluators wrap with GlobalRewardWrapper(alpha=0.5)
        from global_reward_wrapper import GlobalRewardWrapper
        env = GlobalRewardWrapper(env, mixing_alpha=0.5)

    # Wrapper order mirrors evaluate_50x50_and_obstacles.py exactly:
    #   Battery -> [GlobalReward] -> [Obstacle] -> [SpatialRescaling] -> Retain
    if obstacle_mask is not None:
        from obstacle_airspace_wrapper import ObstacleAirspaceWrapper
        env = ObstacleAirspaceWrapper(env, obstacle_mask)

    if grid_size != 25:
        # 25x25-trained policies see a 2x2-average-pooled 25x25 observation
        from spatial_rescaling_wrapper import SpatialRescalingWrapper
        env = SpatialRescalingWrapper(env, target_grid_size=25)

    if positions is None:
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)

    # Wrap with 3D adapter (outermost layer)
    env_3d = DSSE3DAdapter(
        env,
        grid_size=grid_size,
        z_max=z_max,
        obstacle_mask=obstacle_mask,
        obstacle_extrusion_height=obstacle_extrusion_height,
        enable_altitude_controller=enable_altitude_controller,
    )

    return env_3d
