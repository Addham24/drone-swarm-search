import os
import sys
from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper

# Ensure src directory is on sys.path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from battery_station_wrapper import BatteryStationWrapper
from tracking.tracking_drift_wrapper import RandomDriftWrapper
from tracking.tracking_reward_wrapper import TrackingRewardWrapper

def make_tracking_env(
    grid_size=25,
    drone_amount=4,
    person_amount=1,
    person_initial_position=(12, 12),
    timestep_limit=750,
    max_battery=125,
    fault_prob=0.0005,
    is_self_heal=False,
    drift_speed=1.0,
    positions=None,
    render_mode=None
):
    """
    Constructs the dynamic target tracking environment with full wrapper stack:
    DroneSwarmSearch -> AllPositionsWrapper -> BatteryStationWrapper -> RandomDriftWrapper -> TrackingRewardWrapper -> RetainDronePosWrapper
    """
    env_kwargs = {
        "grid_size": grid_size,
        "drone_amount": drone_amount,
        "person_amount": person_amount,
        "person_initial_position": person_initial_position,
        "timestep_limit": timestep_limit,
    }
    if render_mode is not None:
        env_kwargs["render_mode"] = render_mode

    env = DroneSwarmSearch(**env_kwargs)
    
    # 1. Expand observation space to include all drone positions
    env = AllPositionsWrapper(env)
    
    # 2. Add battery depletion, charging at (0,0), and fault injection
    env = BatteryStationWrapper(
        env,
        max_battery=max_battery,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=fault_prob
    )
    
    if not is_self_heal:
        env.COMPENSATION_BONUS = 0.0
        env.COMPENSATION_PENALTY = 0.0
        env.COMPENSATION_HORIZON = 0
    else:
        env.COMPENSATION_BONUS = 2.5
        env.COMPENSATION_PENALTY = -0.5
        env.COMPENSATION_HORIZON = 100

    # 3. Randomize drift direction on reset
    env = RandomDriftWrapper(env, speed=drift_speed)

    # 4. Apply dense reward shaping
    env = TrackingRewardWrapper(env, prob_scale=5.0, step_penalty=-0.1, discovery_bonus=500.0)

    # 5. Fix spawn positions
    if positions is None:
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)

    return env
