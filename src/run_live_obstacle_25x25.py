"""
Live Pygame Window Demo: 25x25 Obstacle Airspace (Layout 3 Skyline)
====================================================================
Runs an interactive Pygame visualization window showing pre-trained RSPO
drones navigating the 25x25 obstacle airspace in real-time.

Usage:
  python3 run_live_obstacle_25x25.py
"""

import os
import sys
import time
import numpy as np
import torch as th

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from obstacle_airspace_wrapper import ObstacleAirspaceWrapper

from evaluate_all_models import load_rllib_policy
from train_rspo_vanilla import RSPOModel as RSPOModelVanilla


def main():
    print("=" * 60)
    print("   LIVE PYGAME DEMO: 25x25 OBSTACLE AIRSPACE (LAYOUT 3 SKYLINE)")
    print("=" * 60)
    print("Launching Pygame Window... Watch the drones search and avoid obstacles!\n")

    obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_25.npy"))
    prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")

    # Create environment with Pygame human rendering enabled
    env = CoverageDroneSwarmSearch(
        render_mode="human",
        timestep_limit=750,
        drone_amount=4,
        prob_matrix_path=prob_path
    )
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0
    env = ObstacleAirspaceWrapper(env, obstacle_mask)
    env = RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])

    # Load RSPO pre-trained policy
    model_path = os.path.join(
        SRC_DIR,
        "ray_res/DSSE_Coverage/RSPO_vanilla_v1/PPO_DSSE_Coverage_RSPO_Vanilla_5bfc7_00000_0_2026-08-01_04-35-01/checkpoint_000199"
    )
    
    policy = load_rllib_policy(
        algo_type="ppo",
        env_name="DSSE_Live_Demo_25",
        model_name="RSPO_Live_Demo_Model_25",
        custom_model_cls=RSPOModelVanilla,
        checkpoint_path=model_path
    )

    obs_dict, infos = env.reset(seed=42)
    step = 0
    total_reward = 0.0

    print("Pygame Window opened! Press Ctrl+C in Terminal to stop.\n")

    try:
        while env.agents:
            step += 1
            actions = {}
            for agent in env.agents:
                if agent in obs_dict:
                    actions[agent] = policy.compute_single_action(observation=obs_dict[agent], policy_id="default_policy")

            obs_dict, rewards, terminations, truncations, infos = env.step(actions)
            total_reward += sum(rewards.values())

            # Control playback speed (50ms per step = 20 FPS)
            time.sleep(0.05)

            # Print live progress every 25 steps
            if step % 25 == 0:
                sample_info = next(iter(infos.values()), {}) if infos else {}
                cov = sample_info.get("coverage_rate", 0.0) * 100
                alive_count = sum(1 for a, inf in infos.items() if isinstance(inf, dict) and inf.get("alive", True))
                print(f"  Step {step:3d}/750 | Coverage: {cov:5.1f}% | Alive Drones: {alive_count}/4 | Return: {total_reward:7.1f}")

    except (KeyboardInterrupt, Exception) as e:
        print(f"\nDemo stopped ({type(e).__name__}).")

    sample_info = next(iter(infos.values()), {}) if infos else {}
    final_cov = sample_info.get("coverage_rate", 0.0) * 100
    print(f"\nEpisode Finished! Final Coverage Rate: {final_cov:.2f}%")
    
    if hasattr(policy, 'stop'):
        policy.stop()


if __name__ == "__main__":
    main()
