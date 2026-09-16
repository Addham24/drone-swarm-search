"""
Live Pygame Window Demo: Dynamic Target Tracking Benchmark
===========================================================
Runs an interactive Pygame visualization window showing trained MARL drones
(RSPO V2, MAPPO, QMIX, MAA2C, COMA, IQL) pursuing a dynamic moving target in real-time.

Usage:
  python src/tracking/run_live_tracking_demo.py --model rspo_selfheal
  python src/tracking/run_live_tracking_demo.py --model qmix_selfheal
  python src/tracking/run_live_tracking_demo.py --model rspo_vanilla
"""

import os
import sys
import time
import argparse
import numpy as np
import torch as th

# Ensure src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from DSSE import DroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from tracking.tracking_drift_wrapper import RandomDriftWrapper
from tracking.tracking_reward_wrapper import TrackingRewardWrapper

from tracking.evaluate_tracking_models import (
    load_rllib_policy,
    load_epymarl_policy,
    find_best_post_10m_checkpoint,
    RSPOModelV2,
    MAPPOModelVanilla,
)


def create_live_tracking_env(is_self_heal=True, fault_prob=0.0005, grid_size=25):
    """Creates Pygame-rendered tracking environment."""
    env = DroneSwarmSearch(
        grid_size=grid_size,
        drone_amount=4,
        person_amount=1,
        person_initial_position=(grid_size // 2, grid_size // 2),
        timestep_limit=750,
        render_mode="human",
    )
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(
        env,
        max_battery=125,
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

    env = RandomDriftWrapper(env, speed=1.0)
    env = TrackingRewardWrapper(env, prob_scale=5.0, step_penalty=-0.1, discovery_bonus=500.0)
    env = RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])
    return env


def main():
    parser = argparse.ArgumentParser(description="Live Pygame Tracking Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="rspo_selfheal",
        choices=[
            "rspo_selfheal", "rspo_vanilla",
            "mappo_selfheal", "mappo_vanilla",
            "qmix_selfheal", "qmix_vanilla",
            "maa2c_selfheal", "maa2c_vanilla",
            "coma_selfheal", "coma_vanilla",
            "iql_selfheal", "iql_vanilla"
        ],
        help="Select model to visualize in live Pygame window"
    )
    parser.add_argument("--fps", type=int, default=20, help="Playback speed (FPS)")
    parser.add_argument("--seed", type=int, default=1000, help="Test seed")
    parser.add_argument("--no_faults", action="store_true", help="Disable random hardware faults")
    args = parser.parse_args()

    print("=" * 70)
    print(f"   LIVE PYGAME DEMO: DYNAMIC TARGET TRACKING ({args.model.upper()})")
    print("=" * 70)
    print("Launching Pygame Window... Watch drones pursue the target!\n")

    is_self_heal = "selfheal" in args.model
    fault_prob = 0.0 if args.no_faults else 0.0005

    model_config_map = {
        "rspo_vanilla": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_Vanilla*/**/checkpoint_*"), "rllib", RSPOModelV2),
        "rspo_selfheal": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_SelfHeal*/**/checkpoint_*"), "rllib", RSPOModelV2),
        "mappo_vanilla": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*MAPPO_Tracking_Vanilla*/**/checkpoint_*"), "rllib", MAPPOModelVanilla),
        "mappo_selfheal": (os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*MAPPO_Tracking_SelfHeal*/**/checkpoint_*"), "rllib", MAPPOModelVanilla),
        "qmix_vanilla": (os.path.join(SRC_DIR, "results/models/*qmix_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "qmix_selfheal": (os.path.join(SRC_DIR, "results/models/*qmix_tracking_selfheal*/**/agent.th"), "epymarl", None),
        "maa2c_vanilla": (os.path.join(SRC_DIR, "results/models/*maa2c_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "maa2c_selfheal": (os.path.join(SRC_DIR, "results/models/*maa2c_tracking_selfheal*/**/agent.th"), "epymarl", None),
        "coma_vanilla": (os.path.join(SRC_DIR, "results/models/*coma_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "coma_selfheal": (os.path.join(SRC_DIR, "results/models/*coma_tracking_selfheal*/**/agent.th"), "epymarl", None),
        "iql_vanilla": (os.path.join(SRC_DIR, "results/models/*iql_tracking_vanilla*/**/agent.th"), "epymarl", None),
        "iql_selfheal": (os.path.join(SRC_DIR, "results/models/*iql_tracking_selfheal*/**/agent.th"), "epymarl", None),
    }

    pattern, framework, model_cls = model_config_map[args.model]
    ckpt_path = find_best_post_10m_checkpoint(pattern, framework, min_timesteps=10_000_000)

    if not ckpt_path:
        print(f"[!] Error: Could not find trained checkpoint for {args.model} matching {pattern}")
        sys.exit(1)

    print(f"Loading {framework.upper()} Checkpoint: {ckpt_path}\n")
    if framework == "rllib":
        policy_fn = load_rllib_policy(model_cls, ckpt_path, model_name=args.model)
    else:
        policy_fn = load_epymarl_policy(ckpt_path)

    env = create_live_tracking_env(is_self_heal=is_self_heal, fault_prob=fault_prob)
    obs_dict, infos = env.reset(seed=args.seed)

    step = 0
    target_found = False
    delay = 1.0 / max(1, args.fps)

    print("Pygame Window opened! Press Ctrl+C in Terminal to stop.\n")

    try:
        while env.agents and step < 750 and not target_found:
            step += 1
            actions = policy_fn(obs_dict, env.agents)
            obs_dict, rewards, term, trunc, infos = env.step(actions)

            # Check target discovery
            for agent, r in rewards.items():
                if r >= 1.0 or (isinstance(infos.get(agent), dict) and infos.get(agent, {}).get("search_and_find", False)):
                    target_found = True
                    print(f"🎯 TARGET INTERCEPTED AT STEP {step}!")
                    break

            time.sleep(delay)

            if step % 25 == 0:
                alive_count = sum(1 for a in env.agents if obs_dict.get(a) is not None)
                print(f"  Step {step:3d}/750 | Alive Drones: {alive_count}/4 | Target Found: {target_found}")

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        print(f"\nDemo error: {e}")

    print(f"\nDemo Complete! Final Result: {'TARGET FOUND in ' + str(step) + ' steps' if target_found else 'TARGET ESCAPED (750 steps)'}")


if __name__ == "__main__":
    main()
