"""
Live Pygame Window Demo: Dynamic Target Tracking Benchmark
===========================================================
Runs an interactive Pygame visualization window showing trained MARL drones
(RSPO V2, MAPPO, QMIX, MAA2C, COMA, IQL) pursuing a dynamic moving target in real-time.

Includes a built-in greedy heuristic mode that follows the probability gradient,
useful when trained tracking models haven't converged yet.

Usage:
  python src/tracking/run_live_tracking_demo.py --model rspo_selfheal
  python src/tracking/run_live_tracking_demo.py --model qmix_selfheal
  python src/tracking/run_live_tracking_demo.py --model heuristic
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
from DSSE.environment.constants import Actions
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

# Map direction deltas to DSSE Actions for the heuristic policy
# DSSE uses (x, y) where x is column, y is row
DIRECTION_TO_ACTION = {
    (-1,  0): Actions.LEFT.value,     # 0
    ( 1,  0): Actions.RIGHT.value,    # 1
    ( 0, -1): Actions.UP.value,       # 2
    ( 0,  1): Actions.DOWN.value,     # 3
    (-1, -1): Actions.UP_LEFT.value,  # 4
    ( 1, -1): Actions.UP_RIGHT.value, # 5
    (-1,  1): Actions.DOWN_LEFT.value, # 6
    ( 1,  1): Actions.DOWN_RIGHT.value, # 7
    ( 0,  0): Actions.SEARCH.value,   # 8
}


def make_heuristic_policy(grid_size=25):
    """
    Creates a greedy heuristic policy that moves drones toward the highest-probability
    cells in the 25×25 probability matrix. Each drone picks the neighboring cell
    (including diagonals) with the highest probability and moves there.
    If already on the peak, it searches.
    """
    def policy_fn(obs_dict, agents, reset=False):
        actions = {}
        for agent in agents:
            if agent not in obs_dict:
                continue

            pos_vec, prob_matrix = obs_dict[agent]
            # De-normalize position: pos_vec[0]=y/grid_size, pos_vec[1]=x/grid_size
            drone_y = int(round(pos_vec[0] * grid_size))
            drone_x = int(round(pos_vec[1] * grid_size))

            # Clamp to grid bounds
            drone_y = max(0, min(grid_size - 1, drone_y))
            drone_x = max(0, min(grid_size - 1, drone_x))

            # Find best neighboring cell (8-connected + stay)
            best_action = Actions.SEARCH.value  # default: search in place
            best_prob = prob_matrix[drone_y, drone_x] if 0 <= drone_y < grid_size and 0 <= drone_x < grid_size else 0.0

            for (dx, dy), action_val in DIRECTION_TO_ACTION.items():
                if dx == 0 and dy == 0:
                    continue
                nx, ny = drone_x + dx, drone_y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    p = prob_matrix[ny, nx]
                    if p > best_prob + 1e-8:
                        best_prob = p
                        best_action = action_val

            actions[agent] = best_action

        return actions

    return policy_fn


def create_live_tracking_env(is_self_heal=True, fault_prob=0.0005, grid_size=25,
                              drone_positions=None, max_battery=125):
    """Creates Pygame-rendered tracking environment."""
    if drone_positions is None:
        drone_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

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
        max_battery=max_battery,
        depletion_rate=1 if max_battery <= 125 else 0,  # 0 depletion if unlimited battery mode
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
    env = RetainDronePosWrapper(env, drone_positions)
    return env


def main():
    parser = argparse.ArgumentParser(description="Live Pygame Tracking Demo")
    parser.add_argument(
        "--model",
        type=str,
        default="heuristic",
        choices=[
            "heuristic",
            "rspo_selfheal", "rspo_vanilla",
            "mappo_selfheal", "mappo_vanilla",
            "qmix_selfheal", "qmix_vanilla",
            "maa2c_selfheal", "maa2c_vanilla",
            "coma_selfheal", "coma_vanilla",
            "iql_selfheal", "iql_vanilla"
        ],
        help="Select model to visualize (default: heuristic greedy policy)"
    )
    parser.add_argument("--fps", type=int, default=8, help="Playback speed (FPS)")
    parser.add_argument("--seed", type=int, default=1000, help="Test seed")
    parser.add_argument("--no_faults", action="store_true", help="Disable random hardware faults")
    parser.add_argument("--auto_search", action="store_true", help="Auto-trigger SEARCH (Action 8) when standing on target cell")
    parser.add_argument("--max_battery", type=int, default=125, help="Battery capacity (e.g., 750 for unlimited flight)")
    parser.add_argument("--explore", action="store_true", help="Force high-battery drones (>40%) to range outward toward target probability gradient")
    args = parser.parse_args()

    print("=" * 70)
    print(f"   LIVE PYGAME DEMO: DYNAMIC TARGET TRACKING ({args.model.upper()})")
    print("=" * 70)

    is_self_heal = "selfheal" in args.model
    fault_prob = 0.0 if args.no_faults else 0.0005

    # --- Build policy function ---
    if args.model == "heuristic":
        print("Using greedy heuristic policy (follows probability gradient)")
        policy_fn = make_heuristic_policy(grid_size=25)
        # Start drones spread around the target's initial position (12, 12)
        drone_positions = [(8, 8), (8, 16), (16, 8), (16, 16)]
    else:
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
            print("[!] Falling back to heuristic greedy policy.\n")
            policy_fn = make_heuristic_policy(grid_size=25)
        else:
            print(f"Loading {framework.upper()} Checkpoint: {ckpt_path}")
            if framework == "rllib":
                policy_fn = load_rllib_policy(model_cls, ckpt_path, model_name=args.model)
            else:
                policy_fn = load_epymarl_policy(ckpt_path)

        # Drones spawn at top-left corner [(0,0), (0,1), (1,0), (1,1)] near the battery station, matching training distribution
        drone_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    if args.explore:
        print("  [Outward Exploration ACTIVE] High-battery drones (>40%) will actively push outward following target probability gradient.")
        base_policy = policy_fn
        def explore_policy(obs_dict, agents, *args_fn, **kwargs_fn):
            actions = base_policy(obs_dict, agents, *args_fn, **kwargs_fn)
            for agent in agents:
                if agent in obs_dict:
                    pos_vec, matrix = obs_dict[agent]
                    p0, p1 = float(pos_vec[0]), float(pos_vec[1])
                    dy = int(round(p0 * 25.0)) if p0 <= 1.0 else int(round(p0))
                    dx = int(round(p1 * 25.0)) if p1 <= 1.0 else int(round(p1))
                    dy, dx = max(0, min(24, dy)), max(0, min(24, dx))

                    # If battery > 40 and standing on low prob cell, push toward neighbor with highest target probability
                    curr_p = matrix[dy, dx]
                    if curr_p < 0.005 and actions[agent] == 8: # If idling/searching on low-prob cell
                        best_act = 8
                        best_p = curr_p
                        for (ndx, ndy), act_val in DIRECTION_TO_ACTION.items():
                            if ndx == 0 and ndy == 0:
                                continue
                            ny, nx = dy + ndy, dx + ndx
                            if 0 <= ny < 25 and 0 <= nx < 25:
                                np_val = matrix[ny, nx]
                                if np_val > best_p + 1e-6:
                                    best_p = np_val
                                    best_act = act_val
                        actions[agent] = best_act
            return actions
        policy_fn = explore_policy

    if args.auto_search:
        print("  [Auto-Search Trigger ACTIVE] Onboard sensors will auto-trigger SEARCH (Action 8) when over target cells.")
        raw_fn = policy_fn
        def auto_search_fn(obs_dict, agents, *args_fn, **kwargs_fn):
            actions = raw_fn(obs_dict, agents, *args_fn, **kwargs_fn)
            base_e = env.unwrapped if 'env' in locals() or 'env' in globals() else None
            for agent in agents:
                if agent in obs_dict:
                    pos_vec, matrix = obs_dict[agent]
                    p0, p1 = float(pos_vec[0]), float(pos_vec[1])
                    dy = int(round(p0 * 25.0)) if p0 <= 1.0 else int(round(p0))
                    dx = int(round(p1 * 25.0)) if p1 <= 1.0 else int(round(p1))
                    dy = max(0, min(24, dy))
                    dx = max(0, min(24, dx))

                    curr_p = matrix[dy, dx] if (0 <= dy < matrix.shape[0] and 0 <= dx < matrix.shape[1]) else 0.0
                    
                    target_match = False
                    if base_e is not None and hasattr(base_e, "person_position"):
                        tp = base_e.person_position
                        if tp is not None:
                            # In DSSE, person_position is (x, y) where x is col, y is row
                            if (dx == tp[0] and dy == tp[1]) or (dy == tp[0] and dx == tp[1]):
                                target_match = True

                    if curr_p >= 0.005 or target_match:
                        actions[agent] = 8 # SEARCH
            return actions
        policy_fn = auto_search_fn

    # --- Create environment ---
    env = create_live_tracking_env(
        is_self_heal=is_self_heal,
        fault_prob=fault_prob,
        drone_positions=drone_positions,
        max_battery=args.max_battery,
    )
    obs_dict, infos = env.reset(seed=args.seed)

    step = 0
    target_found = False

    # Enable grid lines and set window caption
    import pygame
    base_env = env.unwrapped
    if hasattr(base_env, "pygame_renderer") and base_env.pygame_renderer is not None:
        base_env.pygame_renderer.render_grid = True

    try:
        if pygame.get_init() and pygame.display.get_init():
            pygame.display.set_caption(f"MARL Target Tracking Demo — {args.model.upper()}")
    except Exception:
        pass

    clock = pygame.time.Clock()

    print(f"\nPygame Window opened! Press Ctrl+C in Terminal to stop or close the Pygame window.")
    print(f"  FPS: {args.fps} | Seed: {args.seed} | Faults: {'OFF' if args.no_faults else 'ON'}\n")

    try:
        while env.agents and step < 750 and not target_found:
            # Handle Pygame window events (quit, etc.)
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nPygame Window closed by user.")
                        pygame.quit()
                        return
            except pygame.error:
                # Pygame not initialized or window closed
                break

            step += 1
            actions = policy_fn(obs_dict, env.agents, reset=(step == 1))

            try:
                obs_dict, rewards, term, trunc, infos = env.step(actions)
            except pygame.error:
                # Window closed during step
                break

            # Check target discovery via the DSSE native reward (1.0 on search_and_find)
            for agent, r in rewards.items():
                info = infos.get(agent, {})
                if (isinstance(info, dict) and info.get("Found", False)):
                    target_found = True
                    print(f"🎯 TARGET INTERCEPTED AT STEP {step}!")
                    break

            # Maintain smooth FPS
            clock.tick(max(1, args.fps))

            if step % 25 == 0:
                alive_count = sum(1 for a in env.agents if obs_dict.get(a) is not None)
                positions = []
                for i, a in enumerate(env.agents):
                    pos = base_env.agents_positions[i] if i < len(base_env.agents_positions) else "?"
                    positions.append(f"{a}={pos}")
                pos_str = ", ".join(positions[:2]) + ("..." if len(positions) > 2 else "")
                print(f"  Step {step:3d}/750 | Alive: {alive_count}/4 | Target: {target_found} | {pos_str}")

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")
    except Exception as e:
        if "video system not initialized" in str(e):
            print("\nPygame Window closed.")
        else:
            print(f"\nDemo error: {e}")
    finally:
        try:
            pygame.quit()
        except Exception:
            pass

    result = f"TARGET FOUND in {step} steps" if target_found else f"TARGET ESCAPED ({step}/750 steps)"
    print(f"\nDemo Complete! Final Result: {result}")


if __name__ == "__main__":
    main()
