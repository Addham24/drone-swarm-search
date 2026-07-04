import argparse
"""
MAPPO Vanilla Test Script for Drone Swarm Coverage
====================================================
Tests a trained MAPPO Vanilla checkpoint (no self-healing features).

Usage:
    python test_mappo_vanilla.py
"""

import os
import time
import pathlib
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from train_mappo_vanilla import CNNModel

# ANSI color codes for status log
class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def env_creator(render_mode="human"):
    print("-------------------------- MAPPO VANILLA ENV CREATOR (TEST) --------------------------")
    N_AGENTS = 4
    matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")
    if render_mode is None:
        env = CoverageDroneSwarmSearch(
            timestep_limit=750, 
            drone_amount=N_AGENTS, 
            prob_matrix_path=matrix_path
        )
    else:
        env = CoverageDroneSwarmSearch(
            timestep_limit=750, 
            drone_amount=N_AGENTS, 
            prob_matrix_path=matrix_path,
            render_mode=render_mode
        )
    env.reward_scheme = {
        "default": -0.1,
        "exceed_timestep": 0.0,
        "search_cell": 5.0,
        "done": 500.0,
        "reward_poc": 0.0,
    }
    env = AllPositionsWrapper(env)

    # VANILLA: No fault injection, no teammate compensation
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.01)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0

    grid_size = env.grid_size
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


def print_status_log(step, infos):
    """Print a per-step status log showing battery levels and events."""
    batt_parts = []
    events = []
    for agent, info in sorted(infos.items()):
        if not isinstance(info, dict):
            continue
        batt = info.get("battery", "?")
        alive = info.get("alive", True)
        charging = info.get("charging", False)
        if not alive:
            status = f"{Colors.RED}DEAD{Colors.RESET}"
        elif charging:
            status = f"{Colors.GREEN}CHG{Colors.RESET}"
        else:
            status = f"{Colors.YELLOW}FLY{Colors.RESET}"
        batt_parts.append(f"{agent}:{batt}/{status}")

        if info.get("stranded_event", False):
            crash = info.get("crash_coords")
            crash_str = f" at ({crash[0]}, {crash[1]})" if crash else ""
            events.append(
                f"{Colors.RED}{Colors.BOLD}  ⚠ [CRASHED] {agent.upper()} RAN OUT OF BATTERY{crash_str}!{Colors.RESET}"
            )

    print(f"  Step {step:>4}: {Colors.CYAN}[BATTERY]{Colors.RESET} {' | '.join(batt_parts)}")
    for event in events:
        print(event)


if __name__ == "__main__":
    ray.init()
    
    ModelCatalog.register_custom_model("CNNModel", CNNModel)
    register_env("DSSE_Coverage", lambda config: ParallelPettingZooEnv(env_creator(render_mode=None)))
    
    curr_path = pathlib.Path().resolve()
    print("Please enter the absolute path to the MAPPO Vanilla checkpoint.")
    print(f"It should be located somewhere in {curr_path}/ray_res/")
    checkpoint_path = input("Checkpoint path: ").strip()
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit(1)
        
    algo = Algorithm.from_checkpoint(checkpoint_path)
    env = env_creator(render_mode="human")
    observations, infos = env.reset()
    
    total_rewards = {agent: 0.0 for agent in env.agents}
    step_count = 0

    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  MAPPO VANILLA TEST — 4 Drones on 25x25 Grid")
    print(f"  Battery: 125 Max (Deplete:-1 | Charge:+15)")
    print(f"  Self-Healing: OFF (No faults, No compensation)")
    print(f"{'='*60}{Colors.RESET}\n")
    
    while env.agents:
        step_count += 1
        actions = {}
        for agent in env.agents:
            if agent in observations:
                actions[agent] = algo.compute_single_action(
                    observation=observations[agent],
                    policy_id="default_policy"
                )
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print_status_log(step_count, infos)
        
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
            
        time.sleep(0.1)

    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  MAPPO VANILLA TEST FINISHED — {step_count} steps")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n  {Colors.BOLD}Total Rewards:{Colors.RESET}")
    for agent, reward in total_rewards.items():
        status = f"{Colors.GREEN}ALIVE{Colors.RESET}" if infos.get(agent, {}).get("alive", True) else f"{Colors.RED}STRANDED{Colors.RESET}"
        print(f"    {agent}: {reward:>8.2f}  [{status}]")
    
    sample_info = next(iter(infos.values()), {})
    if isinstance(sample_info, dict):
        coverage = sample_info.get("coverage_rate", 0)
        print(f"\n  {Colors.BOLD}Coverage Rate:{Colors.RESET} {coverage*100:.1f}%")
    
    print()
    ray.shutdown()
