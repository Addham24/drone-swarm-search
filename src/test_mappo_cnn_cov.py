import os
import argparse
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
from train_mappo_cnn_cov import CNNModel

# ANSI color codes for status log
class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def env_creator(render_mode="human"):
    print("-------------------------- ENV CREATOR (TEST) --------------------------")
    N_AGENTS = 4
    # 25x25 grid for coverage
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
    # Override reward scheme with fixed constants (no POC/time scaling)
    env.reward_scheme = {
        "default": -0.1,
        "exceed_timestep": 0.0,
        "search_cell": 5.0,
        "done": 500.0,
        "reward_poc": 0.0,
    }
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.01)
    grid_size = env.grid_size    # All drones start in a 2x2 cluster at the top-left corner (the base station)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


def print_status_log(step, infos):
    """Print a per-step status log showing battery levels and events."""
    # Build battery dashboard for every step
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

        # Crash events
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
    
    # Register the custom model
    ModelCatalog.register_custom_model("CNNModel", CNNModel)
    
    # Register the custom environment to prevent the checkpoint load from failing
    register_env("DSSE_Coverage", lambda config: ParallelPettingZooEnv(env_creator(render_mode=None)))
    
    # Checkpoint path
    curr_path = pathlib.Path().resolve()
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--prob-attrition", type=float, default=0.003, help="Probability of agent attrition per step")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        print(f"It should be located somewhere in {curr_path}/ray_res/")
        checkpoint_path = input("Checkpoint path: ").strip()
    
    prob_attrition = args.prob_attrition
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        exit(1)
        
    # Load the trained algorithm
    algo = Algorithm.from_checkpoint(checkpoint_path)
    
    # Create the environment with rendering enabled
    env = env_creator(render_mode="human")
    
    # Run a test episode
    observations, infos = env.reset()
    
    total_rewards = {agent: 0.0 for agent in env.agents}
    step_count = 0

    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  TEST EPISODE STARTED — 4 Drones on 25x25 Grid")
    print(f"  Battery: 125 Max (Deplete:-1 | Charge:+15) | Fixed Rewards | Centre Spawn")
    print(f"{'='*60}{Colors.RESET}\n")
    
    while env.agents:
        step_count += 1
        
        # Compute actions for all available agents
        actions = {}
        for agent in env.agents:
            if agent in observations:
                actions[agent] = algo.compute_single_action(
                    observation=observations[agent],
                    policy_id="default_policy"
                )
        
        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        # Print status log for this step
        print_status_log(step_count, infos)
        
        # Accumulate rewards
        for agent, reward in rewards.items():
            total_rewards[agent] += reward
            
        time.sleep(0.1)  # Sleep briefly to make visualization perceivable

    # --- Episode Summary ---
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  TEST EPISODE FINISHED — {step_count} steps")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n  {Colors.BOLD}Total Rewards:{Colors.RESET}")
    for agent, reward in total_rewards.items():
        status = f"{Colors.GREEN}ALIVE{Colors.RESET}" if infos.get(agent, {}).get("alive", True) else f"{Colors.RED}STRANDED{Colors.RESET}"
        print(f"    {agent}: {reward:>8.2f}  [{status}]")
    
    # Coverage info from last infos
    sample_info = next(iter(infos.values()), {})
    if isinstance(sample_info, dict):
        coverage = sample_info.get("coverage_rate", 0)
        print(f"\n  {Colors.BOLD}Coverage Rate:{Colors.RESET} {coverage*100:.1f}%")
    
    print()
    ray.shutdown()
