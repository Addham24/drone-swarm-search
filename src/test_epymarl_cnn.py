#!/usr/bin/env python3
import os
import sys
import argparse
import time
import pathlib
import torch as th
import torch.nn.functional as F
import numpy as np

# Add src and epymarl/src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from global_reward_wrapper import GlobalRewardWrapper
from modules.agents.cnn_agent import CNNAgent

# ANSI color codes for status log
class Colors:
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


class MockArgs:
    def __init__(self, use_rnn=True, hidden_dim=128, n_actions=9):
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions


def build_env(render_mode="human", disable_self_healing=False):
    matrix_path = os.path.join(SRC_DIR, "uniform_matrix_25.npy")
    
    if render_mode is None or render_mode == "none":
        env = CoverageDroneSwarmSearch(
            timestep_limit=750,
            drone_amount=4,
            prob_matrix_path=matrix_path
        )
    else:
        env = CoverageDroneSwarmSearch(
            timestep_limit=750,
            drone_amount=4,
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
    env = BatteryStationWrapper(
        env,
        max_battery=125,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=0.0005,
    )

    # Self-healing parameters
    env.COMPENSATION_BONUS = 2.5
    env.COMPENSATION_PENALTY = -0.5
    env.COMPENSATION_HORIZON = 0 if disable_self_healing else 100

    env = GlobalRewardWrapper(env, mixing_alpha=0.5)

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate EPyMARL CNN Agent checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory (containing agent.th)")
    parser.add_argument("--mode", type=str, default="greedy", choices=["greedy", "stochastic"],
                        help="Action selection mode: greedy (argmax) or stochastic (multinomial sample)")
    parser.add_argument("--render-mode", type=str, default="human", choices=["human", "none", "rgb_array"],
                        help="Render mode for the environment")
    parser.add_argument("--disable-self-healing", action="store_true",
                        help="Disable self-healing (sets compensation horizon to 0)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="Delay between render steps in seconds")
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension of the GRU cell (default: 128)")
    args = parser.parse_args()

    agent_path = os.path.join(args.checkpoint, "agent.th")
    if not os.path.isfile(agent_path):
        print(f"{Colors.RED}Error: checkpoint file agent.th not found at {agent_path}{Colors.RESET}")
        sys.exit(1)

    print(f"{Colors.GREEN}[→] Building environment...{Colors.RESET}")
    env = build_env(render_mode=None if args.render_mode == "none" else args.render_mode,
                    disable_self_healing=args.disable_self_healing)
    
    # Initialize agent model
    # Inputs: obs (647) + one-hot agent ID (4) = 651
    mock_args = MockArgs(use_rnn=True, hidden_dim=args.hidden_dim, n_actions=9)
    agent = CNNAgent(input_shape=651, args=mock_args)
    
    print(f"{Colors.GREEN}[→] Loading checkpoint from {agent_path}...{Colors.RESET}")
    state_dict = th.load(agent_path, map_location=lambda storage, loc: storage)
    agent.load_state_dict(state_dict)
    agent.eval()
    
    # Run evaluation episode
    obs_dict, info_dict = env.reset()
    
    # EPyMARL constructs agents list in standard order:
    agents = list(env.possible_agents)
    n_agents = len(agents)
    
    # Initialize hidden states
    hidden_states = agent.init_hidden()  # returns shape (1, hidden_dim)
    # Expand hidden states to (n_agents, hidden_dim)
    hidden_states = hidden_states.expand(n_agents, -1).clone()
    
    # Flat observation size (positions = 22, matrix = 625)
    positions_size = 22
    matrix_size = 625
    obs_size = positions_size + matrix_size
    
    total_rewards = {a: 0.0 for a in agents}
    step_count = 0
    
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  EPYMARL CNN EVALUATION STARTED — 4 Drones on 25x25 Grid")
    print(f"  Checkpoint: {os.path.basename(args.checkpoint)} (step {os.path.basename(args.checkpoint.rstrip('/'))})")
    print(f"  Action Mode: {args.mode.upper()} | Self-Healing: {'OFF' if args.disable_self_healing else 'ON'}")
    print(f"{'='*60}{Colors.RESET}\n")

    # Run episode loop
    while env.agents:
        step_count += 1
        
        # Prepare inputs for all agents in batch
        # We construct inputs by concatenating obs with agent ID
        inputs = []
        for i, agent_name in enumerate(agents):
            if agent_name in obs_dict:
                positions, matrix = obs_dict[agent_name]
                flat_obs = np.concatenate([
                    positions.astype(np.float32).flatten(),
                    matrix.astype(np.float32).flatten()
                ])
            else:
                flat_obs = np.zeros(obs_size, dtype=np.float32)
            
            # One-hot agent ID
            agent_id_onehot = np.zeros(n_agents, dtype=np.float32)
            agent_id_onehot[i] = 1.0
            
            full_input = np.concatenate([flat_obs, agent_id_onehot])
            inputs.append(full_input)
            
        inputs_tensor = th.tensor(np.stack(inputs), dtype=th.float32)
        
        # Forward pass through CNN Agent
        with th.no_grad():
            q_logits, hidden_states = agent(inputs_tensor, hidden_states)
            
            if args.mode == "greedy":
                actions_tensor = q_logits.argmax(dim=-1)
            else:
                # Stochastic sampling from policy logits
                probs = F.softmax(q_logits, dim=-1)
                m = th.distributions.Categorical(probs)
                actions_tensor = m.sample()
                
        actions = {}
        for i, agent_name in enumerate(agents):
            if agent_name in env.agents:
                actions[agent_name] = int(actions_tensor[i].item())
                
        # Step the environment
        obs_dict, rewards, terminations, truncations, info_dict = env.step(actions)
        
        # Accumulate rewards
        for agent_name, reward in rewards.items():
            total_rewards[agent_name] += reward
            
        # Logging battery metrics
        print_status_log(step_count, info_dict)
        
        if args.render_mode == "human":
            time.sleep(args.delay)

    # --- Episode Summary ---
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"  EPISODE FINISHED — {step_count} steps")
    print(f"{'='*60}{Colors.RESET}")
    print(f"\n  {Colors.BOLD}Total Rewards:{Colors.RESET}")
    for agent_name, reward in total_rewards.items():
        status = f"{Colors.GREEN}ALIVE{Colors.RESET}" if info_dict.get(agent_name, {}).get("alive", True) else f"{Colors.RED}STRANDED{Colors.RESET}"
        print(f"    {agent_name}: {reward:>8.2f}  [{status}]")
    
    # Coverage info
    sample_info = next(iter(info_dict.values()), {})
    if isinstance(sample_info, dict):
        coverage = sample_info.get("coverage_rate", 0)
        print(f"\n  {Colors.BOLD}Coverage Rate:{Colors.RESET} {coverage*100:.1f}%")
    
    print()
    env.close()

if __name__ == "__main__":
    main()
