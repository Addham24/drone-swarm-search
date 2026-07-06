#!/usr/bin/env python3
"""
EPyMARL Training Launcher for DSSE Coverage Drone Swarm
========================================================
Standalone script that:
  1. Clones EPyMARL (if not already present)
  2. Registers the DSSE environment adapter
  3. Launches training with any EPyMARL-supported algorithm

Supported algorithms (out of the box):
  - qmix      : True QMIX with neural mixing network
  - vdn       : Value Decomposition Networks
  - iql       : Independent Q-Learning
  - coma      : Counterfactual Multi-Agent Policy Gradients
  - mappo     : Multi-Agent PPO (centralized critic)
  - ippo      : Independent PPO
  - maa2c     : Multi-Agent Advantage Actor-Critic
  - ia2c      : Independent A2C
  - pac_ns    : Pareto Actor-Critic (no sharing)

Usage:
    # Train with true QMIX:
    python epymarl_train.py --algo qmix --name "qmix_selfhealing_v1"

    # Train with COMA:
    python epymarl_train.py --algo coma --name "coma_baseline_v1"

    # Train with MAPPO:
    python epymarl_train.py --algo mappo --name "mappo_epymarl_v1"

    # Custom timesteps:
    python epymarl_train.py --algo qmix --name "qmix_short" --t_max 5000000

    # With individual rewards (for algorithms that support it):
    python epymarl_train.py --algo mappo --name "mappo_indiv" --individual_rewards
"""

import argparse
import os
import subprocess
import sys


EPYMARL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epymarl")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_epymarl():
    """Clone EPyMARL if not already present, and install its dependencies."""
    is_cloned = os.path.isdir(EPYMARL_DIR)
    if not is_cloned:
        print("[→] Cloning EPyMARL from GitHub...")
        subprocess.check_call([
            "git", "clone", "https://github.com/uoe-agents/epymarl.git", EPYMARL_DIR
        ])
        print(f"[✓] EPyMARL cloned to: {EPYMARL_DIR}")

    # CRITICAL: Copy custom modifications (like cnn_agent) into the epymarl structure
    custom_dir = os.path.join(SRC_DIR, "epymarl_custom")
    if os.path.isdir(custom_dir):
        print("[→] Applying custom EPyMARL modifications from epymarl_custom...")
        import shutil
        for root, dirs, files in os.walk(custom_dir):
            for file in files:
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, custom_dir)
                dest_path = os.path.join(EPYMARL_DIR, "src", rel_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"    Copied: {rel_path} -> epymarl/src/{rel_path}")
        print("[✓] Custom EPyMARL modifications applied.")

    if not is_cloned:
        # Install EPyMARL core dependencies
        req_file = os.path.join(EPYMARL_DIR, "requirements.txt")
        if os.path.isfile(req_file):
            print("[→] Installing EPyMARL dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("[✓] EPyMARL dependencies installed.")


def register_dsse_env():
    """
    Register our DSSE environment adapter with EPyMARL's environment registry.
    This injects the adapter into EPyMARL's REGISTRY so it can be used
    with `--env-config=dsse`.
    """
    # Add our src directory to Python path so imports work
    if SRC_DIR not in sys.path:
        sys.path.insert(0, SRC_DIR)

    epymarl_src = os.path.join(EPYMARL_DIR, "src")
    if epymarl_src not in sys.path:
        sys.path.insert(0, epymarl_src)

    # Import EPyMARL's registry and our adapter
    from envs import REGISTRY
    from epymarl_env_adapter import DSSEMultiAgentEnv

    def dsse_fn(**kwargs) -> DSSEMultiAgentEnv:
        """Factory function for EPyMARL environment creation."""
        # Extract EPyMARL-specific params
        common_reward = kwargs.pop("common_reward", True)
        reward_scalarisation = kwargs.pop("reward_scalarisation", "sum")
        seed = kwargs.pop("seed", None)

        return DSSEMultiAgentEnv(
            common_reward=common_reward,
            reward_scalarisation=reward_scalarisation,
            seed=seed,
            **kwargs,
        )

    REGISTRY["dsse"] = dsse_fn
    print("[✓] DSSE environment registered with EPyMARL.")


def write_dsse_env_config(timestep_limit=750):
    """
    Write the YAML environment config for EPyMARL.
    This tells EPyMARL how to instantiate the DSSE environment.
    """
    config_dir = os.path.join(EPYMARL_DIR, "src", "config", "envs")
    config_path = os.path.join(config_dir, "dsse.yaml")

    os.makedirs(config_dir, exist_ok=True)

    config_content = f"""# DSSE Coverage Drone Swarm Environment Config for EPyMARL
env: "dsse"

env_args:
  map_name: "dsse_coverage"
  # Environment parameters
  n_agents: 4
  grid_size: 25
  timestep_limit: {timestep_limit}
  # Battery parameters
  max_battery: 125
  depletion_rate: 1
  charge_rate: 15
  fault_prob: 0.0005
  # Reward mixing (VDN-style selective mixing)
  mixing_alpha: 0.5
  # Self-healing compensation
  compensation_bonus: 2.5
  compensation_penalty: -0.5
  compensation_horizon: 100
  seed: 1
"""
    with open(config_path, "w") as f:
        f.write(config_content)
    print(f"[✓] DSSE env config written to: {config_path} with timestep_limit={timestep_limit}")


def run_training(algo, exp_name, t_max, individual_rewards, agent="rnn", episode_limit=750, extra_args=[]):
    """Launches EPyMARL with the bootstrap registry wrapper."""
    main_script = os.path.join(EPYMARL_DIR, "src", "main.py")
    if not os.path.exists(main_script):
        raise FileNotFoundError(f"EPyMARL main.py not found at: {main_script}")

    # Sacred command-line format
    cmd = [
        sys.executable, main_script,
        f"--config={algo}",
        "--env-config=dsse",
        "with",
        f"t_max={t_max}",
        f"name={exp_name}",
        f"agent={agent}",
        "use_tensorboard=True",
        "save_model=True",
        "save_model_interval=50000",
    ]

    # Dynamically scale buffer_size for off-policy algorithms to prevent 78GB RAM pre-allocation
    # Each episode of 750 steps with a 647-float obs takes ~15.75MB, so:
    # - 5000 episodes = 78.75 GB RAM
    # - 200 episodes = 3.15 GB RAM (Perfect for consumer laptops!)
    if algo in ["qmix", "vdn", "iql", "qmix_ns", "vdn_ns", "iql_ns"]:
        cmd.append("buffer_size=200")
        
        # Scale exploration decay based on agent architecture and episode limit:
        # - CNN Agent has natural spatial bias and weight sharing, learning grid geometry 10x faster.
        #   We can safely decay epsilon in exactly 100 episodes (100 * episode_limit) so coverage rate climbs immediately!
        # - RNN/MLP Agent is geometry-blind and needs longer (333 * episode_limit) to learn grid adjacency.
        episodes = 100 if agent == "cnn" else 333
        anneal_steps = episodes * episode_limit
        cmd.append(f"epsilon_anneal_time={anneal_steps}")
        
        # Reduce batch_size from 32 episodes to 8 episodes.
        # Since each episode is episode_limit steps, batch_size=8 executes 4x faster on CPU!
        cmd.append("batch_size=8")
        
        print("[!] Off-policy algorithm detected.")
        print(f"    -> Scaling 'buffer_size' to 200 episodes (~3.1 GB RAM)")
        print(f"    -> Tailoring 'epsilon_anneal_time' to {anneal_steps:,} steps ({episodes} episodes of exploration) for {agent.upper()} agent")
        print(f"    -> Optimizing 'batch_size' to 8 episodes for 4x faster CPU training execution")

    if individual_rewards:
        cmd.append("common_reward=False")

    # Append any extra sacred-style overrides
    cmd.extend(extra_args)

    print(f"\n{'='*70}")
    print(f"  EPyMARL Training: {exp_name}")
    print(f"  Algorithm: {algo.upper()}")
    print(f"  Agent Architecture: {agent.upper()}")
    print(f"  Environment: DSSE Coverage (4 drones, 25x25 grid)")
    print(f"  Timesteps: {t_max:,}")
    print(f"  Individual Rewards: {individual_rewards}")
    print(f"{'='*70}\n")

    # We need to inject our registration before EPyMARL's main.py runs.
    # The cleanest way is to set PYTHONPATH and use a bootstrap approach.
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{os.path.join(EPYMARL_DIR, 'src')}:{env.get('PYTHONPATH', '')}"

    # Create a bootstrap script that registers our env then calls EPyMARL's main
    bootstrap_path = os.path.join(SRC_DIR, "_epymarl_bootstrap.py")
    bootstrap_content = f"""import sys
import os

# Workaround for old tensorboard-logger protobuf issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Ensure paths
sys.path.insert(0, "{SRC_DIR}")
sys.path.insert(0, "{os.path.join(EPYMARL_DIR, 'src')}")

# Register DSSE environment
from envs import REGISTRY
from epymarl_env_adapter import DSSEMultiAgentEnv

def dsse_fn(**kwargs):
    common_reward = kwargs.pop("common_reward", True)
    reward_scalarisation = kwargs.pop("reward_scalarisation", "sum")
    seed = kwargs.pop("seed", None)
    return DSSEMultiAgentEnv(
        common_reward=common_reward,
        reward_scalarisation=reward_scalarisation,
        seed=seed,
        **kwargs,
    )

REGISTRY["dsse"] = dsse_fn
print("[✓] DSSE environment registered with EPyMARL (via bootstrap).")

# Now run EPyMARL's main with overridden __file__ in globals
if __name__ == "__main__":
    sys.argv[0] = "{main_script}"
    g = dict(globals())
    g["__file__"] = "{main_script}"
    exec(open("{main_script}").read(), g)
"""
    with open(bootstrap_path, "w") as f:
        f.write(bootstrap_content)

    # Replace main script with bootstrap
    cmd[1] = bootstrap_path

    print(f"[→] Running: {' '.join(cmd)}\n")
    subprocess.call(cmd, env=env)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EPyMARL Training Launcher for DSSE Coverage Drone Swarm"
    )
    parser.add_argument(
        "--algo", type=str, default="qmix",
        choices=["qmix", "vdn", "iql", "coma", "mappo", "ippo", "maa2c", "ia2c", "pac_ns"],
        help="MARL algorithm to train with (default: qmix)"
    )
    parser.add_argument(
        "--name", type=str, default="dsse_experiment",
        help="Experiment name for logging (default: dsse_experiment)"
    )
    parser.add_argument(
        "--agent", type=str, default="rnn",
        choices=["rnn", "cnn"],
        help="Agent model architecture (default: rnn)"
    )
    parser.add_argument(
        "--t_max", type=int, default=20_000_000,
        help="Total training timesteps (default: 20,000,000)"
    )
    parser.add_argument(
        "--episode_limit", type=int, default=750,
        help="Timestep limit per episode during training (default: 750)"
    )
    parser.add_argument(
        "--individual_rewards", action="store_true",
        help="Use individual rewards instead of common reward (only for supported algorithms)"
    )
    parser.add_argument(
        "--disable_self_healing", action="store_true",
        help="Disable the self-healing/attrition compensation mechanic (sets compensation_horizon=0)"
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, extra_args = parse_args()

    # Apply the disable self healing override if requested
    if args.disable_self_healing:
        extra_args.append("env_args.compensation_horizon=0")

    print(f"\n{'='*70}")
    print(f"  EPyMARL DSSE Training Pipeline")
    print(f"{'='*70}\n")

    # Step 1: Ensure EPyMARL is cloned
    ensure_epymarl()

    # Step 2: Write DSSE environment YAML config with custom episode limit
    write_dsse_env_config(timestep_limit=args.episode_limit)

    # Step 3: Run training
    run_training(
        algo=args.algo,
        exp_name=args.name,
        t_max=args.t_max,
        individual_rewards=args.individual_rewards,
        agent=args.agent,
        episode_limit=args.episode_limit,
        extra_args=extra_args,
    )


