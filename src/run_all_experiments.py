"""
Run All 4 Comparison Experiments Sequentially
==============================================
Trains all 4 configurations back-to-back with the same timestep budget.
Results are saved to ray_res/DSSE_Coverage/ and can be compared in TensorBoard.

Usage:
    python run_all_experiments.py

    # Custom timestep budget (default: 5M)
    python run_all_experiments.py --timesteps 10000000

    # Skip experiments that already have results
    python run_all_experiments.py --skip-existing
"""

import os
import sys
import argparse
import pathlib
import time

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

# --- MONKEY-PATCH FOR RAY RLLIB BUG ---
import ray.rllib.algorithms.algorithm as algo_mod
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.replay_buffers import ReplayBuffer

def safe_create_buffer(self, config):
    if not config.get("replay_buffer_config") or config["replay_buffer_config"].get("no_local_replay_buffer"):
        return None
    
    typ = config["replay_buffer_config"].get("type")
    if isinstance(typ, str) and "EpisodeReplayBuffer" in typ:
        config["replay_buffer_config"]["metrics_num_episodes_for_smoothing"] = \
            self.config.metrics_num_episodes_for_smoothing
    
    return from_config(ReplayBuffer, config["replay_buffer_config"])

algo_mod.Algorithm._create_local_replay_buffer_if_necessary = safe_create_buffer
# ---------------------------------------

# Import all env creators and models
from train_mappo_cnn_cov import (
    env_creator as mappo_sh_env_creator,
    CNNModel,
    BatteryMetricsCallback,
)
from train_mappo_vanilla import (
    env_creator as mappo_vanilla_env_creator,
    CNNModel as VanillaCNNModel,
)
from train_qmix_cnn_cov import (
    env_creator as qmix_sh_env_creator,
    QMIXCNNModel,
)
from train_qmix_vanilla import (
    env_creator as qmix_vanilla_env_creator,
    QMIXVanillaCNNModel,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run all 4 comparison experiments")
    parser.add_argument(
        "--timesteps", type=int, default=5_000_000,
        help="Total timesteps per experiment (default: 5M)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip experiments that already have result directories"
    )
    return parser.parse_args()


def print_banner(num, name, algo, self_healing):
    sh_str = "ON" if self_healing else "OFF"
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT {num}/4: {name}")
    print(f"  Algorithm: {algo} | Self-Healing: {sh_str}")
    print(f"{'='*70}\n")


def run_experiment(exp_num, exp_name, algo_name, config, curr_path, env_name, timesteps):
    """Run a single experiment."""
    print_banner(exp_num, exp_name, algo_name, "selfheal" in exp_name.lower() or "sh" in exp_name.lower())

    start_time = time.time()

    tune.run(
        algo_name,
        name=exp_name,
        stop={"timesteps_total": timesteps},
        checkpoint_at_end=True,
        checkpoint_freq=0,  # Disable periodic checkpoints (they trigger storage validation crash)
        storage_path=f"{curr_path}/ray_res/{env_name}",
        config=config,
        verbose=3,
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    mins = int((elapsed % 3600) // 60)
    print(f"\n  ✅ {exp_name} completed in {hours}h {mins}m")


def _build_dqn_config(env_name, model_name):
    """Build a DQN config with epsilon-greedy exploration.
    
    Anti-Clumping Hyperparameters:
    - Higher final epsilon (0.1) to force diversity
    - Slower target updates (10k) for maximum stability
    - Gradient clipping to prevent CNN saturation
    """
    config = (
        DQNConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=env_name)
        .env_runners(num_env_runners=4, rollout_fragment_length=200)
        .training(
            train_batch_size=512, lr=1e-4, gamma=0.998,
            num_steps_sampled_before_learning_starts=100000,
            dueling=True, double_q=True, n_step=1,
            target_network_update_freq=10000,
            grad_clip=10.0,
            model={"custom_model": model_name, "_disable_preprocessor_api": True},
        )
        .callbacks(BatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=0)
    )
    # MANUALLY SET REPLAY BUFFER TO BYPASS ABCMeta BUG
    config.replay_buffer_config = {
        "type": "MultiAgentReplayBuffer",
        "capacity": 500_000,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "prioritized_replay_beta": 0.4,
    }
    config.exploration_config = {
        "type": "SoftQ",
        "temperature": 1.0,
    }
    return config


if __name__ == "__main__":
    args = parse_args()
    timesteps = args.timesteps

    curr_path = pathlib.Path().resolve()
    env_name = "DSSE_Coverage"
    results_dir = f"{curr_path}/ray_res/{env_name}"

    print(f"\n{'#'*70}")
    print(f"  DRONE SWARM COMPARISON STUDY")
    print(f"  Timesteps per experiment: {timesteps:,}")
    print(f"  Results directory: {results_dir}")
    print(f"{'#'*70}\n")

    # Define all 4 experiments
    experiments = [
        # {
        #     "num": 1,
        #     "name": "MAPPO_selfheal_comparison",
        #     "algo": "PPO",
        #     "env_creator": mappo_sh_env_creator,
        #     "model_name": "CNNModel",
        #     "model_class": CNNModel,
        #     "config_builder": lambda: (
        #         PPOConfig()
        #         .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        #         .environment(env=env_name)
        #         .env_runners(num_env_runners=4, rollout_fragment_length="auto")
        #         .training(
        #             train_batch_size=8192, lr=1e-4, gamma=0.998, lambda_=0.9,
        #             use_gae=True, entropy_coeff=0.05, vf_clip_param=100000,
        #             minibatch_size=300, num_sgd_iter=10,
        #             model={"custom_model": "CNNModel", "_disable_preprocessor_api": True},
        #         )
        #         .callbacks(BatteryMetricsCallback)
        #         .experimental(_disable_preprocessor_api=True)
        #         .debugging(log_level="ERROR")
        #         .framework(framework="torch")
        #         .resources(num_gpus=0)
        #     ),
        # },
        # {
        #     "num": 2,
        #     "name": "MAPPO_vanilla_comparison",
        #     "algo": "PPO",
        #     "env_creator": mappo_vanilla_env_creator,
        #     "model_name": "CNNModel",
        #     "model_class": VanillaCNNModel,
        #     "config_builder": lambda: (
        #         PPOConfig()
        #         .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        #         .environment(env=env_name)
        #         .env_runners(num_env_runners=4, rollout_fragment_length="auto")
        #         .training(
        #             train_batch_size=8192, lr=1e-4, gamma=0.998, lambda_=0.9,
        #             use_gae=True, entropy_coeff=0.05, vf_clip_param=100000,
        #             minibatch_size=300, num_sgd_iter=10,
        #             model={"custom_model": "CNNModel", "_disable_preprocessor_api": True},
        #         )
        #         .callbacks(BatteryMetricsCallback)
        #         .experimental(_disable_preprocessor_api=True)
        #         .debugging(log_level="ERROR")
        #         .framework(framework="torch")
        #         .resources(num_gpus=0)
        #     ),
        # },
        {
            "num": 3,
            "name": "QMIX_selfheal_comparison",
            "algo": "DQN",
            "env_creator": qmix_sh_env_creator,
            "model_name": "QMIXCNNModel",
            "model_class": QMIXCNNModel,
            "config_builder": lambda: _build_dqn_config(env_name, "QMIXCNNModel"),
        },
        {
            "num": 4,
            "name": "QMIX_vanilla_comparison",
            "algo": "DQN",
            "env_creator": qmix_vanilla_env_creator,
            "model_name": "QMIXVanillaCNNModel",
            "model_class": QMIXVanillaCNNModel,
            "config_builder": lambda: _build_dqn_config(env_name, "QMIXVanillaCNNModel"),
        },
    ]

    # Run each experiment sequentially
    total_start = time.time()

    for exp in experiments:
        # Check if we should skip
        exp_dir = os.path.join(results_dir, exp["name"])
        # FORCE QMIX TO ALWAYS RUN, SKIP MAPPO IF IT EXISTS
        if args.skip_existing and os.path.exists(exp_dir) and "MAPPO" in exp["name"]:
            print(f"\n  ⏭️  Skipping {exp['name']} (already exists)")
            continue

        # Shutdown any previous Ray session, then start fresh
        ray.shutdown()
        ray.init()

        # Register env and model for this experiment
        register_env(env_name, lambda config, ec=exp["env_creator"]: ParallelPettingZooEnv(ec(config)))
        ModelCatalog.register_custom_model(exp["model_name"], exp["model_class"])

        # Build config and run
        config = exp["config_builder"]()
        run_experiment(
            exp["num"],
            exp["name"],
            exp["algo"],
            config,
            curr_path,
            env_name,
            timesteps,
        )

    # Final summary
    ray.shutdown()
    total_elapsed = time.time() - total_start
    total_hours = int(total_elapsed // 3600)
    total_mins = int((total_elapsed % 3600) // 60)

    print(f"\n{'#'*70}")
    print(f"  ALL 4 EXPERIMENTS COMPLETED in {total_hours}h {total_mins}m")
    print(f"  Results: {results_dir}")
    print(f"  View: tensorboard --logdir={results_dir}")
    print(f"{'#'*70}\n")
