"""
MAPPO Vanilla Training Script for Dynamic Target Tracking
===========================================================
MAPPO training on DSSE.DroneSwarmSearch (Dynamic Tracking)
WITHOUT teammate compensation rewards (Vanilla Environment).

Usage:
    python src/tracking/train_mappo_tracking_vanilla.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import pathlib
import numpy as np
import torch
from torch import nn

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

# Ensure src directory is on sys.path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tracking.tracking_env_creator import make_tracking_env


class BatteryMetricsCallback(DefaultCallbacks):
    """Logs battery deaths, charge steps, and average level."""

    def on_episode_start(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        episode.user_data["deaths"] = 0
        episode.user_data["charge_steps"] = 0
        episode.user_data["total_battery"] = 0
        episode.user_data["battery_samples"] = 0

    def on_episode_step(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info is None or not isinstance(info, dict):
                continue
            if info.get("stranded_event", False):
                episode.user_data["deaths"] += 1
            if info.get("charging", False):
                episode.user_data["charge_steps"] += 1
            batt = info.get("battery")
            if batt is not None:
                episode.user_data["total_battery"] += batt
                episode.user_data["battery_samples"] += 1

    def on_episode_end(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        episode.custom_metrics["battery_deaths"] = episode.user_data["deaths"]
        episode.custom_metrics["battery_charge_steps"] = episode.user_data["charge_steps"]

        samples = episode.user_data["battery_samples"]
        if samples > 0:
            episode.custom_metrics["battery_avg_level"] = (
                episode.user_data["total_battery"] / samples
            )
        else:
            episode.custom_metrics["battery_avg_level"] = 0.0


class MAPPOCNNModel(TorchModelV2, nn.Module):
    """MAPPO CNN Model for 22-dim position/battery vector + 25x25 matrix."""

    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name, **kw)
        nn.Module.__init__(self)

        def get_flatten_size(grid_size):
            x = (grid_size - 2) // 2
            x = (x - 1) // 2
            return 32 * x * x

        grid_size = obs_space[1].shape[0]
        flatten_size = get_flatten_size(grid_size)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        self.linear = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )

        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].unsqueeze(1).float()

        cnn_features = self.cnn(input_matrix)
        linear_features = self.linear(input_positions)

        joined_features = torch.cat([cnn_features, linear_features], dim=1)
        joined_out = self.join(joined_features)

        self._value_out = self.value_fn(joined_out)
        logits = self.policy_fn(joined_out)
        return logits, state

    def value_function(self):
        return self._value_out.squeeze(1)


def env_creator(args):
    print("-------------------------- MAPPO TRACKING VANILLA ENV CREATOR --------------------------")
    return make_tracking_env(
        grid_size=25,
        drone_amount=4,
        person_amount=1,
        person_initial_position=(12, 12),
        timestep_limit=750,
        max_battery=125,
        fault_prob=0.0005,
        is_self_heal=False,
        drift_speed=1.0
    )


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Tracking_MAPPO_Vanilla"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("MAPPOCNNModel_Tracking_Vanilla", MAPPOCNNModel)

    total_cpus = os.cpu_count() or 4
    num_gpus = 1 if torch.cuda.is_available() else 0
    if "NUM_RUNNERS" in os.environ:
        num_runners = int(os.environ["NUM_RUNNERS"])
    else:
        max_allocatable = max(1, total_cpus - 2)
        num_runners = min(10, max_allocatable)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode (No GPU detected)"
    print(f"======================================================================")
    print(f"  [MAPPO Tracking Vanilla] Starting Training")
    print(f"  • CPUs Available : {total_cpus} (Workers: {num_runners})")
    print(f"  • GPUs Available : {num_gpus} ({gpu_name})")
    print(f"======================================================================")

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .env_runners(num_env_runners=num_runners, rollout_fragment_length="auto")
        .training(
            train_batch_size=8192,
            lr=1e-4,
            gamma=0.998,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.05,
            vf_clip_param=100000,
            minibatch_size=300,
            num_sgd_iter=10,
            model={
                "custom_model": "MAPPOCNNModel_Tracking_Vanilla",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(BatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )

    exp_name = os.environ.get("EXP_NAME", "mappo_vanilla")
    curr_path = pathlib.Path(__file__).resolve().parent.parent

    tune.run(
        "PPO",
        name="MAPPO_Tracking_Vanilla_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/DSSE_Tracking",
        config=config.to_dict(),
    )
