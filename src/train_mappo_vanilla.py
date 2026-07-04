"""
MAPPO Vanilla Training Script for Drone Swarm Coverage
=======================================================
MAPPO training WITH fault injection but WITHOUT self-healing compensation.
- Random fault injection ON (fault_prob=0.0005) — same adversity as self-healing
- No teammate crash compensation rewards — no reward signal to respond to crashes

This serves as the baseline for comparing against the self-healing variant
in train_mappo_cnn_cov.py.

Usage:
    python train_mappo_vanilla.py
"""

import os
import pathlib
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
import torch
import numpy as np


class BatteryMetricsCallback(DefaultCallbacks):
    """Logs custom battery metrics to TensorBoard via RLlib's custom_metrics."""

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

        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info and isinstance(info, dict) and "coverage_rate" in info:
                episode.custom_metrics["coverage_rate"] = info["coverage_rate"]
                break


class CNNModel(TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        def get_flatten_size(grid_size):
            # Conv1: 3x3, Padding: 0 -> (grid_size - 3 + 1) = grid_size - 2
            # MaxPool1: 2x2 -> (grid_size - 2) // 2
            # Conv2: 2x2, Padding: 0 -> ((grid_size - 2) // 2) - 2 + 1 = (grid_size - 2) // 2 - 1
            # MaxPool2: 2x2 -> ((grid_size - 2) // 2 - 1) // 2
            x = (grid_size - 2) // 2
            x = (x - 1) // 2
            return 32 * x * x

        grid_size = obs_space[1].shape[0]
        flatten_size = get_flatten_size(grid_size)
        print(f"Grid size: {grid_size}, Cnn Dense layer input size: {flatten_size}")
        
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
            ),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(2, 2),
            ),
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
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear(input_positions)

        value_input = torch.cat((cnn_out, linear_out), dim=1)
        value_input = self.join(value_input)
        
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()

def env_creator(args):
    print("-------------------------- MAPPO VANILLA ENV CREATOR --------------------------")
    N_AGENTS = 4
    matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(
        timestep_limit=750, 
        drone_amount=N_AGENTS, 
        prob_matrix_path=matrix_path
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

    # VANILLA: Fault injection ON (same adversity), but NO teammate compensation
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0

    grid_size = env.grid_size
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env

if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .env_runners(num_env_runners=6, rollout_fragment_length="auto")
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
                "custom_model": "CNNModel",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(BatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=0)
    )

    curr_path = pathlib.Path().resolve()
    try:
        exp_name = input("Exp name: ")
    except EOFError:
        exp_name = "default"
        
    tune.run(
        "PPO",
        name="MAPPO_vanilla_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
