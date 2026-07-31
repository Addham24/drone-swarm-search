"""
RSPO Vanilla Training Script for Drone Swarm Coverage
======================================================
RSPO training WITH fault injection but WITHOUT teammate compensation rewards (Vanilla Environment).
- Uses full RSPO Architecture (Fleet Health Embedding, DOVD Critic, RAC Gate)
- Random fault injection ON (fault_prob=0.0005)
- Teammate crash compensation rewards OFF (COMPENSATION_BONUS = 0.0)

This serves as the baseline to evaluate RSPO's architectural resilience purely 
on the vanilla environment.

Usage:
    python train_rspo_vanilla.py
"""

import os
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

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper


class RSPOBatteryMetricsCallback(DefaultCallbacks):
    """Logs battery deaths, charge steps, average level, coverage, and RSPO-specific metrics."""

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


class RSPOModel(TorchModelV2, nn.Module):
    """
    RSPO Neural Network Model
    ------------------------
    Integrates:
      - Fleet Health Embedding (FHE)
      - Dual-Objective Value Decomposition (DOVD)
      - Resilience-Adaptive Clipping (RAC)
    """

    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        def get_flatten_size(grid_size):
            x = (grid_size - 2) // 2
            x = (x - 1) // 2
            return 32 * x * x

        grid_size = obs_space[1].shape[0]
        flatten_size = get_flatten_size(grid_size)
        positions_dim = obs_space[0].shape[0]

        print(f"[RSPO-Vanilla] Grid size: {grid_size}, Flatten size: {flatten_size}, Positions Dim: {positions_dim}")

        # ── 1. Spatial Grid Encoder (CNN) ──
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        # ── 2. Local State Linear Encoder ──
        self.linear_obs = nn.Sequential(
            nn.Linear(positions_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        # ── 3. Component 1: Fleet Health Embedding (FHE) Encoder ──
        self.fhe_encoder = nn.Sequential(
            nn.Linear(positions_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

        # ── 4. Policy Network ──
        self.actor_join = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.Tanh(),
        )
        self.policy_fn = nn.Linear(256, num_outputs)

        # ── 5. Component 2: Dual-Objective Value Decomposition (DOVD) Critic ──
        self.critic_join = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.Tanh(),
        )
        self.v_search_head = nn.Linear(256, 1)
        self.v_takeover_head = nn.Linear(256, 1)
        self.alpha_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # ── 6. Component 3: Resilience-Adaptive Clipping (RAC) Gate ──
        self.rac_gate = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._value_out = None
        self._current_eps_t = None

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear_obs(input_positions)

        # 1. Compute Fleet Health Embedding (FHE)
        h_t = self.fhe_encoder(input_positions)

        # Combine all features for Actor & Critic
        combined_features = torch.cat((cnn_out, linear_out, h_t), dim=1)

        # 2. Compute Actor Policy
        actor_features = self.actor_join(combined_features)
        action_logits = self.policy_fn(actor_features)

        # 3. Compute Dual-Objective Critic (DOVD)
        critic_features = self.critic_join(combined_features)
        v_search = self.v_search_head(critic_features)
        v_takeover = self.v_takeover_head(critic_features)
        alpha_t = self.alpha_head(critic_features)

        # Total Value Function
        self._value_out = v_search + alpha_t * v_takeover

        # 4. Compute State-Conditioned Adaptive Clip (RAC)
        self._current_eps_t = 0.1 + 0.2 * self.rac_gate(h_t)

        return action_logits, state

    def value_function(self):
        return self._value_out.flatten()

    def get_adaptive_clip(self):
        return self._current_eps_t.flatten()


def env_creator(args):
    print("-------------------------- RSPO VANILLA ENV CREATOR --------------------------")
    N_AGENTS = 4
    matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(
        timestep_limit=750,
        drone_amount=N_AGENTS,
        prob_matrix_path=matrix_path
    )
    env.reward_scheme = {
        "default": -0.1,
        "exceed_timestep": 0.0,
        "search_cell": 5.0,
        "done": 500.0,
        "reward_poc": 0.0,
    }
    env = AllPositionsWrapper(env)
    
    # RSPO VANILLA: Fault injection ON, but NO teammate compensation rewards!
    env = BatteryStationWrapper(
        env,
        max_battery=125,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=0.0005
    )
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage_RSPO_Vanilla"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("RSPOModelVanilla", RSPOModel)

    num_gpus = 1 if torch.cuda.is_available() else 0
    num_runners = 24 if torch.cuda.is_available() else 6
    print(f"[RSPO-Vanilla] Configured with {num_gpus} GPUs and {num_runners} environment runners.")

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
                "custom_model": "RSPOModelVanilla",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(RSPOBatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )

    curr_path = pathlib.Path().resolve()
    try:
        exp_name = input("Exp name for RSPO Vanilla run: ")
    except EOFError:
        exp_name = "v1"

    tune.run(
        "PPO",
        name="RSPO_vanilla_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/DSSE_Coverage",
        config=config.to_dict(),
    )
