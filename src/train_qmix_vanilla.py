"""
VDN-DQN Vanilla Training Script for Drone Swarm Coverage
==========================================================
DQN with VDN-style cooperative reward mixing but WITHOUT self-healing:
- Global Reward Wrapper (VDN mixing: alpha=0.7)
- Random fault injection ON (fault_prob=0.0005) — same adversity as self-healing
- No teammate crash compensation rewards — no reward signal to respond to crashes
- Tanh activations (matching MAPPO architecture for fair comparison)

This serves as the VDN-DQN baseline for comparing against the
self-healing variant in train_qmix_cnn_cov.py.

Usage:
    python train_qmix_vanilla.py
"""

import os
import pathlib
import numpy as np

import ray
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from global_reward_wrapper import GlobalRewardWrapper

import torch
from torch import nn

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


# ─── Callbacks ────────────────────────────────────────────────────────────────

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


# ─── Custom CNN + Q-Network Model (Tanh — matches MAPPO architecture) ────────

class QMIXVanillaCNNModel(TorchModelV2, nn.Module):
    """
    CNN-based Q-network for VDN-DQN Vanilla training.
    Same architecture as QMIXCNNModel but registered under a different name
    to avoid model registry conflicts when comparing checkpoints.
    """

    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        print("QMIX VANILLA OBSSPACE: ", obs_space)
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
        print(f"QMIX Vanilla Grid size: {grid_size}, CNN Dense layer input size: {flatten_size}")

        # Spatial stream: processes the 25x25 coverage matrix as a 2D image
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

        # Kinematic stream: processes positions, battery levels, crash coords
        self.linear = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        # Fusion layer: combines spatial and kinematic streams
        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )

        self.q_head = nn.Linear(256, num_outputs)
        self.value_head = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()

        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear(input_positions)

        combined = torch.cat((cnn_out, linear_out), dim=1)
        combined = self.join(combined)

        self._value_out = self.value_head(combined)
        return self.q_head(combined), state

    def value_function(self):
        return self._value_out.flatten()


# ─── Environment Creator ─────────────────────────────────────────────────────

def env_creator(args):
    print("-------------------------- VDN-DQN VANILLA ENV CREATOR --------------------------")
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

    # VANILLA: Fault injection ON (same adversity), but NO teammate compensation
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0

    # VDN MIXING: 50% global reward + 50% individual reward
    env = GlobalRewardWrapper(env, mixing_alpha=0.5)

    grid_size = env.grid_size
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("QMIXVanillaCNNModel", QMIXVanillaCNNModel)

    num_gpus = 1 if torch.cuda.is_available() else 0
    num_runners = 16 if torch.cuda.is_available() else 4
    print(f"Using {num_gpus} GPUs and {num_runners} environment runners.")

    config = (
        DQNConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .env_runners(num_env_runners=num_runners, rollout_fragment_length=200)
        .training(
            train_batch_size=512, lr=1e-4, gamma=0.998,
            num_steps_sampled_before_learning_starts=100000,
            dueling=True,
            double_q=True,
            n_step=1,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": 500_000,
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.6,
                "prioritized_replay_beta": 0.4,
            },
            target_network_update_freq=10000,
            grad_clip=10.0,
            model={
                "custom_model": "QMIXVanillaCNNModel",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(BatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )
    config.exploration_config = {
        "type": "SoftQ",
        "temperature": 1.0,
    }

    curr_path = pathlib.Path().resolve()
    try:
        exp_name = input("Exp name: ")
    except EOFError:
        exp_name = "default"

    print(f"\n{'='*60}")
    print(f"  VDN-DQN Vanilla Training: {exp_name}")
    print(f"  Reward Mixing: VDN (alpha=0.7)")
    print(f"  Fault Injection: ON (p=0.0005)")
    print(f"  Teammate Compensation: OFF")
    print(f"{'='*60}\n")

    tune.run(
        "DQN",
        name="QMIX_vanilla_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/" + env_name,
        config=config.to_dict(),
    )
