"""
RSPO V2 Self-Healing Training Script for Drone Swarm Coverage
=============================================================
RSPO V2 training WITH fault injection AND teammate compensation rewards (Self-Healing Environment).

Features Implemented (RSPO V2):
  1. Fleet Health Embedding (FHE): Dedicated 14-dim fleet status vector feeding h_t in R^32.
  2. Crash-Sector Spatial Attention (CSSA): Dynamic Gaussian heatmap spatial attention on CNN feature maps.
  3. Dual-Objective Value Decomposition (DOVD): Separate V_search and V_takeover heads with auxiliary supervision in custom_loss().
  4. Resilience-Adaptive Clipping (RAC): Dynamic per-sample clipping parameter eps_t integrated into RSPOTorchPolicy.

Usage:
    python train_rspo_v2_cnn_cov.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib
import numpy as np
import torch
from torch import nn

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env, register_trainable

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper

from rspo_v2_policy import RSPOPPO, RSPOPPOConfig, RSPOTorchPolicy
from train_rspo_v2_vanilla import RSPOModelV2, RSPOBatteryMetricsCallback


def env_creator(args):
    print("-------------------------- RSPO V2 SELF-HEALING ENV CREATOR --------------------------")
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
    
    # RSPO V2 SELF-HEALING: Fault injection ON + teammate compensation rewards ON!
    env = BatteryStationWrapper(
        env,
        max_battery=125,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=0.0005
    )

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage_RSPO_V2_SelfHeal"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("RSPOModelV2_SelfHeal", RSPOModelV2)
    register_trainable("RSPOPPO", RSPOPPO)

    num_gpus = 1 if torch.cuda.is_available() else 0
    num_runners = 24 if torch.cuda.is_available() else 6
    print(f"[RSPO V2 Self-Heal] Configured with {num_gpus} GPUs and {num_runners} environment runners.")

    config = (
        RSPOPPOConfig()
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
                "custom_model": "RSPOModelV2_SelfHeal",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(RSPOBatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )

    import sys
    exp_name = os.environ.get("EXP_NAME", "")
    if not exp_name:
        if sys.stdin.isatty():
            try:
                exp_name = input("Exp name for RSPO V2 Self-Healing run: ")
            except (EOFError, KeyboardInterrupt):
                exp_name = "v2_selfheal"
        else:
            exp_name = "v2_selfheal"
    if not exp_name:
        exp_name = "v2_selfheal"

    curr_path = pathlib.Path().resolve()
    tune.run(
        "RSPOPPO",
        name="RSPO_V2_SelfHeal_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/DSSE_Coverage",
        config=config.to_dict(),
    )
