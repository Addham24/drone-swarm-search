"""
RSPO V2 Vanilla Training Script for Dynamic Target Tracking
============================================================
RSPO V2 training on DSSE.DroneSwarmSearch (Dynamic Tracking)
WITHOUT teammate compensation rewards (Vanilla Environment).

Features Implemented (RSPO V2):
  1. Fleet Health Embedding (FHE): Dedicated 14-dim fleet status vector feeding h_t in R^32.
  2. Crash-Sector Spatial Attention (CSSA): Dynamic Gaussian heatmap spatial attention on CNN feature maps.
  3. Dual-Objective Value Decomposition (DOVD): Separate V_track and V_takeover heads with auxiliary supervision.
  4. Resilience-Adaptive Clipping (RAC): Dynamic per-sample clipping parameter eps_t integrated into RSPOTorchPolicy.

Usage:
    python src/tracking/train_rspo_v2_tracking_vanilla.py
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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env, register_trainable

# Ensure src directory is on sys.path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from rspo_v2_policy import RSPOPPO, RSPOPPOConfig, RSPOTorchPolicy
from train_rspo_v2_vanilla import RSPOModelV2
from tracking.tracking_env_creator import make_tracking_env


class RSPOTrackingMetricsCallback(DefaultCallbacks):
    """Logs battery deaths, charge steps, average level, and tracking metrics."""

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


def env_creator(args):
    print("-------------------------- RSPO V2 TRACKING VANILLA ENV CREATOR --------------------------")
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

    env_name = "DSSE_Tracking_RSPO_V2_Vanilla"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("RSPOModelV2_Tracking_Vanilla", RSPOModelV2)
    register_trainable("RSPOPPO", RSPOPPO)

    total_cpus = os.cpu_count() or 4
    num_gpus = 1 if torch.cuda.is_available() else 0
    if "NUM_RUNNERS" in os.environ:
        num_runners = int(os.environ["NUM_RUNNERS"])
    else:
        max_allocatable = max(1, total_cpus - 2)
        num_runners = min(10, max_allocatable)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Mode (No GPU detected)"
    print(f"======================================================================")
    print(f"  [RSPO V2 Tracking Vanilla] Starting Training")
    print(f"  • CPUs Available : {total_cpus} (Workers: {num_runners})")
    print(f"  • GPUs Available : {num_gpus} ({gpu_name})")
    print(f"======================================================================")

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
                "custom_model": "RSPOModelV2_Tracking_Vanilla",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(RSPOTrackingMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )

    exp_name = os.environ.get("EXP_NAME", "rspo_v2_vanilla")

    curr_path = pathlib.Path(__file__).resolve().parent.parent
    tune.run(
        "RSPOPPO",
        name="RSPO_V2_Tracking_Vanilla_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/DSSE_Tracking",
        config=config.to_dict(),
    )
