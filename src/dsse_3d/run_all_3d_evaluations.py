"""
DSSE-3D Master Evaluation Suite
=================================
Evaluates all 12 MARL models in 3D voxel environments across:
    - 4 environment configurations (25x25, 25x25+Obs, 50x50, 50x50+Obs)
    - 2 fault probability regimes (No Faults, Active Faults)
    - 2 tasks (Tracking, Coverage)
    = 16 total evaluation runs per task

Each run evaluates 12 models × 100 seeds = 1200 episodes.

Environment Configurations:
    1. 25×25 Standard     (T=750,  battery=125, no obstacles)
    2. 25×25 Obstacles    (T=750,  battery=125, obstacle_skyline_25.npy)
    3. 50×50 Standard     (T=1500, battery=250, no obstacles)
    4. 50×50 Obstacles    (T=1500, battery=250, obstacle_skyline_50.npy)

Fault Regimes:
    - No_Faults_0.0:         fault_prob = 0.0
    - Active_Faults_0.0005:  fault_prob = 0.0005

3D-Specific Metrics (in addition to all 2D metrics):
    - Mean Drone Altitude
    - Altitude Transitions / Episode
    - 3D Obstacle Collisions
    - Mid-Air Inter-Drone Collisions

Outputs:
    ~/Desktop/Results/3D/Coverage/<env_tag>/<fault_tag>/
        - dashboard_coverage_3d.png
        - coverage_3d_summary.csv
        - coverage_3d_raw_seed_by_seed.csv
    ~/Desktop/Results/3D/Tracking/<env_tag>/<fault_tag>/
        - dashboard_tracking_3d.png
        - tracking_3d_summary.csv
        - tracking_3d_raw_seed_by_seed.csv

Usage:
    python -m dsse_3d.run_all_3d_evaluations --task tracking
    python -m dsse_3d.run_all_3d_evaluations --task coverage
    python -m dsse_3d.run_all_3d_evaluations --task all
    python -m dsse_3d.run_all_3d_evaluations --task all --n_seeds 10   # quick test
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import pandas as pd
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from dsse_3d.env_creators import make_tracking_env_3d, make_coverage_env_3d
from dsse_3d.policy_adapters import wrap_rllib_policy_for_3d, wrap_epymarl_policy_for_3d
from dsse_3d.policy_loaders import (
    load_rllib_ppo_policy, load_rllib_dqn_policy, load_epymarl_policy_strict,
    make_resizing_policy,
)
from tracking.evaluate_tracking_models import find_best_post_10m_checkpoint
from train_rspo_v2_vanilla import RSPOModelV2
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla


# ──────────────────────────────────────────────────────────────────────────────
# Environment Configurations
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_obstacle_mask(grid_size):
    """Load obstacle mask .npy for a given grid size, or return None."""
    path = os.path.join(SRC_DIR, f"obstacle_skyline_{grid_size}.npy")
    if os.path.exists(path):
        return np.load(path)
    print(f"    [!] Obstacle mask not found: {path}")
    return None


def _resolve_prob_matrix(grid_size, has_obstacles):
    """Resolve probability matrix path for coverage environments."""
    if has_obstacles:
        path = os.path.join(SRC_DIR, f"obstacle_prob_matrix_{grid_size}.npy")
    else:
        path = os.path.join(SRC_DIR, f"uniform_matrix_{grid_size}.npy")
    if os.path.exists(path):
        return path
    return None


# `target_start` mirrors the 2D tracking evaluators (evaluate_tracking_models.py /
# evaluate_tracking_obstacles_25x25.py use (12,12); the 50x50 evaluators use (26,26)).
# `free_cells` is the number of coverage-eligible cells used by the 2D coverage
# evaluators for the pure-mission-return formula (evaluate_50x50_and_obstacles.py).
ENV_CONFIGS = [
    {
        "tag": "25x25_Standard",
        "grid_size": 25,
        "timestep_limit": 750,
        "max_battery": 125,
        "has_obstacles": False,
        "target_start": (12, 12),
        "free_cells": 625,
        "label": "25×25 Standard",
    },
    {
        "tag": "25x25_Obstacles",
        "grid_size": 25,
        "timestep_limit": 750,
        "max_battery": 125,
        "has_obstacles": True,
        "target_start": (12, 12),
        "free_cells": 609,
        "label": "25×25 Obstacles",
    },
    {
        "tag": "50x50_Standard",
        "grid_size": 50,
        "timestep_limit": 1500,
        "max_battery": 250,
        "has_obstacles": False,
        "target_start": (26, 26),
        "free_cells": 2500,
        "label": "50×50 Standard",
    },
    {
        "tag": "50x50_Obstacles",
        "grid_size": 50,
        "timestep_limit": 1500,
        "max_battery": 250,
        "has_obstacles": True,
        "target_start": (26, 26),
        "free_cells": 2404,
        "label": "50×50 Obstacles",
    },
    {
        "tag": "75x75_Standard",
        "grid_size": 75,
        "timestep_limit": 2250,
        "max_battery": 375,
        "has_obstacles": False,
        "target_start": (38, 38),
        "free_cells": 5625,
        "label": "75×75 Standard",
    },
    {
        "tag": "75x75_Obstacles",
        "grid_size": 75,
        "timestep_limit": 2250,
        "max_battery": 375,
        "has_obstacles": True,
        "target_start": (38, 38),
        "free_cells": 5481,
        "label": "75×75 Obstacles",
    },
]

FAULT_REGIMES = [
    {"tag": "No_Faults_0.0", "fault_prob": 0.0, "label": "No Faults (fault_prob = 0.0)"},
    {"tag": "Fault_0.00025", "fault_prob": 0.00025, "label": "Low Faults (fault_prob = 0.00025)"},
    {"tag": "Active_Faults_0.0005", "fault_prob": 0.0005, "label": "Standard Faults (fault_prob = 0.0005)"},
    {"tag": "Fault_0.00075", "fault_prob": 0.00075, "label": "High Faults (fault_prob = 0.00075)"},
    {"tag": "Fault_0.001", "fault_prob": 0.0010, "label": "Severe Faults (fault_prob = 0.0010)"},
]


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry
# ──────────────────────────────────────────────────────────────────────────────

def build_model_registry(task):
    """Build the 12-model registry with checkpoint glob patterns."""
    if task == "tracking":
        return [
            ("RSPO Vanilla",   "rllib",   RSPOModelV2,      False, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_Vanilla*/**/checkpoint_*"),
            ("RSPO SelfHeal",  "rllib",   RSPOModelV2,      True,  "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_SelfHeal*/**/checkpoint_*"),
            ("MAPPO Vanilla",  "rllib",   MAPPOModelVanilla, False, "ray_res/DSSE_Tracking/*MAPPO_Tracking_Vanilla*/**/checkpoint_*"),
            ("MAPPO SelfHeal", "rllib",   MAPPOModelVanilla, True,  "ray_res/DSSE_Tracking/*MAPPO_Tracking_SelfHeal*/**/checkpoint_*"),
            ("QMIX Vanilla",   "epymarl", None,             False, "results/models/*qmix_tracking_vanilla*/**/agent.th"),
            ("QMIX SelfHeal",  "epymarl", None,             True,  "results/models/*qmix_tracking_selfheal*/**/agent.th"),
            ("MAA2C Vanilla",  "epymarl", None,             False, "results/models/*maa2c_tracking_vanilla*/**/agent.th"),
            ("MAA2C SelfHeal", "epymarl", None,             True,  "results/models/*maa2c_tracking_selfheal*/**/agent.th"),
            ("COMA Vanilla",   "epymarl", None,             False, "results/models/*coma_tracking_vanilla*/**/agent.th"),
            ("COMA SelfHeal",  "epymarl", None,             True,  "results/models/*coma_tracking_selfheal*/**/agent.th"),
            ("I-DQN Vanilla",  "epymarl", None,             False, "results/models/*iql_tracking_vanilla*/**/agent.th"),
            ("I-DQN SelfHeal", "epymarl", None,             True,  "results/models/*iql_tracking_selfheal*/**/agent.th"),
        ]
    else:  # coverage
        return [
            ("RSPO Vanilla",   "rllib",   RSPOModelV2,      False, "ray_res/DSSE_Coverage/RSPO_V2_Vanilla_RSPO_v2/RSPOPPO_DSSE_Coverage_RSPO_V2_Vanilla_45eb7_00000_0_2026-09-04_06-38-43/checkpoint_000139"),
            ("RSPO SelfHeal",  "rllib",   RSPOModelV2,      True,  "ray_res/DSSE_Coverage/RSPO_V2_SelfHeal_rspo_v2_selfheal/RSPOPPO_DSSE_Coverage_RSPO_V2_SelfHeal_86fd5_00000_0_2026-09-04_06-40-32/checkpoint_000230"),
            ("MAPPO Vanilla",  "rllib",   MAPPOModelVanilla, False, "ray_res/DSSE_Coverage/MAPPO_vanilla_proper/PPO_DSSE_Coverage_cc7e1_00000_0_2026-07-05_03-37-08/checkpoint_000141"),
            ("MAPPO SelfHeal", "rllib",   MAPPOModelVanilla, True,  "ray_res/DSSE_Coverage/MAPPO_selfheal_final_v2/checkpoints/checkpoint_iter_980_ts_10002432.pt"),
            ("QMIX Vanilla",   "epymarl", None,             False, "results/models/qmix_true_vanilla_v1_seed533532401_dsse_coverage_2026-07-18 02:06:54.683275/17040000/agent.th"),
            ("QMIX SelfHeal",  "epymarl", None,             True,  "results/models/qmix_selfheal_v2_seed533532401_dsse_coverage_2026-07-17 03:23:47.879210/19044000/agent.th"),
            ("MAA2C Vanilla",  "epymarl", None,             False, "results/models/maa2c_cnn_bigbatch_v3_vanilla_seed782869923_dsse_coverage_2026-06-16 18:45:27.715438/18795000/agent.th"),
            ("MAA2C SelfHeal", "epymarl", None,             True,  "results/models/maa2c_cnn_bigbatch_v3_original_seed776978465_dsse_coverage_2026-07-15 06:30:31.971452/12675000/agent.th"),
            ("COMA Vanilla",   "epymarl", None,             False, "results/models/coma_vanilla_proper_seed452926189_dsse_coverage_2026-07-10 01:59:53.354626/19338000/agent.th"),
            ("COMA SelfHeal",  "epymarl", None,             True,  "results/models/coma_selfhealing_final_stable_seed339532096_dsse_coverage_2026-07-06 10:52:47.933607/19986000/agent.th"),
            ("I-DQN Vanilla",  "rllib",   MAPPOModelVanilla, False, "ray_res/DSSE_Coverage/QMIX_vanilla_I-DQN_vanilla_proper/DQN_DSSE_Coverage_61018_00000_0_2026-07-16_09-47-37/checkpoint_000609"),
            ("I-DQN SelfHeal", "rllib",   MAPPOModelVanilla, True,  "ray_res/DSSE_Coverage/QMIX_I-DQN_selfheal_proper_pt2/DQN_DSSE_Coverage_8aefa_00000_0_2026-07-16_05-38-15/checkpoint_000565"),
        ]


def load_all_policies(task):
    """
    Pre-load all 12 model checkpoints with strict key matching.
    Returns list of (name, policy_fn, is_self_heal, checkpoint_path).

    Protocol (mirrors the 2D evaluators per task):
      * coverage: RLlib models act greedily (argmax), as in evaluate_all_models.py
      * tracking: RLlib PPO-family models sample from the policy, as in
                  tracking/evaluate_tracking_models.py
      * EPyMARL: always greedy argmax over Q-values
    A missing checkpoint or any weight-key mismatch raises (no random stubs).
    """
    registry = build_model_registry(task)
    loaded = []
    greedy_rllib = (task == "coverage")

    print(f"\n{'─'*70}")
    print(f"  LOADING {task.upper()} MODEL CHECKPOINTS (12 models, strict)")
    print(f"{'─'*70}")

    for model_name, framework, model_cls, is_self_heal, pattern in registry:
        full_pattern = os.path.join(SRC_DIR, pattern)
        if os.path.exists(full_pattern):
            ckpt_path = full_pattern
        else:
            ckpt_path = find_best_post_10m_checkpoint(full_pattern, framework, min_timesteps=10_000_000)

        print(f"\n  [▶] {model_name} ({framework.upper()})")
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint for {task} / {model_name}: {pattern}")
        print(f"      Checkpoint: {ckpt_path}")

        if framework == "epymarl":
            raw_fn = load_epymarl_policy_strict(ckpt_path)
            policy_fn = wrap_epymarl_policy_for_3d(raw_fn)
        elif model_name.startswith("I-DQN"):
            raw_fn = load_rllib_dqn_policy(ckpt_path)
            policy_fn = wrap_rllib_policy_for_3d(raw_fn)
        else:
            raw_fn = load_rllib_ppo_policy(model_cls, ckpt_path, model_name, greedy=greedy_rllib)
            policy_fn = wrap_rllib_policy_for_3d(raw_fn)

        loaded.append((model_name, policy_fn, is_self_heal, ckpt_path))

    return loaded



# ──────────────────────────────────────────────────────────────────────────────
# Tracking Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_tracking_3d_episode(
    policy_fn, env_config, fault_prob, is_self_heal, n_seeds, obstacle_mask
):
    """Evaluate tracking in 3D over n_seeds. Returns (summary_dict, raw_records)."""
    grid_size = env_config["grid_size"]
    records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_tracking_env_3d(
            grid_size=grid_size,
            drone_amount=4,
            person_amount=1,
            person_initial_position=env_config["target_start"],
            timestep_limit=env_config["timestep_limit"],
            max_battery=env_config["max_battery"],
            fault_prob=fault_prob,
            is_self_heal=is_self_heal,
            drift_speed=1.0,
            obstacle_mask=obstacle_mask,
        )

        obs, infos = env.reset(seed=seed)
        done = False
        step = 0
        target_found = False
        discovery_step = env_config["timestep_limit"]

        total_battery = 0
        battery_samples = 0
        crash_events = 0
        crashed_agents = set()
        total_altitude = 0
        altitude_samples = 0

        while not done and step < env_config["timestep_limit"]:
            step += 1
            # Stable agent ordering + per-episode recurrent reset (reset=True on step 1)
            actions = policy_fn(obs, env.possible_agents, reset=(step == 1))
            actions = {a: v for a, v in actions.items() if a in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)

            for agent in list(rewards.keys()):
                agent_info = infos.get(agent, {})
                if isinstance(agent_info, dict) and agent_info.get("Found", False):
                    target_found = True
                    discovery_step = step
                    done = True
                    break

            for agent, info in infos.items():
                if isinstance(info, dict):
                    if info.get("stranded_event", False):
                        crash_events += 1
                        crashed_agents.add(agent)
                    batt = info.get("battery")
                    if batt is not None:
                        total_battery += batt
                        battery_samples += 1
                    alt = info.get("altitude")
                    if alt is not None:
                        total_altitude += alt
                        altitude_samples += 1

            if all(term.values()) or all(trunc.values()):
                done = True

        metrics_3d = env.get_3d_metrics()
        n_drones = 4
        survival_rate = ((n_drones - len(crashed_agents)) / n_drones) * 100.0
        avg_batt = ((total_battery / battery_samples) / env_config["max_battery"] * 100.0) if battery_samples > 0 else 0.0
        mean_alt = (total_altitude / altitude_samples) if altitude_samples > 0 else 1.0

        records.append({
            "seed": seed,
            "target_found": 1 if target_found else 0,
            "discovery_step": discovery_step,
            "crash_events": crash_events,
            "survival_rate": survival_rate,
            "avg_battery": avg_batt,
            "mean_altitude": mean_alt,
            "altitude_changes": metrics_3d["altitude_changes"],
        })

    # Aggregate
    tf = np.mean([r["target_found"] for r in records]) * 100.0
    dt = np.mean([r["discovery_step"] for r in records])
    surv = np.mean([r["survival_rate"] for r in records])
    batt = np.mean([r["avg_battery"] for r in records])
    tl = env_config["timestep_limit"]

    f_target = max(0.001, tf / 100.0)
    f_speed = max(0.001, (tl - dt) / tl)
    f_survival = max(0.001, surv / 100.0)
    f_battery = max(0.001, batt / 100.0)
    smei = (f_target**0.20 * f_speed**0.20 * f_survival**0.30 * f_battery**0.30) * 100.0

    ci = lambda vals: 1.96 * np.std(vals) / np.sqrt(len(vals))
    found_vals = [r["target_found"] * 100.0 for r in records]
    disc_vals = [r["discovery_step"] for r in records]

    summary = {
        "target_found_rate": round(tf, 2),
        "target_found_ci95": round(ci(found_vals), 2),
        "mean_discovery_time": round(dt, 2),
        "discovery_time_std": round(np.std(disc_vals), 2),
        "discovery_time_ci95": round(ci(disc_vals), 2),
        "attrition_rate": round(np.mean([r["crash_events"] for r in records]), 2),
        "survival_rate": round(surv, 2),
        "avg_battery": round(batt, 2),
        "smei": round(smei, 2),
        "mean_altitude": round(np.mean([r["mean_altitude"] for r in records]), 2),
        "mean_alt_changes": round(np.mean([r["altitude_changes"] for r in records]), 2),
        "n_seeds": len(records),
    }
    return summary, records


# ──────────────────────────────────────────────────────────────────────────────
# Coverage Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_coverage_3d_episode(
    policy_fn, env_config, fault_prob, is_self_heal, n_seeds, obstacle_mask
):
    """Evaluate coverage in 3D over n_seeds. Returns (summary_dict, raw_records)."""
    grid_size = env_config["grid_size"]
    prob_path = _resolve_prob_matrix(grid_size, env_config["has_obstacles"])
    records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_coverage_env_3d(
            grid_size=grid_size,
            drone_amount=4,
            timestep_limit=env_config["timestep_limit"],
            max_battery=env_config["max_battery"],
            fault_prob=fault_prob,
            is_self_heal=is_self_heal,
            obstacle_mask=obstacle_mask,
            prob_matrix_path=prob_path,
        )

        obs, infos = env.reset(seed=seed)
        # Sanity: 3D coverage denominator must equal the 2D free-cell count
        assert len(env._coverage_universe) == env_config["free_cells"], (
            f"coverage universe {len(env._coverage_universe)} != expected {env_config['free_cells']} "
            f"for {env_config['tag']}"
        )
        done = False
        step = 0
        total_reward = 0.0
        total_battery = 0
        battery_samples = 0
        crash_events = 0
        crashed_agents = set()
        total_altitude = 0
        altitude_samples = 0
        final_coverage = 0.0
        charging_steps = 0

        while not done and step < env_config["timestep_limit"]:
            step += 1
            # Stable agent ordering + per-episode recurrent reset (reset=True on step 1)
            actions = policy_fn(obs, env.possible_agents, reset=(step == 1))
            actions = {a: v for a, v in actions.items() if a in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)

            total_reward += sum(rewards.values())

            for agent, info in infos.items():
                if isinstance(info, dict):
                    if info.get("stranded_event", False):
                        crash_events += 1
                        crashed_agents.add(agent)
                    if info.get("charging", False):
                        charging_steps += 1
                    batt = info.get("battery")
                    if batt is not None:
                        total_battery += batt
                        battery_samples += 1
                    alt = info.get("altitude")
                    if alt is not None:
                        total_altitude += alt
                        altitude_samples += 1
                    cov = info.get("coverage_rate")
                    if cov is not None:
                        final_coverage = max(final_coverage, cov)

            if all(term.values()) or all(trunc.values()):
                done = True

        metrics_3d = env.get_3d_metrics()
        n_drones = 4
        survival_rate = ((n_drones - len(crashed_agents)) / n_drones) * 100.0
        avg_batt = ((total_battery / battery_samples) / env_config["max_battery"] * 100.0) if battery_samples > 0 else 0.0
        mean_alt = (total_altitude / altitude_samples) if altitude_samples > 0 else 1.0

        # Authoritative final coverage straight from the 3D state (2D free-cell denominator)
        final_coverage = env.get_coverage_rate()
        # Pure mission return: identical formula to the 2D coverage evaluators
        # (cells_covered * 5.0 == coverage * free_cells * 5.0). The shaped env return
        # (incl. battery / 3D penalties) is kept separately for transparency.
        mission_return = final_coverage * env_config["free_cells"] * 5.0

        records.append({
            "seed": seed,
            "coverage_pct": final_coverage * 100.0,
            "mission_return": mission_return,
            "shaped_return": total_reward,
            "crash_events": crash_events,
            "survival_rate": survival_rate,
            "avg_battery": avg_batt,
            "charging_steps": charging_steps,
            "mean_altitude": mean_alt,
            "altitude_changes": metrics_3d["altitude_changes"],
        })

    cov_vals = [r["coverage_pct"] for r in records]
    cov_mean = np.mean(cov_vals)
    surv_mean = np.mean([r["survival_rate"] for r in records])
    batt_mean = np.mean([r["avg_battery"] for r in records])

    f_cov = max(0.001, cov_mean / 100.0)
    f_surv = max(0.001, surv_mean / 100.0)
    f_batt = max(0.001, batt_mean / 100.0)
    smei = (f_cov**0.40 * f_surv**0.30 * f_batt**0.30) * 100.0

    summary = {
        "coverage_rate": round(cov_mean, 2),
        "coverage_std": round(np.std(cov_vals), 2),
        "coverage_ci95": round(1.96 * np.std(cov_vals) / np.sqrt(len(cov_vals)), 2),
        "attrition_rate": round(np.mean([r["crash_events"] for r in records]), 2),
        "survival_rate": round(surv_mean, 2),
        "avg_battery": round(batt_mean, 2),
        "smei": round(smei, 2),
        "mean_reward": round(np.mean([r["mission_return"] for r in records]), 2),
        "mean_shaped_return": round(np.mean([r["shaped_return"] for r in records]), 2),
        "mean_charging_steps": round(np.mean([r["charging_steps"] for r in records]), 2),
        "mean_altitude": round(np.mean([r["mean_altitude"] for r in records]), 2),
        "mean_alt_changes": round(np.mean([r["altitude_changes"] for r in records]), 2),
        "n_seeds": len(records),
    }
    return summary, records


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_tracking_3d_dashboard(df, output_dir, title):
    """Generate 8-panel tracking dashboard including 3D-specific metrics."""
def generate_tracking_3d_dashboard(df, output_dir, title):
    """Generate 6-panel tracking dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    models = df["Model"].tolist()
    x = np.arange(len(models))

    panels = [
        ("Target Found Rate (%)",    "#2ca02c", "1. Target Intercept Rate (%)",    0, 115, "%"),
        ("Mean Discovery Time",      "#1f77b4", "2. Mean Time-to-Intercept",       None, None, ""),
        ("Attrition Rate",           "#d62728", "3. Agent Attrition (Crashes/ep)",  0, 4.8, ""),
        ("Survival Rate (%)",        "#9467bd", "4. Swarm Survival Rate (%)",       0, 115, "%"),
        ("Avg Battery (%)",          "#ff7f0e", "5. Fleet Battery (%)",             0, 115, "%"),
        ("SMEI (%)",                 "#e377c2", "6. SMEI (%)",                      0, 115, "%"),
    ]

    for idx, (col, color, panel_title, ylim_lo, ylim_hi, suffix) in enumerate(panels):
        row, col_idx = divmod(idx, 3)
        ax = axes[row, col_idx]
        values = df[col].values if col in df.columns else np.zeros(len(models))
        bars = ax.bar(x, values, color=color, edgecolor="white")
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            val = bar.get_height()
            offset = 0.01 * (ylim_hi if ylim_hi else max(abs(val * 1.1), 1))
            ax.text(bar.get_x() + bar.get_width()/2.0, val + offset,
                    f"{val:.1f}{suffix}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "dashboard_tracking_3d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    [✓] Dashboard: {save_path}")

    csv_path = os.path.join(output_dir, "tracking_3d_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"    [✓] Summary CSV: {csv_path}")


def generate_coverage_3d_dashboard(df, output_dir, title):
    """Generate 6-panel coverage dashboard."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    models = df["Model"].tolist()
    x = np.arange(len(models))

    panels = [
        ("Coverage Rate (%)",  "#2ca02c", "1. Area Coverage (%)",               0, 115, "%"),
        ("Attrition Rate",     "#d62728", "2. Agent Attrition (Crashes/ep)",     0, 4.8, ""),
        ("Survival Rate (%)",  "#9467bd", "3. Swarm Survival Rate (%)",          0, 115, "%"),
        ("Avg Battery (%)",    "#ff7f0e", "4. Fleet Battery (%)",                0, 115, "%"),
        ("Mean Reward",        "#1f77b4", "5. Cumulative Mission Return",        None, None, ""),
        ("Alt Changes",        "#e377c2", "6. Altitude Transitions/ep",          None, None, ""),
    ]

    for idx, (col, color, panel_title, ylim_lo, ylim_hi, suffix) in enumerate(panels):
        row, col_idx = divmod(idx, 3)
        ax = axes[row, col_idx]
        values = df[col].values if col in df.columns else np.zeros(len(models))
        bars = ax.bar(x, values, color=color, edgecolor="white")
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            val = bar.get_height()
            offset = 0.01 * (ylim_hi if ylim_hi else max(abs(val * 1.1), 1))
            ax.text(bar.get_x() + bar.get_width()/2.0, val + offset,
                    f"{val:.1f}{suffix}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "dashboard_coverage_3d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    [✓] Dashboard: {save_path}")

    csv_path = os.path.join(output_dir, "coverage_3d_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"    [✓] Summary CSV: {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_task_evaluations(task, n_seeds, loaded_policies):
    """Run all 20 evaluation configurations for a single task."""
    base_output = os.path.expanduser(f"~/Desktop/Results/3D/{task.title()}")
    os.makedirs(base_output, exist_ok=True)
    total_start = time.time()
    run_count = 0

    # Reproducibility manifest: exact checkpoints + evaluation protocol
    manifest = {
        "task": task,
        "n_seeds": n_seeds,
        "seeds": [1000, 1000 + n_seeds - 1],
        "rllib_action_selection": "greedy_argmax" if task == "coverage" else "stochastic_sample (matches 2D tracking)",
        "epymarl_action_selection": "greedy_argmax",
        "checkpoints": {name: path for name, _, _, path in loaded_policies},
        "env_configs": ENV_CONFIGS,
        "fault_regimes": FAULT_REGIMES,
    }
    with open(os.path.join(base_output, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    total_runs = len(ENV_CONFIGS) * len(FAULT_REGIMES)
    for env_config in ENV_CONFIGS:
        obstacle_mask = _resolve_obstacle_mask(env_config["grid_size"]) if env_config["has_obstacles"] else None

        for fault_regime in FAULT_REGIMES:
            run_count += 1
            run_label = f"{env_config['label']} | {fault_regime['label']}"
            output_dir = os.path.join(base_output, env_config["tag"], fault_regime["tag"])

            print(f"\n{'═'*75}")
            print(f"  3D {task.upper()} RUN {run_count}/{total_runs}: {run_label}")
            print(f"  Output: {output_dir}")
            print(f"{'═'*75}")

            results_list = []
            all_raw = []
            run_start = time.time()

            for model_name, base_policy_fn, is_self_heal, _ckpt in loaded_policies:
                print(f"\n    [▶] {model_name}...", end=" ", flush=True)
                model_start = time.time()

                if env_config["grid_size"] != 25:
                    policy_fn = make_resizing_policy(
                        base_policy_fn, source_grid_size=env_config["grid_size"], target_grid_size=25)
                else:
                    policy_fn = base_policy_fn

                if task == "tracking":
                    summary, raw = evaluate_tracking_3d_episode(
                        policy_fn, env_config, fault_regime["fault_prob"],
                        is_self_heal, n_seeds, obstacle_mask
                    )
                    results_list.append({
                        "Model": model_name,
                        "Target Found Rate (%)": summary["target_found_rate"],
                        "Target Found 95% CI (±%)": summary["target_found_ci95"],
                        "Mean Discovery Time": summary["mean_discovery_time"],
                        "Discovery Time Std": summary["discovery_time_std"],
                        "Discovery Time 95% CI (±)": summary["discovery_time_ci95"],
                        "Attrition Rate": summary["attrition_rate"],
                        "Survival Rate (%)": summary["survival_rate"],
                        "Avg Battery (%)": summary["avg_battery"],
                        "SMEI (%)": summary["smei"],
                        "Mean Altitude": summary["mean_altitude"],
                        "Alt Changes": summary["mean_alt_changes"],
                        "N Seeds": summary["n_seeds"],
                    })
                else:  # coverage
                    summary, raw = evaluate_coverage_3d_episode(
                        policy_fn, env_config, fault_regime["fault_prob"],
                        is_self_heal, n_seeds, obstacle_mask
                    )
                    results_list.append({
                        "Model": model_name,
                        "Coverage Rate (%)": summary["coverage_rate"],
                        "Coverage Std (%)": summary["coverage_std"],
                        "Coverage 95% CI (±%)": summary["coverage_ci95"],
                        "Attrition Rate": summary["attrition_rate"],
                        "Survival Rate (%)": summary["survival_rate"],
                        "Avg Battery (%)": summary["avg_battery"],
                        "Mean Charging Steps": summary["mean_charging_steps"],
                        "Mean Reward": summary["mean_reward"],
                        "Mean Shaped Return": summary["mean_shaped_return"],
                        "Mean Altitude": summary["mean_altitude"],
                        "Alt Changes": summary["mean_alt_changes"],
                        "N Seeds": summary["n_seeds"],
                    })

                for r in raw:
                    r["Model"] = model_name
                    all_raw.append(r)

                elapsed = time.time() - model_start
                print(f"({elapsed:.1f}s)")

            # Generate dashboard & save CSVs
            df = pd.DataFrame(results_list)
            dashboard_title = f"DSSE-3D {task.title()} — {run_label} ({n_seeds} seeds)"

            if task == "tracking":
                generate_tracking_3d_dashboard(df, output_dir, dashboard_title)
            else:
                generate_coverage_3d_dashboard(df, output_dir, dashboard_title)

            # Save raw seed-by-seed CSV
            df_raw = pd.DataFrame(all_raw)
            raw_path = os.path.join(output_dir, f"{task}_3d_raw_seed_by_seed.csv")
            os.makedirs(output_dir, exist_ok=True)
            df_raw.to_csv(raw_path, index=False)
            print(f"    [✓] Raw CSV: {raw_path}")

            run_elapsed = time.time() - run_start
            print(f"    ⏱ Run completed in {run_elapsed/60:.1f} min")

    total_elapsed = time.time() - total_start
    print(f"\n{'═'*75}")
    print(f"  3D {task.upper()} ALL {total_runs} RUNS COMPLETE! ({total_elapsed/60:.1f} min total)")
    print(f"  Results: {base_output}")
    print(f"{'═'*75}\n")


def main():
    parser = argparse.ArgumentParser(
        description="DSSE-3D Master Evaluation Suite (6 envs × 5 fault regimes × 12 models × N seeds)"
    )
    parser.add_argument("--task", type=str, default="all", choices=["tracking", "coverage", "all"],
                        help="Task to evaluate: 'tracking', 'coverage', or 'all'")
    parser.add_argument("--n_seeds", type=int, default=100,
                        help="Number of test seeds per model (default: 100)")
    args = parser.parse_args()

    print(f"\n{'╔'*75}")
    print(f"  DSSE-3D MASTER EVALUATION SUITE")
    print(f"  Tasks: {args.task.upper()} | Seeds: {args.n_seeds}")
    print(f"  Environments: 6 (25x25, 25x25+Obs, 50x50, 50x50+Obs, 75x75, 75x75+Obs)")
    print(f"  Fault Regimes: 5 (0.0, 0.00025, 0.0005, 0.00075, 0.0010)")
    print(f"  Models: 12 (RSPO, MAPPO, QMIX, MAA2C, COMA, I-DQN × Vanilla/SelfHeal)")
    total_episodes = args.n_seeds * 12 * 30
    if args.task == "all":
        total_episodes *= 2
    print(f"  Total Episodes: {total_episodes:,}")
    print(f"{'╚'*75}\n")

    tasks_to_run = ["tracking", "coverage"] if args.task == "all" else [args.task]

    for task in tasks_to_run:
        loaded_policies = load_all_policies(task)
        run_task_evaluations(task, args.n_seeds, loaded_policies)

    print(f"\n{'═'*75}")
    print(f"  ALL DSSE-3D EVALUATIONS COMPLETE!")
    print(f"  Results saved to: ~/Desktop/Results/3D/")
    print(f"{'═'*75}\n")


if __name__ == "__main__":
    main()
