"""
DSSE-2D Master Evaluation Suite Runner
========================================
Runs 2D evaluations across:
    - 4 Environments: 25x25 Standard, 25x25 Obstacles, 50x50 Standard, 50x50 Obstacles
    - 5 Fault Severity Levels:
        1. fault_prob = 0.0      (No Faults)
        2. fault_prob = 0.00025  (Low Faults)
        3. fault_prob = 0.0005   (Standard Faults)
        4. fault_prob = 0.00075  (High Faults)
        5. fault_prob = 0.0010   (Severe Faults)
    - 12 MARL Models: RSPO, MAPPO, QMIX, MAA2C, COMA, I-DQN × Vanilla/SelfHeal
    - 100 Test Seeds per model

Saved to: ~/Desktop/Results/2D/
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
from tracking.tracking_env_creator import make_tracking_env

from dsse_3d.policy_loaders import (
    load_rllib_ppo_policy, load_rllib_dqn_policy, load_epymarl_policy_strict,
    make_resizing_policy,
)
from train_rspo_v2_vanilla import RSPOModelV2
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla


# ──────────────────────────────────────────────────────────────────────────────
# Environment Configurations & Fault Regimes
# ──────────────────────────────────────────────────────────────────────────────

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
]

FAULT_REGIMES = [
    {"tag": "No_Faults_0.0", "fault_prob": 0.0, "label": "No Faults (fault_prob = 0.0)"},
    {"tag": "Fault_0.00025", "fault_prob": 0.00025, "label": "Low Faults (fault_prob = 0.00025)"},
    {"tag": "Active_Faults_0.0005", "fault_prob": 0.0005, "label": "Standard Faults (fault_prob = 0.0005)"},
    {"tag": "Fault_0.00075", "fault_prob": 0.00075, "label": "High Faults (fault_prob = 0.00075)"},
    {"tag": "Fault_0.001", "fault_prob": 0.0010, "label": "Severe Faults (fault_prob = 0.0010)"},
]


def _resolve_obstacle_mask(grid_size):
    path = os.path.join(SRC_DIR, f"obstacle_skyline_{grid_size}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def _resolve_prob_matrix(grid_size, has_obstacles):
    if has_obstacles:
        path = os.path.join(SRC_DIR, f"obstacle_prob_matrix_{grid_size}.npy")
    else:
        path = os.path.join(SRC_DIR, f"uniform_matrix_{grid_size}.npy")
    if os.path.exists(path):
        return path
    return None


def make_coverage_env_2d(grid_size=25, drone_amount=4, timestep_limit=750, max_battery=125,
                         fault_prob=0.0005, is_self_heal=False, positions=None, obstacle_mask=None,
                         prob_matrix_path=None):
    if prob_matrix_path is None:
        prob_matrix_path = os.path.join(SRC_DIR, f"uniform_matrix_{grid_size}.npy")

    env = CoverageDroneSwarmSearch(
        timestep_limit=timestep_limit,
        drone_amount=drone_amount,
        prob_matrix_path=prob_matrix_path,
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
        max_battery=max_battery,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=fault_prob,
    )

    if not is_self_heal:
        env.COMPENSATION_BONUS = 0.0
        env.COMPENSATION_PENALTY = 0.0
        env.COMPENSATION_HORIZON = 0
    else:
        env.COMPENSATION_BONUS = 2.5
        env.COMPENSATION_PENALTY = -0.5
        env.COMPENSATION_HORIZON = 100
        env = GlobalRewardWrapper(env, mixing_alpha=0.5)

    if obstacle_mask is not None:
        from obstacle_airspace_wrapper import ObstacleAirspaceWrapper
        env = ObstacleAirspaceWrapper(env, obstacle_mask)

    if grid_size != 25:
        from spatial_rescaling_wrapper import SpatialRescalingWrapper
        env = SpatialRescalingWrapper(env, target_grid_size=25)

    if positions is None:
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)

    return env


def build_model_registry(task):
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
            ("I-DQN Vanilla",   "epymarl", None,             False, "results/models/*iql_tracking_vanilla*/**/agent.th"),
            ("I-DQN SelfHeal",  "epymarl", None,             True,  "results/models/*iql_tracking_selfheal*/**/agent.th"),
        ]
    else:  # coverage
        return [
            ("RSPO Vanilla",   "rllib",     RSPOModelV2,      False, "ray_res/DSSE_Coverage/*RSPO_V2_Vanilla*/**/checkpoint_*"),
            ("RSPO SelfHeal",  "rllib",     RSPOModelV2,      True,  "ray_res/DSSE_Coverage/*RSPO_V2_SelfHeal*/**/checkpoint_*"),
            ("MAPPO Vanilla",  "rllib",     MAPPOModelVanilla, False, "ray_res/DSSE_Coverage/*MAPPO_vanilla*/**/checkpoint_*"),
            ("MAPPO SelfHeal", "rllib",     MAPPOModelVanilla, True,  "ray_res/DSSE_Coverage/*MAPPO_selfheal*/**/checkpoint_*"),
            ("QMIX Vanilla",   "rllib_dqn", None,             False, "ray_res/DSSE_Coverage/*QMIX_vanilla_I-DQN_vanilla*/**/checkpoint_*"),
            ("QMIX SelfHeal",  "epymarl",   None,             True,  "results/models/*qmix_selfheal*/**/agent.th"),
            ("MAA2C Vanilla",  "epymarl",   None,             False, "results/models/*maa2c_cnn_bigbatch_v3_vanilla*/**/agent.th"),
            ("MAA2C SelfHeal", "epymarl",   None,             True,  "results/models/*maa2c_cnn_bigbatch_v3_*/**/agent.th"),
            ("COMA Vanilla",   "epymarl",   None,             False, "results/models/*coma_vanilla*/**/agent.th"),
            ("COMA SelfHeal",  "epymarl",   None,             True,  "results/models/*coma_selfhealing*/**/agent.th"),
            ("I-DQN Vanilla",   "rllib_dqn", None,             False, "ray_res/DSSE_Coverage/*QMIX_vanilla_I-DQN_vanilla*/**/checkpoint_*"),
            ("I-DQN SelfHeal",  "rllib_dqn", None,             True,  "ray_res/DSSE_Coverage/*QMIX_I-DQN_selfheal*/**/checkpoint_*"),
        ]


def load_all_2d_policies(task):
    from tracking.evaluate_tracking_models import find_best_post_10m_checkpoint
    registry = build_model_registry(task)
    loaded = []

    for model_name, framework, model_cls, is_self_heal, rel_pattern in registry:
        abs_pattern = os.path.join(SRC_DIR, rel_pattern)
        ckpt_path = find_best_post_10m_checkpoint(abs_pattern, framework="rllib" if "rllib" in framework else "epymarl", min_timesteps=10_000_000)
        
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found for 2D {task} / {model_name}: {rel_pattern}")

        if framework == "rllib":
            raw_fn = load_rllib_ppo_policy(model_cls, ckpt_path, model_name=model_name, greedy=(task == "coverage"))
        elif framework == "rllib_dqn":
            raw_fn = load_rllib_dqn_policy(ckpt_path)
        else:
            raw_fn = load_epymarl_policy_strict(ckpt_path, n_agents=4)

        loaded.append((model_name, raw_fn, is_self_heal, framework))

    return loaded


def evaluate_tracking_2d_episode(policy_fn, env_config, fault_prob, is_self_heal, n_seeds, obstacle_mask):
    grid_size = env_config["grid_size"]
    if grid_size != 25:
        policy_fn = make_resizing_policy(policy_fn, source_grid_size=grid_size, target_grid_size=25)
    records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_tracking_env(
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

        while not done and step < env_config["timestep_limit"]:
            step += 1
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

            if all(term.values()) or all(trunc.values()):
                done = True

        tf = 100.0 if target_found else 0.0
        dt = discovery_step if target_found else env_config["timestep_limit"]
        n_drones = 4
        surv = ((n_drones - len(crashed_agents)) / n_drones) * 100.0
        batt = ((total_battery / battery_samples) / env_config["max_battery"] * 100.0) if battery_samples > 0 else 0.0

        f_target = max(0.001, tf / 100.0)
        f_speed = max(0.001, (env_config["timestep_limit"] - dt) / env_config["timestep_limit"])
        f_survival = max(0.001, surv / 100.0)
        f_battery = max(0.001, batt / 100.0)
        smei = (f_target**0.20 * f_speed**0.20 * f_survival**0.30 * f_battery**0.30) * 100.0

        records.append({
            "seed": seed,
            "target_found": 1 if target_found else 0,
            "discovery_step": dt,
            "crash_events": crash_events,
            "survival_rate": surv,
            "avg_battery": batt,
            "smei": smei,
        })

    tf_vals = [r["target_found"] * 100.0 for r in records]
    dt_vals = [r["discovery_step"] for r in records]
    ci = lambda vals: 1.96 * np.std(vals) / np.sqrt(len(vals))

    summary = {
        "target_found_rate": round(np.mean(tf_vals), 2),
        "target_found_ci95": round(ci(tf_vals), 2),
        "mean_discovery_time": round(np.mean(dt_vals), 2),
        "discovery_time_std": round(np.std(dt_vals), 2),
        "discovery_time_ci95": round(ci(dt_vals), 2),
        "attrition_rate": round(np.mean([r["crash_events"] for r in records]), 2),
        "survival_rate": round(np.mean([r["survival_rate"] for r in records]), 2),
        "avg_battery": round(np.mean([r["avg_battery"] for r in records]), 2),
        "smei": round(np.mean([r["smei"] for r in records]), 2),
        "n_seeds": len(records),
    }
    return summary, records


def evaluate_coverage_2d_episode(policy_fn, env_config, fault_prob, is_self_heal, n_seeds, obstacle_mask):
    grid_size = env_config["grid_size"]
    if grid_size != 25:
        policy_fn = make_resizing_policy(policy_fn, source_grid_size=grid_size, target_grid_size=25)
    prob_path = _resolve_prob_matrix(grid_size, env_config["has_obstacles"])
    records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_coverage_env_2d(
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
        done = False
        step = 0
        total_reward = 0.0
        total_battery = 0
        battery_samples = 0
        crash_events = 0
        crashed_agents = set()

        while not done and step < env_config["timestep_limit"]:
            step += 1
            actions = policy_fn(obs, env.possible_agents, reset=(step == 1))
            actions = {a: v for a, v in actions.items() if a in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)

            total_reward += sum(rewards.values())

            for agent, info in infos.items():
                if isinstance(info, dict):
                    if info.get("stranded_event", False):
                        crash_events += 1
                        crashed_agents.add(agent)
                    batt = info.get("battery")
                    if batt is not None:
                        total_battery += batt
                        battery_samples += 1

            if all(term.values()) or all(trunc.values()):
                done = True

        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'unwrapped'):
            base_env = base_env.unwrapped

        final_coverage = len(getattr(base_env, 'seen_states', set())) / env_config["free_cells"] if env_config["free_cells"] > 0 else 0.0
        mission_return = final_coverage * env_config["free_cells"] * 5.0

        n_drones = 4
        surv = ((n_drones - len(crashed_agents)) / n_drones) * 100.0
        batt = ((total_battery / battery_samples) / env_config["max_battery"] * 100.0) if battery_samples > 0 else 0.0

        f_cov = max(0.001, final_coverage)
        f_surv = max(0.001, surv / 100.0)
        f_batt = max(0.001, batt / 100.0)
        smei = (f_cov**0.40 * f_surv**0.30 * f_batt**0.30) * 100.0

        records.append({
            "seed": seed,
            "coverage_pct": final_coverage * 100.0,
            "mission_return": mission_return,
            "shaped_return": total_reward,
            "crash_events": crash_events,
            "survival_rate": surv,
            "avg_battery": batt,
            "smei": smei,
        })

    cov_vals = [r["coverage_pct"] for r in records]
    cov_mean = np.mean(cov_vals)
    surv_mean = np.mean([r["survival_rate"] for r in records])
    batt_mean = np.mean([r["avg_battery"] for r in records])
    smei_mean = np.mean([r["smei"] for r in records])

    summary = {
        "coverage_rate": round(cov_mean, 2),
        "coverage_std": round(np.std(cov_vals), 2),
        "coverage_ci95": round(1.96 * np.std(cov_vals) / np.sqrt(len(cov_vals)), 2),
        "attrition_rate": round(np.mean([r["crash_events"] for r in records]), 2),
        "survival_rate": round(surv_mean, 2),
        "avg_battery": round(batt_mean, 2),
        "smei": round(smei_mean, 2),
        "mean_reward": round(np.mean([r["mission_return"] for r in records]), 2),
        "mean_shaped_return": round(np.mean([r["shaped_return"] for r in records]), 2),
        "n_seeds": len(records),
    }
    return summary, records


def generate_tracking_2d_dashboard(df, output_dir, title):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    models = df["Model"].tolist()
    x = np.arange(len(models))

    panels = [
        ("target_found_rate",  "#2ca02c", "1. Target Intercept Rate (%)",    0, 115, "%"),
        ("mean_discovery_time","#1f77b4", "2. Mean Time-to-Intercept",       None, None, ""),
        ("attrition_rate",     "#d62728", "3. Agent Attrition (Crashes/ep)",  0, 4.8, ""),
        ("survival_rate",      "#9467bd", "4. Swarm Survival Rate (%)",       0, 115, "%"),
        ("avg_battery",        "#ff7f0e", "5. Fleet Battery (%)",             0, 115, "%"),
        ("smei",               "#e377c2", "6. SMEI (%)",                      0, 115, "%"),
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
    save_path = os.path.join(output_dir, "dashboard_tracking_2d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    csv_path = os.path.join(output_dir, "tracking_2d_summary.csv")
    df.to_csv(csv_path, index=False)


def generate_coverage_2d_dashboard(df, output_dir, title):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    models = df["Model"].tolist()
    x = np.arange(len(models))

    panels = [
        ("coverage_rate",  "#2ca02c", "1. Area Coverage (%)",               0, 115, "%"),
        ("attrition_rate", "#d62728", "2. Agent Attrition (Crashes/ep)",     0, 4.8, ""),
        ("survival_rate",  "#9467bd", "3. Swarm Survival Rate (%)",          0, 115, "%"),
        ("avg_battery",    "#ff7f0e", "4. Fleet Battery (%)",                0, 115, "%"),
        ("mean_reward",    "#1f77b4", "5. Cumulative Mission Return",        None, None, ""),
        ("smei",           "#e377c2", "6. SMEI (%)",                      0, 115, "%"),
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
    save_path = os.path.join(output_dir, "dashboard_coverage_2d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    csv_path = os.path.join(output_dir, "coverage_2d_summary.csv")
    df.to_csv(csv_path, index=False)


def run_2d_evaluations(task, n_seeds, loaded_policies, grid_size_filter=None):
    base_output = os.path.expanduser(f"~/Desktop/Results/2D/{task.title()}")
    os.makedirs(base_output, exist_ok=True)

    configs_to_run = [c for c in ENV_CONFIGS if c["grid_size"] == grid_size_filter] if grid_size_filter else ENV_CONFIGS
    total_runs = len(configs_to_run) * len(FAULT_REGIMES)
    run_idx = 0

    for env_cfg in configs_to_run:
        obstacle_mask = _resolve_obstacle_mask(env_cfg["grid_size"]) if env_cfg["has_obstacles"] else None

        for fault_cfg in FAULT_REGIMES:
            run_idx += 1
            run_tag = f"{env_cfg['tag']}/{fault_cfg['tag']}"
            output_dir = os.path.join(base_output, env_cfg["tag"], fault_cfg["tag"])
            os.makedirs(output_dir, exist_ok=True)

            print(f"\n{'═'*75}")
            print(f"  2D {task.upper()} RUN {run_idx}/{total_runs}: {env_cfg['label']} | {fault_cfg['label']}")
            print(f"  Output: {output_dir}")
            print(f"{'═'*75}\n")

            results_list = []
            all_raw = []

            for model_name, raw_fn, is_self_heal, framework in loaded_policies:
                print(f"    [▶] {model_name}...", end="", flush=True)
                t0 = time.time()

                if task == "tracking":
                    summary, records = evaluate_tracking_2d_episode(
                        raw_fn, env_cfg, fault_cfg["fault_prob"], is_self_heal, n_seeds, obstacle_mask
                    )
                else:
                    summary, records = evaluate_coverage_2d_episode(
                        raw_fn, env_cfg, fault_cfg["fault_prob"], is_self_heal, n_seeds, obstacle_mask
                    )

                t_el = time.time() - t0
                print(f" ({t_el:.1f}s)")

                summary["Model"] = model_name
                results_list.append(summary)

                for r in records:
                    r_copy = dict(r)
                    r_copy["Model"] = model_name
                    all_raw.append(r_copy)

            df = pd.DataFrame(results_list)
            dashboard_title = f"DSSE-2D {task.title()} — {env_cfg['label']} ({fault_cfg['label']})"

            if task == "tracking":
                generate_tracking_2d_dashboard(df, output_dir, dashboard_title)
            else:
                generate_coverage_2d_dashboard(df, output_dir, dashboard_title)

            df_raw = pd.DataFrame(all_raw)
            raw_path = os.path.join(output_dir, f"{task}_2d_raw_seed_by_seed.csv")
            df_raw.to_csv(raw_path, index=False)
            print(f"    [✓] Raw CSV: {raw_path}")


def main():
    parser = argparse.ArgumentParser(description="DSSE-2D Master Evaluation Suite")
    parser.add_argument("--task", type=str, default="all", choices=["tracking", "coverage", "all"])
    parser.add_argument("--grid_size", type=int, default=None, choices=[25, 50, 75],
                        help="Filter evaluations by grid size (e.g., --grid_size 75 for 75x75 environments only)")
    parser.add_argument("--n_seeds", type=int, default=100)
    args = parser.parse_args()

    print(f"  Environments: 4 (25x25, 25x25+Obs, 50x50, 50x50+Obs)")
    print(f"  Fault Regimes: 5 (0.0, 0.00025, 0.0005, 0.00075, 0.0010)")
    print(f"{'╚'*75}\n")

    tasks_to_run = ["tracking", "coverage"] if args.task == "all" else [args.task]

    for task in tasks_to_run:
        loaded = load_all_2d_policies(task)
        run_2d_evaluations(task, args.n_seeds, loaded, grid_size_filter=args.grid_size)

    print(f"\n{'═'*75}")
    print(f"  ALL DSSE-2D EVALUATIONS COMPLETE!")
    print(f"  Results saved to: ~/Desktop/Results/2D/")
    print(f"{'═'*75}\n")


if __name__ == "__main__":
    main()
