"""
50x50 Grid & Obstacle Airspace Evaluation Suite (All 12 MARL Models x 100 Seeds)
=================================================================================
Extends the standard 25x25 evaluation with 3 additional test suites:

Suite 1: 50x50 Extended Grid (T=1500 Steps) — Zero-shot spatial scaling
Suite 2: 25x25 Obstacle Airspace (T=750 Steps) — Layout 3 Scattered Skyline
Suite 3: 50x50 Obstacle Airspace (T=1500 Steps) — Combined scaling + obstacles

Uses SpatialRescalingWrapper for 50x50 → 25x25 observation downsampling
so pre-trained 25x25 policies can run zero-shot on 4x larger grids.

Adjusted Coverage Rate = Searched Free Cells / (Total Cells - N_obstacles) * 100%

Exports:
  - evaluation_50x50_summary.csv
  - evaluation_obstacles_summary.csv
  - fig_50x50_spatial_scaling.png
  - fig_obstacle_airspace_resilience.png
  - fig_master_50x50_obstacles_dashboard.png
"""

import os
import sys
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt

# Add source paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from global_reward_wrapper import GlobalRewardWrapper
from obstacle_airspace_wrapper import ObstacleAirspaceWrapper
from spatial_rescaling_wrapper import SpatialRescalingWrapper
from modules.agents.cnn_agent import CNNAgent
from modules.agents.rnn_agent import RNNAgent

# Import model classes and evaluation functions from existing suite
from evaluate_all_models import (
    load_rllib_policy, evaluate_rllib_model, evaluate_epymarl_model,
    DirectPyTorchPolicyWrapper, MockArgs
)
from train_rspo_vanilla import RSPOModel as RSPOModelVanilla
from train_rspo_cnn_cov import RSPOModel as RSPOModelSelfHeal
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla


# ══════════════════════════════════════════════════════════════════════
# ENVIRONMENT CREATORS
# ══════════════════════════════════════════════════════════════════════

# --- Suite 1: 50x50 Extended Grid (T=1500, No Obstacles) ---
def create_50x50_vanilla_env():
    matrix_path = os.path.join(SRC_DIR, "uniform_matrix_50.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=1500, drone_amount=4, prob_matrix_path=matrix_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=250, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0
    env = SpatialRescalingWrapper(env, target_grid_size=25)
    return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])


def create_50x50_selfheal_env():
    matrix_path = os.path.join(SRC_DIR, "uniform_matrix_50.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=1500, drone_amount=4, prob_matrix_path=matrix_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=250, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 2.5
    env.COMPENSATION_PENALTY = -0.5
    env.COMPENSATION_HORIZON = 100
    env = GlobalRewardWrapper(env, mixing_alpha=0.5)
    env = SpatialRescalingWrapper(env, target_grid_size=25)
    return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])


# --- Suite 2: 25x25 Obstacle Airspace (T=750, Layout 3 Skyline) ---
def create_25x25_obstacle_vanilla_env():
    obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_25.npy"))
    prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=750, drone_amount=4, prob_matrix_path=prob_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0
    env = ObstacleAirspaceWrapper(env, obstacle_mask)
    return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])


def create_25x25_obstacle_selfheal_env():
    obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_25.npy"))
    prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=750, drone_amount=4, prob_matrix_path=prob_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 2.5
    env.COMPENSATION_PENALTY = -0.5
    env.COMPENSATION_HORIZON = 100
    env = GlobalRewardWrapper(env, mixing_alpha=0.5)
    env = ObstacleAirspaceWrapper(env, obstacle_mask)
    return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])


# --- Suite 3: 50x50 Obstacle Airspace (T=1500, Layout 3 Skyline + Rescaling) ---
def create_50x50_obstacle_vanilla_env():
    obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_50.npy"))
    prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=1500, drone_amount=4, prob_matrix_path=prob_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=250, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0
    env = ObstacleAirspaceWrapper(env, obstacle_mask)
    env = SpatialRescalingWrapper(env, target_grid_size=25)
    return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])


def create_50x50_obstacle_selfheal_env():
    obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_50.npy"))
    prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=1500, drone_amount=4, prob_matrix_path=prob_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=250, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 2.5
    env.COMPENSATION_PENALTY = -0.5
    env.COMPENSATION_HORIZON = 100
    env = GlobalRewardWrapper(env, mixing_alpha=0.5)
    env = ObstacleAirspaceWrapper(env, obstacle_mask)
    env = SpatialRescalingWrapper(env, target_grid_size=25)
    return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])


# ══════════════════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════════════════

def run_suite(suite_name, rllib_models, epymarl_models, num_seeds=100, seed_start=1000, free_cells=625):
    """Run a full evaluation suite across all 12 models."""
    raw_rows = []
    summary_rows = []

    print(f"\n{'='*70}")
    print(f"  {suite_name}")
    print(f"{'='*70}")

    # RLlib models
    for model_key, meta in rllib_models.items():
        print(f"\n[Evaluating RLlib Model: {model_key}]")
        algo = load_rllib_policy(
            meta["algo_type"], meta["env_name"], meta["model_name"],
            meta["model_cls"], meta["path"]
        )
        records = evaluate_rllib_model(algo, meta["env_fn"], num_seeds=num_seeds, seed_start=seed_start)
        algo.stop()

        _process_records(model_key, "RLlib", records, raw_rows, summary_rows, free_cells=free_cells)

    # EPyMARL models
    for model_key, meta in epymarl_models.items():
        print(f"\n[Evaluating EPyMARL Model: {model_key}]")
        if not os.path.isfile(meta["path"]):
            print(f"  Warning: Checkpoint not found. Skipping...")
            continue
        records = evaluate_epymarl_model(meta["path"], meta["env_fn"], num_seeds=num_seeds, seed_start=seed_start)
        _process_records(model_key, "EPyMARL", records, raw_rows, summary_rows, free_cells=free_cells)

    return raw_rows, summary_rows


def _process_records(model_key, framework, records, raw_rows, summary_rows, free_cells=625):
    """Process evaluation records into raw rows and summary statistics."""
    covs = [r["coverage_rate"] * 100 for r in records]
    deaths = [r["agent_deaths"] for r in records]
    # Pure Mission Return: cells_searched * 5.0 reward per cell
    rewards_list = [r["coverage_rate"] * free_cells * 5.0 for r in records]
    batts = [r["avg_battery_level"] for r in records]
    chg_steps = [r["charging_steps"] for r in records]

    for r in records:
        pure_reward = r["coverage_rate"] * free_cells * 5.0
        raw_rows.append({
            "Algorithm": model_key, "Framework": framework, "Seed": r["seed"],
            "Coverage_Rate_%": round(r["coverage_rate"] * 100, 2),
            "Agent_Deaths": r["agent_deaths"],
            "Cumulative_Reward": round(pure_reward, 2),
            "Avg_Battery_Level": round(r["avg_battery_level"], 2),
            "Charging_Steps": r["charging_steps"],
        })

    cov_mean, cov_std = np.mean(covs), np.std(covs)
    ci95 = 1.96 * (cov_std / np.sqrt(len(covs)))

    print(f"  - Coverage Rate:    {cov_mean:.2f}% ± {cov_std:.2f}%  [95% CI: ±{ci95:.2f}%]")
    print(f"  - Attrition Rate:   {np.mean(deaths):.2f} crashes/ep  (Survival: {(1 - np.mean(deaths)/4)*100:.1f}%)")
    print(f"  - Cumulative Return: {np.mean(rewards_list):.2f}")
    print(f"  - Battery Health:   {np.mean(batts):.2f}% | {np.mean(chg_steps):.1f} charging steps")

    summary_rows.append({
        "Algorithm": model_key, "Framework": framework,
        "Coverage Rate Mean (%)": round(cov_mean, 2),
        "Coverage Std (%)": round(cov_std, 2),
        "Coverage 95% CI (±%)": round(ci95, 2),
        "Agent Attrition Rate (Crashes/Ep)": round(np.mean(deaths), 2),
        "Swarm Survival Rate (%)": round((1 - np.mean(deaths)/4.0)*100, 1),
        "Cumulative Return Mean": round(np.mean(rewards_list), 2),
        "Fleet Battery Level Mean (%)": round(np.mean(batts), 2),
        "Mean Charging Steps": round(np.mean(chg_steps), 1),
    })


# ══════════════════════════════════════════════════════════════════════
# CHART GENERATION
# ══════════════════════════════════════════════════════════════════════

COLOR_MAP = {
    "RSPO_Vanilla": "#8A2BE2", "RSPO_SelfHeal": "#BA55D3",
    "MAPPO_Vanilla": "#1F77B4", "MAPPO_SelfHeal": "#6BAED6",
    "QMIX_Vanilla": "#FF7F0E", "QMIX_SelfHeal": "#FDBE85",
    "MAA2C_Vanilla": "#17BECF", "MAA2C_SelfHeal": "#9EDAE5",
    "I-DQN_Vanilla": "#2CA02C", "I-DQN_SelfHeal": "#A1D99B",
    "COMA_Vanilla": "#D62728", "COMA_SelfHeal": "#FF9896",
}


def generate_comparison_chart(df_25, df_50, metric_col, title, ylabel, filename, higher_is_better=True):
    """Generate a grouped bar chart comparing 25x25 vs 50x50 performance."""
    algorithms = df_25["Algorithm"].tolist()
    x = np.arange(len(algorithms))
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 6))
    bars_25 = ax.bar(x - width/2, df_25[metric_col], width, label="25×25 Grid (T=750)",
                     color=[COLOR_MAP.get(a, "#333") for a in algorithms], edgecolor="black", linewidth=0.5, alpha=0.85)
    bars_50 = ax.bar(x + width/2, df_50[metric_col], width, label="50×50 Grid (T=1500)",
                     color=[COLOR_MAP.get(a, "#333") for a in algorithms], edgecolor="black", linewidth=0.5, alpha=0.5, hatch="//")

    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=25, ha="right", fontweight="bold", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    # Add value labels
    for bar in bars_25:
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    for bar in bars_50:
        h = bar.get_height()
        if h != 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


def generate_master_dashboard(df_25, df_50, df_obs_25, df_obs_50, filename):
    """Generate master 4-panel dashboard comparing all test regimes."""
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    algorithms = df_25["Algorithm"].tolist()
    x = np.arange(len(algorithms))
    width = 0.2
    colors = [COLOR_MAP.get(a, "#333") for a in algorithms]

    # Hatch patterns: Solid, Vertical Stripes, Horizontal Stripes, Dots
    suite_styles = [
        {"label": "25×25 (Solid)",              "hatch": "",   "alpha": 1.0},
        {"label": "50×50 (Vertical Stripes)",    "hatch": "||", "alpha": 0.85},
        {"label": "25×25+Obs (Horizontal Stripes)", "hatch": "--", "alpha": 0.85},
        {"label": "50×50+Obs (Dots)",            "hatch": "..", "alpha": 0.85},
    ]

    def _draw_grouped(ax, metric, ylim=None):
        dfs = [df_25, df_50, df_obs_25, df_obs_50]
        offsets = [-1.5, -0.5, 0.5, 1.5]
        for i, (df, off, style) in enumerate(zip(dfs, offsets, suite_styles)):
            ax.bar(x + off*width, df[metric], width,
                   label=style["label"], color=colors,
                   edgecolor="black", linewidth=0.5,
                   alpha=style["alpha"], hatch=style["hatch"])
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=35, fontsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        if ylim is not None:
            ax.set_ylim(*ylim)

    # Panel 1: Coverage Rate
    _draw_grouped(axs[0, 0], "Coverage Rate Mean (%)", ylim=(0, 100))
    axs[0, 0].set_title("Coverage Rate (%) Across All Test Regimes", fontweight="bold")

    # Panel 2: Agent Attrition
    _draw_grouped(axs[0, 1], "Agent Attrition Rate (Crashes/Ep)", ylim=(0, 4.5))
    axs[0, 1].set_title("Agent Attrition Rate (Crashes/Ep) [Lower is Better]", fontweight="bold")

    # Panel 3: Cumulative Return
    _draw_grouped(axs[1, 0], "Cumulative Return Mean")
    axs[1, 0].set_title("Cumulative Mission Return [Higher is Better]", fontweight="bold")

    # Panel 4: Battery Health
    _draw_grouped(axs[1, 1], "Fleet Battery Level Mean (%)")
    axs[1, 1].set_title("Fleet Battery Health (%) [Higher is Better]", fontweight="bold")

    plt.suptitle("Zero-Shot Spatial Generalization & Obstacle Airspace: Master Dashboard (100 Seeds)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    desktop = os.path.expanduser("~/Desktop")

    # Model checkpoint paths (same as evaluate_all_models.py)
    rllib_checkpoints = {
        "RSPO_Vanilla": {
            "algo_type": "ppo", "model_name": "RSPOModelVanilla_50", "model_cls": RSPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_vanilla_v1/PPO_DSSE_Coverage_RSPO_Vanilla_5bfc7_00000_0_2026-08-01_04-35-01/checkpoint_000199"),
        },
        "RSPO_SelfHeal": {
            "algo_type": "ppo", "model_name": "RSPOModelSelfHeal_50", "model_cls": RSPOModelSelfHeal,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_rspo_v1/PPO_DSSE_Coverage_RSPO_394ce_00000_0_2026-07-26_00-36-29/checkpoint_000165"),
        },
        "MAPPO_Vanilla": {
            "algo_type": "ppo", "model_name": "MAPPOModelVanilla_50", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_vanilla_proper/PPO_DSSE_Coverage_cc7e1_00000_0_2026-07-05_03-37-08/checkpoint_000141"),
        },
        "MAPPO_SelfHeal": {
            "algo_type": "ppo", "model_name": "MAPPOModelSelfHeal_50", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_selfheal_comparison/PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04/checkpoint_000023"),
        },
        "I-DQN_Vanilla": {
            "algo_type": "dqn", "model_name": "IDQNModelVanilla_50", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_vanilla_I-DQN_vanilla_proper/DQN_DSSE_Coverage_61018_00000_0_2026-07-16_09-47-37/checkpoint_000609"),
        },
        "I-DQN_SelfHeal": {
            "algo_type": "dqn", "model_name": "IDQNModelSelfHeal_50", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_I-DQN_selfheal_proper_pt2/DQN_DSSE_Coverage_8aefa_00000_0_2026-07-16_05-38-15/checkpoint_000565"),
        },
    }

    epymarl_checkpoints = {
        "QMIX_Vanilla": os.path.join(SRC_DIR, "results/models/qmix_true_vanilla_v1_seed533532401_dsse_coverage_2026-07-18 02:06:54.683275/17040000/agent.th"),
        "QMIX_SelfHeal": os.path.join(SRC_DIR, "results/models/qmix_selfheal_v2_seed533532401_dsse_coverage_2026-07-17 03:23:47.879210/19044000/agent.th"),
        "MAA2C_Vanilla": os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_vanilla_seed782869923_dsse_coverage_2026-06-16 18:45:27.715438/18795000/agent.th"),
        "MAA2C_SelfHeal": os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_original_seed776978465_dsse_coverage_2026-07-15 06:30:31.971452/12675000/agent.th"),
        "COMA_Vanilla": os.path.join(SRC_DIR, "results/models/coma_vanilla_proper_seed452926189_dsse_coverage_2026-07-10 01:59:53.354626/19338000/agent.th"),
        "COMA_SelfHeal": os.path.join(SRC_DIR, "results/models/coma_selfhealing_final_stable_seed339532096_dsse_coverage_2026-07-06 10:52:47.933607/19986000/agent.th"),
    }

    # Map environment functions for each suite
    def build_suite_models(vanilla_env_fn, selfheal_env_fn):
        """Build model dicts with correct env_fn for each suite."""
        from evaluate_all_models import create_vanilla_env, create_selfheal_env

        rllib = {}
        for key, meta in rllib_checkpoints.items():
            env_fn = vanilla_env_fn if "Vanilla" in key else selfheal_env_fn
            # For RLlib policy loading, we still need a 25x25 env for config registration
            env_name_suffix = key.replace("-", "_")
            rllib[key] = {
                **meta,
                "env_name": f"DSSE_50x50_{env_name_suffix}",
                "env_fn": env_fn,
            }

        epymarl = {}
        for key, path in epymarl_checkpoints.items():
            env_fn = vanilla_env_fn if "Vanilla" in key else selfheal_env_fn
            epymarl[key] = {"type": "epymarl", "env_fn": env_fn, "path": path}

        return rllib, epymarl

    # Load existing 25x25 baseline results from CSV
    baseline_csv = os.path.join(desktop, "evaluation_summary_all_metrics.csv")
    if os.path.exists(baseline_csv):
        df_25_baseline = pd.read_csv(baseline_csv)
        print(f"Loaded 25x25 baseline from: {baseline_csv}")
    else:
        df_25_baseline = None
        print("WARNING: No 25x25 baseline CSV found. Run evaluate_all_models.py first.")

    # ════════════════════════════════
    # Suite 1: 50x50 Extended Grid
    # ════════════════════════════════
    rllib_50, epymarl_50 = build_suite_models(create_50x50_vanilla_env, create_50x50_selfheal_env)
    raw_50, summary_50 = run_suite("Suite 1: 50×50 Extended Grid (T=1500, Zero-Shot Transfer)", rllib_50, epymarl_50, free_cells=2500)

    df_50 = pd.DataFrame(summary_50)
    df_50.to_csv(os.path.join(desktop, "evaluation_50x50_summary.csv"), index=False)
    pd.DataFrame(raw_50).to_csv(os.path.join(desktop, "evaluation_50x50_raw.csv"), index=False)

    # ════════════════════════════════
    # Suite 2: 25x25 Obstacle Airspace
    # ════════════════════════════════
    rllib_obs25, epymarl_obs25 = build_suite_models(create_25x25_obstacle_vanilla_env, create_25x25_obstacle_selfheal_env)
    raw_obs25, summary_obs25 = run_suite("Suite 2: 25×25 Obstacle Airspace (T=750, Layout 3 Skyline)", rllib_obs25, epymarl_obs25, free_cells=529)

    df_obs25 = pd.DataFrame(summary_obs25)
    df_obs25.to_csv(os.path.join(desktop, "evaluation_obstacles_25x25_summary.csv"), index=False)
    pd.DataFrame(raw_obs25).to_csv(os.path.join(desktop, "evaluation_obstacles_25x25_raw.csv"), index=False)

    # ════════════════════════════════
    # Suite 3: 50x50 Obstacle Airspace
    # ════════════════════════════════
    rllib_obs50, epymarl_obs50 = build_suite_models(create_50x50_obstacle_vanilla_env, create_50x50_obstacle_selfheal_env)
    raw_obs50, summary_obs50 = run_suite("Suite 3: 50×50 Obstacle Airspace (T=1500, Layout 3 + Rescaling)", rllib_obs50, epymarl_obs50, free_cells=2404)

    df_obs50 = pd.DataFrame(summary_obs50)
    df_obs50.to_csv(os.path.join(desktop, "evaluation_obstacles_50x50_summary.csv"), index=False)
    pd.DataFrame(raw_obs50).to_csv(os.path.join(desktop, "evaluation_obstacles_50x50_raw.csv"), index=False)

    # ════════════════════════════════
    # Generate Charts
    # ════════════════════════════════
    print("\n=================== GENERATING CHARTS ===================")

    if df_25_baseline is not None:
        # 50x50 Spatial Scaling Comparison
        generate_comparison_chart(
            df_25_baseline, df_50,
            "Coverage Rate Mean (%)",
            "Zero-Shot Spatial Generalization: 25×25 vs 50×50 Coverage Rate",
            "Coverage Rate (%)",
            os.path.join(desktop, "fig_50x50_spatial_scaling.png")
        )

        # Master Dashboard
        generate_master_dashboard(
            df_25_baseline, df_50, df_obs25, df_obs50,
            os.path.join(desktop, "fig_master_50x50_obstacles_dashboard.png")
        )

    # Obstacle Resilience Chart
    generate_comparison_chart(
        df_obs25, df_obs50,
        "Coverage Rate Mean (%)",
        "Obstacle Airspace Resilience: 25×25 vs 50×50 (Layout 3 Skyline)",
        "Adjusted Coverage Rate (%)",
        os.path.join(desktop, "fig_obstacle_airspace_resilience.png")
    )

    print("\n=================== ALL SUITES COMPLETE ===================")
    print(f"Saved CSVs and charts to: {desktop}")

    ray.shutdown()
