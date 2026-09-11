"""
Evaluation and Dashboard Generation Script for Dynamic Target Tracking
========================================================================
Evaluates trained tracking models over 100 test seeds per model.

Metrics Evaluated:
  1. Target Found Rate (%)
  2. Mean Time-to-Discovery (steps)
  3. Agent Attrition Rate (crashes per episode)
  4. Swarm Survival Rate (%)
  5. Fleet Average Battery Level (%)

Outputs:
  - ~/Desktop/Results/Tracking_No_Faults/dashboard_tracking_25x25.png & summary.csv
  - ~/Desktop/Results/Tracking_Active_Faults/dashboard_tracking_25x25.png & summary.csv
"""

import os
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure src directory is on sys.path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tracking.tracking_env_creator import make_tracking_env


def evaluate_tracking_model(model_eval_fn, is_fault_active=True, n_seeds=100, grid_size=25):
    """
    Evaluates a policy function over n_seeds in the tracking environment.
    """
    fault_prob = 0.0005 if is_fault_active else 0.0
    
    target_found_list = []
    discovery_times = []
    attrition_list = []
    survival_list = []
    battery_list = []

    for seed in range(1, n_seeds + 1):
        env = make_tracking_env(
            grid_size=grid_size,
            drone_amount=4,
            person_amount=1,
            person_initial_position=(grid_size // 2, grid_size // 2),
            timestep_limit=750,
            max_battery=125,
            fault_prob=fault_prob,
            is_self_heal=False,
            drift_speed=1.0,
        )

        obs, infos = env.reset(seed=seed)
        done = False
        step = 0
        target_found = False
        discovery_step = 750

        total_battery = 0
        battery_samples = 0
        crashes = 0

        while not done and step < 750:
            step += 1
            actions = model_eval_fn(obs, env.agents)
            obs, rewards, term, trunc, infos = env.step(actions)

            # Check target discovery
            for agent, r in rewards.items():
                if r >= 1.0 or infos.get(agent, {}).get("search_and_find", False):
                    target_found = True
                    discovery_step = step
                    done = True
                    break

            # Collect battery and crash stats
            for agent, info in infos.items():
                if isinstance(info, dict):
                    if info.get("stranded_event", False):
                        crashes += 1
                    batt = info.get("battery")
                    if batt is not None:
                        total_battery += batt
                        battery_samples += 1

            if all(term.values()) or all(trunc.values()):
                done = True

        target_found_list.append(100.0 if target_found else 0.0)
        if target_found:
            discovery_times.append(discovery_step)
        attrition_list.append(crashes)
        
        alive_drones = sum(1 for agent in env.agents if obs.get(agent) is not None)
        survival_list.append((alive_drones / 4.0) * 100.0)
        
        avg_batt = (total_battery / battery_samples) if battery_samples > 0 else 0.0
        battery_list.append(avg_batt)

    return {
        "target_found_rate": float(np.mean(target_found_list)),
        "mean_discovery_time": float(np.mean(discovery_times)) if discovery_times else 750.0,
        "attrition_rate": float(np.mean(attrition_list)),
        "survival_rate": float(np.mean(survival_list)),
        "avg_battery": float(np.mean(battery_list)),
    }


def generate_tracking_dashboard(df_results, output_dir, title_prefix="Tracking Benchmark"):
    """Generates a 5-panel dashboard PNG for dynamic target tracking performance."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"{title_prefix} — Dynamic Target Pursuit & Tracking (25x25)", fontsize=18, fontweight="bold", y=0.98)

    palette = sns.color_palette("muted")

    # 1. Target Found Rate (%)
    sns.barplot(data=df_results, x="Model", y="Target Found Rate (%)", ax=axes[0, 0], palette="Greens_d")
    axes[0, 0].set_title("1. Target Found Rate (%)", fontsize=13, fontweight="bold")
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylim(0, 105)

    # 2. Mean Time-to-Discovery (steps)
    sns.barplot(data=df_results, x="Model", y="Mean Discovery Time (steps)", ax=axes[0, 1], palette="Blues_d")
    axes[0, 1].set_title("2. Mean Time-to-Discovery (steps)", fontsize=13, fontweight="bold")
    axes[0, 1].tick_params(axis='x', rotation=45)

    # 3. Agent Attrition Rate (crashes/ep)
    sns.barplot(data=df_results, x="Model", y="Attrition Rate", ax=axes[0, 2], palette="Reds_d")
    axes[0, 2].set_title("3. Agent Attrition Rate (Crashes/ep)", fontsize=13, fontweight="bold")
    axes[0, 2].tick_params(axis='x', rotation=45)

    # 4. Swarm Survival Rate (%)
    sns.barplot(data=df_results, x="Model", y="Survival Rate (%)", ax=axes[1, 0], palette="Purples_d")
    axes[1, 0].set_title("4. Swarm Survival Rate (%)", fontsize=13, fontweight="bold")
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylim(0, 105)

    # 5. Fleet Avg Battery Level (%)
    sns.barplot(data=df_results, x="Model", y="Avg Battery Level (%)", ax=axes[1, 1], palette="Oranges_d")
    axes[1, 1].set_title("5. Fleet Avg Battery Level (%)", fontsize=13, fontweight="bold")
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Hide unused 6th subplot
    fig.delaxes(axes[1, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "dashboard_tracking_25x25.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Tracking Dashboard PNG saved to: {save_path}")

    csv_path = os.path.join(output_dir, "tracking_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[✓] Tracking Summary CSV saved to: {csv_path}")


if __name__ == "__main__":
    print("Evaluate tracking models utility module loaded successfully.")
