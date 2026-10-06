"""
DSSE-3D Evaluation Benchmark Suite (Tracking & Coverage)
=========================================================
Evaluates all 12 MARL models in the 3D voxel environment across 100 test
seeds, generating leaderboard dashboards and CSV summaries.

This script mirrors the 2D evaluation scripts but runs in 3D:
    - evaluate_tracking_models.py -> 3D tracking benchmark
    - evaluate_all_models.py      -> 3D coverage benchmark

3D-Specific Metrics (in addition to all 2D metrics):
    - Mean Drone Altitude: Average altitude across all drones and steps.
    - Altitude Transitions: Number of altitude changes per episode.
    - 3D Obstacle Collisions: Count of collisions with extruded obstacle prisms.
    - Mid-Air Collisions: Count of inter-drone 3D voxel collisions.

Usage:
    python -m dsse_3d.evaluate_3d --task tracking
    python -m dsse_3d.evaluate_3d --task coverage
    python -m dsse_3d.evaluate_3d --task tracking --grid_size 25 --map_type obstacles
"""

import os
import sys
import argparse
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


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation Functions
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_tracking_3d(
    policy_fn,
    is_fault_active=True,
    n_seeds=100,
    grid_size=25,
    is_self_heal=False,
    obstacle_mask=None,
):
    """
    Evaluate a policy in the 3D tracking environment over n_seeds.

    Returns dict with all tracking + 3D-specific metrics.
    """
    fault_prob = 0.0005 if is_fault_active else 0.0

    target_found_list = []
    discovery_times = []
    attrition_list = []
    survival_list = []
    battery_list = []
    altitude_changes_list = []
    midair_collisions_list = []
    obstacle_collisions_3d_list = []
    mean_altitude_list = []
    seed_records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_tracking_env_3d(
            grid_size=grid_size,
            drone_amount=4,
            person_amount=1,
            person_initial_position=(grid_size // 2, grid_size // 2),
            timestep_limit=750,
            max_battery=125,
            fault_prob=fault_prob,
            is_self_heal=is_self_heal,
            drift_speed=1.0,
            obstacle_mask=obstacle_mask,
        )

        obs, infos = env.reset(seed=seed)
        done = False
        step = 0
        target_found = False
        discovery_step = 750

        total_battery = 0
        battery_samples = 0
        crash_events = 0
        crashed_agents = set()
        total_altitude = 0
        altitude_samples = 0

        while not done and step < 750:
            step += 1
            actions = policy_fn(obs, env.agents)
            obs, rewards, term, trunc, infos = env.step(actions)

            # Check target discovery
            for agent in list(rewards.keys()):
                agent_info = infos.get(agent, {})
                if isinstance(agent_info, dict) and agent_info.get("Found", False):
                    target_found = True
                    discovery_step = step
                    done = True
                    break

            # Collect stats
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

        # 3D metrics
        metrics_3d = env.get_3d_metrics()

        target_found_list.append(100.0 if target_found else 0.0)
        discovery_times.append(discovery_step if target_found else 750)
        attrition_list.append(crash_events)

        n_drones = 4
        unique_crashed = len(crashed_agents)
        survival_rate = ((n_drones - unique_crashed) / n_drones) * 100.0
        survival_list.append(survival_rate)

        avg_batt = ((total_battery / battery_samples) / 125.0 * 100.0) if battery_samples > 0 else 0.0
        battery_list.append(avg_batt)

        mean_alt = (total_altitude / altitude_samples) if altitude_samples > 0 else 1.0
        mean_altitude_list.append(mean_alt)
        altitude_changes_list.append(metrics_3d["altitude_changes"])
        midair_collisions_list.append(metrics_3d["midair_collisions"])
        obstacle_collisions_3d_list.append(metrics_3d["obstacle_collisions_3d"])

        seed_records.append({
            "seed": seed,
            "target_found": 1 if target_found else 0,
            "discovery_step": discovery_step,
            "crash_events": crash_events,
            "survival_rate": survival_rate,
            "avg_battery": avg_batt,
            "mean_altitude": mean_alt,
            "altitude_changes": metrics_3d["altitude_changes"],
            "midair_collisions": metrics_3d["midair_collisions"],
            "obstacle_collisions_3d": metrics_3d["obstacle_collisions_3d"],
        })

    res_tf = float(np.mean(target_found_list))
    res_dt = float(np.mean(discovery_times))
    res_surv = float(np.mean(survival_list))
    res_batt = float(np.mean(battery_list))

    # SMEI
    f_target = max(0.001, res_tf / 100.0)
    f_speed = max(0.001, (750.0 - res_dt) / 750.0)
    f_survival = max(0.001, res_surv / 100.0)
    f_battery = max(0.001, res_batt / 100.0)
    smei = (f_target**0.20 * f_speed**0.20 * f_survival**0.30 * f_battery**0.30) * 100.0

    return {
        "target_found_rate": res_tf,
        "mean_discovery_time": res_dt,
        "attrition_rate": float(np.mean(attrition_list)),
        "survival_rate": res_surv,
        "avg_battery": res_batt,
        "smei": smei,
        "mean_altitude": float(np.mean(mean_altitude_list)),
        "mean_altitude_changes": float(np.mean(altitude_changes_list)),
        "mean_midair_collisions": float(np.mean(midair_collisions_list)),
        "mean_obstacle_collisions_3d": float(np.mean(obstacle_collisions_3d_list)),
        "raw_records": seed_records,
    }


def evaluate_coverage_3d(
    policy_fn,
    is_fault_active=True,
    n_seeds=100,
    grid_size=25,
    is_self_heal=False,
    obstacle_mask=None,
):
    """
    Evaluate a policy in the 3D coverage environment over n_seeds.

    Returns dict with all coverage + 3D-specific metrics.
    """
    fault_prob = 0.0005 if is_fault_active else 0.0

    coverage_list = []
    attrition_list = []
    survival_list = []
    battery_list = []
    reward_list = []
    altitude_changes_list = []
    midair_collisions_list = []
    obstacle_collisions_3d_list = []
    mean_altitude_list = []
    seed_records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_coverage_env_3d(
            grid_size=grid_size,
            drone_amount=4,
            timestep_limit=750,
            max_battery=125,
            fault_prob=fault_prob,
            is_self_heal=is_self_heal,
            obstacle_mask=obstacle_mask,
        )

        obs, infos = env.reset(seed=seed)
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

        while not done and step < 750:
            step += 1
            actions = policy_fn(obs, env.agents)
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
                    alt = info.get("altitude")
                    if alt is not None:
                        total_altitude += alt
                        altitude_samples += 1
                    cov = info.get("coverage_rate")
                    if cov is not None:
                        final_coverage = cov

            if all(term.values()) or all(trunc.values()):
                done = True

        metrics_3d = env.get_3d_metrics()

        coverage_list.append(final_coverage * 100.0)
        attrition_list.append(crash_events)

        n_drones = 4
        unique_crashed = len(crashed_agents)
        survival_rate = ((n_drones - unique_crashed) / n_drones) * 100.0
        survival_list.append(survival_rate)

        avg_batt = ((total_battery / battery_samples) / 125.0 * 100.0) if battery_samples > 0 else 0.0
        battery_list.append(avg_batt)
        reward_list.append(total_reward)

        mean_alt = (total_altitude / altitude_samples) if altitude_samples > 0 else 1.0
        mean_altitude_list.append(mean_alt)
        altitude_changes_list.append(metrics_3d["altitude_changes"])
        midair_collisions_list.append(metrics_3d["midair_collisions"])
        obstacle_collisions_3d_list.append(metrics_3d["obstacle_collisions_3d"])

        seed_records.append({
            "seed": seed,
            "coverage_pct": final_coverage * 100.0,
            "crash_events": crash_events,
            "survival_rate": survival_rate,
            "avg_battery": avg_batt,
            "total_reward": total_reward,
            "mean_altitude": mean_alt,
            "altitude_changes": metrics_3d["altitude_changes"],
            "midair_collisions": metrics_3d["midair_collisions"],
            "obstacle_collisions_3d": metrics_3d["obstacle_collisions_3d"],
        })

    return {
        "coverage_rate": float(np.mean(coverage_list)),
        "attrition_rate": float(np.mean(attrition_list)),
        "survival_rate": float(np.mean(survival_list)),
        "avg_battery": float(np.mean(battery_list)),
        "mean_reward": float(np.mean(reward_list)),
        "mean_altitude": float(np.mean(mean_altitude_list)),
        "mean_altitude_changes": float(np.mean(altitude_changes_list)),
        "mean_midair_collisions": float(np.mean(midair_collisions_list)),
        "mean_obstacle_collisions_3d": float(np.mean(obstacle_collisions_3d_list)),
        "raw_records": seed_records,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dashboard Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_3d_tracking_dashboard(df_results, output_dir, title_prefix="DSSE-3D Tracking"):
    """Generate an 8-panel dashboard including 3D-specific metrics."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(f"{title_prefix} — 3D Voxel Performance Dashboard", fontsize=17, fontweight="bold", y=0.98)

    models = df_results["Model"].tolist()
    x = np.arange(len(models))

    metrics_config = [
        ("Target Found Rate (%)", "#2ca02c", "1. Target Intercept Rate (%)", 0, 115, "%"),
        ("Mean Discovery Time", "#1f77b4", "2. Mean Time-to-Intercept (steps)", None, None, ""),
        ("Attrition Rate", "#d62728", "3. Agent Attrition (Crashes/ep)", 0, 4.8, ""),
        ("Survival Rate (%)", "#9467bd", "4. Swarm Survival Rate (%)", 0, 115, "%"),
        ("Avg Battery (%)", "#ff7f0e", "5. Fleet Battery (%)", 0, 115, "%"),
        ("SMEI (%)", "#e377c2", "6. SMEI (%)", 0, 115, "%"),
        ("Mean Altitude", "#17becf", "7. Mean Drone Altitude (Z)", 0, 6, ""),
        ("3D Collisions", "#bcbd22", "8. 3D Obstacle + MidAir Collisions", None, None, ""),
    ]

    for idx, (col, color, title, ylim_lo, ylim_hi, suffix) in enumerate(metrics_config):
        row, col_idx = divmod(idx, 4)
        ax = axes[row, col_idx]

        if col in df_results.columns:
            values = df_results[col].values
        else:
            values = np.zeros(len(models))

        bars = ax.bar(x, values, color=color, edgecolor="white")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha='right', fontsize=7)
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.0,
                val + (0.01 * (ylim_hi or val * 1.1)),
                f"{val:.1f}{suffix}", ha='center', va='bottom', fontsize=6.5, fontweight='bold'
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "dashboard_tracking_3d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] 3D Tracking Dashboard saved to: {save_path}")

    csv_path = os.path.join(output_dir, "tracking_3d_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[✓] 3D Tracking CSV saved to: {csv_path}")


def generate_3d_coverage_dashboard(df_results, output_dir, title_prefix="DSSE-3D Coverage"):
    """Generate an 8-panel dashboard for 3D coverage results."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 4, figsize=(24, 11))
    fig.suptitle(f"{title_prefix} — 3D Voxel Performance Dashboard", fontsize=17, fontweight="bold", y=0.98)

    models = df_results["Model"].tolist()
    x = np.arange(len(models))

    metrics_config = [
        ("Coverage Rate (%)", "#2ca02c", "1. Area Coverage (%)", 0, 115, "%"),
        ("Attrition Rate", "#d62728", "2. Agent Attrition (Crashes/ep)", 0, 4.8, ""),
        ("Survival Rate (%)", "#9467bd", "3. Swarm Survival Rate (%)", 0, 115, "%"),
        ("Avg Battery (%)", "#ff7f0e", "4. Fleet Battery (%)", 0, 115, "%"),
        ("Mean Reward", "#1f77b4", "5. Cumulative Mission Return", None, None, ""),
        ("Mean Altitude", "#17becf", "6. Mean Drone Altitude (Z)", 0, 6, ""),
        ("Alt Changes", "#e377c2", "7. Altitude Transitions/ep", None, None, ""),
        ("3D Collisions", "#bcbd22", "8. 3D Obstacle + MidAir Collisions", None, None, ""),
    ]

    for idx, (col, color, title, ylim_lo, ylim_hi, suffix) in enumerate(metrics_config):
        row, col_idx = divmod(idx, 4)
        ax = axes[row, col_idx]

        if col in df_results.columns:
            values = df_results[col].values
        else:
            values = np.zeros(len(models))

        bars = ax.bar(x, values, color=color, edgecolor="white")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha='right', fontsize=7)
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.grid(axis='y', linestyle='--', alpha=0.4)

        for bar in bars:
            val = bar.get_height()
            offset = 0.01 * (ylim_hi if ylim_hi else max(abs(val * 1.1), 1))
            ax.text(
                bar.get_x() + bar.get_width()/2.0,
                val + offset,
                f"{val:.1f}{suffix}", ha='center', va='bottom', fontsize=6.5, fontweight='bold'
            )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "dashboard_coverage_3d.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] 3D Coverage Dashboard saved to: {save_path}")

    csv_path = os.path.join(output_dir, "coverage_3d_summary.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[✓] 3D Coverage CSV saved to: {csv_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DSSE-3D Evaluation Benchmark")
    parser.add_argument("--task", type=str, default="tracking", choices=["tracking", "coverage"])
    parser.add_argument("--grid_size", type=int, default=25)
    parser.add_argument("--map_type", type=str, default="standard", choices=["standard", "obstacles"])
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--no_faults", action="store_true")
    args = parser.parse_args()

    from tracking.evaluate_tracking_models import (
        load_rllib_policy, load_epymarl_policy, find_best_post_10m_checkpoint,
    )
    from train_rspo_v2_vanilla import RSPOModelV2
    from train_mappo_vanilla import CNNModel as MAPPOModelVanilla

    # Load obstacle mask
    obstacle_mask = None
    if args.map_type == "obstacles":
        obs_path = os.path.join(SRC_DIR, f"obstacle_skyline_{args.grid_size}.npy")
        if os.path.exists(obs_path):
            obstacle_mask = np.load(obs_path)

    # Model definitions
    task_prefix = "Tracking" if args.task == "tracking" else "Coverage"

    tracking_models = [
        ("RSPO Vanilla", f"ray_res/DSSE_{task_prefix}/*RSPO_V2_*Vanilla*/**/checkpoint_*", "rllib", RSPOModelV2, False),
        ("RSPO SelfHeal", f"ray_res/DSSE_{task_prefix}/*RSPO_V2_*SelfHeal*/**/checkpoint_*", "rllib", RSPOModelV2, True),
        ("MAPPO Vanilla", f"ray_res/DSSE_{task_prefix}/*MAPPO_*Vanilla*/**/checkpoint_*", "rllib", MAPPOModelVanilla, False),
        ("MAPPO SelfHeal", f"ray_res/DSSE_{task_prefix}/*MAPPO_*SelfHeal*/**/checkpoint_*", "rllib", MAPPOModelVanilla, True),
        ("QMIX Vanilla", f"results/models/*qmix_{'tracking_' if args.task == 'tracking' else ''}vanilla*/**/agent.th", "epymarl", None, False),
        ("QMIX SelfHeal", f"results/models/*qmix_{'tracking_' if args.task == 'tracking' else ''}selfheal*/**/agent.th", "epymarl", None, True),
        ("MAA2C Vanilla", f"results/models/*maa2c_{'tracking_' if args.task == 'tracking' else ''}vanilla*/**/agent.th", "epymarl", None, False),
        ("MAA2C SelfHeal", f"results/models/*maa2c_{'tracking_' if args.task == 'tracking' else ''}selfheal*/**/agent.th", "epymarl", None, True),
        ("COMA Vanilla", f"results/models/*coma_{'tracking_' if args.task == 'tracking' else ''}vanilla*/**/agent.th", "epymarl", None, False),
        ("COMA SelfHeal", f"results/models/*coma_{'tracking_' if args.task == 'tracking' else ''}selfheal*/**/agent.th", "epymarl", None, True),
        ("I-DQN Vanilla", f"results/models/*iql_{'tracking_' if args.task == 'tracking' else ''}vanilla*/**/agent.th", "epymarl", None, False),
        ("I-DQN SelfHeal", f"results/models/*iql_{'tracking_' if args.task == 'tracking' else ''}selfheal*/**/agent.th", "epymarl", None, True),
    ]

    is_fault_active = not args.no_faults
    fault_tag = "No_Faults_0.0" if not is_fault_active else "Active_Faults_0.0005"
    map_tag = f"_{args.map_type}" if args.map_type != "standard" else ""

    base_output_dir = os.path.expanduser(f"~/Desktop/Results/3D/{args.task.title()}{map_tag}_{args.grid_size}x{args.grid_size}")

    print(f"\n{'='*75}")
    print(f"  DSSE-3D EVALUATION: {args.task.upper()} | {args.grid_size}x{args.grid_size} | {args.map_type}")
    print(f"{'='*75}\n")

    # Load policies
    loaded_policies = []
    for model_name, pattern, framework, model_cls, is_self_heal in tracking_models:
        full_pattern = os.path.join(SRC_DIR, pattern)
        ckpt_path = find_best_post_10m_checkpoint(full_pattern, framework, min_timesteps=10_000_000)

        print(f"[▶] Resolving: {model_name}")
        if ckpt_path:
            print(f"    Loading {framework.upper()} from: {ckpt_path}")
            if framework == "rllib":
                policy_fn = load_rllib_policy(model_cls, ckpt_path, model_name=model_name)
                policy_fn = wrap_rllib_policy_for_3d(policy_fn)
            else:
                policy_fn = load_epymarl_policy(ckpt_path)
                policy_fn = wrap_epymarl_policy_for_3d(policy_fn)
        else:
            print(f"    [!] No checkpoint found. Using random stub.")
            def make_stub():
                def stub_fn(obs_dict, agents, *a, **kw):
                    return {agent: np.random.randint(0, 9) for agent in agents}
                return wrap_rllib_policy_for_3d(stub_fn)
            policy_fn = make_stub()

        loaded_policies.append((model_name, policy_fn, is_self_heal))

    # Run evaluations
    results_list = []
    all_raw = []

    for model_name, policy_fn, is_self_heal in loaded_policies:
        print(f"\n  [▶] Evaluating {model_name} in 3D...")

        if args.task == "tracking":
            res = evaluate_tracking_3d(
                policy_fn, is_fault_active=is_fault_active,
                n_seeds=args.n_seeds, grid_size=args.grid_size,
                is_self_heal=is_self_heal, obstacle_mask=obstacle_mask,
            )
            results_list.append({
                "Model": model_name,
                "Target Found Rate (%)": round(res["target_found_rate"], 2),
                "Mean Discovery Time": round(res["mean_discovery_time"], 2),
                "Attrition Rate": round(res["attrition_rate"], 2),
                "Survival Rate (%)": round(res["survival_rate"], 2),
                "Avg Battery (%)": round(res["avg_battery"], 2),
                "SMEI (%)": round(res["smei"], 2),
                "Mean Altitude": round(res["mean_altitude"], 2),
                "3D Collisions": round(res["mean_midair_collisions"] + res["mean_obstacle_collisions_3d"], 2),
            })
        else:
            res = evaluate_coverage_3d(
                policy_fn, is_fault_active=is_fault_active,
                n_seeds=args.n_seeds, grid_size=args.grid_size,
                is_self_heal=is_self_heal, obstacle_mask=obstacle_mask,
            )
            results_list.append({
                "Model": model_name,
                "Coverage Rate (%)": round(res["coverage_rate"], 2),
                "Attrition Rate": round(res["attrition_rate"], 2),
                "Survival Rate (%)": round(res["survival_rate"], 2),
                "Avg Battery (%)": round(res["avg_battery"], 2),
                "Mean Reward": round(res["mean_reward"], 2),
                "Mean Altitude": round(res["mean_altitude"], 2),
                "Alt Changes": round(res["mean_altitude_changes"], 2),
                "3D Collisions": round(res["mean_midair_collisions"] + res["mean_obstacle_collisions_3d"], 2),
            })

        for r in res["raw_records"]:
            r["Model"] = model_name
            all_raw.append(r)

    df_results = pd.DataFrame(results_list)
    regime_dir = os.path.join(base_output_dir, fault_tag)

    if args.task == "tracking":
        generate_3d_tracking_dashboard(df_results, regime_dir, title_prefix=f"DSSE-3D Tracking {args.grid_size}x{args.grid_size}")
    else:
        generate_3d_coverage_dashboard(df_results, regime_dir, title_prefix=f"DSSE-3D Coverage {args.grid_size}x{args.grid_size}")

    # Save raw data
    df_raw = pd.DataFrame(all_raw)
    raw_path = os.path.join(regime_dir, f"{args.task}_3d_raw_seed_by_seed.csv")
    df_raw.to_csv(raw_path, index=False)
    print(f"[✓] Raw seed-by-seed CSV saved to: {raw_path}")

    print(f"\n{'='*75}")
    print(f"  DSSE-3D {args.task.upper()} EVALUATION COMPLETE!")
    print(f"  Results: {regime_dir}")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
