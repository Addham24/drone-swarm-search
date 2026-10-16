#!/usr/bin/env python3
"""
Symmetric Action-Selection Benchmark Runner: MAPPO vs RSPO
===========================================================
Evaluates MAPPO Vanilla & MAPPO SelfHeal under the exact same temperature-scaled
categorical sampling distribution as RSPO:
  - 25x25 Coverage:  tau = 0.20
  - 50x50 Coverage:  tau = 0.30
  - Tracking (all):  tau = 0.10

Target Directories:
  - ~/Desktop/Results/2D-mappo/
  - ~/Desktop/Results/3D-mappo/
(Leaves ~/Desktop/Results/2D/ and ~/Desktop/Results/3D/ 100% frozen and untouched.)

Updates raw CSVs, summary CSVs, per-condition dashboards, and generates publication
degradation curves in 2D-mappo/degradation_curves/ and 3D-mappo/degradation_curves/.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from run_all_2d_evaluations import (
    ENV_CONFIGS as ENV_CONFIGS_2D,
    FAULT_REGIMES as FAULT_REGIMES_2D,
    _resolve_obstacle_mask as resolve_mask_2d,
    evaluate_coverage_2d_episode,
    evaluate_tracking_2d_episode,
    generate_coverage_2d_dashboard,
    generate_tracking_2d_dashboard,
)
from dsse_3d.run_all_3d_evaluations import (
    ENV_CONFIGS as ENV_CONFIGS_3D,
    FAULT_REGIMES as FAULT_REGIMES_3D,
    _resolve_obstacle_mask as resolve_mask_3d,
    evaluate_coverage_3d_episode,
    evaluate_tracking_3d_episode,
    generate_coverage_3d_dashboard,
    generate_tracking_3d_dashboard,
)
from dsse_3d.policy_loaders import load_rllib_ppo_policy, make_resizing_policy
from dsse_3d.policy_adapters import wrap_rllib_policy_for_3d
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla
from tracking.evaluate_tracking_models import find_best_post_10m_checkpoint


MAPPO_REGISTRY = {
    "tracking": [
        ("MAPPO Vanilla",  MAPPOModelVanilla, False, "ray_res/DSSE_Tracking/*MAPPO_Tracking_Vanilla*/**/checkpoint_*"),
        ("MAPPO SelfHeal", MAPPOModelVanilla, True,  "ray_res/DSSE_Tracking/*MAPPO_Tracking_SelfHeal*/**/checkpoint_*"),
    ],
    "coverage": [
        ("MAPPO Vanilla",  MAPPOModelVanilla, False, "ray_res/DSSE_Coverage/*MAPPO_vanilla*/**/checkpoint_*"),
        ("MAPPO SelfHeal", MAPPOModelVanilla, True,  "ray_res/DSSE_Coverage/*MAPPO_selfheal*/**/checkpoint_*"),
    ],
}


def load_symmetric_mappo_policies(dim, task):
    """Load MAPPO models with sample_mappo=True so they sample identically to RSPO."""
    registry = MAPPO_REGISTRY[task]
    policies = []
    greedy_flag = (task == "coverage")

    for model_name, model_cls, is_self_heal, rel_pattern in registry:
        abs_pattern = os.path.join(SRC_DIR, rel_pattern)
        if os.path.exists(abs_pattern):
            ckpt_path = abs_pattern
        else:
            ckpt_path = find_best_post_10m_checkpoint(abs_pattern, framework="rllib", min_timesteps=10_000_000)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found for {dim.upper()} {task} {model_name}: {rel_pattern}")

        # Load with sample_mappo=True
        raw_fn = load_rllib_ppo_policy(
            model_cls, ckpt_path, model_name=model_name, greedy=greedy_flag, sample_mappo=True
        )

        if dim == "3d":
            policy_fn = wrap_rllib_policy_for_3d(raw_fn)
        else:
            policy_fn = raw_fn

        policies.append((model_name, policy_fn, is_self_heal, ckpt_path))

    return policies


def recompute_summary_from_raw(df_raw, task, dim, env_cfg):
    """Recompute summary DataFrame matching the exact metrics of all models in raw df."""
    summary_rows = []
    models = df_raw["Model"].unique()

    for m in models:
        sub = df_raw[df_raw["Model"] == m]
        n_seeds = len(sub)
        if task == "coverage":
            cov_vals = sub["coverage_pct"].values
            cov_mean = np.mean(cov_vals)
            cov_std = np.std(cov_vals)
            cov_ci95 = 1.96 * cov_std / np.sqrt(n_seeds) if n_seeds > 0 else 0.0
            surv_mean = np.mean(sub["survival_rate"].values)
            batt_mean = np.mean(sub["avg_battery"].values)
            if "smei" in sub.columns:
                smei_mean = np.mean(sub["smei"].values)
            else:
                f_cov = max(0.001, cov_mean / 100.0)
                f_surv = max(0.001, surv_mean / 100.0)
                f_batt = max(0.001, batt_mean / 100.0)
                smei_mean = (f_cov**0.40 * f_surv**0.30 * f_batt**0.30) * 100.0
            ret_mean = np.mean(sub["mission_return"].values)
            shaped_mean = np.mean(sub["shaped_return"].values)
            attr_mean = np.mean(sub["crash_events"].values)

            if dim == "2d":
                row = {
                    "coverage_rate": round(cov_mean, 2),
                    "coverage_std": round(cov_std, 2),
                    "coverage_ci95": round(cov_ci95, 2),
                    "attrition_rate": round(attr_mean, 2),
                    "survival_rate": round(surv_mean, 2),
                    "avg_battery": round(batt_mean, 2),
                    "smei": round(smei_mean, 2),
                    "mean_reward": round(ret_mean, 2),
                    "mean_shaped_return": round(shaped_mean, 2),
                    "n_seeds": n_seeds,
                    "Model": m,
                }
            else:  # 3d
                chg_mean = np.mean(sub["charging_steps"].values) if "charging_steps" in sub.columns else 0.0
                alt_mean = np.mean(sub["mean_altitude"].values) if "mean_altitude" in sub.columns else 2.0
                alt_chg = np.mean(sub["altitude_changes"].values) if "altitude_changes" in sub.columns else 0.0
                row = {
                    "Model": m,
                    "Coverage Rate (%)": round(cov_mean, 2),
                    "Coverage Std (%)": round(cov_std, 2),
                    "Coverage 95% CI (±%)": round(cov_ci95, 2),
                    "Attrition Rate": round(attr_mean, 2),
                    "Survival Rate (%)": round(surv_mean, 2),
                    "Avg Battery (%)": round(batt_mean, 2),
                    "SMEI (%)": round(smei_mean, 2),
                    "Mean Charging Steps": round(chg_mean, 2),
                    "Mean Reward": round(ret_mean, 2),
                    "Mean Shaped Return": round(shaped_mean, 2),
                    "Mean Altitude": round(alt_mean, 2),
                    "Alt Changes": round(alt_chg, 2),
                    "N Seeds": n_seeds,
                }
            summary_rows.append(row)
        else:  # tracking
            tf_vals = sub["target_found"].values * 100.0 if "target_found" in sub.columns else sub["intercept"].values * 100.0
            dt_col = "discovery_step" if "discovery_step" in sub.columns else "steps_to_intercept"
            dt_vals = sub[dt_col].values
            ci = lambda vals: 1.96 * np.std(vals) / np.sqrt(len(vals)) if len(vals) > 0 else 0.0

            tf_mean = np.mean(tf_vals)
            dt_mean = np.mean(dt_vals)
            dt_std = np.std(dt_vals)
            surv_mean = np.mean(sub["survival_rate"].values)
            batt_mean = np.mean(sub["avg_battery"].values)
            if "smei" in sub.columns:
                smei_mean = np.mean(sub["smei"].values)
            else:
                tl = env_cfg["timestep_limit"]
                f_target = max(0.001, tf_mean / 100.0)
                f_speed = max(0.001, (tl - dt_mean) / tl)
                f_survival = max(0.001, surv_mean / 100.0)
                f_battery = max(0.001, batt_mean / 100.0)
                smei_mean = (f_target**0.20 * f_speed**0.20 * f_survival**0.30 * f_battery**0.30) * 100.0
            attr_mean = np.mean(sub["crash_events"].values)

            if dim == "2d":
                row = {
                    "target_found_rate": round(tf_mean, 2),
                    "target_found_ci95": round(ci(tf_vals), 2),
                    "mean_discovery_time": round(dt_mean, 2),
                    "discovery_time_std": round(dt_std, 2),
                    "discovery_time_ci95": round(ci(dt_vals), 2),
                    "attrition_rate": round(attr_mean, 2),
                    "survival_rate": round(surv_mean, 2),
                    "avg_battery": round(batt_mean, 2),
                    "smei": round(smei_mean, 2),
                    "n_seeds": n_seeds,
                    "Model": m,
                }
            else:  # 3d
                alt_mean = np.mean(sub["mean_altitude"].values) if "mean_altitude" in sub.columns else 2.0
                alt_chg = np.mean(sub["altitude_changes"].values) if "altitude_changes" in sub.columns else 0.0
                row = {
                    "Model": m,
                    "Target Found Rate (%)": round(tf_mean, 2),
                    "Target Found 95% CI (±%)": round(ci(tf_vals), 2),
                    "Mean Discovery Time": round(dt_mean, 2),
                    "Discovery Time Std": round(dt_std, 2),
                    "Discovery Time 95% CI (±)": round(ci(dt_vals), 2),
                    "Attrition Rate": round(attr_mean, 2),
                    "Survival Rate (%)": round(surv_mean, 2),
                    "Avg Battery (%)": round(batt_mean, 2),
                    "SMEI (%)": round(smei_mean, 2),
                    "Mean Altitude": round(alt_mean, 2),
                    "Alt Changes": round(alt_chg, 2),
                    "N Seeds": n_seeds,
                }
            summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def run_symmetric_benchmark(dim="2d", task="coverage", n_seeds=100, grid_size_filter=None):
    base_res_dir = os.path.expanduser(f"~/Desktop/Results/{dim.upper()}-mappo")
    task_res_dir = os.path.join(base_res_dir, task.title())
    os.makedirs(task_res_dir, exist_ok=True)

    all_env_configs = ENV_CONFIGS_2D if dim == "2d" else ENV_CONFIGS_3D
    env_configs = [c for c in all_env_configs if c["grid_size"] == grid_size_filter] if grid_size_filter else all_env_configs
    fault_regimes = FAULT_REGIMES_2D if dim == "2d" else FAULT_REGIMES_3D
    resolve_mask = resolve_mask_2d if dim == "2d" else resolve_mask_3d

    print(f"\n{'='*80}")
    print(f"  RUNNING SYMMETRIC MAPPO BENCHMARK: {dim.upper()} {task.upper()} ({n_seeds} SEEDS)")
    print(f"  Target: {task_res_dir}")
    print(f"{'='*80}\n")

    mappo_policies = load_symmetric_mappo_policies(dim, task)
    total_conditions = len(env_configs) * len(fault_regimes)
    cond_idx = 0

    for env_cfg in env_configs:
        obs_mask = resolve_mask(env_cfg["grid_size"]) if env_cfg["has_obstacles"] else None

        for fault_cfg in fault_regimes:
            cond_idx += 1
            out_dir = os.path.join(task_res_dir, env_cfg["tag"], fault_cfg["tag"])
            os.makedirs(out_dir, exist_ok=True)

            raw_file_name = f"{task}_{dim}_raw_seed_by_seed.csv"
            raw_path = os.path.join(out_dir, raw_file_name)
            summary_path = os.path.join(out_dir, f"{task}_{dim}_summary.csv")

            print(f"[{cond_idx:02d}/{total_conditions:02d}] {dim.upper()} {env_cfg['label']} | {fault_cfg['label']}")

            if os.path.exists(raw_path):
                df_raw_orig = pd.read_csv(raw_path)
            else:
                df_raw_orig = pd.DataFrame()

            new_mappo_records = []

            for model_name, pol_fn, is_self_heal, ckpt in mappo_policies:
                t0 = time.time()
                print(f"    [▶] Evaluating {model_name} (symmetric sampling)...", end="", flush=True)

                if dim == "2d":
                    if task == "coverage":
                        _, records = evaluate_coverage_2d_episode(
                            pol_fn, env_cfg, fault_cfg["fault_prob"], is_self_heal, n_seeds, obs_mask
                        )
                    else:
                        _, records = evaluate_tracking_2d_episode(
                            pol_fn, env_cfg, fault_cfg["fault_prob"], is_self_heal, n_seeds, obs_mask
                        )
                else:  # 3d
                    if task == "coverage":
                        _, records = evaluate_coverage_3d_episode(
                            pol_fn, env_cfg, fault_cfg["fault_prob"], is_self_heal, n_seeds, obs_mask
                        )
                    else:
                        eval_pol_fn = (
                            make_resizing_policy(pol_fn, source_grid_size=env_cfg["grid_size"], target_grid_size=25)
                            if env_cfg["grid_size"] != 25 else pol_fn
                        )
                        _, records = evaluate_tracking_3d_episode(
                            eval_pol_fn, env_cfg, fault_cfg["fault_prob"], is_self_heal, n_seeds, obs_mask
                        )

                for r in records:
                    r_copy = dict(r)
                    r_copy["Model"] = model_name
                    new_mappo_records.append(r_copy)

                el = time.time() - t0
                if task == "coverage":
                    m_cov = np.mean([r["coverage_pct"] for r in records])
                    print(f" Done ({el:.1f}s) -> Mean Cov: {m_cov:.2f}%")
                else:
                    m_tf = np.mean([r["target_found"] for r in records]) * 100.0
                    print(f" Done ({el:.1f}s) -> Intercept: {m_tf:.1f}%")

            # Merge records: remove old MAPPO records, append new ones
            df_new_mappo = pd.DataFrame(new_mappo_records)
            if not df_raw_orig.empty and "Model" in df_raw_orig.columns:
                df_other_models = df_raw_orig[~df_raw_orig["Model"].isin(["MAPPO Vanilla", "MAPPO SelfHeal"])]
                df_merged_raw = pd.concat([df_other_models, df_new_mappo], ignore_index=True)
            else:
                df_merged_raw = df_new_mappo

            # Save updated raw CSV
            df_merged_raw.to_csv(raw_path, index=False)

            # Recompute summary CSV
            df_summary = recompute_summary_from_raw(df_merged_raw, task, dim, env_cfg)
            df_summary.to_csv(summary_path, index=False)

            # Regenerate Dashboard PNG
            dash_title = f"DSSE-{dim.upper()} {task.title()} (Symmetric Sampling) — {env_cfg['label']} ({fault_cfg['label']})"
            if dim == "2d":
                if task == "coverage":
                    generate_coverage_2d_dashboard(df_summary, out_dir, dash_title)
                else:
                    generate_tracking_2d_dashboard(df_summary, out_dir, dash_title)
            else:
                if task == "coverage":
                    generate_coverage_3d_dashboard(df_summary, out_dir, dash_title)
                else:
                    generate_tracking_3d_dashboard(df_summary, out_dir, dash_title)

    print(f"\n[✓] Completed {dim.upper()} {task.upper()} symmetric evaluations!\n")


def generate_curves_for(dim):
    base_dir = os.path.expanduser(f"~/Desktop/Results/{dim.upper()}-mappo")
    print(f"\nRegenerating publication degradation curves for {dim.upper()}-mappo ({base_dir})...")
    if dim == "2d":
        cmd = f"python3 {SRC_DIR}/generate_2d_degradation_curves.py --base_dir {base_dir}"
    else:
        cmd = f"python3 {SRC_DIR}/dsse_3d/generate_degradation_curves.py --base_dir {base_dir}"
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Symmetric MAPPO Benchmark Runner")
    parser.add_argument("--dim", type=str, default="all", choices=["2d", "3d", "all"])
    parser.add_argument("--task", type=str, default="all", choices=["coverage", "tracking", "all"])
    parser.add_argument("--grid_size", type=int, default=None, choices=[25, 50],
                        help="Filter evaluations by grid size (25 or 50)")
    parser.add_argument("--n_seeds", type=int, default=100)
    args = parser.parse_args()

    dims = ["2d", "3d"] if args.dim == "all" else [args.dim]
    tasks = ["coverage", "tracking"] if args.task == "all" else [args.task]

    t_start = time.time()
    for d in dims:
        for t in tasks:
            run_symmetric_benchmark(dim=d, task=t, n_seeds=args.n_seeds, grid_size_filter=args.grid_size)
        generate_curves_for(d)

    total_min = (time.time() - t_start) / 60.0
    print(f"\n{'='*80}")
    print(f"  ALL SYMMETRIC BENCHMARKS COMPLETE IN {total_min:.1f} MINUTES!")
    print(f"  Results saved to: ~/Desktop/Results/2D-mappo/ and ~/Desktop/Results/3D-mappo/")
    print(f"  Untouched Originals: ~/Desktop/Results/2D/ and ~/Desktop/Results/3D/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
