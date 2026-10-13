#!/usr/bin/env python3
"""
Peak-Baseline Benchmark Runner: 2D-mappo & 3D-mappo
===================================================
Evaluates the 8 baseline models (QMIX, MAA2C, COMA, I-DQN x Vanilla/SelfHeal)
using their newly patched PEAK post-10M checkpoints.

Strict Constraints:
  - RSPO & MAPPO results in ~/Desktop/Results/2D-mappo/ and ~/Desktop/Results/3D-mappo/
    are strictly PRESERVED and UNCHANGED (maintaining identical symmetric sampling temperatures).
  - Original ~/Desktop/Results/2D/ and ~/Desktop/Results/3D/ remain 100% frozen and untouched.
  - Updates raw CSVs, summary CSVs, per-condition dashboards, and regenerates
    publication degradation curves in 2D-mappo/degradation_curves/ and 3D-mappo/degradation_curves/.
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
    load_all_2d_policies,
)
from dsse_3d.run_all_3d_evaluations import (
    ENV_CONFIGS as ENV_CONFIGS_3D,
    FAULT_REGIMES as FAULT_REGIMES_3D,
    _resolve_obstacle_mask as resolve_mask_3d,
    evaluate_coverage_3d_episode,
    evaluate_tracking_3d_episode,
    generate_coverage_3d_dashboard,
    generate_tracking_3d_dashboard,
    load_all_policies as load_all_3d_policies,
)
from dsse_3d.policy_loaders import make_resizing_policy
from run_symmetric_mappo_benchmark import recompute_summary_from_raw

PRESERVED_MODELS = {"RSPO Vanilla", "RSPO SelfHeal", "MAPPO Vanilla", "MAPPO SelfHeal"}


def load_peak_baseline_policies(dim, task):
    """Load only the non-RSPO, non-MAPPO policies using the patched peak checkpoint loader."""
    if dim == "2d":
        all_policies = load_all_2d_policies(task)
        # Format: (model_name, raw_fn, is_self_heal, framework)
        baselines = [p for p in all_policies if p[0] not in PRESERVED_MODELS]
    else:  # 3d
        all_policies = load_all_3d_policies(task)
        # Format: (name, policy_fn, is_self_heal, checkpoint_path)
        baselines = [(p[0], p[1], p[2], p[3]) for p in all_policies if p[0] not in PRESERVED_MODELS]

    print(f"\n[✓] Loaded {len(baselines)} peak baseline policies for {dim.upper()} {task.upper()}:")
    for b in baselines:
        print(f"    - {b[0]} (SelfHeal={b[2]})")
    return baselines


def run_baseline_updates(dim, task, n_seeds=100, grid_size_filter=None):
    base_dir = os.path.expanduser(f"~/Desktop/Results/{dim.upper()}-mappo")
    task_res_dir = os.path.join(base_dir, task.title())

    if dim == "2d":
        env_configs = [c for c in ENV_CONFIGS_2D if c["grid_size"] == grid_size_filter] if grid_size_filter else ENV_CONFIGS_2D
        fault_regimes = FAULT_REGIMES_2D
        resolve_mask = resolve_mask_2d
    else:
        env_configs = [c for c in ENV_CONFIGS_3D if c["grid_size"] == grid_size_filter] if grid_size_filter else ENV_CONFIGS_3D
        fault_regimes = FAULT_REGIMES_3D
        resolve_mask = resolve_mask_3d

    baseline_policies = load_peak_baseline_policies(dim, task)
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

            print(f"\n[{cond_idx:02d}/{total_conditions:02d}] {dim.upper()} {env_cfg['label']} | {fault_cfg['label']}")

            if os.path.exists(raw_path):
                df_raw_orig = pd.read_csv(raw_path)
            else:
                raise FileNotFoundError(f"Original raw CSV not found at {raw_path}")

            # Strictly preserve RSPO and MAPPO rows
            df_preserved = df_raw_orig[df_raw_orig["Model"].isin(PRESERVED_MODELS)].copy()
            print(f"    [Preserved] {len(df_preserved)} rows from RSPO & MAPPO (symmetric sampling intact)")

            new_baseline_records = []

            for item in baseline_policies:
                model_name = item[0]
                pol_fn = item[1]
                is_self_heal = item[2]

                t0 = time.time()
                print(f"    [▶] Evaluating {model_name} (peak checkpoint)...", end="", flush=True)

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
                    new_baseline_records.append(r_copy)

                el = time.time() - t0
                if task == "coverage":
                    m_cov = np.mean([r["coverage_pct"] for r in records])
                    print(f" Done ({el:.1f}s) -> Mean Cov: {m_cov:.2f}%")
                else:
                    m_tf = np.mean([r["target_found"] for r in records]) * 100.0
                    print(f" Done ({el:.1f}s) -> Intercept: {m_tf:.1f}%")

            # Combine preserved RSPO/MAPPO with updated baselines
            df_new_baselines = pd.DataFrame(new_baseline_records)
            df_merged_raw = pd.concat([df_preserved, df_new_baselines], ignore_index=True)

            # Save updated raw CSV
            df_merged_raw.to_csv(raw_path, index=False)

            # Recompute summary CSV
            df_summary = recompute_summary_from_raw(df_merged_raw, task, dim, env_cfg)
            df_summary.to_csv(summary_path, index=False)

            # Regenerate Dashboard PNG
            dash_title = f"DSSE-{dim.upper()} {task.title()} (Peak Baselines & Symmetric MAPPO) — {env_cfg['label']} ({fault_cfg['label']})"
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

    print(f"\n[✓] Finished baseline updates for {dim.upper()} {task.upper()}!\n")


def generate_curves_for(dim):
    base_dir = os.path.expanduser(f"~/Desktop/Results/{dim.upper()}-mappo")
    print(f"\nRegenerating publication degradation curves for {dim.upper()}-mappo ({base_dir})...")
    if dim == "2d":
        cmd = f"python3 {SRC_DIR}/generate_2d_degradation_curves.py --base_dir {base_dir}"
    else:
        cmd = f"python3 {SRC_DIR}/dsse_3d/generate_degradation_curves.py --base_dir {base_dir}"
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Peak-Baseline Benchmark Runner")
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
            run_baseline_updates(dim=d, task=t, n_seeds=args.n_seeds, grid_size_filter=args.grid_size)
        generate_curves_for(d)

    total_min = (time.time() - t_start) / 60.0
    print(f"\n{'='*80}")
    print(f"  PEAK BASELINES BENCHMARK COMPLETE IN {total_min:.1f} MINUTES!")
    print(f"  Results saved to: ~/Desktop/Results/2D-mappo/ and ~/Desktop/Results/3D-mappo/")
    print(f"  Untouched Originals: ~/Desktop/Results/2D/ and ~/Desktop/Results/3D/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
