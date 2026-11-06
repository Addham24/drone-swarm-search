#!/usr/bin/env python3
"""
Update 2D-final Benchmarks for Compliant Checkpoints:
  - I-DQN SelfHeal -> checkpoint_000587 (post-10M peak return at 18.79M steps)
  - MAA2C Vanilla -> 10035000/agent.th (post-10M peak return at 10.07M steps)

This script evaluates only these 2 models across all 20 2D conditions,
merges the new 100 seeds into coverage_2d_raw_seed_by_seed.csv,
recomputes coverage_2d_summary.csv, and regenerates dashboard_coverage_2d.png.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch as th
from concurrent.futures import ProcessPoolExecutor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_master_final_benchmarks import (
    MODEL_SPECS, ENV_CONFIGS, FAULT_REGIMES, resolve_obstacle_mask, worker_eval_2d
)
from run_all_2d_evaluations import generate_coverage_2d_dashboard

# Target models to re-evaluate
TARGET_MODELS = [
    ("MAA2C Vanilla",  "epymarl",   None, os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_vanilla_seed782869923_dsse_coverage_2026-06-16 18:45:27.715438/10035000/agent.th"), False),
    ("I-DQN SelfHeal", "rllib_dqn", None, os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_I-DQN_selfheal_proper_pt2/DQN_DSSE_Coverage_8aefa_00000_0_2026-07-16_05-38-15/checkpoint_000587"), True),
]

# Canonical order of 12 models
CANONICAL_MODEL_ORDER = [name for name, _, _, _, _ in MODEL_SPECS]


def main():
    base_2d = os.path.expanduser("~/Desktop/Results/2D-final/Coverage")
    log_file_path = os.path.expanduser("~/Desktop/Results/2D-final/evaluation_log.txt")
    os.makedirs(base_2d, exist_ok=True)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    log_f = open(log_file_path, "a", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    log(f"\n{'='*80}")
    log(f"  2D BENCHMARK UPDATE: COMPLIANT CHECKPOINTS RE-RUN")
    log(f"  Models: MAA2C Vanilla (10035000/agent.th) & I-DQN SelfHeal (checkpoint_000587)")
    log(f"  Target Directory: {base_2d}")
    log(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*80}\n")

    t_start = time.time()
    cond_idx = 0
    total_conditions = len(ENV_CONFIGS) * len(FAULT_REGIMES)

    for env_cfg in ENV_CONFIGS:
        obs_mask = resolve_obstacle_mask(env_cfg["grid_size"]) if env_cfg["has_obstacles"] else None

        for fault_cfg in FAULT_REGIMES:
            cond_idx += 1
            cond_start = time.time()
            out_dir = os.path.join(base_2d, env_cfg["tag"], fault_cfg["tag"])
            raw_path = os.path.join(out_dir, "coverage_2d_raw_seed_by_seed.csv")
            summary_path = os.path.join(out_dir, "coverage_2d_summary.csv")
            dash_path = os.path.join(out_dir, "dashboard_coverage_2d.png")

            if not os.path.exists(raw_path):
                log(f"[{cond_idx:02d}/{total_conditions:02d}] WARNING: {raw_path} not found! Skipping.")
                continue

            log(f"[{cond_idx:02d}/{total_conditions:02d}] Evaluating {env_cfg['label']} | {fault_cfg['label']} ...")

            tasks = []
            for name, fw, cls_name, path, is_self_heal in TARGET_MODELS:
                tasks.append((name, fw, cls_name, path, is_self_heal, env_cfg, fault_cfg["fault_prob"], 100, obs_mask))

            with ProcessPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(worker_eval_2d, tasks))

            results_dict = {name: (s, r) for name, s, r in results}

            # Read existing raw CSV
            df_old = pd.read_csv(raw_path)
            # Remove existing rows for the target models
            target_names = [m[0] for m in TARGET_MODELS]
            df_kept = df_old[~df_old["Model"].isin(target_names)].copy()

            # Append new records
            new_records = []
            for name in target_names:
                _, recs = results_dict[name]
                new_records.extend(recs)
            df_new = pd.DataFrame(new_records)

            df_merged = pd.concat([df_kept, df_new], ignore_index=True)

            # Re-sort to canonical model order
            df_merged["model_rank"] = df_merged["Model"].apply(lambda m: CANONICAL_MODEL_ORDER.index(m) if m in CANONICAL_MODEL_ORDER else 99)
            df_merged = df_merged.sort_values(by=["model_rank", "seed"]).drop(columns=["model_rank"]).reset_index(drop=True)

            # Assert exactly 1200 rows
            assert len(df_merged) == 1200, f"Expected 1200 rows in {raw_path}, got {len(df_merged)}"
            df_merged.to_csv(raw_path, index=False)

            # Recompute summary dataframe
            summary_rows = []
            for m_name in CANONICAL_MODEL_ORDER:
                m_df = df_merged[df_merged["Model"] == m_name]
                if len(m_df) == 0:
                    continue
                cov_vals = m_df["coverage_pct"].values
                cov_mean = np.mean(cov_vals)
                surv_mean = np.mean(m_df["survival_rate"].values)
                batt_mean = np.mean(m_df["avg_battery"].values)
                smei_mean = np.mean(m_df["smei"].values)
                attr_mean = np.mean(m_df["crash_events"].values)
                rew_mean = np.mean(m_df["mission_return"].values)
                shaped_mean = np.mean(m_df["shaped_return"].values)

                summary_rows.append({
                    "coverage_rate": round(cov_mean, 2),
                    "coverage_std": round(np.std(cov_vals), 2),
                    "coverage_ci95": round(1.96 * np.std(cov_vals) / np.sqrt(len(cov_vals)), 2),
                    "attrition_rate": round(attr_mean, 2),
                    "survival_rate": round(surv_mean, 2),
                    "avg_battery": round(batt_mean, 2),
                    "smei": round(smei_mean, 2),
                    "mean_reward": round(rew_mean, 2),
                    "mean_shaped_return": round(shaped_mean, 2),
                    "n_seeds": len(m_df),
                    "Model": m_name,
                })

            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_csv(summary_path, index=False)

            # Regenerate Dashboard PNG
            dash_title = f"DSSE-2D Coverage (Final Synchronized) — {env_cfg['label']} ({fault_cfg['label']})"
            generate_coverage_2d_dashboard(df_summary, out_dir, dash_title)

            el_cond = time.time() - cond_start
            maa2c_cov = df_summary[df_summary["Model"] == "MAA2C Vanilla"]["coverage_rate"].values[0]
            idqn_cov = df_summary[df_summary["Model"] == "I-DQN SelfHeal"]["coverage_rate"].values[0]
            log(f"    [✓] Done in {el_cond:.1f}s | MAA2C Vanilla: {maa2c_cov:.1f}% | I-DQN SelfHeal: {idqn_cov:.1f}%")

    # Update 2D run_manifest.json
    manifest_paths = [
        os.path.join(base_2d, "run_manifest.json"),
        os.path.expanduser("~/Desktop/Results/2D-final/run_manifest.json"),
    ]
    manifest_data = {
        "task": "coverage",
        "dimension": "2D",
        "git_commit_hash": "decd8ff8e50e2c8b292d8d774797341a3f8af048",
        "git_adapter_fix_commit": "c52c997adf074fcbea2bae50d34e1ba363ee0f16",
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_seeds": 100,
        "seeds": [1000, 1099],
        "free_cells": {
            "25x25_Standard": 625,
            "25x25_Obstacles": 529,
            "50x50_Standard": 2500,
            "50x50_Obstacles": 2041,
        },
        "action_selection": {
            "RSPO": "temperature_sampling (tau=0.20 on 25x25, tau=0.30 on 50x50)",
            "MAPPO": "temperature_sampling (tau=0.20 on 25x25, tau=0.30 on 50x50)",
            "EPyMARL": "greedy_argmax",
            "I-DQN": "greedy_argmax",
        },
        "checkpoints": {name: path for name, _, _, path, _ in MODEL_SPECS},
        "re_evaluated_models": [
            "MAA2C Vanilla (switched to 10035000/agent.th per peak post-10M greedy return)",
            "I-DQN SelfHeal (switched to checkpoint_000587 per peak post-10M episode return)",
        ],
        "re_evaluated_scenarios": "All 20 2D scenarios",
        "env_configs": ENV_CONFIGS,
        "fault_regimes": FAULT_REGIMES,
    }

    for mp in manifest_paths:
        with open(mp, "w") as f:
            json.dump(manifest_data, f, indent=2)
        log(f"  [✓] Updated manifest at {mp}")

    el_total = (time.time() - t_start) / 60.0
    log(f"\n{'='*80}")
    log(f"  2D BENCHMARK UPDATE COMPLETE IN {el_total:.1f} MINUTES!")
    log(f"{'='*80}\n")
    log_f.close()


if __name__ == "__main__":
    main()
