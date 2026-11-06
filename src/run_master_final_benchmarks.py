#!/usr/bin/env python3
"""
Master Final Benchmark Runner: 2D-final & 3D-final
===================================================
Executes the comprehensive, publication-grade coverage benchmark across:
  - 12 MARL Models (Standardized peak training coverage checkpoints across 2D and 3D)
  - 4 Environments (25x25 Standard, 25x25 Obstacles, 50x50 Standard, 50x50 Obstacles)
  - 5 Fault Regimes (No Faults 0.0, Low 0.00025, Standard 0.0005, High 0.00075, Severe 0.0010)
  - 100 Seeds per condition (1000 - 1099)
  - Active Urban Warfare Obstacle Maps (529 free cells on 25x25, 2041 free cells on 50x50)

Outputs Target:
  - ~/Desktop/Results/2D-final/
  - ~/Desktop/Results/3D-final/
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd
import torch as th
from concurrent.futures import ProcessPoolExecutor

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Model Registry with locked peak-performance checkpoints
MODEL_SPECS = [
    ("RSPO Vanilla",   "rllib",     "RSPOModelV2",       os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_Vanilla_RSPO_v2/RSPOPPO_DSSE_Coverage_RSPO_V2_Vanilla_45eb7_00000_0_2026-09-04_06-38-43/checkpoint_000136"), False),
    ("RSPO SelfHeal",  "rllib",     "RSPOModelV2",       os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_SelfHeal_rspo_v2_selfheal/RSPOPPO_DSSE_Coverage_RSPO_V2_SelfHeal_86fd5_00000_0_2026-09-04_06-40-32/checkpoint_000122"), True),
    ("MAPPO Vanilla",  "rllib",     "MAPPOModelVanilla", os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_vanilla_proper/PPO_DSSE_Coverage_cc7e1_00000_0_2026-07-05_03-37-08/checkpoint_000122"), False),
    ("MAPPO SelfHeal", "rllib",     "MAPPOModelVanilla", os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_selfheal_final_v2/checkpoints/checkpoint_final_ts_20004864.pt"), True),
    ("QMIX Vanilla",   "epymarl",   None,                os.path.join(SRC_DIR, "results/models/qmix_true_vanilla_v1_seed533532401_dsse_coverage_2026-07-18 02:06:54.683275/17040000/agent.th"), False),
    ("QMIX SelfHeal",  "epymarl",   None,                os.path.join(SRC_DIR, "results/models/qmix_selfheal_v2_seed533532401_dsse_coverage_2026-07-17 03:23:47.879210/19044000/agent.th"), True),
    ("MAA2C Vanilla",  "epymarl",   None,                os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_vanilla_seed782869923_dsse_coverage_2026-06-16 18:45:27.715438/10035000/agent.th"), False),
    ("MAA2C SelfHeal", "epymarl",   None,                os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_original_seed776978465_dsse_coverage_2026-07-15 06:30:31.971452/12675000/agent.th"), True),
    ("COMA Vanilla",   "epymarl",   None,                os.path.join(SRC_DIR, "results/models/coma_vanilla_proper_seed452926189_dsse_coverage_2026-07-10 01:59:53.354626/19338000/agent.th"), False),
    ("COMA SelfHeal",  "epymarl",   None,                os.path.join(SRC_DIR, "results/models/coma_selfhealing_final_stable_seed339532096_dsse_coverage_2026-07-06 10:52:47.933607/16692000/agent.th"), True),
    ("I-DQN Vanilla",   "rllib_dqn", None,                os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_vanilla_I-DQN_vanilla_proper/DQN_DSSE_Coverage_61018_00000_0_2026-07-16_09-47-37/checkpoint_000609"), False),
    ("I-DQN SelfHeal",  "rllib_dqn", None,                os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_I-DQN_selfheal_proper_pt2/DQN_DSSE_Coverage_8aefa_00000_0_2026-07-16_05-38-15/checkpoint_000587"), True),
]

ENV_CONFIGS = [
    {
        "tag": "25x25_Standard",
        "grid_size": 25,
        "timestep_limit": 750,
        "max_battery": 125,
        "has_obstacles": False,
        "free_cells": 625,
        "label": "25×25 Standard",
    },
    {
        "tag": "25x25_Obstacles",
        "grid_size": 25,
        "timestep_limit": 750,
        "max_battery": 125,
        "has_obstacles": True,
        "free_cells": 529,
        "label": "25×25 Obstacles",
    },
    {
        "tag": "50x50_Standard",
        "grid_size": 50,
        "timestep_limit": 1500,
        "max_battery": 250,
        "has_obstacles": False,
        "free_cells": 2500,
        "label": "50×50 Standard",
    },
    {
        "tag": "50x50_Obstacles",
        "grid_size": 50,
        "timestep_limit": 1500,
        "max_battery": 250,
        "has_obstacles": True,
        "free_cells": 2041,
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


def resolve_obstacle_mask(grid_size):
    path = os.path.join(SRC_DIR, f"obstacle_skyline_{grid_size}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None


def worker_eval_2d(args):
    model_name, fw, cls_name, path, is_self_heal, env_cfg, fault_prob, n_seeds, obs_mask = args
    import sys, os, torch as th, numpy as np
    SRC_DIR = '/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src'
    sys.path.insert(0, SRC_DIR)
    th.set_num_threads(1)

    from dsse_3d.policy_loaders import load_rllib_ppo_policy, load_rllib_dqn_policy, load_epymarl_policy_strict
    from train_rspo_v2_vanilla import RSPOModelV2
    from train_mappo_vanilla import CNNModel as MAPPOModelVanilla
    from run_all_2d_evaluations import evaluate_coverage_2d_episode

    cls_map = {'RSPOModelV2': RSPOModelV2, 'MAPPOModelVanilla': MAPPOModelVanilla}
    cls = cls_map.get(cls_name)

    if fw == 'rllib':
        policy_fn = load_rllib_ppo_policy(cls, path, model_name, greedy=True, sample_mappo=True)
    elif fw == 'rllib_dqn':
        policy_fn = load_rllib_dqn_policy(path)
    elif fw == 'epymarl':
        policy_fn = load_epymarl_policy_strict(path)

    summary, records = evaluate_coverage_2d_episode(policy_fn, env_cfg, fault_prob, is_self_heal, n_seeds, obs_mask)
    summary['Model'] = model_name
    for r in records:
        r['Model'] = model_name
    return model_name, summary, records


def worker_eval_3d(args):
    model_name, fw, cls_name, path, is_self_heal, env_cfg, fault_prob, n_seeds, obs_mask = args
    import sys, os, torch as th, numpy as np
    SRC_DIR = '/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src'
    sys.path.insert(0, SRC_DIR)
    th.set_num_threads(1)

    from dsse_3d.policy_loaders import load_rllib_ppo_policy, load_rllib_dqn_policy, load_epymarl_policy_strict
    from train_rspo_v2_vanilla import RSPOModelV2
    from train_mappo_vanilla import CNNModel as MAPPOModelVanilla
    from dsse_3d.run_all_3d_evaluations import evaluate_coverage_3d_episode

    cls_map = {'RSPOModelV2': RSPOModelV2, 'MAPPOModelVanilla': MAPPOModelVanilla}
    cls = cls_map.get(cls_name)

    if fw == 'rllib':
        policy_fn = load_rllib_ppo_policy(cls, path, model_name, greedy=True, sample_mappo=True)
    elif fw == 'rllib_dqn':
        policy_fn = load_rllib_dqn_policy(path)
    elif fw == 'epymarl':
        policy_fn = load_epymarl_policy_strict(path)

    summary, records = evaluate_coverage_3d_episode(policy_fn, env_cfg, fault_prob, is_self_heal, n_seeds, obs_mask)
    summary['Model'] = model_name
    for r in records:
        r['Model'] = model_name
    return model_name, summary, records


def run_benchmark_suite(dim="2d", n_seeds=100, max_workers=6, force=False):
    is_3d = (dim.lower() == "3d")
    worker_fn = worker_eval_3d if is_3d else worker_eval_2d
    
    from run_all_2d_evaluations import generate_coverage_2d_dashboard
    from dsse_3d.run_all_3d_evaluations import generate_coverage_3d_dashboard
    gen_dash = generate_coverage_3d_dashboard if is_3d else generate_coverage_2d_dashboard

    base_root = os.path.expanduser(f"~/Desktop/Results/{dim.upper()}-final")
    base_out = os.path.join(base_root, "Coverage")
    os.makedirs(base_out, exist_ok=True)

    manifest = {
        "task": "coverage",
        "dimension": dim.upper(),
        "git_commit_hash": "decd8ff8e50e2c8b292d8d774797341a3f8af048",
        "git_adapter_fix_commit": "c52c997adf074fcbea2bae50d34e1ba363ee0f16",
        "run_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_seeds": n_seeds,
        "seeds": [1000, 1000 + n_seeds - 1],
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
        "re_evaluated_scenarios": f"All 20 {dim.upper()} scenarios (all 12 models)",
        "env_configs": ENV_CONFIGS,
        "fault_regimes": FAULT_REGIMES,
    }
    with open(os.path.join(base_out, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(base_root, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    log_path = os.path.join(base_root, "evaluation_log.txt")
    log_f = open(log_path, "w" if force else "a", buffering=1)

    def log(msg):
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    total_conditions = len(ENV_CONFIGS) * len(FAULT_REGIMES)
    cond_idx = 0
    t_suite_start = time.time()

    log(f"\n{'='*80}")
    log(f"  LAUNCHING {dim.upper()} COVERAGE MASTER BENCHMARK ({total_conditions} CONDITIONS)")
    log(f"  Target Directory: {base_out}")
    log(f"  Workers: {max_workers} Parallel Processes")
    log(f"  Force Re-run: {force}")
    log(f"{'='*80}\n")

    for env_cfg in ENV_CONFIGS:
        obs_mask = resolve_obstacle_mask(env_cfg["grid_size"]) if env_cfg["has_obstacles"] else None

        for fault_cfg in FAULT_REGIMES:
            cond_idx += 1
            cond_start = time.time()
            out_dir = os.path.join(base_out, env_cfg["tag"], fault_cfg["tag"])
            os.makedirs(out_dir, exist_ok=True)

            raw_path = os.path.join(out_dir, f"coverage_{dim.lower()}_raw_seed_by_seed.csv")
            summary_path = os.path.join(out_dir, f"coverage_{dim.lower()}_summary.csv")
            dash_path = os.path.join(out_dir, f"dashboard_coverage_{dim.lower()}.png")
            if not force and os.path.exists(raw_path) and os.path.exists(summary_path) and os.path.exists(dash_path):
                log(f"[{cond_idx:02d}/{total_conditions:02d}] {dim.upper()} {env_cfg['label']} | {fault_cfg['label']} [ALREADY EVALUATED - SKIPPING]")
                continue

            log(f"[{cond_idx:02d}/{total_conditions:02d}] {dim.upper()} {env_cfg['label']} | {fault_cfg['label']} ...")

            tasks = []
            for name, fw, cls_name, path, is_self_heal in MODEL_SPECS:
                tasks.append((name, fw, cls_name, path, is_self_heal, env_cfg, fault_cfg["fault_prob"], n_seeds, obs_mask))

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(worker_fn, tasks))

            results_dict = {name: (s, r) for name, s, r in results}
            # Maintain exact model order
            summary_list = []
            all_raw = []
            for name, _, _, _, _ in MODEL_SPECS:
                s, recs = results_dict[name]
                summary_list.append(s)
                all_raw.extend(recs)

            df_summary = pd.DataFrame(summary_list)
            df_raw = pd.DataFrame(all_raw)

            # Save CSVs
            df_raw.to_csv(raw_path, index=False)
            df_summary.to_csv(summary_path, index=False)

            # Generate Dashboard PNG
            dash_title = f"DSSE-{dim.upper()} Coverage (Final Synchronized) — {env_cfg['label']} ({fault_cfg['label']})"
            gen_dash(df_summary, out_dir, dash_title)

            el_cond = time.time() - cond_start
            cov_col = "coverage_rate" if "coverage_rate" in df_summary.columns else "Coverage Rate (%)"
            rspo_cov = df_summary[df_summary["Model"] == "RSPO SelfHeal"][cov_col].values[0]
            mappo_cov = df_summary[df_summary["Model"] == "MAPPO SelfHeal"][cov_col].values[0]
            log(f"    [✓] Done in {el_cond:.1f}s | RSPO SH: {rspo_cov:.1f}% | MAPPO SH: {mappo_cov:.1f}%")

    total_el = (time.time() - t_suite_start) / 60.0
    log(f"\n{'='*80}")
    log(f"  {dim.upper()} MASTER BENCHMARK COMPLETE IN {total_el:.1f} MINUTES!")
    log(f"  All 20 Conditions & Dashboards Saved to: {base_out}")
    log(f"{'='*80}\n")
    log_f.close()


def run_post_processing():
    import subprocess
    import glob
    import zipfile
    import shutil
    print(f"\n{'='*80}")
    print("  RUNNING POST-PROCESSING: DEGRADATION CURVES, RQ FIGURES & ZIP ARCHIVE")
    print(f"{'='*80}\n")

    b2d = os.path.expanduser("~/Desktop/Results/2D-final")
    b3d = os.path.expanduser("~/Desktop/Results/3D-final")

    # 1. Copy run_manifest.json to root of 2D-final and 3D-final
    for b in [b2d, b3d]:
        m_cov = os.path.join(b, "Coverage", "run_manifest.json")
        m_root = os.path.join(b, "run_manifest.json")
        if os.path.exists(m_cov) and not os.path.exists(m_root):
            shutil.copy2(m_cov, m_root)

    # 2. Degradation curves
    if os.path.exists(os.path.join(b2d, "Coverage")):
        print("  [▶] Generating 2D publication degradation curves...")
        subprocess.run([sys.executable, os.path.join(SRC_DIR, "generate_2d_degradation_curves.py"), "--base_dir", b2d], check=False)

    if os.path.exists(os.path.join(b3d, "Coverage")):
        print("  [▶] Generating 3D publication degradation curves...")
        subprocess.run([sys.executable, os.path.join(SRC_DIR, "dsse_3d/generate_degradation_curves.py"), "--base_dir", b3d], check=False)

    # 3. Master RQ figures
    print("  [▶] Generating All Research Question (RQ1, RQ3, RQ4) Visualizations...")
    subprocess.run([sys.executable, os.path.join(SRC_DIR, "generate_all_final_visualizations.py")], check=False)

    # 4. Build master ZIP archive for the user (v2)
    zip_path = os.path.expanduser("~/Desktop/Results/final_evaluation_results_v2.zip")
    print(f"  [▶] Building master ZIP archive: {zip_path} ...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root_dir in [b2d, b3d]:
            if not os.path.exists(root_dir):
                continue
            base_folder = os.path.basename(root_dir)
            for dirpath, _, filenames in os.walk(root_dir):
                for f in filenames:
                    if f.endswith((".csv", ".json", ".txt", ".png")):
                        full_f = os.path.join(dirpath, f)
                        rel_path = os.path.relpath(full_f, os.path.dirname(root_dir))
                        zipf.write(full_f, rel_path)
    print(f"  [✓] Master ZIP archive created successfully at: {zip_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=str, default="both", choices=["2d", "3d", "both"])
    parser.add_argument("--n_seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Force overwrite existing benchmarks")
    args = parser.parse_args()

    dims_to_run = ["2d", "3d"] if args.dim == "both" else [args.dim]
    for d in dims_to_run:
        run_benchmark_suite(dim=d, n_seeds=args.n_seeds, max_workers=args.workers, force=args.force)

    run_post_processing()
