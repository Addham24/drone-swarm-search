#!/usr/bin/env python3
"""
Verify integrity assertions and generate comparison tables between v2 and v3
for DSSE 2D and 3D obstacle scenarios.
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.expanduser("~/Desktop/Results")
DIR_2D_V2 = os.path.join(RESULTS_DIR, "2D-final", "Coverage")
DIR_2D_V3 = os.path.join(RESULTS_DIR, "2D-final-v3", "Coverage")
DIR_3D_V2 = os.path.join(RESULTS_DIR, "3D-final", "Coverage")
DIR_3D_V3 = os.path.join(RESULTS_DIR, "3D-final-v3", "Coverage")

ENV_CONFIGS = [
    {"tag": "25x25_Standard", "label": "25×25 Standard", "grid_size": 25, "has_obstacles": False, "free_cells": 625},
    {"tag": "25x25_Obstacles", "label": "25×25 Obstacles", "grid_size": 25, "has_obstacles": True, "free_cells": 529},
    {"tag": "50x50_Standard", "label": "50×50 Standard", "grid_size": 50, "has_obstacles": False, "free_cells": 2500},
    {"tag": "50x50_Obstacles", "label": "50×50 Obstacles", "grid_size": 50, "has_obstacles": True, "free_cells": 2041},
]

FAULT_REGIMES = [
    {"tag": "No_Faults_0.0", "label": "No Faults (p=0.0)", "fault_prob": 0.0},
    {"tag": "Fault_0.00025", "label": "Low Faults (p=0.00025)", "fault_prob": 0.00025},
    {"tag": "Active_Faults_0.0005", "label": "Standard Faults (p=0.0005)", "fault_prob": 0.0005},
    {"tag": "Fault_0.00075", "label": "High Faults (p=0.00075)", "fault_prob": 0.00075},
    {"tag": "Fault_0.001", "label": "Severe Faults (p=0.0010)", "fault_prob": 0.001},
]

MODELS = [
    "RSPO Vanilla", "RSPO SelfHeal",
    "MAPPO Vanilla", "MAPPO SelfHeal",
    "I-DQN Vanilla", "I-DQN SelfHeal",
    "QMIX Vanilla", "QMIX SelfHeal",
    "COMA Vanilla", "COMA SelfHeal",
    "MAA2C Vanilla", "MAA2C SelfHeal"
]

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_standard_hashes():
    print("\n" + "="*80)
    print("CHECKING SHA-256 HASHES FOR UNCHANGED STANDARD MAPS (v2 vs v3)")
    print("="*80)
    all_matched = True
    hashes_list = []
    for dim, d_v2, d_v3 in [("2D", DIR_2D_V2, DIR_2D_V3), ("3D", DIR_3D_V2, DIR_3D_V3)]:
        for map_tag in ["25x25_Standard", "50x50_Standard"]:
            for fault in FAULT_REGIMES:
                f_tag = fault["tag"]
                f_name = f"coverage_{dim.lower()}_raw_seed_by_seed.csv"
                path_v2 = os.path.join(d_v2, map_tag, f_tag, f_name)
                path_v3 = os.path.join(d_v3, map_tag, f_tag, f_name)
                if not os.path.exists(path_v2) or not os.path.exists(path_v3):
                    print(f"[-] Missing file: {path_v2} or {path_v3}")
                    all_matched = False
                    continue
                h_v2 = sha256_file(path_v2)
                h_v3 = sha256_file(path_v3)
                match = (h_v2 == h_v3)
                if not match:
                    all_matched = False
                hashes_list.append({
                    "dim": dim, "scenario": f"{map_tag}/{f_tag}",
                    "hash_v2": h_v2, "hash_v3": h_v3, "match": match
                })
                print(f"[{'PASS' if match else 'FAIL'}] {dim} {map_tag}/{f_tag}: {h_v3[:16]}... (match: {match})")
    print(f"\nOverall Standard Maps SHA-256 Match: {'ALL PASS' if all_matched else 'SOME FAILED'}")
    return all_matched, hashes_list

def verify_obstacle_integrity():
    print("\n" + "="*80)
    print("VERIFYING INTEGRITY ASSERTIONS ON OBSTACLE SCENARIOS (v3)")
    print("="*80)
    results = []
    for dim, d_v3 in [("2D", DIR_2D_V3), ("3D", DIR_3D_V3)]:
        for map_cfg in [c for c in ENV_CONFIGS if c["has_obstacles"]]:
            map_tag = map_cfg["tag"]
            n_free = map_cfg["free_cells"]
            for fault in FAULT_REGIMES:
                f_tag = fault["tag"]
                scenario_label = f"{dim} {map_tag} {fault['label']}"
                raw_file = os.path.join(d_v3, map_tag, f_tag, f"coverage_{dim.lower()}_raw_seed_by_seed.csv")
                if not os.path.exists(raw_file):
                    results.append({"scenario": scenario_label, "status": "PENDING / NOT FOUND"})
                    print(f"[-] {scenario_label}: File not found yet")
                    continue
                df = pd.read_csv(raw_file)
                # Assertion 1: exactly 1200 rows, no duplicate (Model, seed)
                n_rows = len(df)
                dup_count = df.duplicated(subset=["Model", "seed"]).sum()
                row_check = (n_rows == 1200 and dup_count == 0)

                # Assertion 2: coverage_pct * N_free / 100 is integer within 1e-4
                # Note: column may be named 'coverage_rate' or 'coverage_pct'
                cov_col = "coverage_pct" if "coverage_pct" in df.columns else "coverage_rate"
                cov_vals = df[cov_col].values
                cells_searched = cov_vals * n_free / 100.0
                int_diff = np.abs(cells_searched - np.round(cells_searched))
                max_int_diff = np.max(int_diff)
                int_check = (max_int_diff < 1e-4)

                # Assertion 3: smei column formula check
                # smei = 100 * max(C, 0.001)^0.4 * max(S, 0.001)^0.3 * max(B, 0.001)^0.3
                surv_col = "survival_rate" if "survival_rate" in df.columns else "survived"
                batt_col = "avg_battery" if "avg_battery" in df.columns else "fleet_battery"
                if batt_col not in df.columns and "battery" in df.columns:
                    batt_col = "battery"

                c_frac = np.clip(cov_vals / 100.0, 0.001, 1.0)
                s_frac = np.clip(df[surv_col].values / 100.0, 0.001, 1.0)
                b_frac = np.clip(df[batt_col].values / 100.0, 0.001, 1.0)
                expected_smei = 100.0 * (c_frac ** 0.4) * (s_frac ** 0.3) * (b_frac ** 0.3)
                actual_smei = df["smei"].values
                max_smei_diff = np.max(np.abs(expected_smei - actual_smei))
                smei_check = (max_smei_diff < 1e-9)

                passed = row_check and int_check and smei_check
                results.append({
                    "dim": dim, "scenario": f"{map_tag}/{f_tag}",
                    "label": scenario_label,
                    "n_rows": n_rows, "dup_count": dup_count,
                    "max_int_diff": max_int_diff,
                    "max_smei_diff": max_smei_diff,
                    "passed": passed
                })
                print(f"[{'PASS' if passed else 'FAIL'}] {scenario_label}: rows={n_rows}, dups={dup_count}, max_int_err={max_int_diff:.2e}, max_smei_err={max_smei_diff:.2e}")
    return results

if __name__ == "__main__":
    check_standard_hashes()
    verify_obstacle_integrity()
