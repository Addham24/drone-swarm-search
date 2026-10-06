"""
Master Evaluation Runner for Dynamic Target Tracking
=====================================================
Executes all 4 tracking benchmark evaluation scripts sequentially:
  1. evaluate_tracking_models.py           (25x25 Standard)
  2. evaluate_tracking_obstacles_25x25.py (25x25 Obstacles)
  3. evaluate_tracking_50x50.py           (50x50 Standard Zero-Shot)
  4. evaluate_tracking_obstacles_50x50.py (50x50 Obstacles Zero-Shot)

Updates all output CSVs and dashboard PNGs across No_Faults_0.0 and Active_Faults_0.0005.
"""

import os
import sys
import subprocess

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

eval_scripts = [
    ("25x25 Standard", os.path.join(SRC_DIR, "tracking", "evaluate_tracking_models.py")),
    ("25x25 Obstacles", os.path.join(SRC_DIR, "tracking", "evaluate_tracking_obstacles_25x25.py")),
    ("50x50 Standard", os.path.join(SRC_DIR, "tracking", "evaluate_tracking_50x50.py")),
    ("50x50 Obstacles", os.path.join(SRC_DIR, "tracking", "evaluate_tracking_obstacles_50x50.py")),
]

def run_all():
    for name, script_path in eval_scripts:
        print("\n" + "=" * 80)
        print(f"  RUNNING EVALUATION SUITE: {name}")
        print("=" * 80 + "\n")
        res = subprocess.run([sys.executable, script_path])
        if res.returncode != 0:
            print(f"[!] Warning: Evaluation script for {name} returned exit code {res.returncode}")

if __name__ == "__main__":
    run_all()
