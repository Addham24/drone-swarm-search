"""
Master Runner Script for Dynamic Target Tracking Benchmark (Paired Execution)
===============================================================================
Executes training for all 12 MARL models on the tracking environment in 6 PAIRS:

  Pair 1: RSPO V2 Vanilla  +  RSPO V2 SelfHeal    (Parallel)
  Pair 2: MAPPO Vanilla    +  MAPPO SelfHeal      (Parallel)
  Pair 3: QMIX Vanilla     +  QMIX SelfHeal       (Parallel)
  Pair 4: MAA2C Vanilla    +  MAA2C SelfHeal      (Parallel)
  Pair 5: COMA Vanilla     +  COMA SelfHeal       (Parallel)
  Pair 6: IQL Vanilla      +  IQL SelfHeal        (Parallel)

Both jobs in a pair run simultaneously. The runner waits for BOTH to finish
before moving to the next algorithm pair!

Usage on RunPod:
    python src/tracking/run_all_tracking_experiments.py
"""

import os
import sys
import time
import subprocess
import argparse

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PAIRS = [
    {
        "algorithm": "RSPO V2",
        "vanilla": {
            "name": "RSPO_V2_Tracking_Vanilla",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "tracking", "train_rspo_v2_tracking_vanilla.py")]
        },
        "selfheal": {
            "name": "RSPO_V2_Tracking_SelfHeal",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "tracking", "train_rspo_v2_tracking_selfheal.py")]
        }
    },
    {
        "algorithm": "MAPPO",
        "vanilla": {
            "name": "MAPPO_Tracking_Vanilla",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "tracking", "train_mappo_tracking_vanilla.py")]
        },
        "selfheal": {
            "name": "MAPPO_Tracking_SelfHeal",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "tracking", "train_mappo_tracking_selfheal.py")]
        }
    },
    {
        "algorithm": "QMIX",
        "vanilla": {
            "name": "QMIX_Tracking_Vanilla",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "qmix", "--env_type", "tracking", "--name", "qmix_tracking_vanilla", "--disable_self_healing"]
        },
        "selfheal": {
            "name": "QMIX_Tracking_SelfHeal",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "qmix", "--env_type", "tracking", "--name", "qmix_tracking_selfheal"]
        }
    },
    {
        "algorithm": "MAA2C",
        "vanilla": {
            "name": "MAA2C_Tracking_Vanilla",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "maa2c", "--env_type", "tracking", "--name", "maa2c_tracking_vanilla", "--disable_self_healing"]
        },
        "selfheal": {
            "name": "MAA2C_Tracking_SelfHeal",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "maa2c", "--env_type", "tracking", "--name", "maa2c_tracking_selfheal"]
        }
    },
    {
        "algorithm": "COMA",
        "vanilla": {
            "name": "COMA_Tracking_Vanilla",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "coma", "--env_type", "tracking", "--name", "coma_tracking_vanilla", "--disable_self_healing"]
        },
        "selfheal": {
            "name": "COMA_Tracking_SelfHeal",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "coma", "--env_type", "tracking", "--name", "coma_tracking_selfheal"]
        }
    },
    {
        "algorithm": "IQL",
        "vanilla": {
            "name": "IQL_Tracking_Vanilla",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "iql", "--env_type", "tracking", "--name", "iql_tracking_vanilla", "--disable_self_healing"]
        },
        "selfheal": {
            "name": "IQL_Tracking_SelfHeal",
            "cmd": [sys.executable, os.path.join(SRC_DIR, "epymarl_train.py"), "--algo", "iql", "--env_type", "tracking", "--name", "iql_tracking_selfheal"]
        }
    },
]


def run_paired():
    print(f"\n{'='*75}")
    print(f"  LAUNCHING TRACKING BENCHMARK IN ALGORITHM PAIRS (RUNPOD PARALLEL MODE)")
    print(f"{'='*75}\n")

    for i, pair in enumerate(PAIRS, 1):
        algo_name = pair["algorithm"]
        vanilla_job = pair["vanilla"]
        selfheal_job = pair["selfheal"]

        print(f"\n{'─'*75}")
        print(f"  PAIR [{i}/{len(PAIRS)}]: {algo_name.upper()} (Vanilla + SelfHeal)")
        print(f"{'─'*75}")
        print(f"[→] Launching Vanilla : {vanilla_job['name']}")
        print(f"    Command: {' '.join(vanilla_job['cmd'])}\n")
        
        # Popen Vanilla
        p_vanilla = subprocess.Popen(vanilla_job['cmd'], cwd=SRC_DIR)

        # Brief delay to avoid port/resource collision during init
        time.sleep(3)

        print(f"[→] Launching SelfHeal: {selfheal_job['name']}")
        print(f"    Command: {' '.join(selfheal_job['cmd'])}\n")
        
        # Popen SelfHeal
        p_selfheal = subprocess.Popen(selfheal_job['cmd'], cwd=SRC_DIR)

        print(f"\n[⏳] Both {algo_name} jobs are running in parallel. Waiting for PAIR {i} to complete...")

        # Wait for both processes to complete
        code_v = p_vanilla.wait()
        code_s = p_selfheal.wait()

        if code_v == 0:
            print(f"[✓] {vanilla_job['name']} completed successfully!")
        else:
            print(f"[✗] {vanilla_job['name']} failed with exit code: {code_v}")

        if code_s == 0:
            print(f"[✓] {selfheal_job['name']} completed successfully!")
        else:
            print(f"[✗] {selfheal_job['name']} failed with exit code: {code_s}")

        print(f"[✓] PAIR [{i}/{len(PAIRS)}] ({algo_name}) FINISHED. Moving to next pair...\n")

    print(f"\n{'='*75}")
    print(f"  ALL 6 ALGORITHM PAIRS (12 MODELS TOTAL) COMPLETED SUCCESSFULLY!")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paired Tracking Benchmark Runner")
    parser.add_argument("--sequential", action="store_true", help="Run 12 jobs strictly 1-by-1 instead of in parallel pairs")
    args = parser.parse_args()

    if args.sequential:
        print("[!] Running in sequential mode (1 job at a time)...")
        for pair in PAIRS:
            for job in [pair["vanilla"], pair["selfheal"]]:
                print(f"Executing: {job['name']}")
                subprocess.call(job['cmd'], cwd=SRC_DIR)
    else:
        run_paired()
