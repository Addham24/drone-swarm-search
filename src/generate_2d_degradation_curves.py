"""
Fault Severity Degradation Curve Generator for DSSE-2D
========================================================
Parses 2D summary CSV files across all 5 fault severity levels:
    - fault_prob = 0.0     (No Faults)
    - fault_prob = 0.00025 (Low Faults)
    - fault_prob = 0.0005  (Standard Faults)
    - fault_prob = 0.00075 (High Faults)
    - fault_prob = 0.0010  (Severe Faults)

Saved to: ~/Desktop/Results/2D/degradation_curves/
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.expanduser("~/Desktop/Results/2D")
OUTPUT_DIR = os.path.join(BASE_DIR, "degradation_curves")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FAULT_LEVELS = [
    ("No_Faults_0.0", 0.0),
    ("Fault_0.00025", 0.00025),
    ("Active_Faults_0.0005", 0.0005),
    ("Fault_0.00075", 0.00075),
    ("Fault_0.001", 0.0010),
]

ENVS = ["25x25_Standard", "25x25_Obstacles", "50x50_Standard", "50x50_Obstacles"]
TASKS = ["Coverage", "Tracking"]

MODEL_ORDER = [
    "RSPO SelfHeal", "RSPO Vanilla",
    "MAPPO SelfHeal", "MAPPO Vanilla",
    "QMIX SelfHeal", "QMIX Vanilla",
    "MAA2C SelfHeal", "MAA2C Vanilla",
    "COMA SelfHeal", "COMA Vanilla",
    "I-DQN SelfHeal", "I-DQN Vanilla",
]

COLOR_MAP = {
    "RSPO SelfHeal": "#1f77b4",
    "RSPO Vanilla": "#3182bd",
    "MAPPO SelfHeal": "#ff7f0e",
    "MAPPO Vanilla": "#fdd0a2",
    "QMIX SelfHeal": "#2ca02c",
    "QMIX Vanilla": "#a1d99b",
    "MAA2C SelfHeal": "#d62728",
    "MAA2C Vanilla": "#fdae6b",
    "COMA SelfHeal": "#9467bd",
    "COMA Vanilla": "#bcbddc",
    "I-DQN SelfHeal": "#8c564b",
    "I-DQN Vanilla": "#c49c94",
}

MARKER_MAP = {
    "RSPO SelfHeal": "o",
    "RSPO Vanilla": "s",
    "MAPPO SelfHeal": "^",
    "MAPPO Vanilla": "v",
    "QMIX SelfHeal": "D",
    "QMIX Vanilla": "d",
    "MAA2C SelfHeal": "P",
    "MAA2C Vanilla": "X",
    "COMA SelfHeal": "*",
    "COMA Vanilla": "h",
    "I-DQN SelfHeal": "p",
    "I-DQN Vanilla": "8",
}

LINESTYLE_MAP = {
    "RSPO SelfHeal": "-",
    "RSPO Vanilla": "--",
    "MAPPO SelfHeal": "-",
    "MAPPO Vanilla": "-.",
    "QMIX SelfHeal": "-",
    "QMIX Vanilla": ":",
    "MAA2C SelfHeal": "-",
    "MAA2C Vanilla": "--",
    "COMA SelfHeal": "-",
    "COMA Vanilla": ":",
    "I-DQN SelfHeal": "-",
    "I-DQN Vanilla": "-.",
}


def generate_degradation_plots_2d():
    for task in TASKS:
        primary_metric = "coverage_rate" if task == "Coverage" else "target_found_rate"
        metric_label = "Coverage Rate (%)" if task == "Coverage" else "Target Intercept Rate (%)"
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"DSSE-2D {task} — Fault Severity Degradation Curves", fontsize=16, fontweight="bold", y=0.98)
        
        for env_idx, env in enumerate(ENVS):
            ax = axes[env_idx // 2, env_idx % 2]
            model_data = {}
            
            for fault_tag, fault_prob in FAULT_LEVELS:
                csv_path = os.path.join(BASE_DIR, task, env, fault_tag, f"{task.lower()}_2d_summary.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        m = row["Model"]
                        val = row[primary_metric] if primary_metric in row else 0.0
                        if m not in model_data:
                            model_data[m] = []
                        model_data[m].append((fault_prob, val))
            
            # Sort models by standard order
            sorted_models = [m for m in MODEL_ORDER if m in model_data] + [m for m in model_data if m not in MODEL_ORDER]
            
            for m_idx, m in enumerate(sorted_models):
                points = model_data[m]
                if points:
                    points.sort(key=lambda p: p[0])
                    x_vals = [p[0] * 10000 for p in points]
                    
                    # Add tiny visual offset for overlapping lines (0.08% per model index)
                    offset = (m_idx - len(sorted_models)/2.0) * 0.08 if task == "Coverage" else 0.0
                    y_vals = [p[1] + offset for p in points]
                    
                    is_rspo = "RSPO" in m
                    lw = 2.2 if is_rspo else 1.4
                    ls = LINESTYLE_MAP.get(m, "-")
                    marker = MARKER_MAP.get(m, "o")
                    color = COLOR_MAP.get(m, None)
                    
                    ax.plot(
                        x_vals, y_vals, label=m, color=color, linestyle=ls,
                        linewidth=lw, alpha=0.9, marker=marker, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5
                    )
            
            ax.set_title(f"Environment: {env.replace('_', ' ')}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Fault Probability (×10⁻⁴ per step)", fontsize=10)
            ax.set_ylabel(metric_label, fontsize=10)
            ax.set_xticks([0.0, 2.5, 5.0, 7.5, 10.0])
            ax.grid(True, linestyle="--", alpha=0.5)
            
            if env_idx == 0:
                ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.9)
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(OUTPUT_DIR, f"{task.lower()}_2d_fault_degradation_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[✓] 2D Degradation plot saved to: {save_path}")


if __name__ == "__main__":
    generate_degradation_plots_2d()
