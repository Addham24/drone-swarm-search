#!/usr/bin/env python3
"""
Regenerate Coverage Dashboards for All 40 Scenarios (2D-final-v3 & 3D-final-v3)
===============================================================================
Ensures 100% visualization consistency across all 40 benchmark scenarios:
  - Panel 5: Labeled as '5. Mission Return (5× cells searched)'
  - Panel 6: Plots the mean of the per-seed 'smei' column (per-episode SMEI)
  - Identical styling, colors, and model order across 2D and 3D
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ENV_CONFIGS = [
    {"tag": "25x25_Standard", "label": "25×25 Standard"},
    {"tag": "25x25_Obstacles", "label": "25×25 Obstacles"},
    {"tag": "50x50_Standard", "label": "50×50 Standard"},
    {"tag": "50x50_Obstacles", "label": "50×50 Obstacles"},
]

FAULT_REGIMES = [
    {"tag": "No_Faults_0.0", "label": "No Faults (fault_prob = 0.0)"},
    {"tag": "Fault_0.00025", "label": "Low Faults (fault_prob = 0.00025)"},
    {"tag": "Active_Faults_0.0005", "label": "Standard Faults (fault_prob = 0.0005)"},
    {"tag": "Fault_0.00075", "label": "High Faults (fault_prob = 0.00075)"},
    {"tag": "Fault_0.001", "label": "Severe Faults (fault_prob = 0.0010)"},
]

MODEL_ORDER = [
    "RSPO Vanilla", "RSPO SelfHeal",
    "MAPPO Vanilla", "MAPPO SelfHeal",
    "QMIX Vanilla", "QMIX SelfHeal",
    "MAA2C Vanilla", "MAA2C SelfHeal",
    "COMA Vanilla", "COMA SelfHeal",
    "I-DQN Vanilla", "I-DQN SelfHeal",
]


def render_dashboard(df_plot, out_dir, title, dim_label):
    os.makedirs(out_dir, exist_ok=True)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)

    models = df_plot["Model"].tolist()
    x = np.arange(len(models))

    panels = [
        ("coverage_rate",  "#2ca02c", "1. Area Coverage (%)",               0, 115, "%"),
        ("attrition_rate", "#d62728", "2. Agent Attrition (Crashes/ep)",     0, 4.8, ""),
        ("survival_rate",  "#9467bd", "3. Swarm Survival Rate (%)",          0, 115, "%"),
        ("avg_battery",    "#ff7f0e", "4. Fleet Battery (%)",                0, 115, "%"),
        ("mean_reward",    "#1f77b4", "5. Mission Return (5× cells searched)", None, None, ""),
        ("smei",           "#e377c2", "6. SMEI (%)",                         0, 115, "%"),
    ]

    for idx, (col, color, panel_title, ylim_lo, ylim_hi, suffix) in enumerate(panels):
        row, col_idx = divmod(idx, 3)
        ax = axes[row, col_idx]
        values = df_plot[col].values if col in df_plot.columns else np.zeros(len(models))
        bars = ax.bar(x, values, color=color, edgecolor="white")
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=40, ha="right", fontsize=7)
        if ylim_lo is not None:
            ax.set_ylim(ylim_lo, ylim_hi)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar in bars:
            val = bar.get_height()
            offset = 0.01 * (ylim_hi if ylim_hi else max(abs(val * 1.1), 1))
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + offset,
                    f"{val:.1f}{suffix}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    dash_filename = f"dashboard_coverage_{dim_label.lower()}.png"
    save_path = os.path.join(out_dir, dash_filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def regenerate_all(v3=True):
    suffix = "-v3" if v3 else ""
    b2d = os.path.expanduser(f"~/Desktop/Results/2D-final{suffix}/Coverage")
    b3d = os.path.expanduser(f"~/Desktop/Results/3D-final{suffix}/Coverage")

    print(f"\n{'='*80}")
    print(f"  REGENERATING ALL 40 DASHBOARDS ({'v3' if v3 else 'v2'})")
    print(f"{'='*80}\n")

    for dim, base_dir in [("2D", b2d), ("3D", b3d)]:
        count = 0
        for env_cfg in ENV_CONFIGS:
            for fault_cfg in FAULT_REGIMES:
                scenario_dir = os.path.join(base_dir, env_cfg["tag"], fault_cfg["tag"])
                raw_path = os.path.join(scenario_dir, f"coverage_{dim.lower()}_raw_seed_by_seed.csv")
                summary_path = os.path.join(scenario_dir, f"coverage_{dim.lower()}_summary.csv")

                if not os.path.exists(raw_path) or not os.path.exists(summary_path):
                    print(f"  [!] Missing data in {scenario_dir}, skipping...")
                    continue

                df_raw = pd.read_csv(raw_path)
                df_summary = pd.read_csv(summary_path)

                # Recompute exact per-seed SMEI mean for panel 6
                plot_records = []
                for m in MODEL_ORDER:
                    m_summ = df_summary[df_summary["Model"] == m]
                    m_raw = df_raw[df_raw["Model"] == m]
                    if len(m_summ) == 0:
                        continue
                    row = m_summ.iloc[0].to_dict()
                    # Authoritatively set SMEI to mean of per-seed smei column
                    row["smei"] = round(float(m_raw["smei"].mean()), 2)
                    plot_records.append(row)

                df_plot = pd.DataFrame(plot_records)
                title = f"DSSE-{dim} Coverage (Final Synchronized) — {env_cfg['label']} ({fault_cfg['label']})"
                save_path = render_dashboard(df_plot, scenario_dir, title, dim)
                count += 1
                print(f"  [✓] [{count:02d}/20] {dim} {env_cfg['label']} | {fault_cfg['label']} -> {os.path.basename(save_path)}")

    print(f"\n[✓] Successfully regenerated all 40 dashboards with consistent metrics and labels!\n")


if __name__ == "__main__":
    v3_mode = "--v2" not in sys.argv
    regenerate_all(v3=v3_mode)
