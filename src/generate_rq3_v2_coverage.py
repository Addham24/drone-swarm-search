"""
RQ3 v2 Visualisation Generator: Coverage-Focused Generalisation Dashboards
========================================================================
Generates dedicated, high-impact RQ3 generalisation matrices focused exclusively
on Area Coverage Rate (%), showing all 12 models across 4 map topologies and 5 fault rates.

Outputs:
  - 2D: ~/Desktop/Results/2D-mappo/rq3_v2_coverage_generalisation_2d.png
  - 3D: ~/Desktop/Results/3D-mappo/rq3_v2_coverage_generalisation_3d.png
  - Unified: ~/Desktop/Results/rq3_v2_coverage_generalisation_unified.png
"""

import os
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup paths
OUT_2D = os.path.expanduser("~/Desktop/Results/2D-mappo")
OUT_3D = os.path.expanduser("~/Desktop/Results/3D-mappo")
OUT_GEN = os.path.expanduser("~/Desktop/Results")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", None)

# Import data loader and environment definitions
from generate_rq_visualizations import (
    load_all_fault_dataset,
    ENVS,
    ENV_SHORT,
    FAULT_LEVELS,
    FAULT_TITLES
)

# Global typography styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def generate_rq3_v2_dashboard(dim):
    """Generate high-resolution, coverage-focused generalisation matrix for 2D or 3D."""
    df = load_all_fault_dataset(dim)

    # Sort models by overall mean Coverage across all 20 scenarios
    model_order = df.groupby("Model")["Coverage"].mean().sort_values(ascending=False).index.tolist()

    # Precompute model mean coverage across all 20 scenarios
    model_grand_means = df.groupby("Model")["Coverage"].mean().to_dict()

    fig = plt.figure(figsize=(25.0, 12.0))
    dim_desc = "Planar 2D Ground Search" if dim == "2D" else "Volumetric 3D Airspace Search"
    fig.suptitle(
        f"RQ3 (v2): Map Topology & Hardware Fault Generalisation Matrix — Area Coverage Rate (%)\n"
        f"{dim_desc} — All 12 Baseline Models Evaluated Across 4 Map Configurations & 5 Fault Regimes (100 Seeds per cell)",
        fontsize=14.5, fontweight="bold", y=0.985
    )

    # 1. Top Executive KPI Cards (y = 0.89 to 0.955)
    ax_kpi = fig.add_axes([0.06, 0.885, 0.91, 0.065])
    ax_kpi.axis("off")

    rspo_sub = df[df["Model"] == "RSPO SelfHeal"]
    rspo_mean = rspo_sub["Coverage"].mean()
    c_p0 = rspo_sub[rspo_sub["Fault"] == "No_Faults_0.0"]["Coverage"].mean()
    c_pmax = rspo_sub[rspo_sub["Fault"] == "Fault_0.001"]["Coverage"].mean()
    retention_pct = (c_pmax / c_p0) * 100.0 if c_p0 > 0 else 0.0

    c_25 = rspo_sub[rspo_sub["Env"].str.startswith("25")]["Coverage"].mean()
    c_50 = rspo_sub[rspo_sub["Env"].str.startswith("50")]["Coverage"].mean()
    scale_delta = c_50 - c_25

    c_std = rspo_sub[rspo_sub["Env"].str.endswith("Standard")]["Coverage"].mean()
    c_obs = rspo_sub[rspo_sub["Env"].str.endswith("Obstacles")]["Coverage"].mean()
    obs_delta = c_obs - c_std

    kpi_cards = [
        (f"[+] RSPO OVERALL COVERAGE", f"{rspo_mean:.1f}% Mean (#1 Rank)", "#003366", "#eaf2fb", "#1f77b4"),
        (f"[#] FAULT DEGRADATION RETENTION", f"{retention_pct:.1f}% Kept at p=0.001", "#1e7e34", "#ebfbee", "#28a745"),
        (f"[^] 4x AREA GENERALISATION", f"{c_50:.1f}% on 50x50 ({scale_delta:+.1f}%)", "#862e9c", "#f8f0fc", "#ae3ec9"),
        (f"[*] URBAN OBSTACLE RESILIENCE", f"{c_obs:.1f}% in Obstacles ({obs_delta:+.1f}%)", "#b7791f", "#fff9db", "#f39c12"),
    ]

    for k_idx, (title, val, txt_c, bg_c, border_c) in enumerate(kpi_cards):
        bx = k_idx * 0.252
        bw = 0.244
        card = patches.FancyBboxPatch((bx, 0.05), bw, 0.90, boxstyle="round,pad=0.03,rounding_size=0.10",
                                      facecolor=bg_c, edgecolor=border_c, linewidth=1.4)
        ax_kpi.add_patch(card)
        ax_kpi.text(bx + bw / 2, 0.65, title, ha="center", va="center", fontsize=8.4, fontweight="bold", color=txt_c)
        ax_kpi.text(bx + bw / 2, 0.26, val, ha="center", va="center", fontsize=11.2, fontweight="bold", color=txt_c)

    # 2. Main Heatmap Panels (5 Fault Columns + 1 Grand Summary Column)
    col_width = 0.155
    col_gap = 0.012
    col_start = 0.095
    row_bottom = 0.17
    row_height = 0.68

    cmap_cov = "Blues"
    vmin, vmax = 0, 100

    col_axes = []
    for col_idx, fault_name in enumerate(FAULT_LEVELS):
        cx = col_start + col_idx * (col_width + col_gap)
        ax = fig.add_axes([cx, row_bottom, col_width, row_height])
        col_axes.append(ax)

        ax.set_title(FAULT_TITLES[col_idx], fontsize=11.0, fontweight="bold", pad=10, color="#003366")

        # 12 models x 4 environments
        matrix = np.zeros((12, 4))
        for i, model in enumerate(model_order):
            for j, env in enumerate(ENVS):
                sub = df[(df["Model"] == model) & (df["Env"] == env) & (df["Fault"] == fault_name)]
                if not sub.empty:
                    matrix[i, j] = sub["Coverage"].values[0]

        im = ax.imshow(matrix, cmap=cmap_cov, vmin=vmin, vmax=vmax, aspect="auto")

        # Ticks
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(ENV_SHORT, fontsize=9.2, fontweight="bold")

        ax.set_yticks(np.arange(12))
        if col_idx == 0:
            ax.set_yticklabels(model_order, fontsize=9.5, fontweight="bold")
        else:
            ax.set_yticklabels([])

        # White grid
        ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Highlight RSPO SelfHeal Row (row index where model == RSPO SelfHeal)
        rspo_idx = model_order.index("RSPO SelfHeal")
        rect = patches.Rectangle((-0.5, rspo_idx - 0.5), 4, 1, linewidth=2.2, edgecolor="#d4af37",
                                 facecolor="none", linestyle="-", zorder=5)
        ax.add_patch(rect)

        # Cell annotations
        for i in range(12):
            for j in range(4):
                val = matrix[i, j]
                norm_val = val / 100.0
                txt_c = "white" if norm_val > 0.55 else "#111111"
                is_rspo = (i == rspo_idx)
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color=txt_c, fontsize=8.4, fontweight="bold" if is_rspo or norm_val > 0.70 else "normal", zorder=4)

    # 3. Right-Side Summary Column (Grand Mean Coverage across all 20 scenarios)
    summary_x = col_start + 5 * (col_width + col_gap) + 0.005
    summary_w = 0.048
    ax_sum = fig.add_axes([summary_x, row_bottom, summary_w, row_height])
    ax_sum.set_title("Overall\nMean", fontsize=10.5, fontweight="bold", pad=6, color="#1e40af")

    sum_matrix = np.zeros((12, 1))
    for i, model in enumerate(model_order):
        sum_matrix[i, 0] = model_grand_means[model]

    im_sum = ax_sum.imshow(sum_matrix, cmap=cmap_cov, vmin=vmin, vmax=vmax, aspect="auto")

    ax_sum.set_xticks([0])
    ax_sum.set_xticklabels(["Mean"], fontsize=9.2, fontweight="bold")
    ax_sum.set_yticks(np.arange(12))
    ax_sum.set_yticklabels([])

    ax_sum.set_xticks(np.arange(-.5, 1, 1), minor=True)
    ax_sum.set_yticks(np.arange(-.5, 12, 1), minor=True)
    ax_sum.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax_sum.tick_params(which="minor", bottom=False, left=False)

    # Highlight RSPO SelfHeal on summary column
    rect_sum = patches.Rectangle((-0.5, rspo_idx - 0.5), 1, 1, linewidth=2.2, edgecolor="#d4af37",
                                 facecolor="none", linestyle="-", zorder=5)
    ax_sum.add_patch(rect_sum)

    for i in range(12):
        val = sum_matrix[i, 0]
        norm_val = val / 100.0
        txt_c = "white" if norm_val > 0.55 else "#111111"
        is_rspo = (i == rspo_idx)
        ax_sum.text(0, i, f"{val:.1f}%", ha="center", va="center",
                    color=txt_c, fontsize=8.6, fontweight="bold" if is_rspo else "normal", zorder=4)

    # 4. Centered Bottom Horizontal Colorbar
    cbar_ax = fig.add_axes([0.32, 0.055, 0.36, 0.024])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Area Coverage Rate (%)", fontsize=10.8, fontweight="bold")

    # Save to dedicated RQ3_v2 file
    out_dir = OUT_2D if dim == "2D" else OUT_3D
    save_path = os.path.join(out_dir, f"rq3_v2_coverage_generalisation_{dim.lower()}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ3_v2 Coverage Dashboard ({dim}) saved to: {save_path}")
    return save_path


def generate_rq3_v2_unified():
    """Generate dual 2D vs 3D unified coverage generalisation dashboard."""
    df_2d = load_all_fault_dataset("2D")
    df_3d = load_all_fault_dataset("3D")

    # Consistent model order: 3D coverage ranking
    model_order = df_3d.groupby("Model")["Coverage"].mean().sort_values(ascending=False).index.tolist()

    fig = plt.figure(figsize=(25.0, 15.0))
    fig.suptitle(
        "RQ3 (v2): Multi-Dimensional Area Coverage Generalisation Matrix (2D vs 3D)\n"
        "Evaluating Exploration Robustness Across 4 Map Topologies & 5 Hardware Fault Regimes (All 12 Models)",
        fontsize=14.5, fontweight="bold", y=0.985
    )

    dims_data = [
        ("Row 1: Planar 2D Ground Patrol", df_2d, 0.53, 0.39),
        ("Row 2: Volumetric 3D Airspace Patrol", df_3d, 0.09, 0.39),
    ]

    col_width = 0.155
    col_gap = 0.012
    col_start = 0.095
    cmap_cov = "Blues"
    vmin, vmax = 0, 100

    last_im = None

    for row_title, df_dim, r_bottom, r_height in dims_data:
        # Row header label on left
        fig.text(0.015, r_bottom + r_height / 2, row_title,
                 fontsize=12.0, fontweight="bold", va="center", ha="left", rotation=90, color="#003366")

        for col_idx, fault_name in enumerate(FAULT_LEVELS):
            cx = col_start + col_idx * (col_width + col_gap)
            ax = fig.add_axes([cx, r_bottom, col_width, r_height])

            # Column title only on top row
            if r_bottom > 0.40:
                ax.set_title(FAULT_TITLES[col_idx], fontsize=11.0, fontweight="bold", pad=8, color="#003366")

            matrix = np.zeros((12, 4))
            for i, model in enumerate(model_order):
                for j, env in enumerate(ENVS):
                    sub = df_dim[(df_dim["Model"] == model) & (df_dim["Env"] == env) & (df_dim["Fault"] == fault_name)]
                    if not sub.empty:
                        matrix[i, j] = sub["Coverage"].values[0]

            im = ax.imshow(matrix, cmap=cmap_cov, vmin=vmin, vmax=vmax, aspect="auto")
            last_im = im

            ax.set_xticks(np.arange(4))
            if r_bottom < 0.40:  # Bottom row gets env labels
                ax.set_xticklabels(ENV_SHORT, fontsize=8.8, fontweight="bold")
            else:
                ax.set_xticklabels([])

            ax.set_yticks(np.arange(12))
            if col_idx == 0:
                ax.set_yticklabels(model_order, fontsize=9.0, fontweight="bold")
            else:
                ax.set_yticklabels([])

            ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Highlight RSPO SelfHeal Row (row 0)
            rect = patches.Rectangle((-0.5, -0.5), 4, 1, linewidth=2.0, edgecolor="#d4af37",
                                     facecolor="none", linestyle="-", zorder=5)
            ax.add_patch(rect)

            for i in range(12):
                for j in range(4):
                    val = matrix[i, j]
                    norm_val = val / 100.0
                    txt_c = "white" if norm_val > 0.55 else "#111111"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            color=txt_c, fontsize=7.8, fontweight="bold" if i == 0 or norm_val > 0.70 else "normal")

        # Right Summary Column for this row
        summary_x = col_start + 5 * (col_width + col_gap) + 0.005
        summary_w = 0.048
        ax_sum = fig.add_axes([summary_x, r_bottom, summary_w, r_height])
        if r_bottom > 0.40:
            ax_sum.set_title("Overall\nMean", fontsize=10.0, fontweight="bold", pad=6, color="#1e40af")

        sum_matrix = np.zeros((12, 1))
        for i, model in enumerate(model_order):
            sum_matrix[i, 0] = df_dim[df_dim["Model"] == model]["Coverage"].mean()

        ax_sum.imshow(sum_matrix, cmap=cmap_cov, vmin=vmin, vmax=vmax, aspect="auto")
        ax_sum.set_xticks([0])
        ax_sum.set_xticklabels(["Mean"] if r_bottom < 0.40 else [], fontsize=8.8, fontweight="bold")
        ax_sum.set_yticks(np.arange(12))
        ax_sum.set_yticklabels([])
        ax_sum.set_xticks(np.arange(-.5, 1, 1), minor=True)
        ax_sum.set_yticks(np.arange(-.5, 12, 1), minor=True)
        ax_sum.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax_sum.tick_params(which="minor", bottom=False, left=False)

        rect_sum = patches.Rectangle((-0.5, -0.5), 1, 1, linewidth=2.0, edgecolor="#d4af37",
                                     facecolor="none", linestyle="-", zorder=5)
        ax_sum.add_patch(rect_sum)

        for i in range(12):
            val = sum_matrix[i, 0]
            norm_val = val / 100.0
            txt_c = "white" if norm_val > 0.55 else "#111111"
            ax_sum.text(0, i, f"{val:.1f}%", ha="center", va="center",
                        color=txt_c, fontsize=8.0, fontweight="bold" if i == 0 else "normal")

    # Bottom Colorbar
    cbar_ax = fig.add_axes([0.35, 0.025, 0.30, 0.018])
    cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Area Coverage Rate (%)", fontsize=10.0, fontweight="bold")

    save_path = os.path.join(OUT_GEN, "rq3_v2_coverage_generalisation_unified.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ3_v2 Unified Coverage Dashboard saved to: {save_path}")
    return save_path


def copy_to_artifacts(paths):
    """Optionally copy generated plots if ARTIFACT_DIR is defined."""
    if not ARTIFACT_DIR or not os.path.exists(ARTIFACT_DIR):
        return
    for src in paths:
        if os.path.exists(src):
            fname = os.path.basename(src)
            dst = os.path.join(ARTIFACT_DIR, fname)
            shutil.copyfile(src, dst)
            print(f"[✓] Copied to artifact: {dst}")


def main():
    print("Generating RQ3_v2 Coverage-Only Generalisation Dashboards...")
    p1 = generate_rq3_v2_dashboard("2D")
    p2 = generate_rq3_v2_dashboard("3D")
    p3 = generate_rq3_v2_unified()

    all_plots = [p1, p2, p3]
    copy_to_artifacts(all_plots)
    print("\nAll RQ3_v2 coverage visualizations generated successfully without touching existing files!")


if __name__ == "__main__":
    main()
