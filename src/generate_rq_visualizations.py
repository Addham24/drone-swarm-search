"""
Visualisation Generator for RQ3 (Generalisation) and RQ4 (Rankings & Leaderboard)
================================================================================
Customized according to user steering:
  - RQ3: Comprehensive dashboard for ALL 12 models across ALL 5 fault rates
         and 4 map topologies (25x25 vs 50x50, obstacles vs none), split by fault rates.
         Evaluates Coverage Rate (%), Survival Rate (%), and SMEI (%).
  - RQ4: Focused, uncluttered executive Leaderboard capturing ONLY overall model
         standings and aggregate means (SMEI, Coverage, Survival) + scenario wins.

Outputs saved to:
  ~/Desktop/Results/2D-mappo/
  ~/Desktop/Results/3D-mappo/
  ~/Desktop/Results/
  and copied to Artifacts directory.
"""

import os
import glob
import shutil
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Setup paths
BASE_2D = os.path.expanduser("~/Desktop/Results/2D-mappo/Coverage")
BASE_3D = os.path.expanduser("~/Desktop/Results/3D-mappo/Coverage")
OUT_2D = os.path.expanduser("~/Desktop/Results/2D-mappo")
OUT_3D = os.path.expanduser("~/Desktop/Results/3D-mappo")
OUT_GEN = os.path.expanduser("~/Desktop/Results")
ARTIFACT_DIR = "/Users/AdhamMotawi/.gemini/antigravity-ide/brain/261d090f-67ca-4f3a-8220-dd983c502482"

ENVS = ["25x25_Standard", "25x25_Obstacles", "50x50_Standard", "50x50_Obstacles"]
ENV_SHORT = ["25 Std", "25 Obs", "50 Std", "50 Obs"]
ENV_LABELS = [
    "25×25 Standard (Baseline)",
    "25×25 Obstacles (Urban)",
    "50×50 Standard (4× Area)",
    "50×50 Obstacles (4× + Urban)"
]

FAULT_LEVELS = [
    "No_Faults_0.0",
    "Fault_0.00025",
    "Active_Faults_0.0005",
    "Fault_0.00075",
    "Fault_0.001"
]
FAULT_TITLES = [
    "p = 0.0 (Zero Faults)",
    "p = 0.00025 (Low Faults)",
    "p = 0.0005 (Active Faults)",
    "p = 0.00075 (High Faults)",
    "p = 0.001 (Severe Faults)"
]

ORDERED_12_MODELS = [
    "RSPO SelfHeal", "RSPO Vanilla",
    "MAPPO SelfHeal", "MAPPO Vanilla",
    "QMIX SelfHeal", "COMA Vanilla",
    "COMA SelfHeal", "MAA2C SelfHeal",
    "QMIX Vanilla", "I-DQN SelfHeal",
    "I-DQN Vanilla", "MAA2C Vanilla",
]

# Matplotlib global styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


# ==============================================================================
# DATA LOADERS
# ==============================================================================
def load_all_fault_dataset(dim):
    """Load all 20 CSV files (4 environments x 5 fault rates) for 2D or 3D."""
    base = BASE_2D if dim == "2D" else BASE_3D
    prefix = dim.lower()
    rows = []
    for env in ENVS:
        for fault in FAULT_LEVELS:
            p = os.path.join(base, env, fault, f"coverage_{prefix}_summary.csv")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing CSV: {p}")
            df = pd.read_csv(p)
            cov_c = "coverage_rate" if "coverage_rate" in df.columns else "Coverage Rate (%)"
            surv_c = "survival_rate" if "survival_rate" in df.columns else "Survival Rate (%)"
            smei_c = "smei" if "smei" in df.columns else "SMEI (%)"
            for _, r in df.iterrows():
                rows.append({
                    "Dim": dim,
                    "Env": env,
                    "Fault": fault,
                    "Model": r["Model"],
                    "Coverage": float(r[cov_c]),
                    "Survival": float(r[surv_c]),
                    "SMEI": float(r[smei_c])
                })
    return pd.DataFrame(rows)


def compute_leaderboard_stats(dim, fault_regime=None):
    """Compute aggregate rankings, win counts, and mean metrics across scenarios.
    If fault_regime is None, aggregates across all 5 fault rates and 4 topologies (all 20 scenarios).
    """
    df_full = load_all_fault_dataset(dim)
    if fault_regime is not None:
        df = df_full[df_full["Fault"] == fault_regime].copy()
        scenarios = [(fault_regime, env) for env in ENVS]
    else:
        df = df_full.copy()
        scenarios = [(f, env) for f in df["Fault"].unique() for env in ENVS]

    # Calculate scenario wins across evaluated scenarios
    cov_wins = {m: 0 for m in ORDERED_12_MODELS}
    surv_wins = {m: 0 for m in ORDERED_12_MODELS}
    smei_wins = {m: 0 for m in ORDERED_12_MODELS}

    for f_val, env in scenarios:
        sub = df[(df["Fault"] == f_val) & (df["Env"] == env)]
        if not sub.empty:
            best_cov = sub.loc[sub["Coverage"].idxmax()]["Model"]
            best_surv = sub.loc[sub["Survival"].idxmax()]["Model"]
            best_smei = sub.loc[sub["SMEI"].idxmax()]["Model"]
            cov_wins[best_cov] += 1
            surv_wins[best_surv] += 1
            smei_wins[best_smei] += 1

    # Aggregate means across scenarios
    agg = df.groupby("Model").agg({
        "Coverage": "mean",
        "Survival": "mean",
        "SMEI": "mean"
    }).reset_index()

    agg["Cov_Wins"] = agg["Model"].map(cov_wins)
    agg["Surv_Wins"] = agg["Model"].map(surv_wins)
    agg["SMEI_Wins"] = agg["Model"].map(smei_wins)
    agg["Total_Wins"] = agg["Cov_Wins"] + agg["Surv_Wins"] + agg["SMEI_Wins"]

    # Sort descending by SMEI
    agg = agg.sort_values(by="SMEI", ascending=False).reset_index(drop=True)
    agg["Rank"] = agg.index + 1

    return agg, df


# ==============================================================================
# RQ3: COMPLETE GENERALISATION DASHBOARD (ALL 12 MODELS & 5 FAULT RATES)
# ==============================================================================
def generate_rq3_dashboard(dim):
    """Generate high-resolution multi-panel heatmap dashboard for DSSE-2D or DSSE-3D.
    Captures:
      - All 12 models (rows)
      - All 4 environment topologies (columns inside each fault panel)
      - All 5 fault rates (split across 5 horizontal columns)
      - 3 core metrics (Rows: Coverage Rate, Survival Rate, SMEI Composite)
    """
    df = load_all_fault_dataset(dim)
    
    # Sort models by overall mean SMEI in this dimension
    model_order = df.groupby("Model")["SMEI"].mean().sort_values(ascending=False).index.tolist()

    metrics = [
        ("Coverage", "1. Coverage Rate (%) Generalisation", "Blues", 0, 100),
        ("Survival", "2. Survival Rate (%) Generalisation", "Greens", 0, 75 if dim == "2D" else 80),
        ("SMEI", "3. SMEI Composite (%) Efficacy", "YlOrRd", 0, 75 if dim == "2D" else 85)
    ]

    fig = plt.figure(figsize=(26.0, 15.0))
    dim_desc = "Planar 2D Ground Patrol" if dim == "2D" else "Volumetric 3D Airspace Patrol"
    fig.suptitle(f"RQ3: DSSE-{dim} Complete Policy Generalisation Matrix Across All 12 Baselines & 5 Fault Rates\n"
                 f"{dim_desc} — Evaluated Across Unseen Map Topologies (25×25 vs 50×50, Obstacles vs None) & Increasing Fault Severity",
                 fontsize=15.0, fontweight="bold", y=0.985)

    # 3 rows of subplots x 5 columns (fault levels)
    # Using explicit add_axes for exact, non-overlapping spacing
    # Row heights: 0.24 each; gaps: 0.05
    # Row 0 (Coverage): top = 0.93 -> bottom = 0.69
    # Row 1 (Survival): top = 0.64 -> bottom = 0.40
    # Row 2 (SMEI):     top = 0.35 -> bottom = 0.11
    row_bottoms = [0.69, 0.40, 0.11]
    row_height = 0.24

    # Col lefts: left_margin = 0.08, width = 0.17, gap = 0.012
    col_lefts = [0.09 + i * 0.178 for i in range(5)]
    col_width = 0.165

    # Store images for colorbars
    last_ims = {}

    for row_idx, (metric_col, metric_title, cmap_name, vmin, vmax) in enumerate(metrics):
        # Row label on the far left
        fig.text(0.015, row_bottoms[row_idx] + row_height / 2, metric_title,
                 fontsize=12.5, fontweight="bold", va="center", ha="left", rotation=90, color="#111111")

        for col_idx, fault_name in enumerate(FAULT_LEVELS):
            ax = fig.add_axes([col_lefts[col_idx], row_bottoms[row_idx], col_width, row_height])

            # Header on top row only
            if row_idx == 0:
                ax.set_title(FAULT_TITLES[col_idx], fontsize=11.0, fontweight="bold", pad=10, color="#003366")

            # Extract 12x4 matrix
            matrix = np.zeros((12, 4))
            for i, model in enumerate(model_order):
                for j, env in enumerate(ENVS):
                    sub = df[(df["Model"] == model) & (df["Env"] == env) & (df["Fault"] == fault_name)]
                    if not sub.empty:
                        matrix[i, j] = sub[metric_col].values[0]

            im = ax.imshow(matrix, cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
            last_ims[metric_col] = im

            # Ticks
            ax.set_xticks(np.arange(4))
            if row_idx == 2:  # Bottom row gets env labels
                ax.set_xticklabels(ENV_SHORT, fontsize=9.0, fontweight="bold")
            else:
                ax.set_xticklabels([])

            ax.set_yticks(np.arange(12))
            if col_idx == 0:  # Far left gets model names
                ax.set_yticklabels(model_order, fontsize=9.2, fontweight="bold")
            else:
                ax.set_yticklabels([])

            # Grid lines
            ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 12, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
            ax.tick_params(which="minor", bottom=False, left=False)

            # Highlight RSPO SelfHeal Row (Row 0)
            rect = patches.Rectangle((-0.5, -0.5), 4, 1, linewidth=2.0, edgecolor="#d4af37", facecolor="none", linestyle="-")
            ax.add_patch(rect)

            # Annotate cells with number
            for i in range(12):
                for j in range(4):
                    val = matrix[i, j]
                    norm_val = (val - vmin) / max(vmax - vmin, 1)
                    txt_color = "white" if norm_val > 0.52 else "#111111"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            color=txt_color, fontsize=8.2, fontweight="bold" if i < 2 else "normal")

    # Dedicated Colorbars at the bottom
    # Colorbar 1 (Coverage)
    cbar_cov_ax = fig.add_axes([0.15, 0.04, 0.20, 0.016])
    cbar_cov = fig.colorbar(last_ims["Coverage"], cax=cbar_cov_ax, orientation="horizontal")
    cbar_cov.set_label("Coverage Rate (%)", fontsize=9.5, fontweight="bold")

    # Colorbar 2 (Survival)
    cbar_surv_ax = fig.add_axes([0.43, 0.04, 0.20, 0.016])
    cbar_surv = fig.colorbar(last_ims["Survival"], cax=cbar_surv_ax, orientation="horizontal")
    cbar_surv.set_label("Survival Rate (%)", fontsize=9.5, fontweight="bold")

    # Colorbar 3 (SMEI)
    cbar_smei_ax = fig.add_axes([0.71, 0.04, 0.20, 0.016])
    cbar_smei = fig.colorbar(last_ims["SMEI"], cax=cbar_smei_ax, orientation="horizontal")
    cbar_smei.set_label("SMEI Efficacy Composite (%)", fontsize=9.5, fontweight="bold")

    out_dir = OUT_2D if dim == "2D" else OUT_3D
    save_path = os.path.join(out_dir, f"rq3_generalisation_{dim.lower()}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ3 Complete Dashboard ({dim}) saved to: {save_path}")
    return save_path


def generate_rq3_unified():
    """Generate master unified generalisation comparison (2D vs 3D) across fault rates.
    Compares the 4 focal architectures: RSPO SelfHeal, RSPO Vanilla, MAPPO SelfHeal, MAPPO Vanilla.
    """
    df_2d = load_all_fault_dataset("2D")
    df_3d = load_all_fault_dataset("3D")

    focal_models = ["RSPO SelfHeal", "RSPO Vanilla", "MAPPO SelfHeal", "MAPPO Vanilla"]
    
    fig, axes = plt.subplots(2, 5, figsize=(25.0, 10.5))
    fig.suptitle("RQ3: Multi-Dimensional Policy Generalisation Matrix (2D vs 3D Comparison Across All 5 Fault Rates)\n"
                 "SMEI Mission Efficacy (%) Across Unseen Topologies (25×25 vs 50×50, Obstacles vs None) and Fault Regimes",
                 fontsize=15, fontweight="bold", y=0.98)

    for row_idx, (dim, df) in enumerate([("2D", df_2d), ("3D", df_3d)]):
        dim_lbl = "DSSE-2D (Surface Patrol)" if dim == "2D" else "DSSE-3D (Airspace Flyover)"
        fig.text(0.012, 0.72 if row_idx == 0 else 0.26, dim_lbl,
                 fontsize=12.5, fontweight="bold", va="center", ha="left", rotation=90)

        for col_idx, fault_name in enumerate(FAULT_LEVELS):
            ax = axes[row_idx, col_idx]
            if row_idx == 0:
                ax.set_title(FAULT_TITLES[col_idx], fontsize=11, fontweight="bold", pad=8)

            mat = np.zeros((4, 4))
            for i, m in enumerate(focal_models):
                for j, env in enumerate(ENVS):
                    sub = df[(df["Model"] == m) & (df["Env"] == env) & (df["Fault"] == fault_name)]
                    if not sub.empty:
                        mat[i, j] = sub["SMEI"].values[0]

            im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=75 if dim == "2D" else 80, aspect="auto")

            ax.set_xticks(np.arange(4))
            if row_idx == 1:
                ax.set_xticklabels(ENV_SHORT, fontsize=9.0, fontweight="bold")
            else:
                ax.set_xticklabels([])

            ax.set_yticks(np.arange(4))
            if col_idx == 0:
                ax.set_yticklabels(["RSPO SH", "RSPO Van", "MAPPO SH", "MAPPO Van"], fontsize=9.5, fontweight="bold")
            else:
                ax.set_yticklabels([])

            # Highlight RSPO SelfHeal Row
            rect = patches.Rectangle((-0.5, -0.5), 4, 1, linewidth=2.0, edgecolor="#d4af37", facecolor="none", linestyle="-")
            ax.add_patch(rect)

            for i in range(4):
                for j in range(4):
                    v = mat[i, j]
                    ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                            color="white" if v > 40 else "#111111", fontsize=9.5, fontweight="bold" if i == 0 else "normal")

            ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
            ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout(rect=[0.03, 0.08, 0.99, 0.94])

    cbar_ax = fig.add_axes([0.35, 0.03, 0.30, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("SMEI Composite Efficacy (%)", fontsize=10, fontweight="bold")

    save_path = os.path.join(OUT_GEN, "rq3_generalisation_unified.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ3 Unified saved to: {save_path}")
    return save_path


# ==============================================================================
# RQ4: FOCUSED LEADERBOARD (STANDINGS & AGGREGATE MEANS ONLY)
# ==============================================================================
def generate_rq4_leaderboard(dim):
    """Generate clean, focused, executive Leaderboard dashboard for DSSE-2D or DSSE-3D.
    Contains:
      - Overall Model Standings (Rank #1 to #12)
      - Aggregate Grand Means for SMEI (%), Coverage (%), Survival (%)
      - Executive summary KPI cards
      Averaged across all 5 fault regimes and 4 map topologies (20 scenarios, 100 seeds each).
    """
    agg, df_raw = compute_leaderboard_stats(dim, fault_regime=None)

    fig = plt.figure(figsize=(16.5, 11.8))
    dim_desc = "Planar 2D Search" if dim == "2D" else "Volumetric 3D Airspace Search"
    fig.suptitle(f"RQ4: DSSE-{dim} Benchmark Leaderboard & Overall Model Standings\n"
                 f"{dim_desc} — Grand Means Across All 5 Fault Severities (p = 0.0 to 0.001) & 4 Topologies (20 Scenarios, 100 Seeds each)",
                 fontsize=14.5, fontweight="bold", y=0.975)

    # 1. Top KPI Summary Cards (width 92%, height ~8%)
    ax_kpi = fig.add_axes([0.04, 0.865, 0.92, 0.065])
    ax_kpi.axis("off")

    top_m = agg.iloc[0]
    top_name = top_m["Model"]
    top_smei = top_m["SMEI"]
    top_cov = top_m["Coverage"]
    top_surv = top_m["Survival"]
    vanilla_mappo_smei = agg[agg["Model"] == "MAPPO Vanilla"]["SMEI"].values[0]
    delta_smei = top_smei - vanilla_mappo_smei

    kpis = [
        ("[#1] OVERALL BENCHMARK WINNER", f"{top_name}", "#003366", "#eaf2fb", "#1f77b4"),
        ("[*] GRAND SMEI SCORE", f"{top_smei:.2f}% Mean Efficacy", "#b7791f", "#fff9db", "#f39c12"),
        ("[+] AREA COVERAGE", f"{top_cov:.2f}% Mean Explored", "#1e7e34", "#ebfbee", "#28a745"),
        ("[^] ATTRITION RESILIENCE", f"+{delta_smei:.2f}% vs Vanilla MAPPO", "#862e9c", "#f8f0fc", "#ae3ec9"),
    ]

    for k_idx, (title, val, txt_c, bg_c, border_c) in enumerate(kpis):
        bx = k_idx * 0.252
        bw = 0.244
        card = patches.FancyBboxPatch((bx, 0.05), bw, 0.90, boxstyle="round,pad=0.03,rounding_size=0.10",
                                      facecolor=bg_c, edgecolor=border_c, linewidth=1.4)
        ax_kpi.add_patch(card)
        ax_kpi.text(bx + bw / 2, 0.65, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=txt_c)
        ax_kpi.text(bx + bw / 2, 0.26, val, ha="center", va="center", fontsize=11.0, fontweight="bold", color=txt_c)

    # 2. Main Leaderboard Card Stack
    ax_main = fig.add_axes([0.04, 0.08, 0.92, 0.76])
    ax_main.set_xlim(-4, 116)
    ax_main.set_ylim(-1.60, 11.7)
    ax_main.invert_yaxis()
    ax_main.axis("off")

    # Column sub-headers (Scenario Wins removed!)
    ax_main.text(-0.5, -0.95, "Rank & Algorithm", fontsize=11, fontweight="bold", color="#333333")
    ax_main.text(42.5, -0.95, "Mean SMEI (%)", fontsize=11, fontweight="bold", color="#d35400", ha="center")
    ax_main.text(71.5, -0.95, "Mean Coverage (%)", fontsize=11, fontweight="bold", color="#1f77b4", ha="center")
    ax_main.text(100.5, -0.95, "Mean Survival (%)", fontsize=11, fontweight="bold", color="#27ae60", ha="center")

    for i, row in agg.iterrows():
        y = i
        rank = row["Rank"]
        model = row["Model"]
        smei = row["SMEI"]
        cov = row["Coverage"]
        surv = row["Survival"]

        is_rspo = "RSPO" in model
        bg_color = "#eaf2fb" if is_rspo else ("#f8f9fa" if i % 2 == 0 else "#ffffff")

        # Row Background Card
        card = patches.FancyBboxPatch((-3.6, y - 0.40), 119.0, 0.80, boxstyle="round,pad=0.06,rounding_size=0.15",
                                      facecolor=bg_color, edgecolor="#1f77b4" if is_rspo else "#e0e0e0",
                                      linewidth=1.6 if is_rspo else 0.8)
        ax_main.add_patch(card)

        # Rank Badge
        badge_color = "#d4af37" if rank == 1 else ("#a0a0a0" if rank == 2 else ("#cd7f32" if rank == 3 else "#777777"))
        badge = patches.FancyBboxPatch((-2.8, y - 0.28), 4.6, 0.56, boxstyle="round,pad=0.05,rounding_size=0.15",
                                       facecolor=badge_color, edgecolor="white", linewidth=1.2)
        ax_main.add_patch(badge)
        ax_main.text(-0.5, y, f"#{rank}", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

        # Model Name
        m_color = "#003366" if is_rspo else "#222222"
        ax_main.text(3.0, y, model, ha="left", va="center", color=m_color, fontsize=10.5, fontweight="bold" if is_rspo else "normal")

        # Gauges with generous inter-column spacing and clear visible tracks
        gw = 14.5

        # 1. SMEI Progress Gauge
        gx1 = 32.0
        smei_w = (smei / 100.0) * gw
        ax_main.barh(y, gw, left=gx1, height=0.34, color="#e9ecef", edgecolor="#ced4da", linewidth=0.8, zorder=1)
        ax_main.barh(y, smei_w, left=gx1, height=0.34, color="#d35400" if is_rspo else "#e67e22", zorder=2)
        ax_main.text(gx1 + gw + 1.8, y, f"{smei:.2f}%", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#d35400")

        # 2. Coverage Progress Gauge
        gx2 = 61.0
        cov_w = (cov / 100.0) * gw
        ax_main.barh(y, gw, left=gx2, height=0.34, color="#e9ecef", edgecolor="#ced4da", linewidth=0.8, zorder=1)
        ax_main.barh(y, cov_w, left=gx2, height=0.34, color="#1f77b4" if is_rspo else "#3498db", zorder=2)
        ax_main.text(gx2 + gw + 1.8, y, f"{cov:.2f}%", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1f77b4")

        # 3. Survival Progress Gauge
        gx3 = 90.0
        surv_w = (surv / 100.0) * gw
        ax_main.barh(y, gw, left=gx3, height=0.34, color="#e9ecef", edgecolor="#ced4da", linewidth=0.8, zorder=1)
        ax_main.barh(y, surv_w, left=gx3, height=0.34, color="#27ae60", zorder=2)
        ax_main.text(gx3 + gw + 1.8, y, f"{surv:.2f}%", va="center", ha="left", fontsize=9.5, fontweight="bold", color="#27ae60")

    # Legend at the bottom
    ax_main.plot([], [], color="#d35400", linewidth=6, label="Mean SMEI Score (%)")
    ax_main.plot([], [], color="#1f77b4", linewidth=6, label="Mean Coverage Rate (%)")
    ax_main.plot([], [], color="#27ae60", linewidth=6, label="Mean Drone Survival (%)")
    ax_main.legend(loc="lower center", bbox_to_anchor=(0.50, -0.07), ncol=3, fontsize=9.5, frameon=False)

    out_dir = OUT_2D if dim == "2D" else OUT_3D
    save_path = os.path.join(out_dir, f"rq4_leaderboard_{dim.lower()}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ4 Leaderboard ({dim}) saved to: {save_path}")
    return save_path


def generate_rq4_unified():
    """Generate side-by-side dual 2D vs 3D Leaderboard comparing overall model standings."""
    agg_2d, _ = compute_leaderboard_stats("2D", fault_regime=None)
    agg_3d, _ = compute_leaderboard_stats("3D", fault_regime=None)

    fig, axes = plt.subplots(1, 2, figsize=(21.0, 11.8))
    fig.suptitle("RQ4: Overall Benchmark Leaderboard & Model Standings (2D vs 3D)\n"
                 "Grand Means Across All 5 Fault Severities (p = 0.0 to 0.001) & 4 Topologies (20 Scenarios, 100 Seeds each)",
                 fontsize=15, fontweight="bold", y=0.975)

    for col_idx, (dim, agg) in enumerate([("2D", agg_2d), ("3D", agg_3d)]):
        ax = axes[col_idx]
        ax.set_title(f"DSSE-{dim} Grand Leaderboard (Ranked by Mean SMEI)", fontsize=13, fontweight="bold", pad=16)
        ax.set_xlim(-3, 105)
        ax.set_ylim(-1.60, 11.8)
        ax.invert_yaxis()
        ax.axis("off")

        # Column headers (Scenario Wins removed!)
        ax.text(0.0, -0.95, "Rank & Model", fontsize=10.5, fontweight="bold", color="#333333")
        ax.text(37.5, -0.95, "Mean SMEI (%)", fontsize=10.0, fontweight="bold", color="#d35400", ha="center")
        ax.text(62.5, -0.95, "Mean Coverage (%)", fontsize=10.0, fontweight="bold", color="#1f77b4", ha="center")
        ax.text(87.5, -0.95, "Mean Survival (%)", fontsize=10.0, fontweight="bold", color="#27ae60", ha="center")

        for i, row in agg.iterrows():
            y = i
            rank = row["Rank"]
            model = row["Model"]
            smei = row["SMEI"]
            cov = row["Coverage"]
            surv = row["Survival"]

            is_rspo = "RSPO" in model
            bg_color = "#eaf2fb" if is_rspo else ("#f8f9fa" if i % 2 == 0 else "#ffffff")

            card = patches.FancyBboxPatch((-2.2, y - 0.40), 106.4, 0.80, boxstyle="round,pad=0.08,rounding_size=0.15",
                                          facecolor=bg_color, edgecolor="#1f77b4" if is_rspo else "#e0e0e0",
                                          linewidth=1.4 if is_rspo else 0.8)
            ax.add_patch(card)

            # Rank Badge
            badge_color = "#d4af37" if rank == 1 else ("#a0a0a0" if rank == 2 else ("#cd7f32" if rank == 3 else "#777777"))
            badge = patches.FancyBboxPatch((-1.6, y - 0.28), 4.6, 0.56, boxstyle="round,pad=0.05,rounding_size=0.15",
                                           facecolor=badge_color, edgecolor="white", linewidth=1.2)
            ax.add_patch(badge)
            ax.text(0.7, y, f"#{rank}", ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

            # Model Name
            m_color = "#003366" if is_rspo else "#222222"
            ax.text(4.2, y, model, ha="left", va="center", color=m_color, fontsize=10.0, fontweight="bold" if is_rspo else "normal")

            gw = 10.5
            # 1. SMEI
            gx1 = 29.0
            smei_w = (smei / 100.0) * gw
            ax.barh(y, gw, left=gx1, height=0.32, color="#e9ecef", edgecolor="#ced4da", linewidth=0.7, zorder=1)
            ax.barh(y, smei_w, left=gx1, height=0.32, color="#d35400" if is_rspo else "#e67e22", zorder=2)
            ax.text(gx1 + gw + 1.3, y, f"{smei:.2f}%", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#d35400")

            # 2. Coverage
            gx2 = 54.0
            cov_w = (cov / 100.0) * gw
            ax.barh(y, gw, left=gx2, height=0.32, color="#e9ecef", edgecolor="#ced4da", linewidth=0.7, zorder=1)
            ax.barh(y, cov_w, left=gx2, height=0.32, color="#1f77b4" if is_rspo else "#3498db", zorder=2)
            ax.text(gx2 + gw + 1.3, y, f"{cov:.2f}%", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#1f77b4")

            # 3. Survival
            gx3 = 79.0
            surv_w = (surv / 100.0) * gw
            ax.barh(y, gw, left=gx3, height=0.32, color="#e9ecef", edgecolor="#ced4da", linewidth=0.7, zorder=1)
            ax.barh(y, surv_w, left=gx3, height=0.32, color="#27ae60", zorder=2)
            ax.text(gx3 + gw + 1.3, y, f"{surv:.2f}%", va="center", ha="left", fontsize=8.6, fontweight="bold", color="#27ae60")

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    save_path = os.path.join(OUT_GEN, "rq4_leaderboard_unified.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ4 Unified saved to: {save_path}")
    return save_path


def copy_to_artifacts(paths):
    """Copy generated plots to antigravity artifact directory."""
    for src in paths:
        if os.path.exists(src):
            fname = os.path.basename(src)
            dst = os.path.join(ARTIFACT_DIR, fname)
            shutil.copyfile(src, dst)
            print(f"[✓] Copied to artifact: {dst}")


def main():
    print("Generating comprehensive RQ3 and focused RQ4 Visualisations...")
    p1 = generate_rq3_dashboard("2D")
    p2 = generate_rq3_dashboard("3D")
    p3 = generate_rq3_unified()

    p4 = generate_rq4_leaderboard("2D")
    p5 = generate_rq4_leaderboard("3D")
    p6 = generate_rq4_unified()

    all_plots = [p1, p2, p3, p4, p5, p6]
    copy_to_artifacts(all_plots)
    print("\nAll 6 RQ visualisations regenerated successfully!")


if __name__ == "__main__":
    main()
