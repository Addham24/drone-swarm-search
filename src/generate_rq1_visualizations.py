"""
RQ1 Visualization Generator: Paired Ablation Analysis (Panel A)
==============================================================
Evaluates whether self-healing (compensation reward and fault-aware signals)
improves Coverage and Fleet Survival compared with Vanilla training across
ALL 6 model families (12 models total).

Generates:
  1. DSSE-3D Panel A: Paired Grouped Ablation (Coverage & Survival)
  2. DSSE-2D Panel A: Paired Grouped Ablation (Coverage & Survival)
  3. Unified Comparative Dashboard (2D vs 3D)

Saves to:
  ~/Desktop/Results/2D-mappo/rq1_self_healing_ablation_2d.png
  ~/Desktop/Results/3D-mappo/rq1_self_healing_ablation_3d.png
  ~/Desktop/Results/rq1_self_healing_ablation_unified.png
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

# Import data loader from generate_rq_visualizations
from generate_rq_visualizations import load_all_fault_dataset

ARCHITECTURES = ["RSPO", "MAPPO", "QMIX", "COMA", "MAA2C", "I-DQN"]
ARCH_LABELS = [
    "RSPO\n(Ours)",
    "MAPPO\n(PPO Base)",
    "QMIX\n(Value Factor)",
    "COMA\n(Counterfactual)",
    "MAA2C\n(Actor-Critic)",
    "I-DQN\n(Independent)"
]

# Matplotlib global styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def compute_ablation_data(dim, fault_regime="Active_Faults_0.0005"):
    """Compute paired statistics for Vanilla vs SelfHeal across all 6 architectures."""
    df = load_all_fault_dataset(dim)
    sub = df[df["Fault"] == fault_regime].copy()

    records = []
    for arch in ARCHITECTURES:
        v_sub = sub[sub["Model"] == f"{arch} Vanilla"]
        s_sub = sub[sub["Model"] == f"{arch} SelfHeal"]

        v_cov = v_sub["Coverage"].mean()
        s_cov = s_sub["Coverage"].mean()
        d_cov = s_cov - v_cov

        v_sur = v_sub["Survival"].mean()
        s_sur = s_sub["Survival"].mean()
        d_sur = s_sur - v_sur

        v_smei = v_sub["SMEI"].mean()
        s_smei = s_sub["SMEI"].mean()
        d_smei = s_smei - v_smei

        records.append({
            "Arch": arch,
            "Vanilla_Cov": v_cov,
            "SelfHeal_Cov": s_cov,
            "Delta_Cov": d_cov,
            "Vanilla_Surv": v_sur,
            "SelfHeal_Surv": s_sur,
            "Delta_Surv": d_sur,
            "Vanilla_SMEI": v_smei,
            "SelfHeal_SMEI": s_smei,
            "Delta_SMEI": d_smei,
        })
    return pd.DataFrame(records)


def draw_paired_bars(ax, x_indices, v_vals, s_vals, d_vals, v_color, s_color, metric_name, y_max):
    """Draw paired bars with value labels and floating delta badges."""
    width = 0.35
    x_v = x_indices - width / 2 - 0.02
    x_s = x_indices + width / 2 + 0.02

    # Draw bars
    bars_v = ax.bar(x_v, v_vals, width, color=v_color, edgecolor="#333333", linewidth=1.1,
                    label="Vanilla Training", alpha=0.90, zorder=3)
    bars_s = ax.bar(x_s, s_vals, width, color=s_color, edgecolor="#111111", linewidth=1.4,
                    label="Self-Healing (Compensation + Fault Signals)", alpha=0.95, zorder=3)

    # Grid
    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=0)
    ax.set_ylim(0, y_max)

    # Annotate bar tops and delta brackets
    for i in range(len(x_indices)):
        v = v_vals[i]
        s = s_vals[i]
        d = d_vals[i]
        xi_v = x_v[i]
        xi_s = x_s[i]
        mid_x = (xi_v + xi_s) / 2

        # Bar value text
        ax.text(xi_v, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.8,
                color="#444444", fontweight="normal", zorder=4)
        ax.text(xi_s, s + 1.2, f"{s:.1f}%", ha="center", va="bottom", fontsize=9.2,
                color="#000000", fontweight="bold", zorder=4)

        # Delta Bracket & Pill
        max_bar = max(v, s)
        bracket_y = max_bar + 7.5
        tick_h = 1.8

        # Connecting bracket lines
        ax.plot([xi_v, xi_v, xi_s, xi_s],
                [bracket_y - tick_h, bracket_y, bracket_y, bracket_y - tick_h],
                color="#666666", linewidth=1.1, zorder=4)

        # Delta Badge Style
        if d > 0.5:
            pill_bg = "#ecfdf5"
            pill_border = "#10b981"
            pill_txt_color = "#065f46"
            delta_str = f"+{d:.1f}%"
        elif d < -0.5:
            pill_bg = "#fff1f2"
            pill_border = "#f43f5e"
            pill_txt_color = "#9f1239"
            delta_str = f"{d:.1f}%"
        else:
            pill_bg = "#f1f5f9"
            pill_border = "#94a3b8"
            pill_txt_color = "#334155"
            delta_str = f"{d:+.1f}%"

        # Pill Patch
        pill_w = 0.58
        pill_h = y_max * 0.052
        pill = patches.FancyBboxPatch((mid_x - pill_w / 2, bracket_y + 1.0), pill_w, pill_h,
                                      boxstyle="round,pad=0.04,rounding_size=0.15",
                                      facecolor=pill_bg, edgecolor=pill_border, linewidth=1.2, zorder=5)
        ax.add_patch(pill)
        ax.text(mid_x, bracket_y + 1.0 + pill_h / 2, delta_str, ha="center", va="center",
                fontsize=8.6, fontweight="bold", color=pill_txt_color, zorder=6)


def generate_rq1_dashboard(dim):
    """Generate high-impact publication-grade Paired Ablation Dashboard (Panel A) for 2D or 3D."""
    df_ab = compute_ablation_data(dim)

    fig = plt.figure(figsize=(19.5, 12.0))
    dim_name = "Planar 2D Ground Search" if dim == "2D" else "Volumetric 3D Airspace Search"
    fig.suptitle(f"RQ1: Impact of Self-Healing Mechanisms on Swarm Performance (Panel A: Paired Ablation)\n"
                 f"{dim_name} — All 6 Model Families Grouped Under Active Faults (p = 0.0005, 100 Seeds per condition)",
                 fontsize=14.5, fontweight="bold", y=0.98)

    # 1. Top Executive KPI Cards (height ~7%)
    ax_kpi = fig.add_axes([0.05, 0.875, 0.90, 0.065])
    ax_kpi.axis("off")

    rspo_cov_d = df_ab[df_ab["Arch"] == "RSPO"]["Delta_Cov"].values[0]
    rspo_sur_d = df_ab[df_ab["Arch"] == "RSPO"]["Delta_Surv"].values[0]
    rspo_smei_d = df_ab[df_ab["Arch"] == "RSPO"]["Delta_SMEI"].values[0]

    max_cov_arch = df_ab.loc[df_ab["Delta_Cov"].idxmax()]["Arch"]
    max_cov_gain = df_ab["Delta_Cov"].max()

    max_sur_arch = df_ab.loc[df_ab["Delta_Surv"].idxmax()]["Arch"]
    max_sur_gain = df_ab["Delta_Surv"].max()

    kpis = [
        (f"[+] RSPO COVERAGE UPLIFT", f"+{rspo_cov_d:.1f}% Area Gain", "#003366", "#eaf2fb", "#1f77b4"),
        (f"[^] RSPO SURVIVAL UPLIFT", f"+{rspo_sur_d:.1f}% Fleet Survival", "#1e7e34", "#ebfbee", "#28a745"),
        (f"[*] RSPO HOLISTIC SMEI", f"+{rspo_smei_d:.1f}% Mission Efficacy", "#b7791f", "#fff9db", "#f39c12"),
        (f"[#1] PEAK BENCHMARK GAIN", f"{max_sur_arch}: +{max_sur_gain:.1f}% Surv", "#862e9c", "#f8f0fc", "#ae3ec9"),
    ]

    for k_idx, (title, val, txt_c, bg_c, border_c) in enumerate(kpis):
        bx = k_idx * 0.252
        bw = 0.244
        card = patches.FancyBboxPatch((bx, 0.05), bw, 0.90, boxstyle="round,pad=0.03,rounding_size=0.10",
                                      facecolor=bg_c, edgecolor=border_c, linewidth=1.4)
        ax_kpi.add_patch(card)
        ax_kpi.text(bx + bw / 2, 0.65, title, ha="center", va="center", fontsize=8.2, fontweight="bold", color=txt_c)
        ax_kpi.text(bx + bw / 2, 0.26, val, ha="center", va="center", fontsize=11.0, fontweight="bold", color=txt_c)

    # 2. Main Paired Bar Plots: Subplot 1 (Coverage) and Subplot 2 (Survival)
    # Axes: left = 0.06, width = 0.42, gap = 0.06
    ax1 = fig.add_axes([0.06, 0.28, 0.42, 0.55])
    ax2 = fig.add_axes([0.54, 0.28, 0.42, 0.55])

    x = np.arange(len(ARCHITECTURES))

    # --- Subplot 1: Area Coverage Rate (%) ---
    ax1.set_title("1. Area Coverage Rate (%) — Exploration Efficacy", fontsize=12.5, fontweight="bold", pad=12, color="#003366")
    draw_paired_bars(
        ax=ax1,
        x_indices=x,
        v_vals=df_ab["Vanilla_Cov"].values,
        s_vals=df_ab["SelfHeal_Cov"].values,
        d_vals=df_ab["Delta_Cov"].values,
        v_color="#94a3b8",      # Slate grey
        s_color="#2563eb",      # Vibrant cobalt
        metric_name="Coverage",
        y_max=108 if dim == "3D" else 85
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(ARCH_LABELS, fontsize=9.8, fontweight="bold")
    ax1.set_ylabel("Mean Area Coverage Rate (%)", fontsize=11.0, fontweight="bold")

    # --- Subplot 2: Fleet Survival Rate (%) ---
    ax2.set_title("2. Fleet Survival Rate (%) — Swarm Longevity", fontsize=12.5, fontweight="bold", pad=12, color="#1e7e34")
    draw_paired_bars(
        ax=ax2,
        x_indices=x,
        v_vals=df_ab["Vanilla_Surv"].values,
        s_vals=df_ab["SelfHeal_Surv"].values,
        d_vals=df_ab["Delta_Surv"].values,
        v_color="#cbd5e1",      # Light grey-blue
        s_color="#059669",      # Emerald green
        metric_name="Survival",
        y_max=70 if dim == "3D" else 60
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(ARCH_LABELS, fontsize=9.8, fontweight="bold")
    ax2.set_ylabel("Mean Fleet Survival Rate (%)", fontsize=11.0, fontweight="bold")

    # Shared Legend at top of main plots
    handles = [
        patches.Patch(facecolor="#94a3b8", edgecolor="#333333", linewidth=1.1, label="Vanilla Training (No Self-Healing)"),
        patches.Patch(facecolor="#2563eb", edgecolor="#111111", linewidth=1.4, label="Self-Healing Enabled (Compensation Reward + Fault Signals)"),
    ]
    ax1.legend(handles=handles, loc="upper left", fontsize=9.5, frameon=True, facecolor="white", framealpha=0.92)

    handles2 = [
        patches.Patch(facecolor="#cbd5e1", edgecolor="#333333", linewidth=1.1, label="Vanilla Training (No Self-Healing)"),
        patches.Patch(facecolor="#059669", edgecolor="#111111", linewidth=1.4, label="Self-Healing Enabled (Compensation Reward + Fault Signals)"),
    ]
    ax2.legend(handles=handles2, loc="upper left", fontsize=9.5, frameon=True, facecolor="white", framealpha=0.92)

    # 3. Bottom Ablation Scorecard Cards (height ~15%)
    ax_bottom = fig.add_axes([0.05, 0.04, 0.90, 0.18])
    ax_bottom.axis("off")
    ax_bottom.text(0.50, 0.94, "Ablation Summary Scorecard Across All 6 Model Families",
                   ha="center", va="center", fontsize=11.0, fontweight="bold", color="#111111")

    cw = 0.155
    cg = 0.014
    for i, row in df_ab.iterrows():
        cx = i * (cw + cg)
        arch = row["Arch"]
        dc = row["Delta_Cov"]
        ds = row["Delta_Surv"]
        dm = row["Delta_SMEI"]

        is_rspo = (arch == "RSPO")
        bg_card = "#f0f7ff" if is_rspo else "#fafafa"
        border_card = "#1f77b4" if is_rspo else "#e0e0e0"

        c_box = patches.FancyBboxPatch((cx, 0.02), cw, 0.84, boxstyle="round,pad=0.03,rounding_size=0.12",
                                       facecolor=bg_card, edgecolor=border_card, linewidth=1.4 if is_rspo else 0.8)
        ax_bottom.add_patch(c_box)

        # Header
        ax_bottom.text(cx + cw / 2, 0.72, f"{arch}", ha="center", va="center",
                       fontsize=10.5, fontweight="bold", color="#003366" if is_rspo else "#222222")

        # Metrics
        cov_sign = "+" if dc > 0 else ""
        surv_sign = "+" if ds > 0 else ""
        smei_sign = "+" if dm > 0 else ""

        ax_bottom.text(cx + 0.015, 0.52, f"Δ Cov:", fontsize=8.6, color="#444444")
        ax_bottom.text(cx + cw - 0.015, 0.52, f"{cov_sign}{dc:.1f}%", fontsize=8.8,
                       fontweight="bold", color="#1e40af" if dc > 0 else "#991b1b", ha="right")

        ax_bottom.text(cx + 0.015, 0.35, f"Δ Surv:", fontsize=8.6, color="#444444")
        ax_bottom.text(cx + cw - 0.015, 0.35, f"{surv_sign}{ds:.1f}%", fontsize=8.8,
                       fontweight="bold", color="#065f46" if ds > 0 else "#991b1b", ha="right")

        ax_bottom.text(cx + 0.015, 0.18, f"Δ SMEI:", fontsize=8.6, color="#444444")
        ax_bottom.text(cx + cw - 0.015, 0.18, f"{smei_sign}{dm:.1f}%", fontsize=8.8,
                       fontweight="bold", color="#b45309" if dm > 0 else "#991b1b", ha="right")

    out_dir = OUT_2D if dim == "2D" else OUT_3D
    save_path = os.path.join(out_dir, f"rq1_self_healing_ablation_{dim.lower()}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ1 Ablation Dashboard ({dim}) saved to: {save_path}")
    return save_path


def generate_rq1_unified():
    """Generate dual 2D vs 3D unified ablation comparison."""
    df_2d = compute_ablation_data("2D")
    df_3d = compute_ablation_data("3D")

    fig, axes = plt.subplots(2, 2, figsize=(21.0, 13.5))
    fig.suptitle("RQ1: Multi-Dimensional Paired Ablation Analysis (2D vs 3D Comparison)\n"
                 "Evaluating Self-Healing (Compensation + Fault Signals) Across All 6 Model Families Under Active Faults (p = 0.0005)",
                 fontsize=14.5, fontweight="bold", y=0.98)

    x = np.arange(len(ARCHITECTURES))

    # Row 0: 2D Coverage & Survival
    # Row 1: 3D Coverage & Survival
    dims_data = [("2D (Planar Surface Patrol)", df_2d, axes[0, :], 85, 60),
                 ("3D (Volumetric Airspace Patrol)", df_3d, axes[1, :], 108, 70)]

    for row_name, df_dim, (ax_c, ax_s), ymax_c, ymax_s in dims_data:
        # Coverage
        ax_c.set_title(f"Area Coverage Rate (%) — {row_name}", fontsize=11.5, fontweight="bold", pad=10, color="#003366")
        draw_paired_bars(
            ax=ax_c, x_indices=x,
            v_vals=df_dim["Vanilla_Cov"].values,
            s_vals=df_dim["SelfHeal_Cov"].values,
            d_vals=df_dim["Delta_Cov"].values,
            v_color="#94a3b8", s_color="#2563eb",
            metric_name="Coverage", y_max=ymax_c
        )
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(ARCH_LABELS, fontsize=9.2, fontweight="bold")
        ax_c.set_ylabel("Coverage Rate (%)", fontsize=10.5, fontweight="bold")

        # Survival
        ax_s.set_title(f"Fleet Survival Rate (%) — {row_name}", fontsize=11.5, fontweight="bold", pad=10, color="#1e7e34")
        draw_paired_bars(
            ax=ax_s, x_indices=x,
            v_vals=df_dim["Vanilla_Surv"].values,
            s_vals=df_dim["SelfHeal_Surv"].values,
            d_vals=df_dim["Delta_Surv"].values,
            v_color="#cbd5e1", s_color="#059669",
            metric_name="Survival", y_max=ymax_s
        )
        ax_s.set_xticks(x)
        ax_s.set_xticklabels(ARCH_LABELS, fontsize=9.2, fontweight="bold")
        ax_s.set_ylabel("Survival Rate (%)", fontsize=10.5, fontweight="bold")

    # Add legends to top subplots
    axes[0, 0].legend(loc="upper left", fontsize=9.0, framealpha=0.92)
    axes[0, 1].legend(loc="upper left", fontsize=9.0, framealpha=0.92)

    plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.95])
    save_path = os.path.join(OUT_GEN, "rq1_self_healing_ablation_unified.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] RQ1 Unified saved to: {save_path}")
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
    print("Generating publication-grade RQ1 Paired Ablation Dashboards (Panel A)...")
    p1 = generate_rq1_dashboard("2D")
    p2 = generate_rq1_dashboard("3D")
    p3 = generate_rq1_unified()

    all_plots = [p1, p2, p3]
    copy_to_artifacts(all_plots)
    print("\nAll RQ1 visualisations generated successfully!")


if __name__ == "__main__":
    main()
