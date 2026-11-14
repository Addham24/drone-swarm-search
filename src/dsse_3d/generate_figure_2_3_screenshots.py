#!/usr/bin/env python3
"""
Generate Publication-Quality 3D Visualiser Snapshots for Figure 2.3
===================================================================
Produces high-resolution 3D renders matching the evaluated environment:
  - Multi-height Urban Warfare obstacles (96 building blocks on 25x25)
  - Semi-transparent building prisms with height-stratified coloring
  - Dynamic altitude trajectories (Z=1 scan, Z=2 cruise, Z>=3 rooftop flyover)
  - Dotted altitude drop-lines and sensor footprints
  - Real-time telemetry HUD overlay
  - Rendered WITHOUT in-image titles per publication requirements

Saves to:
  - figure_2_3_3d_visualizer_urban_obstacles.png
  - figure_2_3_3d_visualizer_standard.png
  - figure_2_3_3d_visualizer_composite.png
"""

import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dsse_3d.env_creators import make_coverage_env_3d
from dsse_3d.visualizer_3d import (
    draw_ground_grid, draw_obstacles_3d, draw_trajectories_3d,
    draw_drones_3d, draw_search_frustum
)
from dsse_3d.policy_loaders import load_rllib_ppo_policy
from train_rspo_v2_vanilla import RSPOModelV2


def generate_3d_snapshot(map_type="obstacles", grid_size=25, n_steps=35, output_path=None):
    obstacle_mask = None
    prob_matrix_path = None
    if map_type == "obstacles":
        obs_path = os.path.join(SRC_DIR, f"obstacle_skyline_{grid_size}.npy")
        if os.path.exists(obs_path):
            obstacle_mask = np.load(obs_path)
        prob_path = os.path.join(SRC_DIR, f"obstacle_prob_matrix_{grid_size}.npy")
        if os.path.exists(prob_path):
            prob_matrix_path = prob_path

    env = make_coverage_env_3d(
        grid_size=grid_size,
        timestep_limit=750,
        max_battery=125,
        fault_prob=0.0005,
        is_self_heal=True,
        obstacle_mask=obstacle_mask,
        prob_matrix_path=prob_matrix_path,
    )

    # Load RSPO SelfHeal policy
    ckpt_path = os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_SelfHeal_rspo_v2_selfheal/RSPOPPO_DSSE_Coverage_RSPO_V2_SelfHeal_86fd5_00000_0_2026-09-04_06-40-32/checkpoint_000122")
    policy_fn = load_rllib_ppo_policy(RSPOModelV2, ckpt_path, "RSPO SelfHeal", greedy=True, sample_mappo=False)

    obs, infos = env.reset(seed=1000)

    # Step environment to build realistic 3D trajectories
    for step in range(n_steps):
        actions = policy_fn(obs, env.possible_agents, grid_size=grid_size)
        obs, rewards, term, trunc, infos = env.step(actions)
        if any(term.values()) or any(trunc.values()):
            break

    # Setup 3D Figure - clean publication presentation without in-image titles
    fig = plt.figure(figsize=(14, 11), facecolor="#141923")
    ax = fig.add_subplot(111, projection="3d", facecolor="#141923")

    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, env.z_max + 1)
    ax.set_xlabel("X (Ground Plane)", fontsize=11, color="white", labelpad=10)
    ax.set_ylabel("Y (Ground Plane)", fontsize=11, color="white", labelpad=10)
    ax.set_zlabel("Altitude Z (Voxel Layer)", fontsize=11, color="white", labelpad=10)

    # Note: in-image title omitted per specification!

    ax.tick_params(colors="#8892b0", labelsize=8)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#334155")
    ax.yaxis.pane.set_edgecolor("#334155")
    ax.zaxis.pane.set_edgecolor("#334155")
    ax.grid(color="#334155", linestyle=":", alpha=0.6)

    # Draw components
    base_env = env._get_base_env()
    prob_mat = base_env.probability_matrix.get_matrix() if hasattr(base_env, "probability_matrix") else None
    covered_cells = set(base_env.seen_states) if hasattr(base_env, "seen_states") else set()

    draw_ground_grid(ax, grid_size, prob_mat, covered_cells)
    if map_type == "obstacles":
        draw_obstacles_3d(ax, env.voxel._obstacle_heights)
    draw_trajectories_3d(ax, env.trajectory_history, max_trail=30)

    alive_mask = [True] * env.n_drones
    draw_drones_3d(ax, env.voxel, alive_mask)

    # Draw frustums for drones scanning
    for i in range(env.n_drones):
        if i in env.voxel.drone_positions and env.voxel.can_search(i):
            x, y, z = env.voxel.get_position(i)
            draw_search_frustum(ax, x, y, z)

    # Telemetry HUD
    cov_pct = env.get_coverage_rate() * 100.0
    hud_lines = [
        "─── DSSE-3D Airspace Telemetry ───",
        f"Step: {n_steps} | Map: {grid_size}×{grid_size} {map_type.capitalize()}",
        f"Area Coverage: {cov_pct:.1f}%",
        f"Controller: Cruise-at-Z=2 & Rooftop Flyover",
        f"Altitude Layers: Z ∈ [0, 5] (Z_scan = 1)",
    ]
    for i in range(env.n_drones):
        if i in env.voxel.drone_positions:
            z = env.voxel.get_altitude(i)
            hud_lines.append(f"  Drone {i}: Alt Z={z} (Operational)")

    fig.text(
        0.03, 0.94, "\n".join(hud_lines),
        fontsize=9.5, color="#38bdf8", fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#0f172a", edgecolor="#0284c7", alpha=0.9)
    )

    ax.view_init(elev=28, azim=-55)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"[✓] Saved 3D Visualizer snapshot to: {output_path}")
    plt.close()
    return output_path


def generate_all_snapshots(target_dirs):
    # 1. Render Urban Obstacles
    temp_obs = os.path.join(target_dirs[0], "figure_2_3_3d_visualizer_urban_obstacles.png")
    generate_3d_snapshot("obstacles", grid_size=25, n_steps=40, output_path=temp_obs)

    # 2. Render Standard Open Grid
    temp_std = os.path.join(target_dirs[0], "figure_2_3_3d_visualizer_standard.png")
    generate_3d_snapshot("standard", grid_size=25, n_steps=35, output_path=temp_std)

    # 3. Create side-by-side composite WITHOUT titles
    im_std = Image.open(temp_std)
    im_obs = Image.open(temp_obs)

    fig, axes = plt.subplots(1, 2, figsize=(24, 10.5), facecolor="#141923")
    axes[0].imshow(im_std)
    axes[0].axis("off")
    axes[1].imshow(im_obs)
    axes[1].axis("off")
    plt.tight_layout(pad=0.5)

    temp_comp = os.path.join(target_dirs[0], "figure_2_3_3d_visualizer_composite.png")
    fig.savefig(temp_comp, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[✓] Saved Composite 3D Visualizer snapshot to: {temp_comp}")

    # Copy to all target directories
    import shutil
    for td in target_dirs[1:]:
        os.makedirs(td, exist_ok=True)
        shutil.copy2(temp_obs, os.path.join(td, "figure_2_3_3d_visualizer_urban_obstacles.png"))
        shutil.copy2(temp_std, os.path.join(td, "figure_2_3_3d_visualizer_standard.png"))
        shutil.copy2(temp_comp, os.path.join(td, "figure_2_3_3d_visualizer_composite.png"))
        print(f"[✓] Replicated snapshots to: {td}")


def main():
    target_dirs = [
        os.path.expanduser("~/Desktop/Results"),
        os.path.expanduser("~/Desktop/Results/3D-final"),
        os.path.expanduser("~/Desktop/Results/3D-final-v3"),
    ]
    generate_all_snapshots(target_dirs)
    print("\n[✓] All Figure 2.3 snapshots (standard, urban_obstacles, composite) generated successfully without in-image titles!")


if __name__ == "__main__":
    main()
