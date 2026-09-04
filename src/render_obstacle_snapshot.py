"""
Obstacle Airspace Snapshot Visualizer & Renderer
=================================================
Generates crisp, high-resolution visual snapshot figures of the obstacle airspace
(Layout 3 Scattered Skyline) for both 25x25 and 50x50 grid configurations.

Exports:
  - ~/Desktop/obstacle_airspace_visual_25x25.png
  - ~/Desktop/obstacle_airspace_visual_50x50.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from obstacle_airspace_wrapper import ObstacleAirspaceWrapper


def render_obstacle_snapshot(grid_size, obstacle_mask_path, prob_matrix_path, title, output_filename, steps=80, seed=42):
    np.random.seed(seed)
    obstacle_mask = np.load(obstacle_mask_path)
    
    # Grid state: 0 = unsearched, 1 = searched, -1 = obstacle, 2 = base station
    grid_render = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    # Mark obstacles (-1.0)
    grid_render[obstacle_mask == 1] = -1.0
    
    # Base station at (0,0)
    grid_render[0, 0] = 2.0
    
    # Simulate a 4-drone swarm search run to show realistic coverage & trajectories
    # 4 drones starting around base station
    drone_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    drone_trajectories = {i: [pos] for i, pos in enumerate(drone_positions)}
    
    # Move drones around avoiding obstacles
    for step in range(steps):
        for i, (y, x) in enumerate(drone_positions):
            # Pick a random valid move that doesn't hit obstacles
            candidates = []
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1), (0,0)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid_size and 0 <= nx < grid_size and obstacle_mask[ny, nx] == 0:
                    candidates.append((ny, nx))
            if candidates:
                # Prefer moving outward to expand coverage
                weights = [1.0 + (ny + nx)*0.1 for ny, nx in candidates]
                weights = np.array(weights) / sum(weights)
                next_pos = candidates[np.random.choice(len(candidates), p=weights)]
                drone_positions[i] = next_pos
                drone_trajectories[i].append(next_pos)
                if grid_render[next_pos] == 0.0:
                    grid_render[next_pos] = 1.0  # Searched cell
    
    # Base station stays marked
    grid_render[0, 0] = 2.0
    
    # --- Custom Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Color map:
    # -1.0 (Obstacle): Dark Charcoal [40, 40, 40]
    #  0.0 (Unsearched): Dark Blue/Gray [30, 40, 60]
    #  1.0 (Searched): Light Cyan/Silver [180, 220, 240]
    #  2.0 (Base Station): Bright Emerald Green [40, 220, 100]
    
    color_grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    for y in range(grid_size):
        for x in range(grid_size):
            val = grid_render[y, x]
            if val == -1.0:  # Obstacle (Charcoal)
                color_grid[y, x] = [0.15, 0.15, 0.18]
            elif val == 0.0:  # Unsearched
                color_grid[y, x] = [0.12, 0.16, 0.24]
            elif val == 1.0:  # Searched
                color_grid[y, x] = [0.75, 0.88, 0.95]
            elif val == 2.0:  # Base Station
                color_grid[y, x] = [0.15, 0.85, 0.40]
    
    ax.imshow(color_grid, origin="upper")
    
    # Draw subtle grid lines
    ax.set_xticks(np.arange(-0.5, grid_size, 1 if grid_size <= 25 else 5), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1 if grid_size <= 25 else 5), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.3, alpha=0.3)
    
    # Plot drone trajectories & current positions
    drone_colors = ["#FF3366", "#33CCFF", "#FFCC00", "#CC66FF"]  # Magenta, Cyan, Gold, Purple
    for i in range(4):
        traj = np.array(drone_trajectories[i])
        ax.plot(traj[:, 1], traj[:, 0], color=drone_colors[i], linewidth=2.0, alpha=0.8, linestyle="-")
        # Current drone position marker
        curr_y, curr_x = drone_positions[i]
        ax.plot(curr_x, curr_y, marker="o", markersize=10, color=drone_colors[i], markeredgecolor="white", markeredgewidth=1.5)
        ax.text(curr_x, curr_y, f"D{i+1}", color="black", fontsize=7, fontweight="bold", ha="center", va="center")
    
    # Title & Labels
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Grid Column (X)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Grid Row (Y)", fontsize=10, fontweight="bold")
    
    # Legend
    legend_elements = [
        Patch(facecolor=[0.15, 0.15, 0.18], edgecolor="white", label="Physical Building Obstacle (-1.0)"),
        Patch(facecolor=[0.15, 0.85, 0.40], edgecolor="white", label="Charging Base Station (0,0)"),
        Patch(facecolor=[0.75, 0.88, 0.95], edgecolor="white", label="Searched Airspace Cell"),
        Patch(facecolor=[0.12, 0.16, 0.24], edgecolor="white", label="Unsearched Frontier Cell"),
    ]
    for i in range(4):
        legend_elements.append(Patch(facecolor=drone_colors[i], edgecolor="white", label=f"Drone {i+1} Path"))
        
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.35, 1.0), fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    desktop_path = os.path.expanduser(f"~/Desktop/{output_filename}")
    plt.savefig(desktop_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved snapshot figure to: {desktop_path}")


if __name__ == "__main__":
    print("Generating Obstacle Airspace Visual Snapshots...")
    
    # 25x25 Obstacle Snapshot
    render_obstacle_snapshot(
        grid_size=25,
        obstacle_mask_path=os.path.join(SRC_DIR, "obstacle_skyline_25.npy"),
        prob_matrix_path=os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy"),
        title="25×25 Obstacle Airspace (Layout 3: Scattered High-Rise Skyline)",
        output_filename="obstacle_airspace_visual_25x25.png"
    )
    
    # 50x50 Obstacle Snapshot
    render_obstacle_snapshot(
        grid_size=50,
        obstacle_mask_path=os.path.join(SRC_DIR, "obstacle_skyline_50.npy"),
        prob_matrix_path=os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy"),
        title="50×50 Extended Obstacle Airspace (Layout 3: Scattered Skyline)",
        output_filename="obstacle_airspace_visual_50x50.png"
    )
    
    print("\nVisual snapshots generated successfully!")
