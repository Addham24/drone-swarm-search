"""
DSSE-3D Interactive Matplotlib Visualizer
==========================================
Real-time 3D visualization of drone swarm operations in the DSSE-3D voxel
workspace using Matplotlib's mplot3d toolkit.

Features:
    - Interactive 3D camera (mouse orbit, pan, zoom)
    - Semi-transparent 3D obstacle prisms (bar3d)
    - Multi-colored 3D trajectory ribbons per drone
    - Dotted altitude drop-lines (drone → ground projection)
    - Sensor frustum spotlight on SEARCH at Z=1
    - Ground-plane heatmap (probability matrix / coverage grid)
    - Real-time telemetry HUD (battery, altitude, coverage/tracking status)
    - GIF/MP4 export support (--export_gif / --export_mp4)

Dual Mode Support:
    --task tracking:  Moving target sphere + probability density heatmap
    --task coverage:  Color-coded floor grid (unvisited vs. covered)

Usage:
    python -m dsse_3d.visualizer_3d --task tracking --model rspo_selfheal
    python -m dsse_3d.visualizer_3d --task coverage --model rspo_selfheal
    python -m dsse_3d.visualizer_3d --task tracking --model heuristic --grid_size 25

Reference:
    Visual style matches Yan et al. (IEEE T-IV 2022) and Panerati et al.
    (gym-pybullet-drones, IEEE RA-L 2021).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
if sys.platform == "darwin":
    try:
        matplotlib.use("macosx")
    except Exception:
        pass
else:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import torch as th

# Ensure src directory is on path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)


# ──────────────────────────────────────────────────────────────────────────────
# Color Palette (Publication-grade)
# ──────────────────────────────────────────────────────────────────────────────

DRONE_COLORS = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
]
DRONE_COLORS_RGBA = [mcolors.to_rgba(c, alpha=0.9) for c in DRONE_COLORS]
TRAIL_COLORS_RGBA = [mcolors.to_rgba(c, alpha=0.35) for c in DRONE_COLORS]

OBSTACLE_COLOR = (0.25, 0.25, 0.28, 0.45)   # Dark charcoal, semi-transparent
GROUND_UNVISITED = (0.12, 0.12, 0.15, 0.8)  # Dark slate
GROUND_COVERED = (0.18, 0.75, 0.45, 0.6)    # Emerald green
GROUND_TARGET = (0.9, 0.2, 0.15, 0.85)      # Red-orange
STATION_COLOR = (0.2, 0.85, 0.45, 0.9)      # Bright emerald

FRUSTUM_COLOR = (1.0, 1.0, 0.3, 0.15)       # Soft yellow, very translucent
DROPLINE_COLOR = (0.5, 0.5, 0.5, 0.4)       # Grey dashed


# ──────────────────────────────────────────────────────────────────────────────
# 3D Drawing Utilities
# ──────────────────────────────────────────────────────────────────────────────

def draw_ground_grid(ax, grid_size, prob_matrix=None, covered_cells=None, target_pos=None):
    """
    Draw the ground plane (Z=0) as a color-coded grid.

    For tracking: colored by probability density.
    For coverage: colored by visited/unvisited status.
    """
    for y in range(grid_size):
        for x in range(grid_size):
            # Determine cell color
            if x == 0 and y == 0:
                color = STATION_COLOR
            elif target_pos is not None and (x, y) == target_pos:
                color = GROUND_TARGET
            elif covered_cells is not None and (x, y) in covered_cells:
                color = GROUND_COVERED
            elif prob_matrix is not None:
                p = prob_matrix[y, x]
                max_p = prob_matrix.max() if prob_matrix.max() > 0 else 1.0
                intensity = p / max_p
                # Blue-to-red colormap
                color = (0.1 + 0.8 * intensity, 0.1 + 0.2 * (1 - intensity), 0.15, 0.6)
            else:
                color = GROUND_UNVISITED

            # Draw flat tile at Z=0
            verts = [
                [x, y, 0],
                [x + 1, y, 0],
                [x + 1, y + 1, 0],
                [x, y + 1, 0],
            ]
            ax.add_collection3d(Poly3DCollection(
                [verts], alpha=color[3], facecolors=[color[:3]], edgecolors=[(0.3, 0.3, 0.3, 0.2)],
                linewidths=0.3
            ))


def draw_obstacles_3d(ax, obstacle_heights):
    """
    Render 3D obstacle prisms as semi-transparent blocks.

    Args:
        obstacle_heights: Dict of (x, y) -> height.
    """
    for (ox, oy), h in obstacle_heights.items():
        # Draw a 3D bar (semi-transparent rectangular prism)
        ax.bar3d(
            ox, oy, 0,    # Base corner
            1, 1, h,      # Width, depth, height
            color=OBSTACLE_COLOR[:3],
            alpha=OBSTACLE_COLOR[3],
            edgecolor=(0.4, 0.4, 0.45, 0.5),
            linewidth=0.5,
        )


def draw_target_3d(ax, target_pos):
    """
    Render the moving target person in 3D as a glowing red avatar marker on the ground.
    """
    if target_pos is None:
        return
    tx, ty = target_pos
    cx, cy = tx + 0.5, ty + 0.5
    # 1. Ground target star marker
    ax.scatter([cx], [cy], [0.05], color='#ff2222', edgecolor='#ffff00', linewidth=1.2, s=200, marker='*', zorder=10)
    # 2. 3D Person Target Sphere / Marker at Z=0.4
    ax.scatter([cx], [cy], [0.4], color='#ff0033', edgecolor='white', linewidth=1.5, s=140, marker='o', zorder=11)
    # 3. Label above target
    ax.text(cx, cy, 0.9, "TARGET", color='#ff3333', fontsize=9.5, fontweight='bold', ha='center', zorder=12)



def draw_drones_3d(ax, voxel_grid, alive_mask=None):
    """
    Draw drone markers at their 3D positions with altitude drop-lines.
    """
    for i in range(voxel_grid.n_drones):
        if i not in voxel_grid.drone_positions:
            continue
        if alive_mask is not None and not alive_mask[i]:
            continue

        x, y, z = voxel_grid.get_position(i)
        cx, cy, cz = x + 0.5, y + 0.5, z  # Center of cell

        # Drone marker (large sphere)
        ax.scatter(
            [cx], [cy], [cz],
            s=120, c=[DRONE_COLORS[i % len(DRONE_COLORS)]],
            marker='o', edgecolors='white', linewidths=1.2,
            zorder=10, depthshade=True,
            label=f"Drone {i}" if cz > 0 else None,
        )

        # Altitude drop-line (dotted vertical line to ground)
        if cz > 0:
            ax.plot(
                [cx, cx], [cy, cy], [0, cz],
                linestyle=':', linewidth=0.8,
                color=DROPLINE_COLOR[:3], alpha=DROPLINE_COLOR[3],
            )

        # Ground projection marker (small X)
        ax.scatter(
            [cx], [cy], [0],
            s=25, c=[DRONE_COLORS[i % len(DRONE_COLORS)]],
            marker='x', alpha=0.3, linewidths=0.8,
        )


def draw_trajectories_3d(ax, trajectory_history, max_trail=50):
    """
    Draw 3D trajectory ribbons trailing each drone.
    """
    for drone_idx, trail in trajectory_history.items():
        if len(trail) < 2:
            continue

        # Only show last max_trail points
        recent = trail[-max_trail:]
        xs = [p[0] + 0.5 for p in recent]
        ys = [p[1] + 0.5 for p in recent]
        zs = [p[2] for p in recent]

        color = TRAIL_COLORS_RGBA[drone_idx % len(TRAIL_COLORS_RGBA)]
        ax.plot(
            xs, ys, zs,
            linewidth=1.5, color=color[:3], alpha=color[3],
        )


def draw_search_frustum(ax, x, y, z):
    """
    Draw a semi-transparent downward frustum cone from drone to ground,
    indicating an active SEARCH action.
    """
    if z <= 0:
        return

    cx, cy = x + 0.5, y + 0.5

    # Simple frustum: lines from drone to corners of ground cell
    for dx, dy in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        ax.plot(
            [cx, x + dx], [cy, y + dy], [z, 0],
            linewidth=0.6, color=FRUSTUM_COLOR[:3], alpha=FRUSTUM_COLOR[3],
        )

    # Highlight ground cell
    verts = [
        [x, y, 0.01],
        [x + 1, y, 0.01],
        [x + 1, y + 1, 0.01],
        [x, y + 1, 0.01],
    ]
    ax.add_collection3d(Poly3DCollection(
        [verts], alpha=0.3, facecolors=[(1.0, 1.0, 0.3)],
        edgecolors=[(1.0, 1.0, 0.3, 0.5)], linewidths=1.0
    ))


# ──────────────────────────────────────────────────────────────────────────────
# Main Visualizer Class
# ──────────────────────────────────────────────────────────────────────────────

class DSSE3DVisualizer:
    """
    Interactive 3D real-time visualizer for DSSE-3D environments.

    Args:
        env_3d: A DSSE3DAdapter-wrapped environment.
        task: "tracking" or "coverage".
        max_trail: Maximum trajectory trail length to render.
        update_interval: Matplotlib pause interval in seconds.
    """

    def __init__(self, env_3d, task="tracking", max_trail=80, update_interval=0.03):
        self.env = env_3d
        self.task = task
        self.max_trail = max_trail
        self.update_interval = update_interval

        # Frames for export
        self.frames = []
        self.export_path = None

    def run(self, policy_fn, n_steps=750, export_path=None):
        """
        Run the live 3D visualization loop.

        Args:
            policy_fn: Policy function (obs_dict, agents) -> actions_dict.
                       Must already be wrapped for 3D (use wrap_*_policy_for_3d).
            n_steps: Maximum timesteps to render.
            export_path: If set, save frames for GIF/MP4 export.
        """
        self.export_path = export_path

        plt.ion()
        fig = plt.figure(figsize=(14, 10), facecolor='#0a0a0f')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0f')

        obs, infos = self.env.reset()

        grid_size = self.env.grid_size
        z_max = self.env.z_max

        # Track coverage cells
        covered_cells = set()

        # Telemetry text objects
        telemetry_text = fig.text(
            0.02, 0.95, "", fontsize=9, color='white',
            fontfamily='monospace', verticalalignment='top',
            transform=fig.transFigure,
        )

        for step in range(n_steps):
            ax.cla()

            # Set 3D axes
            ax.set_xlim(0, grid_size)
            ax.set_ylim(0, grid_size)
            ax.set_zlim(0, z_max + 1)
            ax.set_xlabel("X", fontsize=10, color='white', labelpad=8)
            ax.set_ylabel("Y", fontsize=10, color='white', labelpad=8)
            ax.set_zlabel("Altitude (Z)", fontsize=10, color='white', labelpad=8)

            title = f"DSSE-3D {'Tracking' if self.task == 'tracking' else 'Coverage'} — Step {step}"
            ax.set_title(title, fontsize=13, fontweight='bold', color='white', pad=15)

            # Style axes
            ax.tick_params(colors='grey', labelsize=7)
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('grey')
            ax.yaxis.pane.set_edgecolor('grey')
            ax.zaxis.pane.set_edgecolor('grey')

            # Extract state
            prob_matrix = None
            target_pos = None

            if obs:
                first_agent = list(obs.keys())[0]
                if isinstance(obs[first_agent], tuple) and len(obs[first_agent]) == 2:
                    prob_matrix = obs[first_agent][1]

            # Get target position for tracking mode
            if self.task == "tracking":
                base_env = self.env._get_base_env()
                if hasattr(base_env, 'persons_set') and base_env.persons_set:
                    p = list(base_env.persons_set)[0]
                    if hasattr(p, 'get_position'):
                        pos = p.get_position()
                        # DSSE Person get_position() returns (y, x) = (row, col)
                        # 3D plot expects (x, y) = (col, row)
                        target_pos = (int(pos[1]), int(pos[0]))
                    elif hasattr(p, 'x') and hasattr(p, 'y'):
                        target_pos = (int(p.x), int(p.y))
                elif hasattr(base_env, 'person_positions') and base_env.person_positions:
                    tp = base_env.person_positions[0]
                    target_pos = (int(tp[1]), int(tp[0]))
                elif hasattr(base_env, 'persons') and base_env.persons:
                    tp = base_env.persons[0]
                    if hasattr(tp, 'get_position'):
                        pos = tp.get_position()
                        target_pos = (int(pos[1]), int(pos[0]))
                    elif hasattr(tp, 'x') and hasattr(tp, 'y'):
                        target_pos = (int(tp.x), int(tp.y))

            # Track coverage
            if self.task == "coverage":
                base_env = self.env._get_base_env()
                if hasattr(base_env, 'seen_states'):
                    covered_cells = set(base_env.seen_states)

            # Build obstacle heights dict
            obstacle_heights = self.env.voxel._obstacle_heights

            # Draw scene
            draw_ground_grid(ax, grid_size, prob_matrix, covered_cells, target_pos)
            draw_obstacles_3d(ax, obstacle_heights)
            draw_target_3d(ax, target_pos)
            draw_trajectories_3d(ax, self.env.trajectory_history, self.max_trail)

            # Get alive status
            alive_mask = [True] * self.env.n_drones
            batt_env = self.env.env
            while hasattr(batt_env, 'env'):
                if hasattr(batt_env, 'alive'):
                    for i, agent in enumerate(self.env.possible_agents):
                        alive_mask[i] = batt_env.alive.get(agent, True)
                    break
                batt_env = batt_env.env

            draw_drones_3d(ax, self.env.voxel, alive_mask)

            # Draw search frustums for drones at scan altitude
            for i in range(self.env.n_drones):
                if i in self.env.voxel.drone_positions and self.env.voxel.can_search(i):
                    x, y, z = self.env.voxel.get_position(i)
                    if alive_mask[i]:
                        draw_search_frustum(ax, x, y, z)

            # Draw target as a sphere (tracking mode)
            if target_pos is not None:
                ax.scatter(
                    [target_pos[0] + 0.5], [target_pos[1] + 0.5], [0.15],
                    s=200, c=['#ff4444'], marker='*',
                    edgecolors='yellow', linewidths=1.0,
                    zorder=15, label="Target",
                )

            # Build telemetry string
            telemetry_lines = [f"─── DSSE-3D Telemetry ───"]
            telemetry_lines.append(f"Step: {step}/{n_steps}")

            # Battery & altitude per drone
            for i, agent in enumerate(self.env.possible_agents):
                if i in self.env.voxel.drone_positions:
                    alt = self.env.voxel.get_altitude(i)
                    batt_pct = "N/A"
                    batt_wrapper = self.env.env
                    while hasattr(batt_wrapper, 'env'):
                        if hasattr(batt_wrapper, 'battery') and agent in batt_wrapper.battery:
                            batt_pct = f"{batt_wrapper.battery[agent] / batt_wrapper.max_battery * 100:.0f}%"
                            break
                        batt_wrapper = batt_wrapper.env
                    alive_str = "✓" if alive_mask[i] else "✗"
                    telemetry_lines.append(f"  D{i}: Z={alt}  Batt={batt_pct}  [{alive_str}]")

            if self.task == "coverage":
                total = grid_size * grid_size
                cov_pct = len(covered_cells) / total * 100 if total > 0 else 0
                telemetry_lines.append(f"Coverage: {cov_pct:.1f}% ({len(covered_cells)}/{total})")

            metrics = self.env.get_3d_metrics()
            telemetry_lines.append(f"Alt Changes: {metrics['altitude_changes']}")
            telemetry_lines.append(f"3D Collisions: {metrics['midair_collisions']}")

            telemetry_text.set_text("\n".join(telemetry_lines))

            # Camera angle
            ax.view_init(elev=25, azim=-60 + step * 0.15)

            plt.draw()
            plt.pause(self.update_interval)

            # Check for window close
            if not plt.fignum_exists(fig.number):
                break

            # Get actions from policy
            actions = policy_fn(obs, self.env.agents)
            obs, rewards, term, trunc, infos = self.env.step(actions)

            # Check target found (tracking)
            if self.task == "tracking":
                for agent_info in infos.values():
                    if isinstance(agent_info, dict) and agent_info.get("Found", False):
                        print(f"[✓] TARGET FOUND at step {step}!")
                        telemetry_lines.append(f">>> TARGET FOUND at step {step}! <<<")
                        telemetry_text.set_text("\n".join(telemetry_lines))
                        plt.draw()
                        plt.pause(3.0)
                        plt.ioff()
                        plt.show()
                        return

            if all(term.values()) or all(trunc.values()):
                break

        plt.ioff()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def make_heuristic_policy(grid_size=25):
    """
    Creates a greedy heuristic policy for demo/testing purposes.
    Moves toward highest-probability cells in the 2D observation matrix.
    """
    from DSSE.environment.constants import Actions

    DIRECTION_TO_ACTION = {
        (-1,  0): Actions.LEFT.value,
        ( 1,  0): Actions.RIGHT.value,
        ( 0, -1): Actions.UP.value,
        ( 0,  1): Actions.DOWN.value,
        (-1, -1): Actions.UP_LEFT.value,
        ( 1, -1): Actions.UP_RIGHT.value,
        (-1,  1): Actions.DOWN_LEFT.value,
        ( 1,  1): Actions.DOWN_RIGHT.value,
        ( 0,  0): Actions.SEARCH.value,
    }

    def policy_fn(obs_dict, agents, *args, **kwargs):
        actions = {}
        for agent in agents:
            if agent not in obs_dict:
                continue
            pos_vec, prob_matrix = obs_dict[agent]
            drone_y = int(round(pos_vec[0] * grid_size))
            drone_x = int(round(pos_vec[1] * grid_size))
            drone_y = max(0, min(grid_size - 1, drone_y))
            drone_x = max(0, min(grid_size - 1, drone_x))

            best_action = Actions.SEARCH.value
            best_prob = prob_matrix[drone_y, drone_x] if 0 <= drone_y < grid_size and 0 <= drone_x < grid_size else 0.0

            for (dx, dy), action_val in DIRECTION_TO_ACTION.items():
                if dx == 0 and dy == 0:
                    continue
                nx, ny = drone_x + dx, drone_y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    p = prob_matrix[ny, nx]
                    if p > best_prob + 1e-8:
                        best_prob = p
                        best_action = action_val

            actions[agent] = best_action
        return actions

    return policy_fn


def main():
    parser = argparse.ArgumentParser(description="DSSE-3D Interactive Visualizer")
    parser.add_argument("--task", type=str, default="tracking", choices=["tracking", "coverage"],
                        help="Task mode: 'tracking' or 'coverage'")
    parser.add_argument("--model", type=str, default="heuristic",
                        help="Model to visualize (e.g., rspo_selfheal, mappo_vanilla, heuristic)")
    parser.add_argument("--grid_size", type=int, default=25, help="Grid size (25 or 50)")
    parser.add_argument("--map_type", type=str, default="standard",
                        choices=["standard", "obstacles"], help="Map type")
    parser.add_argument("--max_steps", type=int, default=750, help="Max episode steps")
    parser.add_argument("--export_gif", type=str, default=None, help="Export frames to GIF path")
    parser.add_argument("--no_faults", action="store_true", help="Disable fault injection")
    args = parser.parse_args()

    from dsse_3d.env_creators import make_tracking_env_3d, make_coverage_env_3d
    from dsse_3d.policy_adapters import wrap_rllib_policy_for_3d, wrap_epymarl_policy_for_3d

    # Load obstacle mask if needed
    obstacle_mask = None
    if args.map_type == "obstacles":
        obs_path = os.path.join(SRC_DIR, f"obstacle_skyline_{args.grid_size}.npy")
        if os.path.exists(obs_path):
            obstacle_mask = np.load(obs_path)
            print(f"[✓] Loaded obstacle mask from {obs_path}")

    fault_prob = 0.0 if args.no_faults else 0.0005

    # Create 3D environment
    if args.task == "tracking":
        env = make_tracking_env_3d(
            grid_size=args.grid_size,
            fault_prob=fault_prob,
            is_self_heal="selfheal" in args.model,
            obstacle_mask=obstacle_mask,
        )
    else:
        env = make_coverage_env_3d(
            grid_size=args.grid_size,
            fault_prob=fault_prob,
            is_self_heal="selfheal" in args.model,
            obstacle_mask=obstacle_mask,
        )

    # Load policy
    if args.model == "heuristic":
        policy_fn = make_heuristic_policy(args.grid_size)
        # Wrap for 3D (strips altitude from obs before forwarding)
        policy_fn = wrap_rllib_policy_for_3d(policy_fn)
    else:
        # Load trained model checkpoint
        policy_fn = _load_trained_model(args.model, args.task, args.grid_size)

    # Run visualizer
    viz = DSSE3DVisualizer(env, task=args.task)
    viz.run(policy_fn, n_steps=args.max_steps, export_path=args.export_gif)


def _load_trained_model(model_name, task, grid_size):
    """Load a trained MARL policy and wrap it for 3D."""
    from dsse_3d.policy_adapters import wrap_rllib_policy_for_3d, wrap_epymarl_policy_for_3d

    # Import model loaders from existing evaluation scripts
    from tracking.evaluate_tracking_models import (
        load_rllib_policy, load_epymarl_policy, find_best_post_10m_checkpoint,
    )
    from train_rspo_v2_vanilla import RSPOModelV2
    from train_mappo_vanilla import CNNModel as MAPPOModelVanilla

    # Model registry — maps model names to checkpoint patterns and frameworks
    model_registry = {
        # RLlib models
        "rspo_vanilla": ("rllib", RSPOModelV2, f"ray_res/DSSE_{'Tracking' if task == 'tracking' else 'Coverage'}/*RSPO_V2_*Vanilla*/**/checkpoint_*"),
        "rspo_selfheal": ("rllib", RSPOModelV2, f"ray_res/DSSE_{'Tracking' if task == 'tracking' else 'Coverage'}/*RSPO_V2_*SelfHeal*/**/checkpoint_*"),
        "mappo_vanilla": ("rllib", MAPPOModelVanilla, f"ray_res/DSSE_{'Tracking' if task == 'tracking' else 'Coverage'}/*MAPPO_*Vanilla*/**/checkpoint_*"),
        "mappo_selfheal": ("rllib", MAPPOModelVanilla, f"ray_res/DSSE_{'Tracking' if task == 'tracking' else 'Coverage'}/*MAPPO_*SelfHeal*/**/checkpoint_*"),
        # EPyMARL models
        "qmix_vanilla": ("epymarl", None, f"results/models/*qmix_{'tracking_' if task == 'tracking' else ''}vanilla*/**/agent.th"),
        "qmix_selfheal": ("epymarl", None, f"results/models/*qmix_{'tracking_' if task == 'tracking' else ''}selfheal*/**/agent.th"),
        "maa2c_vanilla": ("epymarl", None, f"results/models/*maa2c_{'tracking_' if task == 'tracking' else ''}vanilla*/**/agent.th"),
        "maa2c_selfheal": ("epymarl", None, f"results/models/*maa2c_{'tracking_' if task == 'tracking' else ''}selfheal*/**/agent.th"),
        "coma_vanilla": ("epymarl", None, f"results/models/*coma_{'tracking_' if task == 'tracking' else ''}vanilla*/**/agent.th"),
        "coma_selfheal": ("epymarl", None, f"results/models/*coma_{'tracking_' if task == 'tracking' else ''}selfheal*/**/agent.th"),
        "iql_vanilla": ("epymarl", None, f"results/models/*iql_{'tracking_' if task == 'tracking' else ''}vanilla*/**/agent.th"),
        "iql_selfheal": ("epymarl", None, f"results/models/*iql_{'tracking_' if task == 'tracking' else ''}selfheal*/**/agent.th"),
    }

    if model_name not in model_registry:
        print(f"[!] Unknown model '{model_name}'. Using heuristic fallback.")
        from dsse_3d.visualizer_3d import make_heuristic_policy
        from dsse_3d.policy_adapters import wrap_rllib_policy_for_3d
        return wrap_rllib_policy_for_3d(make_heuristic_policy(grid_size))

    framework, model_cls, pattern = model_registry[model_name]
    full_pattern = os.path.join(SRC_DIR, pattern)

    ckpt_path = find_best_post_10m_checkpoint(full_pattern, framework, min_timesteps=10_000_000)

    if ckpt_path is None:
        print(f"[!] No checkpoint found for {model_name}. Using random policy.")
        def random_policy(obs_dict, agents, *a, **kw):
            return {agent: np.random.randint(0, 9) for agent in agents}
        return wrap_rllib_policy_for_3d(random_policy)

    print(f"[✓] Loading {model_name} from {ckpt_path}")

    if framework == "rllib":
        policy_fn = load_rllib_policy(model_cls, ckpt_path, model_name=model_name)
        return wrap_rllib_policy_for_3d(policy_fn)
    else:
        policy_fn = load_epymarl_policy(ckpt_path)
        return wrap_epymarl_policy_for_3d(policy_fn)


if __name__ == "__main__":
    main()
