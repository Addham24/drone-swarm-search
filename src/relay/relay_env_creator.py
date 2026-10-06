"""
Swarm Communication Mesh Relay (FANET) Environment Creator
===========================================================
Wraps DSSE Coverage parallel environment to evaluate Ad-Hoc Network Mesh Connectivity.

Network Physics & Graph Topology:
  - Base Station B: Fixed at (0, 0)
  - Remote Zone R: Fixed at (grid_size - 1, grid_size - 1)
  - Radio Link Range: R_comm = 6.5 cells (Euclidean distance)
  - Line-of-Sight (LOS) Obstacle Attenuation: R_comm_blocked = 3.25 cells if line intersects building obstacle
  - Link Connected L(t) = 1 if graph path exists B -> D_i -> ... -> R
"""

import os
import sys
import math
import collections
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper


def is_los_blocked(p1, p2, obstacle_mask):
    """Bresenham's line algorithm to check if line of sight between p1 and p2 intersects obstacle cells."""
    if obstacle_mask is None:
        return False
    x1, y1 = p1
    x2, y2 = p2
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    x, y = int(x1), int(y1)
    n = 1 + dx + dy
    x_inc = 1 if x2 > x1 else -1
    y_inc = 1 if y2 > y1 else -1
    error = dx - dy
    dx *= 2
    dy *= 2

    grid_h, grid_w = obstacle_mask.shape

    for _ in range(n):
        if 0 <= y < grid_h and 0 <= x < grid_w:
            if obstacle_mask[y, x] == 1:
                return True
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return False


def check_relay_connectivity(drone_positions, grid_size, R_comm=6.5, obstacle_mask=None):
    """
    Checks graph connectivity from Base Station (0,0) to Remote Zone (grid_size-1, grid_size-1).
    Returns (is_connected: bool, active_nodes: int).
    """
    base = (0, 0)
    remote = (grid_size - 1, grid_size - 1)
    nodes = [base] + list(drone_positions) + [remote]
    n = len(nodes)
    adj = collections.defaultdict(list)

    for i in range(n):
        for j in range(i + 1, n):
            p1, p2 = nodes[i], nodes[j]
            dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            max_range = R_comm
            if obstacle_mask is not None and is_los_blocked(p1, p2, obstacle_mask):
                max_range = R_comm / 2.0  # Attenuation under obstacle LOS block

            if dist <= max_range:
                adj[i].append(j)
                adj[j].append(i)

    # BFS from Base (node 0) to Remote (node n-1)
    visited = set([0])
    queue = collections.deque([0])
    target_node = n - 1

    while queue:
        curr = queue.popleft()
        if curr == target_node:
            return True, len(visited) - 1  # Connected!
        for neighbor in adj[curr]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return False, len(visited) - 1


class RelayEnvironmentWrapper(BaseParallelWrapper):
    """Wraps Coverage environment and computes FANET Communication Mesh connectivity."""
    def __init__(self, env, grid_size=25, R_comm=6.5, obstacle_mask=None):
        super().__init__(env)
        self.grid_size = grid_size
        self.R_comm = R_comm
        self.obstacle_mask = obstacle_mask
        self.uptime_steps = 0
        self.total_steps = 0
        self.link_breaks = 0
        self.repair_steps = []
        self.current_break_duration = 0
        self.was_connected = False

    def reset(self, **kwargs):
        obs, infos = self.env.reset(**kwargs)
        self.uptime_steps = 0
        self.total_steps = 0
        self.link_breaks = 0
        self.repair_steps = []
        self.current_break_duration = 0
        self.was_connected = False
        return obs, infos

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.total_steps += 1

        # Extract active drone positions
        base_env = self.env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'unwrapped'):
            base_env = base_env.unwrapped

        drone_positions = []
        if hasattr(base_env, 'agents_positions'):
            drone_positions = base_env.agents_positions

        is_connected, active_nodes = check_relay_connectivity(
            drone_positions, self.grid_size, R_comm=self.R_comm, obstacle_mask=self.obstacle_mask
        )

        if is_connected:
            self.uptime_steps += 1
            if not self.was_connected and self.current_break_duration > 0:
                self.repair_steps.append(self.current_break_duration)
                self.current_break_duration = 0
            self.was_connected = True
        else:
            if self.was_connected:
                self.link_breaks += 1
            self.current_break_duration += 1
            self.was_connected = False

        # Inject relay metrics into infos
        uptime_pct = (self.uptime_steps / self.total_steps) * 100.0 if self.total_steps > 0 else 0.0
        for agent in infos:
            if isinstance(infos[agent], dict):
                infos[agent] = dict(infos[agent])
                infos[agent]["relay_connected"] = is_connected
                infos[agent]["relay_uptime"] = uptime_pct

        return obs, rewards, terminations, truncations, infos


def make_relay_env(
    grid_size=25,
    drone_amount=4,
    timestep_limit=750,
    max_battery=125,
    fault_prob=0.0005,
    is_self_heal=False,
    obstacle_mask=None
):
    """Constructs the Swarm Communication Mesh Relay environment stack."""
    if grid_size == 25:
        prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy" if obstacle_mask is not None else "uniform_matrix_25.npy")
    else:
        prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy" if obstacle_mask is not None else "uniform_matrix_50.npy")

    env = CoverageDroneSwarmSearch(
        timestep_limit=timestep_limit,
        drone_amount=drone_amount,
        prob_matrix_path=prob_path
    )
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(
        env,
        max_battery=max_battery,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=fault_prob
    )

    if not is_self_heal:
        env.COMPENSATION_BONUS = 0.0
        env.COMPENSATION_PENALTY = 0.0
        env.COMPENSATION_HORIZON = 0
    else:
        env.COMPENSATION_BONUS = 2.5
        env.COMPENSATION_PENALTY = -0.5
        env.COMPENSATION_HORIZON = 100

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)

    if obstacle_mask is not None:
        from obstacle_airspace_wrapper import ObstacleAirspaceWrapper
        env = ObstacleAirspaceWrapper(env, obstacle_mask)

    env = RelayEnvironmentWrapper(env, grid_size=grid_size, R_comm=6.5, obstacle_mask=obstacle_mask)
    return env
