"""
VoxelGrid: 3D Spatial State Manager for DSSE-3D
=================================================
Manages discrete 3D voxel coordinates (X, Y, Z) for a drone swarm operating
in a volumetric workspace above a 2D ground search grid.

Design Rationale:
    The 2D DSSE environments handle all ground-plane search logic (probability
    matrices, target drift, cell coverage). This VoxelGrid adds a vertical
    dimension Z ∈ [0, Z_max] on top, managing:
        - Per-drone altitude state and transitions
        - 3D obstacle prism extrusions (buildings from Z=0 to H_obs)
        - Altitude-dependent sensor field-of-view (FOV) footprint
        - Inter-drone vertical deconfliction
        - Search validity constraints (only at Z=1 scanning layer)

Coordinate Convention:
    (x, y) matches DSSE convention (column, row).
    z = 0: Ground level (charging station / target plane)
    z = 1: Low-altitude scanning layer (high-res FOV, SEARCH action valid)
    z = 2..Z_max-1: Mid-altitude transit layers
    z = Z_max: Altitude ceiling

Reference:
    Yan et al., "Multi-UAV 3D Target Search", IEEE T-IV, 2022.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set


class VoxelGrid:
    """
    Manages 3D spatial state for a swarm of drones over a 2D search grid.

    Args:
        grid_size: Side length N of the NxN ground grid.
        z_max: Maximum altitude layer (inclusive). Default 5 creates layers 0..5.
        n_drones: Number of drones in the swarm.
        scan_altitude: The altitude layer at which SEARCH actions are effective.
        obstacle_heights: Optional dict mapping (x, y) -> max obstacle height h,
            meaning voxels (x, y, z) for z in [1, h] are blocked.
    """

    # Altitude semantics
    Z_GROUND = 0       # Ground level: charging station, target plane
    Z_SCAN = 1         # Scanning altitude: high-resolution search FOV
    Z_TRANSIT_MIN = 2  # Start of transit altitude band
    Z_CEILING = 5      # Default altitude ceiling

    # FOV footprint radii by altitude (in ground cells)
    # Lower altitude = smaller but higher-resolution FOV
    # Higher altitude = wider but lower-resolution FOV
    FOV_RADIUS = {
        0: 0,  # On ground, no aerial FOV
        1: 0,  # Scanning: single cell directly below (high-res)
        2: 1,  # Transit: 3x3 neighborhood (low-res awareness)
        3: 2,  # Transit: 5x5 neighborhood
        4: 2,  # Transit: 5x5 neighborhood
        5: 3,  # High: 7x7 neighborhood (wide but inaccurate)
    }

    def __init__(
        self,
        grid_size: int = 25,
        z_max: int = 5,
        n_drones: int = 4,
        scan_altitude: int = 1,
        obstacle_heights: Optional[Dict[Tuple[int, int], int]] = None,
    ):
        self.grid_size = grid_size
        self.z_max = z_max
        self.n_drones = n_drones
        self.scan_altitude = scan_altitude

        # Per-drone 3D position: (x, y, z)
        self.drone_positions: Dict[int, Tuple[int, int, int]] = {}

        # 3D obstacle occupancy grid: True = blocked voxel
        self.obstacle_grid = np.zeros((grid_size, grid_size, z_max + 1), dtype=bool)

        # Build obstacle extrusions from 2D mask heights
        self._obstacle_heights = obstacle_heights or {}
        for (ox, oy), h in self._obstacle_heights.items():
            for z in range(1, min(h + 1, z_max + 1)):
                self.obstacle_grid[oy, ox, z] = True

        # Set of blocked (x, y) columns for quick ground-cell lookup
        self._obstacle_columns: Set[Tuple[int, int]] = set(self._obstacle_heights.keys())

    def reset(self, initial_positions_2d: List[Tuple[int, int]], initial_z: int = 1):
        """
        Reset all drones to given 2D positions at a specified altitude.

        Args:
            initial_positions_2d: List of (x, y) ground positions per drone.
            initial_z: Starting altitude for all drones (default: scan altitude 1).
        """
        self.drone_positions.clear()
        for i, (x, y) in enumerate(initial_positions_2d):
            z = initial_z
            # If spawn altitude is blocked, find next clear altitude
            while z <= self.z_max and self.obstacle_grid[y, x, z]:
                z += 1
            if z > self.z_max:
                z = initial_z  # Fallback: shouldn't happen with valid spawn
            self.drone_positions[i] = (x, y, z)

    def get_position(self, drone_idx: int) -> Tuple[int, int, int]:
        """Return (x, y, z) position of a drone."""
        return self.drone_positions[drone_idx]

    def get_altitude(self, drone_idx: int) -> int:
        """Return current altitude of a drone."""
        return self.drone_positions[drone_idx][2]

    def get_2d_position(self, drone_idx: int) -> Tuple[int, int]:
        """Return (x, y) ground projection of a drone."""
        x, y, _ = self.drone_positions[drone_idx]
        return (x, y)

    def apply_2d_action(self, drone_idx: int, new_x: int, new_y: int):
        """
        Update the horizontal (x, y) position of a drone, keeping altitude fixed.
        Validates against 3D obstacle collisions at the drone's current altitude.

        Args:
            drone_idx: Index of the drone.
            new_x: New x coordinate from 2D env.
            new_y: New y coordinate from 2D env.

        Returns:
            True if the move was valid, False if blocked by a 3D obstacle.
        """
        _, _, z = self.drone_positions[drone_idx]

        # Check 3D obstacle at (new_x, new_y, z)
        if self._is_blocked(new_x, new_y, z):
            # Stay in place — blocked by obstacle prism at this altitude
            return False

        self.drone_positions[drone_idx] = (new_x, new_y, z)
        return True

    def change_altitude(self, drone_idx: int, delta_z: int) -> bool:
        """
        Change a drone's altitude by delta_z (+1 = ascend, -1 = descend).
        Validates against altitude bounds and 3D obstacle occupancy.

        Args:
            drone_idx: Index of the drone.
            delta_z: Altitude change (-1, 0, or +1).

        Returns:
            True if altitude change succeeded, False if blocked.
        """
        x, y, z = self.drone_positions[drone_idx]
        new_z = z + delta_z

        # Altitude bounds
        if new_z < self.Z_GROUND or new_z > self.z_max:
            return False

        # Obstacle check at new altitude
        if self._is_blocked(x, y, new_z):
            return False

        self.drone_positions[drone_idx] = (x, y, new_z)
        return True

    def can_search(self, drone_idx: int) -> bool:
        """
        Check if a drone is at the scanning altitude and thus able to
        perform a valid SEARCH action on the ground cell below.
        """
        return self.get_altitude(drone_idx) == self.scan_altitude

    def get_fov_footprint(self, drone_idx: int) -> List[Tuple[int, int]]:
        """
        Compute the set of ground cells (x, y) visible from the drone's
        current 3D position, based on altitude-dependent FOV radius.

        Returns:
            List of (x, y) ground coordinates in the sensor footprint.
        """
        x, y, z = self.drone_positions[drone_idx]
        radius = self.FOV_RADIUS.get(z, 0)

        cells = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    cells.append((nx, ny))
        return cells

    def check_inter_drone_collision(self) -> List[Tuple[int, int]]:
        """
        Detect pairs of drones occupying the same 3D voxel.

        Returns:
            List of (drone_i, drone_j) collision pairs.
        """
        collisions = []
        positions = list(self.drone_positions.items())
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i][1] == positions[j][1]:
                    collisions.append((positions[i][0], positions[j][0]))
        return collisions

    def get_altitude_observation(self) -> np.ndarray:
        """
        Build a normalized altitude observation vector for all drones.

        Returns:
            Array of shape (n_drones,) with values in [0, 1] representing
            normalized altitude z / z_max.
        """
        alt_obs = np.zeros(self.n_drones, dtype=np.float32)
        for i in range(self.n_drones):
            if i in self.drone_positions:
                alt_obs[i] = self.drone_positions[i][2] / self.z_max
        return alt_obs

    def get_obstacle_height_map(self) -> np.ndarray:
        """
        Return a 2D array of obstacle heights for visualization.

        Returns:
            Array of shape (grid_size, grid_size) with integer heights.
        """
        height_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        for (ox, oy), h in self._obstacle_heights.items():
            height_map[oy, ox] = h
        return height_map

    def _is_blocked(self, x: int, y: int, z: int) -> bool:
        """Check if a voxel is occupied by an obstacle."""
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        if z < 0 or z > self.z_max:
            return True
        return bool(self.obstacle_grid[y, x, z])

    def to_state_dict(self) -> dict:
        """Serialize current state for logging/debugging."""
        return {
            "drone_positions": dict(self.drone_positions),
            "grid_size": self.grid_size,
            "z_max": self.z_max,
        }
