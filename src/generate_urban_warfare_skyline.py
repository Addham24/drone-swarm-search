"""
Urban Warfare Environment Generator (25x25 and 50x50)
=====================================================
Generates realistic, asymmetric urban warfare (MOUT) obstacle environments
for 2D and 3D drone swarm coverage benchmarks.

Design Features:
  - Spread-out, non-aligned layout simulating ruined compounds, barricades, and towers.
  - Multi-tier height distribution:
      Z=1: Low rubble, destroyed walls, blast barriers
      Z=2: Mid-rise fortified compounds, depots, residential ruins
      Z=3: High vantage towers, sniper perches, communication masts
  - Central open town square plaza + peripheral courtyards.
  - Guaranteed 100% BFS reachability from base station (0, 0).
  - Clean base station perimeter [0:3, 0:3] (25x25) or [0:5, 0:5] (50x50).
  - Open space for swarm coverage: ~80% - 83% free space.

Outputs:
  - obstacle_skyline_25.npy
  - obstacle_skyline_50.npy
  - obstacle_prob_matrix_25.npy
  - obstacle_prob_matrix_50.npy
"""

import os
from collections import deque
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_urban_warfare_25():
    """Generate 25x25 Urban Warfare obstacle map with heights in {1, 2, 3}."""
    grid = np.zeros((25, 25), dtype=np.int32)

    # 1. Northwest Quadrant
    # L-shaped compound (Heights 1 and 2)
    grid[4:7, 3:5] = 1
    grid[4:5, 5:7] = 2

    # Standalone sniper perch (Height 3)
    grid[8:10, 5:7] = 3

    # 2. Southwest Quadrant
    # Barricade / low wall (Height 1)
    grid[14:17, 2:4] = 1
    # Multi-level outpost (Heights 2 and 3)
    grid[19:22, 3:6] = 2
    grid[20:22, 5:7] = 3
    # Corner guard post (Height 1)
    grid[23:25, 2:4] = 1

    # 3. North-Central Sector
    # Commercial ruin (Heights 2 and 3)
    grid[4:7, 10:13] = 2
    grid[5:7, 13:15] = 3

    # Central Plaza: [10:15, 10:15] left open as a 5x5 town square!

    # 4. South-Central Sector
    # U-shaped market compound (Heights 1 and 2)
    grid[18:21, 9:11] = 1
    grid[20:22, 11:14] = 2
    grid[18:21, 14:16] = 1
    # Warehouse ruin (Height 1)
    grid[23:25, 10:13] = 1

    # 5. North-East Quadrant
    # High-rise command tower (Height 3) & wing (Height 2)
    grid[4:7, 18:20] = 3
    grid[6:9, 20:23] = 2
    # Staggered barrier (Height 1)
    grid[2:4, 18:21] = 1

    # 6. East-Central Sector
    # Heavy fortification (Height 3)
    grid[12:15, 18:21] = 3
    # Small bunker (Height 1)
    grid[11:13, 22:24] = 1

    # 7. South-East Quadrant
    # Staggered residential blocks (Heights 2 and 1)
    grid[17:20, 18:21] = 2
    grid[22:24, 18:20] = 1
    grid[19:22, 22:24] = 2

    # Base station protection
    assert np.all(grid[0:3, 0:3] == 0), "Base station area must be clear!"

    return grid


def generate_urban_warfare_50():
    """Generate 50x50 Urban Warfare obstacle map with heights in {1, 2, 3}."""
    grid = np.zeros((50, 50), dtype=np.int32)

    # 1. Base station area [0:5, 0:5] strictly free

    # 2. Quadrant 1 (North-West)
    grid[6:9, 5:8] = 1       # Rubble compound
    grid[6:8, 8:10] = 2
    grid[12:15, 4:7] = 3     # Sniper tower
    grid[17:20, 2:5] = 1     # Low barricade
    grid[15:18, 8:11] = 2    # Mid-rise depot
    grid[7:10, 18:21] = 2    # Commercial block
    grid[8:10, 21:23] = 3
    grid[12:15, 16:19] = 1   # Low wall
    grid[18:21, 14:17] = 2   # Residential ruin

    # 3. Quadrant 2 (North-East)
    grid[4:7, 26:29] = 1
    grid[5:8, 29:32] = 2
    grid[3:6, 36:39] = 2
    grid[4:6, 39:41] = 3     # North-east tower
    grid[9:12, 42:45] = 1
    grid[8:11, 27:30] = 3    # Vantage tower
    grid[12:15, 31:34] = 2
    grid[14:17, 38:42] = 1   # Long barrier
    grid[18:21, 26:29] = 2
    grid[17:20, 33:36] = 3   # High-rise ruin
    grid[19:22, 44:47] = 2

    # 4. Central Sector & Cross Corridors
    # Main Plaza at [22:28, 22:28] left completely open!
    grid[22:25, 17:20] = 3   # West gate tower
    grid[25:28, 18:21] = 2
    grid[22:25, 30:33] = 3   # East gate tower
    grid[25:28, 29:32] = 2

    # 5. Quadrant 3 (South-West)
    grid[29:32, 4:7] = 2
    grid[31:34, 7:10] = 3    # Watchtower
    grid[36:39, 3:6] = 1     # Rubble
    grid[42:45, 5:8] = 2
    grid[43:45, 8:11] = 1
    grid[29:32, 13:16] = 1   # Bunker
    grid[33:36, 17:20] = 2   # Mid block
    grid[39:42, 14:17] = 3   # Heavy structure
    grid[42:45, 18:21] = 1
    grid[46:48, 12:15] = 1

    # 6. South-Central Sector
    grid[32:35, 23:26] = 2
    grid[34:37, 26:29] = 1
    grid[40:43, 22:25] = 3   # South avenue tower
    grid[41:44, 25:28] = 2
    grid[46:49, 23:27] = 1   # South perimeter wall

    # 7. Quadrant 4 (South-East)
    grid[28:31, 35:38] = 1
    grid[30:33, 38:41] = 2
    grid[29:32, 43:46] = 3   # High-rise tower
    grid[35:38, 30:33] = 2
    grid[37:40, 34:37] = 3   # Industrial tower
    grid[35:38, 42:45] = 1   # Low depot
    grid[41:44, 32:35] = 1
    grid[40:43, 38:41] = 2   # Residential block
    grid[41:44, 44:47] = 2
    grid[45:48, 35:38] = 1
    grid[46:48, 40:43] = 3   # Corner outpost

    # Base station protection
    assert np.all(grid[0:4, 0:4] == 0), "Base station area must be clear!"

    return grid


def validate_bfs_reachability(grid):
    """Verify that all free cells are 100% reachable from base station (0, 0)."""
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    visited[0, 0] = True

    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                if not visited[ny, nx] and grid[ny, nx] == 0:
                    visited[ny, nx] = True
                    q.append((ny, nx))

    free_cells = (grid == 0)
    unreachable = free_cells & (~visited)
    n_unreachable = int(unreachable.sum())
    return (n_unreachable == 0), n_unreachable


def generate_prob_matrix(grid):
    """
    Generate normalized coverage probability matrix.
    Obstacle cells have 0.0 probability.
    Free cells have uniform probability summing to exactly 1.0.
    """
    h, w = grid.shape
    n_obstacles = int((grid > 0).sum())
    n_free = (h * w) - n_obstacles

    prob_matrix = np.full((h, w), 1.0 / n_free, dtype=np.float64)
    prob_matrix[grid > 0] = 0.0

    assert abs(prob_matrix.sum() - 1.0) < 1e-10, f"Sum: {prob_matrix.sum()} != 1.0"
    return prob_matrix


def main():
    print("=" * 65)
    print("  GENERATING ASYMMETRIC URBAN WARFARE ENVIRONMENTS")
    print("=" * 65)

    # 1. 25x25 Map
    grid_25 = generate_urban_warfare_25()
    ok_25, unreach_25 = validate_bfs_reachability(grid_25)
    assert ok_25, f"25x25 map has {unreach_25} unreachable cells!"

    n_obs_25 = int((grid_25 > 0).sum())
    n_free_25 = 625 - n_obs_25
    prob_25 = generate_prob_matrix(grid_25)

    path_mask_25 = os.path.join(SRC_DIR, "obstacle_skyline_25.npy")
    path_prob_25 = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")
    np.save(path_mask_25, grid_25)
    np.save(path_prob_25, prob_25)

    print(f"\n[✓] 25x25 Urban Warfare Map Generated:")
    print(f"    File: {path_mask_25}")
    print(f"    Obstacle Cells: {n_obs_25} ({n_obs_25/625*100:.1f}%) | Free Cells: {n_free_25} ({n_free_25/625*100:.1f}%)")
    print(f"    Heights: Z=1: {(grid_25==1).sum()}, Z=2: {(grid_25==2).sum()}, Z=3: {(grid_25==3).sum()}")
    print(f"    BFS Reachability: 100.0% (0 unreachable)")

    # 2. 50x50 Map
    grid_50 = generate_urban_warfare_50()
    ok_50, unreach_50 = validate_bfs_reachability(grid_50)
    assert ok_50, f"50x50 map has {unreach_50} unreachable cells!"

    n_obs_50 = int((grid_50 > 0).sum())
    n_free_50 = 2500 - n_obs_50
    prob_50 = generate_prob_matrix(grid_50)

    path_mask_50 = os.path.join(SRC_DIR, "obstacle_skyline_50.npy")
    path_prob_50 = os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy")
    np.save(path_mask_50, grid_50)
    np.save(path_prob_50, prob_50)

    print(f"\n[✓] 50x50 Urban Warfare Map Generated:")
    print(f"    File: {path_mask_50}")
    print(f"    Obstacle Cells: {n_obs_50} ({n_obs_50/2500*100:.1f}%) | Free Cells: {n_free_50} ({n_free_50/2500*100:.1f}%)")
    print(f"    Heights: Z=1: {(grid_50==1).sum()}, Z=2: {(grid_50==2).sum()}, Z=3: {(grid_50==3).sum()}")
    print(f"    BFS Reachability: 100.0% (0 unreachable)")
    print("\n[✓] Successfully generated and verified all urban warfare obstacle environments.")


if __name__ == "__main__":
    main()
