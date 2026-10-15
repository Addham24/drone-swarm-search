"""
City Grid Environment Generator (25x25 and 50x50)
==================================================
Generates realistic, aligned urban cityscapes with structured street networks,
boulevards, cross avenues, and city blocks of varying building heights.

Urban Planning Architecture:
  - Road Grid:
      * Major Boulevards (3-4 cells wide) running through the center.
      * Cross Avenues and perimeter streets running across the map.
      * Central open plazas / civic squares.
  - Aligned City Blocks:
      * Clearly defined rectangular building blocks separated by streets.
      * Mixed heights within blocks:
          Z=1: Low-rise residential / commercial shops
          Z=2: Mid-rise corporate / residential complexes
          Z=3: High-rise towers & skyscrapers on major street corners
  - Airspace & Maneuver Space:
      * ~13% - 16% building density, leaving ~84% - 87% wide open streets and plazas.
      * Base station [0:3, 0:3] (25x25) or [0:5, 0:5] (50x50) completely clear.
      * 100% BFS reachability guaranteed across all road corridors.

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


def generate_city_grid_25():
    """
    Generate 25x25 Aligned City Grid:
      - Continuous road network: Rows 0-2, 6-7, 11-14, 18-19, 23-24.
      - Continuous road network: Cols 0-2, 5-6, 11-14, 18-19, 23-24.
      - 12 clearly aligned 2x3 city blocks with stratified heights.
    """
    grid = np.zeros((25, 25), dtype=np.int32)

    # 12 cleanly defined city blocks (each 2x3)
    # NW Quadrant (Rows 3-5, 8-10; Cols 2-5, 7-10)
    grid[3:5, 2:5] = 1   # Low residential
    grid[8:10, 2:5] = 2  # Mid-rise block
    grid[3:5, 7:10] = 2  # Commercial
    grid[4:5, 9:10] = 3  # Corner tower
    grid[8:10, 7:10] = 1 # Low depot

    # NE Quadrant (Rows 3-5, 8-10; Cols 15-18, 20-23)
    grid[3:5, 15:18] = 2 # Mid-rise
    grid[3:5, 17:18] = 3 # Tower
    grid[8:10, 15:18] = 1 # Low shops
    grid[3:5, 20:23] = 1 # Low residential
    grid[8:10, 20:23] = 3 # High-rise tower

    # SW Quadrant (Rows 15-17, 20-22; Cols 2-5, 7-10)
    grid[15:17, 2:5] = 2 # Mid-rise
    grid[20:22, 2:5] = 1 # Low-rise
    grid[15:17, 7:10] = 3 # Tower
    grid[20:22, 7:10] = 2 # Mid-rise

    # SE Quadrant (Rows 15-17, 20-22; Cols 15-18, 20-23)
    grid[15:17, 15:18] = 3 # High-rise tower
    grid[20:22, 15:18] = 2 # Mid-rise
    grid[15:17, 20:23] = 2 # Commercial
    grid[20:22, 20:23] = 1 # Low-rise

    assert np.all(grid[0:3, 0:3] == 0), "Base station area must be clear!"
    return grid


def generate_city_grid_50():
    """
    Generate 50x50 Aligned City Grid:
      - Central Grand Boulevard: X in [23:27] (4 cells wide)
      - Central Grand Parkway: Y in [23:27] (4 cells wide)
      - Perimeter Avenues: X in [0:3], [47:50], Y in [0:3], [47:50]
      - 36 cleanly aligned 3x3 city blocks with stratified heights.
    """
    grid = np.zeros((50, 50), dtype=np.int32)

    # Blocks in NW:
    nw_blocks = [
        ((4, 7), (4, 7), 1),
        ((4, 7), (11, 14), 2),
        ((4, 7), (18, 21), 3),
        ((11, 14), (4, 7), 2),
        ((11, 14), (11, 14), 1),
        ((11, 14), (18, 21), 2),
        ((18, 21), (4, 7), 3),
        ((18, 21), (11, 14), 2),
        ((18, 21), (18, 21), 1),
    ]

    # Blocks in NE:
    ne_blocks = [
        ((4, 7), (29, 32), 2),
        ((4, 7), (36, 39), 1),
        ((4, 7), (43, 46), 2),
        ((11, 14), (29, 32), 3),
        ((11, 14), (36, 39), 2),
        ((11, 14), (43, 46), 1),
        ((18, 21), (29, 32), 1),
        ((18, 21), (36, 39), 3),
        ((18, 21), (43, 46), 2),
    ]

    # Blocks in SW:
    sw_blocks = [
        ((29, 32), (4, 7), 1),
        ((29, 32), (11, 14), 3),
        ((29, 32), (18, 21), 2),
        ((36, 39), (4, 7), 2),
        ((36, 39), (11, 14), 1),
        ((36, 39), (18, 21), 3),
        ((43, 46), (4, 7), 2),
        ((43, 46), (11, 14), 2),
        ((43, 46), (18, 21), 1),
    ]

    # Blocks in SE:
    se_blocks = [
        ((29, 32), (29, 32), 3),
        ((29, 32), (36, 39), 2),
        ((29, 32), (43, 46), 1),
        ((36, 39), (29, 32), 2),
        ((36, 39), (36, 39), 1),
        ((36, 39), (43, 46), 3),
        ((43, 46), (29, 32), 1),
        ((43, 46), (36, 39), 2),
        ((43, 46), (43, 46), 2),
    ]

    all_blocks = nw_blocks + ne_blocks + sw_blocks + se_blocks
    for (ys, ye), (xs, xe), h in all_blocks:
        grid[ys:ye, xs:xe] = h

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
    print("  GENERATING ALIGNED URBAN CITY GRID ENVIRONMENTS")
    print("=" * 65)

    # 1. 25x25 Map
    grid_25 = generate_city_grid_25()
    ok_25, unreach_25 = validate_bfs_reachability(grid_25)
    assert ok_25, f"25x25 map has {unreach_25} unreachable cells!"

    n_obs_25 = int((grid_25 > 0).sum())
    n_free_25 = 625 - n_obs_25
    prob_25 = generate_prob_matrix(grid_25)

    path_mask_25 = os.path.join(SRC_DIR, "obstacle_skyline_25.npy")
    path_prob_25 = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")
    np.save(path_mask_25, grid_25)
    np.save(path_prob_25, prob_25)

    print(f"\n[✓] 25x25 Aligned City Grid Generated:")
    print(f"    File: {path_mask_25}")
    print(f"    Building Cells: {n_obs_25} ({n_obs_25/625*100:.1f}%) | Free Road Cells: {n_free_25} ({n_free_25/625*100:.1f}%)")
    print(f"    Heights: Z=1: {(grid_25==1).sum()}, Z=2: {(grid_25==2).sum()}, Z=3: {(grid_25==3).sum()}")
    print(f"    BFS Reachability: 100.0% (0 unreachable)")

    # 2. 50x50 Map
    grid_50 = generate_city_grid_50()
    ok_50, unreach_50 = validate_bfs_reachability(grid_50)
    assert ok_50, f"50x50 map has {unreach_50} unreachable cells!"

    n_obs_50 = int((grid_50 > 0).sum())
    n_free_50 = 2500 - n_obs_50
    prob_50 = generate_prob_matrix(grid_50)

    path_mask_50 = os.path.join(SRC_DIR, "obstacle_skyline_50.npy")
    path_prob_50 = os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy")
    np.save(path_mask_50, grid_50)
    np.save(path_prob_50, prob_50)

    print(f"\n[✓] 50x50 Aligned City Grid Generated:")
    print(f"    File: {path_mask_50}")
    print(f"    Building Cells: {n_obs_50} ({n_obs_50/2500*100:.1f}%) | Free Road Cells: {n_free_50} ({n_free_50/2500*100:.1f}%)")
    print(f"    Heights: Z=1: {(grid_50==1).sum()}, Z=2: {(grid_50==2).sum()}, Z=3: {(grid_50==3).sum()}")
    print(f"    BFS Reachability: 100.0% (0 unreachable)")
    print("\n[✓] Successfully generated and verified all aligned city obstacle environments.")


if __name__ == "__main__":
    main()
