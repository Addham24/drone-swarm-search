"""
Matrix & Obstacle Layout Generator for 50x50 Grid & Layout 3 (Scattered Skyline)
=================================================================================
Generates:
  - uniform_matrix_50.npy          : 50x50 uniform search probability matrix
  - obstacle_skyline_25.npy        : 25x25 Layout 3 obstacle mask (4 x 2x2 building clusters)
  - obstacle_skyline_50.npy        : 50x50 Layout 3 obstacle mask (6 x 4x4 building clusters)
  - obstacle_prob_matrix_25.npy    : 25x25 uniform prob matrix with obstacle cells zeroed out
  - obstacle_prob_matrix_50.npy    : 50x50 uniform prob matrix with obstacle cells zeroed out
"""

import numpy as np
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def generate_uniform_matrix(size):
    """Generate a uniform probability matrix that sums to 1.0."""
    total_cells = size * size
    return np.full((size, size), 1.0 / total_cells, dtype=np.float64)


def generate_skyline_obstacle_mask_25():
    """
    Layout 3 (Scattered High-Rise Skyline) for 25x25 grid.
    
    Places 4 scattered 2x2 building blocks across the 4 grid quadrants.
    Base station (0,0) and adjacent cells are guaranteed obstacle-free.
    
    Quadrant layout (25x25):
      Q1 (top-left):     Building at [4:6, 4:6]
      Q2 (top-right):    Building at [3:5, 18:20]
      Q3 (bottom-left):  Building at [19:21, 5:7]
      Q4 (bottom-right): Building at [18:20, 20:22]
    
    Total obstacle cells: 16 out of 625 (2.6%)
    Accessible free cells: 609
    """
    mask = np.zeros((25, 25), dtype=np.int32)
    
    # Q1: Top-left quadrant building
    mask[4:6, 4:6] = 1
    
    # Q2: Top-right quadrant building
    mask[3:5, 18:20] = 1
    
    # Q3: Bottom-left quadrant building
    mask[19:21, 5:7] = 1
    
    # Q4: Bottom-right quadrant building
    mask[18:20, 20:22] = 1
    
    # Safety: ensure base station area is clear
    assert mask[0, 0] == 0, "Base station (0,0) must be obstacle-free!"
    assert mask[0, 1] == 0, "Adjacent cell (0,1) must be obstacle-free!"
    assert mask[1, 0] == 0, "Adjacent cell (1,0) must be obstacle-free!"
    assert mask[1, 1] == 0, "Adjacent cell (1,1) must be obstacle-free!"
    
    return mask


def generate_skyline_obstacle_mask_50():
    """
    Layout 3 (Scattered High-Rise Skyline) for 50x50 grid.
    
    Places 6 scattered 4x4 building blocks across the grid.
    Base station (0,0) and adjacent cells are guaranteed obstacle-free.
    
    Building placement (50x50):
      B1: [8:12,  8:12]    (upper-left sector)
      B2: [6:10,  36:40]   (upper-right sector)
      B3: [22:26, 22:26]   (center)
      B4: [38:42, 8:12]    (lower-left sector)
      B5: [35:39, 38:42]   (lower-right sector)
      B6: [14:18, 28:32]   (mid-right sector)
    
    Total obstacle cells: 96 out of 2500 (3.8%)
    Accessible free cells: 2404
    """
    mask = np.zeros((50, 50), dtype=np.int32)
    
    # B1: Upper-left sector
    mask[8:12, 8:12] = 1
    
    # B2: Upper-right sector
    mask[6:10, 36:40] = 1
    
    # B3: Center
    mask[22:26, 22:26] = 1
    
    # B4: Lower-left sector
    mask[38:42, 8:12] = 1
    
    # B5: Lower-right sector
    mask[35:39, 38:42] = 1
    
    # B6: Mid-right sector
    mask[14:18, 28:32] = 1
    
    # Safety: ensure base station area is clear
    assert mask[0, 0] == 0, "Base station (0,0) must be obstacle-free!"
    assert mask[0, 1] == 0, "Adjacent cell (0,1) must be obstacle-free!"
    assert mask[1, 0] == 0, "Adjacent cell (1,0) must be obstacle-free!"
    assert mask[1, 1] == 0, "Adjacent cell (1,1) must be obstacle-free!"
    
    return mask


def generate_obstacle_prob_matrix(size, obstacle_mask):
    """
    Generate a uniform probability matrix with obstacle cells zeroed out.
    Probability is redistributed uniformly across accessible free cells.
    """
    n_obstacles = int(obstacle_mask.sum())
    n_free = size * size - n_obstacles
    
    prob_matrix = np.full((size, size), 1.0 / n_free, dtype=np.float64)
    prob_matrix[obstacle_mask == 1] = 0.0
    
    # Verify probability sums to 1.0
    assert abs(prob_matrix.sum() - 1.0) < 1e-10, f"Probability sum = {prob_matrix.sum()}, expected 1.0"
    
    return prob_matrix


if __name__ == "__main__":
    # 1. Generate uniform 50x50 matrix
    uniform_50 = generate_uniform_matrix(50)
    np.save(os.path.join(SRC_DIR, "uniform_matrix_50.npy"), uniform_50)
    print(f"[✓] uniform_matrix_50.npy: shape={uniform_50.shape}, sum={uniform_50.sum():.6f}")
    
    # 2. Generate Layout 3 obstacle masks
    mask_25 = generate_skyline_obstacle_mask_25()
    np.save(os.path.join(SRC_DIR, "obstacle_skyline_25.npy"), mask_25)
    n_obs_25 = int(mask_25.sum())
    print(f"[✓] obstacle_skyline_25.npy: {n_obs_25} obstacle cells, {625 - n_obs_25} free cells")
    
    mask_50 = generate_skyline_obstacle_mask_50()
    np.save(os.path.join(SRC_DIR, "obstacle_skyline_50.npy"), mask_50)
    n_obs_50 = int(mask_50.sum())
    print(f"[✓] obstacle_skyline_50.npy: {n_obs_50} obstacle cells, {2500 - n_obs_50} free cells")
    
    # 3. Generate obstacle probability matrices
    obs_prob_25 = generate_obstacle_prob_matrix(25, mask_25)
    np.save(os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy"), obs_prob_25)
    print(f"[✓] obstacle_prob_matrix_25.npy: shape={obs_prob_25.shape}, sum={obs_prob_25.sum():.6f}")
    
    obs_prob_50 = generate_obstacle_prob_matrix(50, mask_50)
    np.save(os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy"), obs_prob_50)
    print(f"[✓] obstacle_prob_matrix_50.npy: shape={obs_prob_50.shape}, sum={obs_prob_50.sum():.6f}")
    
    print("\n[✓] All matrices generated successfully!")
