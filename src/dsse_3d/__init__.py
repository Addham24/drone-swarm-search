"""
DSSE-3D: Three-Dimensional Voxel Grid Extension for Drone Swarm Search Environments
=====================================================================================
Extends the 2D DSSE multi-UAV benchmarks (Coverage & Tracking) into a 3D voxel
workspace (X × Y × Z) using a Zero-Shot 3D Adapter architecture.

This module is self-contained and does NOT modify any existing 2D code.

Components:
    - voxel_grid:       Core 3D voxel state manager (altitude, obstacles, FOV)
    - adapter:          Zero-Shot 3D Adapter wrapping 2D DSSE environments
    - visualizer_3d:    Interactive Matplotlib 3D real-time visualizer
    - evaluate_3d:      3D benchmark evaluation suite (tracking & coverage)

Reference Architecture:
    Yan et al., "Multi-UAV 3D Target Search in Complex Environments Using DRL", 2022.
    Panerati et al., "gym-pybullet-drones", IEEE RA-L, 2021.
"""

from dsse_3d.voxel_grid import VoxelGrid
from dsse_3d.adapter import DSSE3DAdapter

__all__ = ["VoxelGrid", "DSSE3DAdapter"]
