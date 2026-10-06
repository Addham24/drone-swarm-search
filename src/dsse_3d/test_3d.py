"""
DSSE-3D Unit Test Suite
========================
Verifies correctness of the 3D voxel grid, zero-shot adapter, policy
adapters, and environment integration for both tracking and coverage tasks.

Usage:
    python -m dsse_3d.test_3d
    python -m pytest src/dsse_3d/test_3d.py -v
"""

import os
import sys
import unittest
import numpy as np

# Ensure src is on path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from dsse_3d.voxel_grid import VoxelGrid
from dsse_3d.adapter import DSSE3DAdapter
from dsse_3d.policy_adapters import strip_3d_obs, wrap_rllib_policy_for_3d


class TestVoxelGrid(unittest.TestCase):
    """Tests for the VoxelGrid 3D spatial state manager."""

    def setUp(self):
        self.voxel = VoxelGrid(grid_size=10, z_max=5, n_drones=4)

    def test_reset_positions(self):
        """Drones should be initialized at the correct 3D positions."""
        positions = [(0, 0), (1, 0), (0, 1), (1, 1)]
        self.voxel.reset(positions, initial_z=1)

        for i, (x, y) in enumerate(positions):
            self.assertEqual(self.voxel.get_position(i), (x, y, 1))
            self.assertEqual(self.voxel.get_2d_position(i), (x, y))
            self.assertEqual(self.voxel.get_altitude(i), 1)

    def test_altitude_bounds(self):
        """Altitude changes should respect [0, z_max] bounds."""
        self.voxel.reset([(5, 5)], initial_z=0)

        # Cannot descend below 0
        self.assertFalse(self.voxel.change_altitude(0, -1))
        self.assertEqual(self.voxel.get_altitude(0), 0)

        # Ascend to max
        for _ in range(5):
            self.voxel.change_altitude(0, +1)
        self.assertEqual(self.voxel.get_altitude(0), 5)

        # Cannot ascend above z_max
        self.assertFalse(self.voxel.change_altitude(0, +1))
        self.assertEqual(self.voxel.get_altitude(0), 5)

    def test_obstacle_blocking(self):
        """Obstacle prisms should block movement at affected altitudes."""
        obstacles = {(5, 5): 3}  # Obstacle at (5,5) from z=1 to z=3
        voxel = VoxelGrid(grid_size=10, z_max=5, n_drones=1, obstacle_heights=obstacles)
        voxel.reset([(4, 5)], initial_z=2)

        # Try to move into obstacle at z=2 (should be blocked)
        self.assertFalse(voxel.apply_2d_action(0, 5, 5))
        self.assertEqual(voxel.get_2d_position(0), (4, 5))  # Stayed in place

        # Move above obstacle (z=4)
        voxel.change_altitude(0, +1)  # z=3, still blocked
        voxel.change_altitude(0, +1)  # z=4, clear
        self.assertTrue(voxel.apply_2d_action(0, 5, 5))  # Should succeed at z=4
        self.assertEqual(voxel.get_2d_position(0), (5, 5))

    def test_can_search(self):
        """SEARCH should only be valid at scan altitude (Z=1)."""
        self.voxel.reset([(5, 5)], initial_z=1)
        self.assertTrue(self.voxel.can_search(0))

        self.voxel.change_altitude(0, +1)
        self.assertFalse(self.voxel.can_search(0))

        self.voxel.change_altitude(0, -1)
        self.assertTrue(self.voxel.can_search(0))

    def test_inter_drone_collision(self):
        """Collision detection should find drones at the same voxel."""
        self.voxel.reset([(5, 5), (5, 5)], initial_z=1)
        collisions = self.voxel.check_inter_drone_collision()
        self.assertEqual(len(collisions), 1)
        self.assertEqual(collisions[0], (0, 1))

    def test_altitude_observation(self):
        """Altitude observation should be normalized [0, 1]."""
        self.voxel.reset([(0, 0), (1, 1), (2, 2), (3, 3)], initial_z=1)
        self.voxel.change_altitude(1, +2)  # z=3

        alt_obs = self.voxel.get_altitude_observation()
        self.assertEqual(len(alt_obs), 4)
        self.assertAlmostEqual(alt_obs[0], 1.0 / 5.0, places=3)  # z=1
        self.assertAlmostEqual(alt_obs[1], 3.0 / 5.0, places=3)  # z=3

    def test_fov_footprint(self):
        """FOV footprint should respect altitude-dependent radius."""
        self.voxel.reset([(5, 5)], initial_z=1)
        fov_z1 = self.voxel.get_fov_footprint(0)
        self.assertEqual(len(fov_z1), 1)  # Radius 0 at Z=1 → single cell

        self.voxel.change_altitude(0, +1)  # z=2, radius=1
        fov_z2 = self.voxel.get_fov_footprint(0)
        self.assertEqual(len(fov_z2), 9)  # 3x3 = 9 cells

    def test_altitude_controller_charging_landing(self):
        """Drones at station (0,0) with battery < 95% should descend to Z=0 landing pad and hold until charged."""
        from dsse_3d.adapter import AltitudeController
        ctrl = AltitudeController(self.voxel)

        # Drone at station (0,0), z=2, battery=0.50 -> Should descend (-1)
        self.voxel.reset([(0, 0)], initial_z=2)
        action_z = ctrl.compute_altitude_action(drone_idx=0, action_2d=0, battery_level=0.50, at_station=True)
        self.assertEqual(action_z, -1, "Drone at station with low battery should descend")

        # Drone at station (0,0), z=0 (landed), battery=0.50 -> Should hold at Z=0 (0)
        self.voxel.reset([(0, 0)], initial_z=0)
        action_z_landed = ctrl.compute_altitude_action(drone_idx=0, action_2d=0, battery_level=0.50, at_station=True)
        self.assertEqual(action_z_landed, 0, "Landed drone at station should remain at Z=0 while charging")

        # Drone at station (0,0), z=0 (landed), fully charged battery=0.98, action=move -> Should ascend (+1)
        action_z_takeoff = ctrl.compute_altitude_action(drone_idx=0, action_2d=1, battery_level=0.98, at_station=True)
        self.assertEqual(action_z_takeoff, +1, "Fully charged drone should take off from Z=0 to Z=1")


class TestPolicyAdapters(unittest.TestCase):
    """Tests for the zero-shot policy observation adapters."""

    def test_strip_3d_obs(self):
        """Stripping should remove exactly the altitude suffix."""
        n_drones = 4
        original_pos = np.random.rand(22).astype(np.float32)
        altitude_suffix = np.random.rand(n_drones + 1).astype(np.float32)
        augmented_pos = np.concatenate([original_pos, altitude_suffix])
        matrix = np.random.rand(25, 25).astype(np.float32)

        obs_3d = {"drone0": (augmented_pos, matrix)}
        obs_2d = strip_3d_obs(obs_3d, n_extra_dims=n_drones + 1)

        np.testing.assert_array_almost_equal(obs_2d["drone0"][0], original_pos)
        np.testing.assert_array_almost_equal(obs_2d["drone0"][1], matrix)

    def test_wrapped_policy_receives_2d_obs(self):
        """Wrapped policy should receive 2D-shaped observations."""
        received_obs = {}

        def mock_2d_policy(obs_dict, agents, *args, **kwargs):
            for agent, obs in obs_dict.items():
                received_obs[agent] = obs[0].shape[0]
            return {agent: 0 for agent in agents}

        wrapped = wrap_rllib_policy_for_3d(mock_2d_policy, n_drones=4)

        # Create 3D augmented obs
        pos_3d = np.random.rand(27).astype(np.float32)  # 22 + 5
        mat = np.random.rand(25, 25).astype(np.float32)
        obs_3d = {"drone0": (pos_3d, mat)}

        wrapped(obs_3d, ["drone0"])
        self.assertEqual(received_obs["drone0"], 22)  # Should be stripped to 22


class TestDSSE3DAdapterTracking(unittest.TestCase):
    """Integration tests for DSSE3DAdapter with tracking environment."""

    def test_tracking_env_creation(self):
        """3D tracking environment should create and reset without errors."""
        from dsse_3d.env_creators import make_tracking_env_3d

        env = make_tracking_env_3d(
            grid_size=25,
            fault_prob=0.0,
            is_self_heal=False,
        )
        obs, infos = env.reset(seed=42)

        # Check obs structure
        self.assertGreater(len(obs), 0)
        first_agent = list(obs.keys())[0]
        pos, mat = obs[first_agent]

        # Position should be augmented (22 + n_drones + 1 = 27)
        self.assertEqual(pos.shape[0], 22 + 4 + 1)
        # Matrix should remain 25x25
        self.assertEqual(mat.shape, (25, 25))

        # Check 3D info
        self.assertIn("altitude", infos[first_agent])
        self.assertIn("position_3d", infos[first_agent])

    def test_tracking_step(self):
        """3D tracking environment should step without errors."""
        from dsse_3d.env_creators import make_tracking_env_3d

        env = make_tracking_env_3d(grid_size=25, fault_prob=0.0)
        obs, _ = env.reset(seed=42)

        # Random actions
        actions = {agent: np.random.randint(0, 9) for agent in env.agents}
        obs2, rewards, term, trunc, infos = env.step(actions)

        self.assertGreater(len(obs2), 0)
        self.assertGreater(len(rewards), 0)

    def test_tracking_zero_shot_policy(self):
        """Zero-shot wrapped policy should work correctly in 3D tracking."""
        from dsse_3d.env_creators import make_tracking_env_3d

        env = make_tracking_env_3d(grid_size=25, fault_prob=0.0)
        obs, _ = env.reset(seed=42)

        # Create a mock 2D policy
        def mock_policy(obs_dict, agents, *a, **kw):
            return {agent: 8 for agent in agents}  # Always SEARCH

        wrapped = wrap_rllib_policy_for_3d(mock_policy)

        # Run for 10 steps
        for _ in range(10):
            actions = wrapped(obs, env.agents)
            obs, rewards, term, trunc, infos = env.step(actions)
            if all(term.values()) or all(trunc.values()):
                break


class TestDSSE3DAdapterCoverage(unittest.TestCase):
    """Integration tests for DSSE3DAdapter with coverage environment."""

    def test_coverage_env_creation(self):
        """3D coverage environment should create and reset without errors."""
        from dsse_3d.env_creators import make_coverage_env_3d

        env = make_coverage_env_3d(
            grid_size=25,
            fault_prob=0.0,
            is_self_heal=False,
        )
        obs, infos = env.reset(seed=42)

        self.assertGreater(len(obs), 0)
        first_agent = list(obs.keys())[0]
        pos, mat = obs[first_agent]

        # Position should be augmented
        expected_pos_dim = 22 + 4 + 1  # 27
        self.assertEqual(pos.shape[0], expected_pos_dim)

    def test_coverage_step(self):
        """3D coverage environment should step correctly."""
        from dsse_3d.env_creators import make_coverage_env_3d

        env = make_coverage_env_3d(grid_size=25, fault_prob=0.0)
        obs, _ = env.reset(seed=42)

        for step in range(20):
            actions = {agent: np.random.randint(0, 9) for agent in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)
            if all(term.values()) or all(trunc.values()):
                break

    def test_3d_coverage_altitude_dependency(self):
        """Verification that 3D coverage scales with altitude FOV footprint (Z=1: 1x1, Z=2: 3x3, Z=0: 0)."""
        from dsse_3d.env_creators import make_coverage_env_3d

        # Create env with manual altitude control (no auto-controller) to test Z-dependency
        env = make_coverage_env_3d(grid_size=25, fault_prob=0.0, enable_altitude_controller=False)
        obs, infos = env.reset(seed=100)

        # Initially drones start at Z=1 (scan altitude, FOV radius=0 -> 1x1 cell per drone)
        initial_coverage = len(env.seen_states_3d)
        self.assertGreater(initial_coverage, 0)

        # Move drones to Z=2 (transit altitude, FOV radius=1 -> 3x3 = 9 cells footprint per drone)
        for i in range(env.n_drones):
            env.voxel.drone_positions[i] = (env.voxel.drone_positions[i][0], env.voxel.drone_positions[i][1], 2)

        # Step once at Z=2
        actions = {agent: 3 for agent in env.agents}
        env.step(actions)

        coverage_at_z2 = len(env.seen_states_3d)
        # Because Z=2 has a 3x3 footprint per drone, coverage at Z=2 should grow significantly faster than 1x1
        self.assertGreater(coverage_at_z2, initial_coverage + 4, "Higher altitude (Z=2) expands sensor FOV footprint (3x3)")

        # Move drones to Z=0 (ground level, landed at station)
        for i in range(env.n_drones):
            env.voxel.drone_positions[i] = (env.voxel.drone_positions[i][0], env.voxel.drone_positions[i][1], 0)

        coverage_before_z0 = len(env.seen_states_3d)
        actions_idle = {agent: 8 for agent in env.agents}
        env.step(actions_idle)
        coverage_after_z0 = len(env.seen_states_3d)

        self.assertEqual(coverage_before_z0, coverage_after_z0, "Landed drones at Z=0 accumulate 0 aerial coverage")


class TestDSSE3DWithObstacles(unittest.TestCase):
    """Tests for 3D environments with extruded obstacle prisms."""

    def test_obstacle_extrusion(self):
        """Obstacles should be extruded vertically in the voxel grid."""
        obstacle_mask = np.zeros((10, 10), dtype=int)
        obstacle_mask[5, 5] = 1  # Single obstacle column

        voxel = VoxelGrid(
            grid_size=10, z_max=5, n_drones=1,
            obstacle_heights={(5, 5): 3}
        )

        # Voxels (5, 5, 1..3) should be blocked
        self.assertTrue(voxel._is_blocked(5, 5, 1))
        self.assertTrue(voxel._is_blocked(5, 5, 2))
        self.assertTrue(voxel._is_blocked(5, 5, 3))

        # Voxels above the obstacle should be clear
        self.assertFalse(voxel._is_blocked(5, 5, 4))
        self.assertFalse(voxel._is_blocked(5, 5, 5))

        # Ground level should be clear (target plane)
        self.assertFalse(voxel._is_blocked(5, 5, 0))

    def test_3d_metrics(self):
        """3D metrics should track altitude changes and collisions."""
        from dsse_3d.env_creators import make_tracking_env_3d

        env = make_tracking_env_3d(grid_size=25, fault_prob=0.0)
        env.reset(seed=42)

        for _ in range(10):
            actions = {agent: np.random.randint(0, 9) for agent in env.agents}
            env.step(actions)

        metrics = env.get_3d_metrics()
        self.assertIn("altitude_changes", metrics)
        self.assertIn("midair_collisions", metrics)
        self.assertIn("obstacle_collisions_3d", metrics)
        self.assertIn("trajectories", metrics)


if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  DSSE-3D Unit Test Suite")
    print(f"{'='*60}\n")
    unittest.main(verbosity=2)
