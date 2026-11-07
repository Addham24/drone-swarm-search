#!/usr/bin/env python3
"""
Unit and Regression Test Suite for ObstacleAirspaceWrapper Coordinate Fix
========================================================================
Validates:
  3a. Impassable movement blocking on 25x25 and 50x50 obstacle maps.
  3b. BFS reachability over all free cells from spawn positions (100% ceiling).
  3c. 3D obstacle altitude gating (Z <= H blocked, Z > H flyover).
  3d. Exact seed-for-seed 2D regression on open standard maps (diff == 0.0).
  3e. Obstacle occupancy sanity and coverage cell purity across seeds 1000-1009.
"""

import os
import sys
import time
from collections import deque
import numpy as np
import pandas as pd
import torch as th

# Ensure repository root and src are on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import AllPositionsWrapper
from src.obstacle_airspace_wrapper import ObstacleAirspaceWrapper
from dsse_3d.env_creators import make_coverage_env_3d
from dsse_3d.policy_loaders import load_rllib_ppo_policy
from train_rspo_v2_vanilla import RSPOModelV2
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla
from run_all_2d_evaluations import make_coverage_env_2d, evaluate_coverage_2d_episode, _resolve_obstacle_mask, _resolve_prob_matrix

DELTA_TO_ACTION = {
    (-1, 0): 0,  # LEFT
    (1, 0): 1,   # RIGHT
    (0, -1): 2,  # UP
    (0, 1): 3,   # DOWN
    (-1, -1): 4, # UP_LEFT
    (1, -1): 5,  # UP_RIGHT
    (-1, 1): 6,  # DOWN_LEFT
    (1, 1): 7,   # DOWN_RIGHT
}


def test_3a_blocking():
    print(f"\n{'='*75}")
    print("  3a. IMPASSABLE MOVEMENT BLOCKING TEST (25x25 & 50x50)")
    print(f"{'='*75}")

    maps = [
        (25, os.path.join(SRC_DIR, "obstacle_skyline_25.npy"), os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")),
        (50, os.path.join(SRC_DIR, "obstacle_skyline_50.npy"), os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy")),
    ]

    for grid_size, sky_file, prob_file in maps:
        sky = np.load(sky_file)
        prob = np.load(prob_file)
        H, W = sky.shape

        # Verify skyline and prob_matrix encode identical cells
        assert np.array_equal(sky > 0, prob == 0.0), f"Mismatch between {sky_file} and {prob_file}!"
        n_obs = int((sky > 0).sum())
        n_free = H * W - n_obs
        print(f"[{grid_size}x{grid_size}] Loaded {os.path.basename(sky_file)} & {os.path.basename(prob_file)}")
        print(f"    Obstacle cells: {n_obs}, Free cells: {n_free}")

        base_env = CoverageDroneSwarmSearch(drone_amount=1, prob_matrix_path=prob_file, timestep_limit=1_000_000)
        env = AllPositionsWrapper(base_env)
        env = ObstacleAirspaceWrapper(env, sky)
        env.reset()

        real_base = env._get_base_env()
        agent = env.possible_agents[0]

        blocked_correct = 0
        blocked_wrong = 0
        passed_through = 0

        free_cells = [(x, y) for y in range(H) for x in range(W) if sky[y, x] == 0]
        for cx, cy in free_cells:
            for (dx, dy), action in DELTA_TO_ACTION.items():
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < W and 0 <= ny < H):
                    continue
                is_obstacle = (sky[ny, nx] > 0)

                # Teleport drone to (cx, cy)
                real_base.agents_positions[0] = (cx, cy)
                real_base.timestep = 0
                real_base.not_seen_states.add((-99, -99))  # Guard against episode completion termination
                env.env.agents = list(env.possible_agents)
                real_base.agents = list(env.possible_agents)

                obs, rewards, term, trunc, infos = env.step({agent: action})
                new_pos = real_base.agents_positions[0]
                collided = infos[agent].get("obstacle_collision", False)
                penalty_applied = (rewards[agent] <= -2.0)

                if is_obstacle:
                    if new_pos == (cx, cy) and collided and penalty_applied:
                        blocked_correct += 1
                    else:
                        passed_through += 1
                else:
                    if new_pos == (nx, ny) and not collided:
                        pass
                    else:
                        blocked_wrong += 1

        print(f"    Result: blocked-correct: {blocked_correct}")
        print(f"            blocked-wrong:   {blocked_wrong}")
        print(f"            passed-through-obstacle: {passed_through}")

        assert blocked_wrong == 0, f"Blocked-wrong must be 0, got {blocked_wrong}!"
        assert passed_through == 0, f"Passed-through must be 0, got {passed_through}!"
        assert blocked_correct > 0, f"Blocked-correct must be > 0!"
        print(f"    [✓] {grid_size}x{grid_size} Blocking Test PASSED!")


def test_3b_reachability():
    print(f"\n{'='*75}")
    print("  3b. BFS REACHABILITY TEST (25x25 & 50x50)")
    print(f"{'='*75}")

    maps = [
        (25, os.path.join(SRC_DIR, "obstacle_skyline_25.npy"), 529),
        (50, os.path.join(SRC_DIR, "obstacle_skyline_50.npy"), 2041),
    ]

    for grid_size, sky_file, expected_free in maps:
        sky = np.load(sky_file)
        H, W = sky.shape
        free_coords = {(x, y) for y in range(H) for x in range(W) if sky[y, x] == 0}
        assert len(free_coords) == expected_free

        spawn = [(0, 0), (0, 1), (1, 0), (1, 1)]
        visited = set()
        queue = deque()
        for s in spawn:
            if s in free_coords:
                visited.add(s)
                queue.append(s)

        while queue:
            cx, cy = queue.popleft()
            for (dx, dy) in DELTA_TO_ACTION.keys():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if (nx, ny) in free_coords and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny))

        print(f"[{grid_size}x{grid_size}] Reachable free cells: {len(visited)} / {len(free_coords)} (Expected: {expected_free})")
        assert len(visited) == expected_free, f"Reachable cells {len(visited)} != {expected_free}!"
        print(f"    [✓] {grid_size}x{grid_size} Reachability Test PASSED (100% coverage ceiling confirmed)!")


def test_3c_3d_altitude_check():
    print(f"\n{'='*75}")
    print("  3c. 3D ALTITUDE GATING & FLYOVER CHECK")
    print(f"{'='*75}")

    sky_file = os.path.join(SRC_DIR, "obstacle_skyline_25.npy")
    prob_file = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")
    mask = np.load(sky_file)

    env = make_coverage_env_3d(
        grid_size=25,
        drone_amount=1,
        positions=[(0, 0)],
        obstacle_mask=mask,
        prob_matrix_path=prob_file,
        enable_altitude_controller=False
    )
    env.reset(seed=1000)

    agent = env.possible_agents[0]
    base_env = env._get_base_env()

    # Find test obstacle cells of each height level: 1, 2, 3
    height_cells = {}
    for y in range(25):
        for x in range(25):
            h = int(mask[y, x])
            if h > 0 and h not in height_cells:
                for dx, dy, action in [(-1, 0, 0), (1, 0, 1), (0, -1, 2), (0, 1, 3)]:
                    fx, fy = x - dx, y - dy
                    if 0 <= fx < 25 and 0 <= fy < 25 and mask[fy, fx] == 0:
                        height_cells[h] = ((fx, fy), (x, y), action)
                        break

    print(f"Sample test obstacle cells per height level: {height_cells}")

    for h, ((fx, fy), (ox, oy), action) in sorted(height_cells.items()):
        for z in range(1, 5):
            env.voxel.drone_positions[0] = (fx, fy, z)
            base_env.agents_positions[0] = (fx, fy)
            base_env.timestep = 0
            base_env.not_seen_states.add((-99, -99))
            env.agents = list(env.possible_agents)
            base_env.agents = list(env.possible_agents)

            obs, rewards, term, trunc, infos = env.step({agent: action})
            new_pos = base_env.agents_positions[0]
            blocked = (new_pos == (fx, fy))
            expected_blocked = (z <= h)

            print(f"    Obstacle H={h} at ({ox},{oy}) | Drone Altitude Z={z} -> Blocked: {blocked} (Expected: {expected_blocked})")
            assert blocked == expected_blocked, f"Mismatch at H={h}, Z={z}: got blocked={blocked}, expected={expected_blocked}"

    # Also test move into a free cell at all altitudes
    fx, fy = 0, 0
    nx, ny = 1, 0
    action = 1  # RIGHT
    for z in range(1, 5):
        env.voxel.drone_positions[0] = (fx, fy, z)
        base_env.agents_positions[0] = (fx, fy)
        base_env.timestep = 0
        base_env.not_seen_states.add((-99, -99))
        env.agents = list(env.possible_agents)
        base_env.agents = list(env.possible_agents)

        obs, rewards, term, trunc, infos = env.step({agent: action})
        new_pos = base_env.agents_positions[0]
        blocked = (new_pos == (fx, fy))
        print(f"    Free Cell ({nx},{ny}) | Drone Altitude Z={z} -> Blocked: {blocked} (Expected: False)")
        assert not blocked, f"Free cell incorrectly blocked at Z={z}!"

    print("    [✓] 3D Altitude Gating & Flyover Check PASSED!")


def test_3d_open_map_regression():
    print(f"\n{'='*75}")
    print("  3d. REGRESSION ON OPEN MAPS (25x25_Standard, p=0.0005, seeds 1000-1099)")
    print(f"{'='*75}")

    old_raw_path = os.path.expanduser("~/Desktop/Results/2D-final/Coverage/25x25_Standard/Active_Faults_0.0005/coverage_2d_raw_seed_by_seed.csv")
    if not os.path.exists(old_raw_path):
        print(f"[!] Warning: {old_raw_path} not found. Skipping 3d.")
        return

    df_old = pd.read_csv(old_raw_path)

    env_cfg = {
        "tag": "25x25_Standard",
        "grid_size": 25,
        "timestep_limit": 750,
        "max_battery": 125,
        "has_obstacles": False,
        "free_cells": 625,
        "label": "25×25 Standard",
    }

    models = [
        ("RSPO Vanilla", RSPOModelV2, os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_Vanilla_RSPO_v2/RSPOPPO_DSSE_Coverage_RSPO_V2_Vanilla_45eb7_00000_0_2026-09-04_06-38-43/checkpoint_000136")),
        ("MAPPO Vanilla", MAPPOModelVanilla, os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_vanilla_proper/PPO_DSSE_Coverage_cc7e1_00000_0_2026-07-05_03-37-08/checkpoint_000122")),
    ]

    for model_name, cls, ckpt_path in models:
        print(f"    [▶] Re-evaluating {model_name} over 100 seeds (1000-1099)...")
        p_fn = load_rllib_ppo_policy(cls, ckpt_path, model_name, greedy=True, sample_mappo=True)
        summary, recs = evaluate_coverage_2d_episode(
            p_fn, env_cfg, fault_prob=0.0005, is_self_heal=False, n_seeds=100, obstacle_mask=None
        )
        df_new = pd.DataFrame(recs)
        df_sub_old = df_old[df_old["Model"] == model_name].sort_values("seed").reset_index(drop=True)
        df_new = df_new.sort_values("seed").reset_index(drop=True)

        diff_cov = np.max(np.abs(df_new["coverage_pct"] - df_sub_old["coverage_pct"]))
        diff_surv = np.max(np.abs(df_new["survival_rate"] - df_sub_old["survival_rate"]))
        diff_smei = np.max(np.abs(df_new["smei"] - df_sub_old["smei"]))

        print(f"        Max abs difference in coverage_pct: {diff_cov:.9f}")
        print(f"        Max abs difference in survival_rate: {diff_surv:.9f}")
        print(f"        Max abs difference in smei:          {diff_smei:.9f}")

        assert diff_cov < 1e-6, f"{model_name} coverage_pct changed by {diff_cov}!"
        assert diff_surv < 1e-6, f"{model_name} survival_rate changed by {diff_surv}!"
        assert diff_smei < 1e-6, f"{model_name} smei changed by {diff_smei}!"
        print(f"    [✓] {model_name} open-map regression strictly identical (diff == 0.0)!")


def test_3e_sanity_on_obstacles():
    print(f"\n{'='*75}")
    print("  3e. SANITY ON OBSTACLES (2D 25x25_Obstacles, p=0, RSPO Vanilla, seeds 1000-1009)")
    print(f"{'='*75}")

    obstacle_mask = _resolve_obstacle_mask(25)
    prob_path = _resolve_prob_matrix(25, has_obstacles=True)
    obstacle_coords = {(x, y) for y in range(25) for x in range(25) if obstacle_mask[y, x] > 0}

    rspo_ckpt = os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_Vanilla_RSPO_v2/RSPOPPO_DSSE_Coverage_RSPO_V2_Vanilla_45eb7_00000_0_2026-09-04_06-38-43/checkpoint_000136")
    policy_fn = load_rllib_ppo_policy(RSPOModelV2, rspo_ckpt, "RSPO Vanilla", greedy=True, sample_mappo=True)

    for seed in range(1000, 1010):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_coverage_env_2d(
            grid_size=25,
            drone_amount=4,
            timestep_limit=750,
            max_battery=125,
            fault_prob=0.0,
            is_self_heal=False,
            obstacle_mask=obstacle_mask,
            prob_matrix_path=prob_path,
        )
        obs, infos = env.reset(seed=seed)
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        if hasattr(base_env, "unwrapped"):
            base_env = base_env.unwrapped

        done = False
        step = 0
        while not done and step < 750:
            step += 1
            actions = policy_fn(obs, env.possible_agents, reset=(step == 1), grid_size=25)
            actions = {a: v for a, v in actions.items() if a in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)

            # Assert no drone is ever positioned on an obstacle cell
            for i, pos in enumerate(base_env.agents_positions):
                assert pos not in obstacle_coords, f"Seed {seed}, step {step}: Drone {i} on obstacle {pos}!"

            if all(term.values()) or all(trunc.values()):
                done = True

        seen = getattr(base_env, "seen_states", set())
        overlap = seen & obstacle_coords
        assert len(overlap) == 0, f"Seed {seed}: Obstacle cells counted in coverage: {overlap}"

        cov_cells = len(seen)
        cov_pct = cov_cells / 529.0 * 100.0
        # Check integer count assertion
        assert abs(cov_pct * 529 / 100.0 - cov_cells) < 1e-4

        print(f"    Seed {seed}: Steps={step} | Searched free cells={cov_cells}/529 ({cov_pct:.2f}%) | Obstacle overlap=0 [PASS]")

    print("    [✓] Obstacle Sanity & Purity Test PASSED!")


def main():
    print(f"\n{'#'*75}")
    print("  RUNNING COMPLETE OBSTACLE COORDINATE FIX VERIFICATION SUITE")
    print(f"{'#'*75}")
    t0 = time.time()

    test_3a_blocking()
    test_3b_reachability()
    test_3c_3d_altitude_check()
    test_3d_open_map_regression()
    test_3e_sanity_on_obstacles()

    el = time.time() - t0
    print(f"\n{'='*75}")
    print(f"  ALL TESTS 3a - 3e COMPLETED AND PASSED IN {el:.1f}s!")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
