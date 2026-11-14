# DSSE Obstacle Coordinate Transposition Bug Fix & Verification Report

**Author / Runner**: Adham Motawi  
**Date**: November 7, 2026  
**Git Branch**: `fix/obstacle-transpose`  
**Base Commit**: `91ddba5` / `c52c997`  
**Fix Commit**: `213be59`  
**Evaluation Scope**: Re-evaluation of ONLY the 20 Obstacle Scenarios (10 in 2D, 10 in 3D) across all 12 models (seeds 1000–1099). All 20 Standard-map scenarios remain strictly untouched and byte-identical.

---

## 1. Executive Summary & The Bug

### 1.1 The Bug Identification
In `src/obstacle_airspace_wrapper.py`, the `step()` method intercepts agent actions to enforce 2D obstacle collisions. The PettingZoo environment `env_base.py` in DSSE stores agent positions as `(x, y) = (col, row)`. Specifically:
- `LEFT` decrements `position[0]`: `(position[0] - 1, position[1])` $\to$ `position[0]` is column ($x$).
- `UP` decrements `position[1]`: `(position[0], position[1] - 1)` $\to$ `position[1]` is row ($y$).

However, `ObstacleAirspaceWrapper.step()` contained an inverted comment and transposed unpacking:
```python
# BEFORE (BUGGY):
current_pos = base_env.agents_positions[i]  # (y, x) in DSSE      <-- WRONG COMMENT
new_pos = base_env.move_drone(current_pos, actions[agent])
target_x, target_y = new_pos[1], new_pos[0]                        <-- BUG: TRANSPOSED
if (target_x, target_y) in self.obstacle_coords:
    target_h = int(self.obstacle_mask[target_y, target_x])
```
Because `self.obstacle_coords` is constructed as `(c, r) = (x, y)` from `obstacle_mask[y, x] > 0`, looking up `(target_x, target_y) = (new_pos[1], new_pos[0])` queried the mirror cell $(y \leftrightarrow x)$.

### 1.2 Physical Consequences
- **25×25 Obstacle Map**: 32 of 96 real obstacle cells could be entered without collision penalties, while 32 free cells were mirrored as false obstacles, permanently blocked with collision penalties ($-2.0$), capping theoretical 2D coverage at $497 / 529 = 93.95\%$.
- **50×50 Obstacle Map**: 18 obstacle cells each way were transposed, capping coverage at $2023 / 2041 = 99.12\%$.
- **3D Airspace**: In 3D, false obstacle heights blocked drones at mirror coordinates unless they ascended above the mirrored cell's building height.

---

## 2. The Code Fix & Audit

### 2.1 Changes to `src/obstacle_airspace_wrapper.py`
In `src/obstacle_airspace_wrapper.py`, the unpacking was corrected to `target_x, target_y = new_pos[0], new_pos[1]` and comments were updated.

```diff
--- a/src/obstacle_airspace_wrapper.py
+++ b/src/obstacle_airspace_wrapper.py
@@ -27,7 +27,7 @@ class ObstacleAirspaceWrapper(BaseParallelWrapper):
     Mechanics:
         - Obstacle heights range 1..3. Obstacle cells have mask > 0.
         - Drones default to altitude Z=1.
-        - Moving into a cell (y', x') where obstacle_mask[y', x'] > 0:
+        - Moving into a cell (x', y') where obstacle_mask[y', x'] > 0:
             * In 2D: blocked! Drone stays in place, receives penalty.
             * In 3D: blocked only if current drone altitude <= building height.
@@ -140,8 +140,8 @@ class ObstacleAirspaceWrapper(BaseParallelWrapper):
         for i, agent in enumerate(base_env.agents):
             if agent not in actions:
                 continue
-            current_pos = base_env.agents_positions[i]  # (y, x) in DSSE
+            current_pos = base_env.agents_positions[i]  # (x, y) = (col, row) in DSSE
             new_pos = base_env.move_drone(current_pos, actions[agent])
-            target_x, target_y = new_pos[1], new_pos[0]
+            target_x, target_y = new_pos[0], new_pos[1]
             if (target_x, target_y) in self.obstacle_coords:
                 target_h = int(self.obstacle_mask[target_y, target_x])
```

### 2.2 Grep Audit Across the Repository
A full repository audit was conducted across `src/obstacle_airspace_wrapper.py`, `src/spatial_rescaling_wrapper.py`, `src/dsse_3d/adapter.py`, and rendering routines:
1. `self.obstacle_coords`: Constructed at initialization via `self.obstacle_coords.add((c, r))` where `r` is row and `c` is col. `(x, y) = (c, r)`. Correct.
2. `base_env.seen_states - self.obstacle_coords`: In DSSE, `seen_states` stores `(x, y)`. The set difference correctly subtracts obstacle cells. Correct.
3. `_encode_obstacles_in_obs`: Iterates `for x, y in self.obstacle_coords: matrix[y, x] = -1.0`. `y` is row index and `x` is col index in numpy array `matrix`. Correct.
4. `custom_draw`: Iterates `for counter_x ...: for counter_y ...:` and accesses `obstacle_mask[counter_y, counter_x]`. Matches `(col, row)`. Correct.
5. `src/spatial_rescaling_wrapper.py`: Uses 2D block reshaping and average pooling (`reshape(th, sf, tw, sf)`). Drone positions vector is passed through without coordinate transposition. Correct.
6. `src/dsse_3d/adapter.py`: Commit `c52c997` already correctly indexed `base_env.agents_positions` as `(x, y) = (col, row)` when calling `self.voxel.apply_2d_action(idx, x, y)` and sets `curr_env.current_altitudes = current_alts`. Correct.

### 2.3 RSPO CSSA Attention Investigation (Report Only)
In `src/train_rspo_v2_vanilla.py`, `RSPOModelV2._compute_cssa_attention` (lines 225–231) extracts crash coordinates from `input_positions[:, 12:20]`:
```python
crash_r = input_positions[:, 12:20:2]  # Unpacked as row/y
crash_c = input_positions[:, 13:20:2]  # Unpacked as col/x
```
**Observation Analysis**:
In `src/custom_wrappers.py` (`BatteryStationWrapper.get_positions_vector`):
- `agents_positions[i]` is stored as `pos[0], pos[1]` $\to$ `(col, row) = (x, y)`.
- Crash coordinate slots (indices 12 to 19) store the crash location `(pos[0], pos[1])` when a drone crashes.
- Therefore, `input_positions[:, 12:20:2]` (even indices: 12, 14, 16, 18) actually holds $x$ (column), and `input_positions[:, 13:20:2]` (odd indices: 13, 15, 17, 19) holds $y$ (row).
- **Finding**: `_compute_cssa_attention` conceptually treats slot 0 as row/$y$ and slot 1 as col/$x$, which inverts $(x \leftrightarrow y)$ for the spatial Gaussian distance kernel computation against grid coordinates. Per instructions, **no model code or checkpoint was altered**, as modifying this would require retraining the neural network from scratch.

---

## 3. Pre-Run Verification Test Suite Results

The comprehensive test suite `tests/test_obstacle_wrapper.py` was executed. All checks passed:

### Check 3a: Blocking Test (Both 25×25 and 50×50 Maps)
Tested every free cell adjacent to an obstacle, attempting every cardinal move into adjacent cells:
- **25×25 Map** (`src/obstacle_skyline_25.npy`):
  - Blocked-correct: **416**
  - Blocked-wrong: **0**
  - Passed-through-obstacle: **0**
  - Status: **PASSED**
- **50×50 Map** (`src/obstacle_skyline_50.npy`):
  - Blocked-correct: **1,632**
  - Blocked-wrong: **0**
  - Passed-through-obstacle: **0**
  - Status: **PASSED**

### Check 3b: Free Cell Reachability (BFS from Spawn Cells)
BFS explored all reachable free cells using the wrapper's blocking rule:
- **25×25 Map**: **529 / 529** free cells reachable (**100.0% coverage ceiling**).
- **50×50 Map**: **2,041 / 2,041** free cells reachable (**100.0% coverage ceiling**).
- Status: **PASSED**

### Check 3c: 3D Altitude Gating
Tested in 3D adapter across building height levels ($h \in \{1, 2, 3\}$):
- Building height 1: Blocked at $Z=1$; cleanly passes over at $Z \in \{2, 3, 4\}$.
- Building height 2: Blocked at $Z \in \{1, 2\}$; cleanly passes over at $Z \in \{3, 4\}$.
- Building height 3: Blocked at $Z \in \{1, 2, 3\}$; cleanly passes over at $Z=4$.
- Free cells never block at any altitude $Z \in \{1, 2, 3, 4\}$.
- Status: **PASSED**

### Check 3d: Open Map Regression (Seeds 1000–1099)
Re-evaluated 2D `25x25_Standard` at $p = 0.0005$ across 100 seeds for RSPO Vanilla and MAPPO Vanilla against the baseline v2 results:
- RSPO Vanilla: $\max |\Delta \text{Coverage}| = \mathbf{0.000000000}$, $\max |\Delta \text{Survival}| = \mathbf{0.000000000}$, $\max |\Delta \text{SMEI}| = \mathbf{0.000000000}$.
- MAPPO Vanilla: $\max |\Delta \text{Coverage}| = \mathbf{0.000000000}$, $\max |\Delta \text{Survival}| = \mathbf{0.000000000}$, $\max |\Delta \text{SMEI}| = \mathbf{0.000000000}$.
- Status: **PASSED** (Strict zero-diff regression).

### Check 3e: Obstacle Purity Sanity (Seeds 1000–1009)
Evaluated 2D `25x25_Obstacles` at $p = 0$ for RSPO Vanilla across 10 seeds:
- Obstacle cells included in coverage across all episodes: **0**
- Drones positioned on obstacle cells across every timestep of every episode: **0**
- Status: **PASSED**

---

## 4. Provenance & Dashboard Fixes

1. **Dynamic Git Provenance**: Replaced hardcoded git commit hashes in `src/run_master_final_benchmarks.py` and `src/run_2d_checkpoint_updates.py` with runtime `git rev-parse HEAD`. Added assertion enforcing that `git status --porcelain` is clean before benchmarking begins.
2. **Dashboard SMEI Metric (Panel 6)**: Fixed dashboard generation in `src/dsse_3d/run_all_3d_evaluations.py` and `src/run_all_2d_evaluations.py`. Panel 6 now strictly plots the **mean of the per-seed `smei` column** (per-episode SMEI), rather than recomputing SMEI from aggregate means.
3. **Panel 5 Relabeling**: Relabeled Panel 5 across all 2D and 3D dashboards as `"5. Mission Return (5× cells searched)"` to accurately reflect the PettingZoo base reward ($+5.0$ per newly discovered cell) rather than net shaped return.
4. **Figure 2.3 Visualizer Re-renders**: Re-rendered all three Figure 2.3 3D visualizer images without in-image title banners while preserving the HUD telemetry box. Confirmed standard open-map render contains zero building obstacle cells.

---

## 5. Answers to Section 6 Facts

### Fact 1: Exact Scripts and Commands for v2 Results
- **3D-final v2**: Generated by:
  `python3 src/run_master_final_benchmarks.py --dim 3d --workers 6 --force`
  Git commit: `decd8ff` (with adapter fix `c52c997`), timestamp: Nov 7, 2026 00:55:14.
- **2D-final v2**: 10 models generated by `python3 src/run_master_final_benchmarks.py --dim 2d --workers 6` (commit `dee7b58`, Nov 6, 2026), and 2 models (MAA2C Vanilla `10035000/agent.th` and I-DQN SelfHeal `checkpoint_000587`) updated by `python3 src/run_2d_checkpoint_updates.py --workers 6` (commit `b06a65c`, Nov 7, 2026 00:54:17).

### Fact 2: QMIX Network & Buffer Configuration
From `src/epymarl_train.py:177` and `src/epymarl/src/config/algs/qmix.yaml`:
- `use_rnn = False` (Feedforward neural network architecture).
- `hidden_dim = 64` (from `src/epymarl/src/config/default.yaml`).
- `buffer_size = 200` episodes (from `src/epymarl_train.py:177` and Sacred config `run.json`).

### Fact 3: Learning Rates for COMA, MAA2C, and QMIX Checkpoints
From Sacred run configurations (`config.json`):
- **COMA**: `lr = 0.00005` ($5 \times 10^{-5}$), `critic_lr = 0.0005` ($5 \times 10^{-4}$).
- **MAA2C**: `lr = 0.0003` ($3 \times 10^{-4}$), `critic_lr = 0.0003`.
- **QMIX**: `lr = 0.0005` ($5 \times 10^{-4}$).

### Fact 4: Worker/Runner Allocations for RLlib Models
From `params.json` and `progress.csv`:
- **RSPO Vanilla**: `num_env_runners = 10`
- **RSPO SelfHeal**: `num_env_runners = 10`
- **MAPPO Vanilla**: `num_env_runners = 24`
- **MAPPO SelfHeal**: `num_env_runners = 24` (GPU runs) / `6` (CPU fallback)
- **I-DQN Vanilla**: `num_env_runners = 16`
- **I-DQN SelfHeal**: `num_env_runners = 16`

### Fact 5: MAPPO SelfHeal Training Log After 1.97M Steps
- The training directory `src/ray_res/DSSE_Coverage/PPO_DSSE_Coverage_0261b` logged up to step $1,974,272$ ($1.97\text{M}$ steps) where TensorBoard/progress logging ceased.
- The resumed run to 20M steps only wrote raw checkpoint state files to `src/ray_res/DSSE_Coverage/MAPPO_selfheal_final_v2/checkpoints/` terminating at `checkpoint_final_ts_20004864.pt`.
- No post-10M CSV metrics log exists for MAPPO SelfHeal; thus checkpoint selection was fixed at the final completed checkpoint.

### Fact 6: EPyMARL Checkpoint Intervals, Post-10M Checkpoints & Selection Rules
From Sacred logs and checkpoint directories:
- **QMIX Vanilla**: `save_model_interval = 1,000,000` (step delta $1,002,000$). Checkpoints post-10M: `[10026000, 11028000, 12030000, 13032000, 14034000, 15036000, 16038000, 17040000, 18042000, 19044000]`. Post-10M peak test return occurred at step $16,584,000$ (return $1733.2$). Evaluated checkpoint `17040000` is the **first saved checkpoint after the peak** (and nearest).
- **QMIX SelfHeal**: `save_model_interval = 1,000,000` (step delta $1,002,000$). Checkpoints post-10M: `[10026000, 11028000, 12030000, 13032000, 14034000, 15036000, 16038000, 17040000, 18042000, 19044000]`. Post-10M peak test return occurred at step $18,618,000$ (return $1728.8$). Evaluated checkpoint `19044000` is the **first saved checkpoint after the peak** (and nearest).
- **COMA Vanilla**: `save_model_interval = 50,000` (step delta $54,000$). 185 checkpoints post-10M (`10050000` to `19986000`). Post-10M peak test return occurred at step $19,350,000$ (return $115.8$). Evaluated checkpoint `19338000` is the **nearest saved checkpoint** ($-12\text{k}$ diff before peak, vs $+42\text{k}$ for `19392000`).
- **COMA SelfHeal**: `save_model_interval = 50,000` (step delta $54,000$). 185 checkpoints post-10M (`10050000` to `19986000`). Post-10M peak test return occurred at step $16,668,000$ (return $151.7$). Evaluated checkpoint `16692000` is the **first saved checkpoint after the peak** (and nearest, $+24\text{k}$ after vs $-30\text{k}$ before).
- **MAA2C Vanilla**: `save_model_interval = 50,000` (step delta $60,000$). 147 checkpoints post-10M (`10035000` to `18795000`). Post-10M peak test return occurred at step $10,065,000$ (return $1130.1$). Evaluated checkpoint `10035000` was selected (equidistant $-30\text{k}$ before peak, vs $+30\text{k}$ for `10095000`).
- **MAA2C SelfHeal**: `save_model_interval = 50,000` (step delta $60,000$). 167 checkpoints post-10M (`10035000` to `19995000`). Post-10M peak test return occurred at step $12,675,000$ (return $1130.7$). Evaluated checkpoint `12675000` matches the **exact peak step**.

### Fact 7: RLlib Checkpoint Frequency
Confirmed from `PPO/DQN` checkpoint directories: Checkpoint $k$ is written at iteration $10(k+1)$. Examples:
- Checkpoint `000000` is iteration 10.
- Checkpoint `000122` is iteration 1,230.
- Checkpoint `000587` is iteration 5,880.

### Fact 8: Spawn Positions of the Four Drones
From `DSSE/environment/env_base.py:101-105`:
The four drones spawn at coordinates `[(0, 0), (0, 1), (1, 0), (1, 1)]` forming a $2 \times 2$ cluster adjacent to the base station at $(0, 0)$. The thesis description mentioning $(0, 0)$ for all drones is a simplified text description; the actual environment spawns them at adjacent non-overlapping cells.

### Fact 9: Obstacle Skyline vs Probability Matrix Files
Evaluation scripts load `obstacle_skyline_25.npy` and `obstacle_skyline_50.npy` for building heights, and load `obstacle_prob_matrix_25.npy` and `obstacle_prob_matrix_50.npy` for the initial probability distribution.
- Audit check: `(obstacle_skyline > 0) == (obstacle_prob_matrix == 0.0)` for all 625 cells on $25 \times 25$ (96 obstacle cells) and all 2,500 cells on $50 \times 50$ (459 obstacle cells). Both files encode the exact same obstacle locations.

### Fact 10: 3D Altitude Action Penalties
From `src/dsse_3d/adapter.py:440-455`:
- **Search Penalty**: A penalty of $-0.5$ applies if an agent executes action `SEARCH` when $z \neq 1$. This includes $z = 0$ (docked at charging station) and $z \ge 2$ (too high for effective optical search).
- **Altitude Transition Cost**: An action cost of $-0.02$ applies whenever an agent changes altitude ($\Delta z \neq 0$).

---

## 6. Verification of Unchanged Standard-Map Files (SHA-256)

All 40 Standard-map raw seed-by-seed CSV files (20 in 2D, 20 in 3D) are byte-for-byte identical between v2 and v3:

| Dimension | Scenario | SHA-256 Hash | Match |
| :--- | :--- | :--- | :---: |
| 2D | 25x25_Standard / No_Faults_0.0 | `802d56c5bfd57094e9f9050d275bcefa7e467f53f9547dca9637c76251b32bb3` | PASS |
| 2D | 25x25_Standard / Fault_0.00025 | `23bf742a92fbc8623b36ab83e8fa8ff8ca02179a32cba972e7a33b0ea95914be` | PASS |
| 2D | 25x25_Standard / Active_Faults_0.0005 | `3ba12a2627f045424df1eb1a473f8a41a4a4d048d087c2c9d69e46a788bbd6d3` | PASS |
| 2D | 25x25_Standard / Fault_0.00075 | `12fe456ff6dcb215be17ce8b23f0535e6c7031846b0e9a7e6ea8724d9c7e090f` | PASS |
| 2D | 25x25_Standard / Fault_0.001 | `3e4825e5fb318d85e7baecae35d3d04fc3d0c3bc6b9f44db4fe9c323f4a43b2f` | PASS |
| 2D | 50x50_Standard / No_Faults_0.0 | `ea216d9e3afd6364b4cbbbf83e29f3796d19ebf85d1e21b77fba1ffcae2b17f9` | PASS |
| 2D | 50x50_Standard / Fault_0.00025 | `477b479aab92a8e100f946399b247fcfafc596ec6ba6bf4f620e7ae361d76378` | PASS |
| 2D | 50x50_Standard / Active_Faults_0.0005 | `f3f499d7dc05aa6199bc4892c9085ecae7ff71ec44cb3f9e8020a1db48d5d4d3` | PASS |
| 2D | 50x50_Standard / Fault_0.00075 | `fef42e67be1f9f933f7d2f4d6d6396f9a078bfa6335165d44a1aeae232921508` | PASS |
| 2D | 50x50_Standard / Fault_0.001 | `1600ba3c33957b6a94f61f7db16f5c889f074d28ba64c1b96d9ccce9996e1b74` | PASS |
| 3D | 25x25_Standard / No_Faults_0.0 | `012860bdcf18eb68bcbe9618b76c8c49942a275727931c81dcde0218fc81da30` | PASS |
| 3D | 25x25_Standard / Fault_0.00025 | `da04152dd9501cea37ebc529ba4a2e5d95bf1d6d1b76df479b4ce55a9b00cb33` | PASS |
| 3D | 25x25_Standard / Active_Faults_0.0005 | `374c6db1b12068903ca4ff0ca90356612d3df04cb1a5c68b8e847c1a2f6412e8` | PASS |
| 3D | 25x25_Standard / Fault_0.00075 | `c11c00e1ef570b6d9dae691ec85c1c875d97ec9537f7a2db1da1994cc5386dbb` | PASS |
| 3D | 25x25_Standard / Fault_0.001 | `4b91f98dec67c392f4e040432c6dffae48fc7688229b47e53f1ff4f16422d103` | PASS |
| 3D | 50x50_Standard / No_Faults_0.0 | `bfa72d4c75a22e21b069d30ca5794775437895318db820468a73955db369e578` | PASS |
| 3D | 50x50_Standard / Fault_0.00025 | `9a190bfd6a46dfbac4cf760773d44bc083cfc5c56d77c154f8664ecfc94297eb` | PASS |
| 3D | 50x50_Standard / Active_Faults_0.0005 | `a8310115551c74cfcf595e69bf6d6568dd77d5ef9b09d949479b183dd0027f3d` | PASS |
| 3D | 50x50_Standard / Fault_0.00075 | `5568636442e54307ef0cb11cb5dfd5c9576402777174e9089fb23d7ec17c5b6b` | PASS |
| 3D | 50x50_Standard / Fault_0.001 | `6529894ff3e6f3460f089e60a3734a9ca3b8efad42ce146ba31c77f0a1491e57` | PASS |

---

## 7. Integrity Assertions on Obstacle Scenarios

All 20 obstacle scenario raw CSV files in v3 strictly satisfy all integrity criteria:
1. Exactly **1,200 rows** per raw CSV, with **0 duplicate `(Model, seed)` pairs**.
2. **Integer Searched Free Cells**: $\text{coverage\_pct} \times N_{\text{free}} / 100$ is an integer within $10^{-4}$ ($N_{\text{free}} = 529$ on $25 \times 25$, $2041$ on $50 \times 50$).
3. **SMEI Formula Precision**: $\text{smei} = 100 \times \max(C, 0.001)^{0.4} \times \max(S, 0.001)^{0.3} \times \max(B, 0.001)^{0.3}$ within $10^{-9}$.

| Scenario | N Rows | Duplicates | Max Integer Err | Max SMEI Err | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 2D 25x25_Obstacles No Faults (p=0.0) | 1200 | 0 | $8.53e-14$ | $1.42e-14$ | PASS |
| 2D 25x25_Obstacles Low Faults (p=0.00025) | 1200 | 0 | $8.53e-14$ | $1.42e-14$ | PASS |
| 2D 25x25_Obstacles Standard Faults (p=0.0005) | 1200 | 0 | $8.53e-14$ | $1.42e-14$ | PASS |
| 2D 25x25_Obstacles High Faults (p=0.00075) | 1200 | 0 | $8.53e-14$ | $1.42e-14$ | PASS |
| 2D 25x25_Obstacles Severe Faults (p=0.0010) | 1200 | 0 | $8.53e-14$ | $1.42e-14$ | PASS |
| 2D 50x50_Obstacles No Faults (p=0.0) | 1200 | 0 | $2.27e-13$ | $1.42e-14$ | PASS |
| 2D 50x50_Obstacles Low Faults (p=0.00025) | 1200 | 0 | $1.71e-13$ | $1.42e-14$ | PASS |
| 2D 50x50_Obstacles Standard Faults (p=0.0005) | 1200 | 0 | $1.14e-13$ | $1.42e-14$ | PASS |
| 2D 50x50_Obstacles High Faults (p=0.00075) | 1200 | 0 | $1.71e-13$ | $1.42e-14$ | PASS |
| 2D 50x50_Obstacles Severe Faults (p=0.0010) | 1200 | 0 | $1.71e-13$ | $7.11e-15$ | PASS |
| 3D 25x25_Obstacles No Faults (p=0.0) | 1200 | 0 | $1.71e-13$ | $1.42e-14$ | PASS |
| 3D 25x25_Obstacles Low Faults (p=0.00025) | 1200 | 0 | $1.71e-13$ | $1.42e-14$ | PASS |
| 3D 25x25_Obstacles Standard Faults (p=0.0005) | 1200 | 0 | $1.14e-13$ | $1.42e-14$ | PASS |
| 3D 25x25_Obstacles High Faults (p=0.00075) | 1200 | 0 | $1.71e-13$ | $1.42e-14$ | PASS |
| 3D 25x25_Obstacles Severe Faults (p=0.0010) | 1200 | 0 | $1.71e-13$ | $1.42e-14$ | PASS |
| 3D 50x50_Obstacles No Faults (p=0.0) | 1200 | 0 | $6.82e-13$ | $2.84e-14$ | PASS |
| 3D 50x50_Obstacles Low Faults (p=0.00025) | 1200 | 0 | $6.82e-13$ | $2.84e-14$ | PASS |
| 3D 50x50_Obstacles Standard Faults (p=0.0005) | 1200 | 0 | $6.82e-13$ | $2.84e-14$ | PASS |
| 3D 50x50_Obstacles High Faults (p=0.00075) | 1200 | 0 | $6.82e-13$ | $1.42e-14$ | PASS |
| 3D 50x50_Obstacles Severe Faults (p=0.0010) | 1200 | 0 | $6.82e-13$ | $1.42e-14$ | PASS |

---

## 8. Before (v2) vs After (v3) Scenario Comparison Tables
*(Note: Per-episode SMEI is reported as the mean of the per-seed `smei` column across 100 seeds.)*

## 2D Obstacle Scenarios Comparison

### 2D 25×25 Obstacles — No Faults (p=0.0)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 32.77 | 35.31 | +2.54 | 27.25 | 13.25 | -14.00 | 29.30 | 21.20 | -8.10 |
| RSPO SelfHeal | 36.33 | 43.23 | +6.90 | 26.00 | 6.75 | -19.25 | 30.19 | 14.94 | -15.26 |
| MAPPO Vanilla | 44.83 | 48.01 | +3.18 | 23.00 | 15.50 | -7.50 | 31.40 | 25.91 | -5.48 |
| MAPPO SelfHeal | 31.95 | 37.93 | +5.98 | 13.00 | 11.25 | -1.75 | 18.14 | 16.84 | -1.30 |
| I-DQN Vanilla | 9.26 | 9.26 | +0.00 | 0.00 | 0.00 | +0.00 | 4.00 | 4.00 | +0.00 |
| I-DQN SelfHeal | 4.35 | 4.35 | +0.00 | 0.00 | 0.00 | +0.00 | 2.91 | 2.91 | +0.00 |
| QMIX Vanilla | 17.77 | 15.12 | -2.65 | 0.00 | 0.00 | +0.00 | 5.11 | 4.79 | -0.32 |
| QMIX SelfHeal | 5.10 | 5.10 | +0.00 | 50.00 | 50.00 | +0.00 | 24.12 | 24.12 | +0.00 |
| COMA Vanilla | 0.76 | 0.76 | +0.00 | 50.00 | 50.00 | +0.00 | 11.25 | 11.25 | +0.00 |
| COMA SelfHeal | 0.38 | 0.38 | +0.00 | 75.00 | 75.00 | +0.00 | 9.77 | 9.77 | +0.00 |
| MAA2C Vanilla | 0.95 | 1.32 | +0.38 | 0.00 | 0.00 | +0.00 | 1.58 | 1.81 | +0.23 |
| MAA2C SelfHeal | 1.32 | 1.89 | +0.57 | 0.00 | 0.00 | +0.00 | 1.81 | 2.09 | +0.28 |

### 2D 25×25 Obstacles — Low Faults (p=0.00025)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 31.69 | 33.37 | +1.69 | 24.75 | 9.00 | -15.75 | 28.36 | 16.34 | -12.02 |
| RSPO SelfHeal | 35.03 | 42.17 | +7.13 | 22.25 | 4.75 | -17.50 | 26.87 | 13.00 | -13.86 |
| MAPPO Vanilla | 43.11 | 45.69 | +2.59 | 18.75 | 13.00 | -5.75 | 28.32 | 23.59 | -4.74 |
| MAPPO SelfHeal | 31.18 | 35.16 | +3.98 | 12.75 | 11.25 | -1.50 | 18.57 | 16.13 | -2.44 |
| I-DQN Vanilla | 9.22 | 9.17 | -0.05 | 0.00 | 0.00 | +0.00 | 4.00 | 3.99 | -0.01 |
| I-DQN SelfHeal | 3.77 | 3.81 | +0.04 | 0.00 | 0.00 | +0.00 | 2.68 | 2.71 | +0.02 |
| QMIX Vanilla | 17.46 | 15.13 | -2.33 | 0.00 | 0.00 | +0.00 | 5.07 | 4.80 | -0.27 |
| QMIX SelfHeal | 5.09 | 5.09 | +0.00 | 40.00 | 40.00 | +0.00 | 21.79 | 21.79 | +0.00 |
| COMA Vanilla | 0.76 | 0.76 | +0.00 | 40.00 | 40.00 | +0.00 | 10.17 | 10.17 | +0.00 |
| COMA SelfHeal | 0.38 | 0.38 | +0.00 | 60.00 | 60.00 | +0.00 | 8.96 | 8.96 | +0.00 |
| MAA2C Vanilla | 0.91 | 1.26 | +0.34 | 0.00 | 0.00 | +0.00 | 1.56 | 1.77 | +0.21 |
| MAA2C SelfHeal | 1.34 | 1.91 | +0.57 | 0.00 | 0.00 | +0.00 | 1.82 | 2.10 | +0.28 |

### 2D 25×25 Obstacles — Standard Faults (p=0.0005)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 30.15 | 32.61 | +2.46 | 19.75 | 6.50 | -13.25 | 24.93 | 13.63 | -11.30 |
| RSPO SelfHeal | 32.65 | 40.99 | +8.33 | 15.25 | 4.75 | -10.50 | 21.33 | 12.70 | -8.62 |
| MAPPO Vanilla | 41.00 | 43.55 | +2.55 | 16.00 | 11.50 | -4.50 | 25.10 | 21.57 | -3.53 |
| MAPPO SelfHeal | 29.21 | 33.26 | +4.05 | 9.50 | 10.50 | +1.00 | 15.50 | 15.72 | +0.22 |
| I-DQN Vanilla | 9.19 | 9.13 | -0.05 | 0.00 | 0.00 | +0.00 | 4.00 | 3.99 | -0.01 |
| I-DQN SelfHeal | 3.50 | 3.55 | +0.05 | 0.00 | 0.00 | +0.00 | 2.58 | 2.61 | +0.03 |
| QMIX Vanilla | 17.31 | 15.39 | -1.92 | 0.00 | 0.00 | +0.00 | 5.06 | 4.84 | -0.22 |
| QMIX SelfHeal | 5.08 | 5.08 | +0.00 | 31.00 | 31.00 | +0.00 | 19.31 | 19.31 | +0.00 |
| COMA Vanilla | 0.76 | 0.76 | +0.00 | 31.00 | 31.00 | +0.00 | 9.02 | 9.02 | +0.00 |
| COMA SelfHeal | 0.38 | 0.38 | +0.00 | 47.75 | 47.75 | +0.00 | 8.17 | 8.17 | +0.00 |
| MAA2C Vanilla | 0.89 | 1.22 | +0.33 | 0.00 | 0.00 | +0.00 | 1.55 | 1.75 | +0.20 |
| MAA2C SelfHeal | 1.38 | 1.91 | +0.53 | 0.00 | 0.00 | +0.00 | 1.84 | 2.10 | +0.26 |

### 2D 25×25 Obstacles — High Faults (p=0.00075)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 29.26 | 32.32 | +3.06 | 16.50 | 6.75 | -9.75 | 21.64 | 14.09 | -7.55 |
| RSPO SelfHeal | 30.64 | 40.09 | +9.45 | 13.25 | 4.00 | -9.25 | 19.12 | 12.25 | -6.87 |
| MAPPO Vanilla | 39.04 | 42.44 | +3.40 | 14.00 | 10.25 | -3.75 | 22.95 | 19.74 | -3.21 |
| MAPPO SelfHeal | 27.90 | 32.18 | +4.29 | 10.50 | 8.75 | -1.75 | 14.97 | 13.18 | -1.78 |
| I-DQN Vanilla | 9.19 | 9.13 | -0.05 | 0.00 | 0.00 | +0.00 | 4.01 | 3.99 | -0.01 |
| I-DQN SelfHeal | 3.16 | 3.26 | +0.10 | 0.00 | 0.00 | +0.00 | 2.44 | 2.50 | +0.05 |
| QMIX Vanilla | 16.96 | 15.11 | -1.85 | 0.00 | 0.00 | +0.00 | 5.02 | 4.80 | -0.22 |
| QMIX SelfHeal | 5.08 | 5.08 | +0.00 | 26.75 | 26.75 | +0.00 | 17.52 | 17.52 | +0.00 |
| COMA Vanilla | 0.76 | 0.76 | +0.00 | 26.75 | 26.75 | +0.00 | 8.19 | 8.19 | +0.00 |
| COMA SelfHeal | 0.38 | 0.38 | +0.00 | 40.50 | 40.50 | +0.00 | 7.66 | 7.66 | +0.00 |
| MAA2C Vanilla | 0.88 | 1.18 | +0.31 | 0.00 | 0.00 | +0.00 | 1.54 | 1.73 | +0.19 |
| MAA2C SelfHeal | 1.39 | 1.94 | +0.55 | 0.00 | 0.00 | +0.00 | 1.85 | 2.11 | +0.26 |

### 2D 25×25 Obstacles — Severe Faults (p=0.0010)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 28.73 | 31.72 | +2.99 | 13.25 | 6.75 | -6.50 | 18.57 | 14.02 | -4.56 |
| RSPO SelfHeal | 30.00 | 38.53 | +8.52 | 12.25 | 3.25 | -9.00 | 17.97 | 11.19 | -6.78 |
| MAPPO Vanilla | 37.85 | 41.43 | +3.58 | 11.50 | 9.00 | -2.50 | 20.91 | 18.44 | -2.47 |
| MAPPO SelfHeal | 26.92 | 31.12 | +4.19 | 9.00 | 6.50 | -2.50 | 14.14 | 11.87 | -2.27 |
| I-DQN Vanilla | 9.19 | 9.13 | -0.05 | 0.00 | 0.00 | +0.00 | 4.01 | 4.00 | -0.01 |
| I-DQN SelfHeal | 2.91 | 3.04 | +0.12 | 0.00 | 0.00 | +0.00 | 2.34 | 2.41 | +0.07 |
| QMIX Vanilla | 16.63 | 14.98 | -1.66 | 0.00 | 0.00 | +0.00 | 4.99 | 4.79 | -0.20 |
| QMIX SelfHeal | 5.07 | 5.07 | +0.00 | 22.50 | 22.50 | +0.00 | 15.28 | 15.28 | +0.00 |
| COMA Vanilla | 0.76 | 0.76 | +0.00 | 22.50 | 22.50 | +0.00 | 7.15 | 7.15 | +0.00 |
| COMA SelfHeal | 0.38 | 0.38 | +0.00 | 34.00 | 34.00 | +0.00 | 6.93 | 6.93 | +0.00 |
| MAA2C Vanilla | 0.86 | 1.15 | +0.29 | 0.00 | 0.00 | +0.00 | 1.53 | 1.72 | +0.19 |
| MAA2C SelfHeal | 1.42 | 1.95 | +0.53 | 0.00 | 0.00 | +0.00 | 1.87 | 2.12 | +0.25 |

### 2D 50×50 Obstacles — No Faults (p=0.0)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 26.20 | 26.53 | +0.33 | 18.50 | 15.75 | -2.75 | 22.58 | 20.85 | -1.73 |
| RSPO SelfHeal | 30.70 | 28.70 | -1.99 | 4.75 | 6.00 | +1.25 | 11.56 | 12.73 | +1.17 |
| MAPPO Vanilla | 37.85 | 38.08 | +0.24 | 31.25 | 33.00 | +1.75 | 34.50 | 36.08 | +1.58 |
| MAPPO SelfHeal | 33.81 | 34.53 | +0.73 | 5.50 | 4.75 | -0.75 | 12.84 | 12.45 | -0.39 |
| I-DQN Vanilla | 4.85 | 4.85 | +0.00 | 0.00 | 0.00 | +0.00 | 3.11 | 3.11 | +0.00 |
| I-DQN SelfHeal | 2.06 | 2.06 | +0.00 | 0.00 | 0.00 | +0.00 | 2.16 | 2.16 | +0.00 |
| QMIX Vanilla | 8.23 | 9.60 | +1.37 | 0.00 | 0.00 | +0.00 | 3.76 | 4.00 | +0.24 |
| QMIX SelfHeal | 4.07 | 4.07 | +0.00 | 50.00 | 50.00 | +0.00 | 21.98 | 21.98 | +0.00 |
| COMA Vanilla | 0.20 | 0.20 | +0.00 | 50.00 | 50.00 | +0.00 | 6.54 | 6.54 | +0.00 |
| COMA SelfHeal | 0.10 | 0.10 | +0.00 | 75.00 | 75.00 | +0.00 | 5.74 | 5.74 | +0.00 |
| MAA2C Vanilla | 1.22 | 1.22 | +0.00 | 0.00 | 0.00 | +0.00 | 1.76 | 1.76 | +0.00 |
| MAA2C SelfHeal | 0.49 | 0.49 | +0.00 | 0.00 | 0.00 | +0.00 | 1.22 | 1.22 | +0.00 |

### 2D 50×50 Obstacles — Low Faults (p=0.00025)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 23.64 | 23.81 | +0.17 | 10.75 | 9.00 | -1.75 | 16.31 | 14.69 | -1.63 |
| RSPO SelfHeal | 28.26 | 26.56 | -1.70 | 3.00 | 4.25 | +1.25 | 9.52 | 10.84 | +1.32 |
| MAPPO Vanilla | 32.74 | 32.66 | -0.08 | 19.00 | 21.00 | +2.00 | 26.82 | 28.12 | +1.30 |
| MAPPO SelfHeal | 29.85 | 29.70 | -0.15 | 4.00 | 2.75 | -1.25 | 10.69 | 9.67 | -1.02 |
| I-DQN Vanilla | 4.78 | 4.78 | +0.00 | 0.00 | 0.00 | +0.00 | 3.11 | 3.11 | +0.00 |
| I-DQN SelfHeal | 1.62 | 1.62 | +0.00 | 0.00 | 0.00 | +0.00 | 1.87 | 1.87 | +0.00 |
| QMIX Vanilla | 7.97 | 9.02 | +1.05 | 0.00 | 0.00 | +0.00 | 3.71 | 3.89 | +0.18 |
| QMIX SelfHeal | 4.03 | 4.03 | +0.00 | 32.75 | 32.75 | +0.00 | 17.77 | 17.77 | +0.00 |
| COMA Vanilla | 0.20 | 0.20 | +0.00 | 32.75 | 32.75 | +0.00 | 5.31 | 5.31 | +0.00 |
| COMA SelfHeal | 0.10 | 0.10 | +0.00 | 47.50 | 47.50 | +0.00 | 4.71 | 4.71 | +0.00 |
| MAA2C Vanilla | 1.12 | 1.12 | +0.00 | 0.00 | 0.00 | +0.00 | 1.68 | 1.68 | +0.00 |
| MAA2C SelfHeal | 0.51 | 0.51 | +0.00 | 0.00 | 0.00 | +0.00 | 1.24 | 1.24 | +0.00 |

### 2D 50×50 Obstacles — Standard Faults (p=0.0005)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 21.30 | 21.28 | -0.02 | 10.75 | 10.50 | -0.25 | 15.95 | 15.66 | -0.28 |
| RSPO SelfHeal | 24.80 | 23.86 | -0.94 | 2.50 | 2.75 | +0.25 | 8.46 | 8.80 | +0.34 |
| MAPPO Vanilla | 29.72 | 29.74 | +0.02 | 17.00 | 15.25 | -1.75 | 24.09 | 23.01 | -1.09 |
| MAPPO SelfHeal | 26.43 | 26.31 | -0.12 | 1.50 | 1.75 | +0.25 | 7.81 | 8.07 | +0.26 |
| I-DQN Vanilla | 4.71 | 4.71 | +0.00 | 0.00 | 0.00 | +0.00 | 3.10 | 3.10 | +0.00 |
| I-DQN SelfHeal | 1.21 | 1.21 | +0.00 | 0.00 | 0.00 | +0.00 | 1.61 | 1.61 | +0.00 |
| QMIX Vanilla | 7.84 | 8.57 | +0.74 | 0.00 | 0.00 | +0.00 | 3.68 | 3.81 | +0.13 |
| QMIX SelfHeal | 4.01 | 4.01 | +0.00 | 21.00 | 21.00 | +0.00 | 13.70 | 13.70 | +0.00 |
| COMA Vanilla | 0.20 | 0.20 | +0.00 | 21.00 | 21.00 | +0.00 | 4.11 | 4.11 | +0.00 |
| COMA SelfHeal | 0.10 | 0.10 | +0.00 | 32.25 | 32.25 | +0.00 | 4.07 | 4.07 | +0.00 |
| MAA2C Vanilla | 1.06 | 1.06 | +0.00 | 0.00 | 0.00 | +0.00 | 1.63 | 1.63 | +0.00 |
| MAA2C SelfHeal | 0.51 | 0.51 | +0.00 | 0.00 | 0.00 | +0.00 | 1.25 | 1.25 | +0.00 |

### 2D 50×50 Obstacles — High Faults (p=0.00075)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 19.07 | 19.24 | +0.17 | 8.25 | 8.00 | -0.25 | 13.33 | 13.24 | -0.09 |
| RSPO SelfHeal | 23.24 | 22.53 | -0.71 | 1.75 | 1.25 | -0.50 | 7.57 | 6.94 | -0.63 |
| MAPPO Vanilla | 27.29 | 27.25 | -0.04 | 11.25 | 11.00 | -0.25 | 18.38 | 18.29 | -0.09 |
| MAPPO SelfHeal | 24.11 | 24.16 | +0.05 | 1.50 | 2.00 | +0.50 | 7.60 | 8.22 | +0.61 |
| I-DQN Vanilla | 4.62 | 4.62 | +0.00 | 0.00 | 0.00 | +0.00 | 3.08 | 3.08 | +0.00 |
| I-DQN SelfHeal | 1.01 | 1.01 | +0.00 | 0.00 | 0.00 | +0.00 | 1.48 | 1.48 | +0.00 |
| QMIX Vanilla | 7.32 | 7.95 | +0.63 | 0.00 | 0.00 | +0.00 | 3.57 | 3.68 | +0.11 |
| QMIX SelfHeal | 3.95 | 3.95 | +0.00 | 15.00 | 15.00 | +0.00 | 10.97 | 10.97 | +0.00 |
| COMA Vanilla | 0.20 | 0.20 | +0.00 | 15.00 | 15.00 | +0.00 | 3.32 | 3.32 | +0.00 |
| COMA SelfHeal | 0.10 | 0.10 | +0.00 | 22.75 | 22.75 | +0.00 | 3.26 | 3.26 | +0.00 |
| MAA2C Vanilla | 0.98 | 0.98 | +0.00 | 0.00 | 0.00 | +0.00 | 1.57 | 1.57 | +0.00 |
| MAA2C SelfHeal | 0.51 | 0.51 | +0.00 | 0.00 | 0.00 | +0.00 | 1.25 | 1.25 | +0.00 |

### 2D 50×50 Obstacles — Severe Faults (p=0.0010)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 17.45 | 17.79 | +0.34 | 5.50 | 5.25 | -0.25 | 10.20 | 10.13 | -0.08 |
| RSPO SelfHeal | 21.81 | 21.54 | -0.27 | 0.75 | 0.50 | -0.25 | 6.45 | 6.19 | -0.27 |
| MAPPO Vanilla | 25.45 | 25.10 | -0.34 | 7.00 | 6.50 | -0.50 | 13.83 | 13.34 | -0.49 |
| MAPPO SelfHeal | 22.39 | 22.57 | +0.18 | 0.50 | 1.00 | +0.50 | 6.25 | 6.86 | +0.60 |
| I-DQN Vanilla | 4.62 | 4.62 | +0.00 | 0.25 | 0.25 | +0.00 | 3.13 | 3.13 | +0.00 |
| I-DQN SelfHeal | 0.88 | 0.88 | +0.00 | 0.00 | 0.00 | +0.00 | 1.40 | 1.40 | +0.00 |
| QMIX Vanilla | 7.21 | 7.74 | +0.53 | 0.00 | 0.00 | +0.00 | 3.55 | 3.64 | +0.09 |
| QMIX SelfHeal | 3.94 | 3.94 | +0.00 | 11.00 | 11.00 | +0.00 | 8.70 | 8.70 | +0.00 |
| COMA Vanilla | 0.20 | 0.20 | +0.00 | 11.00 | 11.00 | +0.00 | 2.63 | 2.63 | +0.00 |
| COMA SelfHeal | 0.10 | 0.10 | +0.00 | 13.25 | 13.25 | +0.00 | 2.23 | 2.23 | +0.00 |
| MAA2C Vanilla | 0.91 | 0.91 | +0.00 | 0.00 | 0.00 | +0.00 | 1.52 | 1.52 | +0.00 |
| MAA2C SelfHeal | 0.51 | 0.51 | +0.00 | 0.00 | 0.00 | +0.00 | 1.25 | 1.25 | +0.00 |

## 3D Obstacle Scenarios Comparison

### 3D 25×25 Obstacles — No Faults (p=0.0)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 98.33 | 98.75 | +0.42 | 62.25 | 96.50 | +34.25 | 66.19 | 82.72 | +16.53 |
| RSPO SelfHeal | 97.23 | 98.78 | +1.55 | 55.00 | 99.00 | +44.00 | 64.03 | 83.46 | +19.44 |
| MAPPO Vanilla | 95.74 | 97.66 | +1.91 | 34.00 | 43.75 | +9.75 | 56.86 | 59.15 | +2.29 |
| MAPPO SelfHeal | 94.01 | 95.75 | +1.75 | 30.75 | 94.00 | +63.25 | 46.81 | 81.64 | +34.83 |
| I-DQN Vanilla | 12.67 | 12.67 | +0.00 | 0.00 | 0.00 | +0.00 | 4.53 | 4.53 | +0.00 |
| I-DQN SelfHeal | 40.08 | 40.08 | +0.00 | 0.00 | 0.00 | +0.00 | 7.08 | 7.08 | +0.00 |
| QMIX Vanilla | 74.10 | 75.05 | +0.95 | 0.00 | 0.00 | +0.00 | 9.05 | 9.09 | +0.05 |
| QMIX SelfHeal | 54.06 | 54.06 | +0.00 | 0.00 | 0.00 | +0.00 | 7.78 | 7.78 | +0.00 |
| COMA Vanilla | 1.70 | 1.70 | +0.00 | 50.00 | 50.00 | +0.00 | 15.56 | 15.56 | +0.00 |
| COMA SelfHeal | 1.51 | 1.51 | +0.00 | 75.00 | 75.00 | +0.00 | 17.02 | 17.02 | +0.00 |
| MAA2C Vanilla | 53.50 | 52.93 | -0.57 | 0.00 | 0.00 | +0.00 | 7.94 | 7.91 | -0.03 |
| MAA2C SelfHeal | 52.93 | 56.33 | +3.40 | 0.00 | 0.00 | +0.00 | 7.91 | 8.11 | +0.20 |

### 3D 25×25 Obstacles — Low Faults (p=0.00025)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 96.83 | 97.26 | +0.43 | 44.25 | 79.50 | +35.25 | 54.72 | 76.51 | +21.78 |
| RSPO SelfHeal | 96.02 | 97.59 | +1.57 | 43.00 | 74.75 | +31.75 | 55.04 | 73.54 | +18.50 |
| MAPPO Vanilla | 93.55 | 95.87 | +2.32 | 30.50 | 43.00 | +12.50 | 53.84 | 59.69 | +5.85 |
| MAPPO SelfHeal | 91.11 | 94.88 | +3.77 | 24.00 | 71.00 | +47.00 | 40.01 | 71.42 | +31.40 |
| I-DQN Vanilla | 12.76 | 12.76 | +0.00 | 0.00 | 0.00 | +0.00 | 4.55 | 4.55 | +0.00 |
| I-DQN SelfHeal | 38.93 | 38.99 | +0.05 | 0.00 | 0.00 | +0.00 | 7.00 | 7.01 | +0.00 |
| QMIX Vanilla | 72.44 | 73.99 | +1.55 | 0.00 | 0.00 | +0.00 | 8.98 | 9.06 | +0.08 |
| QMIX SelfHeal | 52.43 | 53.21 | +0.78 | 1.00 | 1.00 | +0.00 | 8.28 | 8.32 | +0.04 |
| COMA Vanilla | 1.70 | 1.70 | +0.00 | 40.00 | 40.00 | +0.00 | 14.07 | 14.07 | +0.00 |
| COMA SelfHeal | 1.51 | 1.51 | +0.00 | 60.00 | 60.00 | +0.00 | 15.59 | 15.59 | +0.00 |
| MAA2C Vanilla | 51.65 | 51.26 | -0.38 | 0.00 | 0.00 | +0.00 | 7.83 | 7.81 | -0.02 |
| MAA2C SelfHeal | 52.42 | 56.06 | +3.64 | 0.00 | 0.00 | +0.00 | 7.89 | 8.11 | +0.22 |

### 3D 25×25 Obstacles — Standard Faults (p=0.0005)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 95.28 | 97.05 | +1.77 | 29.75 | 65.50 | +35.75 | 44.47 | 70.79 | +26.32 |
| RSPO SelfHeal | 95.34 | 96.81 | +1.47 | 32.50 | 58.00 | +25.50 | 48.79 | 66.61 | +17.82 |
| MAPPO Vanilla | 92.59 | 95.57 | +2.98 | 30.50 | 39.25 | +8.75 | 53.62 | 56.27 | +2.65 |
| MAPPO SelfHeal | 88.75 | 93.77 | +5.02 | 16.75 | 50.00 | +33.25 | 31.78 | 59.15 | +27.37 |
| I-DQN Vanilla | 12.74 | 12.74 | +0.00 | 0.00 | 0.00 | +0.00 | 4.56 | 4.56 | +0.00 |
| I-DQN SelfHeal | 38.15 | 38.20 | +0.05 | 0.00 | 0.00 | +0.00 | 6.95 | 6.95 | +0.00 |
| QMIX Vanilla | 71.48 | 73.53 | +2.05 | 0.00 | 0.00 | +0.00 | 8.94 | 9.05 | +0.11 |
| QMIX SelfHeal | 51.43 | 52.63 | +1.20 | 1.50 | 1.25 | -0.25 | 8.76 | 8.49 | -0.28 |
| COMA Vanilla | 1.70 | 1.70 | +0.00 | 31.00 | 31.00 | +0.00 | 12.48 | 12.48 | +0.00 |
| COMA SelfHeal | 1.51 | 1.51 | +0.00 | 47.75 | 47.75 | +0.00 | 14.22 | 14.22 | +0.00 |
| MAA2C Vanilla | 50.33 | 50.11 | -0.22 | 0.00 | 0.00 | +0.00 | 7.76 | 7.75 | -0.01 |
| MAA2C SelfHeal | 52.39 | 56.13 | +3.74 | 0.00 | 0.00 | +0.00 | 7.90 | 8.12 | +0.22 |

### 3D 25×25 Obstacles — High Faults (p=0.00075)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 93.41 | 95.46 | +2.04 | 21.75 | 56.50 | +34.75 | 38.18 | 66.38 | +28.20 |
| RSPO SelfHeal | 93.29 | 95.11 | +1.82 | 25.50 | 49.25 | +23.75 | 42.00 | 61.17 | +19.16 |
| MAPPO Vanilla | 91.56 | 93.97 | +2.41 | 27.75 | 34.50 | +6.75 | 49.87 | 53.68 | +3.81 |
| MAPPO SelfHeal | 85.12 | 92.68 | +7.56 | 12.75 | 42.50 | +29.75 | 27.06 | 55.03 | +27.97 |
| I-DQN Vanilla | 12.74 | 12.74 | +0.00 | 0.00 | 0.00 | +0.00 | 4.56 | 4.56 | +0.00 |
| I-DQN SelfHeal | 37.09 | 37.15 | +0.05 | 0.00 | 0.00 | +0.00 | 6.88 | 6.88 | +0.00 |
| QMIX Vanilla | 70.24 | 73.36 | +3.12 | 0.00 | 0.00 | +0.00 | 8.89 | 9.06 | +0.17 |
| QMIX SelfHeal | 48.93 | 50.29 | +1.36 | 2.25 | 2.00 | -0.25 | 9.24 | 8.98 | -0.26 |
| COMA Vanilla | 1.70 | 1.70 | +0.00 | 26.75 | 26.75 | +0.00 | 11.33 | 11.33 | +0.00 |
| COMA SelfHeal | 1.51 | 1.51 | +0.00 | 40.50 | 40.50 | +0.00 | 13.34 | 13.34 | +0.00 |
| MAA2C Vanilla | 48.59 | 48.43 | -0.16 | 0.00 | 0.00 | +0.00 | 7.65 | 7.65 | -0.01 |
| MAA2C SelfHeal | 52.02 | 55.95 | +3.92 | 0.00 | 0.00 | +0.00 | 7.89 | 8.13 | +0.24 |

### 3D 25×25 Obstacles — Severe Faults (p=0.0010)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 92.43 | 94.34 | +1.91 | 17.50 | 44.75 | +27.25 | 33.63 | 60.02 | +26.39 |
| RSPO SelfHeal | 92.43 | 94.10 | +1.67 | 21.00 | 39.25 | +18.25 | 36.70 | 54.21 | +17.50 |
| MAPPO Vanilla | 89.92 | 92.80 | +2.88 | 24.50 | 29.75 | +5.25 | 46.79 | 49.85 | +3.06 |
| MAPPO SelfHeal | 83.72 | 91.51 | +7.78 | 10.25 | 33.25 | +23.00 | 24.54 | 48.76 | +24.21 |
| I-DQN Vanilla | 12.74 | 12.74 | +0.00 | 0.00 | 0.00 | +0.00 | 4.57 | 4.57 | +0.00 |
| I-DQN SelfHeal | 36.31 | 36.37 | +0.05 | 0.00 | 0.00 | +0.00 | 6.83 | 6.83 | +0.00 |
| QMIX Vanilla | 69.71 | 73.32 | +3.61 | 0.00 | 0.00 | +0.00 | 8.87 | 9.07 | +0.19 |
| QMIX SelfHeal | 49.31 | 50.72 | +1.41 | 1.75 | 1.50 | -0.25 | 8.83 | 8.59 | -0.23 |
| COMA Vanilla | 1.70 | 1.70 | +0.00 | 22.50 | 22.50 | +0.00 | 9.89 | 9.89 | +0.00 |
| COMA SelfHeal | 1.51 | 1.51 | +0.00 | 34.00 | 34.00 | +0.00 | 12.07 | 12.07 | +0.00 |
| MAA2C Vanilla | 47.45 | 47.44 | -0.01 | 0.00 | 0.00 | +0.00 | 7.59 | 7.59 | +0.00 |
| MAA2C SelfHeal | 51.48 | 55.72 | +4.24 | 0.00 | 0.00 | +0.00 | 7.86 | 8.12 | +0.26 |

### 3D 50×50 Obstacles — No Faults (p=0.0)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 94.77 | 94.65 | -0.11 | 79.50 | 83.25 | +3.75 | 74.82 | 76.61 | +1.78 |
| RSPO SelfHeal | 92.71 | 93.44 | +0.73 | 75.75 | 89.00 | +13.25 | 72.25 | 77.09 | +4.83 |
| MAPPO Vanilla | 92.26 | 93.58 | +1.32 | 50.50 | 61.75 | +11.25 | 63.54 | 67.31 | +3.77 |
| MAPPO SelfHeal | 90.10 | 91.76 | +1.66 | 60.00 | 85.50 | +25.50 | 60.84 | 71.26 | +10.42 |
| I-DQN Vanilla | 7.35 | 7.35 | +0.00 | 0.00 | 0.00 | +0.00 | 3.67 | 3.67 | +0.00 |
| I-DQN SelfHeal | 22.34 | 22.34 | +0.00 | 0.00 | 0.00 | +0.00 | 5.61 | 5.61 | +0.00 |
| QMIX Vanilla | 39.15 | 39.15 | +0.00 | 0.00 | 0.00 | +0.00 | 7.02 | 7.02 | +0.00 |
| QMIX SelfHeal | 6.61 | 6.61 | +0.00 | 50.00 | 50.00 | +0.00 | 26.70 | 26.70 | +0.00 |
| COMA Vanilla | 0.44 | 0.44 | +0.00 | 50.00 | 50.00 | +0.00 | 9.05 | 9.05 | +0.00 |
| COMA SelfHeal | 0.39 | 0.39 | +0.00 | 75.00 | 75.00 | +0.00 | 9.92 | 9.92 | +0.00 |
| MAA2C Vanilla | 18.23 | 18.23 | +0.00 | 0.00 | 0.00 | +0.00 | 5.17 | 5.17 | +0.00 |
| MAA2C SelfHeal | 37.92 | 37.92 | +0.00 | 0.00 | 0.00 | +0.00 | 6.93 | 6.93 | +0.00 |

### 3D 50×50 Obstacles — Low Faults (p=0.00025)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 89.55 | 91.40 | +1.85 | 49.00 | 55.75 | +6.75 | 57.75 | 65.42 | +7.67 |
| RSPO SelfHeal | 86.70 | 88.19 | +1.48 | 38.25 | 47.50 | +9.25 | 50.36 | 57.49 | +7.12 |
| MAPPO Vanilla | 84.38 | 84.30 | -0.09 | 36.00 | 40.00 | +4.00 | 55.12 | 56.66 | +1.54 |
| MAPPO SelfHeal | 84.72 | 87.49 | +2.77 | 30.75 | 52.00 | +21.25 | 45.65 | 57.80 | +12.15 |
| I-DQN Vanilla | 7.41 | 7.41 | +0.00 | 0.00 | 0.00 | +0.00 | 3.70 | 3.70 | +0.00 |
| I-DQN SelfHeal | 21.23 | 21.27 | +0.04 | 0.00 | 0.00 | +0.00 | 5.51 | 5.51 | +0.00 |
| QMIX Vanilla | 37.91 | 37.95 | +0.04 | 0.00 | 0.00 | +0.00 | 6.94 | 6.94 | +0.00 |
| QMIX SelfHeal | 6.55 | 6.55 | +0.00 | 32.75 | 32.75 | +0.00 | 21.58 | 21.58 | +0.00 |
| COMA Vanilla | 0.44 | 0.44 | +0.00 | 32.75 | 32.75 | +0.00 | 7.35 | 7.35 | +0.00 |
| COMA SelfHeal | 0.39 | 0.39 | +0.00 | 47.50 | 47.50 | +0.00 | 8.14 | 8.14 | +0.00 |
| MAA2C Vanilla | 18.15 | 18.16 | +0.02 | 0.00 | 0.00 | +0.00 | 5.18 | 5.18 | +0.00 |
| MAA2C SelfHeal | 37.80 | 37.93 | +0.13 | 0.00 | 0.00 | +0.00 | 6.94 | 6.95 | +0.01 |

### 3D 50×50 Obstacles — Standard Faults (p=0.0005)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 85.15 | 87.60 | +2.45 | 29.50 | 38.50 | +9.00 | 44.53 | 53.73 | +9.20 |
| RSPO SelfHeal | 81.76 | 83.05 | +1.29 | 22.25 | 25.75 | +3.50 | 36.64 | 40.80 | +4.16 |
| MAPPO Vanilla | 79.54 | 79.49 | -0.06 | 24.25 | 27.75 | +3.50 | 43.05 | 46.89 | +3.84 |
| MAPPO SelfHeal | 78.13 | 80.78 | +2.65 | 15.50 | 27.50 | +12.00 | 30.38 | 39.25 | +8.87 |
| I-DQN Vanilla | 7.42 | 7.42 | +0.00 | 0.00 | 0.00 | +0.00 | 3.73 | 3.73 | +0.00 |
| I-DQN SelfHeal | 20.35 | 20.39 | +0.04 | 0.00 | 0.00 | +0.00 | 5.43 | 5.44 | +0.00 |
| QMIX Vanilla | 36.56 | 36.61 | +0.05 | 0.00 | 0.00 | +0.00 | 6.86 | 6.86 | +0.00 |
| QMIX SelfHeal | 6.49 | 6.49 | +0.00 | 21.00 | 21.00 | +0.00 | 16.61 | 16.61 | +0.00 |
| COMA Vanilla | 0.44 | 0.44 | +0.00 | 21.00 | 21.00 | +0.00 | 5.68 | 5.68 | +0.00 |
| COMA SelfHeal | 0.39 | 0.39 | +0.00 | 32.25 | 32.25 | +0.00 | 7.02 | 7.02 | +0.00 |
| MAA2C Vanilla | 18.07 | 18.10 | +0.03 | 0.00 | 0.00 | +0.00 | 5.19 | 5.19 | +0.00 |
| MAA2C SelfHeal | 36.44 | 36.78 | +0.34 | 0.00 | 0.00 | +0.00 | 6.86 | 6.88 | +0.02 |

### 3D 50×50 Obstacles — High Faults (p=0.00075)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 81.02 | 83.11 | +2.09 | 19.25 | 27.50 | +8.25 | 33.93 | 42.78 | +8.85 |
| RSPO SelfHeal | 79.10 | 80.44 | +1.34 | 18.50 | 19.00 | +0.50 | 33.95 | 34.85 | +0.90 |
| MAPPO Vanilla | 74.61 | 73.28 | -1.33 | 19.75 | 23.50 | +3.75 | 37.43 | 41.96 | +4.53 |
| MAPPO SelfHeal | 74.24 | 76.92 | +2.67 | 11.75 | 17.00 | +5.25 | 26.11 | 31.27 | +5.17 |
| I-DQN Vanilla | 7.42 | 7.42 | +0.00 | 0.00 | 0.00 | +0.00 | 3.74 | 3.74 | +0.00 |
| I-DQN SelfHeal | 19.77 | 19.81 | +0.04 | 0.00 | 0.00 | +0.00 | 5.39 | 5.39 | +0.00 |
| QMIX Vanilla | 35.20 | 35.18 | -0.02 | 0.00 | 0.00 | +0.00 | 6.76 | 6.76 | -0.00 |
| QMIX SelfHeal | 6.39 | 6.39 | +0.00 | 15.00 | 15.00 | +0.00 | 13.29 | 13.29 | +0.00 |
| COMA Vanilla | 0.44 | 0.44 | +0.00 | 15.00 | 15.00 | +0.00 | 4.59 | 4.59 | +0.00 |
| COMA SelfHeal | 0.39 | 0.39 | +0.00 | 22.75 | 22.75 | +0.00 | 5.63 | 5.63 | +0.00 |
| MAA2C Vanilla | 17.90 | 17.94 | +0.05 | 0.00 | 0.00 | +0.00 | 5.19 | 5.20 | +0.01 |
| MAA2C SelfHeal | 35.65 | 36.00 | +0.35 | 0.00 | 0.00 | +0.00 | 6.82 | 6.84 | +0.02 |

### 3D 50×50 Obstacles — Severe Faults (p=0.0010)

| Model | v2 Cov (%) | v3 Cov (%) | Δ Cov | v2 Surv (%) | v3 Surv (%) | Δ Surv | v2 SMEI | v3 SMEI | Δ SMEI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| RSPO Vanilla | 75.46 | 77.65 | +2.19 | 12.75 | 16.75 | +4.00 | 27.60 | 32.66 | +5.06 |
| RSPO SelfHeal | 76.94 | 77.97 | +1.02 | 11.50 | 13.00 | +1.50 | 26.07 | 28.88 | +2.81 |
| MAPPO Vanilla | 69.65 | 68.94 | -0.72 | 12.00 | 14.50 | +2.50 | 27.86 | 31.65 | +3.79 |
| MAPPO SelfHeal | 70.69 | 72.58 | +1.89 | 6.50 | 9.75 | +3.25 | 19.47 | 24.17 | +4.70 |
| I-DQN Vanilla | 7.44 | 7.44 | +0.00 | 0.25 | 0.25 | +0.00 | 3.83 | 3.83 | +0.00 |
| I-DQN SelfHeal | 19.26 | 19.30 | +0.04 | 0.00 | 0.00 | +0.00 | 5.35 | 5.35 | +0.00 |
| QMIX Vanilla | 34.55 | 34.50 | -0.05 | 0.00 | 0.00 | +0.00 | 6.73 | 6.73 | -0.00 |
| QMIX SelfHeal | 6.38 | 6.38 | +0.00 | 11.00 | 11.00 | +0.00 | 10.54 | 10.54 | +0.00 |
| COMA Vanilla | 0.44 | 0.44 | +0.00 | 11.00 | 11.00 | +0.00 | 3.64 | 3.64 | +0.00 |
| COMA SelfHeal | 0.39 | 0.39 | +0.00 | 13.25 | 13.25 | +0.00 | 3.86 | 3.86 | +0.00 |
| MAA2C Vanilla | 17.75 | 17.80 | +0.04 | 0.00 | 0.00 | +0.00 | 5.19 | 5.19 | +0.01 |
| MAA2C SelfHeal | 35.58 | 36.02 | +0.45 | 0.00 | 0.00 | +0.00 | 6.83 | 6.86 | +0.03 |
