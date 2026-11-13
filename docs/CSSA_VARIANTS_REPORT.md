# RSPO Crash-Sector Spatial Attention (CSSA) Inference-Time Defect Investigation Report

## 1. Executive Summary & Provenance

- **Branch:** `cssa-inference-variants`
- **Commit Hash:** `2f7374a2ce92ec001a248780b5428b2258f8b8c8`
- **Base Commit:** `213be590aabc5c3206aefa56f5a920a9e0bc0d9f`
- **Evaluation Runner:** `src/run_master_final_benchmarks.py` (extended with `--cssa_mode` and `--models` filters)
- **Models Evaluated:** `RSPO Vanilla` and `RSPO SelfHeal` (all other models and environment code remain identical)
- **Checkpoints Used (Identical to v3 Benchmark):**
  - RSPO Vanilla: `ray_res/DSSE_Coverage/RSPO_V2_Vanilla_RSPO_v2/RSPOPPO_DSSE_Coverage_RSPO_V2_Vanilla_45eb7_00000_0_2026-09-04_06-38-43/checkpoint_000136`
  - RSPO SelfHeal: `ray_res/DSSE_Coverage/RSPO_V2_SelfHeal_rspo_v2_selfheal/RSPOPPO_DSSE_Coverage_RSPO_V2_SelfHeal_86fd5_00000_0_2026-09-04_06-40-32/checkpoint_000122`
- **Seeds Evaluated:** 100 seeds per scenario (seeds 1000–1099)
- **Scenarios Evaluated:** All 40 scenarios (20 in 2D, 20 in 3D: 25×25 / 50×50; Standard / Obstacles; $p \in \{0, 0.00025, 0.0005, 0.00075, 0.001\}$)

### Execution Commands & Wall-Clock Times

1. **`off` Mode:**
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE python3 src/run_master_final_benchmarks.py \
       --dim both --n_seeds 100 --cssa_mode off \
       --models "RSPO Vanilla,RSPO SelfHeal" \
       --output_base /Users/AdhamMotawi/Desktop/Results/CSSA-variants/off \
       --skip_post_processing
   ```
   - **2D Suite Runtime:** 130.69 minutes (Start: 2026-11-08 12:57:24, End: 2026-11-08 15:08:05)
   - **3D Suite Runtime:** 182.74 minutes (Start: 2026-11-08 15:08:05, End: 2026-11-08 18:10:50)
   - **Total `off` Runtime:** 313.43 minutes (5.22 hours)

2. **`corrected` Mode:**
   ```bash
   KMP_DUPLICATE_LIB_OK=TRUE python3 src/run_master_final_benchmarks.py \
       --dim both --n_seeds 100 --cssa_mode corrected \
       --models "RSPO Vanilla,RSPO SelfHeal" \
       --output_base /Users/AdhamMotawi/Desktop/Results/CSSA-variants/corrected \
       --skip_post_processing
   ```
   - **2D Suite Runtime:** 152.72 minutes (Start: 2026-11-08 12:57:34, End: 2026-11-08 15:30:17)
   - **3D Suite Runtime:** 189.65 minutes (Start: 2026-11-08 15:30:17, End: 2026-11-08 18:39:56)
   - **Total `corrected` Runtime:** 342.37 minutes (5.71 hours)

---

## 2. Pre-Run Verification Results

### Check 2a: Baseline Reproducibility with `--cssa_mode trained`
RSPO Vanilla was evaluated on 2D `25x25_Standard`, $p = 0.0005$ (`Active_Faults_0.0005`), seeds 1000–1099, with `--cssa_mode trained` and compared per-seed against the locked reference in `Results/2D-final-v3/Coverage/25x25_Standard/Active_Faults_0.0005/coverage_2d_raw_seed_by_seed.csv`:

| Metric | Maximum Absolute Difference (Seeds 1000–1099) | Verification Status |
| :--- | :--- | :--- |
| `coverage_pct` | $7.1054 \times 10^{-15}$ | **PASS** (Machine Epsilon) |
| `survival_rate` | $0.0000 \times 10^{0}$ | **PASS** (Exact 0.0) |
| `avg_battery` | $7.1054 \times 10^{-15}$ | **PASS** (Machine Epsilon) |
| `smei` | $7.1054 \times 10^{-15}$ | **PASS** (Machine Epsilon) |

Overall maximum absolute difference across all metrics: **$7.1054 \times 10^{-15}$**.

### Check 2b: Unit Verification of Attention Maps on Crash at $(x = 3, y = 20)$
Constructed a normalized telemetry vector on $25 \times 25$ with drone 0 deceased at coordinates $(x = 3, y = 20)$ ($x_{\text{norm}} = 3/25 = 0.12, y_{\text{norm}} = 20/25 = 0.80$):

- **Feature Grid Discretization ($5 \times 5$ spatial grid, cell centers at $0.5, 1.5, 2.5, 3.5, 4.5$):**
  - Continuous coordinates: row $y = 0.80 \times 5.0 = 4.00$, col $x = 0.12 \times 5.0 = 0.60$.
- **Attention Heatmap Peak Locations:**
  - `corrected`: Attention peak is located at **row 3, col 0** (corresponding to row index for $y = 20$ and col index for $x = 3$, peak value = 0.9137).
  - `trained`: Attention peak is located at **row 0, col 3** (transposed cell across the matrix diagonal, peak value = 0.9137).
  - Transpose Identity: $\max |m_{\text{trained}} - m_{\text{corrected}}^T| = \mathbf{0.0000 \times 10^0}$ (exact matrix transpose).
  - `off`: Modulation matrix is strictly all-ones ($M \equiv 1.0$), and the attended feature map satisfies $\tilde{F} \equiv F$ exactly.

---

## 3. Macro Aggregates across Dimensions and Topologies

Mean per-episode SMEI, area coverage (%), and swarm survival (%) across scenario groupings (20 scenarios total per dimension; 10 open maps and 10 obstacle maps per dimension; 100 seeds per scenario):

| Grouping | Model | Mode | Mean SMEI (%) | Mean Coverage (%) | Mean Survival (%) | N Scenarios |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2D Overall | RSPO Vanilla | `trained` | 34.46 | 47.04 | 30.34 | 20 |
| 2D Overall | RSPO Vanilla | `off` | 34.36 | 46.61 | 30.23 | 20 |
| 2D Overall | RSPO Vanilla | `corrected` | 34.84 | 47.04 | 30.66 | 20 |
| 2D Overall | RSPO SelfHeal | `trained` | 31.93 | 49.46 | 27.59 | 20 |
| 2D Overall | RSPO SelfHeal | `off` | 32.61 | 49.17 | 27.69 | 20 |
| 2D Overall | RSPO SelfHeal | `corrected` | 32.03 | 49.53 | 27.54 | 20 |
| 2D Open Maps | RSPO Vanilla | `trained` | 53.54 | 66.68 | 51.60 | 10 |
| 2D Open Maps | RSPO Vanilla | `off` | 52.54 | 65.83 | 50.75 | 10 |
| 2D Open Maps | RSPO Vanilla | `corrected` | 53.50 | 66.54 | 51.45 | 10 |
| 2D Open Maps | RSPO SelfHeal | `trained` | 52.91 | 66.10 | 51.35 | 10 |
| 2D Open Maps | RSPO SelfHeal | `off` | 52.36 | 65.46 | 50.10 | 10 |
| 2D Open Maps | RSPO SelfHeal | `corrected` | 52.79 | 66.17 | 50.98 | 10 |
| 2D Obstacle Maps | RSPO Vanilla | `trained` | 15.38 | 27.40 | 9.07 | 10 |
| 2D Obstacle Maps | RSPO Vanilla | `off` | 16.18 | 27.39 | 9.70 | 10 |
| 2D Obstacle Maps | RSPO Vanilla | `corrected` | 16.19 | 27.53 | 9.88 | 10 |
| 2D Obstacle Maps | RSPO SelfHeal | `trained` | 10.96 | 32.82 | 3.83 | 10 |
| 2D Obstacle Maps | RSPO SelfHeal | `off` | 12.86 | 32.88 | 5.28 | 10 |
| 2D Obstacle Maps | RSPO SelfHeal | `corrected` | 11.28 | 32.88 | 4.10 | 10 |
| 3D Overall | RSPO Vanilla | `trained` | 64.70 | 92.06 | 56.04 | 20 |
| 3D Overall | RSPO Vanilla | `off` | 62.27 | 91.81 | 55.33 | 20 |
| 3D Overall | RSPO Vanilla | `corrected` | 62.77 | 92.03 | 56.02 | 20 |
| 3D Overall | RSPO SelfHeal | `trained` | 62.20 | 90.51 | 52.73 | 20 |
| 3D Overall | RSPO SelfHeal | `off` | 59.56 | 90.45 | 52.39 | 20 |
| 3D Overall | RSPO SelfHeal | `corrected` | 59.42 | 90.54 | 52.67 | 20 |
| 3D Open Maps | RSPO Vanilla | `trained` | 66.64 | 92.40 | 55.62 | 10 |
| 3D Open Maps | RSPO Vanilla | `off` | 62.42 | 91.83 | 55.33 | 10 |
| 3D Open Maps | RSPO Vanilla | `corrected` | 62.92 | 92.27 | 55.75 | 10 |
| 3D Open Maps | RSPO SelfHeal | `trained` | 66.60 | 90.48 | 54.00 | 10 |
| 3D Open Maps | RSPO SelfHeal | `off` | 60.48 | 90.57 | 52.80 | 10 |
| 3D Open Maps | RSPO SelfHeal | `corrected` | 61.06 | 90.48 | 53.85 | 10 |
| 3D Obstacle Maps | RSPO Vanilla | `trained` | 62.76 | 91.73 | 56.45 | 10 |
| 3D Obstacle Maps | RSPO Vanilla | `off` | 62.12 | 91.79 | 55.33 | 10 |
| 3D Obstacle Maps | RSPO Vanilla | `corrected` | 62.62 | 91.78 | 56.30 | 10 |
| 3D Obstacle Maps | RSPO SelfHeal | `trained` | 57.81 | 90.55 | 51.45 | 10 |
| 3D Obstacle Maps | RSPO SelfHeal | `off` | 58.64 | 90.33 | 51.98 | 10 |
| 3D Obstacle Maps | RSPO SelfHeal | `corrected` | 57.78 | 90.59 | 51.50 | 10 |

---

## 4. Per-Scenario SMEI Comparison Side by Side (All 40 Scenarios)

### 4.1 RSPO Vanilla (40 Scenarios)

| Dimension | Environment | Fault Regime | Trained SMEI | Off SMEI | Corrected SMEI | Diff (Off - Trained) | Diff (Corr - Trained) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2D | 25×25 Standard | p = 0.0 | 77.16 | 76.98 | 77.08 | -0.18 | -0.08 |
| 2D | 25×25 Standard | p = 0.00025 | 70.55 | 70.81 | 70.46 | +0.26 | -0.09 |
| 2D | 25×25 Standard | p = 0.0005 | 64.07 | 63.98 | 64.23 | -0.09 | +0.16 |
| 2D | 25×25 Standard | p = 0.00075 | 59.47 | 59.30 | 59.83 | -0.17 | +0.36 |
| 2D | 25×25 Standard | p = 0.0010 | 52.65 | 51.90 | 52.65 | -0.75 | +0.00 |
| 2D | 25×25 Obstacles | p = 0.0 | 21.20 | 18.78 | 20.76 | -2.42 | -0.44 |
| 2D | 25×25 Obstacles | p = 0.00025 | 16.34 | 18.65 | 16.68 | +2.31 | +0.34 |
| 2D | 25×25 Obstacles | p = 0.0005 | 13.63 | 17.06 | 14.56 | +3.43 | +0.93 |
| 2D | 25×25 Obstacles | p = 0.00075 | 14.09 | 17.24 | 15.05 | +3.15 | +0.96 |
| 2D | 25×25 Obstacles | p = 0.0010 | 14.02 | 15.10 | 14.61 | +1.08 | +0.59 |
| 2D | 50×50 Standard | p = 0.0 | 65.06 | 63.82 | 65.16 | -1.24 | +0.10 |
| 2D | 50×50 Standard | p = 0.00025 | 50.83 | 47.54 | 50.62 | -3.29 | -0.21 |
| 2D | 50×50 Standard | p = 0.0005 | 40.60 | 37.78 | 40.44 | -2.82 | -0.16 |
| 2D | 50×50 Standard | p = 0.00075 | 30.76 | 30.75 | 30.61 | -0.01 | -0.15 |
| 2D | 50×50 Standard | p = 0.0010 | 24.26 | 22.50 | 23.89 | -1.76 | -0.37 |
| 2D | 50×50 Obstacles | p = 0.0 | 20.85 | 23.70 | 23.25 | +2.85 | +2.40 |
| 2D | 50×50 Obstacles | p = 0.00025 | 14.69 | 15.77 | 17.57 | +1.08 | +2.88 |
| 2D | 50×50 Obstacles | p = 0.0005 | 15.66 | 13.86 | 15.97 | -1.80 | +0.31 |
| 2D | 50×50 Obstacles | p = 0.00075 | 13.24 | 11.74 | 12.93 | -1.50 | -0.31 |
| 2D | 50×50 Obstacles | p = 0.0010 | 10.13 | 9.86 | 10.52 | -0.27 | +0.39 |
| 3D | 25×25 Standard | p = 0.0 | 82.22 | 82.14 | 82.09 | -0.08 | -0.13 |
| 3D | 25×25 Standard | p = 0.00025 | 77.19 | 76.20 | 76.43 | -0.99 | -0.76 |
| 3D | 25×25 Standard | p = 0.0005 | 71.97 | 70.35 | 70.38 | -1.62 | -1.59 |
| 3D | 25×25 Standard | p = 0.00075 | 68.07 | 63.29 | 64.07 | -4.78 | -4.00 |
| 3D | 25×25 Standard | p = 0.0010 | 63.26 | 57.89 | 57.70 | -5.37 | -5.56 |
| 3D | 25×25 Obstacles | p = 0.0 | 82.72 | 82.85 | 82.72 | +0.13 | +0.00 |
| 3D | 25×25 Obstacles | p = 0.00025 | 76.51 | 75.94 | 76.45 | -0.57 | -0.06 |
| 3D | 25×25 Obstacles | p = 0.0005 | 70.79 | 69.96 | 70.84 | -0.83 | +0.05 |
| 3D | 25×25 Obstacles | p = 0.00075 | 66.38 | 66.17 | 66.47 | -0.21 | +0.09 |
| 3D | 25×25 Obstacles | p = 0.0010 | 60.02 | 58.00 | 59.99 | -2.02 | -0.03 |
| 3D | 50×50 Standard | p = 0.0 | 77.09 | 76.37 | 76.77 | -0.72 | -0.32 |
| 3D | 50×50 Standard | p = 0.00025 | 67.69 | 65.52 | 66.38 | -2.17 | -1.31 |
| 3D | 50×50 Standard | p = 0.0005 | 59.81 | 54.06 | 55.07 | -5.75 | -4.74 |
| 3D | 50×50 Standard | p = 0.00075 | 53.35 | 44.25 | 45.22 | -9.10 | -8.13 |
| 3D | 50×50 Standard | p = 0.0010 | 45.76 | 34.11 | 35.07 | -11.65 | -10.69 |
| 3D | 50×50 Obstacles | p = 0.0 | 76.61 | 75.78 | 76.60 | -0.83 | -0.01 |
| 3D | 50×50 Obstacles | p = 0.00025 | 65.42 | 64.63 | 64.87 | -0.79 | -0.55 |
| 3D | 50×50 Obstacles | p = 0.0005 | 53.73 | 53.69 | 53.83 | -0.04 | +0.10 |
| 3D | 50×50 Obstacles | p = 0.00075 | 42.78 | 42.02 | 41.76 | -0.76 | -1.02 |
| 3D | 50×50 Obstacles | p = 0.0010 | 32.66 | 32.14 | 32.66 | -0.52 | +0.00 |

### 4.2 RSPO SelfHeal (40 Scenarios)

| Dimension | Environment | Fault Regime | Trained SMEI | Off SMEI | Corrected SMEI | Diff (Off - Trained) | Diff (Corr - Trained) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2D | 25×25 Standard | p = 0.0 | 79.77 | 80.34 | 79.77 | +0.57 | +0.00 |
| 2D | 25×25 Standard | p = 0.00025 | 69.25 | 70.24 | 69.95 | +0.99 | +0.70 |
| 2D | 25×25 Standard | p = 0.0005 | 62.39 | 61.93 | 62.44 | -0.46 | +0.05 |
| 2D | 25×25 Standard | p = 0.00075 | 55.68 | 55.53 | 55.25 | -0.15 | -0.43 |
| 2D | 25×25 Standard | p = 0.0010 | 50.10 | 49.55 | 49.80 | -0.55 | -0.30 |
| 2D | 25×25 Obstacles | p = 0.0 | 14.94 | 19.67 | 15.64 | +4.73 | +0.70 |
| 2D | 25×25 Obstacles | p = 0.00025 | 13.00 | 16.41 | 13.06 | +3.41 | +0.06 |
| 2D | 25×25 Obstacles | p = 0.0005 | 12.70 | 14.97 | 12.46 | +2.27 | -0.24 |
| 2D | 25×25 Obstacles | p = 0.00075 | 12.25 | 13.60 | 11.91 | +1.35 | -0.34 |
| 2D | 25×25 Obstacles | p = 0.0010 | 11.19 | 12.16 | 11.18 | +0.97 | -0.01 |
| 2D | 50×50 Standard | p = 0.0 | 67.43 | 67.59 | 67.42 | +0.16 | -0.01 |
| 2D | 50×50 Standard | p = 0.00025 | 51.21 | 50.35 | 50.60 | -0.86 | -0.61 |
| 2D | 50×50 Standard | p = 0.0005 | 38.65 | 37.75 | 37.89 | -0.90 | -0.76 |
| 2D | 50×50 Standard | p = 0.00075 | 31.03 | 29.14 | 30.79 | -1.89 | -0.24 |
| 2D | 50×50 Standard | p = 0.0010 | 23.58 | 21.14 | 23.94 | -2.44 | +0.36 |
| 2D | 50×50 Obstacles | p = 0.0 | 12.73 | 13.35 | 13.70 | +0.62 | +0.97 |
| 2D | 50×50 Obstacles | p = 0.00025 | 10.84 | 12.63 | 12.51 | +1.79 | +1.67 |
| 2D | 50×50 Obstacles | p = 0.0005 | 8.80 | 10.00 | 8.88 | +1.20 | +0.08 |
| 2D | 50×50 Obstacles | p = 0.00075 | 6.94 | 8.40 | 7.29 | +1.46 | +0.35 |
| 2D | 50×50 Obstacles | p = 0.0010 | 6.19 | 7.37 | 6.19 | +1.18 | +0.00 |
| 3D | 25×25 Standard | p = 0.0 | 85.11 | 85.09 | 85.09 | -0.02 | -0.02 |
| 3D | 25×25 Standard | p = 0.00025 | 78.74 | 75.65 | 76.78 | -3.09 | -1.96 |
| 3D | 25×25 Standard | p = 0.0005 | 72.75 | 66.81 | 68.77 | -5.94 | -3.98 |
| 3D | 25×25 Standard | p = 0.00075 | 67.97 | 59.18 | 61.60 | -8.79 | -6.37 |
| 3D | 25×25 Standard | p = 0.0010 | 62.39 | 52.83 | 54.30 | -9.56 | -8.09 |
| 3D | 25×25 Obstacles | p = 0.0 | 83.46 | 83.55 | 83.46 | +0.09 | +0.00 |
| 3D | 25×25 Obstacles | p = 0.00025 | 73.54 | 73.76 | 73.49 | +0.22 | -0.05 |
| 3D | 25×25 Obstacles | p = 0.0005 | 66.61 | 67.99 | 66.57 | +1.38 | -0.04 |
| 3D | 25×25 Obstacles | p = 0.00075 | 61.17 | 61.83 | 61.04 | +0.66 | -0.13 |
| 3D | 25×25 Obstacles | p = 0.0010 | 54.21 | 55.97 | 54.19 | +1.76 | -0.02 |
| 3D | 50×50 Standard | p = 0.0 | 80.05 | 80.09 | 79.68 | +0.04 | -0.37 |
| 3D | 50×50 Standard | p = 0.00025 | 66.86 | 62.63 | 62.92 | -4.23 | -3.94 |
| 3D | 50×50 Standard | p = 0.0005 | 57.49 | 49.90 | 49.85 | -7.59 | -7.64 |
| 3D | 50×50 Standard | p = 0.00075 | 51.52 | 42.51 | 41.44 | -9.01 | -10.08 |
| 3D | 50×50 Standard | p = 0.0010 | 43.11 | 30.10 | 30.12 | -13.01 | -12.99 |
| 3D | 50×50 Obstacles | p = 0.0 | 77.09 | 77.03 | 77.09 | -0.06 | +0.00 |
| 3D | 50×50 Obstacles | p = 0.00025 | 57.49 | 59.46 | 57.02 | +1.97 | -0.47 |
| 3D | 50×50 Obstacles | p = 0.0005 | 40.80 | 42.23 | 41.95 | +1.43 | +1.15 |
| 3D | 50×50 Obstacles | p = 0.00075 | 34.85 | 35.68 | 34.55 | +0.83 | -0.30 |
| 3D | 50×50 Obstacles | p = 0.0010 | 28.88 | 28.87 | 28.46 | -0.01 | -0.42 |

---

## 5. CSSA Attention Activity Share

Percentage of evaluated steps where at least one crash was actively registered in fleet health ($crash\_active > 0$), per scenario across fault injection rates ($p \in \{0, 0.00025, 0.0005, 0.00075, 0.001\}$):

| Dimension | Environment | Fault Regime | $p$ Value | RSPO Vanilla Active (%) | RSPO SelfHeal Active (%) | Mean Active Steps (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2D | 25×25 Obstacles | p = 0.0 | 0.00000 | 20.07% | 19.81% | 19.94% |
| 2D | 25×25 Obstacles | p = 0.00025 | 0.00025 | 19.71% | 19.31% | 19.51% |
| 2D | 25×25 Obstacles | p = 0.0005 | 0.00050 | 19.93% | 18.64% | 19.29% |
| 2D | 25×25 Obstacles | p = 0.00075 | 0.00075 | 20.55% | 18.80% | 19.68% |
| 2D | 25×25 Obstacles | p = 0.0010 | 0.00100 | 20.19% | 18.30% | 19.24% |
| 2D | 25×25 Standard | p = 0.0 | 0.00000 | 6.04% | 0.91% | 3.48% |
| 2D | 25×25 Standard | p = 0.00025 | 0.00025 | 27.13% | 22.50% | 24.82% |
| 2D | 25×25 Standard | p = 0.0005 | 0.00050 | 37.50% | 32.71% | 35.10% |
| 2D | 25×25 Standard | p = 0.00075 | 0.00075 | 40.20% | 35.71% | 37.96% |
| 2D | 25×25 Standard | p = 0.0010 | 0.00100 | 39.31% | 35.44% | 37.38% |
| 2D | 50×50 Obstacles | p = 0.0 | 0.00000 | 25.92% | 19.62% | 22.77% |
| 2D | 50×50 Obstacles | p = 0.00025 | 0.00025 | 23.88% | 19.39% | 21.63% |
| 2D | 50×50 Obstacles | p = 0.0005 | 0.00050 | 22.08% | 18.56% | 20.32% |
| 2D | 50×50 Obstacles | p = 0.00075 | 0.00075 | 20.65% | 17.96% | 19.30% |
| 2D | 50×50 Obstacles | p = 0.0010 | 0.00100 | 19.22% | 16.51% | 17.86% |
| 2D | 50×50 Standard | p = 0.0 | 0.00000 | 11.98% | 7.84% | 9.91% |
| 2D | 50×50 Standard | p = 0.00025 | 0.00025 | 32.16% | 32.84% | 32.50% |
| 2D | 50×50 Standard | p = 0.0005 | 0.00050 | 34.52% | 37.88% | 36.20% |
| 2D | 50×50 Standard | p = 0.00075 | 0.00075 | 32.52% | 35.08% | 33.80% |
| 2D | 50×50 Standard | p = 0.0010 | 0.00100 | 28.23% | 30.41% | 29.32% |
| 3D | 25×25 Obstacles | p = 0.0 | 0.00000 | 2.96% | 1.33% | 2.14% |
| 3D | 25×25 Obstacles | p = 0.00025 | 0.00025 | 25.96% | 21.03% | 23.50% |
| 3D | 25×25 Obstacles | p = 0.0005 | 0.00050 | 37.05% | 30.45% | 33.75% |
| 3D | 25×25 Obstacles | p = 0.00075 | 0.00075 | 40.31% | 31.44% | 35.88% |
| 3D | 25×25 Obstacles | p = 0.0010 | 0.00100 | 39.69% | 30.98% | 35.34% |
| 3D | 25×25 Standard | p = 0.0 | 0.00000 | 5.54% | 0.01% | 2.77% |
| 3D | 25×25 Standard | p = 0.00025 | 0.00025 | 25.73% | 22.11% | 23.92% |
| 3D | 25×25 Standard | p = 0.0005 | 0.00050 | 37.44% | 31.27% | 34.35% |
| 3D | 25×25 Standard | p = 0.00075 | 0.00075 | 39.89% | 33.63% | 36.76% |
| 3D | 25×25 Standard | p = 0.0010 | 0.00100 | 39.47% | 32.17% | 35.82% |
| 3D | 50×50 Obstacles | p = 0.0 | 0.00000 | 18.16% | 8.03% | 13.09% |
| 3D | 50×50 Obstacles | p = 0.00025 | 0.00025 | 37.45% | 27.06% | 32.26% |
| 3D | 50×50 Obstacles | p = 0.0005 | 0.00050 | 40.96% | 29.15% | 35.05% |
| 3D | 50×50 Obstacles | p = 0.00075 | 0.00075 | 39.42% | 28.52% | 33.97% |
| 3D | 50×50 Obstacles | p = 0.0010 | 0.00100 | 33.95% | 24.62% | 29.29% |
| 3D | 50×50 Standard | p = 0.0 | 0.00000 | 15.11% | 6.20% | 10.65% |
| 3D | 50×50 Standard | p = 0.00025 | 0.00025 | 37.95% | 31.38% | 34.66% |
| 3D | 50×50 Standard | p = 0.0005 | 0.00050 | 41.76% | 34.46% | 38.11% |
| 3D | 50×50 Standard | p = 0.00075 | 0.00075 | 39.50% | 34.21% | 36.86% |
| 3D | 50×50 Standard | p = 0.0010 | 0.00100 | 33.53% | 28.59% | 31.06% |

---

## 6. Directory Structure & Deliverable Manifest

All output directories and CSV files preserve the exact schema and layout of `Results/2D-final-v3` and `Results/3D-final-v3`:

```
Results/CSSA-variants/
├── CSSA_VARIANTS_REPORT.md             <- This comprehensive diagnostic report
├── off/
│   ├── run_manifest.json               <- Top-level manifest for off mode
│   ├── 2D-final/
│   │   ├── run_manifest.json
│   │   ├── evaluation_log.txt
│   │   └── Coverage/<Env>/<Fault>/     <- 20 scenario folders (raw CSV, summary CSV, dashboard PNG)
│   └── 3D-final/
│       ├── run_manifest.json
│       ├── evaluation_log.txt
│       └── Coverage/<Env>/<Fault>/     <- 20 scenario folders (raw CSV, summary CSV, dashboard PNG)
└── corrected/
    ├── run_manifest.json               <- Top-level manifest for corrected mode
    ├── 2D-final/
    │   ├── run_manifest.json
    │   ├── evaluation_log.txt
    │   └── Coverage/<Env>/<Fault>/     <- 20 scenario folders (raw CSV, summary CSV, dashboard PNG)
    └── 3D-final/
        ├── run_manifest.json
        ├── evaluation_log.txt
        └── Coverage/<Env>/<Fault>/     <- 20 scenario folders (raw CSV, summary CSV, dashboard PNG)
```

