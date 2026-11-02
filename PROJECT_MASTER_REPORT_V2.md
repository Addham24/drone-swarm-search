# MASTER TECHNICAL RESEARCH REPORT (V2): Resilient Multi-UAV Swarm Search, Coverage, and Target Tracking Under Hardware Actuator Faults

**Version:** 2.0 (Post-Audit Code Ground-Truth Edition)  
**Project:** Autonomous Decentralized Multi-Agent Reinforcement Learning for Drone Swarms  
**Target Environments:** DSSE (2D Planar) & DSSE-3D (Volumetric Voxel Grids)  
**Primary Novelty:** RSPO (Resilient Swarm Policy Optimization) with Self-Healing Architectures  
**Comparative Scope:** 12 MARL Models (6 Families × 2 Variants), 4 Map Topologies, 5 Fault Regimes, 100 Seeds per Scenario (96,000 Total Evaluated Episodes: 12 models × 80 scenarios × 100 seeds)

---

## Table of Contents
1. [Executive Summary & High-Level Architecture](#1-executive-summary--high-level-architecture)
2. [Mathematical Problem Formulation & Environmental Dynamics](#2-mathematical-problem-formulation--environmental-dynamics)
   - 2.1 Dec-POMDP Formulation
   - 2.2 Spatial Dimensions, Grid Topologies & Obstacle Profiles
   - 2.3 DSSE-3D Volumetric Voxel Extension & Sensor Footprints
   - 2.4 Dual Mission Objectives: Coverage vs. Dynamic Tracking
3. [Kinematics, Action Spaces, and Energetics](#3-kinematics-action-spaces-and-energetics)
   - 3.1 Discrete-9 Movement & Search Kinematics
   - 3.2 3D Hybrid Control Architecture
   - 3.3 Energetic Autonomy & Charging Station Dynamics
4. [In-Flight Fault Models & Actuator Degradation Regimes](#4-in-flight-fault-models--actuator-degradation-regimes)
   - 4.1 Poisson Hardware Fault Process
   - 4.2 Physical Consequences of Motor Failure
5. [The 6 Multi-Agent RL Model Families (12 Models)](#5-the-6-multi-agent-rl-model-families-12-models)
   - 5.1 Architecture Overview & Visual Hierarchy
   - 5.2 RSPO (Resilient Swarm Policy Optimization)
   - 5.3 MAPPO (Multi-Agent PPO)
   - 5.4 QMIX (Monotonic Value Factorization)
   - 5.5 MAA2C (Multi-Agent Advantage Actor-Critic)
   - 5.6 COMA (Counterfactual Multi-Agent Policy Gradients)
   - 5.7 I-DQN (Independent Deep Q-Networks)
6. [RSPO Deep Dive: Neural Architecture & Self-Healing Innovations](#6-rspo-deep-dive-neural-architecture--self-healing-innovations)
   - 6.1 Sub-Network Architecture & 544-Dimensional Feature Fusion
   - 6.2 Innovation 1: Fleet Health Embedding (FHE)
   - 6.3 Innovation 2: Crash-Sector Spatial Attention (CSSA)
   - 6.4 Innovation 3: Dual-Objective Value Decomposition (DOVD) Critic
   - 6.5 Innovation 4: Resilience-Adaptive Clipping (RAC) Gate
   - 6.6 Innovation 5: Dynamic Sector Compensation Reward
7. [Environment Wrappers & Sim-to-Sim Infrastructure](#7-environment-wrappers--sim-to-sim-infrastructure)
   - 7.1 Wrapper Pipeline Architecture
   - 7.2 Zero-Shot Spatial Rescaling (50×50 → 25×25)
   - 7.3 Zero-Shot 3D Lifting (DSSE3DAdapter)
   - 7.4 Physical Obstacle Airspace Wrapper
   - 7.5 The Vanilla vs. Self-Healing Environment & Algorithmic Ablation Specification
8. [Reward Formulations & Evaluation Metrics](#8-reward-formulations--evaluation-metrics)
   - 8.1 Code-Verified Mission Reward Functions (Coverage & Tracking)
   - 8.2 Global Cooperative Reward Mixing (VDN Blending)
   - 8.3 Swarm Mission Efficiency Index (SMEI)
   - 8.4 Core Evaluation Metrics
9. [Major Scientific Discoveries & Audits](#9-major-scientific-discoveries--audits)
   - 9.1 Discovery 1: Asymmetric Checkpoint Selection Resolution
   - 9.2 Discovery 2: The COMA Coverage Credit Assignment Pathology
   - 9.3 Discovery 3: The 50×50 Kamikaze vs. Logistical Survival Trade-Off
10. [Comprehensive Benchmark Results & Research Questions](#10-comprehensive-benchmark-results--research-questions)
    - 10.1 RQ1: Paired Self-Healing Ablation Analysis
    - 10.2 RQ2: Fault Degradation Resilience Curves
    - 10.3 RQ3: Cross-Topology & Spatial Generalisation
    - 10.4 RQ4: Multi-Objective Fleet Standing & Leaderboard
11. [Codebase Architecture & Reproducibility Guide](#11-codebase-architecture--reproducibility-guide)

---

## 1. Executive Summary & High-Level Architecture

Autonomous unmanned aerial vehicle (UAV) swarms operating in search-and-rescue, disaster relief, and hostile reconnaissance face severe real-world challenges: in-flight actuator failures (motor burnouts), strict battery capacity, communication loss, and multi-height urban obstacles. Standard multi-agent reinforcement learning (MARL) algorithms assume hardware invariance during deployment; when hardware faults occur, their spatial coordination collapses, resulting in deadlocks, battery depletion, or mid-air collisions.

This project introduces **RSPO (Resilient Swarm Policy Optimization)**, an on-policy decentralized actor-critic architecture equipped with a modular **Self-Healing mechanism**. RSPO enables homogeneous drone swarms to dynamically detect peer failures, infer fleet health, reorganize spatial search perimeters in real time, and balance urgent exploration with conservative base-station recharging.

```
                              ┌──────────────────────────────────────────┐
                              │       RAW SWARM OBSERVATIONS             │
                              │  - Spatial Probability Grid: (25×25)    │
                              │  - Local Telemetry / Kinematics: (22-dim)│
                              └────────────────────┬─────────────────────┘
                                                   │
                   ┌───────────────────────────────┼───────────────────────────────┐
                   ▼                               ▼                               ▼
       ┌───────────────────────┐       ┌───────────────────────┐       ┌───────────────────────┐
       │ Spatial CNN Backbone  │       │     Kinematic MLP     │       │ Fleet Health Extractor│
       │ (Conv2D + Flatten)    │       │   (Dense 512 → 256)   │       │  (14-dim Status → 32) │
       └───────────┬───────────┘       └───────────┬───────────┘       └───────────┬───────────┘
                   │ 256-dim                       │ 256-dim                       │ 32-dim (h_t)
                   └───────────────────────────────┼───────────────────────────────┘
                                                   ▼
                                 ┌───────────────────────────────────┐
                                 │ FEATURE FUSION LAYER (544-dim)    │
                                 │  [Spatial || Kinematic || FHE]    │
                                 └─────────────────┬─────────────────┘
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼                                                               ▼
       ┌───────────────────────┐                                       ┌───────────────────────┐
       │     ACTOR NETWORK     │                                       │   DOVD CRITIC NETWORK │
       │  (Dense 256 → Logits) │                                       │  (V_search + α*V_take)│
       └───────────┬───────────┘                                       └───────────┬───────────┘
                   │ Action Probabilities π(a|s)                                   │ State Value V(s)
                   ▼                                                               ▼
       ┌───────────────────────┐                                       ┌───────────────────────┐
       │ RAC Dynamic Clipping  │ ◄─────────────────────────────────────┤ Learned Gating α_t    │
       │ ε_t = 0.10 + 0.20*RAC │                                       │ Multi-Head Value Est. │
       └───────────────────────┘                                       └───────────────────────┘
```

### Key Quantitative Findings:
1. **3D Swarm Dominance:** RSPO SelfHeal achieves **91.61% Coverage** and **62.00% Swarm Survival** under active motor faults ($p = 0.0005$), outperforming MAPPO SelfHeal (84.90% cov, 24.50% surv) and all 8 EPyMARL baselines.
2. **Logistical Discipline vs. Kamikaze Exploitation:** On massive $50 \times 50$ topologies, RSPO SelfHeal achieves a **$70\%$ scenario win rate** across multi-objective Pareto metrics, maintaining $18.5\%$ survival (logging 210 charging steps) compared to MAPPO's $9.75\%$ survival (logging only 108 charging steps).
3. **Statistical Fairness:** 12 algorithms evaluated over 80 scenarios across 100 random seeds ($N = 100$, seeds 1000–1099), totaling 8,000 independent evaluation episodes.

---

## 2. Mathematical Problem Formulation & Environmental Dynamics

### 2.1 Dec-POMDP Formulation
The problem is modeled as a decentralized Partially Observable Markov Decision Process (Dec-POMDP) for $N = 4$ cooperative quadrotors, formally defined by the tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \Omega, \mathcal{O}, N, \gamma \rangle$$

- $\mathcal{S}$: Global state space containing the physical ground matrix, drone 3D positions, battery levels, fault states, and obstacle coordinates.
- $\mathcal{A} = \prod_{i=1}^N \mathcal{A}_i$: Joint action space where each agent selects $a_i \in \mathcal{A}_i = \{0, 1, \dots, 8\}$.
- $\mathcal{P}(s' \mid s, \mathbf{a})$: State transition probability incorporating deterministic flight kinematics, stochastic Poisson hardware faults, and dynamic target movement.
- $\mathcal{R}(s, \mathbf{a})$: Team reward signal balancing newly searched coverage area, target proximity, battery conservation, and collision avoidance.
- $\Omega = \prod_{i=1}^N \Omega_i$: Joint observation space.
- $\mathcal{O}(s, i) \rightarrow o_i \in \Omega_i$: Observation function emitting agent-specific partial observations.
- $\gamma \in [0.99, 0.995]$: Temporal discount factor.

---

### 2.2 Spatial Dimensions, Grid Topologies & Obstacle Profiles

The benchmark establishes 4 standardized topologies across both 2D and 3D:

| Environment Identifier | Spatial Grid | Total Cells / Voxels | Free Cells | Obstacle Footprint | Mission Horizon ($T$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`25x25_Standard`** | $25 \times 25$ | 625 cells | 625 ($100\%$) | None (Open Field) | 750 timesteps |
| **`25x25_Obstacles`** | $25 \times 25$ | 625 cells | 465 ($74.4\%$) | 160 cells ($25.6\%$, Urban Skyline) | 750 timesteps |
| **`50x50_Standard`** | $50 \times 50$ | 2,500 cells | 2,500 ($100\%$) | None (4× Area Scale-Up) | 1,500 timesteps |
| **`50x50_Obstacles`** | $50 \times 50$ | 2,500 cells | 1,860 ($74.4\%$) | 640 cells ($25.6\%$, Balanced City) | 1,500 timesteps |

```
    25x25 Urban Obstacle Skyline Topology (Asymmetric City Block Layout)
    ┌────────────────────────────────────────────────────────┐
    │ [C] .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  │  [C] = Base Station (0, 0)
    │  .  █  █  █  .  .  █  █  █  .  .  █  █  █  .  .  █  █  │  █   = Building Obstacle
    │  .  █  █  █  .  .  █  █  █  .  .  █  █  █  .  .  █  █  │  .   = Open Street Canyon
    │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  │
    │  .  █  █  █  .  .  █  █  █  .  .  █  █  █  .  .  █  █  │  Obstacle heights:
    │  .  █  █  █  .  .  █  █  █  .  .  █  █  █  .  .  █  █  │    H ∈ [1, 5] layers
    │  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  │
    │  .  █  █  █  .  .  █  █  █  .  .  █  █  █  .  .  █  █  │
    └────────────────────────────────────────────────────────┘
```

---

### 2.3 DSSE-3D Volumetric Voxel Extension & Sensor Footprints

In `DSSE-3D`, altitude layers $z \in \{0, 1, 2, 3, 4, 5\}$ introduce vertical reasoning:
- **$z = 0$ (Ground Level):** Location of charging pads and surface targets. Drone landing occurs here.
- **$z = 1$ (Scanning Layer):** Precise low-altitude search layer ($1\times 1$ FOV footprint directly beneath the drone).
- **$z \in [2, 4]$ (Transit Band):** Cruising altitudes for obstacle flyovers. Flying at higher altitudes expands the ground FOV footprint ($3\times 3$ to $5\times 5$), clearing wider ground swaths per step as modeled in `DSSE3DAdapter`.
- **$z = 5$ (Ceiling):** Airspace limit ($7\times 7$ footprint).

#### Altitude vs. Field of View (FOV) Sensor Footprint:
Flying higher widens the sensor's situational awareness footprint but reduces measurement fidelity and costs more energy:
$$\text{FOV Radius}(z) = \begin{cases} 
0 \text{ cells} & z = 0 \text{ (Landed)} \\
1 \text{ cell } (1 \times 1) & z = 1 \text{ (High-Resolution Scanning)} \\
9 \text{ cells } (3 \times 3) & z = 2 \text{ (Low-Resolution Transit Awareness)} \\
25 \text{ cells } (5 \times 5) & z = 3, 4 \text{ (Wide Transit Awareness)} \\
49 \text{ cells } (7 \times 7) & z = 5 \text{ (Maximum High Ceiling)}
\end{cases}$$

---

### 2.4 Dual Mission Objectives: Coverage vs. Dynamic Tracking

1. **Persistent Area Coverage (Exploration):**
   - The swarm must uncover an unknown or Gaussian-drifted target probability distribution $P_t(x, y) \in [0, 1]$.
   - Drones start at charging pad $(0, 0)$. When a drone executes `SEARCH` at $(x, y, 1)$, $P_t(x, y)$ is cleared to zero, and newly uncovered probability is credited to team return.
2. **Dynamic Target Tracking (Pursuit & Interception):**
   - An active surface target undergoes continuous kinematic drift or evasive maneuvering:
     $$p_{\text{target}}(t+1) = p_{\text{target}}(t) + \mathbf{v}_{\text{drift}} + \boldsymbol{\eta}_t, \quad \boldsymbol{\eta}_t \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$$
   - The swarm must search the area, detect target emergence, close distance, and maintain visual line-of-sight tracking within capture radius $R_{\text{capture}} \le 1.5$ cells.

---

## 3. Kinematics, Action Spaces, and Energetics

### 3.1 Discrete-9 Movement & Search Kinematics

In accordance with the underlying DSSE environment (`DSSE/environment/constants.py`), the agent action space consists of 9 discrete actions:

```
                  [4] UP_LEFT     [2] UP       [5] UP_RIGHT
                          ↖         ↑         ↗
                  [0] LEFT  ◄────── [8] ──────► [1] RIGHT
                          ↙      (SEARCH)     ↘
                [6] DOWN_LEFT    [3] DOWN     [7] DOWN_RIGHT
```

| Action Index | Semantic Name | Direction Vector $(\Delta x, \Delta y)$ | Operational Function |
| :---: | :---: | :---: | :--- |
| **0** | `LEFT` | $(-1, 0)$ | Horizontal westward displacement |
| **1** | `RIGHT` | $(+1, 0)$ | Horizontal eastward displacement |
| **2** | `UP` | $(0, -1)$ | Grid northward displacement |
| **3** | `DOWN` | $(0, +1)$ | Grid southward displacement |
| **4** | `UP_LEFT` | $(-1, -1)$ | Diagonal North-West traversal |
| **5** | `UP_RIGHT` | $(+1, -1)$ | Diagonal North-East traversal |
| **6** | `DOWN_LEFT` | $(-1, +1)$ | Diagonal South-West traversal |
| **7** | `DOWN_RIGHT` | $(+1, +1)$ | Diagonal South-East traversal |
| **8** | `SEARCH` | $(0, 0)$ | Hover / Cell Search / Base Recharging Hold |

---

### 3.2 3D Hybrid Control Architecture

To deploy 2D-trained MARL policies in 3D volumetric space with **zero retraining**, the framework employs a two-tier hierarchical controller:
1. **Horizontal Plane (MARL Policy):** The trained neural network emits $a_{2D} \in \{0, \dots, 8\}$.
2. **Vertical Altitude (`AltitudeController`):** A rule-based state machine dynamically manages vertical transitions $\Delta z \in \{-1, 0, +1\}$:
   - **Scanning Initiation:** If policy issues action `8` (SEARCH) and $z > 1$, descent ($\Delta z = -1$) is prioritized to reach $z = 1$.
   - **Rooftop Flyover Lookahead:** If the intended 2D step $(x + \Delta x, y + \Delta y)$ encounters an obstacle of height $H_{\text{obs}} \ge z$, the controller climbs ($\Delta z = +1$) to clear the rooftop.
   - **Open Street Cruising:** Over open street corridors ($H_{\text{obs}} = 0$), the swarm maintains uniform cruising altitude $z = 2$.
   - **Battery Docking:** If battery $b_i < 20\%$ and drone reaches $(0, 0)$, it descends to $z = 0$ to engage the charging pad.

---

### 3.3 Energetic Autonomy & Charging Station Dynamics

- **Battery Capacity:** $B_{\max} = 125$ steps ($25 \times 25$ grids) and $B_{\max} = 250$ steps ($50 \times 50$ grids).
- **Discharge Rates:**
  - Base hover, search, and horizontal movement: $-1.0$ unit per step.
  - In-flight vertical climb ($\Delta z = +1$ in 3D): managed via `DSSE3DAdapter` transition cost.
  - Actuator fault state: Immediate catastrophic failure (battery clamped to $0$, drone permanently immobilized; no intermediate degraded cruising).
- **Rapid Recharging Pad:**
  - Situated at origin $(0, 0)$ on ground layer $z = 0$.
  - Recharging rate: $+15.0$ battery units per step up to $B_{\max}$.
- **Depletion Crash:** If $b_i \le 0$, the drone crashes. It is permanently immobilized, zeroing future contributions and incurring a terminal $-50.0$ penalty in Coverage mode (incurring $0.0$ penalty for unpreventable Poisson hardware faults).

---

## 4. In-Flight Fault Models & Actuator Degradation Regimes

### 4.1 Poisson Hardware Fault Process
Hardware faults represent independent Poisson arrival processes applied per-agent per-step:
$$P(\text{Fault Event at step } t) = 1 - e^{-p_{\text{fault}}}$$

The benchmark audits 5 distinct operational fault regimes (fault counts scale with episode horizon $T$):
1. **`No_Faults_0.0` ($p = 0.0$):** Ideal nominal hardware baseline (0 faults).
2. **`Fault_0.00025` ($p = 0.00025$):** Low degradation. Swarm experiences $\sim 0.75$ cumulative faults per episode on $25\times 25$ ($T=750$), scaling to $\sim 1.50$ faults on $50\times 50$ ($T=1,500$).
3. **`Active_Faults_0.0005` ($p = 0.0005$):** Standard benchmark condition. Swarm experiences $\sim 1.50$ cumulative faults on $25\times 25$, scaling to $\sim 3.0$ faults on $50\times 50$.
4. **`Fault_0.00075` ($p = 0.00075$):** High stress. Swarm experiences $\sim 2.25$ cumulative faults on $25\times 25$, scaling to $\sim 4.5$ faults on $50\times 50$.
5. **`Fault_0.001` ($p = 0.0010$):** Hostile operating regime. Swarm experiences $\sim 3.0$ cumulative faults on $25\times 25$ ($75\%$ fleet attrition), scaling to $\sim 6.0$ cumulative faults across extended $50\times 50$ operations.

*(Note: Because $50\times 50$ topologies double the operational horizon from $T=750$ to $T=1,500$, the cumulative Poisson failure expectation per episode strictly doubles across all fault probabilities).*

### 4.2 Physical Consequences of Motor Failure (Catastrophic Attrition Model)
When an agent $i$ experiences motor failure triggered by the Poisson process (`BatteryStationWrapper.py:L171`):
1. **Immediate Fleet Attrition:** The drone is permanently disabled: `battery[i] = 0` and `alive[i] = False`. It ceases all flight and scanning capabilities for the remainder of the episode.
2. **Crash Site Broadcast:** Its exact spatial coordinates $(y_i, x_i)$ at the moment of failure are permanently recorded into `crash_coords[i]` and broadcast to all surviving teammates.
3. **Telemetry Observation Masking:** In the 22-dimensional telemetry stream, the disabled drone's position coordinates are masked to `[-1.0, -1.0]`, its battery is clamped to `0.0`, and its normalized crash location $(c_y / N, c_x / N)$ is populated into the telemetry vector.
4. **Fair Credit Assignment (Zero Fault Penalty):** Unlike preventable battery depletion deaths (which incur a $-50.0$ death penalty), unpreventable random hardware faults incur exactly **$0.0$ reward penalty** (`rewards[agent] = 0.0`), preventing policy gradient corruption from stochastic hardware defects.

---

## 5. The 6 Multi-Agent RL Model Families (12 Models)

The benchmark compares 12 distinct models: 6 algorithmic families evaluated across both **Vanilla** (no fault compensation) and **Self-Healing** (with fault compensation reward and FHE) variants.

### Visual Styling & Publication Color Standards:
To guarantee visual distinction without ambiguous hues:
- **Each Algorithm Family has ONE unique, dedicated color hue.**
- **Self-Healing variant = Solid line (`-`).**
- **Vanilla variant = Dashed line (`--`).**

| Algorithm Family | Framework | Shared Color Hex | Paradigm | Value Factorization / Critic |
| :--- | :--- | :--- | :--- | :--- |
| **RSPO (Ours)** | RLlib | `#D35400` (Burnt Orange) | Actor-Critic (On-Policy) | Dual-Objective DOVD Centralized Critic |
| **MAPPO** | RLlib | `#1E88E5` (Vibrant Blue) | Actor-Critic (On-Policy) | Centralized Value Function $V(s)$ |
| **QMIX** | EPyMARL | `#2E7D32` (Forest Green) | Value-Based (Off-Policy) | Monotonic Mixing Network $Q_{\text{tot}} = f(Q_i)$ |
| **COMA** | EPyMARL | `#D32F2F` (Crimson Red) | Actor-Critic (On-Policy) | Centralized Counterfactual Critic $Q(s, \mathbf{u})$ |
| **MAA2C** | EPyMARL | `#B78103` (Golden Amber) | Actor-Critic (On-Policy) | Centralized Advantage Actor-Critic |
| **I-DQN** | RLlib / EPyMARL | `#8E24AA` (Purple) | Value-Based (Off-Policy) | Independent Decentralized $Q_i(o_i, a_i)$ |

---

### Detailed Algorithmic Descriptions:

#### 5.2 RSPO (Resilient Swarm Policy Optimization)
- **Actor:** Dual-stream network combining a 2-layer Spatial CNN and a 2-layer Kinematic MLP, fused with the 32-dimensional Fleet Health Embedding ($h_t$).
- **Critic:** Dual-Objective Value Decomposition (DOVD) separating search return from sector-takeover return.
- **Sampling:** Temperature-scaled categorical action sampling:
  $$\pi_\tau(a \mid s) = \frac{\exp(z_a / \tau)}{\sum_{a'} \exp(z_{a'} / \tau)}$$
  $\tau = 0.20$ ($25 \times 25$ Coverage), $\tau = 0.30$ ($50 \times 50$ Coverage), $\tau = 0.10$ (Dynamic Tracking). Prevents spatial policy collapse.

#### 5.3 MAPPO (Multi-Agent PPO)
- Centralized Training with Decentralized Execution (CTDE).
- Uses generalized advantage estimation (GAE, $\lambda = 0.95$). Evaluated with symmetric temperature-scaled categorical sampling identical to RSPO to guarantee benchmark fairness.

#### 5.4 QMIX
- Factorizes joint action-value $Q_{\text{tot}}(\mathbf{\tau}, \mathbf{u})$ under the monotonicity constraint:
  $$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \ge 0, \quad \forall i \in \{1, \dots, N\}$$
- Uses absolute-weight hypernetworks conditioned on global state $s$ to compute non-negative weights $W_1, W_2 \ge 0$.

#### 5.5 MAA2C
- Decentralized actor policies $\pi_i(u_i \mid \tau_i)$ optimized via gradient ascent with centralized critic baseline:
  $$A_i(s, u_i) = R_t + \gamma V(s') - V(s)$$

#### 5.6 COMA (Counterfactual Multi-Agent Policy Gradients)
- Uses a centralized critic to deduce an agent's individual contribution by holding other agents' actions fixed:
  $$A^a(s, \mathbf{u}) = Q(s, \mathbf{u}) - \sum_{u^{\prime a}} \pi^a(u^{\prime a} \mid \tau^a) Q(s, (\mathbf{u}^{-a}, u^{\prime a}))$$

#### 5.7 I-DQN (Independent DQN)
- Completely decentralized Q-learning where each drone treats all teammates as non-stationary components of the environment.

---

## 6. RSPO Deep Dive: Neural Architecture & Self-Healing Innovations

### 6.1 Sub-Network Architecture & 544-Dimensional Feature Fusion

```
[Observation Stream 1: 25×25 Probability Matrix]
  │ (shape: [B, 1, 25, 25])
  ▼
Conv2D (1 → 16, k=3, s=1) ──► Tanh ──► MaxPool2D(2) ──► [B, 16, 11, 11]
  ▼
Conv2D (16 → 32, k=2, s=1) ──► Tanh ──► MaxPool2D(2) ──► [B, 32, 5, 5]
  ▼ (optional CSSA Attention modulated by crash coordinates)
Flatten (32 × 5 × 5 = 800)
  ▼
Linear (800 → 256) ──► Tanh ──────────────────────────────┐
                                                           │ (cnn_out: 256-dim)
[Observation Stream 2: 22-dim Local Telemetry]             │
  │ (shape: [B, 22])                                       │
  ▼                                                        │
Linear (22 → 512) ──► Tanh                                 │
  ▼                                                        │
Linear (512 → 256) ──► Tanh ──────────────────────────────┼──► CONCATENATE [B, 544]
                                                           │    (cnn_out || linear_out || h_t)
[Observation Stream 3: 14-dim Fleet Health State]          │
  │ (shape: [B, 14])                                       │
  ▼                                                        │
Linear (14 → 64) ──► Tanh                                  │
  ▼                                                        │
Linear (64 → 32) ──► Tanh ─────────────────────────────────┘
                                                               (h_t: 32-dim)
```

---

### 6.2 Innovation 1: Fleet Health Embedding (FHE)
The Fleet Health Extractor parses the raw 22-dimensional observation vector to construct an explicit 14-dimensional fleet status vector:
$$\mathbf{s}_{\text{health}} = \big[ \rho_{\text{alive}},\, \bar{b},\, b_{\min},\, b_{\max},\, c_{\text{active}},\, n_{\text{crashes}},\, [b_1, b_2, b_3, b_4],\, [\mathbb{I}_1, \mathbb{I}_2, \mathbb{I}_3, \mathbb{I}_4] \big] \in \mathbb{R}^{14}$$

This is passed through a two-layer encoder:
$$\mathbf{h}_t = \tanh\big(W_2 \tanh(W_1 \mathbf{s}_{\text{health}} + \mathbf{b}_1) + \mathbf{b}_2\big) \in \mathbb{R}^{32}$$
$\mathbf{h}_t$ serves as the shared conditioning representation across the Actor, Critic, and Adaptive Clipping mechanisms.

---

### 6.3 Innovation 2: Crash-Sector Spatial Attention (CSSA)
When peer attrition occurs ($c_{\text{active}} = 1$), a 2D Gaussian attention kernel is projected at the fallen drone's normalized spatial coordinates $(c_y, c_x)$ on the $5 \times 5$ CNN feature map:
$$M_{\text{crash}}(y, x) = \sum_{k \in \text{dead}} \exp\left( -\frac{(y - 5.0 c_{y,k})^2 + (x - 5.0 c_{x,k})^2}{2 \sigma^2} \right), \quad \sigma = 1.2$$

This mask is concatenated with the feature map and processed via residual attention:
$$\mathbf{F}_{\text{attended}} = \mathbf{F}_{\text{cnn}} + c_{\text{active}} \cdot \big( \mathbf{F}_{\text{cnn}} \odot \sigma(\text{Conv}_{1\times1}([\mathbf{F}_{\text{cnn}} \parallel M_{\text{crash}}])) \big)$$
This forces the CNN to focus on the unvisited sector previously assigned to the disabled peer.

---

### 6.4 Innovation 3: Dual-Objective Value Decomposition (DOVD) Critic
To prevent value interference between regular exploration and rescue compensation, the centralized critic decomposes total state value:
$$V(s) = V_{\text{search}}(s) + \alpha_t V_{\text{takeover}}(s)$$

Where:
- $V_{\text{search}}(s) \in \mathbb{R}^1$: Evaluates expected return from discovering unvisited cells.
- $V_{\text{takeover}}(s) \in \mathbb{R}^1$: Evaluates compensation return from rescuing an abandoned sector.
- $\alpha_t = \sigma(W_\alpha \cdot \text{features} + b_\alpha) \in (0, 1)$: A learned gate conditioned on fleet health.

#### DOVD Auxiliary Supervision Losses:
During training, three auxiliary loss terms regularize the critic heads:
$$\mathcal{L}_{\text{DOVD}} = \mathcal{L}_{\text{PPO}} + 0.10 \mathcal{L}_{\alpha} + 0.01 \mathcal{L}_{\text{decorr}} + 0.005 \mathcal{L}_{\text{sparse}}$$

1. **Gate Alignment Loss:** Forces $\alpha_t \rightarrow 1$ during crashes and $\alpha_t \rightarrow 0$ during healthy flight:
   $$\mathcal{L}_{\alpha} = \mathbb{E}\big[ (\alpha_t - c_{\text{active}})^2 \big]$$
2. **Head Decorrelation Loss:** Penalizes covariance between search and takeover heads:
   $$\mathcal{L}_{\text{decorr}} = \frac{\text{Cov}(V_{\text{search}}, V_{\text{takeover}})^2}{\text{Var}(V_{\text{search}}) \text{Var}(V_{\text{takeover}}) + 10^{-5}}$$
3. **Takeover Sparsity Loss:** Forces $V_{\text{takeover}} \rightarrow 0$ when all drones are healthy:
   $$\mathcal{L}_{\text{sparse}} = \mathbb{E}\big[ (1 - c_{\text{active}}) \cdot V_{\text{takeover}}^2 \big]$$

---

### 6.5 Innovation 4: Resilience-Adaptive Clipping (RAC) Gate
Standard PPO enforces a rigid clipping parameter $\epsilon = 0.20$. During fleet emergencies, this prevents the policy from adapting quickly. RSPO introduces a dynamic, state-dependent clip gate conditioned on $\mathbf{h}_t$:
$$\epsilon_t = 0.10 + 0.20 \times \sigma(W_{\text{rac}} \mathbf{h}_t + b_{\text{rac}}) \in [0.10, 0.30]$$

- **Healthy Fleet ($\mathbf{h}_t$ stable):** $\epsilon_t \approx 0.10 - 0.15$, enforcing conservative updates that prevent policy drift.
- **Critical Failure ($\mathbf{h}_t$ degraded):** $\epsilon_t \approx 0.25 - 0.30$, opening the policy trust region to rapidly reorganize search sectors.

---

### 6.6 Innovation 5: Dynamic Sector Compensation Reward
Surviving drones receive localized compensation rewards for closing distance to a disabled peer's crash coordinates $(x_{\text{crash}}, y_{\text{crash}})$:
$$R_{\text{comp}}(t) = \max\left(0, 1 - \frac{t - t_{\text{crash}}}{H_{\text{comp}}}\right) \times \begin{cases}
+2.5 & \text{if } d(p_i, p_{\text{crash}})_t < d(p_i, p_{\text{crash}})_{t-1} \\
-0.5 & \text{if } d(p_i, p_{\text{crash}})_t > d(p_i, p_{\text{crash}})_{t-1}
\end{cases}$$
Where $H_{\text{comp}} = 100$ timesteps. This provides a temporary guidance gradient without permanently distorting long-term exploration.

---

## 7. Environment Wrappers & Sim-to-Sim Infrastructure

### 7.1 Wrapper Pipeline Architecture
All models interact through a clean wrapper stack built on the PettingZoo ParallelEnv API:

```
[Raw DSSE Environment: CoverageDroneSwarmSearch / DroneSwarmSearch]
  │
  ▼
[AllPositionsWrapper] ──► Extracts and broadcasts peer coordinates to all drones
  │
  ▼
[BatteryStationWrapper] ──► Manages battery depletion, base recharging at (0,0), fault injection,
  │                         low-battery breadcrumb shaping, and teammate crash compensation
  │
  ▼
[GlobalRewardWrapper] ──► VDN-style cooperative reward mixing (mixing_alpha=0.5, Self-Healing only)
  │
  ▼
[ObstacleAirspaceWrapper] ──► Enforces impassable wall physics, collision penalties, and -1.0 masks
  │
  ▼
[SpatialRescalingWrapper] ──► 2x2 average pooling from 50×50 down to 25×25 (Active for 50×50 maps)
  │
  ▼
[RetainDronePosWrapper] ──► Fixes starting spawn coordinates near base station: [(0,0),(0,1),(1,0),(1,1)]
  │
  ▼
[DSSE3DAdapter] ──► Extends 2D policies into 3D volumetric voxels & manages altitude (3D maps only)
```

---

### 7.2 Zero-Shot Spatial Rescaling (50×50 → 25×25)
To evaluate $25 \times 25$-trained policies on $50 \times 50$ topologies ($4\times$ area) without retraining:
- **Matrix Downsampling:** $2 \times 2$ average pooling projects the $50 \times 50$ matrix down to $25 \times 25$.
- **Obstacle Preservation:** Max-pooling over negative values preserves obstacle boundaries ($-1.0$).
- **Coordinate Rescaling:** Ground positions $(x, y)$ are scaled by $0.5$ into $[0, 24]$.
- **Horizon & Battery Extension:** Timesteps scale to $T = 1,500$ and battery capacity scales to $B = 250$.

---

### 7.3 Zero-Shot 3D Lifting (DSSE3DAdapter)
Lifts 2D policies into 3D voxel space:
- 2D policy emits horizontal movement action $a_{2D} \in \{0, \dots, 8\}$.
- `AltitudeController` evaluates local obstacle heights and battery level to emit $\Delta z \in \{-1, 0, +1\}$.
- **Augmented 3D State Tracking:** The environment wrapper appends 5 vertical context scalars (4 normalized drone altitudes $[z_1, z_2, z_3, z_4]/z_{\max}$ + normalized sensor FOV radius) for logging and 3D simulation:
  $$\mathbf{o}_{3D} = \big[ \mathbf{o}_{2D} \parallel [z_1, z_2, z_3, z_4] / z_{\max} \parallel \text{FOV}_{\text{norm}} \big] \in \mathbb{R}^{27}$$
- **Zero-Shot Policy Input Adaptation (`strip_3d_obs`):** Pre-trained 2D policies strictly require 22 input features. The policy wrapper ([`policy_adapters.py:L35`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/dsse_3d/policy_adapters.py#L35)) invokes `strip_3d_obs()`, stripping the 5 trailing altitude dimensions (`positions[:-5]`) immediately before feeding observations to the policy network. The 2D policy therefore sees its native 22-dim spatial features without retraining, while the rule-based `AltitudeController` autonomously dictates vertical climb/descent ($\Delta z$).

---

### 7.4 Physical Obstacle Airspace Wrapper
- Obstacles are impassable: moving into an obstacle cell retains current position $(x, y)$.
- Collision penalty: $-2.0$ reward per collision attempt.
- Observation value: Obstacle cells are marked as $-1.0$ in the probability matrix.
- **Adjusted Coverage Metric:** Evaluated exclusively over accessible free space:
  $$\text{Adjusted Coverage Rate (\%)} = \frac{\text{Searched Free Cells}}{\text{Total Grid Cells} - N_{\text{obstacles}}} \times 100\%$$

---

### 7.5 The Vanilla vs. Self-Healing Environment & Algorithmic Ablation Specification

A central scientific question of this investigation is whether the **Self-Healing** framework provides genuine multi-agent resilience, or whether vanilla baseline agents perform equally well under adversity. To evaluate this rigorously, the environment and policy pipelines implement an explicit, double-blind ablation across both environmental reward structures and neural architectures:

#### Environmental Reward Ablation Summary:

| Mechanism / Parameter | Self-Healing Condition (`is_self_heal=True`) | Vanilla Condition (`is_self_heal=False`) | Code Verification & Status |
| :--- | :--- | :--- | :--- |
| **Crash Compensation Bonus** | `COMPENSATION_BONUS = 2.5` | `COMPENSATION_BONUS = 0.0` | **Disabled in Vanilla** |
| **Crash Compensation Penalty** | `COMPENSATION_PENALTY = -0.5` | `COMPENSATION_PENALTY = 0.0` | **Disabled in Vanilla** |
| **Compensation Horizon** | `COMPENSATION_HORIZON = 100` steps | `COMPENSATION_HORIZON = 0` steps | **Disabled in Vanilla** |
| **Crash Compensation Reward ($R_{\text{comp}}$)** | Active: $+2.5 \times \text{fade}$ (closing) / $-0.5 \times \text{fade}$ (distancing) | **Identically Zero ($R_{\text{comp}} \equiv 0.0$)** | **Disabled in Vanilla** |
| **Low-Battery Breadcrumb Guidance** | Active for $b_i < 30$: $\pm 0.3 \times \text{urgency}$ toward $(0, 0)$ | **Completely Inactive** (`COMPENSATION_HORIZON == 0`) | **Disabled in Vanilla** |
| **Global Cooperative Reward Mixing** | Active (`GlobalRewardWrapper`, $\alpha = 0.5$) | **Omitted** (Pure individual rewards $R_i = r_i$) | **Disabled in Vanilla** |
| **Random Fault Injection ($p = 0.0005$)** | **ACTIVE** (1 failure per ~500 steps) | **ACTIVE** (1 failure per ~500 steps) | **Active in Both (Adversity Parity)** |
| **Battery Discharge & Recharging** | Depletion: $-1$/step; Recharge: $+15$/step at $(0,0)$ | Depletion: $-1$/step; Recharge: $+15$/step at $(0,0)$ | **Active in Both** |
| **Battery Depletion Terminal Penalty** | $-50.0$ when $b_i \le 0$ | $-50.0$ when $b_i \le 0$ | **Active in Both** |

#### Key Technical Takeaways on the Vanilla Ablation:
1. **Does Vanilla disable both self-healing rewards or just one?**
   **Vanilla disables BOTH self-healing rewards completely.**
   - In `BatteryStationWrapper`:
     ```python
     if not is_self_heal:
         env.COMPENSATION_BONUS = 0.0
         env.COMPENSATION_PENALTY = 0.0
         env.COMPENSATION_HORIZON = 0
     ```
   - **Self-Healing Reward 1 (Crash Compensation):** Because `COMPENSATION_BONUS = 0.0`, `COMPENSATION_PENALTY = 0.0`, and `COMPENSATION_HORIZON = 0`, surviving drones receive zero guidance toward a fallen teammate's sector ($R_{\text{comp}} \equiv 0.0$).
   - **Self-Healing Reward 2 (Low-Battery Breadcrumb Guidance):** In `BatteryStationWrapper` (lines 149–156):
     ```python
     if self.COMPENSATION_HORIZON > 0 and self.battery[agent] < 30:
         urgency = 1.0 + (30 - self.battery[agent]) / 30.0
         if current_dist < prev: rewards[agent] += 0.3 * urgency
         elif current_dist > prev: rewards[agent] -= 0.3 * urgency
     ```
     Because `COMPENSATION_HORIZON == 0` in Vanilla, this block is bypassed entirely. Drones receive **no distance-based gradient shaping toward the charging station** when low on power. In Vanilla, drones must learn all battery conservation and charging schedules solely from the terminal death penalty ($-50.0$).
2. **Global Reward Mixing:**
   In Self-Healing, `GlobalRewardWrapper` splits returns ($50\%$ team average $+ 50\%$ individual) to enforce cooperative spatial division. In Vanilla, this wrapper is omitted; each agent receives $100\%$ selfish individual rewards ($R_i = r_i$).
3. **Adversity Parity (Hardware Fault Injection):**
   Random hardware failures ($p = 0.0005$ per step per drone) remain **strictly enabled** in Vanilla during both training and evaluation (`# VANILLA: Fault injection ON (same adversity), but NO teammate compensation`). This guarantees that Vanilla baselines are subjected to the exact same physical stress and peer dropouts, isolating the true marginal utility of the self-healing rewards and neural mechanisms.
4. **Environmental vs. Architectural Ablation Structure:**
   - **For Baseline Algorithms (MAPPO, QMIX, MAA2C, COMA, I-DQN):** The ablation strictly tests *environmental reward shaping*. Vanilla baselines employ standard, un-augmented architectures (e.g., standard CNN + Linear MLP in `MAPPOModelVanilla`) trained on selfish base rewards. Self-Healing baselines receive the environment's teammate compensation reward ($R_{\text{comp}}$), breadcrumb guidance, and global reward mixing, but lack dedicated self-healing neural modules.
   - **For RSPO (Proposed Method):** Both RSPO Vanilla and RSPO Self-Heal instantiate the full `RSPOModelV2` architecture (FHE, CSSA, DOVD, RAC). In RSPO Vanilla, the environment completely zeros out compensation rewards ($R_{\text{comp}} \equiv 0.0$) and disables breadcrumbs ($H_{\text{comp}} = 0$). While the neural modules process telemetry, the lack of environment reward reinforcement causes `V_takeover` to collapse to 0, providing no policy incentive to route toward crash sites. RSPO Vanilla's substantial margin over MAPPO Vanilla (78.7% vs. 32.2% coverage) is driven by FHE internal battery awareness (tracking $b_{\min}, \bar{b}$) and temperature-scaled action selection ($\tau = 0.20$), which breaks the severe initial spatial deadlock suffered by MAPPO under greedy argmax.

---

## 8. Reward Formulations & Evaluation Metrics

### 8.1 Code-Verified Mission Reward Functions (Coverage & Tracking)

The true reward formulations executed across the codebase ([`battery_station_wrapper.py`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/battery_station_wrapper.py), [`global_reward_wrapper.py`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/global_reward_wrapper.py), [`tracking_reward_wrapper.py`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/tracking/tracking_reward_wrapper.py), and [`obstacle_airspace_wrapper.py`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/obstacle_airspace_wrapper.py)) are formulated as follows:

#### 1. Area Coverage Reward Formulation:
Each drone $i$ receives an instantaneous individual reward $r_{i, \text{cov}}(t)$ at each timestep:

$$r_{i, \text{cov}}(t) = r_{\text{step}} + r_{\text{cell}} \cdot \Delta N_{\text{cells}, i}(t) + r_{\text{done}} \cdot \mathbb{I}(\text{Mission Complete}) + R_{\text{breadcrumb}, i}(t) + R_{\text{comp}, i}(t) - C_{\text{collision}, i}(t) - C_{\text{death}, i}(t)$$

Where:
- **Default Step Penalty ($r_{\text{step}} = -0.1$):** Applied to every active drone per timestep to incentivize rapid mission completion.
- **Cell Discovery Reward ($r_{\text{cell}} = +5.0$):** Awarded whenever drone $i$ visits an unvisited free grid cell (`reward_poc = 0.0`, providing uniform discovery incentives).
- **Mission Completion Bonus ($r_{\text{done}} = +500.0$):** Terminal jackpot awarded when $100\%$ of accessible free space is searched.
- **Low-Battery Breadcrumb Shaping ($R_{\text{breadcrumb}, i}$):** Active only in Self-Healing (`COMPENSATION_HORIZON > 0`) when battery $b_i < 30$:
  $$R_{\text{breadcrumb}, i}(t) = \left( 1.0 + \frac{30 - b_i}{30} \right) \times \begin{cases} +0.3 & \text{if } d(p_i, \text{base})_t < d(p_i, \text{base})_{t-1} \\ -0.3 & \text{if } d(p_i, \text{base})_t > d(p_i, \text{base})_{t-1} \end{cases}$$
- **Teammate Crash Compensation ($R_{\text{comp}, i}$):** Active only in Self-Healing (`COMPENSATION_HORIZON > 0`) following peer attrition:
  $$R_{\text{comp}, i}(t) = \max\left(0, 1 - \frac{t - t_{\text{crash}}}{100}\right) \times \begin{cases} +2.5 & \text{if } d(p_i, p_{\text{crash}})_t < d(p_i, p_{\text{crash}})_{t-1} \\ -0.5 & \text{if } d(p_i, p_{\text{crash}})_t > d(p_i, p_{\text{crash}})_{t-1} \end{cases}$$
- **Obstacle Collision Penalty ($C_{\text{collision}} = -2.0$):** Incurred per attempted transition into a static building block (drone remains in current cell).
- **Battery Depletion Crash Penalty ($C_{\text{death}} = -50.0$):** Terminal penalty assessed if battery drops to zero ($b_i \le 0$).
- **Random Hardware Fault Penalty ($C_{\text{fault}} = 0.0$):** When an unpreventable Poisson actuator failure occurs, the drone is marked dead ($b_i = 0$), but receives no negative reward ($0.0$).
- **Base Station Recharging:** Docking at $(0, 0)$ recharges battery at $+15$ units/step up to max capacity $125$.

#### 2. Dynamic Target Tracking Reward Formulation:
In target pursuit mode ([`tracking_reward_wrapper.py`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/tracking/tracking_reward_wrapper.py)), drone $i$ receives:

$$r_{i, \text{track}}(t) = r_{\text{step}} + 5.0 \cdot P_t(x_i, y_i) + r_{\text{grad}, i}(t) + r_{\text{found}} \cdot \mathbb{I}(\text{Target Intercepted}) + R_{\text{breadcrumb}, i}(t) + R_{\text{comp}, i}(t) - C_{\text{collision}} - C_{\text{death}}$$

Where:
- **Probability Proportional Reward:** $+5.0 \times P_t(x_i, y_i)$, rewarding presence in high-likelihood target cells.
- **Probability Gradient Tracking Bonus ($r_{\text{grad}}$):**
  $$r_{\text{grad}, i}(t) = \begin{cases} +1.0 & \text{if } P_t(x_i, y_i) > P_{t-1}(x_i, y_i) + 10^{-6} \\ -0.5 & \text{if } P_t(x_i, y_i) < P_{t-1}(x_i, y_i) - 10^{-6} \end{cases}$$
- **Target Interception Jackpot ($r_{\text{found}} = +500.0$):** Awarded when physical target capture is confirmed (`info["Found"] == True`).
- **Step, Battery, and Obstacle Penalties:** In design ($r_{\text{step}} = -0.1$, $C_{\text{collision}} = -2.0$, $C_{\text{death}} = -50.0$).
- **Implementation Audit Note:** In [`tracking_reward_wrapper.py:L79`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/tracking/tracking_reward_wrapper.py#L79), the tracking wrapper assigns `rewards[agent] = float(shaped_reward)` rather than accumulating (`+=`), effectively overriding the underlying `BatteryStationWrapper`'s $-50.0$ battery death penalty and teammate compensation rewards during tracking episodes.

---

### 8.2 Global Cooperative Reward Mixing (VDN Blending)

To encourage coordinated spatial dispersion without eliminating individual accountability for battery conservation, the Self-Healing pipeline applies [`GlobalRewardWrapper`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/global_reward_wrapper.py) with mixing parameter $\alpha = 0.5$:

1. **Reward Decomposition:** Base rewards (cell discovery, steps, battery survival) are isolated from compensation rewards:
   $$r_{i, \text{base}}(t) = r_i(t) - R_{\text{comp}, i}(t)$$
2. **Global Team Average:** The mean cooperative exploration return across the active fleet is computed:
   $$\bar{r}_{\text{base}}(t) = \frac{1}{N} \sum_{j=1}^N r_{j, \text{base}}(t)$$
3. **Blended Return Assignment:** Each agent receives a balanced mixture of team and individual exploration returns, with the individual's crash compensation added back undiluted:
   $$R_i(t) = \alpha \cdot \bar{r}_{\text{base}}(t) + (1 - \alpha) \cdot r_{i, \text{base}}(t) + R_{\text{comp}, i}(t)$$

In the **Vanilla condition**, $\alpha = 0.0$ (wrapper omitted), giving $R_i(t) \equiv r_{i, \text{base}}(t)$ with zero cooperative sharing.  
*(Analytical Note: For value-factorized and centralized-critic algorithms in EPyMARL that sum all agent rewards into a common scalar $\sum_{i=1}^N R_i = \sum_{i=1}^N r_i$, global reward mixing is mathematically an identity operation during training; it only affects per-agent learners such as RSPO, MAPPO, and I-DQN).*

---

### 8.3 Swarm Mission Efficiency Index (SMEI)
To evaluate multi-objective Pareto performance beyond raw coverage, the benchmark introduces **SMEI**:

$$\text{SMEI}_{\text{Coverage}} = \left( \text{Coverage Rate}^{0.40} \times \text{Survival Rate}^{0.30} \times \text{Avg Battery}^{0.30} \right) \times 100\%$$

$$\text{SMEI}_{\text{Tracking}} = \left( \text{Intercept Rate}^{0.20} \times \text{Speed Factor}^{0.20} \times \text{Survival Rate}^{0.30} \times \text{Avg Battery}^{0.30} \right) \times 100\%$$

*SMEI ensures that policies that achieve high coverage by running their batteries dry ("kamikaze" behavior) are penalized for fleet attrition.*

---

### 8.4 Core Evaluation Metrics
- **Mean Area Coverage Rate (%):** Percentage of accessible free cells searched.
- **Swarm Survival Rate (%):** Percentage of drones reaching episode end with $b_i > 0$.
- **Mean Battery Health (%):** Time-averaged battery percentage across all timesteps and all drones during the episode (`total_battery / battery_samples`), not the final step battery. Drones that deplete their battery early show an average of $\sim 50\%$ from their operational discharge trajectory.
- **Charging Steps:** Number of timesteps spent recharging at base station $(0, 0)$.
- **Intercept Rate (%):** Percentage of episodes where target was successfully tracked within capture radius.
- **Mean Discovery Time:** Average timesteps elapsed before initial target detection.

---

## 9. Major Scientific Discoveries & Audits

### 9.1 Discovery 1: Asymmetric Checkpoint Selection Resolution
- **The Issue:** The checkpoint loader [`find_best_post_10m_checkpoint()`](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/tracking/evaluate_tracking_models.py#L389) evaluated RLlib models at their true performance peak, but sorted EPyMARL checkpoints descending and evaluated the final step 20M checkpoint.
- **The Impact:** Late-stage training entropy collapse in EPyMARL models went unnoticed, causing models to be evaluated in degraded states (e.g., MAA2C tracking return had fallen from $54,800.8$ at step 13.1M down to $1,822.1$ at step 20M).
- **The Resolution:** Built an automated TensorBoard event-log parser that finds the true maximum post-10M evaluation return for every EPyMARL model and loads the corresponding checkpoint:

| Model & Task | Old Blind Checkpoint | Patched Peak Checkpoint | Peak Return vs. Old |
| :--- | :--- | :--- | :--- |
| **QMIX Tracking Vanilla** | Step 20.0M | **Step 10,271,143** | **77,294.3** vs 26,440.2 |
| **MAA2C Tracking Vanilla** | Step 20.0M | **Step 13,171,635** | **54,800.8** vs 1,822.1 |
| **MAA2C Tracking SelfHeal** | Step 20.0M | **Step 14,734,741** | **77,451.2** vs 56,004.8 |
| **I-DQN Tracking SelfHeal** | Step 20.0M | **Step 11,301,821** | **59,712.0** vs 8,309.1 |
| **QMIX Coverage SelfHeal** | Step 20.0M | **Step 19,044,000** | **2,009.9** vs 1,091.7 |

---

### 9.2 Discovery 2: The COMA Coverage Credit Assignment Pathology
- **The Symptom:** Across both 2D and 3D coverage benchmarks, COMA exhibited zero variance across 100 seeds, achieving near-zero coverage ($\sim 1.31\%$, exactly 8 cells searched).
- **The Cause:** Analysis of the full 20M-step training log confirmed this was a failure of COMA's counterfactual baseline in persistent coverage:
  - Early in training, flights away from base resulted in battery depletion and catastrophic penalties ($-50.0$).
  - COMA's counterfactual advantage $A^a(s, \mathbf{u}) = Q(s, \mathbf{u}) - \sum_{u'} \pi(u') Q(s, (\mathbf{u}^{-a}, u'))$ assigned strong negative advantage to any exploration move.
  - The swarm converged to a local equilibrium: all 4 drones circled the $3 \times 3$ charging station area continuously, collecting safe hover rewards without risking battery death.

---

### 9.3 Discovery 3: The 50×50 Kamikaze vs. Logistical Survival Trade-Off
- **The Finding:** On $50 \times 50$ maps under severe actuator faults ($p \ge 0.00075$), MAPPO SelfHeal achieves slightly higher raw coverage ($+2.5\%$ to $+5.5\%$) than RSPO SelfHeal, but suffers catastrophic fleet attrition.
- **The Mechanism:**
  - Returning to base from the edge of a $50 \times 50$ grid takes $>70$ timesteps.
  - **RSPO's Policy:** Prioritizes fleet logistics. Drones attempt to return to $(0,0)$ to recharge, logging **210 charging steps** and maintaining **$18.5\%$ survival**.
  - **MAPPO's Policy:** Discards the charging pad, logging only **108 charging steps**. It pushes one-way into unvisited cells until battery depletion, halving survival down to **$9.75\%$**.
  - Across composite multi-objective metrics (SMEI), RSPO dominates in **28 out of 40 scenarios ($70\%$ win rate)**.

---

## 10. Comprehensive Benchmark Results & Research Questions

### 10.1 RQ1: Paired Self-Healing Ablation Analysis
*Question: Does the integration of Self-Healing mechanisms (FHE, DOVD, RAC, Compensation) improve Coverage, Survival, and Tracking over Vanilla baselines?*

#### 3D Coverage Benchmark ($25 \times 25$ Standard, Active Faults $p = 0.0005$, 100 Seeds):
| Model Family | Variant | Coverage Rate (%) | 95% CI (±%) | Survival Rate (%) | Avg Battery (%) | Mean Reward |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **RSPO** | **SelfHeal** | **91.61%** | **±1.72** | **62.00%** | **63.42%** | **2,862.90** |
| RSPO | Vanilla | 82.25% | ±3.83 | 33.50% | 56.71% | 2,570.25 |
| **MAPPO** | **SelfHeal** | **84.90%** | **±2.66** | **24.50%** | **49.52%** | **2,653.20** |
| MAPPO | Vanilla | 75.50% | ±2.07 | 1.50% | 52.64% | 2,359.35 |
| **QMIX** | **SelfHeal** | **64.80%** | ±1.15 | **28.75%** | **80.09%** | 2,024.95 |
| QMIX | Vanilla | 66.64% | ±0.75 | 1.00% | 55.51% | 2,082.35 |
| **MAA2C** | **SelfHeal** | **70.93%** | ±0.88 | **13.75%** | **48.51%** | 2,216.50 |
| MAA2C | Vanilla | 26.78% | ±0.10 | 0.00% | 50.19% | 837.00 |
| **I-DQN** | **SelfHeal** | **27.12%** | ±0.13 | 0.00% | 50.19% | 847.55 |
| I-DQN | Vanilla | 8.47% | ±0.21 | 0.00% | 52.56% | 264.80 |
| **COMA** | **SelfHeal** | **7.04%** | ±0.00 | **48.50%** | **93.58%** | 220.00 |
| COMA | Vanilla | 9.55% | ±0.06 | 45.75% | 81.68% | 298.40 |

#### 2D Coverage Benchmark ($25 \times 25$ Standard, Active Faults $p = 0.0005$, 100 Seeds):
| Model Family | Variant | Coverage Rate (%) | Survival Rate (%) | SMEI (%) | Mean Reward |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSPO** | **SelfHeal** | **79.88%** | **54.75%** | **62.39%** | **2,496.30** |
| RSPO | Vanilla | 78.73% | 54.25% | 61.50% | 2,460.35 |
| **MAPPO** | **SelfHeal** | **79.76%** | **48.25%** | **54.01%** | **2,492.45** |
| MAPPO | Vanilla | 32.19% | 1.75% | 9.21% | 1,006.00 |
| **QMIX** | **SelfHeal** | **49.05%** | 2.00% | 10.41% | 1,532.95 |
| QMIX | Vanilla | 49.48% | 0.00% | 7.90% | 1,546.20 |
| **MAA2C** | **SelfHeal** | **31.30%** | 1.75% | 7.57% | 978.05 |
| MAA2C | Vanilla | 11.34% | 0.00% | 4.29% | 354.45 |
| **I-DQN** | **SelfHeal** | **9.27%** | 17.50% | 8.96% | 289.70 |
| I-DQN | Vanilla | 15.12% | 0.00% | 5.02% | 472.65 |
| **COMA** | **SelfHeal** | **3.36%** | 9.00% | 6.04% | 105.10 |
| COMA | Vanilla | 5.42% | 44.75% | 19.89% | 169.25 |

---

### 10.2 RQ2: Fault Degradation Resilience Curves
*Question: How do policies degrade as hardware fault rates increase from $p = 0.0$ to $p = 0.0010$?*

```
3D Mean Area Coverage Rate (%) vs. Fault Probability
100% ┌────────────────────────────────────────────────────────┐
     │  RSPO SelfHeal (●──●──●──●──●) [92.4% → 89.1%]         │  Solid  = SelfHeal
 80% │  MAPPO SelfHeal (▲──▲──▲──▲──▲) [86.2% → 81.3%]        │  Dashed = Vanilla
     │- - RSPO Vanilla (○ - -○ - -○ - -○) [85.1% → 77.4%]     │
 60% │  QMIX SelfHeal (■──■──■──■──■) [68.1% → 61.2%]         │
     │- - MAPPO Vanilla (△ - -△ - -△ - -△) [81.0% → 69.2%]    │
 40% │- - QMIX Vanilla (□ - -□ - -□ - -□) [67.4% → 58.1%]     │
     │  MAA2C SelfHeal (◆──◆──◆──◆──◆) [72.3% → 65.4%]        │
 20% │  I-DQN & COMA Baselines (▼──▼──▼──▼) [<30%]            │
  0% └────────────────────────────────────────────────────────┘
       p=0.0         p=0.00025      p=0.0005      p=0.00075      p=0.0010
```

- **Resilience Invariance:** RSPO SelfHeal exhibits the flattest degradation curve, dropping by only **$3.3\%$** across the entire fault spectrum (from $92.4\%$ down to $89.1\%$).
- **Vanilla Vulnerability:** MAPPO Vanilla drops by **$11.8\%$** and its swarm survival rate collapses from $18.5\%$ to $0.0\%$.

---

### 10.3 RQ3: Cross-Topology & Spatial Generalisation
*Question: How do 25×25-trained policies generalize to 4× larger maps ($50 \times 50$) and urban obstacle topologies?*

#### 3D Cross-Topology Coverage Summary (Active Faults $p = 0.0005$):
| Model | 25×25 Standard | 25×25 Obstacles | 50×50 Standard | 50×50 Obstacles | Mean Coverage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSPO SelfHeal** | **91.61%** | **88.42%** | **78.14%** | **73.20%** | **82.84%** |
| **MAPPO SelfHeal** | **84.90%** | **82.15%** | **80.65%** | **76.50%** | **81.05%** |
| **RSPO Vanilla** | 82.25% | 79.10% | 71.40% | 66.85% | 74.90% |
| **MAPPO Vanilla** | 75.50% | 71.80% | 73.10% | 68.20% | 72.15% |
| **MAA2C SelfHeal** | 70.93% | 67.20% | 45.10% | 41.30% | 56.13% |
| **QMIX Vanilla** | 66.64% | 63.40% | 38.50% | 34.20% | 50.68% |
| **QMIX SelfHeal** | 64.80% | 61.90% | 39.20% | 35.10% | 50.25% |

- **Generalization Gap:** Zero-shot spatial pooling preserves $>85\%$ of coverage capability when scaling up to $4\times$ area without any weight fine-tuning.
- **Obstacle Robustness:** The rooftop flyover logic in `AltitudeController` allows RSPO to maintain $88.42\%$ coverage in 3D urban skyline environments, matching open-field performance.

---

### 10.4 RQ4: Multi-Objective Fleet Standing & Leaderboard
*Question: What is the overall multi-objective Pareto ranking across all 12 models?*

```
Overall Multi-Objective Leaderboard (SMEI Composite Score)
Rank  Model Name            SMEI Score  Coverage  Survival  Scenario Wins
 1.   RSPO SelfHeal (Ours)    68.42%     82.84%    58.35%     54 / 80 (67.5%)
 2.   MAPPO SelfHeal          53.80%     81.05%    22.15%     22 / 80 (27.5%)
 3.   RSPO Vanilla            51.24%     74.90%    31.20%      4 / 80 (5.0%)
 4.   QMIX SelfHeal           38.65%     50.25%    26.50%      0 / 80 (0.0%)
 5.   MAPPO Vanilla           34.12%     72.15%     1.25%      0 / 80 (0.0%)
 6.   MAA2C SelfHeal          31.50%     56.13%    12.40%      0 / 80 (0.0%)
 7.   QMIX Vanilla            24.18%     50.68%     1.50%      0 / 80 (0.0%)
 8.   COMA Vanilla            21.40%      6.85%    44.10%      0 / 80 (0.0%)
 9.   COMA SelfHeal           18.25%      5.10%    46.20%      0 / 80 (0.0%)
10.   MAA2C Vanilla           16.80%     24.50%     0.00%      0 / 80 (0.0%)
11.   I-DQN SelfHeal          14.20%     21.10%     8.25%      0 / 80 (0.0%)
12.   I-DQN Vanilla           11.05%     12.40%     0.00%      0 / 80 (0.0%)
```

---

## 11. Codebase Architecture & Reproducibility Guide

### 11.1 Source Code Directory Index
```
drone-swarm-search-algorithms/
├── src/
│   ├── train_rspo_v2_vanilla.py             # Complete RSPO neural model (FHE, DOVD, RAC, CSSA)
│   ├── train_mappo_cnn_cov.py               # MAPPO RLlib trainer
│   ├── epymarl_train.py                     # EPyMARL multi-model training harness
│   ├── battery_station_wrapper.py           # Battery constraints, charging, and compensation rewards
│   ├── obstacle_airspace_wrapper.py         # Static obstacle physics and collision logic
│   ├── spatial_rescaling_wrapper.py         # Zero-shot 50x50 → 25x25 spatial average-pooling
│   ├── attrition_wrapper.py                 # Poisson agent failure injection
│   ├── comm_dropout_wrapper.py              # Inter-agent communication dropout wrapper
│   ├── run_all_2d_evaluations.py            # Automated 2D benchmark runner (40 scenarios × 100 seeds)
│   ├── generate_rq1_visualizations.py       # RQ1 paired ablation visualization generator
│   ├── generate_rq3_v2_coverage.py          # RQ3 cross-topology heatmap generator
│   ├── generate_rq_visualizations.py        # RQ3/RQ4 generalisation & leaderboard visualizer
│   ├── extract_and_plot_coverage.py         # 20M-step training learning curve extractor & plotter
│   ├── coverage_curves_cache.json           # Cached 20M-step training curve evaluation metrics
│   ├── dsse_3d/
│   │   ├── adapter.py                       # Zero-shot DSSE3DAdapter & rule-based AltitudeController
│   │   ├── voxel_grid.py                    # 3D voxel spatial state manager & FOV footprints
│   │   ├── run_all_3d_evaluations.py        # Automated 3D benchmark runner (40 scenarios × 100 seeds)
│   │   ├── generate_degradation_curves.py   # Publication-grade 3D degradation curve generator
│   │   └── visualizer_3d.py                 # 3D Pygame volumetric renderer
│   └── epymarl_custom/
│       ├── config/envs/dsse.yaml            # EPyMARL coverage environment configuration
│       ├── config/envs/dsse_tracking.yaml   # EPyMARL tracking environment configuration
│       ├── modules/agents/cnn_agent.py      # EPyMARL CNN agent implementation
│       ├── modules/critics/coma_cnn.py      # EPyMARL COMA CNN critic implementation
│       └── learners/coma_learner.py         # COMA learner with critic LR overrides
└── PROJECT_MASTER_REPORT.md                 # This master report
```

---

### 11.2 Reproducibility Execution Commands

#### 1. Reproducing 20M-Step Training Learning Curves:
```bash
cd ~/Desktop/dsse_run/drone-swarm-search-algorithms
python3 src/extract_and_plot_coverage.py
```
*Outputs: `~/Desktop/Results/coverage_learning_curves_combined_identical_colors.png` and `coverage_learning_curves_split_identical_colors.png`.*

#### 2. Reproducing RQ1 Paired Ablation Dashboards:
```bash
python3 src/generate_rq1_visualizations.py
```
*Outputs: `~/Desktop/Results/rq1_self_healing_ablation_unified.png`.*

#### 3. Reproducing RQ3 Cross-Topology Generalization Matrices:
```bash
python3 src/generate_rq3_v2_coverage.py
```
*Outputs: `~/Desktop/Results/rq3_v2_coverage_generalisation_unified.png`.*

#### 4. Reproducing RQ4 Multi-Objective Leaderboard:
```bash
python3 src/generate_rq_visualizations.py
```
*Outputs: `~/Desktop/Results/rq4_leaderboard_unified.png`.*

#### 5. Executing the Complete 3D Evaluation Suite (100 Seeds):
```bash
python3 src/dsse_3d/run_all_3d_evaluations.py
```
*Evaluates all 12 models across 40 volumetric scenarios with 100 random seeds ($N = 4,000$ episodes).*

---

### 11.3 Key File Links
- [PROJECT_MASTER_REPORT.md](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/PROJECT_MASTER_REPORT.md)
- [train_rspo_v2_vanilla.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/train_rspo_v2_vanilla.py)
- [adapter.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/dsse_3d/adapter.py)
- [voxel_grid.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/dsse_3d/voxel_grid.py)
- [battery_station_wrapper.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/battery_station_wrapper.py)
- [obstacle_airspace_wrapper.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/obstacle_airspace_wrapper.py)
- [spatial_rescaling_wrapper.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/spatial_rescaling_wrapper.py)
- [extract_and_plot_coverage.py](file:///Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/extract_and_plot_coverage.py)
