"""
Dual Benchmark Results Suites (No_Faults_0.0 vs Active_Faults_0.0005)
=====================================================================
Generates 8 dashboards (4 spatial regimes × 2 fault modes) across 12 MARL models × 100 seeds.

Output Structure:
  ~/Desktop/Results/
  ├── No_Faults_0.0/
  │   ├── 1_dashboard_25x25_standard.png
  │   ├── 2_dashboard_25x25_obstacles.png
  │   ├── 3_dashboard_50x50_spatial_scaling.png
  │   ├── 4_dashboard_50x50_obstacles.png
  │   └── no_faults_summary.csv
  └── Active_Faults_0.0005/
      ├── 1_dashboard_25x25_standard.png
      ├── 2_dashboard_25x25_obstacles.png
      ├── 3_dashboard_50x50_spatial_scaling.png
      ├── 4_dashboard_50x50_obstacles.png
      └── active_faults_summary.csv
"""

import os
import sys
import gc
import pickle
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt

# Add source paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

import ray
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from obstacle_airspace_wrapper import ObstacleAirspaceWrapper
from spatial_rescaling_wrapper import SpatialRescalingWrapper
from modules.agents.cnn_agent import CNNAgent
from modules.agents.rnn_agent import RNNAgent

from train_rspo_v2_vanilla import RSPOModelV2
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla


# ══════════════════════════════════════════════════════════════════════
# Utility Classes (Pickle compatibility, Policy Wrapper, MockArgs)
# ══════════════════════════════════════════════════════════════════════

class MockArgs:
    def __init__(self, use_rnn=True, hidden_dim=128, n_actions=9):
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions


class DummyVersion:
    def __init__(self, *args, **kwargs): pass
    def __setstate__(self, state): pass

class VersionCompatibilityUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Version" or "version" in module:
            return DummyVersion
        return super().find_class(module, name)


class DirectPyTorchPolicyWrapper:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def compute_single_action(self, observation, policy_id="default_policy"):
        pos, matrix = observation
        pos_t = th.tensor(pos, dtype=th.float32).unsqueeze(0)
        mat_t = th.tensor(matrix, dtype=th.float32).unsqueeze(0)
        with th.no_grad():
            obs_dict = {"obs": (pos_t, mat_t)}
            logits, _ = self.model(obs_dict, [], None)
            action = int(logits.argmax(dim=-1).item())
        return action

    def stop(self): pass


# ══════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════

def load_rllib_policy(custom_model_cls, model_name, checkpoint_path):
    from gymnasium.spaces import Box, Tuple as GymTuple, Discrete
    obs_space = GymTuple([Box(-1.0, 1.0, (22,), dtype=np.float32), Box(0.0, 1.0, (25, 25), dtype=np.float32)])
    act_space = Discrete(9)

    model = custom_model_cls(obs_space, act_space, 9, {}, model_name)

    if checkpoint_path.endswith(".pt"):
        ckpt = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        tensor_weights = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(tensor_weights, strict=False)
    else:
        policy_state_path = os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")
        if not os.path.exists(policy_state_path):
            policy_state_path = os.path.join(checkpoint_path, "policy_state.pkl")

        with open(policy_state_path, "rb") as f:
            policy_state = VersionCompatibilityUnpickler(f).load()

        if "weights" in policy_state and isinstance(policy_state["weights"], dict):
            tensor_weights = {k: th.from_numpy(v) if not isinstance(v, th.Tensor) else v for k, v in policy_state["weights"].items()}
            model.load_state_dict(tensor_weights, strict=False)

    return DirectPyTorchPolicyWrapper(model)


# ══════════════════════════════════════════════════════════════════════
# Environment Creator Factory
# ══════════════════════════════════════════════════════════════════════

def make_env_creator(grid_size, fault_prob, selfheal, obstacles):
    """
    Factory function that returns an environment creator callable.

    Args:
        grid_size: 25 or 50
        fault_prob: 0.0 or 0.0005
        selfheal: True for SelfHeal compensation, False for Vanilla
        obstacles: True to include obstacle airspace mask
    """
    def create_env():
        if grid_size == 25:
            timestep_limit = 750
            max_battery = 125
            if obstacles:
                obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_25.npy"))
                prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_25.npy")
            else:
                obstacle_mask = None
                prob_path = os.path.join(SRC_DIR, "uniform_matrix_25.npy")
        elif grid_size == 50:
            timestep_limit = 1500
            max_battery = 250  # Doubled battery for 50x50
            if obstacles:
                obstacle_mask = np.load(os.path.join(SRC_DIR, "obstacle_skyline_50.npy"))
                prob_path = os.path.join(SRC_DIR, "obstacle_prob_matrix_50.npy")
            else:
                obstacle_mask = None
                prob_path = os.path.join(SRC_DIR, "uniform_matrix_50.npy")
        else:
            raise ValueError(f"Unsupported grid_size: {grid_size}")

        env = CoverageDroneSwarmSearch(timestep_limit=timestep_limit, drone_amount=4, prob_matrix_path=prob_path)
        env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
        env = AllPositionsWrapper(env)
        env = BatteryStationWrapper(env, max_battery=max_battery, depletion_rate=1, charge_rate=15, fault_prob=fault_prob)

        if selfheal:
            env.COMPENSATION_BONUS = 2.5
            env.COMPENSATION_PENALTY = -0.5
            env.COMPENSATION_HORIZON = 100
        else:
            env.COMPENSATION_BONUS = 0.0
            env.COMPENSATION_PENALTY = 0.0
            env.COMPENSATION_HORIZON = 0

        if obstacles and obstacle_mask is not None:
            env = ObstacleAirspaceWrapper(env, obstacle_mask)

        if grid_size == 50:
            env = SpatialRescalingWrapper(env, target_grid_size=25)

        return RetainDronePosWrapper(env, [(0, 0), (0, 1), (1, 0), (1, 1)])

    return create_env


# ══════════════════════════════════════════════════════════════════════
# Helper to Get Base Probability Matrix
# ══════════════════════════════════════════════════════════════════════

def _get_base_prob_matrix(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    if hasattr(base, "probability_matrix") and hasattr(base.probability_matrix, "get_matrix"):
        return base.probability_matrix.get_matrix()
    return None


# ══════════════════════════════════════════════════════════════════════
# Evaluators with CSRR
# ══════════════════════════════════════════════════════════════════════

def evaluate_rllib_model(algo, env_fn, num_seeds=100, seed_start=1000):
    episode_records = []
    for seed in range(seed_start, seed_start + num_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)
        env = env_fn()
        obs, infos = env.reset(seed=seed)
        batt_samples = []
        chg_steps = 0
        step_idx = 0
        ep_crashes = 0
        crash_unsearched_masks = []

        while env.agents:
            step_idx += 1
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    actions[agent] = algo.compute_single_action(observation=obs[agent], policy_id="default_policy")

            obs, rewards, terminations, truncations, infos = env.step(actions)

            for a, info in infos.items():
                if isinstance(info, dict):
                    batt = info.get("battery")
                    if batt is not None:
                        batt_samples.append(batt)
                    if info.get("charging", False):
                        chg_steps += 1
                    if info.get("stranded_event", False):
                        ep_crashes += 1
                        crash_coords = info.get("crash_coords")
                        prob_mat = _get_base_prob_matrix(env)
                        if crash_coords is not None and prob_mat is not None:
                            cy, cx = crash_coords
                            H, W = prob_mat.shape
                            y_min, y_max = max(0, cy - 2), min(H, cy + 3)
                            x_min, x_max = max(0, cx - 2), min(W, cx + 3)
                            unsearched_sub = (prob_mat[y_min:y_max, x_min:x_max] > 0.0).copy()
                            crash_unsearched_masks.append(((y_min, y_max, x_min, x_max), unsearched_sub))

        prob_mat = _get_base_prob_matrix(env)
        sample_info = next(iter(infos.values()), {}) if infos else {}
        cov = sample_info.get("coverage_rate", 0.0) if isinstance(sample_info, dict) else 0.0
        avg_batt = np.mean(batt_samples) if batt_samples else 0.0

        # Compute CSRR
        csrr = 100.0
        if crash_unsearched_masks and prob_mat is not None:
            recovered_pcts = []
            for (y_min, y_max, x_min, x_max), unsearched_at_crash in crash_unsearched_masks:
                still_unsearched = (prob_mat[y_min:y_max, x_min:x_max] > 0.0)
                initial_count = unsearched_at_crash.sum()
                if initial_count > 0:
                    recovered = (unsearched_at_crash & (~still_unsearched)).sum()
                    recovered_pcts.append((recovered / float(initial_count)) * 100.0)
                else:
                    recovered_pcts.append(100.0)
            csrr = float(np.mean(recovered_pcts))

        episode_records.append({
            "seed": seed,
            "coverage_rate": cov,
            "agent_deaths": ep_crashes,
            "avg_battery_level": avg_batt,
            "charging_steps": chg_steps,
            "csrr": csrr,
        })
    return episode_records


def evaluate_epymarl_model(model_th_path, env_fn, num_seeds=100, seed_start=1000):
    state_dict = th.load(model_th_path, map_location=lambda storage, loc: storage)
    if "fc1.weight" in state_dict:
        hidden_dim = state_dict["fc1.weight"].shape[0]
        mock_args = MockArgs(use_rnn=True, hidden_dim=hidden_dim, n_actions=9)
        agent = RNNAgent(input_shape=651, args=mock_args)
    else:
        hidden_dim = state_dict["fc2.weight"].shape[1] if "fc2.weight" in state_dict else 128
        mock_args = MockArgs(use_rnn=True, hidden_dim=hidden_dim, n_actions=9)
        agent = CNNAgent(input_shape=651, args=mock_args)

    agent.load_state_dict(state_dict, strict=False)
    agent.eval()

    episode_records = []
    for seed in range(seed_start, seed_start + num_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)
        env = env_fn()
        obs_dict, info_dict = env.reset(seed=seed)
        agents = list(env.possible_agents)
        n_agents = len(agents)
        hidden_states = agent.init_hidden().expand(n_agents, -1).clone()
        obs_size = 22 + 625
        batt_samples = []
        chg_steps = 0
        step_idx = 0
        ep_crashes = 0
        crash_unsearched_masks = []

        while env.agents:
            step_idx += 1
            inputs = []
            for i, agent_name in enumerate(agents):
                if agent_name in obs_dict:
                    positions, matrix = obs_dict[agent_name]
                    flat_obs = np.concatenate([positions.astype(np.float32).flatten(), matrix.astype(np.float32).flatten()])
                else:
                    flat_obs = np.zeros(obs_size, dtype=np.float32)

                agent_id_onehot = np.zeros(n_agents, dtype=np.float32)
                agent_id_onehot[i] = 1.0
                inputs.append(np.concatenate([flat_obs, agent_id_onehot]))

            inputs_tensor = th.tensor(np.stack(inputs), dtype=th.float32)
            with th.no_grad():
                q_logits, hidden_states = agent(inputs_tensor, hidden_states)
                actions_tensor = q_logits.argmax(dim=-1)

            actions = {agent_name: int(actions_tensor[i].item()) for i, agent_name in enumerate(agents) if agent_name in env.agents}
            obs_dict, rewards, terminations, truncations, info_dict = env.step(actions)

            for a, info in info_dict.items():
                if isinstance(info, dict):
                    batt = info.get("battery")
                    if batt is not None:
                        batt_samples.append(batt)
                    if info.get("charging", False):
                        chg_steps += 1
                    if info.get("stranded_event", False):
                        ep_crashes += 1
                        crash_coords = info.get("crash_coords")
                        prob_mat = _get_base_prob_matrix(env)
                        if crash_coords is not None and prob_mat is not None:
                            cy, cx = crash_coords
                            H, W = prob_mat.shape
                            y_min, y_max = max(0, cy - 2), min(H, cy + 3)
                            x_min, x_max = max(0, cx - 2), min(W, cx + 3)
                            unsearched_sub = (prob_mat[y_min:y_max, x_min:x_max] > 0.0).copy()
                            crash_unsearched_masks.append(((y_min, y_max, x_min, x_max), unsearched_sub))

        prob_mat = _get_base_prob_matrix(env)
        sample_info = next(iter(info_dict.values()), {}) if info_dict else {}
        cov = sample_info.get("coverage_rate", 0.0) if isinstance(sample_info, dict) else 0.0
        avg_batt = np.mean(batt_samples) if batt_samples else 0.0

        csrr = 100.0
        if crash_unsearched_masks and prob_mat is not None:
            recovered_pcts = []
            for (y_min, y_max, x_min, x_max), unsearched_at_crash in crash_unsearched_masks:
                still_unsearched = (prob_mat[y_min:y_max, x_min:x_max] > 0.0)
                initial_count = unsearched_at_crash.sum()
                if initial_count > 0:
                    recovered = (unsearched_at_crash & (~still_unsearched)).sum()
                    recovered_pcts.append((recovered / float(initial_count)) * 100.0)
                else:
                    recovered_pcts.append(100.0)
            csrr = float(np.mean(recovered_pcts))

        episode_records.append({
            "seed": seed,
            "coverage_rate": cov,
            "agent_deaths": ep_crashes,
            "avg_battery_level": avg_batt,
            "charging_steps": chg_steps,
            "csrr": csrr,
        })
    return episode_records


# ══════════════════════════════════════════════════════════════════════
# Dashboard Plotting Function (5-Panel)
# ══════════════════════════════════════════════════════════════════════

COLOR_MAP = {
    "RSPO_Vanilla": "#8A2BE2",
    "RSPO_SelfHeal": "#BA55D3",
    "MAPPO_Vanilla": "#1F77B4",
    "MAPPO_SelfHeal": "#6BAED6",
    "QMIX_Vanilla": "#FF7F0E",
    "QMIX_SelfHeal": "#FDBE85",
    "MAA2C_Vanilla": "#17BECF",
    "MAA2C_SelfHeal": "#9EDAE5",
    "I-DQN_Vanilla": "#2CA02C",
    "I-DQN_SelfHeal": "#A1D99B",
    "COMA_Vanilla": "#D62728",
    "COMA_SelfHeal": "#FF9896",
}

def generate_5panel_dashboard(df_summary, title, save_path):
    bar_colors = [COLOR_MAP.get(alg, "#333333") for alg in df_summary["Algorithm"]]

    fig, axs = plt.subplots(2, 3, figsize=(22, 11))

    # Panel 1: Coverage Rate (%)
    axs[0, 0].bar(df_summary["Algorithm"], df_summary["Coverage Rate Mean (%)"], yerr=df_summary["Coverage 95% CI (±%)"], capsize=3, color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[0, 0].set_title("1. Mean Area Coverage Rate (%) [Higher is Better]", fontweight='bold', fontsize=10)
    axs[0, 0].set_ylim(0, 100)
    axs[0, 0].tick_params(axis='x', rotation=40, labelsize=7)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in axs[0, 0].patches:
        h = bar.get_height()
        if h > 0:
            axs[0, 0].text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Panel 2: Agent Attrition Rate (Crashes/Ep)
    axs[0, 1].bar(df_summary["Algorithm"], df_summary["Agent Attrition Rate (Crashes/Ep)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[0, 1].set_title("2. Agent Attrition Rate (Crashes / Ep) [Lower is Better]", fontweight='bold', fontsize=10)
    axs[0, 1].set_ylim(0, 4.5)
    axs[0, 1].tick_params(axis='x', rotation=40, labelsize=7)
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in axs[0, 1].patches:
        h = bar.get_height()
        axs[0, 1].text(bar.get_x() + bar.get_width()/2., h + 0.05, f'{h:.2f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Panel 3: Fleet Battery Health Mean (%)
    axs[0, 2].bar(df_summary["Algorithm"], df_summary["Fleet Battery Level Mean (%)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[0, 2].set_title("3. Fleet Battery Health Mean (%) [Higher is Better]", fontweight='bold', fontsize=10)
    axs[0, 2].tick_params(axis='x', rotation=40, labelsize=7)
    axs[0, 2].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in axs[0, 2].patches:
        h = bar.get_height()
        if h > 0:
            axs[0, 2].text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Panel 4: Resilience Index (RI %)
    axs[1, 0].bar(df_summary["Algorithm"], df_summary["Resilience Index (RI %)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[1, 0].set_title("4. Resilience Index (RI %) [Fault Retained Capacity]", fontweight='bold', fontsize=10)
    axs[1, 0].set_ylim(0, 115)
    axs[1, 0].tick_params(axis='x', rotation=40, labelsize=7)
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in axs[1, 0].patches:
        h = bar.get_height()
        if h > 0:
            axs[1, 0].text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Panel 5: Crash-Sector Recovery Rate (CSRR %)
    axs[1, 1].bar(df_summary["Algorithm"], df_summary["Crash-Sector Recovery Rate (CSRR %)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[1, 1].set_title("5. Crash-Sector Recovery Rate (CSRR %) [Gap-Filling]", fontweight='bold', fontsize=10)
    axs[1, 1].set_ylim(0, 115)
    axs[1, 1].tick_params(axis='x', rotation=40, labelsize=7)
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    for bar in axs[1, 1].patches:
        h = bar.get_height()
        if h > 0:
            axs[1, 1].text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')

    # Panel 6: Empty — hide it
    axs[1, 2].axis('off')

    plt.suptitle(title, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Saved: {save_path}")


# ══════════════════════════════════════════════════════════════════════
# Suite Runner
# ══════════════════════════════════════════════════════════════════════

def run_suite_and_generate_dashboard(suite_name, dashboard_title, save_path,
                                     vanilla_env_fn, selfheal_env_fn,
                                     rllib_checkpoints, epymarl_checkpoints,
                                     zero_fault_means=None, free_cells=625, num_seeds=100):
    print(f"\n  =================== {suite_name} ===================")
    summary_rows = []

    for model_key, meta in rllib_checkpoints.items():
        print(f"    Evaluating RLlib: {model_key}...")
        env_fn = vanilla_env_fn if "Vanilla" in model_key else selfheal_env_fn
        algo = load_rllib_policy(meta["model_cls"], meta["model_name"], meta["path"])
        records = evaluate_rllib_model(algo, env_fn, num_seeds=num_seeds, seed_start=1000)
        algo.stop()
        gc.collect()

        covs = [r["coverage_rate"] * 100 for r in records]
        deaths = [r["agent_deaths"] for r in records]
        rewards = [r["coverage_rate"] * free_cells * 5.0 for r in records]
        batts = [r["avg_battery_level"] for r in records]
        chg_steps = [r["charging_steps"] for r in records]
        csrr_list = [r["csrr"] for r in records]

        cov_mean, cov_std = np.mean(covs), np.std(covs)
        ci95 = 1.96 * (cov_std / np.sqrt(len(covs)))

        zero_cov = zero_fault_means.get(model_key, cov_mean) if zero_fault_means else cov_mean
        ri_pct = (cov_mean / zero_cov * 100.0) if zero_cov > 0 else 0.0

        summary_rows.append({
            "Algorithm": model_key, "Framework": "RLlib",
            "Coverage Rate Mean (%)": round(cov_mean, 2),
            "Coverage Std (%)": round(cov_std, 2),
            "Coverage 95% CI (±%)": round(ci95, 2),
            "Agent Attrition Rate (Crashes/Ep)": round(np.mean(deaths), 2),
            "Swarm Survival Rate (%)": round((1 - np.mean(deaths)/4.0)*100, 1),
            "Cumulative Return Mean": round(np.mean(rewards), 2),
            "Fleet Battery Level Mean (%)": round(np.mean(batts), 2),
            "Mean Charging Steps": round(np.mean(chg_steps), 1),
            "Resilience Index (RI %)": round(ri_pct, 2),
            "Crash-Sector Recovery Rate (CSRR %)": round(np.mean(csrr_list), 2),
        })

    for model_key, path in epymarl_checkpoints.items():
        print(f"    Evaluating EPyMARL: {model_key}...")
        if not os.path.isfile(path):
            print(f"      Warning: Checkpoint not found at {path}. Skipping...")
            continue
        env_fn = vanilla_env_fn if "Vanilla" in model_key else selfheal_env_fn
        records = evaluate_epymarl_model(path, env_fn, num_seeds=num_seeds, seed_start=1000)
        gc.collect()

        covs = [r["coverage_rate"] * 100 for r in records]
        deaths = [r["agent_deaths"] for r in records]
        rewards = [r["coverage_rate"] * free_cells * 5.0 for r in records]
        batts = [r["avg_battery_level"] for r in records]
        chg_steps = [r["charging_steps"] for r in records]
        csrr_list = [r["csrr"] for r in records]

        cov_mean, cov_std = np.mean(covs), np.std(covs)
        ci95 = 1.96 * (cov_std / np.sqrt(len(covs)))

        zero_cov = zero_fault_means.get(model_key, cov_mean) if zero_fault_means else cov_mean
        ri_pct = (cov_mean / zero_cov * 100.0) if zero_cov > 0 else 0.0

        summary_rows.append({
            "Algorithm": model_key, "Framework": "EPyMARL",
            "Coverage Rate Mean (%)": round(cov_mean, 2),
            "Coverage Std (%)": round(cov_std, 2),
            "Coverage 95% CI (±%)": round(ci95, 2),
            "Agent Attrition Rate (Crashes/Ep)": round(np.mean(deaths), 2),
            "Swarm Survival Rate (%)": round((1 - np.mean(deaths)/4.0)*100, 1),
            "Cumulative Return Mean": round(np.mean(rewards), 2),
            "Fleet Battery Level Mean (%)": round(np.mean(batts), 2),
            "Mean Charging Steps": round(np.mean(chg_steps), 1),
            "Resilience Index (RI %)": round(ri_pct, 2),
            "Crash-Sector Recovery Rate (CSRR %)": round(np.mean(csrr_list), 2),
        })

    df_summary = pd.DataFrame(summary_rows)
    generate_5panel_dashboard(df_summary, dashboard_title, save_path)
    return df_summary


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    # ── Output Directories ──
    RESULTS_DIR = os.path.expanduser("~/Desktop/Results")
    NO_FAULTS_DIR = os.path.join(RESULTS_DIR, "No_Faults_0.0")
    ACTIVE_FAULTS_DIR = os.path.join(RESULTS_DIR, "Active_Faults_0.0005")
    os.makedirs(NO_FAULTS_DIR, exist_ok=True)
    os.makedirs(ACTIVE_FAULTS_DIR, exist_ok=True)

    NUM_SEEDS = 100

    # ── Model Checkpoint Paths ──
    rllib_checkpoints = {
        "RSPO_Vanilla": {
            "model_name": "RSPOModelV2_Vanilla", "model_cls": RSPOModelV2,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_Vanilla_RSPO_v2/RSPOPPO_DSSE_Coverage_RSPO_V2_Vanilla_45eb7_00000_0_2026-09-04_06-38-43/checkpoint_000139"),
        },
        "RSPO_SelfHeal": {
            "model_name": "RSPOModelV2_SelfHeal", "model_cls": RSPOModelV2,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_V2_SelfHeal_rspo_v2_selfheal/RSPOPPO_DSSE_Coverage_RSPO_V2_SelfHeal_86fd5_00000_0_2026-09-04_06-40-32/checkpoint_000230"),
        },
        "MAPPO_Vanilla": {
            "model_name": "MAPPOModelVanilla", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_vanilla_proper/PPO_DSSE_Coverage_cc7e1_00000_0_2026-07-05_03-37-08/checkpoint_000243"),
        },
        "MAPPO_SelfHeal": {
            "model_name": "MAPPOModelSelfHeal", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_selfheal_final_v2/checkpoints/checkpoint_iter_980_ts_10002432.pt"),
        },
        "I-DQN_Vanilla": {
            "model_name": "IDQNModelVanilla", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_vanilla_I-DQN_vanilla_proper/DQN_DSSE_Coverage_61018_00000_0_2026-07-16_09-47-37/checkpoint_000609"),
        },
        "I-DQN_SelfHeal": {
            "model_name": "IDQNModelSelfHeal", "model_cls": MAPPOModelVanilla,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_I-DQN_selfheal_proper_pt2/DQN_DSSE_Coverage_8aefa_00000_0_2026-07-16_05-38-15/checkpoint_000565"),
        },
    }

    epymarl_checkpoints = {
        "QMIX_Vanilla": os.path.join(SRC_DIR, "results/models/qmix_true_vanilla_v1_seed533532401_dsse_coverage_2026-07-18 02:06:54.683275/17040000/agent.th"),
        "QMIX_SelfHeal": os.path.join(SRC_DIR, "results/models/qmix_selfheal_v2_seed533532401_dsse_coverage_2026-07-17 03:23:47.879210/19044000/agent.th"),
        "MAA2C_Vanilla": os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_vanilla_seed782869923_dsse_coverage_2026-06-16 18:45:27.715438/18795000/agent.th"),
        "MAA2C_SelfHeal": os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_original_seed776978465_dsse_coverage_2026-07-15 06:30:31.971452/12675000/agent.th"),
        "COMA_Vanilla": os.path.join(SRC_DIR, "results/models/coma_vanilla_proper_seed452926189_dsse_coverage_2026-07-10 01:59:53.354626/19338000/agent.th"),
        "COMA_SelfHeal": os.path.join(SRC_DIR, "results/models/coma_selfhealing_final_stable_seed339532096_dsse_coverage_2026-07-06 10:52:47.933607/19986000/agent.th"),
    }

    # Free cells per regime
    FREE_CELLS_25 = 625
    FREE_CELLS_25_OBS = 609   # 625 - 16 obstacles
    FREE_CELLS_50 = 2500
    FREE_CELLS_50_OBS = 2404  # 2500 - 96 obstacles

    # ══════════════════════════════════════════════════════════════════
    # SUITE 1: No_Faults_0.0 (fault_prob = 0.0)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  SUITE 1: No_Faults_0.0 (fault_prob = 0.0)")
    print("="*80)

    # Dashboard 1: 25x25 Standard (No Faults)
    df_nf_25 = run_suite_and_generate_dashboard(
        "No Faults — 25×25 Standard Grid",
        "25×25 Standard Grid (fault_prob = 0.0, T=750, N=100 seeds)",
        os.path.join(NO_FAULTS_DIR, "1_dashboard_25x25_standard.png"),
        make_env_creator(25, 0.0, False, False),
        make_env_creator(25, 0.0, True, False),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=None, free_cells=FREE_CELLS_25, num_seeds=NUM_SEEDS
    )

    # Use the 25x25 no-fault baseline for RI calculation across all No_Faults dashboards
    nf_zero_fault_means = dict(zip(df_nf_25["Algorithm"], df_nf_25["Coverage Rate Mean (%)"]))

    # Dashboard 2: 25x25 Obstacles (No Faults)
    df_nf_25obs = run_suite_and_generate_dashboard(
        "No Faults — 25×25 Obstacle Airspace",
        "25×25 Obstacle Airspace (fault_prob = 0.0, Layout 3 Skyline, N=100 seeds)",
        os.path.join(NO_FAULTS_DIR, "2_dashboard_25x25_obstacles.png"),
        make_env_creator(25, 0.0, False, True),
        make_env_creator(25, 0.0, True, True),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=nf_zero_fault_means, free_cells=FREE_CELLS_25_OBS, num_seeds=NUM_SEEDS
    )

    # Dashboard 3: 50x50 Spatial Scaling (No Faults)
    df_nf_50 = run_suite_and_generate_dashboard(
        "No Faults — 50×50 Spatial Scaling",
        "50×50 Zero-Shot Spatial Scaling (fault_prob = 0.0, T=1500, battery=250, N=100 seeds)",
        os.path.join(NO_FAULTS_DIR, "3_dashboard_50x50_spatial_scaling.png"),
        make_env_creator(50, 0.0, False, False),
        make_env_creator(50, 0.0, True, False),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=nf_zero_fault_means, free_cells=FREE_CELLS_50, num_seeds=NUM_SEEDS
    )

    # Dashboard 4: 50x50 Obstacles (No Faults)
    df_nf_50obs = run_suite_and_generate_dashboard(
        "No Faults — 50×50 Obstacle Airspace",
        "50×50 Obstacle Airspace (fault_prob = 0.0, T=1500, battery=250, N=100 seeds)",
        os.path.join(NO_FAULTS_DIR, "4_dashboard_50x50_obstacles.png"),
        make_env_creator(50, 0.0, False, True),
        make_env_creator(50, 0.0, True, True),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=nf_zero_fault_means, free_cells=FREE_CELLS_50_OBS, num_seeds=NUM_SEEDS
    )

    # Combine No-Faults CSV
    df_nf_25["Regimen"] = "25x25 Standard"
    df_nf_25obs["Regimen"] = "25x25 Obstacles"
    df_nf_50["Regimen"] = "50x50 Spatial Scaling"
    df_nf_50obs["Regimen"] = "50x50 Obstacles"
    df_nf_master = pd.concat([df_nf_25, df_nf_25obs, df_nf_50, df_nf_50obs], ignore_index=True)
    nf_csv_path = os.path.join(NO_FAULTS_DIR, "no_faults_summary.csv")
    df_nf_master.to_csv(nf_csv_path, index=False)
    print(f"\n  -> No-Faults Summary CSV saved: {nf_csv_path}")

    # ══════════════════════════════════════════════════════════════════
    # SUITE 2: Active_Faults_0.0005 (fault_prob = 0.0005)
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  SUITE 2: Active_Faults_0.0005 (fault_prob = 0.0005)")
    print("="*80)

    # Use the same no-fault 25x25 baseline for RI calculation
    af_zero_fault_means = nf_zero_fault_means

    # Dashboard 1: 25x25 Standard (Active Faults)
    df_af_25 = run_suite_and_generate_dashboard(
        "Active Faults — 25×25 Standard Grid",
        "25×25 Standard Grid (fault_prob = 0.0005, T=750, N=100 seeds)",
        os.path.join(ACTIVE_FAULTS_DIR, "1_dashboard_25x25_standard.png"),
        make_env_creator(25, 0.0005, False, False),
        make_env_creator(25, 0.0005, True, False),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=af_zero_fault_means, free_cells=FREE_CELLS_25, num_seeds=NUM_SEEDS
    )

    # Dashboard 2: 25x25 Obstacles (Active Faults)
    df_af_25obs = run_suite_and_generate_dashboard(
        "Active Faults — 25×25 Obstacle Airspace",
        "25×25 Obstacle Airspace (fault_prob = 0.0005, Layout 3 Skyline, N=100 seeds)",
        os.path.join(ACTIVE_FAULTS_DIR, "2_dashboard_25x25_obstacles.png"),
        make_env_creator(25, 0.0005, False, True),
        make_env_creator(25, 0.0005, True, True),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=af_zero_fault_means, free_cells=FREE_CELLS_25_OBS, num_seeds=NUM_SEEDS
    )

    # Dashboard 3: 50x50 Spatial Scaling (Active Faults)
    df_af_50 = run_suite_and_generate_dashboard(
        "Active Faults — 50×50 Spatial Scaling",
        "50×50 Zero-Shot Spatial Scaling (fault_prob = 0.0005, T=1500, battery=250, N=100 seeds)",
        os.path.join(ACTIVE_FAULTS_DIR, "3_dashboard_50x50_spatial_scaling.png"),
        make_env_creator(50, 0.0005, False, False),
        make_env_creator(50, 0.0005, True, False),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=af_zero_fault_means, free_cells=FREE_CELLS_50, num_seeds=NUM_SEEDS
    )

    # Dashboard 4: 50x50 Obstacles (Active Faults)
    df_af_50obs = run_suite_and_generate_dashboard(
        "Active Faults — 50×50 Obstacle Airspace",
        "50×50 Obstacle Airspace (fault_prob = 0.0005, T=1500, battery=250, N=100 seeds)",
        os.path.join(ACTIVE_FAULTS_DIR, "4_dashboard_50x50_obstacles.png"),
        make_env_creator(50, 0.0005, False, True),
        make_env_creator(50, 0.0005, True, True),
        rllib_checkpoints, epymarl_checkpoints,
        zero_fault_means=af_zero_fault_means, free_cells=FREE_CELLS_50_OBS, num_seeds=NUM_SEEDS
    )

    # Combine Active-Faults CSV
    df_af_25["Regimen"] = "25x25 Standard"
    df_af_25obs["Regimen"] = "25x25 Obstacles"
    df_af_50["Regimen"] = "50x50 Spatial Scaling"
    df_af_50obs["Regimen"] = "50x50 Obstacles"
    df_af_master = pd.concat([df_af_25, df_af_25obs, df_af_50, df_af_50obs], ignore_index=True)
    af_csv_path = os.path.join(ACTIVE_FAULTS_DIR, "active_faults_summary.csv")
    df_af_master.to_csv(af_csv_path, index=False)
    print(f"\n  -> Active-Faults Summary CSV saved: {af_csv_path}")

    # ══════════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("  ALL 8 DASHBOARDS GENERATED SUCCESSFULLY")
    print("="*80)
    print(f"\n  No_Faults_0.0 ({NO_FAULTS_DIR}):")
    print("    1. 1_dashboard_25x25_standard.png")
    print("    2. 2_dashboard_25x25_obstacles.png")
    print("    3. 3_dashboard_50x50_spatial_scaling.png")
    print("    4. 4_dashboard_50x50_obstacles.png")
    print("    5. no_faults_summary.csv")
    print(f"\n  Active_Faults_0.0005 ({ACTIVE_FAULTS_DIR}):")
    print("    1. 1_dashboard_25x25_standard.png")
    print("    2. 2_dashboard_25x25_obstacles.png")
    print("    3. 3_dashboard_50x50_spatial_scaling.png")
    print("    4. 4_dashboard_50x50_obstacles.png")
    print("    5. active_faults_summary.csv")

    ray.shutdown()
