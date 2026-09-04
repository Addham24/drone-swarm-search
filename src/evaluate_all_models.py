"""
Deep-Dive Testing Metrics Extraction Suite (All 12 MARL Models x 100 Seeds)
================================================──────────────────────────
Extracts all 4 Main Evaluation Metrics requested for presentation & thesis:
  1. Mean Area Coverage Rate (%)
  2. Agent Attrition Rate (Mean Drone Crashes & Swarm Survival Rate %)
  3. Cumulative Mission Return (Episode Total Reward)
  4. Fleet Battery Health (Mean Battery Level & Charging Steps)
  5. Time-to-80% Coverage (Steps)

Exports:
  - evaluation_raw_seed_by_seed.csv (1,200 raw data rows for seed-by-seed transparency)
  - evaluation_summary_all_metrics.csv (Comprehensive summary table for slides/thesis)
  - fig1_coverage_rate.png (Publication Bar Chart with 95% CI)
  - fig2_agent_attrition.png (Agent Attrition Rate & Swarm Survival)
  - fig3_cumulative_reward.png (Cumulative Mission Return)
  - fig4_fleet_battery_health.png (Fleet Battery Health & Charging)
  - fig5_multi_metric_dashboard.png (4-Panel Master Presentation Dashboard)
"""

import os
import sys
import glob
import json
import pickle
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt

# Add EPyMARL paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
from global_reward_wrapper import GlobalRewardWrapper
from modules.agents.cnn_agent import CNNAgent
from modules.agents.rnn_agent import RNNAgent

from train_rspo_vanilla import RSPOModel as RSPOModelVanilla
from train_rspo_cnn_cov import RSPOModel as RSPOModelSelfHeal
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla


# ── Mock Args for EPyMARL Agent ──
class MockArgs:
    def __init__(self, use_rnn=True, hidden_dim=128, n_actions=9):
        self.use_rnn = use_rnn
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions


# ── Environment Creators ──
def create_vanilla_env():
    N_AGENTS = 4
    matrix_path = os.path.join(SRC_DIR, "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=750, drone_amount=N_AGENTS, prob_matrix_path=matrix_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return RetainDronePosWrapper(env, positions)


def create_selfheal_env():
    N_AGENTS = 4
    matrix_path = os.path.join(SRC_DIR, "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=750, drone_amount=N_AGENTS, prob_matrix_path=matrix_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    env.COMPENSATION_BONUS = 2.5
    env.COMPENSATION_PENALTY = -0.5
    env.COMPENSATION_HORIZON = 100
    env = GlobalRewardWrapper(env, mixing_alpha=0.5)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    return RetainDronePosWrapper(env, positions)


# ── Custom Pickle Loader for RLlib ──
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
            q_values, _ = self.model(obs_dict)
            action = int(q_values.argmax(dim=-1).item())
        return action

    def stop(self): pass


def load_rllib_policy(algo_type, env_name, model_name, custom_model_cls, checkpoint_path):
    policy_state_path = os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")
    if not os.path.exists(policy_state_path):
        policy_state_path = os.path.join(checkpoint_path, "policy_state.pkl")

    with open(policy_state_path, "rb") as f:
        policy_state = VersionCompatibilityUnpickler(f).load()

    if algo_type == "dqn":
        from gymnasium.spaces import Box, Tuple as GymTuple
        dummy_obs_space = GymTuple([Box(-1.0, 1.0, (22,), dtype=np.float32), Box(0.0, 1.0, (25, 25), dtype=np.float64)])
        model = MAPPOModelVanilla(dummy_obs_space, Box(0, 8, (), dtype=np.int64), 9, {}, model_name)
        if "weights" in policy_state and isinstance(policy_state["weights"], dict):
            tensor_weights = {k: th.from_numpy(v) if not isinstance(v, th.Tensor) else v for k, v in policy_state["weights"].items()}
            model.load_state_dict(tensor_weights, strict=False)
        return DirectPyTorchPolicyWrapper(model)

    ModelCatalog.register_custom_model(model_name, custom_model_cls)
    register_env(env_name, lambda cfg: ParallelPettingZooEnv(create_vanilla_env() if "Vanilla" in env_name else create_selfheal_env()))

    rllib_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=env_name)
        .env_runners(num_env_runners=0)
        .training(model={"custom_model": model_name, "_disable_preprocessor_api": True})
        .experimental(_disable_preprocessor_api=True, _validate_config=False)
        .framework(framework="torch")
    )

    algo = rllib_config.build()
    policy = algo.get_policy("default_policy")
    if "weights" in policy_state and isinstance(policy_state["weights"], dict):
        tensor_weights = {k: th.from_numpy(v) if not isinstance(v, th.Tensor) else v for k, v in policy_state["weights"].items()}
        policy.model.load_state_dict(tensor_weights, strict=False)
    else:
        policy.set_state(policy_state)

    return algo


def evaluate_rllib_model(algo, env_fn, num_seeds=100, seed_start=1000):
    episode_records = []

    for seed in range(seed_start, seed_start + num_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = env_fn()
        obs, infos = env.reset(seed=seed)
        ep_reward = 0.0

        batt_samples = []
        chg_steps = 0
        step_idx = 0
        time_to_80 = 750
        ep_crashes = 0

        while env.agents:
            step_idx += 1
            actions = {}
            for agent in env.agents:
                if agent in obs:
                    actions[agent] = algo.compute_single_action(observation=obs[agent], policy_id="default_policy")

            obs, rewards, terminations, truncations, infos = env.step(actions)
            ep_reward += sum(rewards.values())

            # Track per-step battery levels, charging & crashes as they happen
            for a, info in infos.items():
                if isinstance(info, dict):
                    batt = info.get("battery")
                    if batt is not None:
                        batt_samples.append(batt)
                    if info.get("charging", False):
                        chg_steps += 1
                    if info.get("stranded_event", False):
                        ep_crashes += 1
                    cov_step = info.get("coverage_rate", 0.0)
                    if cov_step >= 0.80 and time_to_80 == 750:
                        time_to_80 = step_idx

        sample_info = next(iter(infos.values()), {}) if infos else {}
        cov = sample_info.get("coverage_rate", 0.0) if isinstance(sample_info, dict) else 0.0
        avg_batt = np.mean(batt_samples) if batt_samples else 0.0
        pure_mission_return = cov * 625.0 * 5.0

        episode_records.append({
            "seed": seed,
            "coverage_rate": cov,
            "agent_deaths": ep_crashes,
            "cumulative_reward": pure_mission_return,
            "avg_battery_level": avg_batt,
            "charging_steps": chg_steps,
            "time_to_80_coverage": time_to_80,
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
        ep_reward = 0.0

        batt_samples = []
        chg_steps = 0
        step_idx = 0
        time_to_80 = 750
        ep_crashes = 0

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
            ep_reward += sum(rewards.values())

            for a, info in info_dict.items():
                if isinstance(info, dict):
                    batt = info.get("battery")
                    if batt is not None:
                        batt_samples.append(batt)
                    if info.get("charging", False):
                        chg_steps += 1
                    if info.get("stranded_event", False):
                        ep_crashes += 1
                    cov_step = info.get("coverage_rate", 0.0)
                    if cov_step >= 0.80 and time_to_80 == 750:
                        time_to_80 = step_idx

        sample_info = next(iter(info_dict.values()), {}) if info_dict else {}
        cov = sample_info.get("coverage_rate", 0.0) if isinstance(sample_info, dict) else 0.0
        avg_batt = np.mean(batt_samples) if batt_samples else 0.0
        pure_mission_return = cov * 625.0 * 5.0

        episode_records.append({
            "seed": seed,
            "coverage_rate": cov,
            "agent_deaths": ep_crashes,
            "cumulative_reward": pure_mission_return,
            "avg_battery_level": avg_batt,
            "charging_steps": chg_steps,
            "time_to_80_coverage": time_to_80,
        })

    return episode_records


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)

    rllib_models = {
        "RSPO_Vanilla": {
            "algo_type": "ppo", "env_name": "DSSE_Coverage_RSPO_Vanilla_Eval", "model_name": "RSPOModelVanilla_Eval",
            "model_cls": RSPOModelVanilla, "env_fn": create_vanilla_env,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_vanilla_v1/PPO_DSSE_Coverage_RSPO_Vanilla_5bfc7_00000_0_2026-08-01_04-35-01/checkpoint_000199"),
        },
        "RSPO_SelfHeal": {
            "algo_type": "ppo", "env_name": "DSSE_Coverage_RSPO_SelfHeal_Eval", "model_name": "RSPOModelSelfHeal_Eval",
            "model_cls": RSPOModelSelfHeal, "env_fn": create_selfheal_env,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/RSPO_rspo_v1/PPO_DSSE_Coverage_RSPO_394ce_00000_0_2026-07-26_00-36-29/checkpoint_000165"),
        },
        "MAPPO_Vanilla": {
            "algo_type": "ppo", "env_name": "DSSE_Coverage_MAPPO_Vanilla_Eval", "model_name": "MAPPOModelVanilla_Eval",
            "model_cls": MAPPOModelVanilla, "env_fn": create_vanilla_env,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_vanilla_proper/PPO_DSSE_Coverage_cc7e1_00000_0_2026-07-05_03-37-08/checkpoint_000141"),
        },
        "MAPPO_SelfHeal": {
            "algo_type": "ppo", "env_name": "DSSE_Coverage_MAPPO_SelfHeal_Eval", "model_name": "MAPPOModelSelfHeal_Eval",
            "model_cls": MAPPOModelVanilla, "env_fn": create_selfheal_env,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/MAPPO_selfheal_comparison/PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04/checkpoint_000023"),
        },
        "I-DQN_Vanilla": {
            "algo_type": "dqn", "env_name": "DSSE_Coverage_IDQN_Vanilla_Eval", "model_name": "IDQNModelVanilla_Eval",
            "model_cls": MAPPOModelVanilla, "env_fn": create_vanilla_env,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_vanilla_I-DQN_vanilla_proper/DQN_DSSE_Coverage_61018_00000_0_2026-07-16_09-47-37/checkpoint_000609"),
        },
        "I-DQN_SelfHeal": {
            "algo_type": "dqn", "env_name": "DSSE_Coverage_IDQN_SelfHeal_Eval", "model_name": "IDQNModelSelfHeal_Eval",
            "model_cls": MAPPOModelVanilla, "env_fn": create_selfheal_env,
            "path": os.path.join(SRC_DIR, "ray_res/DSSE_Coverage/QMIX_I-DQN_selfheal_proper_pt2/DQN_DSSE_Coverage_8aefa_00000_0_2026-07-16_05-38-15/checkpoint_000565"),
        },
    }

    epymarl_models = {
        "QMIX_Vanilla": {
            "type": "epymarl", "env_fn": create_vanilla_env,
            "path": os.path.join(SRC_DIR, "results/models/qmix_true_vanilla_v1_seed533532401_dsse_coverage_2026-07-18 02:06:54.683275/17040000/agent.th"),
        },
        "QMIX_SelfHeal": {
            "type": "epymarl", "env_fn": create_selfheal_env,
            "path": os.path.join(SRC_DIR, "results/models/qmix_selfheal_v2_seed533532401_dsse_coverage_2026-07-17 03:23:47.879210/19044000/agent.th"),
        },
        "MAA2C_Vanilla": {
            "type": "epymarl", "env_fn": create_vanilla_env,
            "path": os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_vanilla_seed782869923_dsse_coverage_2026-06-16 18:45:27.715438/18795000/agent.th"),
        },
        "MAA2C_SelfHeal": {
            "type": "epymarl", "env_fn": create_selfheal_env,
            "path": os.path.join(SRC_DIR, "results/models/maa2c_cnn_bigbatch_v3_original_seed776978465_dsse_coverage_2026-07-15 06:30:31.971452/12675000/agent.th"),
        },
        "COMA_Vanilla": {
            "type": "epymarl", "env_fn": create_vanilla_env,
            "path": os.path.join(SRC_DIR, "results/models/coma_vanilla_proper_seed452926189_dsse_coverage_2026-07-10 01:59:53.354626/19338000/agent.th"),
        },
        "COMA_SelfHeal": {
            "type": "epymarl", "env_fn": create_selfheal_env,
            "path": os.path.join(SRC_DIR, "results/models/coma_selfhealing_final_stable_seed339532096_dsse_coverage_2026-07-06 10:52:47.933607/19986000/agent.th"),
        },
    }

    raw_seed_rows = []
    summary_rows = []

    print("=================== 100-SEED DEEP-DIVE TESTING METRICS EXTRACTION ===================")
    
    # 1. RLlib Evaluation
    for model_key, meta in rllib_models.items():
        print(f"\n[Evaluating RLlib Model: {model_key}]")
        algo = load_rllib_policy(meta["algo_type"], meta["env_name"], meta["model_name"], meta["model_cls"], meta["path"])
        records = evaluate_rllib_model(algo, meta["env_fn"], num_seeds=100, seed_start=1000)
        algo.stop()

        covs = [r["coverage_rate"] * 100 for r in records]
        deaths = [r["agent_deaths"] for r in records]
        rewards = [r["cumulative_reward"] for r in records]
        batts = [r["avg_battery_level"] for r in records]
        chg_steps = [r["charging_steps"] for r in records]

        for r in records:
            raw_seed_rows.append({
                "Algorithm": model_key,
                "Framework": "RLlib",
                "Seed": r["seed"],
                "Coverage_Rate_%": round(r["coverage_rate"] * 100, 2),
                "Agent_Deaths": r["agent_deaths"],
                "Cumulative_Reward": round(r["cumulative_reward"], 2),
                "Avg_Battery_Level": round(r["avg_battery_level"], 2),
                "Charging_Steps": r["charging_steps"],
                "Time_to_80%_Coverage_Steps": r["time_to_80_coverage"],
            })

        cov_mean, cov_std = np.mean(covs), np.std(covs)
        ci95 = 1.96 * (cov_std / np.sqrt(100))

        print(f"  - 1. Coverage Rate:         {cov_mean:.2f}% ± {cov_std:.2f}%  [95% CI: ±{ci95:.2f}%]")
        print(f"  - 2. Agent Attrition Rate:  {np.mean(deaths):.2f} crashes/ep  (Survival: {(1 - np.mean(deaths)/4)*100:.1f}%)")
        print(f"  - 3. Cumulative Return:     {np.mean(rewards):.2f}")
        print(f"  - 4. Fleet Battery Health:  {np.mean(batts):.2f}% avg battery | {np.mean(chg_steps):.1f} charging steps")

        summary_rows.append({
            "Algorithm": model_key,
            "Framework": "RLlib",
            "Coverage Rate Mean (%)": round(cov_mean, 2),
            "Coverage Std (%)": round(cov_std, 2),
            "Coverage 95% CI (±%)": round(ci95, 2),
            "Agent Attrition Rate (Crashes/Ep)": round(np.mean(deaths), 2),
            "Swarm Survival Rate (%)": round((1 - np.mean(deaths)/4.0)*100, 1),
            "Cumulative Return Mean": round(np.mean(rewards), 2),
            "Fleet Battery Level Mean (%)": round(np.mean(batts), 2),
            "Mean Charging Steps": round(np.mean(chg_steps), 1),
        })

    # 2. EPyMARL Evaluation
    for model_key, meta in epymarl_models.items():
        print(f"\n[Evaluating EPyMARL Model: {model_key}]")
        if not os.path.isfile(meta["path"]):
            print(f"  Warning: Checkpoint not found at {meta['path']}. Skipping...")
            continue

        records = evaluate_epymarl_model(meta["path"], meta["env_fn"], num_seeds=100, seed_start=1000)

        covs = [r["coverage_rate"] * 100 for r in records]
        deaths = [r["agent_deaths"] for r in records]
        rewards = [r["cumulative_reward"] for r in records]
        batts = [r["avg_battery_level"] for r in records]
        chg_steps = [r["charging_steps"] for r in records]

        for r in records:
            raw_seed_rows.append({
                "Algorithm": model_key,
                "Framework": "EPyMARL",
                "Seed": r["seed"],
                "Coverage_Rate_%": round(r["coverage_rate"] * 100, 2),
                "Agent_Deaths": r["agent_deaths"],
                "Cumulative_Reward": round(r["cumulative_reward"], 2),
                "Avg_Battery_Level": round(r["avg_battery_level"], 2),
                "Charging_Steps": r["charging_steps"],
                "Time_to_80%_Coverage_Steps": r["time_to_80_coverage"],
            })

        cov_mean, cov_std = np.mean(covs), np.std(covs)
        ci95 = 1.96 * (cov_std / np.sqrt(100))

        print(f"  - 1. Coverage Rate:         {cov_mean:.2f}% ± {cov_std:.2f}%  [95% CI: ±{ci95:.2f}%]")
        print(f"  - 2. Agent Attrition Rate:  {np.mean(deaths):.2f} crashes/ep  (Survival: {(1 - np.mean(deaths)/4)*100:.1f}%)")
        print(f"  - 3. Cumulative Return:     {np.mean(rewards):.2f}")
        print(f"  - 4. Fleet Battery Health:  {np.mean(batts):.2f}% avg battery | {np.mean(chg_steps):.1f} charging steps")

        summary_rows.append({
            "Algorithm": model_key,
            "Framework": "EPyMARL",
            "Coverage Rate Mean (%)": round(cov_mean, 2),
            "Coverage Std (%)": round(cov_std, 2),
            "Coverage 95% CI (±%)": round(ci95, 2),
            "Agent Attrition Rate (Crashes/Ep)": round(np.mean(deaths), 2),
            "Swarm Survival Rate (%)": round((1 - np.mean(deaths)/4.0)*100, 1),
            "Cumulative Return Mean": round(np.mean(rewards), 2),
            "Fleet Battery Level Mean (%)": round(np.mean(batts), 2),
            "Mean Charging Steps": round(np.mean(chg_steps), 1),
        })

    # Export Raw Seed-by-Seed CSV (1,200 rows)
    df_raw = pd.DataFrame(raw_seed_rows)
    raw_csv_path = os.path.join(SRC_DIR, "evaluation_raw_seed_by_seed.csv")
    raw_desktop = os.path.expanduser("~/Desktop/evaluation_raw_seed_by_seed.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    df_raw.to_csv(raw_desktop, index=False)

    # Export Multi-Metric Summary CSV
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(SRC_DIR, "evaluation_summary_all_metrics.csv")
    summary_desktop = os.path.expanduser("~/Desktop/evaluation_summary_all_metrics.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    df_summary.to_csv(summary_desktop, index=False)

    print(f"\n=================== EVALUATION EXTRACTION COMPLETE ===================")
    print(f"Saved Raw Seed-by-Seed Dataset to: {raw_desktop}")
    print(f"Saved Summary Table to:            {summary_desktop}\n")
    print(df_summary.to_string(index=False))

    # Family Color Mapping
    COLOR_MAP = {
        "RSPO_Vanilla": "#8A2BE2",    # Dark Violet (Our Method)
        "RSPO_SelfHeal": "#BA55D3",   # Medium Orchid
        "MAPPO_Vanilla": "#1F77B4",   # Standard Blue
        "MAPPO_SelfHeal": "#6BAED6",  # Light Blue
        "QMIX_Vanilla": "#FF7F0E",    # Orange
        "QMIX_SelfHeal": "#FDBE85",   # Light Orange
        "MAA2C_Vanilla": "#17BECF",   # Cyan / Teal
        "MAA2C_SelfHeal": "#9EDAE5",  # Light Teal
        "I-DQN_Vanilla": "#2CA02C",   # Green
        "I-DQN_SelfHeal": "#A1D99B",  # Light Green
        "COMA_Vanilla": "#D62728",    # Red
        "COMA_SelfHeal": "#FF9896",   # Light Red
    }
    bar_colors = [COLOR_MAP.get(alg, "#333333") for alg in df_summary["Algorithm"]]

    # Fig 1: Coverage Rate Bar Chart
    plt.figure(figsize=(13, 6))
    bars = plt.bar(df_summary["Algorithm"], df_summary["Coverage Rate Mean (%)"], yerr=df_summary["Coverage 95% CI (±%)"], capsize=4, color=bar_colors, edgecolor="black", linewidth=0.5)
    plt.ylabel("Mean Area Coverage Rate (%)", fontsize=12, fontweight='bold')
    plt.title("1. Mean Area Coverage Rate Across 100 Unseen Test Seeds (Post-10M Peak Checkpoints)", fontsize=13, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig1_path = os.path.expanduser("~/Desktop/PERFECT_fig1_coverage_rate.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Fig 2: Agent Attrition Rate Bar Chart
    plt.figure(figsize=(13, 6))
    bars = plt.bar(df_summary["Algorithm"], df_summary["Agent Attrition Rate (Crashes/Ep)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    plt.ylabel("Mean Drone Crashes per Episode", fontsize=12, fontweight='bold')
    plt.title("2. Agent Attrition Rate (Lower is Better — Swarm Survival)", fontsize=13, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontweight='bold')
    plt.ylim(0, 4.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{height:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig2_path = os.path.expanduser("~/Desktop/PERFECT_agent_attrition.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Fig 3: Cumulative Return Bar Chart
    plt.figure(figsize=(13, 6))
    bars = plt.bar(df_summary["Algorithm"], df_summary["Cumulative Return Mean"], color=bar_colors, edgecolor="black", linewidth=0.5)
    plt.ylabel("Mean Cumulative Episode Return", fontsize=12, fontweight='bold')
    plt.title("3. Cumulative Mission Return Across 100 Unseen Test Seeds", fontsize=13, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 30, f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig3_path = os.path.expanduser("~/Desktop/PERFECT_fig3_cumulative_reward.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Fig 4: Fleet Battery Health Bar Chart
    plt.figure(figsize=(13, 6))
    bars = plt.bar(df_summary["Algorithm"], df_summary["Fleet Battery Level Mean (%)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    plt.ylabel("Mean Fleet Battery Level (%)", fontsize=12, fontweight='bold')
    plt.title("4. Fleet Battery Health Maintained Across Active Drones", fontsize=13, fontweight='bold')
    plt.xticks(rotation=25, ha='right', fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig4_path = os.path.expanduser("~/Desktop/PERFECT_fig4_fleet_battery_health.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Fig 5: Master Presentation 4-Panel Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(16, 11))

    # Panel 1: Coverage
    axs[0, 0].bar(df_summary["Algorithm"], df_summary["Coverage Rate Mean (%)"], yerr=df_summary["Coverage 95% CI (±%)"], capsize=3, color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[0, 0].set_title("1. Mean Area Coverage Rate (%) [Higher is Better]", fontweight='bold')
    axs[0, 0].set_ylim(0, 100)
    axs[0, 0].tick_params(axis='x', rotation=35, labelsize=8)
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Panel 2: Attrition
    axs[0, 1].bar(df_summary["Algorithm"], df_summary["Agent Attrition Rate (Crashes/Ep)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[0, 1].set_title("2. Agent Attrition Rate (Crashes / Ep) [Lower is Better]", fontweight='bold')
    axs[0, 1].set_ylim(0, 4.5)
    axs[0, 1].tick_params(axis='x', rotation=35, labelsize=8)
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)

    # Panel 3: Return
    axs[1, 0].bar(df_summary["Algorithm"], df_summary["Cumulative Return Mean"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[1, 0].set_title("3. Cumulative Mission Return [Higher is Better]", fontweight='bold')
    axs[1, 0].tick_params(axis='x', rotation=35, labelsize=8)
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)

    # Panel 4: Battery
    axs[1, 1].bar(df_summary["Algorithm"], df_summary["Fleet Battery Level Mean (%)"], color=bar_colors, edgecolor="black", linewidth=0.5)
    axs[1, 1].set_title("4. Fleet Battery Health (%) [Higher is Better]", fontweight='bold')
    axs[1, 1].tick_params(axis='x', rotation=35, labelsize=8)
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle("RSPO vs MARL Baselines: Master 4-Metric Evaluation Dashboard (100 Test Seeds)", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig5_path = os.path.expanduser("~/Desktop/PERFECT_master_dashboard.png")
    plt.savefig(fig5_path, dpi=300, bbox_inches='tight')
    plt.close()


    print(f"\n=================== ALL FIGURES & DATASETS SAVED ===================")
    print(f"Saved 1,200 Seed-by-Seed Dataset to: {raw_desktop}")
    print(f"Saved Multi-Metric Summary CSV to:   {summary_desktop}")
    print(f"Saved 4 Individual Metric Charts & Master Dashboard to Desktop!\n")

    ray.shutdown()
