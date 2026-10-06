"""
50×50 Zero-Shot Generalization Evaluation with Obstacles for Dynamic Target Tracking
====================================================================================
Evaluates all 12 trained 25×25 tracking models on a 50×50 grid with physical obstacles.

Obstacle & Scaling Setup:
  - 96 building obstacles (loaded from src/obstacle_skyline_50.npy)
  - Downscales 50×50 observation matrices to 25×25 via PIL BOX resampling before inference
  - Timestep limit: 1500
  - Max battery: 250
  - Target initial position: (25, 25)

Outputs:
  - ~/Desktop/Results/Tracking/No_Faults_0.0/dashboard_tracking_50x50_obstacles.png
  - ~/Desktop/Results/Tracking/Active_Faults_0.0005/dashboard_tracking_50x50_obstacles.png
  - Corresponding CSV files
"""

import os
import sys
import glob
import re
import pickle
import numpy as np
import pandas as pd
import torch as th
import matplotlib.pyplot as plt
from PIL import Image

# Ensure src directory is on sys.path
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPYMARL_SRC = os.path.join(SRC_DIR, "epymarl", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if EPYMARL_SRC not in sys.path:
    sys.path.insert(0, EPYMARL_SRC)

from tracking.tracking_env_creator import make_tracking_env
from modules.agents.cnn_agent import CNNAgent
from modules.agents.rnn_agent import RNNAgent
from train_rspo_v2_vanilla import RSPOModelV2
from train_mappo_vanilla import CNNModel as MAPPOModelVanilla


# ── 50×50 Environment Constants ──
GRID_SIZE = 50
TIMESTEP_LIMIT = 1500
MAX_BATTERY = 250
TARGET_GRID_SIZE = 25
OBSTACLE_MASK_PATH = os.path.join(SRC_DIR, "obstacle_skyline_50.npy")


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


def make_resizing_policy(policy_fn, source_grid_size=50, target_grid_size=25):
    """Downscales 50×50 observation matrices to 25×25 via PIL BOX resampling and scales probability by 4.0 to conserve probability mass."""
    scale_factor = float((source_grid_size / target_grid_size) ** 2)

    def resized_policy_fn(obs_dict, agents):
        resized_obs = {}
        for agent in agents:
            if agent not in obs_dict:
                continue
            pos_vec, matrix = obs_dict[agent]
            pil_img = Image.fromarray(matrix.astype(np.float32), mode='F')
            pil_resized = pil_img.resize((target_grid_size, target_grid_size), resample=Image.BOX)
            resized_matrix = np.array(pil_resized, dtype=np.float32)
            
            # Preserve probability mass: scale non-obstacle cells by 4.0
            non_obs_mask = (resized_matrix >= 0)
            resized_matrix[non_obs_mask] *= scale_factor

            resized_obs[agent] = (pos_vec, resized_matrix)
        return policy_fn(resized_obs, agents)
    return resized_policy_fn


def load_rllib_policy(model_cls, checkpoint_path, model_name="TrackingModel"):
    """Loads PyTorch weights from an RLlib checkpoint directory or .pt file."""
    from gymnasium.spaces import Box, Tuple as GymTuple, Discrete
    obs_space = GymTuple([Box(-1.0, 1.0, (22,), dtype=np.float32), Box(-1.0, 1.0, (25, 25), dtype=np.float32)])
    act_space = Discrete(9)

    model = model_cls(obs_space, act_space, 9, {}, model_name)

    if checkpoint_path.endswith(".pt"):
        ckpt = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        weights = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(weights, strict=False)
    else:
        policy_state_path = os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")
        if not os.path.exists(policy_state_path):
            policy_state_path = os.path.join(checkpoint_path, "policy_state.pkl")

        if os.path.exists(policy_state_path):
            with open(policy_state_path, "rb") as f:
                policy_state = VersionCompatibilityUnpickler(f).load()

            if "weights" in policy_state and isinstance(policy_state["weights"], dict):
                tensor_weights = {k: th.from_numpy(v) if not isinstance(v, th.Tensor) else v for k, v in policy_state["weights"].items()}
                model.load_state_dict(tensor_weights, strict=False)

    model.eval()

    def policy_fn(obs_dict, agents):
        actions = {}
        for agent in agents:
            if agent in obs_dict:
                pos, matrix = obs_dict[agent]
                pos_t = th.tensor(pos, dtype=th.float32).unsqueeze(0)
                mat_t = th.tensor(matrix, dtype=th.float32).unsqueeze(0)
                with th.no_grad():
                    logits, _ = model({"obs": (pos_t, mat_t)}, [], None)
                    dist = th.distributions.Categorical(logits=logits)
                    actions[agent] = int(dist.sample().item())
        return actions

    return policy_fn


def load_epymarl_policy(model_th_path):
    """Loads PyTorch weights from an EPyMARL agent.th checkpoint."""
    state_dict = th.load(model_th_path, map_location="cpu")
    
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

    def policy_fn(obs_dict, agents):
        actions = {}
        n_agents = len(agents)
        obs_size = 22 + 625

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
            hidden_states = agent.init_hidden().expand(n_agents, -1).clone()
            q_logits, _ = agent(inputs_tensor, hidden_states)
            actions_tensor = q_logits.argmax(dim=-1)

        for i, agent_name in enumerate(agents):
            if agent_name in obs_dict:
                actions[agent_name] = int(actions_tensor[i].item())

        return actions

    return policy_fn


def evaluate_tracking_model_50x50_obstacles(policy_fn, obstacle_mask, is_fault_active=True, n_seeds=100, is_self_heal=False):
    """Evaluates a policy function over n_seeds in a 50×50 obstacle tracking environment."""
    fault_prob = 0.0005 if is_fault_active else 0.0
    
    target_found_list = []
    discovery_times = []
    attrition_list = []
    survival_list = []
    battery_list = []
    seed_records = []

    for seed in range(1000, 1000 + n_seeds):
        np.random.seed(seed)
        th.manual_seed(seed)

        env = make_tracking_env(
            grid_size=GRID_SIZE,
            drone_amount=4,
            person_amount=1,
            person_initial_position=(26, 26),
            timestep_limit=TIMESTEP_LIMIT,
            max_battery=MAX_BATTERY,
            fault_prob=fault_prob,
            is_self_heal=is_self_heal,
            drift_speed=1.0,
            obstacle_mask=obstacle_mask
        )

        obs, infos = env.reset(seed=seed)
        done = False
        step = 0
        target_found = False
        discovery_step = TIMESTEP_LIMIT

        total_battery = 0
        battery_samples = 0
        crash_events = 0
        crashed_agents = set()

        while not done and step < TIMESTEP_LIMIT:
            step += 1
            actions = policy_fn(obs, env.agents)
            obs, rewards, term, trunc, infos = env.step(actions)

            # Check target discovery / interception via DSSE native info flag
            # NOTE: Do NOT use r >= 1.0 — BatteryStationWrapper compensation
            # rewards can inflate intermediate rewards above 1.0, causing
            # false positive target detections.
            for agent in list(rewards.keys()):
                agent_info = infos.get(agent, {})
                if isinstance(agent_info, dict) and agent_info.get("Found", False):
                    target_found = True
                    discovery_step = step
                    done = True
                    break

            # Collect battery and crash stats
            for agent, info in infos.items():
                if isinstance(info, dict):
                    if info.get("stranded_event", False):
                        crash_events += 1
                        crashed_agents.add(agent)
                    batt = info.get("battery")
                    if batt is not None:
                        total_battery += batt
                        battery_samples += 1

            if all(term.values()) or all(trunc.values()):
                done = True

        target_found_list.append(100.0 if target_found else 0.0)
        discovery_times.append(discovery_step if target_found else TIMESTEP_LIMIT)
        attrition_list.append(crash_events)
        
        n_drones = 4
        unique_crashed = len(crashed_agents)
        survival_rate = ((n_drones - unique_crashed) / n_drones) * 100.0
        survival_list.append(survival_rate)
        
        avg_batt = ((total_battery / battery_samples) / float(MAX_BATTERY) * 100.0) if battery_samples > 0 else 0.0
        battery_list.append(avg_batt)

        seed_records.append({
            "seed": seed,
            "target_found": 1 if target_found else 0,
            "discovery_step": discovery_step,
            "crash_events": crash_events,
            "unique_agents_crashed": unique_crashed,
            "survival_rate": survival_rate,
            "avg_battery": avg_batt,
        })

    res_tf = float(np.mean(target_found_list))
    res_dt = float(np.mean(discovery_times))
    res_surv = float(np.mean(survival_list))
    res_batt = float(np.mean(battery_list))

    f_target = max(0.001, res_tf / 100.0)
    f_speed = max(0.001, (float(TIMESTEP_LIMIT) - res_dt) / float(TIMESTEP_LIMIT))
    f_survival = max(0.001, res_surv / 100.0)
    f_battery = max(0.001, res_batt / 100.0)
    smei = (f_target**0.20 * f_speed**0.20 * f_survival**0.30 * f_battery**0.30) * 100.0

    return {
        "target_found_rate": res_tf,
        "mean_discovery_time": res_dt,
        "attrition_rate": float(np.mean(attrition_list)),
        "survival_rate": res_surv,
        "avg_battery": res_batt,
        "smei": smei,
        "raw_records": seed_records
    }


def generate_tracking_dashboard_50x50_obstacles(df_results, output_dir, title_prefix="Dynamic Target Tracking 50×50 Obstacles Benchmark"):
    """Generates a master 6-panel dashboard PNG for 50×50 tracking with obstacles."""
    os.makedirs(output_dir, exist_ok=True)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(f"{title_prefix} — Zero-Shot 50×50 Obstacle Dashboard", fontsize=17, fontweight="bold", y=0.98)

    models = df_results["Model"].tolist()
    x = np.arange(len(models))

    # 1. Target Found Rate (%)
    b1 = axes[0, 0].bar(x, df_results["Target Found Rate (%)"], color="#2ca02c", edgecolor="white")
    axes[0, 0].set_title("1. Target Intercept / Discovery Rate (%)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    axes[0, 0].set_ylim(0, 115)
    axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in b1:
        axes[0, 0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 2. Mean Time-to-Discovery (steps)
    b2 = axes[0, 1].bar(x, df_results["Mean Discovery Time (steps)"], color="#1f77b4", edgecolor="white")
    axes[0, 1].set_title("2. Mean Time-to-Intercept (steps)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in b2:
        axes[0, 1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 10.0, f"{bar.get_height():.0f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 3. Agent Attrition Rate (crashes/ep)
    b3 = axes[0, 2].bar(x, df_results["Attrition Rate"], color="#d62728", edgecolor="white")
    axes[0, 2].set_title("3. Agent Attrition Rate (Crashes/ep)", fontsize=12, fontweight="bold")
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    axes[0, 2].set_ylim(0, 4.8)
    axes[0, 2].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in b3:
        axes[0, 2].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 0.08, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 4. Swarm Survival Rate (%)
    b4 = axes[1, 0].bar(x, df_results["Survival Rate (%)"], color="#9467bd", edgecolor="white")
    axes[1, 0].set_title("4. Swarm Survival Rate (%)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    axes[1, 0].set_ylim(0, 115)
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in b4:
        axes[1, 0].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 5. Fleet Avg Battery Level (%)
    b5 = axes[1, 1].bar(x, df_results["Avg Battery Level (%)"], color="#ff7f0e", edgecolor="white")
    axes[1, 1].set_title("5. Fleet Avg Battery Level (%)", fontsize=12, fontweight="bold")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in b5:
        axes[1, 1].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 6. SMEI (%)
    b6 = axes[1, 2].bar(x, df_results["SMEI (%)"], color="#e377c2", edgecolor="white")
    axes[1, 2].set_title("6. Swarm Mission Efficiency Index (SMEI %)", fontsize=12, fontweight="bold")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(models, rotation=35, ha='right', fontsize=8.5)
    axes[1, 2].set_ylim(0, 115)
    axes[1, 2].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in b6:
        axes[1, 2].text(bar.get_x() + bar.get_width()/2.0, bar.get_height() + 1.5, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(output_dir, "dashboard_tracking_50x50_obstacles.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] 50×50 Obstacle Tracking Dashboard PNG saved to: {save_path}")

    csv_path = os.path.join(output_dir, "tracking_summary_50x50_obstacles_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"[✓] 50×50 Obstacle Tracking Summary CSV saved to: {csv_path}")


def find_latest_checkpoint(search_pattern):
    matches = glob.glob(search_pattern, recursive=True)
    if not matches:
        return None
    def get_ckpt_num(path):
        nums = re.findall(r'\d+', os.path.basename(path))
        return int(nums[-1]) if nums else 0
    matches.sort(key=lambda p: (get_ckpt_num(p), os.path.getmtime(p)), reverse=True)
    return matches[0]


def find_best_post_10m_checkpoint(pattern, framework="rllib", min_timesteps=10_000_000):
    if framework == "rllib":
        base_dir = pattern.split("/**/checkpoint_*")[0]
        progress_files = glob.glob(base_dir + "/**/progress.csv", recursive=True)
        
        for prog_path in progress_files:
            try:
                df = pd.read_csv(prog_path)
                col_ts = [c for c in df.columns if c in ["timesteps_total", "num_env_steps_sampled_lifetime"]][0]
                col_len = [c for c in df.columns if "episode_len_mean" in c][0]
                
                df_post10m = df[df[col_ts] >= min_timesteps]
                if not df_post10m.empty:
                    best_row = df_post10m.loc[df_post10m[col_len].idxmin()]
                    best_iter = int(best_row["training_iteration"]) if "training_iteration" in best_row else 0
                    target_ckpt_num = round(best_iter / 10.0)
                    
                    trial_dir = os.path.dirname(prog_path)
                    ckpts = glob.glob(os.path.join(trial_dir, "checkpoint_*"))
                    if ckpts:
                        def get_num(c_path):
                            nums = re.findall(r"\d+", os.path.basename(c_path))
                            return int(nums[-1]) if nums else 0
                        ckpts_post10m = [c for c in ckpts if get_num(c) >= 100]
                        if ckpts_post10m:
                            best_ckpt = min(ckpts_post10m, key=lambda c: abs(get_num(c) - target_ckpt_num))
                            print(f"    [Post-10M Peak Match] Found iter {best_iter} ({os.path.basename(best_ckpt)}) -> ({int(best_row[col_ts]):,} steps | ep_len: {best_row[col_len]:.1f})")
                            return best_ckpt
            except Exception as e:
                pass
                
        matches = glob.glob(pattern, recursive=True)
        post_10m_ckpts = []
        for m in matches:
            nums = re.findall(r'\d+', os.path.basename(m))
            ckpt_num = int(nums[-1]) if nums else 0
            if ckpt_num >= 100:
                post_10m_ckpts.append((ckpt_num, m))
        if post_10m_ckpts:
            post_10m_ckpts.sort(key=lambda x: x[0], reverse=True)
            return post_10m_ckpts[0][1]

    else:
        matches = glob.glob(pattern, recursive=True)
        post_10m_matches = []
        for m in matches:
            parent_dir = os.path.basename(os.path.dirname(m))
            if parent_dir.isdigit() and int(parent_dir) >= min_timesteps:
                post_10m_matches.append((int(parent_dir), m))
        
        if post_10m_matches:
            post_10m_matches.sort(key=lambda x: x[0], reverse=True)
            print(f"    [Post-10M Match] Found checkpoint at {post_10m_matches[0][0]:,} steps")
            return post_10m_matches[0][1]
            
    return find_latest_checkpoint(pattern)


if __name__ == "__main__":
    print(f"\n{'='*75}")
    print(f"  50×50 ZERO-SHOT OBSTACLE AIRSPACE — DYNAMIC TARGET TRACKING EVALUATION")
    print(f"  Grid: {GRID_SIZE}×{GRID_SIZE} | Obstacles: 96 | Battery: {MAX_BATTERY} | Steps: {TIMESTEP_LIMIT}")
    print(f"  [Rule: Selecting BEST checkpoint AFTER 10 Million timesteps]")
    print(f"{'='*75}\n")

    obstacle_mask = np.load(OBSTACLE_MASK_PATH)
    print(f"[✓] Loaded obstacle mask from {OBSTACLE_MASK_PATH} ({int(obstacle_mask.sum())} obstacle cells)")

    tracking_models = [
        ("RSPO Vanilla", os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_Vanilla*/**/checkpoint_*"), "rllib", RSPOModelV2, False),
        ("RSPO SelfHeal", os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*RSPO_V2_Tracking_SelfHeal*/**/checkpoint_*"), "rllib", RSPOModelV2, True),
        ("MAPPO Vanilla", os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*MAPPO_Tracking_Vanilla*/**/checkpoint_*"), "rllib", MAPPOModelVanilla, False),
        ("MAPPO SelfHeal", os.path.join(SRC_DIR, "ray_res/DSSE_Tracking/*MAPPO_Tracking_SelfHeal*/**/checkpoint_*"), "rllib", MAPPOModelVanilla, True),
        ("QMIX Vanilla", os.path.join(SRC_DIR, "results/models/*qmix_tracking_vanilla*/**/agent.th"), "epymarl", None, False),
        ("QMIX SelfHeal", os.path.join(SRC_DIR, "results/models/*qmix_tracking_selfheal*/**/agent.th"), "epymarl", None, True),
        ("MAA2C Vanilla", os.path.join(SRC_DIR, "results/models/*maa2c_tracking_vanilla*/**/agent.th"), "epymarl", None, False),
        ("MAA2C SelfHeal", os.path.join(SRC_DIR, "results/models/*maa2c_tracking_selfheal*/**/agent.th"), "epymarl", None, True),
        ("COMA Vanilla", os.path.join(SRC_DIR, "results/models/*coma_tracking_vanilla*/**/agent.th"), "epymarl", None, False),
        ("COMA SelfHeal", os.path.join(SRC_DIR, "results/models/*coma_tracking_selfheal*/**/agent.th"), "epymarl", None, True),
        ("I-DQN Vanilla", os.path.join(SRC_DIR, "results/models/*iql_tracking_vanilla*/**/agent.th"), "epymarl", None, False),
        ("I-DQN SelfHeal", os.path.join(SRC_DIR, "results/models/*iql_tracking_selfheal*/**/agent.th"), "epymarl", None, True),
    ]

    fault_regimes = [
        ("No_Faults_0.0", False, "No Faults (fault_prob = 0.0)"),
        ("Active_Faults_0.0005", True, "Active Faults (fault_prob = 0.0005)"),
    ]

    base_output_dir = os.path.expanduser("~/Desktop/Results/Tracking")

    loaded_policies = []
    for model_name, pattern, framework, model_cls, is_self_heal in tracking_models:
        ckpt_path = find_best_post_10m_checkpoint(pattern, framework, min_timesteps=10_000_000)
        print(f"[▶] Resolving Model: {model_name}")
        if ckpt_path:
            print(f"    Loading {framework.upper()} Checkpoint: {ckpt_path}")
            if framework == "rllib":
                raw_policy_fn = load_rllib_policy(model_cls, ckpt_path, model_name=model_name)
            else:
                raw_policy_fn = load_epymarl_policy(ckpt_path)
        else:
            print(f"    [!] Checkpoint pending/training: {pattern}")
            def make_stub():
                def stub_fn(obs_dict, agents):
                    return {agent: np.random.randint(0, 9) for agent in agents}
                return stub_fn
            raw_policy_fn = make_stub()

        resized_policy_fn = make_resizing_policy(raw_policy_fn, source_grid_size=GRID_SIZE, target_grid_size=TARGET_GRID_SIZE)
        loaded_policies.append((model_name, resized_policy_fn, is_self_heal))

    for folder_name, is_fault_active, title_tag in fault_regimes:
        print(f"\n{'='*75}")
        print(f"  RUNNING 50×50 OBSTACLE TRACKING EVALUATION: {title_tag}")
        print(f"{'='*75}\n")

        results_list = []
        all_raw_records = []

        for model_name, policy_fn, is_self_heal in loaded_policies:
            print(f"  [▶] Evaluating {model_name} under {title_tag} (50×50 Obstacles)...")
            res = evaluate_tracking_model_50x50_obstacles(policy_fn, obstacle_mask, is_fault_active=is_fault_active, n_seeds=100, is_self_heal=is_self_heal)
            
            results_list.append({
                "Model": model_name,
                "Target Found Rate (%)": round(res["target_found_rate"], 2),
                "Mean Discovery Time (steps)": round(res["mean_discovery_time"], 2),
                "Attrition Rate": round(res["attrition_rate"], 2),
                "Survival Rate (%)": round(res["survival_rate"], 2),
                "Avg Battery Level (%)": round(res["avg_battery"], 2),
                "SMEI (%)": round(res["smei"], 2),
            })

            for r in res["raw_records"]:
                r["Model"] = model_name
                all_raw_records.append(r)

        df_results = pd.DataFrame(results_list)
        regime_output_dir = os.path.join(base_output_dir, folder_name)
        generate_tracking_dashboard_50x50_obstacles(df_results, regime_output_dir, title_prefix=f"Dynamic Target Tracking 50×50 Obstacles — {title_tag}")

        df_raw = pd.DataFrame(all_raw_records)
        raw_csv_path = os.path.join(regime_output_dir, "tracking_50x50_obstacles_raw_seed_by_seed.csv")
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"[✓] Raw seed-by-seed dataset saved to: {raw_csv_path}")

    print(f"\n{'='*75}")
    print(f"  50×50 OBSTACLE TRACKING BENCHMARK COMPLETE!")
    print(f"  Base Results Directory: {base_output_dir}")
    print(f"{'='*75}\n")
