import os
import pathlib
from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class BatteryMetricsCallback(DefaultCallbacks):
    """Logs custom battery metrics to TensorBoard via RLlib's custom_metrics."""

    def on_episode_start(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        episode.user_data["deaths"] = 0
        episode.user_data["charge_steps"] = 0
        episode.user_data["total_battery"] = 0
        episode.user_data["battery_samples"] = 0

    def on_episode_step(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info is None or not isinstance(info, dict):
                continue
            if info.get("stranded_event", False):
                episode.user_data["deaths"] += 1
            if info.get("charging", False):
                episode.user_data["charge_steps"] += 1
            batt = info.get("battery")
            if batt is not None:
                episode.user_data["total_battery"] += batt
                episode.user_data["battery_samples"] += 1

    def on_episode_end(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        episode.custom_metrics["battery_deaths"] = episode.user_data["deaths"]
        episode.custom_metrics["battery_charge_steps"] = episode.user_data["charge_steps"]
        samples = episode.user_data["battery_samples"]
        if samples > 0:
            episode.custom_metrics["battery_avg_level"] = (
                episode.user_data["total_battery"] / samples
            )
        else:
            episode.custom_metrics["battery_avg_level"] = 0.0
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info and isinstance(info, dict) and "coverage_rate" in info:
                episode.custom_metrics["coverage_rate"] = info["coverage_rate"]
                break


class CNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        print("OBSSPACE: ", obs_space)
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name, **kw)
        nn.Module.__init__(self)

        def get_flatten_size(grid_size):
            x = (grid_size - 2) // 2
            x = (x - 1) // 2
            return 32 * x * x

        grid_size = obs_space[1].shape[0]
        flatten_size = get_flatten_size(grid_size)
        print(f"Grid size: {grid_size}, Cnn Dense layer input size: {flatten_size}")

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )
        self.linear = nn.Sequential(
            nn.Linear(obs_space[0].shape[0], 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )
        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh(),
        )
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float()
        input_matrix = input_matrix.unsqueeze(1)
        cnn_out = self.cnn(input_matrix)
        linear_out = self.linear(input_positions)
        value_input = torch.cat((cnn_out, linear_out), dim=1)
        value_input = self.join(value_input)
        self._value_out = self.value_fn(value_input)
        return self.policy_fn(value_input), state

    def value_function(self):
        return self._value_out.flatten()


def env_creator(args):
    print("-------------------------- ENV CREATOR --------------------------")
    N_AGENTS = 4
    matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(
        timestep_limit=750,
        drone_amount=N_AGENTS,
        prob_matrix_path=matrix_path,
    )
    env.reward_scheme = {
        "default": -0.1,
        "exceed_timestep": 0.0,
        "search_cell": 5.0,
        "done": 500.0,
        "reward_poc": 0.0,
    }
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("CNNModel", CNNModel)

    num_gpus = 1 if torch.cuda.is_available() else 0
    num_runners = 28 if torch.cuda.is_available() else 6
    print(f"Using {num_gpus} GPUs and {num_runners} environment runners.")

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .env_runners(num_env_runners=num_runners, rollout_fragment_length="auto")
        .training(
            train_batch_size=8192,
            lr=2e-5,
            gamma=0.998,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.05,
            vf_clip_param=100000,
            minibatch_size=300,
            num_sgd_iter=10,
            model={
                "custom_model": "CNNModel",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(BatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )

    curr_path = pathlib.Path().resolve()

    # CHECKPOINT_PATH relative to workspace (pointing to clean original 92% checkpoint)
    CHECKPOINT_PATH = os.path.join(
        curr_path, "ray_res", env_name, "MAPPO_selfheal_comparison",
        "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04", "checkpoint_000023"
    )

    # --- Build algo fresh, then load ONLY the policy weights from checkpoint ---
    print(f"Building fresh PPO algorithm...")
    algo = config.build()

    print(f"Loading policy weights from checkpoint: {CHECKPOINT_PATH}")
    import pickle
    policy_state_path = os.path.join(CHECKPOINT_PATH, "policies", "default_policy", "policy_state.pkl")
    with open(policy_state_path, "rb") as f:
        policy_state = pickle.load(f)

    # Restore ONLY the model weights — skip optimizer state
    policy = algo.get_policy("default_policy")
    policy.set_weights(policy_state["weights"])
    print("Model weights restored on local worker (fresh optimizer).")

    # CRITICAL: Sync restored weights to ALL remote workers
    algo.env_runner_group.sync_weights()
    print("Weights synced to all remote workers via env_runner_group!")

    # Verify restore worked
    sample_weight = list(policy.model.parameters())[0].data.mean().item()
    checkpoint_weight = policy_state["weights"]["cnn.0.weight"].mean()
    print(f"Local weight mean:      {sample_weight:.6f}")
    print(f"Checkpoint weight mean: {checkpoint_weight:.6f}")
    print(f"Match: {abs(sample_weight - checkpoint_weight) < 1e-6}")
    print("Policy weights restored successfully!")

    # --- Set up TensorBoard logging (dual writers) ---
    # Writer 1: Original run directory (port 6007) — continues the same graph line
    original_tb_dir = os.path.expanduser(
        "~/Desktop/dsse_run/ray_res/DSSE_Coverage/"
        "MAPPO_selfheal_comparison/"
        "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04"
    )
    if not os.path.exists(original_tb_dir):
        # Fallback to repo-relative path if running on RunPod
        original_tb_dir = os.path.join(
            curr_path, "ray_res", env_name,
            "MAPPO_selfheal_comparison",
            "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04"
        )
    writer_6007 = SummaryWriter(log_dir=original_tb_dir)
    print(f"TensorBoard writer 1 (port 6007): {original_tb_dir}")

    # Writer 2: src/ray_res directory (port 6008) — shows alongside other runs
    port_6008_tb_dir = os.path.join(
        curr_path, "ray_res", env_name,
        "MAPPO_selfheal_comparison",
        "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04"
    )
    os.makedirs(port_6008_tb_dir, exist_ok=True)
    writer_6008 = SummaryWriter(log_dir=port_6008_tb_dir)
    print(f"TensorBoard writer 2 (port 6008): {port_6008_tb_dir}")

    writers = [writer_6007, writer_6008]

    # The original checkpoint timestep is the start point
    TIMESTEP_OFFSET = 1_974_272
    print(f"Timestep offset: {TIMESTEP_OFFSET:,} (continuing from original run)")

    # --- Set up checkpoint saving ---
    ckpt_dir = os.path.join(
        curr_path, "ray_res", env_name, "MAPPO_selfheal_final_v2", "checkpoints"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training loop ---
    TARGET_TIMESTEPS = 20_000_000  # Total including offset
    CHECKPOINT_FREQ = 10  # Save every 10 iterations
    iteration = 240

    print(f"\nStarting training loop (target: {TARGET_TIMESTEPS:,} total timesteps)...")
    print("=" * 60)

    # Recursive logger for all metrics
    def log_result_to_tb(result_dict, prefix="ray/tune", step=0):
        """Recursively log all scalar values from the result dict."""
        SKIP_KEYS = {
            "config", "hist_stats", "episodes_timesteps_total",
            "experiment_tag", "trial_id", "experiment_id", "hostname",
            "node_ip", "pid", "date", "timestamp", "done",
            "training_iteration", "time_this_iter_s", "time_total_s",
            "episode_media", "connector_metrics", "sampler_perf",
        }
        for key, value in result_dict.items():
            if key in SKIP_KEYS:
                continue
            full_key = f"{prefix}/{key}" if prefix else key
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                for w in writers:
                    w.add_scalar(full_key, value, step)
            elif isinstance(value, dict):
                log_result_to_tb(value, prefix=full_key, step=step)

    while True:
        result = algo.train()
        iteration += 1

        # Raw timesteps from this session + offset from original run
        raw_timesteps = result.get("timesteps_total", result.get("num_env_steps_sampled_lifetime", 0))
        total_timesteps = raw_timesteps + TIMESTEP_OFFSET

        reward_mean = result.get("env_runners", {}).get("episode_reward_mean",
                      result.get("episode_reward_mean", 0))
        coverage = result.get("env_runners", {}).get("custom_metrics", {}).get(
            "coverage_rate_mean", 0)
        battery_deaths = result.get("env_runners", {}).get("custom_metrics", {}).get(
            "battery_deaths_mean", 0)

        # Log ALL metrics with offset timestep so graph continues seamlessly
        log_result_to_tb(result, prefix="ray/tune", step=total_timesteps)
        for w in writers:
            w.flush()

        print(f"Iter {iteration:>4d} | timesteps: {total_timesteps:>10,} | "
              f"reward: {reward_mean:>8.1f} | coverage: {coverage:.4f} | "
              f"deaths: {battery_deaths:.1f}")

        # Save checkpoint (torch.save to bypass Python 3.13 cloudpickle bug)
        if iteration % CHECKPOINT_FREQ == 0:
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_iter_{iteration}_ts_{total_timesteps}.pt")
            torch.save({
                "model_state_dict": policy.model.state_dict(),
                "iteration": iteration,
                "timesteps": total_timesteps,
                "coverage": coverage,
                "reward": reward_mean,
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

        # Stop condition
        if total_timesteps >= TARGET_TIMESTEPS:
            print(f"\nReached {TARGET_TIMESTEPS:,} timesteps. Training complete!")
            break

    # Final checkpoint
    final_ckpt_path = os.path.join(ckpt_dir, f"checkpoint_final_ts_{total_timesteps}.pt")
    torch.save({
        "model_state_dict": policy.model.state_dict(),
        "iteration": iteration,
        "timesteps": total_timesteps,
        "coverage": coverage,
        "reward": reward_mean,
    }, final_ckpt_path)
    print(f"Final checkpoint: {final_ckpt_path}")
    for w in writers:
        w.close()
    algo.stop()
    ray.shutdown()
    print("Done!")

