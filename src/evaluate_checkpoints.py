"""Evaluate original checkpoint vs resumed checkpoint to check for performance degradation"""
import os, pickle, sys
sys.path.insert(0, os.path.dirname(__file__))

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
import torch
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CNNModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, model_config, name, **kw):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, model_config, name, **kw)
        nn.Module.__init__(self)
        def get_flatten_size(grid_size):
            x = (grid_size - 2) // 2; x = (x - 1) // 2; return 32 * x * x
        grid_size = obs_space[1].shape[0]
        flatten_size = get_flatten_size(grid_size)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (3,3)), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (2,2)), nn.Tanh(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(flatten_size, 256), nn.Tanh(),
        )
        self.linear = nn.Sequential(nn.Linear(obs_space[0].shape[0], 512), nn.Tanh(), nn.Linear(512, 256), nn.Tanh())
        self.join = nn.Sequential(nn.Linear(256*2, 256), nn.Tanh())
        self.policy_fn = nn.Linear(256, num_outputs)
        self.value_fn = nn.Linear(256, 1)
    def forward(self, input_dict, state, seq_lens):
        inp_pos = input_dict["obs"][0].float(); inp_mat = input_dict["obs"][1].float().unsqueeze(1)
        v = self.join(torch.cat((self.cnn(inp_mat), self.linear(inp_pos)), 1))
        self._value_out = self.value_fn(v); return self.policy_fn(v), state
    def value_function(self): return self._value_out.flatten()

def env_creator(args):
    matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(timestep_limit=750, drone_amount=4, prob_matrix_path=matrix_path)
    env.reward_scheme = {"default": -0.1, "exceed_timestep": 0.0, "search_cell": 5.0, "done": 500.0, "reward_poc": 0.0}
    env = AllPositionsWrapper(env)
    env = BatteryStationWrapper(env, max_battery=125, depletion_rate=1, charge_rate=15, fault_prob=0.0005)
    return RetainDronePosWrapper(env, [(0,0),(0,1),(1,0),(1,1)])

ray.init(log_to_driver=False)
register_env = lambda name, creator: None # registered below if needed, but we can do it via Ray API
from ray.tune.registry import register_env
register_env("DSSE_Coverage", lambda config: ParallelPettingZooEnv(env_creator(config)))
ModelCatalog.register_custom_model("CNNModel", CNNModel)

config = (PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env="DSSE_Coverage")
    .env_runners(num_env_runners=0) # Only local worker for evaluation
    .training(model={"custom_model": "CNNModel", "_disable_preprocessor_api": True})
    .experimental(_disable_preprocessor_api=True)
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(num_gpus=0)
)
algo = config.build()
policy = algo.get_policy("default_policy")

def evaluate_weights(weights_dict, name):
    policy.set_weights(weights_dict)
    
    # Run 10 evaluation episodes
    eval_env = env_creator(None)
    coverages = []
    deaths_list = []
    
    for ep in range(10):
        obs, infos = eval_env.reset()
        terminated = truncated = False
        
        while not (terminated or truncated):
            # Compute actions greedily (explore=False)
            actions = {}
            for agent in eval_env.possible_agents:
                action = algo.compute_single_action(obs[agent], explore=True, policy_id="default_policy")
                actions[agent] = action
            
            obs, rewards, terminations, truncations, infos = eval_env.step(actions)
            terminated = any(terminations.values())
            truncated = any(truncations.values())
            
        # Get final episode metrics
        coverages.append(infos["drone0"].get("coverage_rate", 0))
        # Count deaths
        deaths = sum(1 for a in eval_env.possible_agents if not eval_env.alive[a])
        deaths_list.append(deaths)
        
    print(f"\nEvaluation Results for {name}:")
    print(f"  Mean Coverage: {np.mean(coverages):.4f} (Min: {np.min(coverages):.4f}, Max: {np.max(coverages):.4f})")
    print(f"  Mean Deaths:   {np.mean(deaths_list):.1f}")

# --- Load Original Checkpoint weights ---
orig_path = os.path.expanduser(
    "~/Desktop/dsse_run/ray_res/DSSE_Coverage/"
    "MAPPO_selfheal_comparison/"
    "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04/"
    "checkpoint_000023"
)
policy_state_path = os.path.join(orig_path, "policies", "default_policy", "policy_state.pkl")
with open(policy_state_path, "rb") as f:
    orig_state = pickle.load(f)
evaluate_weights(orig_state["weights"], "Original Checkpoint (Checkpoint 23)")

# --- Load Resumed Checkpoint weights ---
resumed_path = os.path.expanduser(
    "~/Desktop/dsse_run/drone-swarm-search-algorithms/src/"
    "ray_res/DSSE_Coverage/MAPPO_selfheal_final_v2/checkpoints/"
    "checkpoint_iter_1730_ts_16146432.pt"
)
if os.path.exists(resumed_path):
    resumed_data = torch.load(resumed_path, weights_only=False)
    # The saved dict format in torch.save is model_state_dict, convert to rllib format
    # RLlib policy expects weights to match state_dict keys directly
    evaluate_weights(resumed_data["model_state_dict"], f"Resumed Checkpoint (Iter 1730)")
else:
    print(f"\nResumed checkpoint not found at: {resumed_path}")

# Print an observation for debugging
env = env_creator(None)
obs, infos = env.reset()
print("\nDebug Observation (positions vector):")
print("  Shape:", obs["drone0"][0].shape)
print("  Values:", obs["drone0"][0])
print("  Wind/Matrix shape:", obs["drone0"][1].shape)

algo.stop()
ray.shutdown()
