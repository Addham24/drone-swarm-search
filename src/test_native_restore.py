"""Test if native algo.restore() actually restores the weights correctly on all workers"""
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
from ray.tune.registry import register_env
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
register_env("DSSE_Coverage", lambda config: ParallelPettingZooEnv(env_creator(config)))
ModelCatalog.register_custom_model("CNNModel", CNNModel)

config = (PPOConfig()
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    .environment(env="DSSE_Coverage")
    .env_runners(num_env_runners=2, rollout_fragment_length="auto")
    .training(train_batch_size=8192, lr=1e-4, gamma=0.998, lambda_=0.9, use_gae=True,
              entropy_coeff=0.05, vf_clip_param=100000, minibatch_size=300, num_sgd_iter=10,
              model={"custom_model": "CNNModel", "_disable_preprocessor_api": True})
    .experimental(_disable_preprocessor_api=True)
    .debugging(log_level="ERROR")
    .framework(framework="torch")
    .resources(num_gpus=0)
)

algo = config.build()

CHECKPOINT_PATH = os.path.expanduser(
    "~/Desktop/dsse_run/ray_res/DSSE_Coverage/"
    "MAPPO_selfheal_comparison/"
    "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04/"
    "checkpoint_000023"
)

# Load checkpoint weights
policy_state_path = os.path.join(CHECKPOINT_PATH, "policies", "default_policy", "policy_state.pkl")
with open(policy_state_path, "rb") as f:
    policy_state = pickle.load(f)
checkpoint_weight = policy_state["weights"]["cnn.0.weight"].mean()

print(f"Checkpoint cnn.0.weight mean: {checkpoint_weight:.6f}")

# Call native restore
print("Restoring...")
algo.restore(CHECKPOINT_PATH)

local_policy = algo.get_policy("default_policy")
local_weight = list(local_policy.model.parameters())[0].data.mean().item()
print(f"Local worker weight mean: {local_weight:.6f}")

try:
    remote_weights = algo.env_runner_group.foreach_env_runner(
        lambda w: list(w.get_policy("default_policy").model.parameters())[0].data.mean().item()
    )
    print(f"Remote workers weight means: {remote_weights}")
except Exception as e:
    print(f"Error checking remote: {e}")

algo.stop()
ray.shutdown()
