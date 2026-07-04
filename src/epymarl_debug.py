"""
EPyMARL Env Initialization Debugger
===================================
Manually imports and reproduces the exact env initialization sequence
to isolate where the silent exit or crash is happening.
"""

import sys
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EPYMARL_DIR = os.path.join(SRC_DIR, "epymarl")

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(EPYMARL_DIR, "src"))

print("[1] Importing PyTorch...")
import torch as th
print(f"  PyTorch version: {th.__version__}")

print("[2] Importing EPyMARL modules...")
from envs import REGISTRY as env_REGISTRY
from epymarl_env_adapter import DSSEMultiAgentEnv

# Register env
def dsse_fn(**kwargs):
    common_reward = kwargs.pop("common_reward", True)
    reward_scalarisation = kwargs.pop("reward_scalarisation", "sum")
    seed = kwargs.pop("seed", None)
    return DSSEMultiAgentEnv(
        common_reward=common_reward,
        reward_scalarisation=reward_scalarisation,
        seed=seed,
        **kwargs,
    )

env_REGISTRY["dsse"] = dsse_fn
print("  Environment dsse registered.")

# Try creating the env manually using the exact same arguments as QMIX config
env_args = {
    'charge_rate': 15,
    'compensation_bonus': 2.5,
    'compensation_horizon': 100,
    'compensation_penalty': -0.5,
    'depletion_rate': 1,
    'fault_prob': 0.0005,
    'grid_size': 25,
    'map_name': 'dsse_coverage',
    'max_battery': 125,
    'mixing_alpha': 0.5,
    'n_agents': 4,
    'seed': 12345,
    'timestep_limit': 750
}

print("[3] Attempting manual env creation...")
env = env_REGISTRY["dsse"](
    **env_args,
    common_reward=True,
    reward_scalarisation="sum"
)
print("  Env created successfully!")
print(f"  Env details: {env.get_env_info()}")

print("[4] Attempting single step...")
obs, info = env.reset()
actions = [0, 1, 2, 3]
obs, reward, done, trunc, info = env.step(actions)
print("  Step completed successfully!")
print(f"  Step reward: {reward}")

print("[5] Manually importing all EPyMARL registrables...")
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY

print("  All registrables imported successfully!")
print(f"  Available runners: {list(r_REGISTRY.keys())}")
print(f"  Available learners: {list(le_REGISTRY.keys())}")
print(f"  Available controllers: {list(mac_REGISTRY.keys())}")

print("[6] Attempting EpisodeRunner instantiation...")
from types import SimpleNamespace as SN
class MockLogger:
    def log_stat(self, *args, **kwargs): pass
    def console_logger(self): pass

args = SN(
    runner="episode",
    env="dsse",
    env_args=env_args,
    common_reward=True,
    reward_scalarisation="sum",
    batch_size_run=1,
)

runner = r_REGISTRY[args.runner](args=args, logger=MockLogger())
print("  EpisodeRunner instantiated successfully!")
print(f"  Runner env_info: {runner.get_env_info()}")

print("[✓] ALL INITIALIZATION TESTS PASSED! No silent exits detected in debugger.")
