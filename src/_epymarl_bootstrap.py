import sys
import os

# Workaround for old tensorboard-logger protobuf issue
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Ensure paths
sys.path.insert(0, "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src")
sys.path.insert(0, "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src")

# Register DSSE environment
from envs import REGISTRY
from epymarl_env_adapter import DSSEMultiAgentEnv

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

REGISTRY["dsse"] = dsse_fn
print("[✓] DSSE environment registered with EPyMARL (via bootstrap).")

# Now run EPyMARL's main with overridden __file__ in globals
if __name__ == "__main__":
    sys.argv[0] = "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src/main.py"
    g = dict(globals())
    g["__file__"] = "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src/main.py"
    exec(open("/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src/main.py").read(), g)
