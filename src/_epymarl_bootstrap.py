import sys
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

sys.path.insert(0, "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src")
sys.path.insert(0, "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src")

from envs import REGISTRY
from epymarl_env_adapter import DSSEMultiAgentEnv
from tracking.epymarl_tracking_env_adapter import DSSETrackingMultiAgentEnv

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

def dsse_tracking_fn(**kwargs):
    common_reward = kwargs.pop("common_reward", True)
    reward_scalarisation = kwargs.pop("reward_scalarisation", "sum")
    seed = kwargs.pop("seed", None)
    return DSSETrackingMultiAgentEnv(
        seed=seed,
        **kwargs,
    )

REGISTRY["dsse"] = dsse_fn
REGISTRY["dsse_tracking"] = dsse_tracking_fn
print("[✓] DSSE coverage & tracking environments registered with EPyMARL (via bootstrap).")

if __name__ == "__main__":
    sys.argv[0] = "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src/main.py"
    g = dict(globals())
    g["__file__"] = "/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src/main.py"
    exec(open("/Users/AdhamMotawi/Desktop/dsse_run/drone-swarm-search-algorithms/src/epymarl/src/main.py").read(), g)
