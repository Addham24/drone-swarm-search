"""Quick diagnostic to check what's inside the checkpoint policy_state.pkl"""
import pickle
import os

CHECKPOINT_PATH = os.path.expanduser(
    "~/Desktop/dsse_run/ray_res/DSSE_Coverage/"
    "MAPPO_selfheal_comparison/"
    "PPO_DSSE_Coverage_0261b_00000_0_2026-05-07_12-54-04/"
    "checkpoint_000023"
)

policy_state_path = os.path.join(CHECKPOINT_PATH, "policies", "default_policy", "policy_state.pkl")
print(f"Loading: {policy_state_path}")

with open(policy_state_path, "rb") as f:
    policy_state = pickle.load(f)

print(f"\nType: {type(policy_state)}")
print(f"Top-level keys: {list(policy_state.keys())}")

for key in policy_state:
    val = policy_state[key]
    if isinstance(val, dict):
        print(f"\n  '{key}' -> dict with keys: {list(val.keys())[:10]}")
        # Check for model weights
        for subkey in list(val.keys())[:5]:
            subval = val[subkey]
            if hasattr(subval, 'shape'):
                print(f"    '{subkey}' -> tensor shape: {subval.shape}, mean: {subval.mean():.6f}")
            else:
                print(f"    '{subkey}' -> {type(subval).__name__}")
    elif hasattr(val, 'shape'):
        print(f"\n  '{key}' -> tensor shape: {val.shape}")
    else:
        print(f"\n  '{key}' -> {type(val).__name__}: {str(val)[:100]}")
