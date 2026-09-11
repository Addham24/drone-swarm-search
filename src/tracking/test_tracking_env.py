import os
import sys

# Ensure src directory is on sys.path
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tracking.tracking_env_creator import make_tracking_env

def run_smoke_test():
    print("==================================================")
    print("  TRACKING ENVIRONMENT WRAPPER SMOKE TEST  ")
    print("==================================================")
    
    # 1. Create Vanilla and SelfHeal tracking envs
    env_vanilla = make_tracking_env(is_self_heal=False)
    env_selfheal = make_tracking_env(is_self_heal=True)
    
    print("SUCCESS: Environment instances created.")
    
    # 2. Reset test & inspect drift vector randomization
    obs, infos = env_vanilla.reset(seed=42)
    first_vector = getattr(env_vanilla.unwrapped, "vector", None)
    print(f"Reset 1 Drift Vector: {first_vector}")
    
    obs2, infos2 = env_vanilla.reset(seed=123)
    second_vector = getattr(env_vanilla.unwrapped, "vector", None)
    print(f"Reset 2 Drift Vector: {second_vector}")
    
    assert first_vector != second_vector, "Drift vector was not randomized between resets!"
    print("SUCCESS: Drift vector successfully randomized across resets.")
    
    # 3. Inspect Observation Space Shape
    first_agent = env_vanilla.agents[0]
    agent_obs = obs[first_agent]
    assert isinstance(agent_obs, tuple) and len(agent_obs) == 2, f"Expected tuple obs, got {type(agent_obs)}"
    
    pos_vec, matrix = agent_obs
    print(f"Vector Shape: {pos_vec.shape} (Expected: (22,))")
    print(f"Matrix Shape: {matrix.shape} (Expected: (25, 25))")
    
    assert pos_vec.shape == (22,), f"Vector shape mismatch: {pos_vec.shape}"
    assert matrix.shape == (25, 25), f"Matrix shape mismatch: {matrix.shape}"
    print("SUCCESS: Observation shapes 100% matched (22-dim vector + 25x25 matrix).")
    
    # 4. Step execution test
    actions = {agent: 1 for agent in env_vanilla.agents} # Move UP
    next_obs, rewards, term, trunc, infos = env_vanilla.step(actions)
    
    print(f"Step 1 Rewards: {rewards}")
    print("SUCCESS: Step execution completed without errors.")
    print("==================================================")
    print("   ALL TRACKING WRAPPER SMOKE TESTS PASSED!   ")
    print("==================================================")

if __name__ == "__main__":
    run_smoke_test()
