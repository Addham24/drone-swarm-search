"""
DSSE-3D Policy Adapters
========================
Adapts pre-trained 2D MARL policy checkpoints to work with the augmented 3D
observation space produced by DSSE3DAdapter.

Problem:
    2D policies expect observation vectors of dimension D_2d (e.g., 22 for
    RLlib position vectors, 651 for EPyMARL flat obs). The 3D adapter appends
    altitude context, increasing the dimension to D_3d = D_2d + n_drones + 1.

Solution:
    These adapter functions strip the 3D altitude suffix from the observation
    before feeding it to the pre-trained 2D policy. The policy sees exactly
    the same observation shape it was trained on.

    This is the "zero-shot" aspect: no weight modification, no retraining,
    no fine-tuning. The 2D spatial reasoning is preserved perfectly.

Usage:
    from dsse_3d.policy_adapters import wrap_rllib_policy_for_3d, wrap_epymarl_policy_for_3d
"""

import numpy as np
import torch as th
from typing import Callable, Dict, List


# Number of extra dimensions appended by DSSE3DAdapter
# = n_drones (altitude per drone) + 1 (FOV radius)
DEFAULT_N_DRONES = 4
ALTITUDE_OBS_DIM = DEFAULT_N_DRONES + 1  # 5


def strip_3d_obs(obs_dict: dict, n_extra_dims: int = ALTITUDE_OBS_DIM) -> dict:
    """
    Strip the 3D altitude context suffix from each agent's position vector,
    restoring the original 2D observation shape.

    The DSSE3DAdapter appends [alt_0, alt_1, ..., alt_{n-1}, fov_radius]
    to the end of the position vector. This function removes those dimensions.

    Args:
        obs_dict: Dict of agent -> (positions_augmented, matrix).
        n_extra_dims: Number of 3D-appended dimensions to strip.

    Returns:
        Dict of agent -> (positions_original, matrix) with 2D-compatible shapes.
    """
    stripped = {}
    for agent, obs in obs_dict.items():
        if isinstance(obs, tuple) and len(obs) == 2:
            positions, matrix = obs
            # Strip last n_extra_dims from position vector
            positions_2d = positions[:-n_extra_dims] if n_extra_dims > 0 else positions
            stripped[agent] = (positions_2d, matrix)
        else:
            stripped[agent] = obs
    return stripped


def wrap_rllib_policy_for_3d(
    policy_fn_2d: Callable,
    n_drones: int = DEFAULT_N_DRONES,
) -> Callable:
    """
    Wrap a pre-trained RLlib policy function to work with 3D observations.

    The wrapper strips 3D altitude dimensions before calling the 2D policy,
    ensuring weight compatibility.

    Args:
        policy_fn_2d: The original 2D policy function with signature
                       (obs_dict, agents, *args, **kwargs) -> actions_dict.
        n_drones: Number of drones (determines altitude obs dimensions).

    Returns:
        A policy function with the same signature that handles 3D observations.
    """
    n_extra = n_drones + 1

    def policy_fn_3d(obs_dict, agents, *args, **kwargs):
        obs_2d = strip_3d_obs(obs_dict, n_extra_dims=n_extra)
        return policy_fn_2d(obs_2d, agents, *args, **kwargs)

    return policy_fn_3d


def wrap_epymarl_policy_for_3d(
    policy_fn_2d: Callable,
    n_drones: int = DEFAULT_N_DRONES,
) -> Callable:
    """
    Wrap a pre-trained EPyMARL policy function to work with 3D observations.

    EPyMARL policies concatenate positions + flattened matrix + agent_id_onehot
    internally. The wrapper strips altitude dimensions from the position
    portion before the policy processes them.

    Args:
        policy_fn_2d: The original 2D EPyMARL policy function.
        n_drones: Number of drones.

    Returns:
        A policy function compatible with 3D augmented observations.
    """
    n_extra = n_drones + 1

    def policy_fn_3d(obs_dict, agents, *args, **kwargs):
        obs_2d = strip_3d_obs(obs_dict, n_extra_dims=n_extra)
        return policy_fn_2d(obs_2d, agents, *args, **kwargs)

    return policy_fn_3d


def extract_altitude_context(obs_dict: dict, n_drones: int = DEFAULT_N_DRONES) -> Dict[str, dict]:
    """
    Extract the 3D altitude context from augmented observations for logging
    and visualization purposes.

    Args:
        obs_dict: Dict of agent -> (positions_augmented, matrix).
        n_drones: Number of drones.

    Returns:
        Dict of agent -> {"altitudes": array, "fov_radius": float}.
    """
    context = {}
    n_extra = n_drones + 1

    for agent, obs in obs_dict.items():
        if isinstance(obs, tuple) and len(obs) == 2:
            positions = obs[0]
            if len(positions) >= n_extra:
                alt_suffix = positions[-n_extra:]
                context[agent] = {
                    "altitudes": alt_suffix[:n_drones],
                    "fov_radius": float(alt_suffix[-1]),
                }

    return context
