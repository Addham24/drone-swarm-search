"""
DSSE-3D Strict Policy Loaders
==============================
Checkpoint loaders used ONLY by the 3D evaluation suite.

Why this module exists
----------------------
The legacy 2D loaders call ``load_state_dict(..., strict=False)``. That silently
ignores any key mismatch, which was found to hide three real problems:

1. Coverage I-DQN checkpoints are RLlib *Dueling-DQN* networks
   (``q_head`` + ``advantage_module`` + ``value_module``). Loading them into a
   PPO ``CNNModel`` left the policy head randomly initialised.
2. QMIX EPyMARL checkpoints were trained with ``use_rnn=False`` (the ``rnn``
   layer is a feed-forward ``nn.Linear``), but the legacy loader always builds a
   ``GRUCell``, so the recurrent layer was randomly initialised.
3. EPyMARL hidden state was never reset between episodes.

Every loader here uses ``strict=True`` (or an explicit key check) so that a
mismatched checkpoint raises immediately instead of producing silently-wrong
numbers.
"""

import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from tracking.evaluate_tracking_models import VersionCompatibilityUnpickler, MockArgs
from modules.agents.cnn_agent import CNNAgent
from modules.agents.rnn_agent import RNNAgent


# ──────────────────────────────────────────────────────────────────────────────
# RLlib PPO-style models (RSPO V2, MAPPO)
# ──────────────────────────────────────────────────────────────────────────────

def _read_rllib_weights(checkpoint_path):
    """Return the weight dict stored in an RLlib checkpoint dir or a raw .pt file."""
    if checkpoint_path.endswith(".pt"):
        ckpt = th.load(checkpoint_path, map_location="cpu", weights_only=False)
        return ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    policy_state_path = os.path.join(checkpoint_path, "policies", "default_policy", "policy_state.pkl")
    if not os.path.exists(policy_state_path):
        policy_state_path = os.path.join(checkpoint_path, "policy_state.pkl")
    if not os.path.exists(policy_state_path):
        raise FileNotFoundError(f"No policy_state.pkl under {checkpoint_path}")

    with open(policy_state_path, "rb") as f:
        policy_state = VersionCompatibilityUnpickler(f).load()
    weights = policy_state["weights"]
    return {k: (th.from_numpy(v) if not isinstance(v, th.Tensor) else v) for k, v in weights.items()}


def load_rllib_ppo_policy(model_cls, checkpoint_path, model_name, greedy):
    """
    Load an RLlib PPO-family model with strict key matching.

    Args:
        greedy: True -> argmax action (matches the 2D *coverage* protocol).
                False -> sample from the categorical policy (matches the 2D
                *tracking* protocol).
    """
    from gymnasium.spaces import Box, Tuple as GymTuple, Discrete

    obs_space = GymTuple([
        Box(-1.0, 1.0, (22,), dtype=np.float32),
        Box(-1.0, 1.0, (25, 25), dtype=np.float32),
    ])
    model = model_cls(obs_space, Discrete(9), 9, {}, model_name)
    model.load_state_dict(_read_rllib_weights(checkpoint_path), strict=True)
    model.eval()

    def policy_fn(obs_dict, agents, *args, **kwargs):
        actions = {}
        for agent in agents:
            if agent not in obs_dict:
                continue
            pos, matrix = obs_dict[agent]
            pos_t = th.tensor(pos, dtype=th.float32).unsqueeze(0)
            mat_t = th.tensor(matrix, dtype=th.float32).unsqueeze(0)
            with th.no_grad():
                logits, _ = model({"obs": (pos_t, mat_t)}, [], None)
                if greedy:
                    actions[agent] = int(logits.argmax(dim=-1).item())
                else:
                    actions[agent] = int(th.distributions.Categorical(logits=logits).sample().item())
        return actions

    return policy_fn


# ──────────────────────────────────────────────────────────────────────────────
# RLlib Dueling-DQN (coverage I-DQN)
# ──────────────────────────────────────────────────────────────────────────────

class _SlimFC(nn.Module):
    """Mirror of ray.rllib.models.torch.misc.SlimFC key layout (``_model.0``)."""

    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self._model = nn.Sequential(nn.Linear(n_in, n_out))
        self.activation = activation

    def forward(self, x):
        x = self._model(x)
        return F.relu(x) if self.activation == "relu" else x


class DuelingCNNQNet(nn.Module):
    """
    Exact re-implementation of the trained coverage I-DQN network:
    QMIXCNNModel / QMIXVanillaCNNModel (train_qmix_cnn_cov.py / train_qmix_vanilla.py)
    followed by RLlib's dueling heads (dueling=True, q_hiddens=[256], relu).
    """

    def __init__(self, pos_dim=22, grid=25, n_actions=9):
        super().__init__()
        x = (grid - 2) // 2
        x = (x - 1) // 2
        flatten = 32 * x * x
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3)), nn.Tanh(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, (2, 2)), nn.Tanh(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(flatten, 256), nn.Tanh(),
        )
        self.linear = nn.Sequential(
            nn.Linear(pos_dim, 512), nn.Tanh(), nn.Linear(512, 256), nn.Tanh(),
        )
        self.join = nn.Sequential(nn.Linear(512, 256), nn.Tanh())
        self.q_head = nn.Linear(256, 256)      # RLlib "num_outputs" = dueling hidden size
        self.value_head = nn.Linear(256, 1)    # unused by DQN, present in checkpoint

        self.advantage_module = nn.Module()
        self.advantage_module.dueling_A_0 = _SlimFC(256, 256, "relu")
        self.advantage_module.A = _SlimFC(256, n_actions, None)
        self.value_module = nn.Module()
        self.value_module.dueling_V_0 = _SlimFC(256, 256, "relu")
        self.value_module.V = _SlimFC(256, 1, None)

    def q_values(self, pos, matrix):
        combined = self.join(th.cat((self.cnn(matrix.unsqueeze(1)), self.linear(pos)), dim=1))
        model_out = self.q_head(combined)
        adv = self.advantage_module.A(self.advantage_module.dueling_A_0(model_out))
        val = self.value_module.V(self.value_module.dueling_V_0(model_out))
        return val + adv - adv.mean(dim=1, keepdim=True)


def load_rllib_dqn_policy(checkpoint_path):
    """Load a coverage I-DQN checkpoint (strict) and act greedily on Q-values."""
    net = DuelingCNNQNet()
    net.load_state_dict(_read_rllib_weights(checkpoint_path), strict=True)
    net.eval()

    def policy_fn(obs_dict, agents, *args, **kwargs):
        actions = {}
        for agent in agents:
            if agent not in obs_dict:
                continue
            pos, matrix = obs_dict[agent]
            pos_t = th.tensor(pos, dtype=th.float32).unsqueeze(0)
            mat_t = th.tensor(matrix, dtype=th.float32).unsqueeze(0)
            with th.no_grad():
                actions[agent] = int(net.q_values(pos_t, mat_t).argmax(dim=-1).item())
        return actions

    return policy_fn


# ──────────────────────────────────────────────────────────────────────────────
# EPyMARL agents (QMIX, MAA2C, COMA, IQL)
# ──────────────────────────────────────────────────────────────────────────────

def load_epymarl_policy_strict(model_th_path, n_agents=4):
    """
    Load an EPyMARL ``agent.th`` with strict key matching.

    Architecture is inferred from the checkpoint itself:
      * ``fc1.weight`` present  -> RNNAgent, else CNNAgent
      * ``rnn.weight_ih`` present -> GRU (use_rnn=True); ``rnn.weight`` -> feed-forward

    The returned policy_fn accepts ``reset=True`` to zero the hidden state; the
    evaluator must pass it on the first step of every episode.
    """
    state_dict = th.load(model_th_path, map_location="cpu")
    use_rnn = "rnn.weight_ih" in state_dict

    if "fc1.weight" in state_dict:
        hidden_dim = state_dict["fc1.weight"].shape[0]
        agent = RNNAgent(input_shape=651, args=MockArgs(use_rnn=use_rnn, hidden_dim=hidden_dim, n_actions=9))
    else:
        hidden_dim = state_dict["fc2.weight"].shape[1]
        agent = CNNAgent(input_shape=651, args=MockArgs(use_rnn=use_rnn, hidden_dim=hidden_dim, n_actions=9))

    agent.load_state_dict(state_dict, strict=True)
    agent.eval()

    hidden = [None]
    obs_size = 22 + 625

    def policy_fn(obs_dict, agents, reset=False, *args, **kwargs):
        agents = list(agents)
        n = len(agents)
        if reset or hidden[0] is None or hidden[0].shape[0] != n:
            hidden[0] = agent.init_hidden().expand(n, -1).clone()

        inputs = []
        for i, name in enumerate(agents):
            if name in obs_dict:
                positions, matrix = obs_dict[name]
                flat = np.concatenate([
                    np.asarray(positions, dtype=np.float32).flatten(),
                    np.asarray(matrix, dtype=np.float32).flatten(),
                ])
            else:
                flat = np.zeros(obs_size, dtype=np.float32)
            onehot = np.zeros(n, dtype=np.float32)
            onehot[i] = 1.0
            inputs.append(np.concatenate([flat, onehot]))

        with th.no_grad():
            q, hidden[0] = agent(th.tensor(np.stack(inputs), dtype=th.float32), hidden[0])
            acts = q.argmax(dim=-1)
        return {name: int(acts[i].item()) for i, name in enumerate(agents) if name in obs_dict}

    return policy_fn


# ──────────────────────────────────────────────────────────────────────────────
# 50x50 -> 25x25 observation resizing (tracking protocol)
# ──────────────────────────────────────────────────────────────────────────────

def make_resizing_policy(policy_fn, source_grid_size=50, target_grid_size=25):
    """
    Identical to ``make_resizing_policy`` in tracking/evaluate_tracking_50x50.py:
    PIL BOX downsample + probability-mass rescaling by (source/target)^2 on
    non-obstacle cells. Extended to forward ``reset``/kwargs to the wrapped policy.
    """
    scale_factor = float((source_grid_size / target_grid_size) ** 2)

    def resized_policy_fn(obs_dict, agents, *args, **kwargs):
        resized = {}
        for agent in agents:
            if agent not in obs_dict:
                continue
            pos_vec, matrix = obs_dict[agent]
            img = Image.fromarray(matrix.astype(np.float32), mode="F")
            small = np.array(img.resize((target_grid_size, target_grid_size), resample=Image.BOX), dtype=np.float32)
            small[small >= 0] *= scale_factor
            resized[agent] = (pos_vec, small)
        return policy_fn(resized, agents, *args, **kwargs)

    return resized_policy_fn
