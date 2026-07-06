"""
CNN-based COMA Critic for DSSE Coverage Environment
====================================================
Replaces the flat MLP critic with a spatial CNN that processes the 25x25
coverage matrices through Conv2D layers, exactly like the CNN agent does.

The key insight: the standard COMA critic receives the global state as a
flat 2588-float vector and tries to learn spatial relationships through
fully-connected layers. This is extremely inefficient for grid-based
environments. By processing the coverage matrices through convolutions,
the critic can understand spatial coverage patterns directly.

Architecture:
  1. Extract coverage matrices from state → (batch, n_agents, 1, 25, 25)
  2. Process each through shared Conv2D layers → spatial features (256 per agent)
  3. Extract position vectors from state → FC → position features (64 per agent)
  4. Concatenate: spatial + positions + other_agents_actions + agent_id
  5. FC layers → 9 Q-values (one per action)
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class COMACNNCritic(nn.Module):
    def __init__(self, scheme, args):
        super(COMACNNCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.output_type = "q"

        # DSSE observation layout: [positions(22) | matrix(625)] per agent
        self.positions_size = 22
        self.matrix_size = 625
        self.grid_size = 25
        self.obs_size = self.positions_size + self.matrix_size  # 647

        # --- CNN Branch: processes coverage matrices spatially ---
        # Shared Conv2D layers across all agents' matrices
        # Input: (batch * n_agents_channels, 1, 25, 25)
        # Conv1: (1, 25, 25) -> (16, 23, 23) -> MaxPool -> (16, 11, 11)
        # Conv2: (16, 11, 11) -> (32, 10, 10) -> MaxPool -> (32, 5, 5)
        # Flatten: 32 * 5 * 5 = 800
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        # 800 -> 128 per agent's matrix
        self.cnn_fc = nn.Sequential(
            nn.Linear(800, 128),
            nn.ReLU(),
        )

        # --- Position Branch: processes position/battery vectors ---
        # positions_size * n_agents = 22 * 4 = 88 floats from the global state
        self.pos_fc = nn.Sequential(
            nn.Linear(self.positions_size * self.n_agents, 128),
            nn.ReLU(),
        )

        # --- Action + Agent ID inputs ---
        # Other agents' actions (masked): n_agents * n_actions
        action_input_size = self.n_agents * self.n_actions
        # Agent ID: n_agents
        agent_id_size = self.n_agents if self.args.obs_agent_id else 0

        # --- Merge MLP ---
        # CNN features (128 * n_agents) + position features (128) + actions + agent_id
        merge_input = 128 * self.n_agents + 128 + action_input_size + agent_id_size

        # Individual obs adds another 647 floats processed through a small FC
        self.use_individual_obs = getattr(self.args, "obs_individual_obs", False)
        if self.use_individual_obs:
            self.ind_obs_fc = nn.Sequential(
                nn.Linear(self.obs_size, 128),
                nn.ReLU(),
            )
            merge_input += 128

        hidden = getattr(args, "hidden_dim", 256)
        self.merge_fc = nn.Sequential(
            nn.Linear(merge_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.n_actions),
        )

    def forward(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        # --- 1. Extract state and parse coverage matrices ---
        # state shape: (bs, max_t, state_size) where state_size = n_agents * obs_size
        state = batch["state"][:, ts]  # (bs, max_t, n_agents * obs_size)

        # Reshape to per-agent observations: (bs, max_t, n_agents, obs_size)
        state_per_agent = state.view(bs, max_t, self.n_agents, self.obs_size)

        # Extract position vectors: (bs, max_t, n_agents, positions_size)
        all_positions = state_per_agent[:, :, :, :self.positions_size]
        # Flatten positions for all agents: (bs, max_t, n_agents * positions_size)
        all_positions_flat = all_positions.reshape(bs, max_t, -1)

        # Extract coverage matrices: (bs, max_t, n_agents, matrix_size)
        all_matrices = state_per_agent[:, :, :, self.positions_size:]

        # --- 2. Process matrices through CNN ---
        # Reshape for Conv2D: (bs * max_t * n_agents, 1, 25, 25)
        matrices_4d = all_matrices.reshape(-1, 1, self.grid_size, self.grid_size)
        cnn_out = self.cnn(matrices_4d)  # (bs * max_t * n_agents, 800)
        cnn_features = self.cnn_fc(cnn_out)  # (bs * max_t * n_agents, 128)
        # Reshape back: (bs, max_t, n_agents * 128)
        cnn_features = cnn_features.view(bs, max_t, self.n_agents * 128)

        # --- 3. Process positions through FC ---
        pos_features = self.pos_fc(all_positions_flat)  # (bs, max_t, 128)

        # --- 4. Build per-agent outputs ---
        # We need to produce Q-values for each agent, so expand to (bs, max_t, n_agents, ...)
        # Expand CNN and position features to per-agent
        cnn_expanded = cnn_features.unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        pos_expanded = pos_features.unsqueeze(2).expand(-1, -1, self.n_agents, -1)

        # --- 5. Actions (masked by agent — standard COMA counterfactual) ---
        actions = batch["actions_onehot"][:, ts].view(bs, max_t, 1, -1).repeat(
            1, 1, self.n_agents, 1
        )
        agent_mask = 1 - th.eye(self.n_agents, device=batch.device)
        agent_mask = (
            agent_mask.view(-1, 1)
            .repeat(1, self.n_actions)
            .view(self.n_agents, -1)
        )
        masked_actions = actions * agent_mask.unsqueeze(0).unsqueeze(0)

        # --- 6. Agent ID ---
        parts = [cnn_expanded, pos_expanded, masked_actions]

        if self.args.obs_agent_id:
            agent_id = (
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, max_t, -1, -1)
            )
            parts.append(agent_id)

        # --- 7. Individual observations (optional) ---
        if self.use_individual_obs:
            ind_obs = batch["obs"][:, ts]  # (bs, max_t, n_agents, obs_size)
            ind_features = self.ind_obs_fc(ind_obs)  # (bs, max_t, n_agents, 128)
            parts.append(ind_features)

        # --- 8. Concatenate and predict Q-values ---
        merged = th.cat(parts, dim=-1)  # (bs, max_t, n_agents, merge_input)
        q = self.merge_fc(merged)  # (bs, max_t, n_agents, n_actions)

        return q
