import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNAgent, self).__init__()
        self.args = args
        
        # Environment dimensions (matches epymarl_env_adapter.py output)
        self.obs_size = 647
        self.positions_size = 22
        self.matrix_size = 625
        self.grid_size = 25

        # 1. Spatial Convolution Branch (matches MAPPO CNN architecture exactly)
        # Conv1: (1, 25, 25) -> (16, 23, 23) -> MaxPool: (16, 11, 11)
        # Conv2: (16, 11, 11) -> (32, 10, 10) -> MaxPool: (32, 5, 5)
        # Flatten size: 32 * 5 * 5 = 800
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.cnn_dense = nn.Sequential(
            nn.Linear(800, 256),
            nn.Tanh()
        )

        # 2. Linear Position & Status Branch (coordinates, battery, station coords)
        self.linear = nn.Sequential(
            nn.Linear(self.positions_size, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh()
        )

        # 3. Join MLP
        self.join = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.Tanh()
        )

        # 4. GRU Layer for temporal coordination
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(256, args.hidden_dim)
        else:
            self.rnn = nn.Linear(256, args.hidden_dim)
            
        # 5. Output Action Q-values
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

    def init_hidden(self):
        # Create hidden state tensor on the correct device
        return self.join[0].weight.new(1, self.args.hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        # EPyMARL controller packs observations first: inputs[:, :obs_size] is guaranteed raw obs
        obs = inputs[:, :self.obs_size]
        
        # Slice raw observations into positions vector and 2D grid matrix
        positions = obs[:, :self.positions_size]
        matrix_flat = obs[:, self.positions_size:self.obs_size]
        
        # Reshape to (Batch, Channels, Height, Width) for PyTorch Conv2D
        matrix = matrix_flat.view(-1, 1, self.grid_size, self.grid_size)
        
        # Forward through both branches
        cnn_out = self.cnn_dense(self.cnn(matrix))
        linear_out = self.linear(positions)
        
        # Concat, joint process and GRU update
        merged = torch.cat((cnn_out, linear_out), dim=1)
        x = self.join(merged)
        
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
            
        # Predict Q-values
        q = self.fc2(h)
        return q, h
