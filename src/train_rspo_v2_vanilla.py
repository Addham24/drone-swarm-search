"""
RSPO V2 Vanilla Training Script for Drone Swarm Coverage
=========================================================
RSPO V2 training WITH fault injection but WITHOUT teammate compensation rewards (Vanilla Environment).

Features Implemented (RSPO V2):
  1. Fleet Health Embedding (FHE): Dedicated 14-dim fleet status vector feeding h_t in R^32.
  2. Crash-Sector Spatial Attention (CSSA): Dynamic Gaussian heatmap spatial attention on CNN feature maps.
  3. Dual-Objective Value Decomposition (DOVD): Separate V_search and V_takeover heads with auxiliary supervision in custom_loss().
  4. Resilience-Adaptive Clipping (RAC): Dynamic per-sample clipping parameter eps_t integrated into RSPOTorchPolicy.

Usage:
    python train_rspo_v2_vanilla.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pathlib
import numpy as np
import torch
from torch import nn

import ray
from ray import tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env, register_trainable

from DSSE import CoverageDroneSwarmSearch
from DSSE.environment.wrappers import RetainDronePosWrapper, AllPositionsWrapper
from battery_station_wrapper import BatteryStationWrapper

from rspo_v2_policy import RSPOPPO, RSPOPPOConfig, RSPOTorchPolicy


class RSPOBatteryMetricsCallback(DefaultCallbacks):
    """Logs battery deaths, charge steps, average level, coverage, and RSPO-specific metrics."""

    def on_episode_start(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        episode.user_data["deaths"] = 0
        episode.user_data["charge_steps"] = 0
        episode.user_data["total_battery"] = 0
        episode.user_data["battery_samples"] = 0

    def on_episode_step(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info is None or not isinstance(info, dict):
                continue
            if info.get("stranded_event", False):
                episode.user_data["deaths"] += 1
            if info.get("charging", False):
                episode.user_data["charge_steps"] += 1
            batt = info.get("battery")
            if batt is not None:
                episode.user_data["total_battery"] += batt
                episode.user_data["battery_samples"] += 1

    def on_episode_end(self, *, episode, env_runner=None, worker=None, base_env=None, **kwargs):
        episode.custom_metrics["battery_deaths"] = episode.user_data["deaths"]
        episode.custom_metrics["battery_charge_steps"] = episode.user_data["charge_steps"]

        samples = episode.user_data["battery_samples"]
        if samples > 0:
            episode.custom_metrics["battery_avg_level"] = (
                episode.user_data["total_battery"] / samples
            )
        else:
            episode.custom_metrics["battery_avg_level"] = 0.0

        for agent_id in episode.get_agents():
            info = episode.last_info_for(agent_id)
            if info and isinstance(info, dict) and "coverage_rate" in info:
                episode.custom_metrics["coverage_rate"] = info["coverage_rate"]
                break


class RSPOModelV2(TorchModelV2, nn.Module):
    """
    RSPO V2 Neural Network Model
    ----------------------------
    Integrates:
      1. Fleet Health Embedding (FHE): Dedicated 14-dim fleet status vector -> h_t (dim 32)
      2. Crash-Sector Spatial Attention (CSSA): Dynamic Gaussian heatmap spatial attention on CNN maps
      3. Dual-Objective Value Decomposition (DOVD): Value heads + custom_loss auxiliary supervision
      4. Resilience-Adaptive Clipping (RAC): Per-sample eps_t state-conditioned clip parameter
    """

    def __init__(
        self,
        obs_space,
        act_space,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        TorchModelV2.__init__(
            self, obs_space, act_space, num_outputs, model_config, name, **kw
        )
        nn.Module.__init__(self)

        def get_flatten_size(grid_size):
            x = (grid_size - 2) // 2
            x = (x - 1) // 2
            return 32 * x * x

        grid_size = obs_space[1].shape[0]
        flatten_size = get_flatten_size(grid_size)
        positions_dim = obs_space[0].shape[0]

        print(f"[RSPO V2] Grid size: {grid_size}, Flatten size: {flatten_size}, Positions Dim: {positions_dim}")

        # ── 1. Spatial Grid Encoder (CNN Backbone) ──
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
        )

        # ── 2. Component 4: Crash-Sector Spatial Attention (CSSA) Gate ──
        # 1x1 Conv that combines CNN 32-channel feature maps with 1-channel Gaussian crash heatmap
        self.cssa_conv = nn.Conv2d(in_channels=33, out_channels=32, kernel_size=1)

        self.cnn_flatten = nn.Flatten()
        self.cnn_project = nn.Sequential(
            nn.Linear(flatten_size, 256),
            nn.Tanh(),
        )

        # ── 3. Local State Linear Encoder ──
        self.linear_obs = nn.Sequential(
            nn.Linear(positions_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
        )

        # ── 4. Component 1: Fleet Health Embedding (FHE) Encoder ──
        # Dedicated 14-dim fleet status input:
        # [alive_ratio(1), mean_batt(1), min_batt(1), max_batt(1), crash_active(1), n_crashes(1), batt_levels(4), alive_indicators(4)]
        self.fhe_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )

        # ── 5. Policy Network (conditioned on CNN + Linear + FHE) ──
        self.actor_join = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.Tanh(),
        )
        self.policy_fn = nn.Linear(256, num_outputs)

        # ── 6. Component 2: Dual-Objective Value Decomposition (DOVD) Critic ──
        self.critic_join = nn.Sequential(
            nn.Linear(256 + 256 + 32, 256),
            nn.Tanh(),
        )

        # Head A: Primary Cell Search Value V_search(s)
        self.v_search_head = nn.Linear(256, 1)

        # Head B: Sector Takeover Compensation Value V_takeover(s)
        self.v_takeover_head = nn.Linear(256, 1)

        # Learned Gating Alpha_t in (0, 1)
        self.alpha_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # ── 7. Component 3: Resilience-Adaptive Clipping (RAC) Gate ──
        # State-dependent clip offset from FHE: eps_t = 0.1 + 0.2 * rac_gate(h_t)
        self.rac_gate = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._value_out = None
        self._current_eps_t = None
        self._stored_alpha_t = None
        self._stored_v_search = None
        self._stored_v_takeover = None
        self._stored_crash_active = None
        self.tower_stats_extra = {}

    def _extract_fleet_health_state(self, input_positions):
        """
        Extracts explicit 14-dim fleet status vector from observation:
        [alive_ratio, mean_batt, min_batt, max_batt, crash_active, n_crashes, per_agent_batt(4), per_agent_alive(4)]
        """
        positions_part = input_positions[:, 0:8]
        batt_part = input_positions[:, 8:12]

        alive_mask = (positions_part[:, 0::2] >= 0.0).float()  # [B, 4]
        alive_ratio = alive_mask.mean(dim=1, keepdim=True)    # [B, 1]
        mean_batt = batt_part.mean(dim=1, keepdim=True)        # [B, 1]
        min_batt = batt_part.min(dim=1, keepdim=True).values  # [B, 1]
        max_batt = batt_part.max(dim=1, keepdim=True).values  # [B, 1]
        n_crashes = (1.0 - alive_mask).sum(dim=1, keepdim=True) # [B, 1]
        crash_active = (n_crashes > 0).float()                 # [B, 1]

        fleet_health_raw = torch.cat(
            [alive_ratio, mean_batt, min_batt, max_batt, crash_active, n_crashes, batt_part, alive_mask],
            dim=1,
        )
        return fleet_health_raw, alive_mask, crash_active

    def _compute_cssa_attention(self, cnn_features, input_positions, alive_mask, crash_active):
        """
        Computes Crash-Sector Spatial Attention (CSSA) heatmap on CNN feature maps [B, 32, 5, 5].
        """
        B = input_positions.shape[0]
        device = input_positions.device

        crash_part = input_positions[:, 12:20]  # [B, 8] (4 pairs of cy, cx normalized to [0,1])
        crash_coords = crash_part.view(B, 4, 2)
        dead_mask = 1.0 - alive_mask             # [B, 4]

        # Feature map coordinates (5x5 spatial grid)
        crash_fy = crash_coords[:, :, 0] * 5.0  # [B, 4]
        crash_fx = crash_coords[:, :, 1] * 5.0  # [B, 4]

        grid_y = torch.linspace(0.5, 4.5, 5, device=device).view(1, 1, 5, 1)
        grid_x = torch.linspace(0.5, 4.5, 5, device=device).view(1, 1, 1, 5)

        fy = crash_fy.view(B, 4, 1, 1)
        fx = crash_fx.view(B, 4, 1, 1)

        dist_sq = (grid_y - fy) ** 2 + (grid_x - fx) ** 2
        gauss = torch.exp(-dist_sq / 2.88)  # sigma = 1.2 -> 2*sigma^2 = 2.88
        masked_gauss = gauss * dead_mask.view(B, 4, 1, 1)
        m_crash = masked_gauss.sum(dim=1, keepdim=True)  # [B, 1, 5, 5]

        concat_feat = torch.cat([cnn_features, m_crash], dim=1)  # [B, 33, 5, 5]
        attn_map = torch.sigmoid(self.cssa_conv(concat_feat))    # [B, 32, 5, 5]

        # Residual attention combination (identity when no crash active)
        cnn_attended = cnn_features + crash_active.view(B, 1, 1, 1) * (cnn_features * attn_map)
        return cnn_attended

    def forward(self, input_dict, state, seq_lens):
        input_positions = input_dict["obs"][0].float()
        input_matrix = input_dict["obs"][1].float().unsqueeze(1)

        # 1. Extract Fleet Health State Vector & FHE
        fleet_health_raw, alive_mask, crash_active = self._extract_fleet_health_state(input_positions)
        h_t = self.fhe_encoder(fleet_health_raw)  # [B, 32]

        # 2. CNN Backbone & CSSA Attention
        cnn_f1 = self.cnn_layer1(input_matrix)
        cnn_f2 = self.cnn_layer2(cnn_f1)           # [B, 32, 5, 5]
        cnn_attended = self.compute_cssa_attention_if_needed(cnn_f2, input_positions, alive_mask, crash_active)
        
        cnn_flat = self.cnn_flatten(cnn_attended)
        cnn_out = self.cnn_project(cnn_flat)      # [B, 256]

        # 3. Linear Local Obs Encoder
        linear_out = self.linear_obs(input_positions) # [B, 256]

        # Combine all features for Actor & Critic
        combined_features = torch.cat((cnn_out, linear_out, h_t), dim=1)

        # 4. Policy Network
        actor_features = self.actor_join(combined_features)
        action_logits = self.policy_fn(actor_features)

        # 5. Dual-Objective Critic (DOVD)
        critic_features = self.critic_join(combined_features)
        v_search = self.v_search_head(critic_features)
        v_takeover = self.v_takeover_head(critic_features)
        alpha_t = self.alpha_head(critic_features)

        # Total Value Output
        self._value_out = v_search + alpha_t * v_takeover

        # 6. State-Conditioned Adaptive Clip (RAC)
        eps_t = 0.1 + 0.2 * self.rac_gate(h_t)
        self._current_eps_t = eps_t.squeeze(1)

        # Store for custom_loss auxiliary supervision
        self._stored_alpha_t = alpha_t.squeeze(1)
        self._stored_v_search = v_search.squeeze(1)
        self._stored_v_takeover = v_takeover.squeeze(1)
        self._stored_crash_active = crash_active.squeeze(1)

        return action_logits, state

    def compute_cssa_attention_if_needed(self, cnn_features, input_positions, alive_mask, crash_active):
        return self._compute_cssa_attention(cnn_features, input_positions, alive_mask, crash_active)

    def value_function(self):
        return self._value_out.flatten()

    def get_adaptive_clip(self):
        return self._current_eps_t

    def custom_loss(self, total_loss, train_batch):
        """
        DOVD Auxiliary Supervision Losses:
        1. Alpha Gate Alignment Loss (MSE with crash_active)
        2. Value Head Decorrelation Loss
        3. Takeover Sparsity Loss (when no crash active)
        """
        if isinstance(total_loss, list):
            primary_loss = total_loss[0]
        else:
            primary_loss = total_loss

        if self._stored_alpha_t is None or self._stored_alpha_t.device != primary_loss.device:
            return total_loss

        alpha = self._stored_alpha_t
        vs = self._stored_v_search
        vt = self._stored_v_takeover
        crash_act = self._stored_crash_active

        # 1. Alpha Alignment
        l_alpha = torch.mean((alpha - crash_act) ** 2)

        # 2. Value Head Decorrelation
        vs_centered = vs - vs.mean()
        vt_centered = vt - vt.mean()
        cov = torch.mean(vs_centered * vt_centered)
        var_vs = torch.var(vs, unbiased=False) + 1e-5
        var_vt = torch.var(vt, unbiased=False) + 1e-5
        corr = cov / torch.sqrt(var_vs * var_vt)
        l_decorr = corr ** 2

        # 3. Takeover Sparsity when healthy
        l_sparse = torch.mean((1.0 - crash_act) * (vt ** 2))

        l_aux = 0.1 * l_alpha + 0.01 * l_decorr + 0.005 * l_sparse

        self.tower_stats_extra = {
            "aux_alpha_loss": l_alpha,
            "aux_decorr_loss": l_decorr,
            "aux_sparse_loss": l_sparse,
            "mean_alpha_t": alpha.mean(),
            "mean_eps_t": self._current_eps_t.mean() if self._current_eps_t is not None else torch.tensor(0.2, device=primary_loss.device),
        }

        if isinstance(total_loss, list):
            return [l + l_aux for l in total_loss]
        else:
            return total_loss + l_aux


def env_creator(args):
    print("-------------------------- RSPO V2 VANILLA ENV CREATOR --------------------------")
    N_AGENTS = 4
    matrix_path = os.path.join(os.path.dirname(__file__), "uniform_matrix_25.npy")
    env = CoverageDroneSwarmSearch(
        timestep_limit=750,
        drone_amount=N_AGENTS,
        prob_matrix_path=matrix_path
    )
    env.reward_scheme = {
        "default": -0.1,
        "exceed_timestep": 0.0,
        "search_cell": 5.0,
        "done": 500.0,
        "reward_poc": 0.0,
    }
    env = AllPositionsWrapper(env)
    
    # RSPO V2 VANILLA: Fault injection ON, but NO teammate compensation rewards!
    env = BatteryStationWrapper(
        env,
        max_battery=125,
        depletion_rate=1,
        charge_rate=15,
        fault_prob=0.0005
    )
    env.COMPENSATION_BONUS = 0.0
    env.COMPENSATION_PENALTY = 0.0
    env.COMPENSATION_HORIZON = 0

    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    env = RetainDronePosWrapper(env, positions)
    return env


if __name__ == "__main__":
    ray.init()

    env_name = "DSSE_Coverage_RSPO_V2_Vanilla"

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    ModelCatalog.register_custom_model("RSPOModelV2_Vanilla", RSPOModelV2)
    register_trainable("RSPOPPO", RSPOPPO)

    num_gpus = 1 if torch.cuda.is_available() else 0
    num_runners = 24 if torch.cuda.is_available() else 6
    print(f"[RSPO V2 Vanilla] Configured with {num_gpus} GPUs and {num_runners} environment runners.")

    config = (
        RSPOPPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .env_runners(num_env_runners=num_runners, rollout_fragment_length="auto")
        .training(
            train_batch_size=8192,
            lr=1e-4,
            gamma=0.998,
            lambda_=0.9,
            use_gae=True,
            entropy_coeff=0.05,
            vf_clip_param=100000,
            minibatch_size=300,
            num_sgd_iter=10,
            model={
                "custom_model": "RSPOModelV2_Vanilla",
                "_disable_preprocessor_api": True,
            },
        )
        .callbacks(RSPOBatteryMetricsCallback)
        .experimental(_disable_preprocessor_api=True)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=num_gpus)
    )

    import sys
    exp_name = os.environ.get("EXP_NAME", "")
    if not exp_name:
        if sys.stdin.isatty():
            try:
                exp_name = input("Exp name for RSPO V2 Vanilla run: ")
            except (EOFError, KeyboardInterrupt):
                exp_name = "v2_vanilla"
        else:
            exp_name = "v2_vanilla"
    if not exp_name:
        exp_name = "v2_vanilla"

    curr_path = pathlib.Path().resolve()
    tune.run(
        "RSPOPPO",
        name="RSPO_V2_Vanilla_" + exp_name,
        stop={"timesteps_total": 20_000_000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path=f"{curr_path}/ray_res/DSSE_Coverage",
        config=config.to_dict(),
    )
