"""
RSPO V2 Policy and Algorithm Definition
=======================================
Implements RSPOTorchPolicy (subclassing PPOTorchPolicy) for Resilience-Adaptive Clipping (RAC)
and auxiliary DOVD loss integration, registered with RSPOPPO algorithm.
"""

from typing import Dict, List, Type, Union

import torch
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule, KLCoeffMixin
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import explained_variance, sequence_mask, warn_if_infinite_kl_divergence
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class RSPOTorchPolicy(PPOTorchPolicy):
    """
    RSPO Policy extending PPOTorchPolicy to enable:
    1. Resilience-Adaptive Clipping (RAC): Reads per-sample dynamic eps_t from model.get_adaptive_clip()
    2. Custom Loss Integration: Calls model.custom_loss() to apply DOVD auxiliary losses
    """

    @override(PPOTorchPolicy)
    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        """Compute loss for RSPO PPO with RAC dynamic clipping and DOVD auxiliary losses."""

        logits, state = model(train_batch)
        curr_action_dist = dist_class(logits, model)

        # RNN sequence masking if applicable
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid
        else:
            mask = None
            reduce_mean_valid = torch.mean

        prev_action_dist = dist_class(
            train_batch[SampleBatch.ACTION_DIST_INPUTS], model
        )

        logp_ratio = torch.exp(
            curr_action_dist.logp(train_batch[SampleBatch.ACTIONS])
            - train_batch[SampleBatch.ACTION_LOGP]
        )

        # KL divergence loss calculation
        if self.config["kl_coeff"] > 0.0:
            action_kl = prev_action_dist.kl(curr_action_dist)
            mean_kl_loss = reduce_mean_valid(action_kl)
            warn_if_infinite_kl_divergence(self, mean_kl_loss)
        else:
            mean_kl_loss = torch.tensor(0.0, device=logp_ratio.device)

        curr_entropy = curr_action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        # ── RAC (Resilience-Adaptive Clipping) ──
        # Reads per-sample adaptive clipping eps_t from FHE if available
        if hasattr(model, "get_adaptive_clip"):
            eps_t = model.get_adaptive_clip()
            if isinstance(eps_t, torch.Tensor) and eps_t.dim() > 0:
                clip_low = 1.0 - eps_t
                clip_high = 1.0 + eps_t
            else:
                clip_low = 1.0 - self.config["clip_param"]
                clip_high = 1.0 + self.config["clip_param"]

            surrogate_loss = torch.min(
                train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
                train_batch[Postprocessing.ADVANTAGES]
                * torch.clamp(logp_ratio, clip_low, clip_high),
            )
        else:
            surrogate_loss = torch.min(
                train_batch[Postprocessing.ADVANTAGES] * logp_ratio,
                train_batch[Postprocessing.ADVANTAGES]
                * torch.clamp(
                    logp_ratio,
                    1 - self.config["clip_param"],
                    1 + self.config["clip_param"],
                ),
            )

        # Value function loss
        if self.config["use_critic"]:
            value_fn_out = model.value_function()
            vf_loss = torch.pow(
                value_fn_out - train_batch[Postprocessing.VALUE_TARGETS], 2.0
            )
            vf_loss_clipped = torch.clamp(vf_loss, 0, self.config["vf_clip_param"])
            mean_vf_loss = reduce_mean_valid(vf_loss_clipped)
        else:
            value_fn_out = torch.tensor(0.0).to(surrogate_loss.device)
            vf_loss_clipped = mean_vf_loss = torch.tensor(0.0).to(surrogate_loss.device)

        total_loss = reduce_mean_valid(
            -surrogate_loss
            + self.config["vf_loss_coeff"] * vf_loss_clipped
            - self.entropy_coeff * curr_entropy
        )

        if self.config["kl_coeff"] > 0.0:
            total_loss += self.kl_coeff * mean_kl_loss

        # ── DOVD Auxiliary Loss Integration ──
        if hasattr(model, "custom_loss"):
            total_loss = model.custom_loss(total_loss, train_batch)

        # Store stats for logging
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_policy_loss"] = reduce_mean_valid(-surrogate_loss)
        model.tower_stats["mean_vf_loss"] = mean_vf_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], value_fn_out
        )
        model.tower_stats["mean_entropy"] = mean_entropy
        model.tower_stats["mean_kl_loss"] = mean_kl_loss

        if hasattr(model, "tower_stats_extra"):
            for k, v in model.tower_stats_extra.items():
                model.tower_stats[k] = v

        return total_loss


class RSPOPPOConfig(PPOConfig):
    """Configuration class for RSPOPPO algorithm."""

    def get_default_policy_class(self, config):
        return RSPOTorchPolicy


class RSPOPPO(PPO):
    """PPO Algorithm configured with RSPOTorchPolicy for RSPO V2."""

    @classmethod
    def get_default_config(cls):
        return RSPOPPOConfig()
