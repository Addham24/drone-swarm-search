"""
VDN-Style Global Reward Wrapper
================================
Implements Value Decomposition Network (VDN) reward mixing for
multi-agent environments using PettingZoo's ParallelEnv API.

In VDN, the joint Q-value is the sum of individual Q-values:
    Q_tot = Q_1 + Q_2 + ... + Q_n

By distributing the global reward (R_tot = sum of all individual rewards)
equally to every agent, each independent DQN learner is forced to optimise
for the swarm's collective objective rather than its own selfish reward.

This effectively turns Independent DQN into a cooperative algorithm
that mimics the behaviour of a true mixing network (QMIX / VDN)
without requiring any changes to RLlib's training loop.

Reference:
    Sunehag et al., "Value-Decomposition Networks for Cooperative
    Multi-Agent Learning", AAMAS 2018.
"""

import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper


class GlobalRewardWrapper(BaseParallelWrapper):
    """
    Sums all agent rewards each timestep and distributes the total
    equally to every agent (VDN-style cooperative reward mixing).

    Args:
        env: The PettingZoo parallel environment to wrap.
        mixing_alpha: Blending factor between individual and global reward.
                      1.0 = pure global (full VDN), 0.0 = pure individual (I-DQN).
                      Default 0.7 provides a good balance: agents still get some
                      individual signal for personal battery management while
                      being strongly incentivised to cooperate.
    """

    def __init__(self, env, mixing_alpha=0.5):
        super().__init__(env)
        self.mixing_alpha = mixing_alpha

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)

        if rewards:
            # 1. Extract and isolate teammate compensation rewards
            base_rewards = {}
            comp_rewards = {}
            for agent in rewards:
                comp = 0.0
                if isinstance(infos.get(agent), dict):
                    comp = infos[agent].get("compensation_reward", 0.0)
                
                comp_rewards[agent] = comp
                base_rewards[agent] = rewards[agent] - comp

            # 2. Sum and average ONLY the cooperative base rewards (cell searches, completions)
            all_base = list(base_rewards.values())
            global_base = sum(all_base) / len(all_base)

            # 3. Blend base rewards and add back undiluted compensation rewards
            for agent in rewards:
                rewards[agent] = (
                    self.mixing_alpha * global_base
                    + (1.0 - self.mixing_alpha) * base_rewards[agent]
                    + comp_rewards[agent]
                )

        return obs, rewards, terminations, truncations, infos
