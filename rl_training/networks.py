#!/usr/bin/env python3
"""
networks.py — Neural networks for DQN maze solver.

QNetwork:  state → Q-value per discrete action (7 outputs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_training.config import STATE_DIM, ACTION_DIM, HIDDEN_1, HIDDEN_2


class QNetwork(nn.Module):
    """
    Dueling DQN Q-network.

    Splits into value stream V(s) and advantage stream A(s,a),
    then combines: Q(s,a) = V(s) + A(s,a) - mean(A(s,:)).

    Architecture:
      state_dim → 512 → 256 → split:
        → 256 → 1           (value stream)
        → 256 → n_actions   (advantage stream)
    """

    def __init__(self, state_dim: int = STATE_DIM,
                 n_actions: int = ACTION_DIM):
        super().__init__()
        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(HIDDEN_2, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(HIDDEN_2, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, n_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions, shape (batch, n_actions)."""
        feat = self.feature(state)
        v = self.value(feat)                    # (batch, 1)
        a = self.advantage(feat)                # (batch, n_actions)
        # Dueling combination: Q = V + (A - mean(A))
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
