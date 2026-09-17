#!/usr/bin/env python3
"""
td3.py — Twin Delayed DDPG (TD3) algorithm.

Adapted from DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/TD3/TD3.py

Key features:
  1. Twin Q-networks → reduces Q-value overestimation
  2. Delayed policy updates → actor updated every 2 critic updates
  3. Target policy smoothing → noise added to target actions
  4. Soft target updates → Polyak averaging

Designed as an abstraction class so SAC can be added later with
the same interface: get_action(), train(), save(), load().
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from rl_training.networks import Actor, Critic
from rl_training.replay_buffer import ReplayBuffer
from rl_training import config as C


class TD3:
    """Twin Delayed DDPG algorithm."""

    def __init__(self,
                 device: torch.device = None,
                 load_model: bool = False,
                 load_path: Path = None):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Actor networks
        self.actor = Actor(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.actor_target = Actor(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=C.TD3_LR)

        # Critic networks
        self.critic = Critic(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.critic_target = Critic(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=C.TD3_LR)

        self.max_action = C.MAX_ACTION
        self.iter_count = 0
        self.writer = SummaryWriter(log_dir=str(C.TB_DIR))

        if load_model and load_path:
            self.load(load_path)

    @property
    def total_params(self) -> int:
        return (sum(p.numel() for p in self.actor.parameters()) +
                sum(p.numel() for p in self.critic.parameters()))

    def get_action(self, state: np.ndarray,
                   add_noise: bool = True) -> np.ndarray:
        """
        Get action from actor, optionally with exploration noise.

        Args:
            state: observation vector (STATE_DIM,)
            add_noise: add Gaussian noise for exploration
        Returns:
            action: clipped to [-MAX_ACTION, MAX_ACTION]
        """
        action = self._act(state)
        if add_noise:
            noise = np.random.normal(0, C.EXPLORE_NOISE, size=C.ACTION_DIM)
            action = (action + noise).clip(-self.max_action, self.max_action)
        return action

    def _act(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through actor (no noise)."""
        with torch.no_grad():
            s = torch.FloatTensor(state).to(self.device)
            return self.actor(s).cpu().numpy().flatten()

    def train(self, replay_buffer: ReplayBuffer,
              iterations: int = C.TRAIN_ITERATIONS,
              batch_size: int = C.BATCH_SIZE):
        """
        Run `iterations` gradient steps of TD3 training.

        Args:
            replay_buffer: experience buffer to sample from
            iterations: number of gradient updates
            batch_size: mini-batch size
        """
        av_Q = 0.0
        max_Q = -inf
        av_loss = 0.0

        for it in range(iterations):
            # Sample batch
            (batch_s, batch_a, batch_r,
             batch_d, batch_ns) = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(batch_s).to(self.device)
            next_state = torch.FloatTensor(batch_ns).to(self.device)
            action = torch.FloatTensor(batch_a).to(self.device)
            reward = torch.FloatTensor(batch_r).to(self.device)
            done = torch.FloatTensor(batch_d).to(self.device)

            # ── Critic update ──────────────────────────────────
            # Target policy smoothing: add clipped noise to target action
            with torch.no_grad():
                next_action = self.actor_target(next_state)
                noise = (torch.randn_like(next_action) * C.TD3_POLICY_NOISE
                         ).clamp(-C.TD3_NOISE_CLIP, C.TD3_NOISE_CLIP)
                next_action = (next_action + noise).clamp(
                    -self.max_action, self.max_action)

                # Twin Q-target: take the min
                target_Q1, target_Q2 = self.critic_target(
                    next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = reward + (1 - done) * C.TD3_DISCOUNT * target_Q

            # Current Q estimates
            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = (F.mse_loss(current_Q1, target_Q) +
                           F.mse_loss(current_Q2, target_Q))

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())
            av_loss += critic_loss.item()

            # ── Delayed actor update ───────────────────────────
            if it % C.TD3_POLICY_FREQ == 0:
                # Actor loss: maximize Q1(s, actor(s))
                actor_loss = -self.critic.q1_forward(
                    state, self.actor(state)).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update targets
                for p, tp in zip(self.actor.parameters(),
                                 self.actor_target.parameters()):
                    tp.data.copy_(
                        C.TD3_TAU * p.data + (1 - C.TD3_TAU) * tp.data)

                for p, tp in zip(self.critic.parameters(),
                                 self.critic_target.parameters()):
                    tp.data.copy_(
                        C.TD3_TAU * p.data + (1 - C.TD3_TAU) * tp.data)

        # Logging
        self.iter_count += 1
        self.writer.add_scalar("train/critic_loss",
                               av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Q",
                               av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q",
                               max_Q, self.iter_count)

        if C.SAVE_EVERY > 0 and self.iter_count % C.SAVE_EVERY == 0:
            self.save(C.CKPT_DIR / f"td3_iter_{self.iter_count}")

    def save(self, path: Path):
        """Save actor and critic weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")
        torch.save(self.actor_target.state_dict(),
                   path / "actor_target.pth")
        torch.save(self.critic_target.state_dict(),
                   path / "critic_target.pth")

    def load(self, path: Path):
        """Load actor and critic weights."""
        path = Path(path)
        self.actor.load_state_dict(
            torch.load(path / "actor.pth", map_location=self.device))
        self.critic.load_state_dict(
            torch.load(path / "critic.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(path / "actor_target.pth", map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(path / "critic_target.pth", map_location=self.device))
        print(f"  Loaded TD3 weights from {path}")

    def load_actor_only(self, path: Path):
        """Load only the actor (for deployment)."""
        path = Path(path)
        self.actor.load_state_dict(
            torch.load(path / "actor.pth", map_location=self.device))
        print(f"  Loaded actor from {path}")

    def perturb_weights(self, noise_scale: float = 0.01):
        """Add small Gaussian noise to actor weights to break rollback loops (GA mutation)."""
        for param in self.actor.parameters():
            param.data += torch.randn_like(param.data) * noise_scale
        for param in self.actor_target.parameters():
            param.data += torch.randn_like(param.data) * noise_scale
