#!/usr/bin/env python3
"""
sac.py — Soft Actor-Critic (SAC) algorithm.

Adapted from DRL-Robot-Navigation-ROS2/src/drl_navigation_ros2/SAC/SAC.py

Key features:
  1. Maximum entropy reinforcement learning
  2. Stochastic policy (GaussianActor)
  3. Learnable temperature parameter (alpha)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_training.networks import GaussianActor, Critic
from rl_training.replay_buffer import ReplayBuffer
from rl_training import config as C


class SAC:
    """Soft Actor-Critic algorithm."""

    def __init__(self,
                 device: torch.device = None,
                 load_model: bool = False,
                 load_path: Path = None):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Actor network
        self.actor = GaussianActor(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=C.SAC_ACTOR_LR, betas=C.SAC_BETAS)

        # Critic networks
        self.critic = Critic(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.critic_target = Critic(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=C.SAC_CRITIC_LR, betas=C.SAC_BETAS)

        self.max_action = C.MAX_ACTION
        self.iter_count = 0
        self.writer = SummaryWriter(log_dir=str(C.TB_DIR))

        # Entropy temperature (alpha)
        self.log_alpha = torch.tensor(np.log(C.SAC_INIT_TEMP)).to(self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -C.ACTION_DIM  # heuristic target entropy
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=C.SAC_ALPHA_LR, betas=C.SAC_BETAS)

        if load_model and load_path:
            self.load(load_path)

    @property
    def total_params(self) -> int:
        return (sum(p.numel() for p in self.actor.parameters()) +
                sum(p.numel() for p in self.critic.parameters()))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action(self, state: np.ndarray,
                   add_noise: bool = True) -> np.ndarray:
        """
        Get action from actor. For SAC, add_noise means sampling from the
        stochastic policy. If False, we take the mean.
        """
        obs = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.actor(obs)
            action = dist.sample() if add_noise else dist.mean
            action = action.clamp(-self.max_action, self.max_action)
        return action[0].cpu().numpy()

    def train(self, replay_buffer: ReplayBuffer,
              iterations: int = C.TRAIN_ITERATIONS,
              batch_size: int = C.BATCH_SIZE):
        """
        Run `iterations` gradient steps of SAC training.
        """
        av_Q = 0.0
        max_Q = -float('inf')
        av_critic_loss = 0.0
        av_actor_loss = 0.0
        av_alpha_loss = 0.0

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
            with torch.no_grad():
                next_dist = self.actor(next_state)
                next_action = next_dist.rsample()
                log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)

                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
                target_Q = reward + (1 - done) * C.SAC_DISCOUNT * target_V

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # ── Actor update ───────────────────────────────────
            if it % C.SAC_ACTOR_FREQ == 0:
                dist = self.actor(state)
                pi_action = dist.rsample()
                log_prob = dist.log_prob(pi_action).sum(-1, keepdim=True)

                actor_Q1, actor_Q2 = self.critic(state, pi_action)
                actor_Q = torch.min(actor_Q1, actor_Q2)
                actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # ── Alpha update ───────────────────────────────
                if C.SAC_LEARN_TEMP:
                    self.alpha_optimizer.zero_grad()
                    alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
                    alpha_loss.backward()
                    self.alpha_optimizer.step()
                    av_alpha_loss += alpha_loss.item()

                av_actor_loss += actor_loss.item()

            # ── Target networks update ─────────────────────────
            if it % C.SAC_TARGET_FREQ == 0:
                for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                    tp.data.copy_(C.SAC_TAU * p.data + (1 - C.SAC_TAU) * tp.data)

            av_Q += torch.mean(target_Q).item()
            max_Q = max(max_Q, torch.max(target_Q).item())
            av_critic_loss += critic_loss.item()

        # Logging
        self.iter_count += 1
        self.writer.add_scalar("train/critic_loss", av_critic_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/actor_loss", av_actor_loss / (iterations / C.SAC_ACTOR_FREQ), self.iter_count)
        if C.SAC_LEARN_TEMP:
            self.writer.add_scalar("train/alpha_loss", av_alpha_loss / (iterations / C.SAC_ACTOR_FREQ), self.iter_count)
            self.writer.add_scalar("train/alpha", self.alpha.item(), self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)

        if C.SAVE_EVERY > 0 and self.iter_count % C.SAVE_EVERY == 0:
            self.save(C.CKPT_DIR / f"sac_iter_{self.iter_count}")

    def save(self, path: Path):
        """Save actor and critic weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), path / "actor.pth")
        torch.save(self.critic.state_dict(), path / "critic.pth")
        torch.save(self.critic_target.state_dict(), path / "critic_target.pth")
        torch.save(self.log_alpha, path / "log_alpha.pth")

    def load(self, path: Path):
        """Load actor and critic weights."""
        path = Path(path)
        self.actor.load_state_dict(torch.load(path / "actor.pth", map_location=self.device))
        self.critic.load_state_dict(torch.load(path / "critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(torch.load(path / "critic_target.pth", map_location=self.device))
        log_alpha = torch.load(path / "log_alpha.pth", map_location=self.device)
        self.log_alpha.data.copy_(log_alpha.data)
        print(f"  Loaded SAC weights from {path}")

    def load_actor_only(self, path: Path):
        """Load only the actor (for deployment)."""
        path = Path(path)
        self.actor.load_state_dict(torch.load(path / "actor.pth", map_location=self.device))
        print(f"  Loaded SAC actor from {path}")

    def perturb_weights(self, noise_scale: float = 0.01):
        """Add small Gaussian noise to actor weights to break rollback loops (GA mutation)."""
        for param in self.actor.parameters():
            param.data += torch.randn_like(param.data) * noise_scale
