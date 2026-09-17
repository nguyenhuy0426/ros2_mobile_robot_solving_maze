#!/usr/bin/env python3
"""
dqn.py — Double DQN algorithm with Dueling architecture.

Features:
  1. Double DQN: online network selects action, target network evaluates Q-value
  2. Dueling architecture in QNetwork (value + advantage streams)
  3. Soft target network updates (Polyak averaging)
  4. Epsilon-greedy exploration with decay
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl_training.networks import QNetwork
from rl_training.replay_buffer import ReplayBuffer
from rl_training import config as C


class DQN:
    """Double DQN with Dueling architecture."""

    def __init__(self,
                 device: torch.device = None,
                 load_model: bool = False,
                 load_path: Path = None):

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Q-networks
        self.q_net = QNetwork(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.q_target = QNetwork(C.STATE_DIM, C.ACTION_DIM).to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=C.DQN_LR)

        # Exploration
        self.epsilon = C.DQN_EPS_START

        self.iter_count = 0
        self.writer = SummaryWriter(log_dir=str(C.TB_DIR))

        if load_model and load_path:
            self.load(load_path)

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())

    @property
    def actor(self):
        """Compatibility alias for code that accesses model.actor."""
        return self.q_net

    def get_action(self, state: np.ndarray,
                   add_noise: bool = True) -> int:
        """
        Epsilon-greedy action selection.

        Args:
            state: observation array of shape (STATE_DIM,)
            add_noise: if True, use epsilon-greedy; if False, greedy (deployment)

        Returns:
            action index (int) in [0, N_ACTIONS)
        """
        if add_noise and np.random.random() < self.epsilon:
            return np.random.randint(0, C.N_ACTIONS)

        obs = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs)
        return int(q_values.argmax(dim=1).item())

    def decay_epsilon(self):
        """Call once per episode to decay exploration."""
        self.epsilon = max(C.DQN_EPS_END,
                          self.epsilon * C.DQN_EPS_DECAY)

    def train(self, replay_buffer: ReplayBuffer,
              iterations: int = C.TRAIN_ITERATIONS,
              batch_size: int = C.BATCH_SIZE):
        """
        Run `iterations` gradient steps of Double DQN training.
        """
        av_loss = 0.0
        av_Q = 0.0
        max_Q = -float('inf')

        for it in range(iterations):
            # Sample batch — actions come as (batch, 1) integers
            (batch_s, batch_a, batch_r,
             batch_d, batch_ns) = replay_buffer.sample(batch_size)

            state = torch.FloatTensor(batch_s).to(self.device)
            next_state = torch.FloatTensor(batch_ns).to(self.device)
            action = torch.LongTensor(batch_a).to(self.device)     # (batch, 1)
            reward = torch.FloatTensor(batch_r).to(self.device)    # (batch, 1)
            done = torch.FloatTensor(batch_d).to(self.device)      # (batch, 1)

            # Ensure action shape is (batch, 1) for gather
            if action.dim() == 1:
                action = action.unsqueeze(1)

            # Current Q-values for taken actions
            current_q = self.q_net(state).gather(1, action)  # (batch, 1)

            # Double DQN target
            with torch.no_grad():
                # Online network selects best action
                next_q_online = self.q_net(next_state)
                best_next_action = next_q_online.argmax(dim=1, keepdim=True)
                # Target network evaluates that action
                next_q_target = self.q_target(next_state)
                next_q = next_q_target.gather(1, best_next_action)
                target_q = reward + (1 - done) * C.DQN_DISCOUNT * next_q

            loss = F.smooth_l1_loss(current_q, target_q)  # Huber loss

            self.optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
            self.optimizer.step()

            # Soft update target network
            for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                tp.data.copy_(C.DQN_TAU * p.data + (1 - C.DQN_TAU) * tp.data)

            av_loss += loss.item()
            av_Q += current_q.mean().item()
            max_Q = max(max_Q, current_q.max().item())

        # Logging
        self.iter_count += 1
        self.writer.add_scalar("train/loss", av_loss / iterations, self.iter_count)
        self.writer.add_scalar("train/avg_Q", av_Q / iterations, self.iter_count)
        self.writer.add_scalar("train/max_Q", max_Q, self.iter_count)
        self.writer.add_scalar("train/epsilon", self.epsilon, self.iter_count)

        if C.SAVE_EVERY > 0 and self.iter_count % C.SAVE_EVERY == 0:
            self.save(C.CKPT_DIR / f"dqn_iter_{self.iter_count}")

    def save(self, path: Path):
        """Save Q-network and target network weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path / "q_net.pth")
        torch.save(self.q_target.state_dict(), path / "q_target.pth")
        torch.save({"epsilon": self.epsilon}, path / "meta.pth")

    def load(self, path: Path):
        """Load Q-network and target network weights."""
        path = Path(path)
        self.q_net.load_state_dict(
            torch.load(path / "q_net.pth", map_location=self.device))
        self.q_target.load_state_dict(
            torch.load(path / "q_target.pth", map_location=self.device))
        meta = torch.load(path / "meta.pth", map_location=self.device)
        self.epsilon = meta.get("epsilon", C.DQN_EPS_END)
        print(f"  Loaded DQN weights from {path} (eps={self.epsilon:.4f})")

    def load_actor_only(self, path: Path):
        """Load only Q-network (for deployment). Sets greedy mode."""
        path = Path(path)
        self.q_net.load_state_dict(
            torch.load(path / "q_net.pth", map_location=self.device))
        self.epsilon = 0.0
        print(f"  Loaded DQN Q-network from {path}")

    def perturb_weights(self, noise_scale: float = 0.01):
        """Add small Gaussian noise to Q-network weights (GA mutation)."""
        for param in self.q_net.parameters():
            param.data += torch.randn_like(param.data) * noise_scale
