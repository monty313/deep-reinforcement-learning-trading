"""
gpu_rl_trading/agent/dqn.py
PyTorch DQN agent for GPU-centric FTMO trading.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

from gpu_rl_trading.agent.replay import GPUReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim: int, num_actions: int, cfg: dict, device: torch.device):
        self.state_dim   = state_dim
        self.num_actions = num_actions
        self.cfg         = cfg
        self.device      = device
        self.gamma       = cfg["GAMMA"]
        self.epsilon     = cfg["EPSILON_START"]
        self.eps_min     = cfg["EPSILON_MIN"]
        self.batch_size  = cfg["BATCH_SIZE_RL"]
        self.train_every = cfg["TRAIN_EVERY"]
        self.sync_every  = cfg["SYNC_EVERY"]
        self._step_count = 0

        hidden = cfg.get("HIDDEN", 256)
        self.q_net      = QNetwork(state_dim, num_actions, hidden).to(device)
        self.target_net = QNetwork(state_dim, num_actions, hidden).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg["LR"])
        self.memory    = GPUReplayBuffer(cfg["MEMORY_SIZE"], state_dim, device)

    @torch.no_grad()
    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Epsilon-greedy. state: (B, state_dim) -> actions: (B,)."""
        B = state.shape[0]
        if torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.num_actions, (B,), device=self.device)
        q = self.q_net(state)
        return q.argmax(dim=1)

    def store(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        self._step_count += 1
        if self._step_count % self.train_every != 0:
            return None

        s, a, r, ns, d = self.memory.sample(self.batch_size)
        with torch.no_grad():
            q_next  = self.target_net(ns).max(dim=1).values
            targets = r + self.gamma * q_next * (~d)

        q_pred = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        loss   = F.mse_loss(q_pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self._step_count % self.sync_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def decay_epsilon(self, episode: int):
        self.epsilon = max(self.eps_min,
                           self.cfg["EPSILON_START"] ** (1 + episode / 50))

    def save(self, path: str):
        torch.save({
            "q_net":     self.q_net.state_dict(),
            "target":    self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon":   self.epsilon,
            "step":      self._step_count,
        }, path)
        print(f"[ckpt] saved -> {path}", flush=True)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon     = ckpt.get("epsilon", self.cfg["EPSILON_START"])
        self._step_count = ckpt.get("step", 0)
        print(f"[ckpt] loaded <- {path}", flush=True)
