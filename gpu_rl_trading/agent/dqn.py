"""
gpu_rl_trading/agent/dqn.py
PyTorch DQN agent for GPU-centric FTMO trading.

Transfer-learning compatible: checkpoints store state_dim so partial loads
work automatically when state_dim grows (new features) or rewards change.
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
        self.state_dim   = state_dim
        self.num_actions = num_actions
        self.hidden      = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def load_partial(self, state_dict: dict, old_state_dim: int):
        """
        Load weights from a checkpoint with a different (smaller) state_dim.

        The first layer expands from (old_state_dim, hidden) to (state_dim, hidden).
        Old feature weights are preserved exactly. New feature columns are
        zero-initialised so they start neutral and are learned from scratch.
        All subsequent layers (hidden→hidden/2→actions) load exactly as-is.

        Args:
            state_dict   : the checkpoint's q_net or target_net state_dict
            old_state_dim: state_dim stored in the checkpoint
        """
        own = self.state_dict()

        for name, param in state_dict.items():
            if name not in own:
                continue

            if name == "net.0.weight":
                # shape: (hidden, state_dim)  — expand along dim=1
                new_w = own[name].clone()          # (hidden, new_state_dim)
                cols  = min(old_state_dim, new_w.shape[1])
                new_w[:, :cols] = param[:, :cols]  # copy old columns
                # new columns stay zero-initialised
                own[name] = new_w

            elif name == "net.0.bias":
                own[name] = param                  # bias shape unchanged

            else:
                # all deeper layers: load directly (shapes are identical)
                if own[name].shape == param.shape:
                    own[name] = param
                else:
                    print(f"[transfer] skipping {name}: "
                          f"shape {param.shape} != {own[name].shape}", flush=True)

        self.load_state_dict(own)


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
        # Exponential decay: epsilon halves roughly every 100 episodes
        # eps = start * (min/start)^(episode / decay_episodes)
        # decay_episodes = 500 gives: ep0=0.9, ep100=~0.72, ep300=~0.46, ep500=0.05
        decay_episodes = self.cfg.get("EPSILON_DECAY_EPISODES", 500)
        eps_min = self.cfg.get("EPSILON_MIN", self.eps_min)
        ratio = eps_min / (self.cfg["EPSILON_START"] + 1e-8)
        self.epsilon = max(
            self.eps_min,
            self.cfg["EPSILON_START"] * (ratio ** (episode / decay_episodes))
        )

    def save(self, path: str):
        """Save checkpoint. Always stores state_dim so partial loads work."""
        torch.save({
            "q_net":      self.q_net.state_dict(),
            "target":     self.target_net.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "epsilon":    self.epsilon,
            "step":       self._step_count,
            "state_dim":  self.state_dim,    # ← stored for transfer learning
        }, path)
        print(f"[ckpt] saved -> {path}", flush=True)

    def load(self, path: str, partial: bool = False):
        """
        Load checkpoint.

        Args:
            path    : path to .pt file
            partial : if True, use transfer learning when state_dim differs.
                      Old feature weights are preserved; new columns zero-init.
                      If False (default), standard exact load — raises on mismatch.
        """
        ckpt = torch.load(path, map_location=self.device)
        ckpt_state_dim = ckpt.get("state_dim", self.state_dim)

        if partial and ckpt_state_dim != self.state_dim:
            print(f"[transfer] state_dim {ckpt_state_dim} → {self.state_dim} "
                  f"(+{self.state_dim - ckpt_state_dim} features)", flush=True)
            self.q_net.load_partial(ckpt["q_net"], ckpt_state_dim)
            self.target_net.load_partial(ckpt["target"], ckpt_state_dim)
            # rebuild optimizer for the new network parameters
            self.optimizer = torch.optim.Adam(
                self.q_net.parameters(), lr=self.cfg["LR"])
            print("[transfer] optimizer reset for new architecture", flush=True)
        else:
            self.q_net.load_state_dict(ckpt["q_net"])
            self.target_net.load_state_dict(ckpt["target"])
            self.optimizer.load_state_dict(ckpt["optimizer"])

        self.epsilon     = ckpt.get("epsilon", self.cfg["EPSILON_START"])
        self._step_count = ckpt.get("step", 0)

        mode = "partial/transfer" if (partial and ckpt_state_dim != self.state_dim) else "exact"
        print(f"[ckpt] loaded ({mode}) <- {path}", flush=True)
