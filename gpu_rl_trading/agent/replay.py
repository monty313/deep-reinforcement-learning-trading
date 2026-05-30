"""
gpu_rl_trading/agent/replay.py
GPU replay buffer storing transitions directly in tensors.
"""
from __future__ import annotations

import torch


class GPUReplayBuffer:
    def __init__(self, capacity: int, state_dim: int, device: torch.device):
        self.capacity  = capacity
        self.device    = device
        self.ptr       = 0
        self.size      = 0

        self.states     = torch.zeros((capacity, state_dim), device=device)
        self.next_states = torch.zeros((capacity, state_dim), device=device)
        self.actions    = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rewards    = torch.zeros(capacity, device=device)
        self.dones      = torch.zeros(capacity, dtype=torch.bool, device=device)

    def push(self, state, action, reward, next_state, done):
        """Accept tensors of shape (B, state_dim) and scalars/tensors of shape (B,)."""
        B = state.shape[0]
        idx = torch.arange(self.ptr, self.ptr + B, device=self.device) % self.capacity
        self.states[idx]      = state.detach()
        self.next_states[idx] = next_state.detach()
        self.actions[idx]     = action.detach()
        self.rewards[idx]     = reward.detach()
        self.dones[idx]       = done.detach()
        self.ptr  = (self.ptr + B) % self.capacity
        self.size = min(self.size + B, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self.size
