"""Tests for gpu_rl_trading.env.environment."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
import torch
from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import BatchedFTMOEnv, NUM_ACTIONS


class TestBatchedFTMOEnv:
    @pytest.fixture
    def setup(self):
        """Create a small synthetic environment for testing."""
        cfg = CFG.copy()
        cfg["BATCH_SIZE_ENV"] = 2
        cfg["PHASE"] = 7
        cfg["EPISODE_BARS"] = 500
        
        T = 2000
        ohlcv = (np.random.rand(T, 5).astype(np.float32) + 1.0)
        features = build_feature_matrix(
            ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3], ohlcv[:, 4]
        )
        
        device = torch.device("cpu")
        env = BatchedFTMOEnv(features, cfg, device)
        return env, cfg, device
    
    def test_env_initialization(self, setup):
        env, cfg, device = setup
        assert env.B == 2
        assert env.state_dim == 2166  # based on NUM_FEATURES=27, LOOKBACK=20, 4 TFs, 6 position features
        assert env.phase == 7
    
    def test_reset_returns_state(self, setup):
        env, cfg, device = setup
        state = env.reset()
        assert state.shape == (2, env.state_dim)
        assert state.dtype == torch.float32
    
    def test_step_outputs(self, setup):
        env, cfg, device = setup
        state = env.reset()
        actions = torch.randint(0, NUM_ACTIONS, (2,), device=device)
        next_state, rewards, dones, exec_act = env.step(actions)
        
        assert next_state.shape == (2, env.state_dim)
        assert rewards.shape == (2,)
        assert dones.shape == (2,)
        assert rewards.dtype == torch.float32
        assert dones.dtype == torch.bool
    
    def test_episode_runs_to_completion(self, setup):
        env, cfg, device = setup
        state = env.reset()
        dones = torch.zeros(2, dtype=torch.bool, device=device)
        
        step_count = 0
        max_steps = env.ep_bars + 100  # allow some margin
        
        while not dones.all() and step_count < max_steps:
            actions = torch.randint(0, NUM_ACTIONS, (2,), device=device)
            state, _, dones, _exec = env.step(actions)
            step_count += 1
        
        assert dones.all(), "Episode did not complete within expected steps"
        assert step_count <= max_steps
