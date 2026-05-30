"""
Minimal smoke test for gpu_rl_trading env + agent.
Run with: python smoke_test.py
"""
import numpy as np
import torch
from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import BatchedFTMOEnv, NUM_ACTIONS
from gpu_rl_trading.agent.dqn import DQNAgent

cfg = CFG.copy()
cfg["BATCH_SIZE_ENV"] = 2
cfg["PHASE"] = 7
cfg["EPISODE_BARS"] = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# synthetic OHLCV
T = 2000
ohlcv = (np.random.rand(T, 5).astype(np.float32) + 1.0)
features = build_feature_matrix(ohlcv[:,0], ohlcv[:,1], ohlcv[:,2], ohlcv[:,3], ohlcv[:,4])

env = BatchedFTMOEnv(features, cfg, device)
agent = DQNAgent(env.state_dim, NUM_ACTIONS, cfg, device)

state = env.reset()
actions = agent.select_actions(state)
next_state, rewards, dones = env.step(actions)
print("next_state.shape:", next_state.shape)
print("rewards:", rewards)
print("dones:", dones)
print("SMOKE TEST OK")
