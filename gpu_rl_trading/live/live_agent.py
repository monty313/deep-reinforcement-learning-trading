"""
gpu_rl_trading/live/live_agent.py

Python side of the live FTMO trading system.
Reads bar data written by the MT5 EA, runs the trained DQN model,
and writes the action back for the EA to execute.

HOW IT WORKS:
  MT5 EA writes a bar file  →  Python reads it, runs model  →  Python writes action file  →  EA reads and trades

SETUP (run this FIRST, before starting the EA):
  1. Copy your checkpoint .pt file somewhere on your PC
  2. Open a terminal in this project folder
  3. Run:
       python -m gpu_rl_trading.live.live_agent --checkpoint "C:/path/to/eurusd_gpu_ph7_final.pt"
  4. You will see: [agent] Ready. Waiting for bars from MT5...
  5. Then attach the FTMO_DQN EA to any chart in MT5

SHARED FILES (EA and Python use these to communicate):
  The EA writes to:   C:/MT5Bridge/bar_data.csv
  Python writes to:   C:/MT5Bridge/action.txt
  Python writes to:   C:/MT5Bridge/agent_ready.txt  (signals EA that Python is running)

All paths can be changed via --bridge-dir argument.
"""
from __future__ import annotations

import argparse
import time
import sys
from collections import deque
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from gpu_rl_trading.config.settings import CFG
from gpu_rl_trading.env.indicators import build_feature_matrix
from gpu_rl_trading.env.environment import NUM_ACTIONS
from gpu_rl_trading.agent.dqn import DQNAgent


# action names for logging
ACTION_NAMES = ["FLAT", "BUY_S", "BUY_M", "BUY_L", "SELL_S", "SELL_M", "SELL_L"]

# how many 1m bars we need before indicators are valid
# must match training warmup: max(TF_FACTORS)*2 + LOOKBACK + 200
MIN_BARS = 3100


def load_agent(checkpoint_path: str, device: torch.device) -> DQNAgent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dim     = ckpt.get("state_dim", None)
    if state_dim is None:
        raise ValueError("Checkpoint missing state_dim — was it saved by this codebase?")

    cfg = CFG.copy()
    cfg["STATE_DIM"]      = state_dim
    cfg["BATCH_SIZE_ENV"] = 1
    cfg["MEMORY_SIZE"]    = 1

    agent = DQNAgent(state_dim, NUM_ACTIONS, cfg, device)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.q_net.eval()
    agent.epsilon = 0.0
    print(f"[agent] Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"[agent] state_dim={state_dim}  epsilon=0.0 (greedy)", flush=True)
    return agent, state_dim


def bars_to_features(bar_buffer: deque) -> np.ndarray:
    """Convert the rolling bar buffer to a (T, 5) OHLCV array and build features."""
    arr = np.array(list(bar_buffer), dtype=np.float32)  # (T, 5)
    return build_feature_matrix(
        arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4])


def get_state(features: np.ndarray, cfg: dict, device: torch.device) -> torch.Tensor:
    """
    Build the state tensor from the last bar of the feature matrix.
    Mirrors SequentialBacktestEnv._get_state() but for a rolling buffer.
    Uses the last bar as current, lookback window for history.
    """
    lkbk       = cfg["LOOKBACK"]
    tf_factors = cfg["TF_FACTORS"]
    F          = features.shape[1]
    T          = features.shape[0]

    feat_t = torch.tensor(features, dtype=torch.float32, device=device)

    # precompute resampled tensors
    resampled = {}
    for tf in tf_factors:
        if tf == 1:
            resampled[tf] = feat_t
        else:
            idx = torch.arange(tf - 1, T, tf, device=device)
            if len(idx) == 0:
                idx = torch.tensor([0], device=device)
            resampled[tf] = feat_t[idx]

    abs_t   = torch.tensor(T - 1, device=device)
    parts   = []

    for tf in tf_factors:
        feat   = resampled[tf]
        tf_idx = (abs_t // tf).clamp(0, feat.shape[0] - 1)
        offsets = torch.arange(lkbk - 1, -1, -1, device=device)
        win_idx = (tf_idx - offsets).clamp(0, feat.shape[0] - 1)
        window  = feat[win_idx].unsqueeze(0)          # (1, lkbk, F)
        mu  = window.mean(dim=(1, 2), keepdim=True)
        std = window.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        parts.append(((window - mu) / std).reshape(1, -1))

    # FTMO position features — Python doesn't track position (EA does)
    # so we pass zeros here; the EA enforces hard stops independently
    pos_feat = torch.zeros((1, 6), dtype=torch.float32, device=device)
    parts.append(pos_feat)

    return torch.cat(parts, dim=1)   # (1, state_dim)


def run_agent(
    checkpoint_path: str,
    bridge_dir:      str  = "C:/MT5Bridge",
    poll_ms:         int  = 100,    # how often to check for new bar (milliseconds)
):
    bridge = Path(bridge_dir)
    bridge.mkdir(parents=True, exist_ok=True)

    bar_file    = bridge / "bar_data.csv"
    action_file = bridge / "action.txt"
    ready_file  = bridge / "agent_ready.txt"
    log_file    = bridge / "agent_log.txt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[agent] device={device}", flush=True)

    agent, state_dim = load_agent(checkpoint_path, device)

    cfg = CFG.copy()
    cfg["MEMORY_SIZE"] = 1
    cfg["BATCH_SIZE_ENV"] = 1

    # rolling bar buffer — keeps last MIN_BARS bars
    bar_buffer: deque = deque(maxlen=MIN_BARS + 100)

    # signal EA that Python is ready
    ready_file.write_text("ready")
    print(f"[agent] Ready. Waiting for bars from MT5 ...", flush=True)
    print(f"[agent] Bridge dir: {bridge}", flush=True)
    print(f"[agent] Watching:   {bar_file}", flush=True)

    last_bar_time = ""
    bars_received = 0

    while True:
        try:
            if not bar_file.exists():
                time.sleep(poll_ms / 1000)
                continue

            # read bar file written by EA
            # format: symbol,time,open,high,low,close,volume
            content = bar_file.read_text().strip()
            if not content or content == last_bar_time:
                time.sleep(poll_ms / 1000)
                continue

            parts = content.split(",")
            if len(parts) < 7:
                time.sleep(poll_ms / 1000)
                continue

            symbol   = parts[0]
            bar_time = parts[1]
            o, h, l, c, v = (float(x) for x in parts[2:7])

            # skip if same bar as last time
            if bar_time == last_bar_time:
                time.sleep(poll_ms / 1000)
                continue

            last_bar_time = bar_time
            bar_buffer.append([o, h, l, c, v])
            bars_received += 1

            if bars_received < MIN_BARS:
                # not enough bars yet — tell EA to stay flat
                action_file.write_text("0")
                if bars_received % 100 == 0:
                    print(f"[agent] Warming up: {bars_received}/{MIN_BARS} bars",
                          flush=True)
                continue

            # build features and get action
            features = bars_to_features(bar_buffer)
            state    = get_state(features, cfg, device)

            with torch.no_grad():
                q_vals = agent.q_net(state)
                action = int(q_vals.argmax(dim=1).item())

            # write action for EA to read
            action_file.write_text(str(action))

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] {symbol} bar={bar_time}  "
                  f"close={c:.5f}  action={action}({ACTION_NAMES[action]})",
                  flush=True)

            # append to log
            with open(log_file, "a") as f:
                f.write(f"{ts},{symbol},{bar_time},{c:.5f},{action},{ACTION_NAMES[action]}\n")

        except KeyboardInterrupt:
            print("\n[agent] Stopped by user.", flush=True)
            ready_file.unlink(missing_ok=True)
            action_file.write_text("0")
            break
        except Exception as e:
            print(f"[agent] ERROR: {e}", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live DQN agent for FTMO trading")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to trained .pt checkpoint")
    parser.add_argument("--bridge-dir", default="C:/MT5Bridge",
                        help="Shared folder for MT5 communication (default: C:/MT5Bridge)")
    parser.add_argument("--poll-ms", type=int, default=100,
                        help="Polling interval in ms (default: 100)")
    args = parser.parse_args()

    run_agent(
        checkpoint_path = args.checkpoint,
        bridge_dir      = args.bridge_dir,
        poll_ms         = args.poll_ms,
    )
