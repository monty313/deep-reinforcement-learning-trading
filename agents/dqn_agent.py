"""
agents/dqn_agent.py
Multi-asset DQN agent (Keras).  Extends the Quantra template approach with:
- 7-action space per asset (flat, buy/sell × small/med/large)
- Learnable risk fractions stored alongside model weights
- Shared-trunk + per-asset action heads
- ExperienceReplay with Bellman update
- Weight save / load (transfer learning support)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.optimizers import Adam

from env.ftmo_game import NUM_ACTIONS, FLAT, BUY_SMALL, SELL_SMALL

# ── default risk fractions (learnable; stored in agent state) ─────────────────
DEFAULT_RISK_FRACTIONS = {"small": 0.005, "med": 0.010, "large": 0.020}


# ── Experience Replay (same design as Quantra) ────────────────────────────────
class ExperienceReplay:
    """Bellman experience replay memory."""

    def __init__(self, max_memory: int = 2000, discount: float = 0.95):
        self.max_memory = max_memory
        self.discount   = discount
        self.memory: list = []

    def remember(self, states, game_over: bool):
        self.memory.append([states, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def process(self, q_net, r_net, batch_size: int = 32):
        n           = len(self.memory)
        env_dim     = self.memory[0][0][0].shape[1]
        num_actions = q_net.output_shape[-1]
        bs          = min(n, batch_size)

        idxs       = np.random.randint(0, n, size=bs)
        inputs     = np.zeros((bs, env_dim))
        next_states = np.zeros((bs, env_dim))
        rewards    = np.zeros(bs)
        actions    = np.zeros(bs, dtype=int)
        dones      = np.zeros(bs, dtype=bool)

        for i, idx in enumerate(idxs):
            state_t, action_t, reward_t, state_tp1 = self.memory[idx][0]
            inputs[i]      = state_t
            next_states[i] = state_tp1
            rewards[i]     = reward_t
            actions[i]     = action_t
            dones[i]       = self.memory[idx][1]

        # Single batched predict call each — eliminates per-sample overhead
        targets  = r_net.predict(inputs, verbose=0)
        q_next   = q_net.predict(next_states, verbose=0)
        q_max    = q_next.max(axis=1)

        for i in range(bs):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.discount * q_max[i]

        return inputs, targets


# ── DQN network builder ───────────────────────────────────────────────────────
def build_dqn(state_dim: int, num_assets: int, num_actions: int,
              hidden_mult: int = 2, activation: str = "relu",
              learning_rate: float = 0.001) -> Tuple[Model, Model]:
    """
    Shared trunk -> per-asset action head.
    Returns (q_network, r_network) both with same architecture.
    For simplicity we flatten all per-asset actions into one output of size
    num_assets * num_actions, then reshape in the caller.
    """
    def _make_net():
        # Cap hidden size so memory stays reasonable on large state vectors
        h1 = min(state_dim * hidden_mult, 512)
        h2 = min(state_dim * hidden_mult // 2, 256)
        inp    = Input(shape=(state_dim,))
        hidden = Dense(h1, activation=activation)(inp)
        hidden = Dense(h2, activation=activation)(hidden)
        out    = Dense(num_assets * num_actions, activation="linear")(hidden)
        net    = Model(inputs=inp, outputs=out)
        net.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
        return net

    return _make_net(), _make_net()


# ── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    """
    Wraps Q-network and R-network with Quantra-style training loop.
    Works with FTMOGame's multi-asset action dict.
    """

    def __init__(
        self,
        symbols:        List[str],
        state_dim:      int,
        rl_config:      dict,
        risk_fractions: dict = None,
    ):
        self.symbols     = symbols
        self.num_assets  = len(symbols)
        self.state_dim   = state_dim
        self.rl_config   = rl_config
        self.risk_fractions = risk_fractions or dict(DEFAULT_RISK_FRACTIONS)

        self.q_net, self.r_net = build_dqn(
            state_dim    = state_dim,
            num_assets   = self.num_assets,
            num_actions  = NUM_ACTIONS,
            hidden_mult  = rl_config.get("HIDDEN_MULT", 2),
            activation   = rl_config.get("ACTIVATION_FUN", "relu"),
            learning_rate= rl_config.get("LEARNING_RATE", 0.001),
        )
        self.r_net.set_weights(self.q_net.get_weights())

        self.exp_replay = ExperienceReplay(
            max_memory = rl_config.get("MAX_MEM", 2000),
            discount   = rl_config.get("DISCOUNT_RATE", 0.95),
        )

    # ── exploration / exploitation ────────────────────────────────────────────
    def select_actions(self, state: np.ndarray, epsilon: float) -> Dict[str, int]:
        """Return {symbol: action_int} for all assets."""
        if np.random.rand() <= epsilon:
            return {s: np.random.randint(0, NUM_ACTIONS) for s in self.symbols}

        q_vals = self.q_net(state, training=False).numpy()[0]  # shape: (num_assets * NUM_ACTIONS,)
        q_mat  = q_vals.reshape(self.num_assets, NUM_ACTIONS)
        return {self.symbols[i]: int(np.argmax(q_mat[i])) for i in range(self.num_assets)}

    # ── flat action int for replay memory ─────────────────────────────────────
    def actions_to_flat(self, actions: Dict[str, int]) -> int:
        """Encode the first-asset action for backward-compat with ExperienceReplay."""
        return actions.get(self.symbols[0], FLAT)

    # ── training step ─────────────────────────────────────────────────────────
    def train_step(self, batch_size: int = 32):
        if len(self.exp_replay.memory) < batch_size:
            return
        inputs, targets = self.exp_replay.process(self.q_net, self.r_net, batch_size)
        self.q_net.train_on_batch(inputs, targets)

    def sync_r_net(self):
        self.r_net.set_weights(self.q_net.get_weights())

    # ── save / load ───────────────────────────────────────────────────────────
    def save(self, weights_file: str, replay_file: str, risk_file: str = None):
        self.q_net.save_weights(weights_file, overwrite=True)
        pickle.dump(self.exp_replay.memory, open(replay_file, "wb"))
        if risk_file:
            pickle.dump(self.risk_fractions, open(risk_file, "wb"))

    def load(self, weights_file: str, replay_file: str, risk_file: str = None,
             freeze_layers: int = 0):
        """
        Load weights with optional layer freezing for transfer learning.
        freeze_layers: number of early layers to freeze (0 = no freezing).
        """
        self.q_net.load_weights(weights_file)
        self.r_net.load_weights(weights_file)
        if replay_file and Path(replay_file).exists():
            self.exp_replay.memory = pickle.load(open(replay_file, "rb"))
        if risk_file and Path(risk_file).exists():
            self.risk_fractions = pickle.load(open(risk_file, "rb"))
        if freeze_layers > 0:
            for layer in self.q_net.layers[:freeze_layers]:
                layer.trainable = False
            for layer in self.r_net.layers[:freeze_layers]:
                layer.trainable = False
            self.q_net.compile(optimizer=Adam(self.rl_config.get("LEARNING_RATE", 0.001)), loss="mse")
            self.r_net.compile(optimizer=Adam(self.rl_config.get("LEARNING_RATE", 0.001)), loss="mse")

    # ── epsilon schedule (Quantra style) ─────────────────────────────────────
    @staticmethod
    def epsilon(episode: int, eps_start: float = 0.9, eps_min: float = 0.05) -> float:
        return eps_start ** (np.log10(max(1, episode))) + eps_min
