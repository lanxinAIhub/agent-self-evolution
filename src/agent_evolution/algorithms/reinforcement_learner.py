"""
ReinforcementLearner - 强化学习基础版（无模型）
Model-free reinforcement learning for strategy selection.
Implements:
- Epsilon-Greedy exploration
- Softmax (Boltzmann) exploration  
- Upper Confidence Bound (UCB)
- Q-Learning with experience replay

No GPU required. Pure NumPy/SciPy.
"""

import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class QState:
    """State representation for strategy selection."""
    task_type: str
    context_hash: str = ""  # Compact context fingerprint

    def __hash__(self):
        return hash((self.task_type, self.context_hash))

    def __eq__(self, other):
        return self.task_type == other.task_type and self.context_hash == other.context_hash


@dataclass
class ReplayBuffer:
    """Experience replay buffer for Q-Learning."""
    capacity: int = 10000
    buffer: deque = field(default_factory=lambda: deque(maxlen=10000))

    def push(self, state: str, action: str, reward: float, next_state: str):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class ReinforcementLearner:
    """
    Model-free RL for strategy (action) selection given task states.
    
    Supports:
    - Q-Learning with ε-greedy, softmax, and UCB action selection
    - Experience replay for stable learning
    - Automatic state abstraction (hashes task_type + context features)
    
    All computation is O(1) per step; memory scales with replay buffer.
    """

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 0.1,          # ε for ε-greedy
        temperature: float = 1.0,       # for softmax
        ucb_constant: float = 1.0,
        exploration_mode: str = "epsilon",  # "epsilon", "softmax", "ucb"
        replay_capacity: int = 5000,
    ):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.temperature = temperature
        self.ucb_c = ucb_constant
        self.exploration_mode = exploration_mode
        
        self.Q: dict[str, dict[str, float]] = {}  # state -> {action: q_value}
        self.action_visits: dict[str, dict[str, int]] = {}  # state -> {action: count}
        self.replay = ReplayBuffer(capacity=replay_capacity)
        
        self.total_steps = 0

    def _get_state(self, task_type: str, context: Optional[dict] = None) -> str:
        """Serialize state to string key."""
        if not context:
            return task_type
        
        # Create compact context hash from relevant features
        features = []
        for k in sorted(context.keys()):
            v = context[k]
            if isinstance(v, (int, float, bool, str)):
                features.append(f"{k}={v}")
        
        hash_str = str(hash(tuple(features)))[:8] if features else ""
        return f"{task_type}::{hash_str}"

    def register_actions(self, actions: list[str]):
        """Register available actions (strategies)."""
        for action in actions:
            if action not in self.Q:
                self.Q[action] = {}
                self.action_visits[action] = {}

    def get_q(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        return self.Q.get(state, {}).get(action, 0.0)

    def update(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: str,
        done: bool = False,
        learn: bool = True,
    ):
        """
        Update Q-value for state-action pair using Q-Learning.
        
        Q(s,a) ← Q(s,a) + α[r + γ * max_a' Q(s',a') - Q(s,a)]
        """
        if not learn:
            self.replay.push(state, action, reward, next_state)
            return
        
        # Store in replay buffer
        self.replay.push(state, action, reward, next_state)
        
        # Initialize if new
        if state not in self.Q:
            self.Q[state] = {}
        if action not in self.Q[state]:
            self.Q[state][action] = 0.0
        
        if next_state not in self.Q:
            self.Q[next_state] = {}
        
        # TD target
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * max(
                self.Q[next_state].get(a, 0.0) for a in self.Q[next_state]
            )
        
        # TD error
        td_error = td_target - self.Q[state][action]
        
        # Update
        self.Q[state][action] += self.lr * td_error
        
        # Update visit counts
        if action not in self.action_visits:
            self.action_visits[action] = {}
        self.action_visits[action][state] = self.action_visits[action].get(state, 0) + 1
        
        self.total_steps += 1

    def select_action(
        self,
        state: str,
        available_actions: list[str],
        context: Optional[dict] = None,
        use_ucb: bool = False,
    ) -> str:
        """
        Select action using configured exploration strategy.
        """
        if not available_actions:
            return "default"
        
        # Ensure all actions have Q entries
        for a in available_actions:
            if state not in self.Q:
                self.Q[state] = {}
            if a not in self.Q[state]:
                self.Q[state][a] = 0.0
        
        if use_ucb or self.exploration_mode == "ucb":
            return self._ucb_select(state, available_actions)
        elif self.exploration_mode == "softmax":
            return self._softmax_select(state, available_actions)
        else:
            return self._epsilon_greedy(state, available_actions)

    def _epsilon_greedy(self, state: str, actions: list[str]) -> str:
        """ε-Greedy: explore with probability ε, exploit otherwise."""
        if random.random() < self.epsilon:
            return random.choice(actions)
        
        # Exploit: pick best Q
        q_vals = {a: self.Q[state].get(a, 0.0) for a in actions}
        max_q = max(q_vals.values())
        best = [a for a, q in q_vals.items() if abs(q - max_q) < 1e-9]
        return random.choice(best)

    def _softmax_select(self, state: str, actions: list[str]) -> str:
        """Softmax/Boltzmann: sample action proportionally to exp(Q/T)."""
        q_vals = [self.Q[state].get(a, 0.0) for a in actions]
        max_q = max(q_vals)
        # Numerically stable softmax
        exp_qs = [math.exp((q - max_q) / max(self.temperature, 0.01)) for q in q_vals]
        sum_exp = sum(exp_qs)
        probs = [e / sum_exp for e in exp_qs]
        
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r <= cumsum:
                return actions[i]
        return actions[0]

    def _ucb_select(self, state: str, actions: list[str]) -> str:
        """UCB: Q + c * sqrt(ln(t) / N(s,a))."""
        t = max(1, self.total_steps)
        q_vals = []
        
        for a in actions:
            q = self.Q[state].get(a, 0.0)
            n = self.action_visits.get(a, {}).get(state, 0)
            if n == 0:
                ucb = float('inf')
            else:
                ucb = q + self.ucb_c * math.sqrt(math.log(t) / n)
            q_vals.append(ucb)
        
        max_ucb = max(q_vals)
        best_idx = q_vals.index(max_ucb)
        return actions[best_idx]

    def train_batch(self, batch_size: int = 32):
        """Perform a training update on a random batch from replay buffer."""
        if len(self.replay) < batch_size:
            return 0.0
        
        batch = self.replay.sample(batch_size)
        total_td = 0.0
        
        for state, action, reward, next_state in batch:
            if state not in self.Q:
                self.Q[state] = {}
            if action not in self.Q[state]:
                self.Q[state][action] = 0.0
            if next_state not in self.Q:
                self.Q[next_state] = {}
            
            if next_state:
                td_target = reward + self.gamma * max(
                    self.Q[next_state].get(a, 0.0) for a in self.Q[next_state]
                )
            else:
                td_target = reward
            
            td_error = td_target - self.Q[state][action]
            self.Q[state][action] += self.lr * td_error
            total_td += abs(td_error)
        
        return total_td / batch_size

    def decay_epsilon(self, factor: float = 0.99, min_epsilon: float = 0.01):
        """Decay epsilon for less exploration over time."""
        self.epsilon = max(min_epsilon, self.epsilon * factor)

    def get_value(self, state: str, available_actions: list[str]) -> float:
        """V(s) = max_a Q(s,a) for a given state and action set."""
        if state not in self.Q:
            return 0.0
        if not available_actions:
            return 0.0
        return max(self.Q[state].get(a, 0.0) for a in available_actions)

    def save(self, path: str):
        """Persist learner state."""
        data = {
            "Q": self.Q,
            "action_visits": self.action_visits,
            "total_steps": self.total_steps,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "exploration_mode": self.exploration_mode,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data))

    def load(self, path: str):
        """Load learner state."""
        data = json.loads(Path(path).read_text())
        self.Q = data["Q"]
        self.action_visits = data.get("action_visits", {})
        self.total_steps = data.get("total_steps", 0)
        self.lr = data.get("lr", 0.1)
        self.gamma = data.get("gamma", 0.9)
        self.epsilon = data.get("epsilon", 0.1)
        self.temperature = data.get("temperature", 1.0)
        self.exploration_mode = data.get("exploration_mode", "epsilon")
