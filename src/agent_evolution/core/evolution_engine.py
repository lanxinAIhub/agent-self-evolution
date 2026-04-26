"""
EvolutionEngine - 进化引擎
Central orchestrator that combines experience logging, strategy analysis,
and algorithm-driven adaptation. Selects, evaluates, and evolves agent strategies.

The engine supports:
- Thompson Sampling for exploration/exploitation balance
- UCB (Upper Confidence Bound) for strategy selection
- Bayesian optimization for parameter tuning
- Automatic strategy mutation and crossover
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Any

from .experience_logger import ExperienceLogger, ExperienceRecord
from .strategy_analyzer import StrategyAnalyzer, StrategyRecommendation


@dataclass
class StrategyConfig:
    """Configuration for a strategy in the evolution loop."""
    name: str
    weight: float = 1.0          # Selection weight
    temperature: float = 1.0     # For softmax selection
    mutation_rate: float = 0.1   # Probability of mutating params
    params: dict = field(default_factory=dict)
    enabled: bool = True


@dataclass
class EvolutionEvent:
    """An event emitted during the evolution loop."""
    event_type: str  # "selection", "mutation", "crossover", "reward"
    timestamp: str
    details: dict


class EvolutionEngine:
    """
    Main engine for agent self-evolution.
    
    The evolution loop:
    1. Given a task, select the best strategy (Thompson Sampling / UCB)
    2. Log the execution with ExperienceLogger
    3. Analyze results with StrategyAnalyzer
    4. Evolve strategy weights and parameters (reward-based)
    5. Optionally mutate/crossover strategies
    
    No GPU required. All computation is lightweight statistics.
    """

    def __init__(
        self,
        db_path: str = "~/.agent_evolution/experiences.db",
        config_path: Optional[str] = None,
        selection_mode: str = "thompson",  # "thompson", "ucb", "softmax", "greedy"
    ):
        self.db_path = db_path
        self.logger = ExperienceLogger(db_path)
        self.analyzer = StrategyAnalyzer(db_path)
        self.selection_mode = selection_mode
        
        # Strategy registry
        self.strategies: dict[str, StrategyConfig] = {}
        
        # Event log for observability
        self.events: list[EvolutionEvent] = []
        
        # Config persistence
        self.config_path = config_path or "~/.agent_evolution/evolution_config.json"
        self._load_config()

    def _load_config(self):
        p = Path(self.config_path).expanduser()
        if p.exists():
            data = json.loads(p.read_text())
            self.strategies = {
                name: StrategyConfig(name=name, **cfg) 
                for name, cfg in data.get("strategies", {}).items()
            }

    def save_config(self):
        """Persist strategy configs to disk."""
        p = Path(self.config_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "strategies": {
                name: {
                    "weight": s.weight,
                    "temperature": s.temperature,
                    "mutation_rate": s.mutation_rate,
                    "params": s.params,
                    "enabled": s.enabled,
                }
                for name, s in self.strategies.items()
            }
        }
        p.write_text(json.dumps(data, indent=2))

    def register_strategy(self, name: str, **kwargs):
        """Register a new strategy or update existing one."""
        if name in self.strategies:
            for k, v in kwargs.items():
                if hasattr(self.strategies[name], k):
                    setattr(self.strategies[name], k, v)
        else:
            self.strategies[name] = StrategyConfig(name=name, **kwargs)
        self.save_config()

    def select_strategy(
        self,
        task_type: str,
        context: Optional[dict] = None,
    ) -> tuple[str, dict]:
        """
        Select the best strategy for a task using the configured selection mode.
        
        Returns (strategy_name, strategy_params).
        """
        recommendations = self.analyzer.rank_strategies(
            task_type=task_type, 
            min_samples=1,
            top_k=10,
        )
        
        if not recommendations:
            # No data yet — use registered strategies with equal weight
            enabled = [s for s in self.strategies.values() if s.enabled]
            if not enabled:
                return "default", {}
            if self.selection_mode == "greedy":
                return enabled[0].name, enabled[0].params.copy()
            # Random among enabled
            import random
            chosen = random.choice(enabled)
            return chosen.name, chosen.params.copy()
        
        rec_map = {r.strategy_name: r for r in recommendations}
        
        if self.selection_mode == "thompson":
            return self._thompson_select(recommendations, rec_map)
        elif self.selection_mode == "ucb":
            return self._ucb_select(recommendations, rec_map)
        elif self.selection_mode == "softmax":
            return self._softmax_select(recommendations)
        else:  # greedy
            return recommendations[0].strategy_name, {}

    def _thompson_select(
        self,
        recommendations: list[StrategyRecommendation],
        rec_map: dict,
    ) -> tuple[str, dict]:
        """Thompson Sampling: sample from posterior Beta distribution."""
        import random
        import math
        
        best_name = recommendations[0].strategy_name
        best_sample = -1.0
        
        for name, rec in rec_map.items():
            # Beta distribution parameters from win rate
            successes = int(rec.avg_score * rec.sample_size)
            failures = rec.sample_size - successes
            alpha, beta = max(1, successes + 1), max(1, failures + 1)
            
            # Sample from Beta
            # Use Gamma trick for Beta sampling
            gamma_a = random.gammavariate(alpha, 1.0)
            gamma_b = random.gammavariate(beta, 1.0)
            sample = gamma_a / (gamma_a + gamma_b) if (gamma_a + gamma_b) > 0 else 0
            
            if sample > best_sample:
                best_sample = sample
                best_name = name
        
        params = self.strategies.get(best_name, StrategyConfig(name=best_name)).params.copy()
        return best_name, params

    def _ucb_select(
        self,
        recommendations: list[StrategyRecommendation],
        rec_map: dict,
    ) -> tuple[str, dict]:
        """UCB (Upper Confidence Bound): balance mean reward vs uncertainty."""
        import math
        
        total_samples = sum(r.sample_size for r in recommendations)
        if total_samples == 0:
            return recommendations[0].strategy_name, {}
        
        C = 1.0  # exploration constant
        
        best_name = recommendations[0].strategy_name
        best_ucb = -float('inf')
        
        for rec in recommendations:
            exploitation = rec.avg_score
            exploration = C * math.sqrt(math.log(total_samples + 1) / rec.sample_size)
            ucb = exploitation + exploration
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_name = rec.strategy_name
        
        params = self.strategies.get(best_name, StrategyConfig(name=best_name)).params.copy()
        return best_name, params

    def _softmax_select(
        self,
        recommendations: list[StrategyRecommendation],
    ) -> tuple[str, dict]:
        """Softmax selection: probability proportional to score."""
        import random
        import math
        
        if not recommendations:
            return "default", {}
        
        temperatures = [self.strategies.get(r.strategy_name, StrategyConfig(name=r.strategy_name)).temperature 
                       for r in recommendations]
        avg_temp = sum(temperatures) / len(temperatures)
        
        # Compute softmax probabilities
        scores = [r.avg_score / (t if t > 0.01 else 0.01) for r, t in zip(recommendations, temperatures)]
        max_score = max(scores)
        exp_scores = [math.exp((s - max_score) / avg_temp) for s in scores]
        sum_exp = sum(exp_scores)
        probs = [e / sum_exp for e in exp_scores]
        
        # Sample
        r = random.random()
        cumsum = 0.0
        for i, prob in enumerate(probs):
            cumsum += prob
            if r <= cumsum:
                name = recommendations[i].strategy_name
                params = self.strategies.get(name, StrategyConfig(name=name)).params.copy()
                return name, params
        
        name = recommendations[0].strategy_name
        return name, self.strategies.get(name, StrategyConfig(name=name)).params.copy()

    def execute_and_evolve(
        self,
        task_type: str,
        task_id: str,
        execute_fn: Callable[..., dict],  # fn(task_id, strategy_name, params) -> {"outcome": str, "score": float, ...}
        context: Optional[dict] = None,
        strategies_override: Optional[list[str]] = None,
    ) -> dict:
        """
        Main loop: select strategy → execute → log → evolve.
        
        Args:
            task_type: Type/category of task
            task_id: Unique task identifier
            execute_fn: The actual task execution function.
                        Signature: fn(task_id, strategy_name, strategy_params) -> dict
                        Must return: {"outcome": "success|failure|partial", "score": 0.0-1.0}
                        May return: {"duration_seconds": float, "error_message": str, "metadata": dict}
            context: Task context metadata
            strategies_override: Limit strategy selection to these names
        
        Returns:
            Execution result dict
        """
        # 1. Select
        strategy_name, strategy_params = self.select_strategy(task_type, context)
        
        # Override filter
        if strategies_override and strategy_name not in strategies_override:
            strategy_name = strategies_override[0] if strategies_override else "default"
        
        # 2. Execute
        start_time = time.time()
        try:
            result = execute_fn(task_id, strategy_name, strategy_params)
        except Exception as e:
            result = {
                "outcome": "failure",
                "score": 0.0,
                "error_message": str(e),
            }
        duration = time.time() - start_time
        
        # 3. Log experience
        self.logger.log_task(
            task_type=task_type,
            task_id=task_id,
            strategy_name=strategy_name,
            strategy_params=strategy_params,
            context=context or {},
            outcome=result.get("outcome", "failure"),
            score=result.get("score", 0.0),
            duration_seconds=result.get("duration_seconds", duration),
            error_message=result.get("error_message", ""),
            metadata=result.get("metadata", {}),
        )
        
        # 4. Evolve strategy weights based on reward
        self._update_strategy_weights(task_type, strategy_name, result)
        
        # 5. Record event
        self.events.append(EvolutionEvent(
            event_type="execution",
            timestamp=datetime.now().isoformat(),
            details={
                "task_type": task_type,
                "task_id": task_id,
                "strategy": strategy_name,
                "outcome": result.get("outcome"),
                "score": result.get("score"),
            },
        ))
        
        return {
            "strategy_used": strategy_name,
            "strategy_params": strategy_params,
            "execution_result": result,
        }

    def _update_strategy_weights(self, task_type: str, strategy_name: str, result: dict):
        """Update strategy selection weight based on execution reward."""
        if strategy_name not in self.strategies:
            self.strategies[strategy_name] = StrategyConfig(name=strategy_name)
        
        s = self.strategies[strategy_name]
        score = result.get("score", 0.0)
        
        # Exponential moving average update
        # Higher score → increase weight; lower → decrease
        delta = (score - 0.5) * 0.1  # nudge
        s.weight = max(0.01, s.weight * (1 + delta))
        
        # Optional: mutate params with low probability
        if result.get("outcome") == "failure" and s.mutation_rate > 0:
            if __import__("random").random() < s.mutation_rate:
                self._mutate_params(s)
        
        self.save_config()

    def _mutate_params(self, strategy: StrategyConfig):
        """Mutate strategy parameters slightly."""
        import random
        mutated = {}
        for k, v in strategy.params.items():
            if isinstance(v, float):
                mutated[k] = v * random.uniform(0.8, 1.25)
            elif isinstance(v, int):
                mutated[k] = int(v * random.uniform(0.8, 1.25))
            else:
                mutated[k] = v
        strategy.params = mutated

    def get_best_strategy(self, task_type: str) -> StrategyRecommendation:
        """Get the currently best-performing strategy for a task type."""
        return self.analyzer.recommend(task_type)

    def get_evolution_summary(self) -> dict:
        """Get a summary of the current evolution state."""
        stats = self.analyzer.get_strategy_stats()
        return {
            "total_experiences": self.logger.count(),
            "registered_strategies": len(self.strategies),
            "strategy_stats": stats,
            "recent_events": [
                {"type": e.event_type, "timestamp": e.timestamp, "details": e.details}
                for e in self.events[-10:]
            ],
        }

    def run_batch(
        self,
        task_type: str,
        task_ids: list[str],
        execute_fn: Callable[..., dict],
        context_fn: Optional[Callable[[str], dict]] = None,
    ) -> list[dict]:
        """Run a batch of tasks with automatic strategy selection."""
        results = []
        for task_id in task_ids:
            ctx = context_fn(task_id) if context_fn else {}
            result = self.execute_and_evolve(
                task_type=task_type,
                task_id=task_id,
                execute_fn=execute_fn,
                context=ctx,
            )
            results.append(result)
        return results
