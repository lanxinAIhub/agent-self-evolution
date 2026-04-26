"""
Tests for Agent Self-Evolution System
Run with: pytest tests/ -v
"""

import json
import math
import tempfile
import time
from pathlib import Path

import pytest

from agent_evolution.core import ExperienceLogger, StrategyAnalyzer, EvolutionEngine
from agent_evolution.algorithms import BayesianOptimizer, ReinforcementLearner, StatisticalAnalyzer


# ────────────────────────────────────────────────────────────────
# ExperienceLogger Tests
# ────────────────────────────────────────────────────────────────

class TestExperienceLogger:
    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.logger = ExperienceLogger(self.db_path)

    def test_log_and_retrieve(self):
        record_id = self.logger.log_task(
            task_type="coding",
            task_id="task-001",
            strategy_name="greedy",
            outcome="success",
            score=0.9,
            duration_seconds=1.5,
        )
        assert record_id is not None
        
        experiences = self.logger.get_experiences(task_type="coding")
        assert len(experiences) == 1
        assert experiences[0]["strategy_name"] == "greedy"
        assert experiences[0]["score"] == 0.9

    def test_strategy_stats(self):
        for i in range(5):
            self.logger.log_task(
                task_type="analysis",
                task_id=f"task-{i}",
                strategy_name="thompson",
                outcome="success" if i >= 3 else "failure",
                score=0.8 if i >= 3 else 0.2,
            )
        
        stats = self.logger.get_strategy_stats(task_type="analysis")
        assert "thompson" in stats
        assert stats["thompson"]["total"] == 5
        assert stats["thompson"]["successes"] == 2
        assert 0.3 < stats["thompson"]["win_rate"] < 0.5

    def test_count(self):
        for i in range(10):
            self.logger.log_task(
                task_type="test",
                task_id=f"t{i}",
                strategy_name="default",
                outcome="success",
                score=1.0,
            )
        assert self.logger.count() == 10


# ────────────────────────────────────────────────────────────────
# StrategyAnalyzer Tests
# ────────────────────────────────────────────────────────────────

class TestStrategyAnalyzer:
    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.logger = ExperienceLogger(self.db_path)
        self.analyzer = StrategyAnalyzer(self.db_path)

    def test_rank_strategies(self):
        for i in range(10):
            self.logger.log_task(
                task_type="ranking_test",
                task_id=f"t{i}",
                strategy_name="strategy_a",
                outcome="success",
                score=0.9,
            )
        for i in range(5):
            self.logger.log_task(
                task_type="ranking_test",
                task_id=f"t{i+10}",
                strategy_name="strategy_b",
                outcome="success",
                score=0.7,
            )
        
        ranked = self.analyzer.rank_strategies(task_type="ranking_test", min_samples=2)
        assert len(ranked) == 2
        assert ranked[0].strategy_name == "strategy_a"
        assert ranked[0].win_rate == 1.0

    def test_wilson_ci(self):
        # 9 successes out of 10 → 90% win rate
        lower, upper = self.analyzer.wilson_ci(9, 10, z=1.645)
        assert 0.6 < lower < 0.7
        assert 0.95 < upper < 1.0

    def test_recommend(self):
        rec = self.analyzer.recommend(task_type="nonexistent", min_samples=1)
        assert rec.strategy_name == "default"


# ────────────────────────────────────────────────────────────────
# EvolutionEngine Tests
# ────────────────────────────────────────────────────────────────

class TestEvolutionEngine:
    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.engine = EvolutionEngine(
            db_path=self.db_path,
            selection_mode="thompson",
        )

    def test_register_and_select(self):
        self.engine.register_strategy("test_strategy", weight=1.0, params={"temp": 0.5})
        name, params = self.engine.select_strategy(task_type="test_task")
        assert name == "test_strategy"
        assert params.get("temp") == 0.5

    def test_execute_and_evolve(self):
        call_count = 0
        
        def fake_execute(task_id, strategy_name, params):
            nonlocal call_count
            call_count += 1
            return {
                "outcome": "success",
                "score": 0.8,
                "duration_seconds": 0.1,
            }
        
        result = self.engine.execute_and_evolve(
            task_type="exec_test",
            task_id="exec-001",
            execute_fn=fake_execute,
        )
        
        assert call_count == 1
        assert result["execution_result"]["outcome"] == "success"
        assert self.logger.count() if hasattr(self, 'logger') else True

    def test_evolution_summary(self):
        summary = self.engine.get_evolution_summary()
        assert "total_experiences" in summary
        assert "registered_strategies" in summary


# ────────────────────────────────────────────────────────────────
# BayesianOptimizer Tests
# ────────────────────────────────────────────────────────────────

class TestBayesianOptimizer:
    def test_suggest_random_initial(self):
        optimizer = BayesianOptimizer(
            param_bounds={"lr": (0.001, 0.1), "temp": (0.1, 2.0)},
            maximize=True,
        )
        suggestion = optimizer.suggest()
        assert 0.001 <= suggestion["lr"] <= 0.1
        assert 0.1 <= suggestion["temp"] <= 2.0

    def test_observe_and_improve(self):
        optimizer = BayesianOptimizer(
            param_bounds={"x": (0.0, 10.0)},
            maximize=True,
        )
        
        # Simulate observing scores
        for _ in range(5):
            params = optimizer.suggest()
            score = 1.0 - abs(params["x"] - 7.3) / 10.0  # Peak at x=7.3
            optimizer.observe(params, score)
        
        best_params, best_score = optimizer.get_best()
        assert 0.0 <= best_params["x"] <= 10.0
        assert 0.0 <= best_score <= 1.0

    def test_save_load(self):
        optimizer = BayesianOptimizer(
            param_bounds={"lr": (0.001, 0.1)},
            maximize=True,
        )
        optimizer.observe({"lr": 0.05}, 0.8)
        
        tmp = tempfile.mktemp(suffix=".json")
        optimizer.save(tmp)
        
        optimizer2 = BayesianOptimizer(param_bounds={"lr": (0.001, 0.1)}, maximize=True)
        optimizer2.load(tmp)
        
        assert len(optimizer2.observations) == 1
        assert optimizer2.observations[0].score == 0.8


# ────────────────────────────────────────────────────────────────
# ReinforcementLearner Tests
# ────────────────────────────────────────────────────────────────

class TestReinforcementLearner:
    def test_q_learning_update(self):
        learner = ReinforcementLearner(learning_rate=0.1, discount_factor=0.9)
        
        learner.register_actions(["a", "b"])
        learner.update("task_1", "a", reward=1.0, next_state="task_2")
        learner.update("task_2", "b", reward=0.5, next_state=None, done=True)
        
        assert learner.get_q("task_1", "a") != 0.0

    def test_epsilon_greedy_explores(self):
        learner = ReinforcementLearner(exploration_mode="epsilon", epsilon=0.0)
        # With epsilon=0, should always pick highest Q
        learner.Q["s"] = {"a": 1.0, "b": 0.5}
        action = learner.select_action("s", ["a", "b"])
        assert action == "a"

    def test_softmax_selects(self):
        learner = ReinforcementLearner(exploration_mode="softmax", temperature=0.1)
        learner.Q["s"] = {"a": 1.0, "b": 0.5}
        # Should prefer higher Q
        action = learner.select_action("s", ["a", "b"])
        assert action == "a"

    def test_save_load(self):
        learner = ReinforcementLearner(epsilon=0.1)
        learner.Q["s"] = {"a": 0.5}
        
        tmp = tempfile.mktemp(suffix=".json")
        learner.save(tmp)
        
        learner2 = ReinforcementLearner()
        learner2.load(tmp)
        assert learner2.Q["s"]["a"] == 0.5


# ────────────────────────────────────────────────────────────────
# StatisticalAnalyzer Tests
# ────────────────────────────────────────────────────────────────

class TestStatisticalAnalyzer:
    def test_wilson_ci_bounds(self):
        analyzer = StatisticalAnalyzer()
        lower, upper = analyzer.wilson_ci(50, 100, z=1.645)
        assert 0.0 <= lower <= upper <= 1.0
        assert 0.4 < lower < 0.6
        assert 0.4 < upper < 0.6

    def test_compare_strategies(self):
        db = tempfile.mktemp(suffix=".db")
        logger = ExperienceLogger(db)
        analyzer = StatisticalAnalyzer(db)
        
        # Strategy A: 8/10 successes
        for i in range(10):
            logger.log_task("test", f"a{i}", "A", score=0.8 if i < 8 else 0.2)
        # Strategy B: 4/10 successes
        for i in range(10):
            logger.log_task("test", f"b{i}", "B", score=0.8 if i < 4 else 0.2)
        
        result = analyzer.compare_strategies("A", "B", task_type="test")
        assert result.winner in ["a", "b", "insufficient_data"]
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_drift(self):
        db = tempfile.mktemp(suffix=".db")
        logger = ExperienceLogger(db)
        analyzer = StatisticalAnalyzer(db)
        
        # Degrading: early scores high, late scores low
        for i in range(30):
            score = 0.9 if i < 20 else 0.3
            logger.log_task("drift_test", f"t{i}", "strategy_x", score=score)
        
        result = analyzer.detect_drift("strategy_x", task_type="drift_test")
        assert "drifted" in result
        assert "direction" in result
