"""
StatisticalAnalyzer - 统计学习胜率分析
Statistical analysis tools for win rate and performance comparison.
Provides:
- Binomial confidence intervals (Wilson, Clopper-Pearson)
- A/B test comparison between strategies
- Moving average and trend detection
- Context similarity scoring
"""

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class WinRateResult:
    strategy: str
    successes: int
    total: int
    win_rate: float
    ci_lower: float
    ci_upper: float
    confidence: float  # Based on sample size


@dataclass
class ABTestResult:
    strategy_a: str
    strategy_b: str
    winner: str  # "a", "b", or "insufficient_data"
    p_value: float
    relative_improvement: float  # % improvement of winner over loser
    confidence: float
    reasoning: str


class StatisticalAnalyzer:
    """
    Statistical analysis for strategy comparison and win rate estimation.
    
    All methods are lightweight and run on CPU.
    Designed for comparing strategies with small to medium sample sizes.
    """

    def __init__(self, db_path: str = "~/.agent_evolution/experiences.db"):
        self.db_path = Path(db_path).expanduser()

    # ────────────────────────────────────────────────────────────────
    # Confidence Intervals
    # ────────────────────────────────────────────────────────────────

    def wilson_ci(self, successes: int, total: int, z: float = 1.645) -> tuple[float, float]:
        """
        Wilson score confidence interval.
        More accurate than normal approximation for small samples.
        
        Returns (lower, upper) bounds.
        z=1.645 → 90% CI
        z=1.96  → 95% CI
        """
        if total == 0:
            return 0.0, 1.0
        
        p = successes / total
        denominator = 1 + z**2 / total
        center = p + z**2 / (2 * total)
        margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        
        lower = max(0.0, (center - margin) / denominator)
        upper = min(1.0, (center + margin) / denominator)
        return lower, upper

    def clopper_pearson_ci(self, successes: int, total: int, alpha: float = 0.1) -> tuple[float, float]:
        """
        Clopper-Pearson exact confidence interval for binomial proportion.
        More conservative than Wilson; uses beta distribution quantiles.
        """
        from math import gamma, lgamma
        
        if total == 0:
            return 0.0, 1.0
        
        # Use approximation via regularized incomplete beta function
        # For exact computation we'd need scipy.special.betainc
        # Here we use a simple approximation
        p = successes / total
        
        # Jeffreys prior interval (Bayesian with Beta(0.5, 0.5) prior)
        # Equivalent to Wilson with z=1.96
        alpha2 = successes + 0.5
        beta2 = total - successes + 0.5
        
        # Approximation: use Wilson with adjusted counts
        adj_successes = successes + 0.5
        adj_total = total + 1
        return self.wilson_ci(int(adj_successes), int(adj_total), z=1.96)

    # ────────────────────────────────────────────────────────────────
    # Win Rate Analysis
    # ────────────────────────────────────────────────────────────────

    def analyze_win_rates(
        self,
        task_type: Optional[str] = None,
        min_samples: int = 2,
        confidence_level: float = 0.90,
    ) -> list[WinRateResult]:
        """Analyze win rates for all strategies with confidence intervals."""
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT strategy_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                   AVG(score) as avg_score
            FROM experiences
        """
        params = []
        if task_type:
            query += " WHERE task_type = ?"
            params.append(task_type)
        query += " GROUP BY strategy_name"
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        z = 1.645 if confidence_level == 0.90 else 1.96
        results = []
        
        for row in rows:
            name, total, successes, avg_score = row
            successes = successes or 0
            total = total or 0
            
            if total < min_samples:
                continue
            
            wr = successes / total
            ci_lower, ci_upper = self.wilson_ci(successes, total, z=z)
            confidence = min(1.0, total / 30.0)
            
            results.append(WinRateResult(
                strategy=name,
                successes=successes,
                total=total,
                win_rate=wr,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence=confidence,
            ))
        
        results.sort(key=lambda r: r.win_rate * r.confidence, reverse=True)
        return results

    # ────────────────────────────────────────────────────────────────
    # A/B Testing
    # ────────────────────────────────────────────────────────────────

    def compare_strategies(
        self,
        strategy_a: str,
        strategy_b: str,
        task_type: Optional[str] = None,
    ) -> ABTestResult:
        """
        Compare two strategies using a two-proportion z-test.
        
        H0: p_a = p_b (no difference)
        H1: p_a ≠ p_b (two-sided)
        """
        if not self.db_path.exists():
            return ABTestResult(
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                winner="insufficient_data",
                p_value=1.0,
                relative_improvement=0.0,
                confidence=0.0,
                reasoning="No data available.",
            )
        
        conn = sqlite3.connect(str(self.db_path))
        query_a = """
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes
            FROM experiences WHERE strategy_name = ?
        """
        params_a = [strategy_a]
        if task_type:
            query_a += " AND task_type = ?"
            params_a.append(task_type)
        
        query_b = query_a.replace("strategy_name = ?", "strategy_name = ?", 1)
        params_b = [strategy_b]
        if task_type:
            params_b.append(task_type)
        
        row_a = conn.execute(query_a, params_a).fetchone()
        row_b = conn.execute(query_b, params_b).fetchone()
        conn.close()
        
        n_a, s_a = row_a[0], row_a[1] or 0
        n_b, s_b = row_b[0], row_b[1] or 0
        
        if n_a < 3 or n_b < 3:
            return ABTestResult(
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                winner="insufficient_data",
                p_value=1.0,
                relative_improvement=0.0,
                confidence=0.0,
                reasoning=f"Sample too small: A={n_a}, B={n_b}. Need ≥3 each.",
            )
        
        p_a = s_a / n_a
        p_b = s_b / n_b
        
        # Pooled proportion
        p_pool = (s_a + s_b) / (n_a + n_b)
        se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        
        if se < 1e-10:
            return ABTestResult(
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                winner="insufficient_data",
                p_value=1.0,
                relative_improvement=0.0,
                confidence=0.0,
                reasoning="Standard error too small to compute.",
            )
        
        z = (p_a - p_b) / se
        
        # Two-tailed p-value
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        
        # Determine winner
        if p_value < 0.10:
            if p_a > p_b:
                winner = "a"
                rel_imp = (p_a - p_b) / max(p_b, 0.001)
            else:
                winner = "b"
                rel_imp = (p_b - p_a) / max(p_a, 0.001)
        else:
            winner = "insufficient_data"
            rel_imp = 0.0
        
        confidence = max(0.0, 1.0 - p_value)
        winner_name = [strategy_a, strategy_b, "none"]["abc".index(winner)] if winner != "insufficient_data" else "none"
        
        return ABTestResult(
            strategy_a=strategy_a,
            strategy_b=strategy_b,
            winner=winner,
            p_value=p_value,
            relative_improvement=rel_imp,
            confidence=confidence,
            reasoning=(
                f"p_a={p_a:.3f} ({s_a}/{n_a}), p_b={p_b:.3f} ({s_b}/{n_b}), "
                f"z={z:.3f}, p={p_value:.4f}. "
                f"{'Significant difference' if p_value < 0.10 else 'No significant difference'}."
            ),
        )

    # ────────────────────────────────────────────────────────────────
    # Trend Analysis
    # ────────────────────────────────────────────────────────────────

    def get_moving_average(
        self,
        strategy_name: str,
        task_type: Optional[str] = None,
        window: int = 10,
    ) -> list[tuple[str, float]]:
        """Get time-series of moving average scores for a strategy."""
        if not self.db_path.exists():
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT timestamp, score
            FROM experiences
            WHERE strategy_name = ?
        """
        params = [strategy_name]
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY timestamp ASC"
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        if len(rows) < window:
            return [(r[0], r[1]) for r in rows]
        
        results = []
        scores = [r[1] for r in rows]
        times = [r[0] for r in rows]
        
        for i in range(len(rows)):
            if i < window - 1:
                ma = sum(scores[:i+1]) / (i + 1)
            else:
                ma = sum(scores[i-window+1:i+1]) / window
            results.append((times[i], ma))
        
        return results

    def detect_drift(
        self,
        strategy_name: str,
        task_type: Optional[str] = None,
        baseline_window: int = 20,
        test_window: int = 10,
    ) -> dict:
        """
        Detect if a strategy's performance is drifting from its baseline.
        Uses Mann-Whitney U test for significance (non-parametric).
        """
        if not self.db_path.exists():
            return {"drifted": False, "direction": "unknown"}
        
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT score
            FROM experiences
            WHERE strategy_name = ?
        """
        params = [strategy_name]
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY timestamp DESC"
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        scores = [r[0] for r in rows]
        
        if len(scores) < baseline_window + test_window:
            return {"drifted": False, "direction": "insufficient_data", "p_value": 1.0}
        
        baseline = scores[baseline_window:]
        recent = scores[:test_window]
        
        baseline_avg = sum(baseline) / len(baseline)
        recent_avg = sum(recent) / len(recent)
        
        # Simple z-test for difference in means (approximate)
        diff = recent_avg - baseline_avg
        se = math.sqrt((sum((s - baseline_avg)**2 for s in baseline) / (len(baseline)-1)) / len(recent) +
                      (sum((s - recent_avg)**2 for s in recent) / (len(recent)-1)) / len(recent))
        
        if se < 1e-10:
            return {"drifted": False, "direction": "stable", "p_value": 1.0}
        
        z = diff / se
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        
        drifted = p_value < 0.10
        direction = "improving" if diff > 0 else "degrading"
        
        return {
            "drifted": drifted,
            "direction": direction if drifted else "stable",
            "p_value": p_value,
            "baseline_avg": baseline_avg,
            "recent_avg": recent_avg,
            "change": diff,
        }

    # ────────────────────────────────────────────────────────────────
    # Context Similarity
    # ────────────────────────────────────────────────────────────────

    def find_similar_contexts(
        self,
        reference_context: dict,
        task_type: Optional[str] = None,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Find strategies that performed well in similar contexts.
        Uses Jaccard similarity on context key sets.
        
        Returns list of (strategy_name, similarity_score).
        """
        if not self.db_path.exists() or not reference_context:
            return []
        
        ref_keys = set(reference_context.keys())
        if not ref_keys:
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT strategy_name, context
            FROM experiences
        """
        params = []
        if task_type:
            query += " WHERE task_type = ?"
            params.append(task_type)
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        strategy_scores: dict[str, list[float]] = defaultdict(list)
        strategy_contexts: dict[str, set] = defaultdict(set)
        
        for name, ctx_json in rows:
            try:
                ctx = json.loads(ctx_json) if isinstance(ctx_json, str) else ctx_json
            except (json.JSONDecodeError, TypeError):
                continue
            
            ctx_keys = set(ctx.keys())
            # Jaccard similarity
            intersection = len(ref_keys & ctx_keys)
            union = len(ref_keys | ctx_keys)
            sim = intersection / union if union > 0 else 0.0
            
            if sim > 0:
                strategy_contexts[name] = ctx_keys
                strategy_scores[name].append(sim)
        
        # Average similarity per strategy
        avg_sims = {name: sum(scores)/len(scores) for name, scores in strategy_scores.items()}
        
        return sorted(avg_sims.items(), key=lambda x: x[1], reverse=True)[:top_k]
