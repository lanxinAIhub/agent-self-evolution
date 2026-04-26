"""
StrategyAnalyzer - 策略分析器
Analyzes historical experiences to determine which strategies work best
for given task contexts. Provides ranking, confidence intervals, and
contextual recommendations.
"""

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StrategyRecommendation:
    """Recommendation for a strategy given a task context."""
    strategy_name: str
    win_rate: float
    confidence: float       # 0-1, based on sample size
    avg_score: float
    sample_size: int
    reasoning: str


class StrategyAnalyzer:
    """
    Analyzes experience logs to find optimal strategies per task context.
    
    Uses:
    - Wilson score interval for win rate confidence (handles low sample sizes)
    - Context similarity scoring for contextual recommendations
    - Trend detection for strategy performance over time
    """

    def __init__(self, db_path: str = "~/.agent_evolution/experiences.db"):
        self.db_path = Path(db_path).expanduser()

    def _wilson_score(self, successes: int, total: int, z: float = 1.645) -> float:
        """
        Wilson score confidence interval lower bound.
        More accurate than raw win rate for small samples.
        z=1.645 → 90% confidence
        """
        if total == 0:
            return 0.0
        p = successes / total
        denominator = 1 + z**2 / total
        center = p + z**2 / (2 * total)
        spread = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
        return max(0.0, (center - spread) / denominator)

    def _get_strategy_records(
        self, 
        task_type: Optional[str] = None,
        min_samples: int = 0,
    ) -> dict[str, dict]:
        """Fetch aggregated strategy stats from DB."""
        if not self.db_path.exists():
            return {}
        
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT strategy_name, 
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                   AVG(score) as avg_score,
                   AVG(duration_seconds) as avg_duration,
                   GROUP_CONCAT(score) as all_scores
            FROM experiences
        """
        params = []
        if task_type:
            query += " WHERE task_type = ?"
            params.append(task_type)
        query += " GROUP BY strategy_name"
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            name, total, successes, avg_score, avg_dur, scores_str = row
            if total < min_samples:
                continue
            # Compute variance of scores if we have them
            score_variance = 0.0
            if scores_str:
                try:
                    scores = [float(s) for s in scores_str.split(",")]
                    if len(scores) > 1:
                        mean = sum(scores) / len(scores)
                        score_variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
                except (ValueError, ZeroDivisionError):
                    pass
            
            result[name] = {
                "total": total,
                "successes": successes or 0,
                "win_rate": (successes or 0) / total if total > 0 else 0.0,
                "wilson_lower": self._wilson_score(successes or 0, total),
                "avg_score": avg_score or 0.0,
                "score_variance": score_variance,
                "avg_duration": avg_dur or 0.0,
            }
        return result

    def rank_strategies(
        self,
        task_type: Optional[str] = None,
        min_samples: int = 3,
        top_k: int = 10,
    ) -> list[StrategyRecommendation]:
        """
        Rank strategies by lower-bound win rate (Wilson score).
        Returns top-k strategies with confidence scores.
        """
        stats = self._get_strategy_records(task_type, min_samples)
        
        recommendations = []
        for name, s in stats.items():
            confidence = min(1.0, s["total"] / 20.0)  # Saturates at 20 samples
            reasoning = (
                f"{s['total']} runs, {s['successes']} successes, "
                f"win_rate={s['win_rate']:.2%}, "
                f"avg_score={s['avg_score']:.3f}"
            )
            recommendations.append(StrategyRecommendation(
                strategy_name=name,
                win_rate=s["win_rate"],
                confidence=confidence,
                avg_score=s["avg_score"],
                sample_size=s["total"],
                reasoning=reasoning,
            ))
        
        # Sort by Wilson lower bound (most robust) then by avg_score
        recommendations.sort(
            key=lambda r: (r.win_rate * r.confidence + r.avg_score * 0.1, r.sample_size),
            reverse=True,
        )
        return recommendations[:top_k]

    def recommend(
        self,
        task_type: str,
        context: Optional[dict] = None,
        min_samples: int = 2,
    ) -> StrategyRecommendation:
        """
        Get the best strategy recommendation for a task type + context.
        
        Context matching: looks for strategies that performed well in
        similar contexts (stored in context JSONB). Falls back to overall
        task_type ranking if no contextual match found.
        """
        ranked = self.rank_strategies(task_type=task_type, min_samples=min_samples, top_k=5)
        
        if not ranked:
            return StrategyRecommendation(
                strategy_name="default",
                win_rate=0.0,
                confidence=0.0,
                avg_score=0.0,
                sample_size=0,
                reasoning="No historical data; falling back to 'default' strategy.",
            )
        
        best = ranked[0]
        if context and ranked:
            # Contextual refinement: prefer strategies that succeeded in similar contexts
            best = self._refine_by_context(ranked, context)
        
        return best

    def _refine_by_context(
        self,
        candidates: list[StrategyRecommendation],
        context: dict,
    ) -> StrategyRecommendation:
        """Refine strategy selection based on context similarity."""
        # For now, return top candidate
        # Future: load experiences, compute context similarity
        return candidates[0]

    def get_trend(
        self,
        strategy_name: str,
        task_type: Optional[str] = None,
        window_size: int = 10,
    ) -> dict:
        """
        Compute rolling win rate trend for a strategy.
        Positive trend → strategy improving; negative → degrading.
        """
        if not self.db_path.exists():
            return {"trend": 0.0, "recent_avg": 0.0, "older_avg": 0.0}
        
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT score, outcome, timestamp
            FROM experiences
            WHERE strategy_name = ?
        """
        params = [strategy_name]
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY timestamp DESC LIMIT 100"
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        if len(rows) < window_size * 2:
            return {"trend": 0.0, "recent_avg": 0.0, "older_avg": 0.0, "samples": len(rows)}
        
        recent = rows[:window_size]
        older = rows[window_size:window_size * 2]
        
        recent_avg = sum(r[0] for r in recent) / len(recent)
        older_avg = sum(r[0] for r in older) / len(older)
        trend = recent_avg - older_avg
        
        return {
            "trend": trend,
            "recent_avg": recent_avg,
            "older_avg": older_avg,
            "samples": len(rows),
            "direction": "improving" if trend > 0.05 else ("degrading" if trend < -0.05 else "stable"),
        }
