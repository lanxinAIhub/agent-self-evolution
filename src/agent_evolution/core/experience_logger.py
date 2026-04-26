"""
ExperienceLogger - 经验记录器
Records every task execution: strategy used, context, outcome, and metadata.
Uses SQLite for durable storage with JSONB for flexible context fields.
"""

import json
import sqlite3
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from pathlib import Path
import threading


@dataclass
class ExperienceRecord:
    """Single experience record from a task execution."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    task_id: str = ""
    strategy_name: str = ""
    strategy_params: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    outcome: str = ""          # "success", "failure", "partial"
    score: float = 0.0         # 0.0-1.0 normalized score
    duration_seconds: float = 0.0
    error_message: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)


class ExperienceLogger:
    """
    Records task execution experiences for later analysis.
    
    Thread-safe, append-only log. Data persists in SQLite.
    Designed for high-throughput agent usage (no locks on reads).
    """

    def __init__(self, db_path: str = "~/.agent_evolution/experiences.db"):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    strategy_params TEXT NOT NULL,
                    context TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    score REAL NOT NULL,
                    duration_seconds REAL NOT NULL,
                    error_message TEXT DEFAULT '',
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_task_strategy 
                ON experiences(task_type, strategy_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON experiences(timestamp)
            """)
            conn.commit()
            conn.close()

    def log(self, record: ExperienceRecord) -> str:
        """Log a single experience. Returns record ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                INSERT INTO experiences 
                (id, task_type, task_id, strategy_name, strategy_params, 
                 context, outcome, score, duration_seconds, error_message, 
                 timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.id,
                record.task_type,
                record.task_id,
                record.strategy_name,
                json.dumps(record.strategy_params),
                json.dumps(record.context),
                record.outcome,
                record.score,
                record.duration_seconds,
                record.error_message,
                record.timestamp,
                json.dumps(record.metadata),
            ))
            conn.commit()
            conn.close()
        return record.id

    def log_task(
        self,
        task_type: str,
        task_id: str,
        strategy_name: str,
        strategy_params: Optional[dict] = None,
        context: Optional[dict] = None,
        outcome: str = "success",
        score: float = 1.0,
        duration_seconds: float = 0.0,
        error_message: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """Convenience method to log a task execution directly."""
        record = ExperienceRecord(
            task_type=task_type,
            task_id=task_id,
            strategy_name=strategy_name,
            strategy_params=strategy_params or {},
            context=context or {},
            outcome=outcome,
            score=score,
            duration_seconds=duration_seconds,
            error_message=error_message,
            metadata=metadata or {},
        )
        return self.log(record)

    def get_experiences(
        self,
        task_type: Optional[str] = None,
        strategy_name: Optional[str] = None,
        limit: int = 100,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Retrieve experiences with optional filters."""
        conn = sqlite3.connect(str(self.db_path))
        query = "SELECT * FROM experiences WHERE 1=1"
        params = []
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        if strategy_name:
            query += " AND strategy_name = ?"
            params.append(strategy_name)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        columns = ["id", "task_type", "task_id", "strategy_name", "strategy_params",
                   "context", "outcome", "score", "duration_seconds", "error_message",
                   "timestamp", "metadata"]
        return [dict(zip(columns, row)) for row in rows]

    def get_strategy_stats(self, task_type: Optional[str] = None) -> dict[str, dict]:
        """Aggregate stats per strategy for a task type."""
        conn = sqlite3.connect(str(self.db_path))
        query = """
            SELECT strategy_name, 
                   COUNT(*) as total,
                   SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) as successes,
                   AVG(score) as avg_score,
                   AVG(duration_seconds) as avg_duration
            FROM experiences
        """
        params = []
        if task_type:
            query += " WHERE task_type = ?"
            params.append(task_type)
        query += " GROUP BY strategy_name"
        
        rows = conn.execute(query, params).fetchall()
        conn.close()
        
        return {
            row[0]: {
                "total": row[1],
                "successes": row[2],
                "win_rate": row[2] / row[1] if row[1] > 0 else 0.0,
                "avg_score": row[3] or 0.0,
                "avg_duration": row[4] or 0.0,
            }
            for row in rows
        }

    def count(self) -> int:
        """Total number of logged experiences."""
        conn = sqlite3.connect(str(self.db_path))
        count = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        conn.close()
        return count
