"""
Agent Self-Evolution System
Lightweight evolutionary algorithms for AI agent strategy optimization.
"""

__version__ = "0.1.0"
__author__ = "Gongbu Ministry"

from .core.experience_logger import ExperienceLogger
from .core.strategy_analyzer import StrategyAnalyzer
from .core.evolution_engine import EvolutionEngine

__all__ = [
    "ExperienceLogger",
    "StrategyAnalyzer", 
    "EvolutionEngine",
]
