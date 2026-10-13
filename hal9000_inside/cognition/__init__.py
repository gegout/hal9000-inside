"""HAL 9000 Inside Cognition Package."""

from hal9000_inside.cognition.attention import AttentionTracker
from hal9000_inside.cognition.curiosity import CuriosityEngine
from hal9000_inside.cognition.digest import (
    FactToRemember,
    TurnDigestAnalyzer,
    TurnDigestResult,
    UserCorrection,
)
from hal9000_inside.cognition.interoception import InteroceptionMonitor
from hal9000_inside.cognition.prediction import PredictionEngine
from hal9000_inside.cognition.scheduler import BackgroundJob, CognitiveScheduler
from hal9000_inside.cognition.sleep import SleepBudget, SleepManager, SleepSummary

__all__ = [
    "AttentionTracker",
    "BackgroundJob",
    "CognitiveScheduler",
    "CuriosityEngine",
    "FactToRemember",
    "InteroceptionMonitor",
    "PredictionEngine",
    "SleepBudget",
    "SleepManager",
    "SleepSummary",
    "TurnDigestAnalyzer",
    "TurnDigestResult",
    "UserCorrection",
]
