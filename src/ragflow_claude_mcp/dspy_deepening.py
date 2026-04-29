"""Backward-compat shim. The real module lives at ragflow_claude_mcp.dspy.deepening."""

from .dspy.deepening import DSPyQueryDeepener, get_deepener
from .dspy.deepening import (
    AnalyzeSearchResults, RefineQuery, MergeSearchResults,
)

__all__ = [
    "DSPyQueryDeepener", "get_deepener",
    "AnalyzeSearchResults", "RefineQuery", "MergeSearchResults",
]
