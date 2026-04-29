"""Backward-compat shim. The real definitions live in ragflow_claude_mcp.common.exceptions."""

from .common.exceptions import (
    RAGFlowMCPError, RAGFlowAPIError, RAGFlowConnectionError,
    ConfigurationError, ValidationError, DatasetNotFoundError,
    DocumentNotFoundError, DSPyConfigurationError, CacheError,
)

__all__ = [
    "RAGFlowMCPError", "RAGFlowAPIError", "RAGFlowConnectionError",
    "ConfigurationError", "ValidationError", "DatasetNotFoundError",
    "DocumentNotFoundError", "DSPyConfigurationError", "CacheError",
]
