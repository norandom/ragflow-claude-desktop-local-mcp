"""
RAGFlow Claude MCP Server

MCP server that exposes a RAGFlow instance to Claude Desktop and other MCP clients.
"""

from .server import main
from .client.ragflow import RAGFlowMCPServer
from .common.exceptions import (
    RAGFlowMCPError, RAGFlowAPIError, RAGFlowConnectionError,
    ConfigurationError, ValidationError, DatasetNotFoundError,
    DocumentNotFoundError, DSPyConfigurationError, CacheError,
)

__version__ = "0.2.0"
__all__ = [
    "main",
    "RAGFlowMCPServer",
    "RAGFlowMCPError",
    "RAGFlowAPIError",
    "RAGFlowConnectionError",
    "ConfigurationError",
    "ValidationError",
    "DatasetNotFoundError",
    "DocumentNotFoundError",
    "DSPyConfigurationError",
    "CacheError",
]
