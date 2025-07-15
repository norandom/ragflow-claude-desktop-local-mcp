"""
RAGFlow Claude MCP Server

A Model Context Protocol server that provides seamless integration between 
tools like Claude Desktop and RAGFlow's REST API for knowledge base querying 
and document management.
"""

from .server import main, RAGFlowMCPServer
from .exceptions import (
    RAGFlowMCPError, RAGFlowAPIError, RAGFlowConnectionError,
    ConfigurationError, ValidationError, DatasetNotFoundError,
    DocumentNotFoundError, DSPyConfigurationError, CacheError
)

__version__ = "1.0.0"
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
    "CacheError"
]
