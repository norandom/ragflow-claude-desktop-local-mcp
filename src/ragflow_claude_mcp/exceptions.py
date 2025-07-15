"""
Custom exceptions for RAGFlow MCP Server.

This module defines specific exception types for better error handling
and debugging throughout the application.
"""

from typing import Optional, Dict, Any


class RAGFlowMCPError(Exception):
    """Base exception for all RAGFlow MCP Server errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}


class RAGFlowAPIError(RAGFlowMCPError):
    """Raised when RAGFlow API returns an error response."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_text: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.status_code = status_code
        self.response_text = response_text


class RAGFlowConnectionError(RAGFlowMCPError):
    """Raised when unable to connect to RAGFlow API."""
    pass


class ConfigurationError(RAGFlowMCPError):
    """Raised when there's an issue with configuration."""
    pass


class ValidationError(RAGFlowMCPError):
    """Raised when input validation fails."""
    pass


class DatasetNotFoundError(RAGFlowMCPError):
    """Raised when a requested dataset cannot be found."""
    
    def __init__(self, dataset_name: str, available_datasets: list = None):
        message = f"Dataset '{dataset_name}' not found"
        if available_datasets:
            message += f". Available datasets: {available_datasets}"
        super().__init__(message)
        self.dataset_name = dataset_name
        self.available_datasets = available_datasets or []


class DocumentNotFoundError(RAGFlowMCPError):
    """Raised when a requested document cannot be found."""
    
    def __init__(self, document_name: str, dataset_id: str):
        message = f"Document '{document_name}' not found in dataset {dataset_id}"
        super().__init__(message)
        self.document_name = document_name
        self.dataset_id = dataset_id


class DSPyConfigurationError(RAGFlowMCPError):
    """Raised when DSPy configuration fails."""
    pass


class CacheError(RAGFlowMCPError):
    """Raised when cache operations fail."""
    pass