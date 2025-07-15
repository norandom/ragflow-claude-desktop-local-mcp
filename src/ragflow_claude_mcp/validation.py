"""
Input validation utilities for RAGFlow MCP Server.

This module provides validation functions to ensure data integrity
and security for all user inputs and API parameters.
"""

import re
from typing import Any, Dict, Optional
from .exceptions import ValidationError


def validate_dataset_id(dataset_id: str) -> str:
    """
    Validate dataset ID format.
    
    Args:
        dataset_id: The dataset ID to validate
        
    Returns:
        The validated dataset ID
        
    Raises:
        ValidationError: If the dataset ID is invalid
    """
    if not dataset_id or not isinstance(dataset_id, str):
        raise ValidationError("Dataset ID must be a non-empty string")
    
    # Dataset IDs should be alphanumeric with possible hyphens/underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', dataset_id):
        raise ValidationError("Dataset ID contains invalid characters")
    
    if len(dataset_id) > 100:
        raise ValidationError("Dataset ID too long (max 100 characters)")
    
    return dataset_id.strip()


def validate_document_id(document_id: str) -> str:
    """
    Validate document ID format.
    
    Args:
        document_id: The document ID to validate
        
    Returns:
        The validated document ID
        
    Raises:
        ValidationError: If the document ID is invalid
    """
    if not document_id or not isinstance(document_id, str):
        raise ValidationError("Document ID must be a non-empty string")
    
    # Document IDs should be alphanumeric with possible hyphens/underscores
    if not re.match(r'^[a-zA-Z0-9_-]+$', document_id):
        raise ValidationError("Document ID contains invalid characters")
    
    if len(document_id) > 100:
        raise ValidationError("Document ID too long (max 100 characters)")
    
    return document_id.strip()


def validate_query(query: str) -> str:
    """
    Validate and sanitize search query.
    
    Args:
        query: The search query to validate
        
    Returns:
        The validated and sanitized query
        
    Raises:
        ValidationError: If the query is invalid
    """
    if not query or not isinstance(query, str):
        raise ValidationError("Query must be a non-empty string")
    
    query = query.strip()
    
    if len(query) < 2:
        raise ValidationError("Query must be at least 2 characters long")
    
    if len(query) > 1000:
        raise ValidationError("Query too long (max 1000 characters)")
    
    # Remove potentially dangerous characters but keep most punctuation for search
    # This is a basic sanitization - adjust based on your search requirements
    sanitized_query = re.sub(r'[<>&"\'`]', '', query)
    
    return sanitized_query


def validate_dataset_name(name: str) -> str:
    """
    Validate dataset name.
    
    Args:
        name: The dataset name to validate
        
    Returns:
        The validated dataset name
        
    Raises:
        ValidationError: If the name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Dataset name must be a non-empty string")
    
    name = name.strip()
    
    if len(name) > 200:
        raise ValidationError("Dataset name too long (max 200 characters)")
    
    return name


def validate_document_name(name: str) -> str:
    """
    Validate document name.
    
    Args:
        name: The document name to validate
        
    Returns:
        The validated document name
        
    Raises:
        ValidationError: If the name is invalid
    """
    if not name or not isinstance(name, str):
        raise ValidationError("Document name must be a non-empty string")
    
    name = name.strip()
    
    if len(name) > 200:
        raise ValidationError("Document name too long (max 200 characters)")
    
    return name


def validate_pagination_params(page: Optional[int] = None, page_size: Optional[int] = None) -> Dict[str, int]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number (1-based)
        page_size: Number of items per page
        
    Returns:
        Dictionary with validated page and page_size
        
    Raises:
        ValidationError: If parameters are invalid
    """
    validated = {}
    
    if page is not None:
        if not isinstance(page, int) or page < 1:
            raise ValidationError("Page must be a positive integer starting from 1")
        if page > 1000:
            raise ValidationError("Page number too high (max 1000)")
        validated['page'] = page
    
    if page_size is not None:
        if not isinstance(page_size, int) or page_size < 1:
            raise ValidationError("Page size must be a positive integer")
        if page_size > 100:
            raise ValidationError("Page size too large (max 100)")
        validated['page_size'] = page_size
    
    return validated


def validate_similarity_threshold(threshold: Optional[float] = None) -> Optional[float]:
    """
    Validate similarity threshold parameter.
    
    Args:
        threshold: Similarity threshold (0.0 to 1.0)
        
    Returns:
        The validated threshold or None
        
    Raises:
        ValidationError: If threshold is invalid
    """
    if threshold is None:
        return None
    
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Similarity threshold must be a number")
    
    if not 0.0 <= threshold <= 1.0:
        raise ValidationError("Similarity threshold must be between 0.0 and 1.0")
    
    return float(threshold)


def validate_top_k(top_k: Optional[int] = None) -> Optional[int]:
    """
    Validate top_k parameter.
    
    Args:
        top_k: Number of chunks for vector computation
        
    Returns:
        The validated top_k or None
        
    Raises:
        ValidationError: If top_k is invalid
    """
    if top_k is None:
        return None
    
    if not isinstance(top_k, int) or top_k < 1:
        raise ValidationError("top_k must be a positive integer")
    
    if top_k > 10000:
        raise ValidationError("top_k too large (max 10000)")
    
    return top_k


def validate_deepening_level(level: Optional[int] = None) -> Optional[int]:
    """
    Validate deepening level parameter.
    
    Args:
        level: DSPy deepening level (0-3)
        
    Returns:
        The validated level or None
        
    Raises:
        ValidationError: If level is invalid
    """
    if level is None:
        return None
    
    if not isinstance(level, int):
        raise ValidationError("Deepening level must be an integer")
    
    if not 0 <= level <= 3:
        raise ValidationError("Deepening level must be between 0 and 3")
    
    return level


def redact_sensitive_data(data: Any, sensitive_keys: Optional[list] = None) -> Any:
    """
    Redact sensitive information from data structures for logging.
    
    Args:
        data: The data to redact
        sensitive_keys: List of keys to redact (default includes common sensitive keys)
        
    Returns:
        Data with sensitive information redacted
    """
    if sensitive_keys is None:
        sensitive_keys = [
            'api_key', 'token', 'password', 'secret', 'key', 'authorization',
            'RAGFLOW_API_KEY', 'OPENAI_API_KEY', 'OPENROUTER_API_KEY'
        ]
    
    if isinstance(data, dict):
        return {
            key: '[REDACTED]' if any(sensitive in key.lower() for sensitive in sensitive_keys) else redact_sensitive_data(value, sensitive_keys)
            for key, value in data.items()
        }
    elif isinstance(data, list):
        return [redact_sensitive_data(item, sensitive_keys) for item in data]
    elif isinstance(data, str) and any(sensitive in data.lower() for sensitive in ['bearer ', 'token ', 'key=']):
        return '[REDACTED]'
    else:
        return data