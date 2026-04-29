"""Shared utilities: exceptions, validation, caching, logging."""

from .exceptions import (
    RAGFlowMCPError, RAGFlowAPIError, RAGFlowConnectionError,
    ConfigurationError, ValidationError, DatasetNotFoundError,
    DocumentNotFoundError, DSPyConfigurationError, CacheError,
)
from .cache import TTLCache, DatasetCache
from .validation import (
    validate_dataset_id, validate_dataset_ids, validate_document_id,
    validate_query, validate_dataset_name, validate_dataset_names,
    validate_document_name, validate_pagination_params,
    validate_similarity_threshold, validate_top_k, validate_deepening_level,
    redact_sensitive_data,
)
from .logging_config import (
    setup_logging, get_logger,
    log_info, log_warning, log_error, log_debug,
    log_with_context,
)

__all__ = [
    # exceptions
    "RAGFlowMCPError", "RAGFlowAPIError", "RAGFlowConnectionError",
    "ConfigurationError", "ValidationError", "DatasetNotFoundError",
    "DocumentNotFoundError", "DSPyConfigurationError", "CacheError",
    # cache
    "TTLCache", "DatasetCache",
    # validation
    "validate_dataset_id", "validate_dataset_ids", "validate_document_id",
    "validate_query", "validate_dataset_name", "validate_dataset_names",
    "validate_document_name", "validate_pagination_params",
    "validate_similarity_threshold", "validate_top_k", "validate_deepening_level",
    "redact_sensitive_data",
    # logging
    "setup_logging", "get_logger",
    "log_info", "log_warning", "log_error", "log_debug", "log_with_context",
]
