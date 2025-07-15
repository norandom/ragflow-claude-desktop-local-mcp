"""
Logging configuration for RAGFlow MCP Server.

This module provides structured logging configuration with appropriate
formatting and levels for better observability and debugging.
"""

import logging
import sys
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs structured JSON logs for better parsing.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        # Create base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)
        
        return json.dumps(log_entry, ensure_ascii=False)


class HumanReadableFormatter(logging.Formatter):
    """
    Formatter for human-readable console output.
    """
    
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


def setup_logging(
    level: str = "INFO",
    structured: bool = False,
    log_file: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured JSON logging
        log_file: Optional path to log file
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Choose formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        # Always use structured format for file logs
        file_handler.setFormatter(StructuredFormatter())
        root_logger.addHandler(file_handler)
    
    # Set up specific logger for our application
    app_logger = logging.getLogger("ragflow-mcp")
    app_logger.setLevel(numeric_level)
    
    # Reduce noise from external libraries
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"ragflow-mcp.{name}")


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **context
) -> None:
    """
    Log a message with additional context fields.
    
    Args:
        logger: Logger instance
        level: Log level
        message: Log message
        **context: Additional context fields
    """
    # Create a log record with extra fields
    if context:
        extra = {"extra_fields": context}
        logger.log(level, message, extra=extra)
    else:
        logger.log(level, message)


# Convenience functions for common log levels with context
def log_info(logger: logging.Logger, message: str, **context) -> None:
    """Log info message with context."""
    log_with_context(logger, logging.INFO, message, **context)


def log_warning(logger: logging.Logger, message: str, **context) -> None:
    """Log warning message with context."""
    log_with_context(logger, logging.WARNING, message, **context)


def log_error(logger: logging.Logger, message: str, **context) -> None:
    """Log error message with context."""
    log_with_context(logger, logging.ERROR, message, **context)


def log_debug(logger: logging.Logger, message: str, **context) -> None:
    """Log debug message with context."""
    log_with_context(logger, logging.DEBUG, message, **context)