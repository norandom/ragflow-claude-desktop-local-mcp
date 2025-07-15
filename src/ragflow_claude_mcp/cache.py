"""
Cache management utilities for RAGFlow MCP Server.

This module provides caching functionality with TTL (Time To Live)
and size limits to improve performance and prevent memory leaks.
"""

import time
from typing import Any, Dict, Optional, Tuple
from .exceptions import CacheError


class TTLCache:
    """
    A simple cache implementation with TTL and size limits.
    
    This cache automatically expires entries after a specified time
    and enforces a maximum size limit using LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of entries to store
            default_ttl: Default TTL in seconds (1 hour by default)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float, float]] = {}  # key: (value, expiry_time, access_time)
        self._access_order: Dict[str, float] = {}  # Track access order for LRU
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        if key not in self._cache:
            return None
        
        value, expiry_time, _ = self._cache[key]
        current_time = time.time()
        
        # Check if expired
        if current_time > expiry_time:
            self._remove(key)
            return None
        
        # Update access time for LRU
        self._cache[key] = (value, expiry_time, current_time)
        self._access_order[key] = current_time
        
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: TTL in seconds (uses default if not specified)
        """
        if ttl is None:
            ttl = self.default_ttl
        
        current_time = time.time()
        expiry_time = current_time + ttl
        
        # Add/update the entry
        self._cache[key] = (value, expiry_time, current_time)
        self._access_order[key] = current_time
        
        # Enforce size limit
        self._enforce_size_limit()
    
    def delete(self, key: str) -> bool:
        """
        Delete a specific key from the cache.
        
        Args:
            key: The cache key to delete
            
        Returns:
            True if the key was found and deleted, False otherwise
        """
        if key in self._cache:
            self._remove(key)
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_, expiry_time, _) in self._cache.items()
            if current_time > expiry_time
        ]
        
        for key in expired_keys:
            self._remove(key)
        
        return len(expired_keys)
    
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        current_time = time.time()
        expired_count = sum(
            1 for _, expiry_time, _ in self._cache.values()
            if current_time > expiry_time
        )
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'expired_entries': expired_count,
            'utilization': len(self._cache) / self.max_size if self.max_size > 0 else 0
        }
    
    def _remove(self, key: str) -> None:
        """Remove a key from both cache and access order tracking."""
        self._cache.pop(key, None)
        self._access_order.pop(key, None)
    
    def _enforce_size_limit(self) -> None:
        """Enforce the maximum cache size using LRU eviction."""
        while len(self._cache) > self.max_size:
            # Find the least recently accessed key
            if not self._access_order:
                break
            
            lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
            self._remove(lru_key)


class DatasetCache:
    """
    Specialized cache for dataset information with automatic cleanup.
    """
    
    def __init__(self, ttl: float = 1800):  # 30 minutes default TTL
        """
        Initialize the dataset cache.
        
        Args:
            ttl: TTL for dataset entries in seconds
        """
        self._cache = TTLCache(max_size=500, default_ttl=ttl)
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # Cleanup every 5 minutes
    
    def get_dataset_id(self, name: str) -> Optional[str]:
        """
        Get dataset ID by name.
        
        Args:
            name: Dataset name (case-insensitive)
            
        Returns:
            Dataset ID or None if not found
        """
        self._maybe_cleanup()
        return self._cache.get(name.lower())
    
    def set_datasets(self, datasets: Dict[str, str]) -> None:
        """
        Update the cache with a new dataset mapping.
        
        Args:
            datasets: Dictionary mapping dataset names to IDs
        """
        for name, dataset_id in datasets.items():
            self._cache.set(name.lower(), dataset_id)
    
    def get_all_names(self) -> list:
        """
        Get all cached dataset names.
        
        Returns:
            List of dataset names
        """
        self._maybe_cleanup()
        # Only return non-expired entries
        current_time = time.time()
        valid_names = []
        
        for key in list(self._cache._cache.keys()):
            _, expiry_time, _ = self._cache._cache[key]
            if current_time <= expiry_time:
                valid_names.append(key)
        
        return valid_names
    
    def clear(self) -> None:
        """Clear the dataset cache."""
        self._cache.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._cache.stats()
    
    def _maybe_cleanup(self) -> None:
        """Perform cleanup if enough time has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup > self._cleanup_interval:
            expired_count = self._cache.cleanup_expired()
            self._last_cleanup = current_time
            if expired_count > 0:
                # Could add logging here if needed
                pass