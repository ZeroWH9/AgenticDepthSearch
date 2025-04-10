"""
Cache module for search optimization
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from src.utils.config import settings

class Cache:
    """Cache system for search results"""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize the cache
        
        Args:
            ttl: Time to live in seconds
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get item from cache
        
        Args:
            key: Item key
            
        Returns:
            Optional[Dict[str, Any]]: Cache item or None
        """
        if key not in self._cache:
            return None
        
        item = self._cache[key]
        if self._is_expired(item["timestamp"]):
            del self._cache[key]
            return None
        
        return item["data"]
    
    def set(self, key: str, data: Dict[str, Any]):
        """
        Store item in cache
        
        Args:
            key: Item key
            data: Data to store
        """
        self._cache[key] = {
            "data": data,
            "timestamp": datetime.now()
        }
    
    def clear(self):
        """Clear the cache"""
        self._cache.clear()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """
        Check if item is expired
        
        Args:
            timestamp: Item creation date
            
        Returns:
            bool: True if expired
        """
        return datetime.now() - timestamp > timedelta(seconds=self._ttl)

# Global cache instance
cache = Cache(ttl=settings.cache_ttl) 