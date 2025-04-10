"""
Tavily Search integration module
"""
from typing import List, Dict, Optional, Any
import asyncio
from tavily import TavilyClient
from src.utils.config import settings

class CustomTavilySearch:
    """Custom client for Tavily Search"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the client
        
        Args:
            api_key: Tavily API key
        """
        self.api_key = api_key or settings.tavily_api_key
        if not self.api_key:
            raise ValueError("Tavily API key is required. Please set TAVILY_API_KEY in your environment variables.")
        self.client = TavilyClient(api_key=self.api_key)
    
    async def parallel_search(
        self,
        queries: List[str],
        max_results: int = 5
    ) -> List[List[Dict]]:
        """
        Perform parallel searches for multiple queries
        
        Args:
            queries: List of queries
            max_results: Maximum number of results per query
            
        Returns:
            List[List[Dict]]: Search results
        """
        if not queries:
            print("No queries provided for search")
            return []
            
        try:
            # Create tasks for parallel search
            tasks = [
                self._search(query, max_results)
                for query in queries
                if query and len(query.strip()) > 0
            ]
            
            if not tasks:
                print("No valid queries to search")
                return []
            
            # Execute tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results, handling exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Error in search for query '{queries[i]}': {str(result)}")
                    processed_results.append([])
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            print(f"Error in parallel searches: {str(e)}")
            return [[] for _ in queries]
    
    async def _search(
        self,
        query: str,
        max_results: int
    ) -> List[Dict]:
        """
        Perform a search on Tavily
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List[Dict]: Search results
        """
        if not query or len(query.strip()) == 0:
            print("Empty query provided")
            return []
            
        try:
            # Perform search synchronously (Tavily client is not async)
            response = self.client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results
            )
            
            if not response or "results" not in response:
                print(f"No results found for query: {query}")
                return []
            
            # Process results
            results = []
            for result in response.get("results", []):
                processed = {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "metadata": {
                        "source": result.get("source", ""),
                        "published_date": result.get("published_date", ""),
                        "author": result.get("author", "")
                    }
                }
                results.append(processed)
            
            return results
            
        except Exception as e:
            print(f"Error in search for query '{query}': {str(e)}")
            return []

def get_default_search() -> CustomTavilySearch:
    """
    Returns a search client instance with default configuration
    
    Returns:
        CustomTavilySearch: Configured client
    """
    return CustomTavilySearch() 