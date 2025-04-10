"""
Memory management module
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from src.integrations.vector_store import VectorStoreManager
from src.integrations.llm import LLMFactory
from src.prompts.memory_prompts import (
    MEMORY_SUMMARY_PROMPT,
    TOPIC_EXTRACTION_PROMPT,
    MEMORY_RELEVANCE_PROMPT
)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from src.utils.config import MemorySettings, settings
from src.utils.metrics import metrics_collector
from src.integrations.embeddings import LocalEmbeddings
import logging

logger = logging.getLogger(__name__)

class DeepResearchMemory:
    """
    Memory system for storing and retrieving research information
    """
    
    def __init__(
        self,
        collection_name: str = "deep_research",
        persist_directory: str = "./data/vector_store"
    ):
        """
        Initialize memory system
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Select and initialize embeddings based on settings
        embedding_config = settings.embedding
        logger.info(f"Configuring embeddings of type: {embedding_config.embedding_type}")

        if embedding_config.embedding_type == "openai":
            logger.info(f"Using OpenAI Embeddings: {embedding_config.model_name}")
            if not embedding_config.openai_api_key:
                 logger.warning("OpenAI API key not set for embeddings, trying to continue...")
            self.embeddings = OpenAIEmbeddings(
                model=embedding_config.model_name,
                openai_api_key=embedding_config.openai_api_key,
                openai_api_base=embedding_config.openai_base_url,
                dimensions=embedding_config.dimensions if embedding_config.dimensions > 0 else None 
            )
        elif embedding_config.embedding_type == "local":
            logger.info(f"Using Local Embeddings (OpenAI-compatible API): {embedding_config.model_name}")

            self.embeddings = LocalEmbeddings(
                base_url=embedding_config.base_url, #
                api_key=embedding_config.api_key,     
                model_name=embedding_config.model_name
            )
        else:
            logger.error(f"Unknown embedding type: {embedding_config.embedding_type}. Using OpenAI as fallback.")
            self.embeddings = OpenAIEmbeddings(
                 model=embedding_config.model_name, # Tenta usar o nome mesmo assim
                 openai_api_key=embedding_config.openai_api_key,
                 openai_api_base=embedding_config.openai_base_url,
                 dimensions=embedding_config.dimensions if embedding_config.dimensions > 0 else None
            )

        # Initialize vector store
        logger.info(f"Initializing ChromaDB at: {persist_directory}")
        try:
            self.vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=persist_directory
            )
        except Exception as e:
            logger.exception(f"Failed to initialize ChromaDB: {e}")
            raise # Re-raise the exception to indicate initialization failure
        
        # Initialize metrics
        self.metrics = metrics_collector
        
    def add_memory(
        self,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add research results to memory
        
        Args:
            query: Original query
            results: Search results
            metadata: Additional metadata
        """
        if not results:
            return
            
        documents = []
        for result in results:
            content = f"Query: {query}\n\n"
            content += f"Title: {result.get('title', '')}\n"
            content += f"Content: {result.get('content', '')}\n"
            content += f"URL: {result.get('url', '')}\n"
            
            doc_metadata = {
                "query": query,
                "timestamp": result.get("timestamp", ""),
                "source": result.get("source", ""),
                "relevance": result.get("relevance", 0.0)
            }
            
            if metadata:
                doc_metadata.update(metadata)
                
            documents.append(
                Document(
                    page_content=content,
                    metadata=doc_metadata
                )
            )
            
        self.vector_store.add_documents(documents)
        
    def get_relevant_memories(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get relevant memories for a query
        
        Args:
            query: Query to search for
            k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of relevant memories
        """
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k
        )
        
        memories = []
        for doc, score in results:
            if score >= min_score:
                memories.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
                
        return memories
        
    def clear(self) -> None:
        """
        Clear all memories
        """
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

class MemoryManager:
    """
    Manager for memory operations
    """
    
    def __init__(
        self,
        memory: Optional[DeepResearchMemory] = None
    ):
        """
        Initialize memory manager
        
        Args:
            memory: Memory system to use
        """
        self.memory = memory or DeepResearchMemory()
        self.settings = MemorySettings()
        
    async def add_to_memory(
        self,
        query: str,
        results: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add results to memory
        
        Args:
            query: Original query
            results: Search results
            metadata: Additional metadata
        """
        self.memory.add_memory(query, results, metadata)
        metrics_collector.record_memory_operation("add", len(results))
        
    async def get_relevant_memories(
        self,
        query: str,
        k: int = 5,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Get relevant memories
        
        Args:
            query: Query to search for
            k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of relevant memories
        """
        memories = self.memory.get_relevant_memories(query, k, min_score)
        metrics_collector.record_memory_operation("retrieve", len(memories))
        return memories
        
    def clear_memory(self) -> None:
        """
        Clear all memories
        """
        self.memory.clear()
        metrics_collector.record_memory_operation("clear", 0)

    def clear_buffer(self) -> None:
        """
        Clear the memory buffer
        """
        self.memory_buffer = []
        
    async def clear_vector_store(self) -> None:
        """
        Clear the vector store
        """
        await self.vector_store.clear()
        
    async def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Memory statistics
        """
        try:
            # Get buffer stats
            buffer_stats = {
                "size": len(self.memory_buffer),
                "oldest": self.memory_buffer[0]["timestamp"] if self.memory_buffer else None,
                "newest": self.memory_buffer[-1]["timestamp"] if self.memory_buffer else None
            }
            
            # Get vector store stats
            vector_store_stats = await self.vector_store.get_stats()
            
            return {
                "buffer": buffer_stats,
                "vector_store": vector_store_stats
            }
            
        except Exception as e:
            print(f"Error getting memory stats: {str(e)}")
            return {
                "error": f"Error getting memory stats: {str(e)}"
            } 