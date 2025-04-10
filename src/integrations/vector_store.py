"""
Vector Store integration module for long-term memory
"""
from typing import List, Dict, Optional
from pathlib import Path
import json
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

class VectorStoreManager:
    """Vector Store manager for long-term memory"""
    
    def __init__(
        self,
        persist_directory: str = "./data/vectors",
        collection_name: str = "deep_research",
        embedding_function: Optional[OpenAIEmbeddings] = None
    ):
        """
        Initialize the Vector Store manager
        
        Args:
            persist_directory: Directory for persistence
            collection_name: Collection name
            embedding_function: Embedding function (optional)
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_function = embedding_function or OpenAIEmbeddings()
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )
    
    def add_results(
        self,
        query: str,
        results: List[Dict],
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add search results to vector store
        
        Args:
            query: Original query
            results: List of results
            metadata: Additional metadata
        """
        documents = []
        base_metadata = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        for result in results:
            # Prepare document with text and metadata
            doc = Document(
                page_content=result.get("content", ""),
                metadata={
                    **base_metadata,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0)
                }
            )
            documents.append(doc)
        
        # Add documents to vector store
        self.vector_store.add_documents(documents)
        self.vector_store.persist()
    
    def search_similar(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            k: Number of results
            metadata_filter: Metadata filter
            
        Returns:
            List[Document]: Similar documents
        """
        return self.vector_store.similarity_search(
            query,
            k=k,
            filter=metadata_filter
        )
    
    def get_relevant_history(
        self,
        query: str,
        max_results: int = 3,
        min_score: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve relevant history for a query
        
        Args:
            query: Current query
            max_results: Maximum number of results
            min_score: Minimum similarity score
            
        Returns:
            List[Dict]: Relevant history
        """
        results = self.vector_store.similarity_search_with_score(
            query,
            k=max_results
        )
        
        relevant_history = []
        for doc, score in results:
            if score >= min_score:
                relevant_history.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score
                })
        
        return relevant_history
    
    def clear(self) -> None:
        """Clear all data from vector store"""
        self.vector_store.delete_collection()
        self.vector_store = Chroma(
            persist_directory=str(self.persist_directory),
            collection_name=self.collection_name,
            embedding_function=self.embedding_function
        )

def get_default_vector_store() -> VectorStoreManager:
    """
    Returns a manager instance with default configuration
    
    Returns:
        VectorStoreManager: Configured manager
    """
    return VectorStoreManager() 