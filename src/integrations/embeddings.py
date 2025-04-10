"""
Embeddings module.
"""
from typing import List, Optional
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
import requests
from src.utils.config import settings

class LocalEmbeddings(Embeddings):
    """Class for local embeddings using LM Studio."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize local embedding.
        
        Args:
            base_url: Base API URL (optional)
            api_key: API key (optional)
            model_name: Model name (optional)
        """
        self.base_url = base_url or settings.local.base_url
        self.api_key = api_key or settings.local.api_key
        self.model_name = model_name or settings.local.embedding_model
        
        print(f"Local Embedding Settings:")
        print(f"Base URL: {self.base_url}")
        print(f"Model Name: {self.model_name}")
        
        if not self.base_url:
            raise ValueError("Base URL not configured")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings
        """
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Text embedding
        """
        return self._get_embedding(text)
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using local API.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Text embedding
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": text,
            "model": self.model_name
        }
        
        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        )
        
        if response.status_code != 200:
            raise ValueError(f"Error generating embedding: {response.text}")
        
        return response.json()["data"][0]["embedding"]

def get_embeddings() -> Embeddings:
    """
    Returns an embedding instance based on settings.
    
    Returns:
        Embeddings: Embedding instance
    """
    if settings.embedding.embedding_type == "openai":
        return OpenAIEmbeddings(
            openai_api_key=settings.openai.api_key,
            openai_api_base=settings.openai.base_url,
            model=settings.embedding.model_name
        )
    else:
        return LocalEmbeddings() 