"""
Configuration module
"""
from typing import Optional, Dict, Any, Literal
from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field, validator
import os
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

@dataclass
class LLMSettings:
    model_type: str = os.getenv("LLM_MODEL_TYPE", "local")
    model_name: str = os.getenv("LLM_MODEL_NAME", "gemma-3-12b-it")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "8192"))
    streaming: bool = os.getenv("LLM_STREAMING", "true").lower() == "true"
    context_window: int = int(os.getenv("LLM_CONTEXT_WINDOW", "8192"))

@dataclass
class OpenAISettings:
    api_key: str = os.getenv("OPENAI_API_KEY", "")
    base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")

@dataclass
class OpenRouterSettings:
    api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    site_url: str = os.getenv("OPENROUTER_SITE_URL", "")
    title: str = os.getenv("OPENROUTER_TITLE", "")
    
@dataclass
class AnthropicSettings:
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")

@dataclass
class LocalModelSettings:
    model_type: str = os.getenv("LOCAL_MODEL_TYPE", "lmstudio")
    api_key: str = os.getenv("LOCAL_API_KEY", "lm-studio")
    base_url: str = os.getenv("LOCAL_BASE_URL", "http://localhost:1234/v1")

@dataclass
class EmbeddingSettings:
    embedding_type: str = os.getenv("EMBEDDING_TYPE", "local").lower()
    model_name: str = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-granite-embedding-278m-multilingual")
    dimensions: Optional[int] = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    base_url: Optional[str] = os.getenv("EMBEDDING_BASE_URL")
    api_key: Optional[str] = os.getenv("EMBEDDING_API_KEY")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL")

    @validator('openai_api_key', always=True)
    def check_openai_key(cls, v, values):
        if values.get('embedding_type') == 'openai' and not v:
            raise ValueError('OPENAI_API_KEY must be set for EMBEDDING_TYPE="openai"')
        return v

@dataclass
class AgentSettings:
    max_iterations: int = int(os.getenv("AGENT_MAX_ITERATIONS", "10"))
    min_tool_score: float = float(os.getenv("AGENT_MIN_TOOL_SCORE", "0.6"))
    context_window: int = int(os.getenv("AGENT_CONTEXT_WINDOW", "8000"))
    memory_size: int = int(os.getenv("AGENT_MEMORY_SIZE", "10"))
    max_parallel_tools: int = int(os.getenv("AGENT_MAX_PARALLEL_TOOLS", "4"))
    auto_adjust: bool = os.getenv("AGENT_AUTO_ADJUST", "true").lower() == "true"
    verbose: bool = os.getenv("AGENT_VERBOSE", "true").lower() == "true"

@dataclass
class ResearchSettings:
    min_confidence: float = float(os.getenv("RESEARCH_MIN_CONFIDENCE", "0.7"))
    max_sources: int = int(os.getenv("RESEARCH_MAX_SOURCES", "20"))
    cross_validation: bool = os.getenv("RESEARCH_CROSS_VALIDATION", "true").lower() == "true"
    auto_expand: bool = os.getenv("RESEARCH_AUTO_EXPAND", "true").lower() == "true"

@dataclass
class SourceEvaluationSettings:
    min_credibility: float = float(os.getenv("SOURCE_MIN_CREDIBILITY", "0.6"))
    max_age_days: int = int(os.getenv("SOURCE_MAX_AGE_DAYS", "365"))
    require_citations: bool = os.getenv("SOURCE_REQUIRE_CITATIONS", "true").lower() == "true"

@dataclass
class TavilySettings:
    api_key: str = os.getenv("TAVILY_API_KEY", "")

@dataclass
class VectorStoreSettings:
    store_type: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    store_path: str = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    collection_name: str = os.getenv("VECTOR_STORE_COLLECTION", "deep_research")

@dataclass
class MemorySettings:
    short_term_size: int = int(os.getenv("SHORT_TERM_MEMORY_SIZE", "10"))
    long_term_enabled: bool = os.getenv("LONG_TERM_MEMORY_ENABLED", "true").lower() == "true"
    enabled: bool = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    max_size: int = int(os.getenv("MEMORY_MAX_SIZE", "1000"))

@dataclass
class FeedbackSettings:
    min_score: float = float(os.getenv("FEEDBACK_MIN_SCORE", "0.7"))
    auto_adjust: bool = os.getenv("FEEDBACK_AUTO_ADJUST", "true").lower() == "true"
    learn_from_errors: bool = os.getenv("FEEDBACK_LEARN_FROM_ERRORS", "true").lower() == "true"

@dataclass
class PerformanceSettings:
    max_parallel_searches: int = int(os.getenv("MAX_PARALLEL_SEARCHES", "3"))
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))
    rate_limit_searches: int = int(os.getenv("RATE_LIMIT_SEARCHES", "100"))
    rate_limit_window: int = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))

@dataclass
class Settings:
    """
    Main settings class
    """
    llm: LLMSettings = field(default_factory=LLMSettings)
    openai: OpenAISettings = field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = field(default_factory=AnthropicSettings)
    openrouter: OpenRouterSettings = field(default_factory=OpenRouterSettings)
    local_model: LocalModelSettings = field(default_factory=LocalModelSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    research: ResearchSettings = field(default_factory=ResearchSettings)
    source_evaluation: SourceEvaluationSettings = field(default_factory=SourceEvaluationSettings)
    tavily: TavilySettings = field(default_factory=TavilySettings)
    vector_store: VectorStoreSettings = field(default_factory=VectorStoreSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    feedback: FeedbackSettings = field(default_factory=FeedbackSettings)
    performance: PerformanceSettings = field(default_factory=PerformanceSettings)
    cache_ttl: int = 3600  # Tempo de vida do cache em segundos (1 hora)
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

settings = Settings()

def get_settings() -> Settings:
    """Returns a settings instance."""
    return settings

settings = get_settings() 