"""
LLM (Large Language Model) integration module
"""
from typing import Optional, Dict, Any, List, Union, Mapping
from langchain_community.chat_models import ChatOpenAI, ChatAnthropic
from langchain_community.llms import OpenAI, Anthropic
from langchain.schema import BaseMessage
from langchain_core.language_models.llms import LLM
from src.utils.config import LLMSettings, OpenAISettings, AnthropicSettings, LocalModelSettings, OpenRouterSettings, settings
import logging
import json
import aiohttp

logger = logging.getLogger(__name__)

class OpenRouterLLM(LLM):
    """
    LLM that uses the OpenRouter API directly
    """
    
    model_name: str
    openrouter_api_key: str
    openrouter_api_base: str
    temperature: float = 0.4
    max_tokens: int = 32000
    http_headers: Dict[str, str] = {}
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(self, prompt: str, **kwargs) -> str:
        """
        Synchronous method - not used, but needs to be implemented
        """
        raise NotImplementedError("Use the asynchronous ainvoke method")
    
    async def _acall(self, prompt: str, **kwargs) -> str:
        """
        Makes the asynchronous call to OpenRouter using the Chat Completions endpoint.
        """
        logger.debug(f"Calling OpenRouter chat with prompt: {prompt[:100]}...")
        logger.debug(f"Model: {self.model_name}")
        logger.debug(f"URL Base: {self.openrouter_api_base}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openrouter_api_key}"
        }
        # Add optional headers if provided
        headers.update(self.http_headers or {})

        # Use Chat Completions endpoint and payload format
        url = f"{self.openrouter_api_base}/chat/completions"
        chat_payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        logger.debug(f"Sending POST to {url}")
        logger.debug(f"Chat Payload: {json.dumps(chat_payload)[:500]}...")
        logger.debug(f"Headers: {headers}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=chat_payload
            ) as resp:
                status = resp.status
                response_text = await resp.text()
                logger.debug(f"Status: {status}, Response: {response_text[:500]}...")
                
                if status != 200:
                    logger.error(f"Error in OpenRouter API ({status}): {response_text}")
                    raise Exception(f"Error in OpenRouter API ({status}): {response_text}")
                
                try:
                    result = json.loads(response_text)
                    # Correctly parse the chat completions structure
                    if "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                        choice = result["choices"][0]
                        if "message" in choice and isinstance(choice["message"], dict) and "content" in choice["message"]:
                            return choice["message"]["content"].strip()
                    
                    # If expected chat structure not found, log error and raise
                    logger.error(f"Unexpected JSON structure from OpenRouter (Chat): {result}")
                    raise ValueError(f"Could not extract message content from OpenRouter chat response structure: {list(result.keys())}")

                except json.JSONDecodeError as json_err:
                    logger.error(f"Failed to decode JSON response from OpenRouter: {json_err}")
                    raise ValueError(f"Invalid JSON received from OpenRouter: {response_text[:500]}...")
                except (KeyError, IndexError, TypeError) as structure_err:
                    logger.error(f"Error parsing expected structure from OpenRouter JSON ({structure_err}): {result}")
                    raise ValueError(f"Could not parse expected structure from OpenRouter response: {result}")

class LLMFactory:
    """
    Factory for creating LLM instances
    """
    
    def __init__(
        self,
        llm_settings: Optional[LLMSettings] = None,
        openai_settings: Optional[OpenAISettings] = None,
        anthropic_settings: Optional[AnthropicSettings] = None,
        local_settings: Optional[LocalModelSettings] = None,
        openrouter_settings: Optional[OpenRouterSettings] = None
    ):
        """
        Initialize factory
        
        Args:
            llm_settings: LLM settings
            openai_settings: OpenAI settings
            anthropic_settings: Anthropic settings
            local_settings: Local model settings
            openrouter_settings: OpenRouter settings
        """
        self.llm_settings = llm_settings or settings.llm
        self.openai_settings = openai_settings or settings.openai
        self.anthropic_settings = anthropic_settings or settings.anthropic
        self.local_settings = local_settings or settings.local_model
        self.openrouter_settings = openrouter_settings or settings.openrouter
        
    def get_llm(self, streaming: Optional[bool] = None):
        """
        Get LLM instance based on settings
        
        Args:
            streaming: Whether to enable streaming
            
        Returns:
            LLM instance
        """
        if self.llm_settings.model_type == "openai":
            return self._get_openai_llm(streaming)
        elif self.llm_settings.model_type == "anthropic":
            return self._get_anthropic_llm(streaming)
        elif self.llm_settings.model_type == "local":
            return self._get_local_llm(streaming)
        elif self.llm_settings.model_type == "openrouter":
            return self._get_openrouter_llm(streaming)
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_settings.model_type}")
            
    def _get_openai_llm(self, streaming: Optional[bool] = None):
        """
        Get OpenAI LLM instance
        
        Args:
            streaming: Whether to enable streaming
            
        Returns:
            OpenAI LLM instance (Chat Model)
        """
        return ChatOpenAI(
            model_name=self.llm_settings.model_name,
            temperature=self.llm_settings.temperature,
            max_tokens=self.llm_settings.max_tokens,
            streaming=streaming if streaming is not None else self.llm_settings.streaming,
            openai_api_key=self.openai_settings.api_key,
            openai_api_base=self.openai_settings.base_url
        )
        
    def _get_anthropic_llm(self, streaming: Optional[bool] = None):
        """
        Get Anthropic LLM instance
        
        Args:
            streaming: Whether to enable streaming
            
        Returns:
            Anthropic LLM instance (Chat Model)
        """
        return ChatAnthropic(
            model=self.llm_settings.model_name,
            temperature=self.llm_settings.temperature,
            max_tokens_to_sample=self.llm_settings.max_tokens,
            streaming=streaming if streaming is not None else self.llm_settings.streaming,
            anthropic_api_key=self.anthropic_settings.api_key
        )
        
    def _get_local_llm(self, streaming: Optional[bool] = None):
        """
        Get local LLM instance using OpenAI-compatible API
        
        Args:
            streaming: Whether to enable streaming
            
        Returns:
            Local LLM instance (Chat Model using ChatOpenAI)
        """
        return ChatOpenAI(
            model_name=self.llm_settings.model_name,
            temperature=self.llm_settings.temperature,
            max_tokens=self.llm_settings.max_tokens,
            streaming=streaming if streaming is not None else self.llm_settings.streaming,
            openai_api_key=self.local_settings.api_key,
            openai_api_base=self.local_settings.base_url
        )
        
    def _get_openrouter_llm(self, streaming: Optional[bool] = None):
        """
        Get OpenRouter LLM instance
        
        Args:
            streaming: Whether to enable streaming (Note: OpenRouterLLM currently doesn't support streaming)
            
        Returns:
            OpenRouter LLM instance
        """
        # Set necessary HTTP headers for OpenRouter
        http_headers = {
            "HTTP-Referer": self.openrouter_settings.site_url or "test",
            "X-Title": self.openrouter_settings.title or "test"
        }
        
        # Use the custom implementation for OpenRouter
        return OpenRouterLLM(
            model_name=self.llm_settings.model_name,
            temperature=self.llm_settings.temperature,
            max_tokens=self.llm_settings.max_tokens,
            openrouter_api_key=self.openrouter_settings.api_key,
            openrouter_api_base=self.openrouter_settings.base_url,
            http_headers=http_headers
        )
        
def get_default_llm_factory() -> LLMFactory:
    """
    Get default LLM factory instance
    
    Returns:
        Default LLM factory instance
    """
    return LLMFactory() 