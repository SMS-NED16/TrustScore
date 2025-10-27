"""
TrustScore Pipeline - LLM Provider Factory Module

This module provides a factory for creating LLM providers based on configuration.
"""

from typing import Optional, Dict
from modules.llm_providers.base_llm import BaseLLMProvider
from modules.llm_providers.openai_provider import OpenAIProvider
from modules.llm_providers.llama_provider import LLaMAProvider
from modules.llm_providers.vllm_provider import VLLMProvider

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    # Cache for vLLM providers to avoid multiple GPU initializations
    _provider_cache: Dict[str, BaseLLMProvider] = {}
    
    @staticmethod
    def create_provider(config) -> BaseLLMProvider:
        """Create LLM provider based on configuration"""
        # For vLLM, use caching to ensure only one instance is created
        if config.provider == "vllm":
            # Create a cache key from model name
            cache_key = f"vllm_{config.model}"
            
            # Return cached provider if available
            if cache_key in LLMProviderFactory._provider_cache:
                return LLMProviderFactory._provider_cache[cache_key]
            
            # Create new provider and cache it
            provider = VLLMProvider(config)
            LLMProviderFactory._provider_cache[cache_key] = provider
            return provider
        elif config.provider == "openai":
            return OpenAIProvider(config)
        elif config.provider == "llama":
            return LLaMAProvider(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")
    
    @staticmethod
    def create_span_tagger_provider(config) -> BaseLLMProvider:
        """Create provider specifically for span tagger"""
        return LLMProviderFactory.create_provider(config)
    
    @staticmethod
    def create_judge_provider(config) -> BaseLLMProvider:
        """Create provider specifically for judges"""
        return LLMProviderFactory.create_provider(config)
    
    @staticmethod
    def clear_cache() -> None:
        """Clear the provider cache (useful for testing)"""
        LLMProviderFactory._provider_cache.clear()
