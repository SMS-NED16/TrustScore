"""
TrustScore Pipeline - LLM Provider Factory Module

This module provides a factory for creating LLM providers based on configuration.
"""

from typing import Optional
from modules.llm_providers.base_llm import BaseLLMProvider
from modules.llm_providers.openai_provider import OpenAIProvider
from modules.llm_providers.llama_provider import LLaMAProvider

class LLMProviderFactory:
    """Factory for creating LLM providers"""
    
    @staticmethod
    def create_provider(config) -> BaseLLMProvider:
        """Create LLM provider based on configuration"""
        if config.provider == "openai":
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
