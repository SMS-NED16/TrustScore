"""
TrustScore Pipeline - LLM Providers Module

This module provides multi-provider support for different LLM backends
including OpenAI, LLaMA, vLLM, and other models.
"""

from .base_llm import BaseLLMProvider
from .openai_provider import OpenAIProvider
from .llama_provider import LLaMAProvider

# vLLM is optional - may not be available on all platforms
try:
    from .vllm_provider import VLLMProvider
    VLLM_AVAILABLE = True
except ImportError:
    VLLMProvider = None  # type: ignore
    VLLM_AVAILABLE = False

from .factory import LLMProviderFactory

__all__ = [
    "BaseLLMProvider",
    "OpenAIProvider", 
    "LLaMAProvider",
    "LLMProviderFactory"
]

# Only add VLLMProvider to __all__ if available
if VLLM_AVAILABLE:
    __all__.append("VLLMProvider")
