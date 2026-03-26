"""
TrustScore Pipeline - OpenAI Provider Module

This module implements the OpenAI/ChatGPT provider for LLM interactions.
"""

import threading
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, RateLimitError
from modules.llm_providers.base_llm import BaseLLMProvider

_MAX_RETRIES = 6
_BACKOFF_BASE = 2  # seconds — doubles each retry (2, 4, 8, 16, 32, 64)

# Global semaphore capping simultaneous in-flight API calls across all threads.
# Prevents bursting past the OpenAI RPM limit when sample-level parallelism
# is enabled. Call set_max_concurrent_api_calls() before run() to override.
_api_semaphore = threading.Semaphore(8)


def set_max_concurrent_api_calls(n: int) -> None:
    """
    Set the maximum number of simultaneous OpenAI API calls allowed across
    all threads. Call this before run() when using sample-level parallelism.

    Recommended values:
      Tier 1 (500 RPM):  8–10
      Tier 2 (5000 RPM): 50–80
    """
    global _api_semaphore
    _api_semaphore = threading.Semaphore(n)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI/ChatGPT provider implementation"""

    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key) if config.api_key else None

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API with exponential backoff on rate limits."""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")

        model = self.config.fine_tuned_model or self.config.model

        for attempt in range(_MAX_RETRIES):
            try:
                with _api_semaphore:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        **kwargs
                    )
                return response.choices[0].message.content
            except RateLimitError as e:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = _BACKOFF_BASE ** attempt
                print(f"[OpenAI] Rate limit hit, retrying in {wait}s (attempt {attempt + 1}/{_MAX_RETRIES})...")
                time.sleep(wait)

    def batch_generate(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Generate responses for multiple message sets"""
        results = []
        for messages in messages_list:
            try:
                result = self.generate(messages, **kwargs)
                results.append(result)
            except Exception as e:
                results.append(f"Error: {str(e)}")
        return results

    def is_available(self) -> bool:
        """Check if OpenAI provider is available"""
        return self.client is not None
