"""
TrustScore Pipeline - OpenAI Provider Module

This module implements the OpenAI/ChatGPT provider for LLM interactions.
"""

from typing import List, Dict, Any, Optional
from openai import OpenAI
from modules.llm_providers.base_llm import BaseLLMProvider

class OpenAIProvider(BaseLLMProvider):
    """OpenAI/ChatGPT provider implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.client = OpenAI(api_key=config.api_key) if config.api_key else None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using OpenAI API"""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        # Use fine-tuned model if available
        model = self.config.fine_tuned_model or self.config.model
        
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            **kwargs
        )
        
        return response.choices[0].message.content
    
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
