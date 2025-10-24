"""
TrustScore Pipeline - Base LLM Provider Module

This module provides the abstract base class and common functionality
for different LLM providers (OpenAI, LLaMA, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from models.llm_record import LLMRecord

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, config):
        self.config = config
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from messages"""
        pass
    
    @abstractmethod
    def batch_generate(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Generate responses for multiple message sets"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available"""
        pass
    
    def format_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        """Format system and user prompts into message format"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
