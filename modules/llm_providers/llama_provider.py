"""
TrustScore Pipeline - LLaMA Provider Module

This module implements the LLaMA provider for local LLM inference.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from modules.llm_providers.base_llm import BaseLLMProvider

class LLaMAProvider(BaseLLMProvider):
    """LLaMA provider implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load LLaMA model and tokenizer"""
        if not self.config.model_path:
            raise ValueError("Model path required for LLaMA provider")
        
        try:
            # Configure quantization for memory efficiency
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=getattr(torch, self.config.torch_dtype),
                trust_remote_code=True
            )
            
            # Load fine-tuned weights if available
            if self.config.fine_tuned and self.config.fine_tuned_model:
                self.model.load_state_dict(torch.load(self.config.fine_tuned_model))
                
        except Exception as e:
            print(f"Warning: Could not load LLaMA model: {e}")
            self.model = None
            self.tokenizer = None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using LLaMA model"""
        if not self.model or not self.tokenizer:
            raise ValueError("LLaMA model not loaded")
        
        # Convert messages to prompt format
        prompt = self._format_messages(messages)
        
        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
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
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for LLaMA prompt"""
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"<|system|>\n{content}\n"
            elif role == "user":
                prompt += f"<|user|>\n{content}\n"
            elif role == "assistant":
                prompt += f"<|assistant|>\n{content}\n"
        
        prompt += "<|assistant|>\n"
        return prompt
    
    def is_available(self) -> bool:
        """Check if LLaMA provider is available"""
        return self.model is not None and self.tokenizer is not None
