"""
TrustScore Pipeline - vLLM Provider Module

This module implements the vLLM provider for fast batched LLM inference.
"""

from typing import List, Dict, Any, Optional
from vllm import LLM, SamplingParams
from modules.llm_providers.base_llm import BaseLLMProvider
import os
from huggingface_hub import login


class VLLMProvider(BaseLLMProvider):
    """vLLM provider implementation for fast batched inference"""
    
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.sampling_params = None
        self._load_model()
    
    def _load_model(self):
        """Load vLLM model"""
        if not self.config.model_path and not self.config.model:
            raise ValueError("Model path or model name required for vLLM provider")
        
        model_name = self.config.model_path or self.config.model
        
        try:
            # Try to authenticate with HuggingFace if token is available
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            if hf_token:
                try:
                    login(token=hf_token, add_to_git_credential=True)
                    # Also set as environment variable for vLLM to pick up
                    os.environ["HF_TOKEN"] = hf_token
                    os.environ["HUGGINGFACE_TOKEN"] = hf_token
                except Exception as e:
                    print(f"⚠️  HF authentication warning: {e}")
            
            # Initialize vLLM with configuration
            # vLLM will pick up HF_TOKEN from environment
            self.llm = LLM(
                model=model_name,
                dtype=self.config.torch_dtype if hasattr(self.config, 'torch_dtype') else "auto",
                tensor_parallel_size=1,  # Single GPU for Colab
                gpu_memory_utilization=0.9,
                max_model_len=2048,  # Adjust based on needs
            )
            
            # Configure sampling parameters
            self.sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=0.95,
            )
            
            print(f"✅ vLLM model loaded successfully: {model_name}")
            
        except Exception as e:
            print(f"❌ Failed to load vLLM model: {str(e)}")
            self.llm = None
            self.sampling_params = None
    
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response using vLLM model"""
        if not self.llm:
            raise ValueError("vLLM model not loaded")
        
        # Convert messages to prompt format
        prompt = self._format_messages_to_prompt(messages)
        
        # Override sampling params if provided
        sampling_params = self.sampling_params
        if kwargs:
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                top_p=kwargs.get('top_p', 0.95),
            )
        
        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Extract text from output
        response = outputs[0].outputs[0].text
        return response.strip()
    
    def batch_generate(self, messages_list: List[List[Dict[str, str]]], **kwargs) -> List[str]:
        """Generate responses for multiple message sets using batched inference"""
        if not self.llm:
            raise ValueError("vLLM model not loaded")
        
        # Convert all messages to prompts
        prompts = [self._format_messages_to_prompt(messages) for messages in messages_list]
        
        # Override sampling params if provided
        sampling_params = self.sampling_params
        if kwargs:
            sampling_params = SamplingParams(
                temperature=kwargs.get('temperature', self.config.temperature),
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                top_p=kwargs.get('top_p', 0.95),
            )
        
        # Batch generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text from outputs
        responses = [output.outputs[0].text.strip() for output in outputs]
        return responses
    
    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for LLM prompt using chat template"""
        # Use the tokenizer's apply_chat_template if available, otherwise fallback
        try:
            # Try to use the tokenizer's chat template (supported by vLLM)
            if hasattr(self.llm, 'get_tokenizer') and self.llm.get_tokenizer():
                tokenizer = self.llm.get_tokenizer()
                if hasattr(tokenizer, 'apply_chat_template'):
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    return formatted
        except:
            pass
        
        # Fallback: Detect model and use appropriate template
        model_name = self.config.model_path or self.config.model or ""
        
        # Mistral format
        if "mistral" in model_name.lower():
            prompt = ""
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    prompt += f"<s>[INST] {content}\n"
                elif role == "user":
                    prompt += f"{content} [/INST]\n"
                elif role == "assistant":
                    prompt += f"{content}</s> "
            return prompt
        else:
            # Default Llama format
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
        """Check if vLLM provider is available"""
        return self.llm is not None

