"""
TrustScore Pipeline - vLLM Provider Module

This module implements the vLLM provider for fast batched LLM inference.
"""

from typing import List, Dict, Any, Optional
import os

# Handle vllm import gracefully
try:
    from vllm import LLM, SamplingParams
    from huggingface_hub import login
    VLLM_IMPORTED = True
except ImportError as e:
    VLLM_IMPORTED = False
    _import_error = e
    # Create dummy classes to prevent import errors
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    login = None  # type: ignore

from modules.llm_providers.base_llm import BaseLLMProvider


class VLLMProvider(BaseLLMProvider):
    """vLLM provider implementation for fast batched inference"""
    
    def __init__(self, config):
        if not VLLM_IMPORTED:
            raise ImportError(
                f"vLLM is not available. Original error: {_import_error}. "
                "Please install vllm package. Note: vLLM may not be available on Windows."
            )
        
        super().__init__(config)
        self.llm = None
        self.sampling_params = None
        self._load_model()
    
    def _load_model(self):
        """Load vLLM model"""
        # VLLM can use either model (for HuggingFace IDs) or model_path
        if not self.config.model and not (hasattr(self.config, 'model_path') and self.config.model_path):
            raise ValueError("Model name required for vLLM provider (use HuggingFace model ID)")
        
        # Prefer model_path if available, otherwise use model (HuggingFace ID)
        model_name = getattr(self.config, 'model_path', None) or self.config.model
        
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
                    print(f"[Warning] HF authentication warning: {e}")
            
            # Initialize vLLM with configuration
            # vLLM will pick up HF_TOKEN from environment
            self.llm = LLM(
                model=model_name,
                dtype=self.config.torch_dtype if hasattr(self.config, 'torch_dtype') else "auto",
                tensor_parallel_size=1,  # Single GPU for Colab
                gpu_memory_utilization=0.9,
                max_model_len=4096,  # Increased to handle longer prompts (SummEval samples)
            )
            
            # Configure sampling parameters
            # Use temperature 0.0 for deterministic results if specified
            temp = self.config.temperature if hasattr(self.config, 'temperature') else 0.0
            # Set seed for deterministic generation when temperature=0, None when temperature>0 for randomness
            seed = 42 if temp == 0.0 else None
            self.sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=self.config.max_tokens if hasattr(self.config, 'max_tokens') else 2000,
                top_p=0.95 if temp > 0 else 1.0,  # top_p=1.0 when temperature=0
                seed=seed,  # Deterministic seed for temperature=0, None for temperature>0
            )
            
            print(f"[Info] vLLM model loaded successfully: {model_name}")
            
        except Exception as e:
            print(f"[Error] Failed to load vLLM model: {str(e)}")
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
            temp = kwargs.get('temperature', self.config.temperature if hasattr(self.config, 'temperature') else 0.0)
            # For temperature > 0: use provided seed (random for variability) or None for natural randomness
            # For temperature = 0: use seed 42 for deterministic results (span tagger)
            # This ensures judges produce different outputs even with the same model when using random seeds
            if temp > 0:
                # For judges: use provided seed (which is random) or None for natural randomness
                seed = kwargs.get('seed', None)  # Use provided seed (can be random) or None
            elif 'seed' in kwargs:
                seed = kwargs.get('seed')  # Allow explicit seed for temp=0 (span tagger)
            else:
                seed = 42  # Default deterministic seed for temp=0
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens if hasattr(self.config, 'max_tokens') else 2000),
                top_p=kwargs.get('top_p', 0.95 if temp > 0 else 1.0),
                seed=seed,  # Random seed for temp>0 (variability), deterministic seed for temp=0
            )
        
        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        
        # Extract text from output
        response = outputs[0].outputs[0].text
        return response.strip()
    
    def batch_generate(self, messages_list: List[List[Dict[str, str]]], 
                      seed: Optional[int] = None,
                      seeds: Optional[List[int]] = None,
                      **kwargs) -> List[str]:
        """
        Generate responses for multiple message sets using batched inference.
        
        Args:
            messages_list: List of message sets
            seed: Single seed for all prompts (for backward compatibility)
            seeds: List of seeds, one per prompt (takes precedence over seed)
            **kwargs: Additional generation parameters
        
        Returns:
            List of generated text responses
        """
        if not self.llm:
            raise ValueError("vLLM model not loaded")
        
        # Convert all messages to prompts
        prompts = [self._format_messages_to_prompt(messages) for messages in messages_list]
        
        # Get temperature
        temp = kwargs.get('temperature', self.config.temperature if hasattr(self.config, 'temperature') else 0.0)
        
        # If we have per-item seeds and temperature > 0, generate separately
        # This ensures each prompt gets its own seed for variability
        # vLLM's batch_generate uses ONE SamplingParams for all prompts, so we need
        # to generate separately when we want different seeds per item
        if seeds is not None and len(seeds) == len(prompts) and temp > 0:
            # Generate each prompt separately with its unique seed
            results = []
            for prompt, item_seed in zip(prompts, seeds):
                sampling_params = SamplingParams(
                    temperature=temp,
                    max_tokens=kwargs.get('max_tokens', self.config.max_tokens if hasattr(self.config, 'max_tokens') else 2000),
                    top_p=kwargs.get('top_p', 0.95),
                    seed=item_seed,  # Unique seed per prompt
                )
                outputs = self.llm.generate([prompt], sampling_params)
                results.append(outputs[0].outputs[0].text.strip())
            return results
        
        # Otherwise, use batch generation with single seed (or None)
        # Override sampling params if provided
        sampling_params = self.sampling_params
        if kwargs or seed is not None:
            if temp > 0:
                # For judges: use provided seed or None for natural randomness
                final_seed = seed if seed is not None else None
            elif 'seed' in kwargs:
                final_seed = kwargs.get('seed')
            else:
                final_seed = 42  # Default deterministic seed for temp=0
            
            sampling_params = SamplingParams(
                temperature=temp,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens if hasattr(self.config, 'max_tokens') else 2000),
                top_p=kwargs.get('top_p', 0.95 if temp > 0 else 1.0),
                seed=final_seed,
            )
        
        # Batch generate
        outputs = self.llm.generate(prompts, sampling_params)
        
        # Extract text from outputs
        return [output.outputs[0].text.strip() for output in outputs]
    
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

