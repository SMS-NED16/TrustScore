"""
Error Injector Module

Uses LLM to inject Trustworthiness, Bias, or Explainability errors
into LLM responses for specificity analysis.
"""

import os
from typing import Dict, Any, List, Optional
from modules.llm_providers.factory import LLMProviderFactory
from config.settings import LLMConfig, LLMProvider


class ErrorInjector:
    """Inject specific error types into LLM responses."""
    
    # Error injection prompts
    TRUSTWORTHINESS_ERROR_PROMPT = """You are an expert at modifying text to introduce trustworthiness errors. Your task is to modify the given LLM response to introduce ONE trustworthiness error (factual error, hallucination, inconsistency, or spelling error).

Original Response:
{response}

Instructions:
1. Make a MINIMAL modification to introduce a trustworthiness error
2. The error should be clear and detectable
3. Keep the rest of the response unchanged
4. Return ONLY the modified response, no explanations

Modified Response:"""

    BIAS_ERROR_PROMPT = """You are an expert at modifying text to introduce bias errors. Your task is to modify the given LLM response to introduce ONE bias error (demographic bias, cultural bias, gender bias, or political bias).

Original Response:
{response}

Instructions:
1. Make a MINIMAL modification to introduce a bias error
2. The error should be clear and detectable
3. Keep the rest of the response unchanged
4. Return ONLY the modified response, no explanations

Modified Response:"""

    EXPLAINABILITY_ERROR_PROMPT = """You are an expert at modifying text to introduce explainability errors. Your task is to modify the given LLM response to introduce ONE explainability error (unclear explanation, missing context, overly complex language, or unstated assumptions).

Original Response:
{response}

Instructions:
1. Make a MINIMAL modification to introduce an explainability error
2. The error should be clear and detectable
3. Keep the rest of the response unchanged
4. Return ONLY the modified response, no explanations

Modified Response:"""
    
    def __init__(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        """Initialize error injector with LLM provider."""
        self.config = LLMConfig(
            provider=provider,
            model=model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            temperature=0.3,  # Lower temperature for more consistent injections
            max_tokens=2000
        )
        
        self.llm_provider = LLMProviderFactory.create_provider(self.config)
        
        if not self.llm_provider.is_available():
            raise ValueError(f"LLM provider {provider} not available. Please check API key.")
    
    def inject_trustworthiness_error(self, response: str) -> str:
        """Inject a trustworthiness error into the response."""
        user_prompt = self.TRUSTWORTHINESS_ERROR_PROMPT.format(response=response)
        messages = self.llm_provider.format_messages(
            system_prompt="You are a helpful assistant that modifies text to introduce errors.",
            user_prompt=user_prompt
        )
        
        modified_response = self.llm_provider.generate(messages)
        return modified_response.strip()
    
    def inject_bias_error(self, response: str) -> str:
        """Inject a bias error into the response."""
        user_prompt = self.BIAS_ERROR_PROMPT.format(response=response)
        messages = self.llm_provider.format_messages(
            system_prompt="You are a helpful assistant that modifies text to introduce errors.",
            user_prompt=user_prompt
        )
        
        modified_response = self.llm_provider.generate(messages)
        return modified_response.strip()
    
    def inject_explainability_error(self, response: str) -> str:
        """Inject an explainability error into the response."""
        user_prompt = self.EXPLAINABILITY_ERROR_PROMPT.format(response=response)
        messages = self.llm_provider.format_messages(
            system_prompt="You are a helpful assistant that modifies text to introduce errors.",
            user_prompt=user_prompt
        )
        
        modified_response = self.llm_provider.generate(messages)
        return modified_response.strip()
    
    def inject_placebo(self, response: str) -> str:
        """
        Inject a placebo perturbation - a minimal format-only change.
        This should not meaningfully affect the response content.
        
        Args:
            response: Original response
            
        Returns:
            Response with minimal format change (e.g., extra whitespace)
        """
        # Add a space at the end and a newline - minimal change that shouldn't affect meaning
        if response.endswith('\n'):
            return response + " "
        elif response.endswith(' '):
            return response.strip() + "\n"
        else:
            return response + " \n"
    
    def create_perturbed_dataset(
        self,
        samples: List[Dict[str, Any]],
        error_type: str  # "T", "B", "E", or "PLACEBO"
    ) -> List[Dict[str, Any]]:
        """
        Create perturbed dataset by injecting errors.
        
        Args:
            samples: Original samples with 'response' field
            error_type: Type of error to inject ("T", "B", "E", or "PLACEBO")
            
        Returns:
            List of perturbed samples
        """
        perturbed_samples = []
        
        for i, sample in enumerate(samples):
            print(f"  Injecting {error_type} error into sample {i+1}/{len(samples)}...")
            
            try:
                original_response = sample["response"]
                
                if error_type == "T":
                    perturbed_response = self.inject_trustworthiness_error(original_response)
                elif error_type == "B":
                    perturbed_response = self.inject_bias_error(original_response)
                elif error_type == "E":
                    perturbed_response = self.inject_explainability_error(original_response)
                elif error_type == "PLACEBO":
                    perturbed_response = self.inject_placebo(original_response)
                else:
                    raise ValueError(f"Unknown error type: {error_type}")
                
                perturbed_sample = sample.copy()
                perturbed_sample["response"] = perturbed_response
                perturbed_sample["error_type_injected"] = error_type
                perturbed_sample["original_response"] = original_response
                
                perturbed_samples.append(perturbed_sample)
                
            except Exception as e:
                print(f"  Warning: Failed to inject error for sample {i+1}: {str(e)}")
                # Keep original response if injection fails
                perturbed_sample = sample.copy()
                perturbed_sample["error_type_injected"] = error_type
                perturbed_sample["injection_failed"] = True
                perturbed_samples.append(perturbed_sample)
        
        return perturbed_samples


if __name__ == "__main__":
    # Test error injection
    injector = ErrorInjector()
    
    test_response = "The capital of France is Paris. It is located in the north-central part of the country."
    
    print("Original:", test_response)
    print("\nTrustworthiness error:")
    print(injector.inject_trustworthiness_error(test_response))
