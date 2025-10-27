"""
Minimal test to load LLaMA with vLLM
"""

import os
from vllm import LLM, SamplingParams

# Set your HF token (replace with your actual token or use environment variable)
# os.environ["HF_TOKEN"] = "your_token_here"

# Check if token is set
hf_token = os.getenv("HF_TOKEN")
print(f"HF_TOKEN set: {hf_token is not None}")
if hf_token:
    print(f"Token length: {len(hf_token)}")

try:
    print("\n" + "="*70)
    print("Loading LLaMA with vLLM...")
    print("="*70)
    
    # Initialize vLLM with LLaMA
    llm = LLM(
        model="meta-llama/Llama-3.1-8B-Instruct",
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    
    print("✅ Model loaded successfully!")
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=100,
    )
    
    # Test generation
    print("\nTesting generation...")
    prompts = ["What is 2+2?"]
    outputs = llm.generate(prompts, sampling_params)
    
    generated_text = outputs[0].outputs[0].text
    print(f"\n✅ Generation successful!")
    print(f"Prompt: What is 2+2?")
    print(f"Response: {generated_text}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

