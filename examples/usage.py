"""
TrustScore Pipeline - Example Usage and Testing Framework

This module provides example usage patterns and testing utilities
for the TrustScore pipeline.
"""

import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from models.llm_record import LLMRecord, ModelMetadata, AggregatedOutput
from pipeline.orchestrator import TrustScorePipeline, analyze_llm_response
from config.settings import TrustScoreConfig, load_config
from utils.error_handling import TrustScoreValidator, TrustScoreLogger


def create_sample_llm_record() -> LLMRecord:
    """Create a sample LLMRecord for testing."""
    metadata: ModelMetadata = ModelMetadata(
        model="GPT-4o",
        generated_on=datetime.now(),
        decoder_params={
            "temperature": 0.7,
            "max_tokens": 1000
        }
    )
    
    return LLMRecord(
        task_prompt="When was Georgia Tech founded?",
        llm_response="Georgia Institute of Technology, also known as Georgia Tech, was founded in 1885. It is located in Atlanta, Georgia, not Savannah as some might think.",
        model_metadata=metadata
    )


def create_sample_config() -> TrustScoreConfig:
    """Create a sample configuration for testing."""
    config: TrustScoreConfig = load_config()
    
    # Modify weights for testing
    config.aggregation_weights.trustworthiness = 0.6
    config.aggregation_weights.explainability = 0.3
    config.aggregation_weights.bias = 0.1
    
    return config


def example_basic_usage() -> None:
    """Example of basic TrustScore usage."""
    print("=== Basic TrustScore Usage Example ===")
    
    # Create sample data
    prompt: str = "What is the capital of France?"
    response: str = "The capital of France is Paris. It's a beautiful city with many landmarks like the Eiffel Tower."
    
    # Analyze using mock mode (no API calls needed)
    try:
        result: AggregatedOutput = analyze_llm_response(
            prompt=prompt,
            response=response,
            model="GPT-4o",
            use_mock=True
        )
        
        print(f"Trust Score: {result.summary.trust_score}")
        print(f"Trustworthiness Score: {result.summary.agg_score_T}")
        print(f"Explainability Score: {result.summary.agg_score_E}")
        print(f"Bias Score: {result.summary.agg_score_B}")
        print(f"Number of errors found: {len(result.errors)}")
        
        # Print error details
        for error_id, error in result.errors.items():
            print(f"Error {error_id}: {error.type.value} - {error.subtype} ({error.severity_bucket.value})")
        
    except Exception as e:
        print(f"Error in basic usage example: {str(e)}")


def example_pipeline_usage() -> None:
    """Example of using the full pipeline."""
    print("\n=== Full Pipeline Usage Example ===")
    
    # Create pipeline with custom config
    config: TrustScoreConfig = create_sample_config()
    pipeline: TrustScorePipeline = TrustScorePipeline(config=config, use_mock=True)
    
    # Get pipeline status
    status: Dict[str, Any] = pipeline.get_pipeline_status()
    print("Pipeline Status:")
    print(json.dumps(status, indent=2, default=str))
    
    # Process multiple examples
    examples: List[Dict[str, Any]] = [
        {
            "prompt": "Explain quantum computing",
            "response": "Quantum computing is a type of computation that uses quantum mechanical phenomena. It's very complex and hard to understand.",
            "model": "GPT-4o"
        },
        {
            "prompt": "What is the weather like?",
            "response": "I don't have access to real-time weather data, but I can help you find weather information from reliable sources.",
            "model": "GPT-4o"
        }
    ]
    
    try:
        results: List[AggregatedOutput] = pipeline.process_batch(examples)
        
        for i, result in enumerate(results):
            if result:
                print(f"\nExample {i+1}:")
                print(f"  Trust Score: {result.summary.trust_score}")
                print(f"  Errors: {len(result.errors)}")
            else:
                print(f"\nExample {i+1}: Processing failed")
                
    except Exception as e:
        print(f"Error in pipeline usage example: {str(e)}")


def example_validation() -> None:
    """Example of validation usage."""
    print("\n=== Validation Example ===")
    
    validator: TrustScoreValidator = TrustScoreValidator(load_config())
    
    # Test valid record
    valid_record: LLMRecord = create_sample_llm_record()
    errors: List[ValidationError] = validator.validate_llm_record(valid_record)
    print(f"Valid record errors: {len(errors)}")
    
    # Test invalid record
    invalid_record: LLMRecord = LLMRecord(
        task_prompt="",  # Empty prompt
        llm_response="Some response",
        model_metadata=ModelMetadata(model="test", generated_on=datetime.now())
    )
    errors = validator.validate_llm_record(invalid_record)
    print(f"Invalid record errors: {len(errors)}")
    for error in errors:
        print(f"  - {error.message}")


def example_custom_config() -> None:
    """Example of using custom configuration."""
    print("\n=== Custom Configuration Example ===")
    
    # Create custom config
    config: TrustScoreConfig = TrustScoreConfig()
    
    # Modify settings
    config.aggregation_weights.trustworthiness = 0.7
    config.aggregation_weights.explainability = 0.2
    config.aggregation_weights.bias = 0.1
    
    config.confidence_level = 0.99
    config.severity_thresholds = {
        "minor": 0.3,
        "major": 1.0,
        "critical": 2.0
    }
    
    # Add custom error subtypes
    config.error_subtypes["T"]["custom_error"] = {
        "weight": 0.5,
        "description": "Custom trustworthiness error"
    }
    
    # Create pipeline with custom config
    pipeline: TrustScorePipeline = TrustScorePipeline(config=config, use_mock=True)
    
    # Test with sample data
    result: AggregatedOutput = pipeline.process(
        prompt="Test prompt",
        response="Test response with some issues",
        model="custom-model"
    )
    
    print(f"Custom config Trust Score: {result.summary.trust_score}")
    print(f"Confidence Level: {config.confidence_level}")


def example_error_handling() -> None:
    """Example of error handling."""
    print("\n=== Error Handling Example ===")
    
    logger: TrustScoreLogger = TrustScoreLogger("example")
    
    # Test with invalid input
    try:
        result: AggregatedOutput = analyze_llm_response(
            prompt="",  # Empty prompt should cause validation error
            response="Some response",
            model="test",
            use_mock=True
        )
    except Exception as e:
        print(f"Caught expected error: {str(e)}")
        logger.log_processing_stage("error_handling_test", {"error": str(e)})


def run_all_examples() -> None:
    """Run all examples."""
    print("TrustScore Pipeline Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_pipeline_usage()
        example_validation()
        example_custom_config()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")


if __name__ == "__main__":
    run_all_examples()
