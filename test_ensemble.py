#!/usr/bin/env python3
"""
Test script for the ensemble implementation of TrustScore pipeline.
"""

from pipeline.orchestrator import TrustScorePipeline
from config.settings import load_config
from models.llm_record import ErrorType

def test_ensemble_configuration():
    """Test that the ensemble configuration is loaded correctly."""
    print("Testing ensemble configuration...")
    
    # Load configuration
    config = load_config()
    
    # Check that we have judges for each aspect
    expected_aspects = ["trustworthiness", "bias", "explainability"]
    for aspect in expected_aspects:
        aspect_judges = [name for name in config.judges.keys() if aspect in name.lower()]
        print(f"  {aspect}: {len(aspect_judges)} judges - {aspect_judges}")
        assert len(aspect_judges) >= 3, f"Expected at least 3 judges for {aspect}, got {len(aspect_judges)}"
    
    print("✓ Configuration test passed!")
    return config

def test_pipeline_initialization():
    """Test that the pipeline initializes with ensemble judges."""
    print("\nTesting pipeline initialization...")
    
    # Initialize pipeline in mock mode
    pipeline = TrustScorePipeline(use_mock=True)
    
    # Check that judges are organized by aspect
    assert isinstance(pipeline.judges, dict), "Judges should be a dictionary"
    assert "trustworthiness" in pipeline.judges, "Should have trustworthiness judges"
    assert "bias" in pipeline.judges, "Should have bias judges"
    assert "explainability" in pipeline.judges, "Should have explainability judges"
    
    # Check that each aspect has multiple judges
    for aspect, judges in pipeline.judges.items():
        print(f"  {aspect}: {len(judges)} judges")
        assert len(judges) >= 3, f"Expected at least 3 judges for {aspect}, got {len(judges)}"
    
    # Test pipeline status
    status = pipeline.get_pipeline_status()
    print(f"  Pipeline status: {status['judge_ensemble']}")
    
    print("✓ Pipeline initialization test passed!")
    return pipeline

def test_ensemble_processing():
    """Test that the ensemble processes spans correctly."""
    print("\nTesting ensemble processing...")
    
    # Initialize pipeline in mock mode
    pipeline = TrustScorePipeline(use_mock=True)
    
    # Test with sample data
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris. It is located in the north-central part of the country."
    
    try:
        result = pipeline.process(prompt, response, model="test-model")
        
        # Check that we got a result
        assert result is not None, "Should get a result from processing"
        
        # Check that the result has the expected structure
        assert hasattr(result, 'summary'), "Result should have summary"
        assert hasattr(result, 'errors'), "Result should have errors"
        
        print(f"  Trust Score: {result.summary.trust_score}")
        print(f"  Trustworthiness Score: {result.summary.agg_score_T}")
        print(f"  Explainability Score: {result.summary.agg_score_E}")
        print(f"  Bias Score: {result.summary.agg_score_B}")
        print(f"  Number of errors found: {len(result.errors)}")
        
        print("✓ Ensemble processing test passed!")
        
    except Exception as e:
        print(f"✗ Ensemble processing test failed: {str(e)}")
        raise

def test_ensemble_statistics():
    """Test ensemble statistics methods."""
    print("\nTesting ensemble statistics...")
    
    from models.llm_record import GradedSpan, SpanTag, JudgeAnalysis, JudgeIndicators, JudgeWeights, SeverityBucket
    
    # Create a mock graded span with multiple judge analyses
    span = GradedSpan(
        start=0,
        end=10,
        type=ErrorType.TRUSTWORTHINESS,
        subtype="factual_error",
        explanation="Test error"
    )
    
    # Add mock analyses from multiple judges
    for i in range(3):
        analysis = JudgeAnalysis(
            indicators=JudgeIndicators(
                centrality=0.5 + i * 0.1,
                domain_sensitivity=0.6 + i * 0.05,
                harm_potential=0.3 + i * 0.1,
                instruction_criticality=0.7 + i * 0.05
            ),
            weights=JudgeWeights(),
            confidence=0.8 + i * 0.05,
            severity_score=1.0 + i * 0.2,
            severity_bucket=SeverityBucket.MAJOR
        )
        span.add_judge_analysis(f"judge_{i+1}", analysis)
    
    # Test ensemble statistics
    stats = span.get_ensemble_statistics()
    print(f"  Ensemble statistics: {stats}")
    
    assert stats["judge_count"] == 3, "Should have 3 judges"
    assert "mean_severity" in stats, "Should have mean severity"
    assert "std_severity" in stats, "Should have std severity"
    assert "mean_confidence" in stats, "Should have mean confidence"
    assert "std_confidence" in stats, "Should have std confidence"
    
    # Test robust averages
    robust_severity = span.get_robust_average_severity_score()
    robust_confidence = span.get_robust_average_confidence()
    
    print(f"  Robust average severity: {robust_severity}")
    print(f"  Robust average confidence: {robust_confidence}")
    
    print("✓ Ensemble statistics test passed!")

def main():
    """Run all tests."""
    print("Running TrustScore Ensemble Implementation Tests")
    print("=" * 50)
    
    try:
        # Test configuration
        config = test_ensemble_configuration()
        
        # Test pipeline initialization
        pipeline = test_pipeline_initialization()
        
        # Test ensemble processing
        test_ensemble_processing()
        
        # Test ensemble statistics
        test_ensemble_statistics()
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Ensemble implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
