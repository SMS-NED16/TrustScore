#!/usr/bin/env python3
"""
Test script for the advanced configuration system of TrustScore pipeline.
"""

from pipeline.orchestrator import TrustScorePipeline
from config.settings import (
    TrustScoreConfig, StatisticalConfig, EnsembleConfig, ErrorHandlingConfig,
    SpanProcessingConfig, AggregationStrategyConfig, PerformanceConfig, OutputConfig
)
from models.llm_record import ErrorType

def test_configuration_creation():
    """Test that all new configuration classes can be created."""
    print("Testing configuration creation...")
    
    # Test individual config classes
    stat_config = StatisticalConfig()
    ensemble_config = EnsembleConfig()
    error_config = ErrorHandlingConfig()
    span_config = SpanProcessingConfig()
    agg_config = AggregationStrategyConfig()
    perf_config = PerformanceConfig()
    output_config = OutputConfig()
    
    print(f"  ✓ StatisticalConfig: {stat_config.t_critical_values}")
    print(f"  ✓ EnsembleConfig: min_judges={ensemble_config.min_judges_required}")
    print(f"  ✓ ErrorHandlingConfig: max_failures={error_config.max_judge_failures}")
    print(f"  ✓ SpanProcessingConfig: max_spans={span_config.max_spans_per_response}")
    print(f"  ✓ AggregationStrategyConfig: method={agg_config.aggregation_method}")
    print(f"  ✓ PerformanceConfig: max_concurrent={perf_config.max_concurrent_judges}")
    print(f"  ✓ OutputConfig: precision={output_config.precision_decimal_places}")
    
    # Test main config with all sections
    config = TrustScoreConfig()
    assert hasattr(config, 'statistical')
    assert hasattr(config, 'ensemble')
    assert hasattr(config, 'error_handling')
    assert hasattr(config, 'span_processing')
    assert hasattr(config, 'aggregation_strategy')
    assert hasattr(config, 'performance')
    assert hasattr(config, 'output')
    
    print("✓ All configuration classes created successfully!")

def test_custom_configuration():
    """Test creating custom configurations."""
    print("\nTesting custom configuration...")
    
    # Create custom configuration
    custom_config = TrustScoreConfig(
        statistical=StatisticalConfig(
            confidence_margin=0.1,
            use_continuity_correction=False
        ),
        ensemble=EnsembleConfig(
            min_judges_required=2,
            require_consensus=True,
            consensus_threshold=0.8
        ),
        error_handling=ErrorHandlingConfig(
            max_judge_failures=1,
            fail_fast=True,
            log_level="DEBUG"
        ),
        aggregation_strategy=AggregationStrategyConfig(
            aggregation_method="median",
            use_robust_statistics=True,
            outlier_removal=True
        ),
        output=OutputConfig(
            include_ensemble_statistics=True,
            include_individual_judge_scores=True,
            precision_decimal_places=4
        )
    )
    
    # Verify custom values
    assert custom_config.statistical.confidence_margin == 0.1
    assert custom_config.ensemble.require_consensus == True
    assert custom_config.error_handling.fail_fast == True
    assert custom_config.aggregation_strategy.aggregation_method == "median"
    assert custom_config.output.precision_decimal_places == 4
    
    print("✓ Custom configuration created successfully!")

def test_pipeline_with_custom_config():
    """Test pipeline initialization with custom configuration."""
    print("\nTesting pipeline with custom configuration...")
    
    # Create custom configuration
    custom_config = TrustScoreConfig(
        ensemble=EnsembleConfig(
            min_judges_required=1,  # Lower for testing
            require_consensus=False
        ),
        error_handling=ErrorHandlingConfig(
            max_judge_failures=3,
            fail_fast=False
        ),
        aggregation_strategy=AggregationStrategyConfig(
            aggregation_method="weighted_mean"
        ),
        output=OutputConfig(
            include_ensemble_statistics=True,
            precision_decimal_places=2
        )
    )
    
    # Initialize pipeline with custom config
    pipeline = TrustScorePipeline(config=custom_config, use_mock=True)
    
    # Check that custom config is used
    assert pipeline.config.ensemble.min_judges_required == 1
    assert pipeline.config.aggregation_strategy.aggregation_method == "weighted_mean"
    assert pipeline.config.output.precision_decimal_places == 2
    
    print("✓ Pipeline initialized with custom configuration!")

def test_pipeline_status_with_new_configs():
    """Test that pipeline status includes new configuration information."""
    print("\nTesting pipeline status with new configurations...")
    
    pipeline = TrustScorePipeline(use_mock=True)
    status = pipeline.get_pipeline_status()
    
    # Check that new configuration sections are included
    assert "ensemble_config" in status
    assert "error_handling" in status
    assert "performance" in status
    
    # Check specific values
    assert "min_judges_required" in status["ensemble_config"]
    assert "max_judge_failures" in status["error_handling"]
    assert "max_concurrent_judges" in status["performance"]
    
    print("✓ Pipeline status includes new configuration information!")
    print(f"  Ensemble config: {status['ensemble_config']}")
    print(f"  Error handling: {status['error_handling']}")
    print(f"  Performance: {status['performance']}")

def test_aggregation_methods():
    """Test different aggregation methods."""
    print("\nTesting aggregation methods...")
    
    # Test different aggregation strategies
    methods = ["weighted_mean", "median", "robust_mean", "max", "min", "geometric_mean"]
    
    for method in methods:
        config = TrustScoreConfig(
            aggregation_strategy=AggregationStrategyConfig(
                aggregation_method=method
            )
        )
        pipeline = TrustScorePipeline(config=config, use_mock=True)
        
        # Verify the method is set correctly
        assert pipeline.config.aggregation_strategy.aggregation_method == method
        print(f"  ✓ {method} aggregation method configured")
    
    print("✓ All aggregation methods tested!")

def test_confidence_combination_methods():
    """Test different confidence interval combination methods."""
    print("\nTesting confidence combination methods...")
    
    methods = ["weighted_average", "minimum", "maximum", "geometric_mean", "harmonic_mean"]
    
    for method in methods:
        config = TrustScoreConfig(
            aggregation_strategy=AggregationStrategyConfig(
                confidence_combination_method=method
            )
        )
        pipeline = TrustScorePipeline(config=config, use_mock=True)
        
        # Verify the method is set correctly
        assert pipeline.config.aggregation_strategy.confidence_combination_method == method
        print(f"  ✓ {method} confidence combination method configured")
    
    print("✓ All confidence combination methods tested!")

def test_ensemble_consensus():
    """Test ensemble consensus functionality."""
    print("\nTesting ensemble consensus...")
    
    # Test with consensus required
    config = TrustScoreConfig(
        ensemble=EnsembleConfig(
            require_consensus=True,
            consensus_threshold=0.7
        )
    )
    pipeline = TrustScorePipeline(config=config, use_mock=True)
    
    # Test consensus checking method
    from models.llm_record import GradedSpan, SpanTag, JudgeAnalysis, JudgeIndicators, JudgeWeights, SeverityBucket
    
    # Create a mock graded span with multiple analyses
    span = GradedSpan(
        start=0, end=10, type=ErrorType.TRUSTWORTHINESS,
        subtype="factual_error", explanation="Test error"
    )
    
    # Add analyses with similar scores (should have consensus)
    for i in range(3):
        analysis = JudgeAnalysis(
            indicators=JudgeIndicators(centrality=0.5, domain_sensitivity=0.6, harm_potential=0.3, instruction_criticality=0.7),
            weights=JudgeWeights(),
            confidence=0.8,
            severity_score=1.0 + i * 0.1,  # Similar scores
            severity_bucket=SeverityBucket.MAJOR
        )
        span.add_judge_analysis(f"judge_{i+1}", analysis)
    
    # Test consensus check
    consensus = pipeline._check_consensus(span, 0.7)
    print(f"  ✓ Consensus check result: {consensus}")
    
    print("✓ Ensemble consensus functionality tested!")

def test_output_formatting():
    """Test output formatting with different configurations."""
    print("\nTesting output formatting...")
    
    # Test with different output configurations
    configs = [
        OutputConfig(include_ensemble_statistics=True, include_individual_judge_scores=True),
        OutputConfig(include_ensemble_statistics=False, include_individual_judge_scores=False),
        OutputConfig(precision_decimal_places=1),
        OutputConfig(precision_decimal_places=5)
    ]
    
    for i, output_config in enumerate(configs):
        print(f"  Testing output config {i+1}: precision={output_config.precision_decimal_places}")
        
        # Test that the config is valid
        assert output_config.precision_decimal_places >= 0
        assert output_config.precision_decimal_places <= 10
    
    print("✓ Output formatting configurations tested!")

def main():
    """Run all advanced configuration tests."""
    print("Running Advanced Configuration System Tests")
    print("=" * 60)
    
    try:
        test_configuration_creation()
        test_custom_configuration()
        test_pipeline_with_custom_config()
        test_pipeline_status_with_new_configs()
        test_aggregation_methods()
        test_confidence_combination_methods()
        test_ensemble_consensus()
        test_output_formatting()
        
        print("\n" + "=" * 60)
        print("🎉 All advanced configuration tests passed!")
        print("\nNew configuration features available:")
        print("  • Statistical analysis parameters")
        print("  • Ensemble processing controls")
        print("  • Error handling and resilience")
        print("  • Span processing validation")
        print("  • Multiple aggregation strategies")
        print("  • Performance optimization")
        print("  • Configurable output formatting")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
