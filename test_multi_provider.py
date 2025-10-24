"""
Test script for multi-provider LLM support in TrustScore pipeline.
"""

import pytest
from datetime import datetime
from models.llm_record import LLMRecord, SpansLevelTags, SpanTag, ErrorType
from modules.span_tagger import SpanTagger, MockSpanTagger
from modules.judges.trustworthiness_judge import TrustworthinessJudge
from modules.judges.bias_judge import BiasJudge
from modules.judges.explainability_judge import ExplainabilityJudge
from modules.aggregator import Aggregator
from pipeline.orchestrator import TrustScorePipeline
from config.settings import (
    TrustScoreConfig, JudgeConfig, SpanTaggerConfig, AggregationWeights,
    LLMProvider, StatisticalConfig, EnsembleConfig, ErrorHandlingConfig,
    SpanProcessingConfig, AggregationStrategyConfig, PerformanceConfig, OutputConfig
)

# Mock Judge for testing purposes
class MockJudge(TrustworthinessJudge):
    def __init__(self, config: JudgeConfig, trust_score_config: TrustScoreConfig, api_key: str = None) -> None:
        super().__init__(config, trust_score_config, api_key)
        self.mock_analysis_data = {
            "indicators": {
                "centrality": 0.7,
                "domain_sensitivity": 0.5,
                "harm_potential": 0.4,
                "instruction_criticality": 0.6
            },
            "weights": {
                "centrality": 1.0,
                "domain_sensitivity": 1.0,
                "harm_potential": 1.0,
                "instruction_criticality": 1.0
            },
            "confidence": 0.8,
            "severity_score": 1.0,
            "severity_bucket": "major"
        }

    def analyze_span(self, llm_record: LLMRecord, span: SpanTag):
        # Simulate different scores for different judges
        if "judge_1" in self.config.name:
            self.mock_analysis_data["severity_score"] = 1.1
            self.mock_analysis_data["confidence"] = 0.85
        elif "judge_2" in self.config.name:
            self.mock_analysis_data["severity_score"] = 0.9
            self.mock_analysis_data["confidence"] = 0.75
        elif "judge_3" in self.config.name:
            self.mock_analysis_data["severity_score"] = 1.2
            self.mock_analysis_data["confidence"] = 0.90
        
        from models.llm_record import JudgeAnalysis, JudgeIndicators, JudgeWeights, SeverityBucket
        return JudgeAnalysis(
            indicators=JudgeIndicators(**self.mock_analysis_data["indicators"]),
            weights=JudgeWeights(**self.mock_analysis_data["weights"]),
            confidence=self.mock_analysis_data["confidence"],
            severity_score=self.mock_analysis_data["severity_score"],
            severity_bucket=self.mock_analysis_data["severity_bucket"]
        )

# Override the actual judge classes with MockJudge for testing
class MockTrustworthinessJudge(MockJudge):
    pass

class MockBiasJudge(MockJudge):
    pass

class MockExplainabilityJudge(MockJudge):
    pass

def create_openai_config() -> TrustScoreConfig:
    """Create configuration for OpenAI provider"""
    return TrustScoreConfig(
        span_tagger=SpanTaggerConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4o",
            api_key="test-key"
        ),
        judges={
            "trust_judge_1": JudgeConfig(
                name="trust_judge_1",
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                api_key="test-key"
            ),
            "bias_judge_1": JudgeConfig(
                name="bias_judge_1",
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                api_key="test-key"
            ),
            "explain_judge_1": JudgeConfig(
                name="explain_judge_1",
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                api_key="test-key"
            ),
        },
        aggregation_weights=AggregationWeights(),
        statistical=StatisticalConfig(),
        ensemble=EnsembleConfig(),
        error_handling=ErrorHandlingConfig(),
        span_processing=SpanProcessingConfig(),
        aggregation_strategy=AggregationStrategyConfig(),
        performance=PerformanceConfig(),
        output=OutputConfig(),
        confidence_level=0.95,
        severity_thresholds={"minor": 0.5, "major": 1.5, "critical": 2.5},
        error_subtypes={
            "T": {"spelling": {"weight": 0.1}, "factual_error": {"weight": 0.8}},
            "B": {"demographic_bias": {"weight": 0.9}, "cultural_bias": {"weight": 0.7}},
            "E": {"unclear_explanation": {"weight": 0.4}, "missing_context": {"weight": 0.6}}
        }
    )

def create_llama_config() -> TrustScoreConfig:
    """Create configuration for LLaMA provider"""
    return TrustScoreConfig(
        span_tagger=SpanTaggerConfig(
            provider=LLMProvider.LLAMA,
            model="llama-2-7b-chat",
            model_path="/path/to/llama-2-7b-chat",
            api_key=None
        ),
        judges={
            "trust_judge_1": JudgeConfig(
                name="trust_judge_1",
                provider=LLMProvider.LLAMA,
                model="llama-2-7b-chat",
                model_path="/path/to/llama-2-7b-chat"
            ),
        },
        aggregation_weights=AggregationWeights(),
        statistical=StatisticalConfig(),
        ensemble=EnsembleConfig(),
        error_handling=ErrorHandlingConfig(),
        span_processing=SpanProcessingConfig(),
        aggregation_strategy=AggregationStrategyConfig(),
        performance=PerformanceConfig(),
        output=OutputConfig(),
        confidence_level=0.95,
        severity_thresholds={"minor": 0.5, "major": 1.5, "critical": 2.5},
        error_subtypes={
            "T": {"spelling": {"weight": 0.1}, "factual_error": {"weight": 0.8}},
            "B": {"demographic_bias": {"weight": 0.9}, "cultural_bias": {"weight": 0.7}},
            "E": {"unclear_explanation": {"weight": 0.4}, "missing_context": {"weight": 0.6}}
        }
    )

def test_openai_provider_config():
    """Test OpenAI provider configuration"""
    config = create_openai_config()
    
    # Test span tagger config
    assert config.span_tagger.provider == LLMProvider.OPENAI
    assert config.span_tagger.model == "gpt-4o"
    assert config.span_tagger.api_key == "test-key"
    
    # Test judge configs
    assert config.judges["trust_judge_1"].provider == LLMProvider.OPENAI
    assert config.judges["bias_judge_1"].provider == LLMProvider.OPENAI
    assert config.judges["explain_judge_1"].provider == LLMProvider.OPENAI

def test_llama_provider_config():
    """Test LLaMA provider configuration"""
    config = create_llama_config()
    
    # Test span tagger config
    assert config.span_tagger.provider == LLMProvider.LLAMA
    assert config.span_tagger.model == "llama-2-7b-chat"
    assert config.span_tagger.model_path == "/path/to/llama-2-7b-chat"
    
    # Test judge config
    assert config.judges["trust_judge_1"].provider == LLMProvider.LLAMA
    assert config.judges["trust_judge_1"].model_path == "/path/to/llama-2-7b-chat"

def test_hybrid_config():
    """Test hybrid configuration (LLaMA + OpenAI)"""
    config = TrustScoreConfig(
        span_tagger=SpanTaggerConfig(
            provider=LLMProvider.LLAMA,
            model="llama-2-7b-chat",
            model_path="/path/to/llama-2-7b-chat"
        ),
        judges={
            "trust_judge_1": JudgeConfig(
                name="trust_judge_1",
                provider=LLMProvider.OPENAI,
                model="gpt-4o",
                api_key="test-key"
            ),
        },
        aggregation_weights=AggregationWeights(),
        statistical=StatisticalConfig(),
        ensemble=EnsembleConfig(),
        error_handling=ErrorHandlingConfig(),
        span_processing=SpanProcessingConfig(),
        aggregation_strategy=AggregationStrategyConfig(),
        performance=PerformanceConfig(),
        output=OutputConfig(),
        confidence_level=0.95,
        severity_thresholds={"minor": 0.5, "major": 1.5, "critical": 2.5},
        error_subtypes={
            "T": {"spelling": {"weight": 0.1}, "factual_error": {"weight": 0.8}},
            "B": {"demographic_bias": {"weight": 0.9}, "cultural_bias": {"weight": 0.7}},
            "E": {"unclear_explanation": {"weight": 0.4}, "missing_context": {"weight": 0.6}}
        }
    )
    
    # Test hybrid configuration
    assert config.span_tagger.provider == LLMProvider.LLAMA
    assert config.judges["trust_judge_1"].provider == LLMProvider.OPENAI

def test_provider_factory():
    """Test LLM provider factory"""
    from modules.llm_providers.factory import LLMProviderFactory
    
    # Test OpenAI provider creation
    openai_config = SpanTaggerConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4o",
        api_key="test-key"
    )
    
    openai_provider = LLMProviderFactory.create_provider(openai_config)
    assert openai_provider.__class__.__name__ == "OpenAIProvider"
    
    # Test LLaMA provider creation
    llama_config = SpanTaggerConfig(
        provider=LLMProvider.LLAMA,
        model="llama-2-7b-chat",
        model_path="/path/to/llama-2-7b-chat"
    )
    
    llama_provider = LLMProviderFactory.create_provider(llama_config)
    assert llama_provider.__class__.__name__ == "LLaMAProvider"

def test_span_tagger_with_provider():
    """Test span tagger with different providers"""
    # Test with mock span tagger (should work regardless of provider)
    openai_config = create_openai_config()
    pipeline = TrustScorePipeline(config=openai_config, use_mock=True)
    
    # Ensure mock span tagger returns a span
    pipeline.span_tagger.tag_spans = lambda x: SpansLevelTags(spans={
        "0": SpanTag(start=0, end=5, type=ErrorType.TRUSTWORTHINESS, subtype="spelling", explanation="test")
    })
    
    result = pipeline.process("prompt", "response", "model")
    assert len(result.errors) == 1

if __name__ == "__main__":
    # Run tests
    test_openai_provider_config()
    test_llama_provider_config()
    test_hybrid_config()
    test_provider_factory()
    test_span_tagger_with_provider()
    
    print("✅ All multi-provider tests passed!")
