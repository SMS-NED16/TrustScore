"""
TrustScore Pipeline - Configuration Management

This module handles configuration settings, weights, and parameters
for the TrustScore pipeline.
"""

from typing import Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
from enum import Enum
import yaml
import os


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    LLAMA = "llama"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"


class LLMConfig(BaseModel):
    """Base configuration for LLM providers"""
    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    model: str = Field(default="gpt-4o")
    fine_tuned_model: Optional[str] = None
    model_path: Optional[str] = None  # For local models like LLaMA
    api_key: Optional[str] = None
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=1, ge=1)
    
    # LLaMA-specific settings
    device: str = Field(default="cuda")  # cuda, cpu, mps
    torch_dtype: str = Field(default="float16")  # float16, float32
    use_cache: bool = Field(default=True)
    
    # Fine-tuning settings
    fine_tuned: bool = Field(default=False)
    fine_tuning_data_path: Optional[str] = None
    fine_tuning_epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=5e-5, ge=1e-6, le=1e-3)


class JudgeConfig(LLMConfig):
    """Configuration for individual judges"""
    name: str
    enabled: bool = Field(default=True)


class AggregationWeights(BaseModel):
    """Weights for aggregating T/E/B scores"""
    trustworthiness: float = Field(default=0.5, ge=0, le=1)
    explainability: float = Field(default=0.3, ge=0, le=1)
    bias: float = Field(default=0.2, ge=0, le=1)
    
    def __post_init__(self) -> None:
        total: float = self.trustworthiness + self.explainability + self.bias
        if abs(total - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Weights must sum to 1.0, got {total}")


class SpanTaggerConfig(LLMConfig):
    """Configuration for the span tagger module"""
    pass


class StatisticalConfig(BaseModel):
    """Configuration for statistical analysis and confidence intervals"""
    t_critical_values: Dict[int, float] = Field(default_factory=lambda: {
        3: 2.776, 4: 2.571, 5: 2.447, 6: 2.365, 7: 2.306, 8: 2.262, 9: 2.228, 10: 2.201
    })
    confidence_margin: float = Field(default=0.05, ge=0, le=0.2)
    min_sample_size_for_t_dist: int = Field(default=30)
    fallback_z_scores: Dict[float, float] = Field(default_factory=lambda: {
        0.90: 1.645, 0.95: 1.96, 0.99: 2.576
    })
    use_continuity_correction: bool = Field(default=True)


class EnsembleConfig(BaseModel):
    """Configuration for ensemble processing"""
    min_judges_required: int = Field(default=1, ge=1)
    max_judges_per_aspect: int = Field(default=10, ge=1)
    require_consensus: bool = Field(default=False)
    consensus_threshold: float = Field(default=0.7, ge=0.5, le=1.0)
    outlier_detection: bool = Field(default=False)
    outlier_threshold: float = Field(default=2.0)  # Standard deviations
    use_robust_statistics: bool = Field(default=False)


class ErrorHandlingConfig(BaseModel):
    """Configuration for error handling and resilience"""
    max_judge_failures: int = Field(default=2, ge=0)
    fail_fast: bool = Field(default=False)
    retry_failed_judges: bool = Field(default=False)
    max_retries: int = Field(default=1, ge=0)
    log_level: str = Field(default="INFO")
    continue_on_span_errors: bool = Field(default=True)


class SpanProcessingConfig(BaseModel):
    """Configuration for span processing and validation"""
    min_span_length: int = Field(default=1, ge=1)
    max_span_length: int = Field(default=1000, ge=1)
    allow_overlapping_spans: bool = Field(default=True)
    max_spans_per_response: int = Field(default=50, ge=1)
    span_validation_strict: bool = Field(default=True)
    merge_adjacent_spans: bool = Field(default=False)
    min_span_gap: int = Field(default=0, ge=0)
    min_explanation_length: int = Field(default=10, ge=1, description="Minimum length for error explanations in characters")


class AggregationStrategyConfig(BaseModel):
    """Configuration for aggregation strategies"""
    aggregation_method: str = Field(default="weighted_mean")
    use_robust_statistics: bool = Field(default=False)
    outlier_removal: bool = Field(default=False)
    confidence_combination_method: str = Field(default="weighted_average")
    normalize_scores: bool = Field(default=False)  # Disabled - using sigmoid transformation instead
    score_range: Tuple[float, float] = Field(default=(0.0, 10.0))
    # Sigmoid transformation parameters for severity -> quality conversion
    use_quality_scores: bool = Field(default=True, description="Transform severity scores to quality scores [0-100]")
    sigmoid_steepness: float = Field(default=0.5, ge=0.1, le=2.0, description="Steepness parameter for sigmoid (higher = steeper)")
    sigmoid_shift: float = Field(default=0.0, description="Shift parameter for sigmoid (adjusts center point)")


class PerformanceConfig(BaseModel):
    """Configuration for performance optimization"""
    max_concurrent_judges: int = Field(default=3, ge=1)
    judge_timeout_seconds: int = Field(default=30, ge=1)
    enable_parallel_processing: bool = Field(default=True)
    cache_judge_responses: bool = Field(default=False)
    max_cache_size: int = Field(default=1000, ge=0)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    batch_processing: bool = Field(default=True)
    max_batch_size: int = Field(default=10, ge=1)


class OutputConfig(BaseModel):
    """Configuration for output formatting and detail level"""
    include_ensemble_statistics: bool = Field(default=True)
    include_individual_judge_scores: bool = Field(default=False)
    include_confidence_intervals: bool = Field(default=True)
    precision_decimal_places: int = Field(default=3, ge=0, le=10)
    include_raw_spans: bool = Field(default=False)
    include_judge_metadata: bool = Field(default=False)
    output_format: str = Field(default="json")
    verbose_logging: bool = Field(default=False)


class TrustScoreConfig(BaseModel):
    """Main configuration for TrustScore pipeline"""
    span_tagger: SpanTaggerConfig = Field(default_factory=SpanTaggerConfig)
    judges: Dict[str, JudgeConfig] = Field(default_factory=dict)
    aggregation_weights: AggregationWeights = Field(default_factory=AggregationWeights)
    
    # New configuration sections
    statistical: StatisticalConfig = Field(default_factory=StatisticalConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    error_handling: ErrorHandlingConfig = Field(default_factory=ErrorHandlingConfig)
    span_processing: SpanProcessingConfig = Field(default_factory=SpanProcessingConfig)
    aggregation_strategy: AggregationStrategyConfig = Field(default_factory=AggregationStrategyConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    # Confidence interval settings
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.99)
    
    # Severity scoring thresholds
    severity_thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "minor": 0.5,
        "major": 1.5,
        "critical": 2.5
    })
    
    # Error type mappings
    error_subtypes: Dict[str, Dict[str, Any]] = Field(default_factory=lambda: {
        "T": {
            "spelling": {"weight": 0.1, "description": "Spelling errors"},
            "factual_error": {"weight": 0.8, "description": "Factual inaccuracies"},
            "hallucination": {"weight": 0.9, "description": "Completely fabricated information"},
            "inconsistency": {"weight": 0.6, "description": "Internal contradictions"}
        },
        "B": {
            "demographic_bias": {"weight": 0.9, "description": "Demographic stereotyping"},
            "cultural_bias": {"weight": 0.7, "description": "Cultural assumptions"},
            "gender_bias": {"weight": 0.8, "description": "Gender stereotyping"},
            "political_bias": {"weight": 0.6, "description": "Political leaning"},
            "sycophancy_bias": {"weight": 0.8, "description": "Excessive agreement or flattery toward user views"},
            "confirmation_bias": {"weight": 0.7, "description": "Selectively presenting confirming information while ignoring contradictory evidence"}
        },
        "E": {
            "unclear_explanation": {"weight": 0.4, "description": "Unclear or confusing explanation"},
            "missing_context": {"weight": 0.6, "description": "Missing important context"},
            "overly_complex": {"weight": 0.3, "description": "Unnecessarily complex explanation"},
            "assumption_not_stated": {"weight": 0.5, "description": "Unstated assumptions"}
        }
    })
    
    @classmethod
    def from_yaml(cls, file_path: str) -> 'TrustScoreConfig':
        """Load configuration from YAML file"""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, file_path: str) -> None:
        """Save configuration to YAML file"""
        with open(file_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def get_error_subtype_weight(self, error_type: str, subtype: str) -> float:
        """Get weight for specific error subtype"""
        return self.error_subtypes.get(error_type, {}).get(subtype, {}).get("weight", 1.0)
    
    def get_severity_bucket(self, score: float) -> str:
        """Determine severity bucket based on score"""
        if score <= self.severity_thresholds["minor"]:
            return "minor"
        elif score <= self.severity_thresholds["major"]:
            return "major"
        else:
            return "critical"


# Default configuration
# Update the default configuration to include multiple judges per aspect
DEFAULT_CONFIG = TrustScoreConfig(
    judges={
        # Trustworthiness ensemble (3 judges)
        "trust_judge_1": JudgeConfig(name="trust_judge_1", model="gpt-4o"),
        "trust_judge_2": JudgeConfig(name="trust_judge_2", model="gpt-4o"),  # Can use different models
        "trust_judge_3": JudgeConfig(name="trust_judge_3", model="gpt-4o"),
        
        # Bias ensemble (3 judges)
        "bias_judge_1": JudgeConfig(name="bias_judge_1", model="gpt-4o"),
        "bias_judge_2": JudgeConfig(name="bias_judge_2", model="gpt-4o"),
        "bias_judge_3": JudgeConfig(name="bias_judge_3", model="gpt-4o"),
        
        # Explainability ensemble (3 judges)
        "explain_judge_1": JudgeConfig(name="explain_judge_1", model="gpt-4o"),
        "explain_judge_2": JudgeConfig(name="explain_judge_2", model="gpt-4o"),
        "explain_judge_3": JudgeConfig(name="explain_judge_3", model="gpt-4o"),
    }
)


def load_config(config_path: Optional[str] = None) -> TrustScoreConfig:
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        return TrustScoreConfig.from_yaml(config_path)
    return DEFAULT_CONFIG
