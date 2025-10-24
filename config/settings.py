"""
TrustScore Pipeline - Configuration Management

This module handles configuration settings, weights, and parameters
for the TrustScore pipeline.
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import yaml
import os


class JudgeConfig(BaseModel):
    """Configuration for individual judges"""
    name: str
    model: str
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=1000, ge=1)
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


class SpanTaggerConfig(BaseModel):
    """Configuration for the span tagger module"""
    model: str = Field(default="gpt-4o")
    temperature: float = Field(default=0.1, ge=0, le=2)
    max_tokens: int = Field(default=2000, ge=1)
    batch_size: int = Field(default=1, ge=1)


class TrustScoreConfig(BaseModel):
    """Main configuration for TrustScore pipeline"""
    span_tagger: SpanTaggerConfig = Field(default_factory=SpanTaggerConfig)
    judges: Dict[str, JudgeConfig] = Field(default_factory=dict)
    aggregation_weights: AggregationWeights = Field(default_factory=AggregationWeights)
    
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
            "political_bias": {"weight": 0.6, "description": "Political leaning"}
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
