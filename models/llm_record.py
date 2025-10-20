"""
TrustScore Pipeline - Data Models

This module contains the core data structures for the TrustScore pipeline,
including LLMRecord, SpanTags, and AggregatedOutput schemas.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class ErrorType(str, Enum):
    """Error type categories"""
    TRUSTWORTHINESS = "T"
    BIAS = "B"
    EXPLAINABILITY = "E"


class SeverityBucket(str, Enum):
    """Severity classification buckets"""
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"


class DecoderParams(BaseModel):
    """Decoder parameters for LLM generation"""
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[List[str]] = None


class ModelMetadata(BaseModel):
    """Model metadata for LLM generation"""
    model: str = Field(..., description="Model name/identifier")
    generated_on: datetime = Field(..., description="Generation timestamp")
    decoder_params: Optional[DecoderParams] = None


class LLMRecord(BaseModel):
    """Standardized format for LLM input/output pairs"""
    task_prompt = Field(..., description="Input prompt")
    llm_response: str = Field(..., description="LLM response")
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SpanTag(BaseModel):
    """Individual span-level error tag"""
    start: int = Field(..., ge=0, description="Start character position")
    end: int = Field(..., ge=0, description="End character position")
    type: ErrorType = Field(..., description="Error category")
    subtype: str = Field(..., description="Specific error subtype")
    explanation: str = Field(..., description="Human-readable explanation")
    
    def __post_init__(self):
        if self.end <= self.start:
            raise ValueError("End position must be greater than start position")


class SpansLevelTags(BaseModel):
    """Collection of span-level error tags"""
    spans: Dict[str, SpanTag] = Field(default_factory=dict)
    
    def add_span(self, span_id: str, span: SpanTag) -> None:
        """Add a span tag"""
        self.spans[span_id] = span
    
    def get_spans_by_type(self, error_type: ErrorType) -> Dict[str, SpanTag]:
        """Get all spans of a specific error type"""
        return {k: v for k, v in self.spans.items() if v.type == error_type}


class JudgeIndicators(BaseModel):
    """Indicators used by judges for scoring"""
    centrality: float = Field(..., ge=0, le=1, description="How central the error is to the response")
    domain_sensitivity: float = Field(..., ge=0, le=1, description="Domain-specific sensitivity")
    harm_potential: float = Field(..., ge=0, le=1, description="Potential for harm")
    instruction_criticality: float = Field(..., ge=0, le=1, description="Criticality to instruction")


class JudgeWeights(BaseModel):
    """Weights for different indicators"""
    centrality: float = Field(default=1.0, ge=0)
    domain_sensitivity: float = Field(default=1.0, ge=0)
    harm_potential: float = Field(default=1.0, ge=0)
    instruction_criticality: float = Field(default=1.0, ge=0)


class JudgeAnalysis(BaseModel):
    """Analysis from a single judge"""
    indicators: JudgeIndicators
    weights: JudgeWeights
    confidence: float = Field(..., ge=0, le=1, description="Judge confidence")
    severity_score: float = Field(..., description="Calculated severity score")
    severity_bucket: SeverityBucket = Field(..., description="Severity classification")


class GradedSpan(BaseModel):
    """Span with judge analyses"""
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=0)
    type: ErrorType
    subtype: str
    explanation: str
    analysis: Dict[str, JudgeAnalysis] = Field(default_factory=dict)
    
    def add_judge_analysis(self, judge_name: str, analysis: JudgeAnalysis) -> None:
        """Add analysis from a judge"""
        self.analysis[judge_name] = analysis
    
    def get_average_severity_score(self) -> float:
        """Calculate average severity score across all judges"""
        if not self.analysis:
            return 0.0
        return sum(analysis.severity_score for analysis in self.analysis.values()) / len(self.analysis)
    
    def get_average_confidence(self) -> float:
        """Calculate average confidence across all judges"""
        if not self.analysis:
            return 0.0
        return sum(analysis.confidence for analysis in self.analysis.values()) / len(self.analysis)


class GradedSpans(BaseModel):
    """Collection of graded spans"""
    spans: Dict[str, GradedSpan] = Field(default_factory=dict)
    
    def add_graded_span(self, span_id: str, graded_span: GradedSpan) -> None:
        """Add a graded span"""
        self.spans[span_id] = graded_span
    
    def get_spans_by_type(self, error_type: ErrorType) -> Dict[str, GradedSpan]:
        """Get all spans of a specific error type"""
        return {k: v for k, v in self.spans.items() if v.type == error_type}


class ConfidenceInterval(BaseModel):
    """Confidence interval for a score"""
    lower: Optional[float] = None
    upper: Optional[float] = None
    
    def __post_init__(self):
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("Lower bound must be <= upper bound")


class ErrorSummary(BaseModel):
    """Summary of an individual error"""
    type: ErrorType
    subtype: str
    severity_bucket: SeverityBucket
    severity_score: float
    confidence: ConfidenceInterval


class AggregatedSummary(BaseModel):
    """Aggregated scores and confidence intervals"""
    agg_score_T: float = Field(..., description="Aggregated Trustworthiness score")
    agg_score_T_ci: ConfidenceInterval = Field(..., description="Trustworthiness confidence interval")
    agg_score_E: float = Field(..., description="Aggregated Explainability score")
    agg_score_E_ci: ConfidenceInterval = Field(..., description="Explainability confidence interval")
    agg_score_B: float = Field(..., description="Aggregated Bias score")
    agg_score_B_ci: ConfidenceInterval = Field(..., description="Bias confidence interval")
    trust_score: float = Field(..., description="Final TrustScore")
    trust_score_ci: ConfidenceInterval = Field(..., description="TrustScore confidence interval")


class AggregatedOutput(BaseModel):
    """Final aggregated output from TrustScore pipeline"""
    task_prompt: str = Field(..., description="Original input prompt")
    llm_response: str = Field(..., description="Original LLM response")
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    summary: AggregatedSummary = Field(..., description="Aggregated scores")
    errors: Dict[str, ErrorSummary] = Field(default_factory=dict, description="Error summaries")
    
    def add_error_summary(self, error_id: str, error_summary: ErrorSummary) -> None:
        """Add an error summary"""
        self.errors[error_id] = error_summary
