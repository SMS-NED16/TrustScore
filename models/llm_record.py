"""
TrustScore Pipeline - Data Models

This module contains the core data structures for the TrustScore pipeline,
including LLMRecord, SpanTags, and AggregatedOutput schemas.
"""

from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import statistics


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
    task_prompt: str = Field(..., description="Input prompt")
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
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """Get ensemble statistics for this span."""
        if not self.analysis:
            return {}
        
        severity_scores = [analysis.severity_score for analysis in self.analysis.values()]
        confidences = [analysis.confidence for analysis in self.analysis.values()]
        
        return {
            "mean_severity": statistics.mean(severity_scores),
            "std_severity": statistics.stdev(severity_scores) if len(severity_scores) > 1 else 0,
            "mean_confidence": statistics.mean(confidences),
            "std_confidence": statistics.stdev(confidences) if len(confidences) > 1 else 0,
            "judge_count": len(self.analysis),
            "severity_range": (min(severity_scores), max(severity_scores)),
            "confidence_range": (min(confidences), max(confidences))
        }
    
    def get_robust_average_severity_score(self) -> float:
        """Calculate robust average using median instead of mean."""
        if not self.analysis:
            return 0.0
        severity_scores = [analysis.severity_score for analysis in self.analysis.values()]
        return statistics.median(severity_scores)
    
    def get_robust_average_confidence(self) -> float:
        """Calculate robust average confidence using median."""
        if not self.analysis:
            return 0.0
        confidences = [analysis.confidence for analysis in self.analysis.values()]
        return statistics.median(confidences)
    
    def format_for_output(self, output_config) -> Dict[str, Any]:
        """Format span data for output based on configuration."""
        base_data = {
            "start": self.start,
            "end": self.end,
            "type": self.type.value,
            "subtype": self.subtype,
            "explanation": self.explanation
        }
        
        if output_config.include_ensemble_statistics:
            stats = self.get_ensemble_statistics()
            base_data["ensemble_statistics"] = stats
        
        if output_config.include_individual_judge_scores:
            judge_scores = {}
            for judge_name, analysis in self.analysis.items():
                judge_scores[judge_name] = {
                    "severity_score": analysis.severity_score,
                    "confidence": analysis.confidence,
                    "severity_bucket": analysis.severity_bucket.value
                }
            base_data["judge_scores"] = judge_scores
        
        if output_config.include_judge_metadata:
            judge_metadata = {}
            for judge_name, analysis in self.analysis.items():
                judge_metadata[judge_name] = {
                    "indicators": {
                        "centrality": analysis.indicators.centrality,
                        "domain_sensitivity": analysis.indicators.domain_sensitivity,
                        "harm_potential": analysis.indicators.harm_potential,
                        "instruction_criticality": analysis.indicators.instruction_criticality
                    },
                    "weights": {
                        "centrality": analysis.weights.centrality,
                        "domain_sensitivity": analysis.weights.domain_sensitivity,
                        "harm_potential": analysis.weights.harm_potential,
                        "instruction_criticality": analysis.weights.instruction_criticality
                    }
                }
            base_data["judge_metadata"] = judge_metadata
        
        # Apply precision formatting
        if hasattr(output_config, 'precision_decimal_places'):
            precision = output_config.precision_decimal_places
            for key, value in base_data.items():
                if isinstance(value, float):
                    base_data[key] = round(value, precision)
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, float):
                            base_data[key][sub_key] = round(sub_value, precision)
        
        return base_data


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
    severity_score: float = Field(..., description="Mean severity score across judges (in severity space)")
    severity_score_ci: ConfidenceInterval = Field(..., description="Confidence interval for severity score (in severity space)")
    confidence_level: float = Field(..., ge=0, le=1, description="Mean confidence across judges (in probability space [0-1])")
    confidence_ci: ConfidenceInterval = Field(..., description="Confidence interval for judge confidence (in probability space [0-1])")
    explanation: str = Field(..., description="Human-readable explanation of the error")
    weights: Optional[JudgeWeights] = Field(default=None, description="Weights used by judges for this error type")


class AggregatedSummary(BaseModel):
    """Aggregated scores and confidence intervals"""
    # Severity scores and CIs (in severity space)
    agg_score_T: float = Field(..., description="Aggregated Trustworthiness score (sum of weighted severity scores)")
    agg_score_T_ci: ConfidenceInterval = Field(..., description="Trustworthiness severity score CI (in severity space)")
    agg_score_E: float = Field(..., description="Aggregated Explainability score (sum of weighted severity scores)")
    agg_score_E_ci: ConfidenceInterval = Field(..., description="Explainability severity score CI (in severity space)")
    agg_score_B: float = Field(..., description="Aggregated Bias score (sum of weighted severity scores)")
    agg_score_B_ci: ConfidenceInterval = Field(..., description="Bias severity score CI (in severity space)")
    # Confidence levels and CIs (in probability space [0-1])
    agg_confidence_T: float = Field(..., ge=0, le=1, description="Mean confidence for Trustworthiness category (in probability space [0-1])")
    agg_confidence_T_ci: ConfidenceInterval = Field(..., description="Trustworthiness confidence CI (in probability space [0-1])")
    agg_confidence_E: float = Field(..., ge=0, le=1, description="Mean confidence for Explainability category (in probability space [0-1])")
    agg_confidence_E_ci: ConfidenceInterval = Field(..., description="Explainability confidence CI (in probability space [0-1])")
    agg_confidence_B: float = Field(..., ge=0, le=1, description="Mean confidence for Bias category (in probability space [0-1])")
    agg_confidence_B_ci: ConfidenceInterval = Field(..., description="Bias confidence CI (in probability space [0-1])")
    # Final trust score
    trust_score: float = Field(..., description="Final TrustScore (weighted average of T/E/B severity scores, in severity space)")
    trust_score_ci: ConfidenceInterval = Field(..., description="TrustScore severity score CI (in severity space, with proper error propagation)")
    trust_confidence: float = Field(..., ge=0, le=1, description="Mean confidence for TrustScore (weighted average of T/E/B confidences, in probability space [0-1])")
    trust_confidence_ci: ConfidenceInterval = Field(..., description="TrustScore confidence CI (in probability space [0-1], with proper error propagation)")


class AggregatedOutput(BaseModel):
    """Final aggregated output from TrustScore pipeline"""
    task_prompt: str = Field(..., description="Original input prompt")
    llm_response: str = Field(..., description="Original LLM response")
    model_metadata: ModelMetadata = Field(..., description="Model metadata")
    summary: AggregatedSummary = Field(..., description="Aggregated scores")
    errors: Dict[str, ErrorSummary] = Field(default_factory=dict, description="Error summaries")
    graded_spans: Optional[GradedSpans] = Field(default=None, description="Underlying graded spans for detailed output")
    
    def add_error_summary(self, error_id: str, error_summary: ErrorSummary) -> None:
        """Add an error summary"""
        self.errors[error_id] = error_summary
    
    def format_for_output(self, output_config, judge_info_map: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Format aggregated output based on configuration.
        
        Args:
            output_config: Output configuration
            judge_info_map: Optional map of judge_name -> {"model": str, ...} for judge metadata
        """
        base_data = {
            "task_prompt": self.task_prompt,
            "llm_response": self.llm_response,
            "model_metadata": {
                "model": self.model_metadata.model,
                "generated_on": self.model_metadata.generated_on.isoformat() if self.model_metadata.generated_on else None
            }
        }
        
        if output_config.include_confidence_intervals:
            base_data["summary"] = {
                # Severity scores and CIs (severity space)
                "agg_score_T": self.summary.agg_score_T,
                "agg_score_T_ci": {
                    "lower": self.summary.agg_score_T_ci.lower,
                    "upper": self.summary.agg_score_T_ci.upper
                },
                "agg_score_E": self.summary.agg_score_E,
                "agg_score_E_ci": {
                    "lower": self.summary.agg_score_E_ci.lower,
                    "upper": self.summary.agg_score_E_ci.upper
                },
                "agg_score_B": self.summary.agg_score_B,
                "agg_score_B_ci": {
                    "lower": self.summary.agg_score_B_ci.lower,
                    "upper": self.summary.agg_score_B_ci.upper
                },
                # Confidence levels and CIs (probability space [0-1])
                "agg_confidence_T": self.summary.agg_confidence_T,
                "agg_confidence_T_ci": {
                    "lower": self.summary.agg_confidence_T_ci.lower,
                    "upper": self.summary.agg_confidence_T_ci.upper
                },
                "agg_confidence_E": self.summary.agg_confidence_E,
                "agg_confidence_E_ci": {
                    "lower": self.summary.agg_confidence_E_ci.lower,
                    "upper": self.summary.agg_confidence_E_ci.upper
                },
                "agg_confidence_B": self.summary.agg_confidence_B,
                "agg_confidence_B_ci": {
                    "lower": self.summary.agg_confidence_B_ci.lower,
                    "upper": self.summary.agg_confidence_B_ci.upper
                },
                # Final trust score (severity space)
                "trust_score": self.summary.trust_score,
                "trust_score_ci": {
                    "lower": self.summary.trust_score_ci.lower,
                    "upper": self.summary.trust_score_ci.upper
                },
                # Final trust confidence (probability space [0-1])
                "trust_confidence": self.summary.trust_confidence,
                "trust_confidence_ci": {
                    "lower": self.summary.trust_confidence_ci.lower,
                    "upper": self.summary.trust_confidence_ci.upper
                }
            }
        else:
            base_data["summary"] = {
                # Severity scores (severity space)
                "agg_score_T": self.summary.agg_score_T,
                "agg_score_E": self.summary.agg_score_E,
                "agg_score_B": self.summary.agg_score_B,
                # Confidence levels (probability space [0-1])
                "agg_confidence_T": self.summary.agg_confidence_T,
                "agg_confidence_E": self.summary.agg_confidence_E,
                "agg_confidence_B": self.summary.agg_confidence_B,
                # Final trust score (severity space)
                "trust_score": self.summary.trust_score,
                # Final trust confidence (probability space [0-1])
                "trust_confidence": self.summary.trust_confidence
            }
        
        if output_config.include_raw_spans:
            base_data["raw_spans"] = {}
            for span_id, span in self.errors.items():
                raw_span_data = {
                    "type": span.type.value,
                    "subtype": span.subtype,
                    "severity_bucket": span.severity_bucket.value,
                    "severity_score": span.severity_score,
                    "explanation": span.explanation
                }
                if output_config.include_confidence_intervals:
                    # Include both CIs with clear naming to distinguish spaces
                    raw_span_data["severity_score_ci"] = {
                        "lower": span.severity_score_ci.lower,
                        "upper": span.severity_score_ci.upper
                    }
                    raw_span_data["confidence_level"] = span.confidence_level
                    raw_span_data["confidence_ci"] = {
                        "lower": span.confidence_ci.lower,
                        "upper": span.confidence_ci.upper
                    }
                base_data["raw_spans"][span_id] = raw_span_data
        else:
            base_data["errors"] = {}
            for error_id, error_summary in self.errors.items():
                error_data = {
                    "type": error_summary.type.value,
                    "subtype": error_summary.subtype,
                    "severity_bucket": error_summary.severity_bucket.value,
                    "severity_score": error_summary.severity_score,
                    "explanation": error_summary.explanation
                }
                
                if output_config.include_confidence_intervals:
                    # Include both CIs with clear naming to distinguish spaces
                    # severity_score_ci: CI for severity score (in severity space)
                    # confidence_level: Mean confidence (in probability space [0-1])
                    # confidence_ci: CI for judge confidence (in probability space [0-1])
                    error_data["severity_score_ci"] = {
                        "lower": error_summary.severity_score_ci.lower,
                        "upper": error_summary.severity_score_ci.upper
                    }
                    error_data["confidence_level"] = error_summary.confidence_level
                    error_data["confidence_ci"] = {
                        "lower": error_summary.confidence_ci.lower,
                        "upper": error_summary.confidence_ci.upper
                    }
                
                # Add ensemble statistics if requested
                if output_config.include_ensemble_statistics and self.graded_spans:
                    graded_span = self.graded_spans.spans.get(error_id)
                    if graded_span:
                        stats = graded_span.get_ensemble_statistics()
                        error_data["ensemble_statistics"] = stats
                
                # Add individual judge scores and weights if requested
                if output_config.include_individual_judge_scores and self.graded_spans:
                    graded_span = self.graded_spans.spans.get(error_id)
                    if graded_span and graded_span.analysis:
                        judge_scores = {}
                        for judge_name, analysis in graded_span.analysis.items():
                            judge_entry = {
                                "severity_score": analysis.severity_score,
                                "confidence": analysis.confidence,
                                "severity_bucket": analysis.severity_bucket.value,
                                "indicators": {
                                    "centrality": analysis.indicators.centrality,
                                    "domain_sensitivity": analysis.indicators.domain_sensitivity,
                                    "harm_potential": analysis.indicators.harm_potential,
                                    "instruction_criticality": analysis.indicators.instruction_criticality
                                }
                            }
                            # Add model name if available in judge_info_map
                            if judge_info_map and judge_name in judge_info_map:
                                judge_entry["model"] = judge_info_map[judge_name].get("model", "unknown")
                            judge_scores[judge_name] = judge_entry
                        error_data["judge_scores"] = judge_scores
                        
                        # Add weights at error level (not per judge, since all judges of same type share weights)
                        if error_summary.weights:
                            error_data["weights"] = {
                                "centrality": error_summary.weights.centrality,
                                "domain_sensitivity": error_summary.weights.domain_sensitivity,
                                "harm_potential": error_summary.weights.harm_potential,
                                "instruction_criticality": error_summary.weights.instruction_criticality
                            }
                
                base_data["errors"][error_id] = error_data
        
        # Apply precision formatting
        if hasattr(output_config, 'precision_decimal_places'):
            precision = output_config.precision_decimal_places
            self._apply_precision_formatting(base_data, precision)
        
        return base_data
    
    def _apply_precision_formatting(self, data: Dict[str, Any], precision: int) -> None:
        """Apply precision formatting to float values recursively."""
        for key, value in data.items():
            if isinstance(value, float):
                data[key] = round(value, precision)
            elif isinstance(value, dict):
                self._apply_precision_formatting(value, precision)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, float):
                        value[i] = round(item, precision)
                    elif isinstance(item, dict):
                        self._apply_precision_formatting(item, precision)
