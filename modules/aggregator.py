"""
TrustScore Pipeline - Aggregator Module

This module implements the aggregation logic for combining
Trustworthiness, Bias, and Explainability scores into a final TrustScore.
"""

import statistics
from typing import Dict, List, Tuple, Optional
from models.llm_record import (
    LLMRecord, GradedSpans, GradedSpan, AggregatedOutput, 
    AggregatedSummary, ErrorSummary, ConfidenceInterval, ErrorType
)
from config.settings import TrustScoreConfig


class Aggregator:
    """
    Aggregates T/E/B scores and confidence intervals into final TrustScore.
    
    This module combines individual error scores across all three categories
    and produces both category-level and overall trust scores with confidence intervals.
    """
    
    def __init__(self, config: TrustScoreConfig) -> None:
        self.config: TrustScoreConfig = config
    
    def aggregate(self, llm_record: LLMRecord, graded_spans: GradedSpans) -> AggregatedOutput:
        """
        Aggregate graded spans into final TrustScore output.
        
        Args:
            llm_record: Original LLM input/output pair
            graded_spans: Collection of graded error spans
            
        Returns:
            AggregatedOutput: Final aggregated scores and error summaries
        """
        # Calculate aggregated scores for each category
        t_score: float
        t_ci: ConfidenceInterval
        t_score, t_ci = self._calculate_category_score(graded_spans, ErrorType.TRUSTWORTHINESS)
        
        e_score: float
        e_ci: ConfidenceInterval
        e_score, e_ci = self._calculate_category_score(graded_spans, ErrorType.EXPLAINABILITY)
        
        b_score: float
        b_ci: ConfidenceInterval
        b_score, b_ci = self._calculate_category_score(graded_spans, ErrorType.BIAS)
        
        # Calculate final trust score
        trust_score: float
        trust_ci: ConfidenceInterval
        trust_score, trust_ci = self._calculate_trust_score(t_score, e_score, b_score, t_ci, e_ci, b_ci)
        
        # Create aggregated summary
        summary: AggregatedSummary = AggregatedSummary(
            agg_score_T=t_score,
            agg_score_T_ci=t_ci,
            agg_score_E=e_score,
            agg_score_E_ci=e_ci,
            agg_score_B=b_score,
            agg_score_B_ci=b_ci,
            trust_score=trust_score,
            trust_score_ci=trust_ci
        )
        
        # Create error summaries
        error_summaries: Dict[str, ErrorSummary] = self._create_error_summaries(graded_spans)
        
        # Create aggregated output
        aggregated_output: AggregatedOutput = AggregatedOutput(
            x=llm_record.x,
            y=llm_record.y,
            M=llm_record.M,
            summary=summary,
            errors=error_summaries
        )
        
        return aggregated_output
    
    def _calculate_category_score(self, graded_spans: GradedSpans, error_type: ErrorType) -> Tuple[float, ConfidenceInterval]:
        """
        Calculate aggregated score and confidence interval for a specific error category.
        
        Args:
            graded_spans: Collection of graded spans
            error_type: The error category to aggregate
            
        Returns:
            Tuple of (score, confidence_interval)
        """
        category_spans: Dict[str, GradedSpan] = graded_spans.get_spans_by_type(error_type)
        
        if not category_spans:
            return 0.0, ConfidenceInterval(lower=None, upper=None)
        
        # Collect all severity scores and confidences
        severity_scores: List[float] = []
        confidences: List[float] = []
        
        for span in category_spans.values():
            avg_severity: float = span.get_average_severity_score()
            avg_confidence: float = span.get_average_confidence()
            
            # Apply subtype weight
            subtype_weight: float = self.config.get_error_subtype_weight(error_type.value, span.subtype)
            weighted_severity: float = avg_severity * subtype_weight
            
            severity_scores.append(weighted_severity)
            confidences.append(avg_confidence)
        
        # Calculate aggregated score (sum of weighted scores)
        aggregated_score: float = sum(severity_scores)
        
        # Calculate confidence interval
        confidence_interval: ConfidenceInterval = self._calculate_confidence_interval(confidences, severity_scores)
        
        return aggregated_score, confidence_interval
    
    def _calculate_trust_score(self, t_score: float, e_score: float, b_score: float,
                              t_ci: ConfidenceInterval, e_ci: ConfidenceInterval, 
                              b_ci: ConfidenceInterval) -> Tuple[float, ConfidenceInterval]:
        """
        Calculate final trust score using weighted combination of T/E/B scores.
        
        Args:
            t_score, e_score, b_score: Category scores
            t_ci, e_ci, b_ci: Category confidence intervals
            
        Returns:
            Tuple of (trust_score, confidence_interval)
        """
        weights = self.config.aggregation_weights
        
        # Calculate weighted trust score
        trust_score: float = (
            weights.trustworthiness * t_score +
            weights.explainability * e_score +
            weights.bias * b_score
        )
        
        # Calculate confidence interval for trust score
        trust_ci: ConfidenceInterval = self._combine_confidence_intervals(t_ci, e_ci, b_ci, weights)
        
        return trust_score, trust_ci
    
    def _calculate_confidence_interval(self, confidences: List[float], 
                                     scores: List[float]) -> ConfidenceInterval:
        """
        Calculate confidence interval for a set of scores.
        
        Args:
            confidences: List of confidence values
            scores: List of corresponding scores
            
        Returns:
            ConfidenceInterval: Calculated confidence interval
        """
        if not confidences or not scores:
            return ConfidenceInterval(lower=None, upper=None)
        
        # Use bootstrap-like approach for confidence interval
        # For simplicity, we'll use the mean confidence and score variance
        
        mean_confidence: float = statistics.mean(confidences)
        score_variance: float = statistics.variance(scores) if len(scores) > 1 else 0
        
        # Calculate confidence interval bounds
        confidence_level: float = self.config.confidence_level
        alpha: float = 1 - confidence_level
        
        # Simple approximation using normal distribution
        # In practice, you might want to use more sophisticated methods
        z_score: float = 1.96 if confidence_level == 0.95 else 2.576  # Approximate z-scores
        
        margin_of_error: float = z_score * (score_variance ** 0.5) / (len(scores) ** 0.5)
        
        mean_score: float = statistics.mean(scores)
        lower_bound: float = mean_score - margin_of_error
        upper_bound: float = mean_score + margin_of_error
        
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound)
    
    def _combine_confidence_intervals(self, t_ci: ConfidenceInterval, e_ci: ConfidenceInterval,
                                    b_ci: ConfidenceInterval, weights) -> ConfidenceInterval:
        """
        Combine confidence intervals from different categories.
        
        Args:
            t_ci, e_ci, b_ci: Confidence intervals for each category
            weights: Aggregation weights
            
        Returns:
            ConfidenceInterval: Combined confidence interval
        """
        # Simple approach: use the minimum confidence across categories
        # More sophisticated methods could be implemented
        
        confidences: List[float] = []
        if t_ci.lower is not None and t_ci.upper is not None:
            confidences.append((t_ci.lower + t_ci.upper) / 2)
        if e_ci.lower is not None and e_ci.upper is not None:
            confidences.append((e_ci.lower + e_ci.upper) / 2)
        if b_ci.lower is not None and b_ci.upper is not None:
            confidences.append((b_ci.lower + b_ci.upper) / 2)
        
        if not confidences:
            return ConfidenceInterval(lower=None, upper=None)
        
        # Use weighted average of confidence centers
        weighted_confidence: float = (
            weights.trustworthiness * (confidences[0] if len(confidences) > 0 else 0) +
            weights.explainability * (confidences[1] if len(confidences) > 1 else 0) +
            weights.bias * (confidences[2] if len(confidences) > 2 else 0)
        )
        
        # Create symmetric confidence interval around weighted confidence
        margin: float = 0.05  # 5% margin
        return ConfidenceInterval(
            lower=max(0, weighted_confidence - margin),
            upper=min(1, weighted_confidence + margin)
        )
    
    def _create_error_summaries(self, graded_spans: GradedSpans) -> Dict[str, ErrorSummary]:
        """
        Create error summaries from graded spans.
        
        Args:
            graded_spans: Collection of graded spans
            
        Returns:
            Dict of error summaries
        """
        error_summaries: Dict[str, ErrorSummary] = {}
        
        for span_id, span in graded_spans.spans.items():
            avg_severity: float = span.get_average_severity_score()
            avg_confidence: float = span.get_average_confidence()
            
            # Determine severity bucket
            severity_bucket: str = self.config.get_severity_bucket(avg_severity)
            
            # Create confidence interval for this error
            confidence_ci: ConfidenceInterval = ConfidenceInterval(
                lower=max(0, avg_confidence - 0.05),
                upper=min(1, avg_confidence + 0.05)
            )
            
            error_summary: ErrorSummary = ErrorSummary(
                type=span.type,
                subtype=span.subtype,
                severity_bucket=severity_bucket,
                severity_score=avg_severity,
                confidence=confidence_ci
            )
            
            error_summaries[span_id] = error_summary
        
        return error_summaries
