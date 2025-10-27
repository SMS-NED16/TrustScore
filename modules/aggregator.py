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
            task_prompt=llm_record.task_prompt,
            llm_response=llm_record.llm_response,
            model_metadata=llm_record.model_metadata,
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
        Calculate final trust score using configurable aggregation strategy.
        
        Args:
            t_score, e_score, b_score: Category scores
            t_ci, e_ci, b_ci: Category confidence intervals
            
        Returns:
            Tuple of (trust_score, confidence_interval)
        """
        weights = self.config.aggregation_weights
        agg_strategy = self.config.aggregation_strategy
        
        # Normalize scores if configured
        if agg_strategy.normalize_scores:
            score_min, score_max = agg_strategy.score_range
            t_score = max(score_min, min(score_max, t_score))
            e_score = max(score_min, min(score_max, e_score))
            b_score = max(score_min, min(score_max, b_score))
        
        # Calculate trust score using configurable aggregation method
        if agg_strategy.aggregation_method == "weighted_mean":
            trust_score: float = (
                weights.trustworthiness * t_score +
                weights.explainability * e_score +
                weights.bias * b_score
            )
        elif agg_strategy.aggregation_method == "median":
            trust_score = statistics.median([t_score, e_score, b_score])
        elif agg_strategy.aggregation_method == "robust_mean":
            # Use trimmed mean for robustness
            scores = [t_score, e_score, b_score]
            if agg_strategy.outlier_removal:
                # Remove outliers using IQR method
                q1 = statistics.quantiles(scores, n=4)[0]
                q3 = statistics.quantiles(scores, n=4)[2]
                iqr = q3 - q1
                scores = [s for s in scores if q1 - 1.5*iqr <= s <= q3 + 1.5*iqr]
            trust_score = statistics.mean(scores)
        elif agg_strategy.aggregation_method == "max":
            trust_score = max(t_score, e_score, b_score)
        elif agg_strategy.aggregation_method == "min":
            trust_score = min(t_score, e_score, b_score)
        elif agg_strategy.aggregation_method == "geometric_mean":
            trust_score = statistics.geometric_mean([t_score, e_score, b_score])
        else:
            # Default to weighted mean
            trust_score = (
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
        Calculate confidence interval using ensemble statistics and configurable parameters.
        
        Args:
            confidences: List of confidence values
            scores: List of corresponding scores
            
        Returns:
            ConfidenceInterval: Calculated confidence interval
        """
        if not confidences or not scores:
            return ConfidenceInterval(lower=None, upper=None)
        
        # Use ensemble statistics for better confidence intervals
        mean_confidence: float = statistics.mean(confidences)
        mean_score: float = statistics.mean(scores)
        
        # Calculate standard error of the mean
        if len(scores) > 1:
            score_std: float = statistics.stdev(scores)
            score_sem: float = score_std / (len(scores) ** 0.5)  # Standard Error of Mean
        else:
            score_sem: float = 0
        
        # Use confidence level from config
        confidence_level: float = self.config.confidence_level
        alpha: float = 1 - confidence_level
        
        # Use configurable statistical parameters
        stat_config = self.config.statistical
        
        # Calculate t-statistic for small samples (more accurate than z-score)
        if len(scores) <= stat_config.min_sample_size_for_t_dist:
            # Use t-distribution for small samples with configurable values
            t_critical = stat_config.t_critical_values.get(len(scores), 2.0)
        else:
            # Use z-score for large samples with configurable values
            t_critical = stat_config.fallback_z_scores.get(confidence_level, 1.96)
        
        margin_of_error: float = t_critical * score_sem
        
        # Apply configurable confidence margin
        if stat_config.use_continuity_correction:
            margin_of_error += stat_config.confidence_margin
        
        lower_bound: float = mean_score - margin_of_error
        upper_bound: float = mean_score + margin_of_error
        
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound)
    
    def _combine_confidence_intervals(self, t_ci: ConfidenceInterval, e_ci: ConfidenceInterval,
                                    b_ci: ConfidenceInterval, weights) -> ConfidenceInterval:
        """
        Combine confidence intervals from different categories using configurable strategy.
        
        Args:
            t_ci, e_ci, b_ci: Confidence intervals for each category
            weights: Aggregation weights
            
        Returns:
            ConfidenceInterval: Combined confidence interval
        """
        # Collect confidence centers
        confidences: List[float] = []
        if t_ci.lower is not None and t_ci.upper is not None:
            confidences.append((t_ci.lower + t_ci.upper) / 2)
        if e_ci.lower is not None and e_ci.upper is not None:
            confidences.append((e_ci.lower + e_ci.upper) / 2)
        if b_ci.lower is not None and b_ci.upper is not None:
            confidences.append((b_ci.lower + b_ci.upper) / 2)
        
        if not confidences:
            return ConfidenceInterval(lower=None, upper=None)
        
        # Use configurable combination method
        agg_strategy = self.config.aggregation_strategy
        stat_config = self.config.statistical
        
        if agg_strategy.confidence_combination_method == "weighted_average":
            # Use weighted average of confidence centers
            weighted_confidence: float = (
                weights.trustworthiness * (confidences[0] if len(confidences) > 0 else 0) +
                weights.explainability * (confidences[1] if len(confidences) > 1 else 0) +
                weights.bias * (confidences[2] if len(confidences) > 2 else 0)
            )
        elif agg_strategy.confidence_combination_method == "minimum":
            weighted_confidence = min(confidences)
        elif agg_strategy.confidence_combination_method == "maximum":
            weighted_confidence = max(confidences)
        elif agg_strategy.confidence_combination_method == "geometric_mean":
            weighted_confidence = statistics.geometric_mean(confidences)
        elif agg_strategy.confidence_combination_method == "harmonic_mean":
            weighted_confidence = statistics.harmonic_mean(confidences)
        else:
            # Default to weighted average
            weighted_confidence = statistics.mean(confidences)
        
        # Create symmetric confidence interval around weighted confidence
        margin: float = stat_config.confidence_margin
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
