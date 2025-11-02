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
        # LOG: Input summary
        total_spans = len(graded_spans.spans)
        t_spans = len(graded_spans.get_spans_by_type(ErrorType.TRUSTWORTHINESS))
        b_spans = len(graded_spans.get_spans_by_type(ErrorType.BIAS))
        e_spans = len(graded_spans.get_spans_by_type(ErrorType.EXPLAINABILITY))
        print(f"[DEBUG Aggregation] Input: {total_spans} total spans (T={t_spans}, B={b_spans}, E={e_spans})")
        
        # Calculate aggregated scores for each category
        # Returns: (severity_score, severity_score_ci, mean_confidence, confidence_ci)
        t_score: float
        t_severity_ci: ConfidenceInterval
        t_confidence: float
        t_confidence_ci: ConfidenceInterval
        t_score, t_severity_ci, t_confidence, t_confidence_ci = self._calculate_category_score(graded_spans, ErrorType.TRUSTWORTHINESS)
        
        e_score: float
        e_severity_ci: ConfidenceInterval
        e_confidence: float
        e_confidence_ci: ConfidenceInterval
        e_score, e_severity_ci, e_confidence, e_confidence_ci = self._calculate_category_score(graded_spans, ErrorType.EXPLAINABILITY)
        
        b_score: float
        b_severity_ci: ConfidenceInterval
        b_confidence: float
        b_confidence_ci: ConfidenceInterval
        b_score, b_severity_ci, b_confidence, b_confidence_ci = self._calculate_category_score(graded_spans, ErrorType.BIAS)
        
        # LOG: Scores before final aggregation
        print(f"[DEBUG Aggregation] Category severity scores: T={t_score:.3f}, E={e_score:.3f}, B={b_score:.3f}")
        print(f"[DEBUG Aggregation] Category mean confidences: T={t_confidence:.3f}, E={e_confidence:.3f}, B={b_confidence:.3f}")
        print(f"[DEBUG Aggregation] Aggregation weights: T={self.config.aggregation_weights.trustworthiness:.3f}, "
              f"E={self.config.aggregation_weights.explainability:.3f}, "
              f"B={self.config.aggregation_weights.bias:.3f}")
        
        # Calculate final trust score (with proper error propagation for both severity and confidence)
        trust_score: float
        trust_score_ci: ConfidenceInterval
        trust_confidence: float
        trust_confidence_ci: ConfidenceInterval
        trust_score, trust_score_ci, trust_confidence, trust_confidence_ci = self._calculate_trust_score(
            t_score, e_score, b_score,
            t_severity_ci, e_severity_ci, b_severity_ci,
            t_confidence, e_confidence, b_confidence,
            t_confidence_ci, e_confidence_ci, b_confidence_ci
        )
        
        # LOG: Final trust score
        print(f"[DEBUG Aggregation] Final trust_score: {trust_score:.3f} (severity space)")
        print(f"[DEBUG Aggregation] Final trust_confidence: {trust_confidence:.3f} (probability space [0-1])")
        
        # Create aggregated summary
        summary: AggregatedSummary = AggregatedSummary(
            # Severity scores and CIs (severity space)
            agg_score_T=t_score,
            agg_score_T_ci=t_severity_ci,
            agg_score_E=e_score,
            agg_score_E_ci=e_severity_ci,
            agg_score_B=b_score,
            agg_score_B_ci=b_severity_ci,
            # Confidence levels and CIs (probability space [0-1])
            agg_confidence_T=t_confidence,
            agg_confidence_T_ci=t_confidence_ci,
            agg_confidence_E=e_confidence,
            agg_confidence_E_ci=e_confidence_ci,
            agg_confidence_B=b_confidence,
            agg_confidence_B_ci=b_confidence_ci,
            # Final trust score (severity space)
            trust_score=trust_score,
            trust_score_ci=trust_score_ci,
            # Final trust confidence (probability space [0-1])
            trust_confidence=trust_confidence,
            trust_confidence_ci=trust_confidence_ci
        )
        
        # Create error summaries
        error_summaries: Dict[str, ErrorSummary] = self._create_error_summaries(graded_spans)
        
        # Create aggregated output with graded_spans reference for detailed output
        aggregated_output: AggregatedOutput = AggregatedOutput(
            task_prompt=llm_record.task_prompt,
            llm_response=llm_record.llm_response,
            model_metadata=llm_record.model_metadata,
            summary=summary,
            errors=error_summaries,
            graded_spans=graded_spans
        )
        
        return aggregated_output
    
    def _calculate_category_score(self, graded_spans: GradedSpans, error_type: ErrorType) -> Tuple[float, ConfidenceInterval, float, ConfidenceInterval]:
        """
        Calculate aggregated score and confidence interval for a specific error category.
        
        Args:
            graded_spans: Collection of graded spans
            error_type: The error category to aggregate
            
        Returns:
            Tuple of (severity_score, severity_score_ci, mean_confidence, confidence_ci)
            - severity_score: Sum of weighted severity scores (severity space)
            - severity_score_ci: CI around the sum (severity space)
            - mean_confidence: Mean confidence across spans (probability space [0-1])
            - confidence_ci: CI around the mean confidence (probability space [0-1])
        """
        category_spans: Dict[str, GradedSpan] = graded_spans.get_spans_by_type(error_type)
        
        print(f"[DEBUG Score Calculation] Computing {error_type.value} score: Found {len(category_spans)} span(s)")
        
        if not category_spans:
            print(f"[DEBUG Score Calculation] {error_type.value} score = 0.0 (no spans found)")
            return 0.0, ConfidenceInterval(lower=None, upper=None), 0.0, ConfidenceInterval(lower=None, upper=None)
        
        # Collect all severity scores and confidences
        severity_scores: List[float] = []
        confidences: List[float] = []
        
        for span_id, span in category_spans.items():
            avg_severity: float = span.get_average_severity_score()
            avg_confidence: float = span.get_average_confidence()
            
            # LOG: Individual span scores
            print(f"[DEBUG Score Calculation] {error_type.value} span {span_id}: "
                  f"avg_severity={avg_severity:.3f}, avg_confidence={avg_confidence:.3f}, "
                  f"subtype={span.subtype}, judge_count={len(span.analysis)}")
            
            # Apply subtype weight
            subtype_weight: float = self.config.get_error_subtype_weight(error_type.value, span.subtype)
            weighted_severity: float = avg_severity * subtype_weight
            
            # LOG: Weighting
            print(f"[DEBUG Score Calculation] {error_type.value} span {span_id}: "
                  f"subtype_weight={subtype_weight:.3f}, weighted_severity={weighted_severity:.3f}")
            
            severity_scores.append(weighted_severity)
            confidences.append(avg_confidence)
        
        # Calculate aggregated severity score (sum of weighted scores)
        aggregated_severity_score: float = sum(severity_scores)
        
        # Calculate mean confidence (average across spans)
        mean_confidence: float = statistics.mean(confidences) if confidences else 0.0
        
        # LOG: Final scores
        print(f"[DEBUG Score Calculation] {error_type.value} final severity score: {aggregated_severity_score:.3f} "
              f"(sum of {len(severity_scores)} span(s): {severity_scores})")
        print(f"[DEBUG Score Calculation] {error_type.value} mean confidence: {mean_confidence:.3f} "
              f"(mean of {len(confidences)} span(s): {confidences})")
        
        # Calculate severity score CI (around the sum, in severity space)
        severity_score_ci: ConfidenceInterval = self._calculate_severity_confidence_interval(severity_scores)
        
        # Calculate confidence CI (around the mean, in probability space [0-1])
        confidence_ci: ConfidenceInterval = self._calculate_mean_confidence_interval(confidences)
        
        return aggregated_severity_score, severity_score_ci, mean_confidence, confidence_ci
    
    def _calculate_trust_score(self, t_score: float, e_score: float, b_score: float,
                              t_severity_ci: ConfidenceInterval, e_severity_ci: ConfidenceInterval, 
                              b_severity_ci: ConfidenceInterval,
                              t_confidence: float, e_confidence: float, b_confidence: float,
                              t_confidence_ci: ConfidenceInterval, e_confidence_ci: ConfidenceInterval,
                              b_confidence_ci: ConfidenceInterval) -> Tuple[float, ConfidenceInterval, float, ConfidenceInterval]:
        """
        Calculate final trust score using configurable aggregation strategy.
        
        Args:
            t_score, e_score, b_score: Category severity scores
            t_severity_ci, e_severity_ci, b_severity_ci: Category severity score CIs (in severity space)
            t_confidence, e_confidence, b_confidence: Category confidence levels
            t_confidence_ci, e_confidence_ci, b_confidence_ci: Category confidence CIs (in probability space [0-1])
            
        Returns:
            Tuple of (trust_score, trust_score_ci, trust_confidence, trust_confidence_ci)
            - trust_score: Weighted average of T/E/B severity scores (severity space)
            - trust_score_ci: CI for trust score with proper error propagation (severity space)
            - trust_confidence: Weighted average of T/E/B confidences (probability space [0-1])
            - trust_confidence_ci: CI for trust confidence with proper error propagation (probability space [0-1])
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
        
        # Calculate trust confidence (weighted average of category confidences)
        trust_confidence: float = (
            weights.trustworthiness * t_confidence +
            weights.explainability * e_confidence +
            weights.bias * b_confidence
        )
        
        # Calculate severity score CI with proper error propagation (in severity space)
        trust_score_ci: ConfidenceInterval = self._propagate_severity_ci(
            t_severity_ci, e_severity_ci, b_severity_ci, weights, agg_strategy.aggregation_method
        )
        
        # Calculate confidence CI with proper error propagation (in probability space [0-1])
        trust_confidence_ci: ConfidenceInterval = self._propagate_confidence_ci(
            t_confidence_ci, e_confidence_ci, b_confidence_ci, weights
        )
        
        return trust_score, trust_score_ci, trust_confidence, trust_confidence_ci
    
    def _calculate_severity_confidence_interval(self, severity_scores: List[float]) -> ConfidenceInterval:
        """
        Calculate confidence interval for severity scores (around the sum, in severity space).
        
        Note: The aggregated score is the SUM of severity scores, so the CI is computed
        around the sum to maintain unit consistency.
        
        Args:
            severity_scores: List of weighted severity scores
            
        Returns:
            ConfidenceInterval: Calculated confidence interval around the sum (in severity space)
        """
        if not severity_scores:
            return ConfidenceInterval(lower=None, upper=None)
        
        # Calculate sum of scores (matches aggregated score calculation)
        sum_score: float = sum(severity_scores)
        
        # Calculate standard error of the SUM (not the mean)
        # For independent scores: Var(sum) = n * Var(individual), so SE(sum) = sqrt(n) * std(individual)
        if len(severity_scores) > 1:
            score_std: float = statistics.stdev(severity_scores)
            # Standard error of sum = sqrt(n) * standard deviation
            n: int = len(severity_scores)
            sum_sem: float = score_std * (n ** 0.5)
        else:
            sum_sem: float = 0
        
        # Use confidence level from config
        confidence_level: float = self.config.confidence_level
        
        # Use configurable statistical parameters
        stat_config = self.config.statistical
        
        # Calculate t-statistic for small samples (more accurate than z-score)
        if len(severity_scores) <= stat_config.min_sample_size_for_t_dist:
            # Use t-distribution for small samples with configurable values
            t_critical = stat_config.t_critical_values.get(len(severity_scores), 2.0)
        else:
            # Use z-score for large samples with configurable values
            t_critical = stat_config.fallback_z_scores.get(confidence_level, 1.96)
        
        margin_of_error: float = t_critical * sum_sem
        
        # Apply configurable confidence margin
        if stat_config.use_continuity_correction:
            margin_of_error += stat_config.confidence_margin
        
        # Compute CI bounds around the sum (consistent with aggregated score units, in severity space)
        lower_bound: float = sum_score - margin_of_error
        upper_bound: float = sum_score + margin_of_error
        
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound)
    
    def _calculate_mean_confidence_interval(self, confidences: List[float]) -> ConfidenceInterval:
        """
        Calculate confidence interval for mean confidence (around the mean, in probability space [0-1]).
        
        Note: This calculates a CI around the MEAN of confidences, not the sum,
        since confidence is already bounded in [0-1] probability space.
        
        Args:
            confidences: List of confidence values (in probability space [0-1])
            
        Returns:
            ConfidenceInterval: Calculated confidence interval around the mean (in probability space [0-1])
        """
        if not confidences:
            return ConfidenceInterval(lower=None, upper=None)
        
        # Calculate mean confidence
        mean_confidence: float = statistics.mean(confidences)
        
        # Calculate standard error of the MEAN (not the sum)
        if len(confidences) > 1:
            confidence_std: float = statistics.stdev(confidences)
            # Standard error of mean = std / sqrt(n)
            n: int = len(confidences)
            mean_sem: float = confidence_std / (n ** 0.5)
        else:
            mean_sem: float = 0
        
        # Use confidence level from config
        confidence_level: float = self.config.confidence_level
        
        # Use configurable statistical parameters
        stat_config = self.config.statistical
        
        # Calculate t-statistic for small samples (more accurate than z-score)
        if len(confidences) <= stat_config.min_sample_size_for_t_dist:
            # Use t-distribution for small samples with configurable values
            t_critical = stat_config.t_critical_values.get(len(confidences), 2.0)
        else:
            # Use z-score for large samples with configurable values
            t_critical = stat_config.fallback_z_scores.get(confidence_level, 1.96)
        
        margin_of_error: float = t_critical * mean_sem
        
        # Apply configurable confidence margin
        if stat_config.use_continuity_correction:
            margin_of_error += stat_config.confidence_margin
        
        # Compute CI bounds around the mean (in probability space [0-1], clamped to valid range)
        lower_bound: float = max(0.0, mean_confidence - margin_of_error)
        upper_bound: float = min(1.0, mean_confidence + margin_of_error)
        
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound)
    
    def _propagate_severity_ci(self, t_ci: ConfidenceInterval, e_ci: ConfidenceInterval,
                              b_ci: ConfidenceInterval, weights, aggregation_method: str) -> ConfidenceInterval:
        """
        Propagate uncertainty for severity score CIs using proper error propagation (in severity space).
        
        For a weighted combination Y = w1*X1 + w2*X2 + w3*X3, the variance is:
        Var(Y) = w1^2 * Var(X1) + w2^2 * Var(X2) + w3^2 * Var(X3) (assuming independence)
        
        Args:
            t_ci, e_ci, b_ci: Category severity score CIs (in severity space)
            weights: Aggregation weights
            aggregation_method: Method used to aggregate scores
            
        Returns:
            ConfidenceInterval: Propagated CI for trust score (in severity space)
        """
        if not all(ci.lower is not None and ci.upper is not None 
                   for ci in [t_ci, e_ci, b_ci]):
            return ConfidenceInterval(lower=None, upper=None)
        
        # Extract centers and half-widths from input CIs
        centers = []
        half_widths = []
        cis = [t_ci, e_ci, b_ci]
        weights_list = [weights.trustworthiness, weights.explainability, weights.bias]
        
        for ci, weight in zip(cis, weights_list):
            center = (ci.lower + ci.upper) / 2
            half_width = (ci.upper - ci.lower) / 2
            centers.append(center)
            half_widths.append(half_width)
        
        # Calculate weighted center (this matches the trust_score calculation)
        if aggregation_method == "weighted_mean":
            weighted_center = (
                weights.trustworthiness * centers[0] +
                weights.explainability * centers[1] +
                weights.bias * centers[2]
            )
        elif aggregation_method == "median":
            weighted_center = statistics.median(centers)
        elif aggregation_method == "robust_mean":
            weighted_center = statistics.mean(centers)
        elif aggregation_method == "max":
            weighted_center = max(centers)
        elif aggregation_method == "min":
            weighted_center = min(centers)
        elif aggregation_method == "geometric_mean":
            weighted_center = statistics.geometric_mean(centers)
        else:
            # Default to weighted mean
            weighted_center = (
                weights.trustworthiness * centers[0] +
                weights.explainability * centers[1] +
                weights.bias * centers[2]
            )
        
        # Propagate uncertainty using error propagation
        # For weighted combination: Var(Y) = sum(w_i^2 * Var(X_i))
        # Using half-width as uncertainty measure: SE ≈ half_width
        # Combined SE = sqrt(sum(w_i^2 * half_width_i^2))
        propagated_uncertainty: float = (
            (weights.trustworthiness ** 2) * (half_widths[0] ** 2) +
            (weights.explainability ** 2) * (half_widths[1] ** 2) +
            (weights.bias ** 2) * (half_widths[2] ** 2)
        ) ** 0.5
        
        # Calculate CI bounds (no clamping since we're in severity space)
        lower_bound: float = weighted_center - propagated_uncertainty
        upper_bound: float = weighted_center + propagated_uncertainty
        
        return ConfidenceInterval(lower=lower_bound, upper=upper_bound)
    
    def _propagate_confidence_ci(self, t_ci: ConfidenceInterval, e_ci: ConfidenceInterval,
                                 b_ci: ConfidenceInterval, weights) -> ConfidenceInterval:
        """
        Propagate uncertainty for confidence CIs using proper error propagation (in probability space [0-1]).
        
        For a weighted combination Y = w1*X1 + w2*X2 + w3*X3, the variance is:
        Var(Y) = w1^2 * Var(X1) + w2^2 * Var(X2) + w3^2 * Var(X3) (assuming independence)
        
        Args:
            t_ci, e_ci, b_ci: Category confidence CIs (in probability space [0-1])
            weights: Aggregation weights
            
        Returns:
            ConfidenceInterval: Propagated CI for trust confidence (in probability space [0-1], clamped)
        """
        if not all(ci.lower is not None and ci.upper is not None 
                   for ci in [t_ci, e_ci, b_ci]):
            return ConfidenceInterval(lower=None, upper=None)
        
        # Extract centers and half-widths from input CIs
        centers = []
        half_widths = []
        cis = [t_ci, e_ci, b_ci]
        weights_list = [weights.trustworthiness, weights.explainability, weights.bias]
        
        for ci, weight in zip(cis, weights_list):
            center = (ci.lower + ci.upper) / 2
            half_width = (ci.upper - ci.lower) / 2
            centers.append(center)
            half_widths.append(half_width)
        
        # Calculate weighted center (weighted average of confidences)
        weighted_center: float = (
            weights.trustworthiness * centers[0] +
            weights.explainability * centers[1] +
            weights.bias * centers[2]
        )
        
        # Propagate uncertainty using error propagation
        # For weighted combination: Var(Y) = sum(w_i^2 * Var(X_i))
        # Using half-width as uncertainty measure: SE ≈ half_width
        # Combined SE = sqrt(sum(w_i^2 * half_width_i^2))
        propagated_uncertainty: float = (
            (weights.trustworthiness ** 2) * (half_widths[0] ** 2) +
            (weights.explainability ** 2) * (half_widths[1] ** 2) +
            (weights.bias ** 2) * (half_widths[2] ** 2)
        ) ** 0.5
        
        # Calculate CI bounds (clamped to [0, 1] since we're in probability space)
        lower_bound: float = max(0.0, weighted_center - propagated_uncertainty)
        upper_bound: float = min(1.0, weighted_center + propagated_uncertainty)
        
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
            
            # Collect all judge-level scores and confidences
            severity_scores: List[float] = [analysis.severity_score for analysis in span.analysis.values()]
            confidences: List[float] = [analysis.confidence for analysis in span.analysis.values()]
            
            # Calculate statistical confidence interval for severity score (in severity space)
            # Based on variance in judge-level severity scores
            if len(severity_scores) > 1:
                # Calculate standard error of the mean for severity scores
                severity_std: float = statistics.stdev(severity_scores)
                severity_sem: float = severity_std / (len(severity_scores) ** 0.5)  # Standard Error of Mean
                
                # Use confidence level from config
                confidence_level: float = self.config.confidence_level
                
                # Use configurable statistical parameters
                stat_config = self.config.statistical
                
                # Calculate t-statistic for small samples (more accurate than z-score)
                if len(severity_scores) <= stat_config.min_sample_size_for_t_dist:
                    # Use t-distribution for small samples with configurable values
                    t_critical = stat_config.t_critical_values.get(len(severity_scores), 2.0)
                else:
                    # Use z-score for large samples with configurable values
                    t_critical = stat_config.fallback_z_scores.get(confidence_level, 1.96)
                
                margin_of_error: float = t_critical * severity_sem
                
                # Apply configurable confidence margin
                if stat_config.use_continuity_correction:
                    margin_of_error += stat_config.confidence_margin
                
                # Compute CI bounds around mean severity score (in severity space)
                severity_score_ci: ConfidenceInterval = ConfidenceInterval(
                    lower=avg_severity - margin_of_error,
                    upper=avg_severity + margin_of_error
                )
            else:
                # Single judge or no variance - use fixed margin as fallback
                severity_score_ci: ConfidenceInterval = ConfidenceInterval(
                    lower=avg_severity - 0.1,
                    upper=avg_severity + 0.1
                )
            
            # Calculate statistical confidence interval for confidence (in probability space [0-1])
            # Based on variance in judge-level confidences
            if len(confidences) > 1:
                # Calculate standard error of the mean for confidences
                confidence_std: float = statistics.stdev(confidences)
                confidence_sem: float = confidence_std / (len(confidences) ** 0.5)  # Standard Error of Mean
                
                # Use confidence level from config
                confidence_level: float = self.config.confidence_level
                
                # Use configurable statistical parameters
                stat_config = self.config.statistical
                
                # Calculate t-statistic for small samples (more accurate than z-score)
                if len(confidences) <= stat_config.min_sample_size_for_t_dist:
                    # Use t-distribution for small samples with configurable values
                    t_critical = stat_config.t_critical_values.get(len(confidences), 2.0)
                else:
                    # Use z-score for large samples with configurable values
                    t_critical = stat_config.fallback_z_scores.get(confidence_level, 1.96)
                
                margin_of_error: float = t_critical * confidence_sem
                
                # Apply configurable confidence margin
                if stat_config.use_continuity_correction:
                    margin_of_error += stat_config.confidence_margin
                
                # Compute CI bounds around mean confidence (in probability space [0-1])
                confidence_ci: ConfidenceInterval = ConfidenceInterval(
                    lower=max(0.0, avg_confidence - margin_of_error),
                    upper=min(1.0, avg_confidence + margin_of_error)
                )
            else:
                # Single judge or no variance - use fixed margin as fallback
                confidence_ci: ConfidenceInterval = ConfidenceInterval(
                    lower=max(0.0, avg_confidence - 0.05),
                    upper=min(1.0, avg_confidence + 0.05)
                )
            
            # Extract weights from first judge analysis (all judges of same type should have same weights)
            weights: Optional[JudgeWeights] = None
            if span.analysis:
                first_analysis = next(iter(span.analysis.values()))
                weights = first_analysis.weights
            
            error_summary: ErrorSummary = ErrorSummary(
                type=span.type,
                subtype=span.subtype,
                severity_bucket=severity_bucket,
                severity_score=avg_severity,
                severity_score_ci=severity_score_ci,
                confidence_level=avg_confidence,
                confidence_ci=confidence_ci,
                explanation=span.explanation,
                weights=weights
            )
            
            error_summaries[span_id] = error_summary
        
        return error_summaries
