"""
TrustScore Pipeline - Main Orchestrator

This module coordinates the entire TrustScore pipeline, orchestrating
the span tagger, judges, and aggregator components.
"""

import asyncio
import statistics
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from models.llm_record import LLMRecord, SpansLevelTags, GradedSpans, GradedSpan, AggregatedOutput
from modules.span_tagger import SpanTagger, MockSpanTagger
from modules.judges.base_judge import BaseJudge
from modules.judges.trustworthiness_judge import TrustworthinessJudge
from modules.judges.bias_judge import BiasJudge
from modules.judges.explainability_judge import ExplainabilityJudge
from modules.aggregator import Aggregator
from config.settings import TrustScoreConfig, load_config


class TrustScorePipeline:
    """
    Main orchestrator for the TrustScore pipeline.
    
    This class coordinates the entire process:
    1. Ingests LLM prompt and response
    2. Processes into LLMRecord format
    3. Calls span tagger to identify errors
    4. Calls judges to score error severity
    5. Aggregates scores into final TrustScore
    """
    
    def __init__(self, config: Optional[TrustScoreConfig] = None, api_key: Optional[str] = None, use_mock: bool = False) -> None:
        self.config: TrustScoreConfig = config or load_config()
        self.api_key: Optional[str] = api_key
        self.use_mock: bool = use_mock
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components with ensemble judges."""
        # Initialize span tagger
        if self.use_mock:
            self.span_tagger: SpanTagger = MockSpanTagger(self.config.span_tagger)
        else:
            self.span_tagger: SpanTagger = SpanTagger(self.config.span_tagger, self.api_key)
        
        # Initialize judges by aspect (ensemble approach)
        self.judges: Dict[str, Dict[str, BaseJudge]] = {
            "trustworthiness": {},
            "bias": {},
            "explainability": {}
        }
        
        for judge_name, judge_config in self.config.judges.items():
            if not judge_config.enabled:
                continue
                
            # Route judges by name pattern
            if "trust" in judge_name.lower():
                judge = TrustworthinessJudge(judge_config, self.config, self.api_key)
                self.judges["trustworthiness"][judge_name] = judge
            elif "bias" in judge_name.lower():
                judge = BiasJudge(judge_config, self.config, self.api_key)
                self.judges["bias"][judge_name] = judge
            elif "explain" in judge_name.lower():
                judge = ExplainabilityJudge(judge_config, self.config, self.api_key)
                self.judges["explainability"][judge_name] = judge
            else:
                # Default to trustworthiness judge for generic judges
                judge = TrustworthinessJudge(judge_config, self.config, self.api_key)
                self.judges["trustworthiness"][judge_name] = judge
        
        # Initialize aggregator
        self.aggregator: Aggregator = Aggregator(self.config)
    
    def process(self, prompt: str, response: str, model: str = "unknown", 
               generated_on: Optional[datetime] = None, generation_seed: Optional[int] = None) -> AggregatedOutput:
        """
        Process a single LLM prompt/response pair through the pipeline.
        
        Args:
            prompt: Input prompt to the LLM
            response: LLM's response
            model: Model name/identifier
            generated_on: When the response was generated
            generation_seed: Optional base seed for deterministic generation. 
                           Individual judge seeds will be derived from this.
                           If None, uses natural randomness.
            
        Returns:
            AggregatedOutput: Final TrustScore analysis
        """
        # Step 1: Create LLMRecord
        llm_record: LLMRecord = self._create_llm_record(prompt, response, model, generated_on)
        
        # Step 2: Tag spans
        print("[Pipeline Step 2/4] Span Annotation: Tagging error spans...")
        spans_tags: SpansLevelTags = self.span_tagger.tag_spans(llm_record)
        
        # Count spans by type for debugging
        span_counts = {"T": 0, "B": 0, "E": 0}
        for span_id, span in spans_tags.spans.items():
            span_counts[span.type.value] = span_counts.get(span.type.value, 0) + 1
        
        print(f"[Pipeline Step 2/4] Span Annotation Complete: Detected {len(spans_tags.spans)} error span(s)")
        print(f"  - Trustworthiness (T): {span_counts['T']}")
        print(f"  - Bias (B): {span_counts['B']}")
        print(f"  - Explainability (E): {span_counts['E']}")
        
        # Step 3: Grade spans with judges
        print(f"[Pipeline Step 3/4] Span Grading: Analyzing {len(spans_tags.spans)} span(s) with judges...")
        graded_spans: GradedSpans = self._grade_spans(llm_record, spans_tags, generation_seed=generation_seed)
        print(f"[Pipeline Step 3/4] Span Grading Complete: Successfully graded {len(graded_spans.spans)} span(s)")
        
        # Step 4: Aggregate scores
        print("[Pipeline Step 4/4] Aggregation: Computing final TrustScore...")
        aggregated_output: AggregatedOutput = self.aggregator.aggregate(llm_record, graded_spans)
        print(f"[Pipeline Complete] Final TrustScore: {aggregated_output.summary.trust_score:.3f}")
        
        return aggregated_output
    
    def process_batch(self, inputs: List[Dict[str, Any]]) -> List[AggregatedOutput]:
        """
        Process multiple LLM prompt/response pairs through the pipeline.
        
        Args:
            inputs: List of dictionaries with 'prompt', 'response', 'model' keys
            
        Returns:
            List of AggregatedOutput: Final TrustScore analyses
        """
        results: List[AggregatedOutput] = []
        total_samples = len(inputs)
        
        for idx, input_data in enumerate(inputs, start=1):
            print(f"\n{'='*70}")
            print(f"Processing sample {idx}/{total_samples} ({total_samples - idx} remaining)")
            print(f"{'='*70}")
            try:
                result: AggregatedOutput = self.process(
                    prompt=input_data['prompt'],
                    response=input_data['response'],
                    model=input_data.get('model', 'unknown'),
                    generated_on=input_data.get('generated_on')
                )
                results.append(result)
                print(f"[Batch Processing] Sample {idx}/{total_samples} completed successfully")
            except Exception as e:
                # Create empty result for failed processing
                print(f"[Batch Processing] Sample {idx}/{total_samples} failed: {str(e)}")
                results.append(None)
        
        return results
    
    def _create_llm_record(self, prompt: str, response: str, model: str, 
                          generated_on: Optional[datetime]) -> LLMRecord:
        """Create LLMRecord from input data."""
        if generated_on is None:
            generated_on = datetime.now()
        
        from models.llm_record import ModelMetadata
        metadata: ModelMetadata = ModelMetadata(
            model=model,
            generated_on=generated_on
        )
        
        return LLMRecord(
            task_prompt=prompt,
            llm_response=response,
            model_metadata=metadata
        )
    
    def _grade_spans(self, llm_record: LLMRecord, spans_tags: SpansLevelTags, 
                     generation_seed: Optional[int] = None) -> GradedSpans:
        """
        Grade spans using ensemble of judges for each aspect with configurable error handling.
        
        Args:
            llm_record: The original LLM input/output pair
            spans_tags: Collection of identified error spans
            generation_seed: Optional base seed for deterministic generation.
                           Individual judge seeds will be derived from this.
                           If None, uses natural randomness.
            
        Returns:
            GradedSpans: Collection of graded spans with ensemble analyses
        """
        graded_spans: GradedSpans = GradedSpans()
        ensemble_config = self.config.ensemble
        error_config = self.config.error_handling
        
        # LOG: Count spans by type before processing
        span_type_counts = {"T": 0, "B": 0, "E": 0}
        for span_id, span in spans_tags.spans.items():
            span_type_counts[span.type.value] = span_type_counts.get(span.type.value, 0) + 1
        
        print(f"[DEBUG Judge Calls] Spans detected: T={span_type_counts['T']}, B={span_type_counts['B']}, E={span_type_counts['E']}")
        print(f"[DEBUG Judge Calls] Available judges: T={list(self.judges['trustworthiness'].keys())}, "
              f"B={list(self.judges['bias'].keys())}, E={list(self.judges['explainability'].keys())}")
        
        for span_id, span in spans_tags.spans.items():
            graded_span: GradedSpan = GradedSpan(
                start=span.start,
                end=span.end,
                type=span.type,
                subtype=span.subtype,
                explanation=span.explanation
            )
            
            # Get appropriate judges for this error type
            aspect_judges: Dict[str, BaseJudge] = {}
            if span.type.value == "T":  # Trustworthiness
                aspect_judges = self.judges["trustworthiness"]
            elif span.type.value == "B":  # Bias
                aspect_judges = self.judges["bias"]
            elif span.type.value == "E":  # Explainability
                aspect_judges = self.judges["explainability"]
            
            # LOG: Judge selection for this span
            print(f"[DEBUG Judge Calls] Span {span_id} (type={span.type.value}, subtype={span.subtype}): "
                  f"Found {len(aspect_judges)} judge(s)")
            
            if len(aspect_judges) == 0:
                print(f"[WARNING Judge Calls] No judges available for span type {span.type.value}! "
                      f"Span {span_id} will not be graded.")
            
            # Limit judges per aspect if configured
            if len(aspect_judges) > ensemble_config.max_judges_per_aspect:
                aspect_judges = dict(list(aspect_judges.items())[:ensemble_config.max_judges_per_aspect])
            
            # Track judge failures
            judge_failures = 0
            successful_analyses = 0
            
            # Get analyses from all judges for this aspect
            for judge_idx, (judge_name, judge) in enumerate(aspect_judges.items()):
                print(f"[DEBUG Judge Calls] Calling judge '{judge_name}' for span {span_id} (type={span.type.value})")
                try:
                    # Generate deterministic seed for this judge if base seed is provided
                    judge_seed = None
                    if generation_seed is not None:
                        # Create unique seed: base_seed + span_id hash + judge_idx
                        # Use a simple hash of span_id for uniqueness
                        span_hash = hash(span_id) % 10000  # Limit to reasonable range
                        judge_seed = generation_seed + span_hash + judge_idx
                        print(f"[DEBUG Seed] Judge '{judge_name}' for span {span_id}: base_seed={generation_seed}, span_hash={span_hash}, judge_idx={judge_idx}, final_seed={judge_seed}")
                    else:
                        print(f"[DEBUG Seed] Judge '{judge_name}' for span {span_id}: No seed provided (using natural randomness)")
                    
                    analysis = judge.analyze_span(llm_record, span, seed=judge_seed)
                    
                    # LOG: Successful judge call with details
                    print(f"[DEBUG Judge Calls] ✓ Judge '{judge_name}' succeeded for span {span_id}: "
                          f"severity_score={analysis.severity_score:.3f}, "
                          f"confidence={analysis.confidence:.3f}, "
                          f"severity_bucket={analysis.severity_bucket.value}")
                    
                    graded_span.add_judge_analysis(judge_name, analysis)
                    successful_analyses += 1
                except Exception as e:
                    judge_failures += 1
                    # LOG: Judge failure with full details
                    print(f"[ERROR Judge Calls] ✗ Judge '{judge_name}' FAILED for span {span_id} (type={span.type.value}): {str(e)}")
                    print(f"[ERROR Judge Calls]   Exception type: {type(e).__name__}")
                    import traceback
                    print(f"[ERROR Judge Calls]   Traceback: {traceback.format_exc()}")
                    
                    if error_config.log_level in ["DEBUG", "INFO", "WARNING"]:
                        print(f"Error from judge {judge_name}: {str(e)}")
                    
                    # Check if we should fail fast
                    if error_config.fail_fast and judge_failures > error_config.max_judge_failures:
                        if error_config.continue_on_span_errors:
                            break
                        else:
                            raise RuntimeError(f"Too many judge failures ({judge_failures})")
                    
                    continue
            
            # LOG: Summary for this span
            print(f"[DEBUG Judge Calls] Span {span_id} summary: {successful_analyses} successful, {judge_failures} failed")
            
            # Check if we have enough analyses based on configuration
            if successful_analyses >= ensemble_config.min_judges_required:
                # Apply consensus requirements if configured
                if ensemble_config.require_consensus:
                    if self._check_consensus(graded_span, ensemble_config.consensus_threshold):
                        graded_spans.add_graded_span(span_id, graded_span)
                        print(f"[DEBUG Judge Calls] Span {span_id} added to graded_spans (consensus passed)")
                    else:
                        print(f"[DEBUG Judge Calls] Span {span_id} NOT added (consensus failed)")
                else:
                    graded_spans.add_graded_span(span_id, graded_span)
                    print(f"[DEBUG Judge Calls] Span {span_id} added to graded_spans")
            elif not error_config.continue_on_span_errors:
                raise RuntimeError(f"Insufficient judge analyses: {successful_analyses} < {ensemble_config.min_judges_required}")
            else:
                print(f"[WARNING Judge Calls] Span {span_id} NOT added: insufficient analyses "
                      f"({successful_analyses} < {ensemble_config.min_judges_required})")
        
        # LOG: Final summary
        graded_type_counts = {"T": 0, "B": 0, "E": 0}
        for span_id, span in graded_spans.spans.items():
            graded_type_counts[span.type.value] = graded_type_counts.get(span.type.value, 0) + 1
        
        print(f"[DEBUG Judge Calls] Final graded spans: T={graded_type_counts['T']}, "
              f"B={graded_type_counts['B']}, E={graded_type_counts['E']}")
        
        return graded_spans
    
    def _check_consensus(self, graded_span: GradedSpan, consensus_threshold: float) -> bool:
        """
        Check if there's consensus among judges for a span.
        
        Args:
            graded_span: The graded span to check
            consensus_threshold: Threshold for consensus (0.0 to 1.0)
            
        Returns:
            bool: True if consensus is reached
        """
        if len(graded_span.analysis) < 2:
            return True  # No consensus needed for single judge
        
        # Get severity scores from all judges
        severity_scores = [analysis.severity_score for analysis in graded_span.analysis.values()]
        
        # Calculate coefficient of variation as a measure of consensus
        mean_score = statistics.mean(severity_scores)
        if mean_score == 0:
            return True  # All scores are 0, perfect consensus
        
        std_score = statistics.stdev(severity_scores)
        cv = std_score / abs(mean_score)  # Coefficient of variation
        
        # Lower CV means higher consensus
        consensus_achieved = cv <= (1.0 - consensus_threshold)
        
        return consensus_achieved
    
    def get_judge_info_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Get judge information map for output formatting.
        
        Returns:
            Dict mapping judge_name -> {"model": str, ...}
        """
        judge_info_map: Dict[str, Dict[str, Any]] = {}
        for aspect, judges in self.judges.items():
            for judge_name, judge in judges.items():
                judge_info_map[judge_name] = {
                    "model": judge.config.model,
                    "provider": judge.config.provider.value if hasattr(judge.config.provider, 'value') else str(judge.config.provider),
                    "aspect": aspect
                }
        return judge_info_map
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get status information about the pipeline.
        
        Returns:
            Dict with pipeline status information
        """
        # Flatten judges for status display
        all_judges = {}
        for aspect, judges in self.judges.items():
            for name, judge in judges.items():
                all_judges[f"{aspect}_{name}"] = {
                    "type": type(judge).__name__,
                    "model": judge.config.model,
                    "enabled": judge.config.enabled,
                    "aspect": aspect
                }
        
        return {
            "span_tagger": {
                "type": type(self.span_tagger).__name__,
                "model": self.config.span_tagger.model,
                "mock_mode": self.use_mock
            },
            "judges": all_judges,
            "judge_ensemble": {
                aspect: {
                    "count": len(judges),
                    "judges": list(judges.keys())
                }
                for aspect, judges in self.judges.items()
            },
            "aggregator": {
                "weights": {
                    "trustworthiness": self.config.aggregation_weights.trustworthiness,
                    "explainability": self.config.aggregation_weights.explainability,
                    "bias": self.config.aggregation_weights.bias
                },
                "confidence_level": self.config.confidence_level,
                "aggregation_method": self.config.aggregation_strategy.aggregation_method,
                "use_robust_statistics": self.config.aggregation_strategy.use_robust_statistics
            },
            "ensemble_config": {
                "min_judges_required": self.config.ensemble.min_judges_required,
                "require_consensus": self.config.ensemble.require_consensus,
                "consensus_threshold": self.config.ensemble.consensus_threshold,
                "outlier_detection": self.config.ensemble.outlier_detection
            },
            "error_handling": {
                "max_judge_failures": self.config.error_handling.max_judge_failures,
                "fail_fast": self.config.error_handling.fail_fast,
                "log_level": self.config.error_handling.log_level
            },
            "performance": {
                "max_concurrent_judges": self.config.performance.max_concurrent_judges,
                "enable_parallel_processing": self.config.performance.enable_parallel_processing,
                "batch_processing": self.config.performance.batch_processing
            }
        }


# Convenience function for quick usage
def analyze_llm_response(prompt: str, response: str, model: str = "unknown", 
                        api_key: Optional[str] = None, use_mock: bool = False) -> AggregatedOutput:
    """
    Quick function to analyze a single LLM response.
    
    Args:
        prompt: Input prompt
        response: LLM response
        model: Model name
        api_key: OpenAI API key
        use_mock: Whether to use mock components
        
    Returns:
        AggregatedOutput: TrustScore analysis
    """
    pipeline: TrustScorePipeline = TrustScorePipeline(api_key=api_key, use_mock=use_mock)
    return pipeline.process(prompt, response, model)
