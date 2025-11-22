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
from modules.judges.trustworthiness_judge import TrustworthinessJudge, MockTrustworthinessJudge
from modules.judges.bias_judge import BiasJudge, MockBiasJudge
from modules.judges.explainability_judge import ExplainabilityJudge, MockExplainabilityJudge
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
                
            # Route judges by name pattern - use mock judges if in mock mode
            if "trust" in judge_name.lower():
                if self.use_mock:
                    judge = MockTrustworthinessJudge(judge_config, self.config, self.api_key)
                else:
                    judge = TrustworthinessJudge(judge_config, self.config, self.api_key)
                self.judges["trustworthiness"][judge_name] = judge
            elif "bias" in judge_name.lower():
                if self.use_mock:
                    judge = MockBiasJudge(judge_config, self.config, self.api_key)
                else:
                    judge = BiasJudge(judge_config, self.config, self.api_key)
                self.judges["bias"][judge_name] = judge
            elif "explain" in judge_name.lower():
                if self.use_mock:
                    judge = MockExplainabilityJudge(judge_config, self.config, self.api_key)
                else:
                    judge = ExplainabilityJudge(judge_config, self.config, self.api_key)
                self.judges["explainability"][judge_name] = judge
            else:
                # Default to trustworthiness judge for generic judges
                if self.use_mock:
                    judge = MockTrustworthinessJudge(judge_config, self.config, self.api_key)
                else:
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
    
    def _is_vllm_provider(self, judge: BaseJudge) -> bool:
        """Check if a judge uses vLLM provider"""
        return (hasattr(judge, 'llm_provider') and 
                hasattr(judge.llm_provider, '__class__') and
                judge.llm_provider.__class__.__name__ == 'VLLMProvider')
    
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
        
        Uses optimized strategies based on provider:
        - vLLM: Batches spans by judge type for efficient batch inference
        - Other providers: Parallelizes judge calls using ThreadPoolExecutor
        - Fallback: Sequential processing if parallelization is disabled
        
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
        
        performance_config = self.config.performance
        
        # Group spans by error type for efficient processing
        spans_by_type: Dict[str, List[tuple[str, Any]]] = {"T": [], "B": [], "E": []}
        for span_id, span in spans_tags.spans.items():
            spans_by_type[span.type.value].append((span_id, span))
        
        # Process each error type with optimized strategy
        for error_type, span_list in spans_by_type.items():
            if not span_list:
                continue
            
            # Get appropriate judges for this error type
            aspect_judges: Dict[str, BaseJudge] = {}
            if error_type == "T":
                aspect_judges = self.judges["trustworthiness"]
            elif error_type == "B":
                aspect_judges = self.judges["bias"]
            elif error_type == "E":
                aspect_judges = self.judges["explainability"]
            
            if len(aspect_judges) == 0:
                print(f"[WARNING Judge Calls] No judges available for error type {error_type}!")
                continue
            
            # Limit judges per aspect if configured
            if len(aspect_judges) > ensemble_config.max_judges_per_aspect:
                aspect_judges = dict(list(aspect_judges.items())[:ensemble_config.max_judges_per_aspect])
            
            # Check if any judge uses vLLM (for batching optimization)
            use_vllm = any(self._is_vllm_provider(judge) for judge in aspect_judges.values())
            
            if use_vllm and performance_config.enable_parallel_processing:
                # vLLM: Use batching for efficiency
                self._grade_spans_with_batching(
                    llm_record, span_list, aspect_judges, error_type,
                    graded_spans, ensemble_config, error_config,
                    generation_seed=generation_seed
                )
            elif performance_config.enable_parallel_processing:
                # Other providers: Use parallelization
                self._grade_spans_with_parallelization(
                    llm_record, span_list, aspect_judges, error_type,
                    graded_spans, ensemble_config, error_config, performance_config
                )
            else:
                # Sequential processing (fallback)
                self._grade_spans_sequential(
                    llm_record, span_list, aspect_judges, error_type,
                    graded_spans, ensemble_config, error_config
                )
        
        # LOG: Final summary
        graded_type_counts = {"T": 0, "B": 0, "E": 0}
        for span_id, span in graded_spans.spans.items():
            graded_type_counts[span.type.value] = graded_type_counts.get(span.type.value, 0) + 1
        
        print(f"[DEBUG Judge Calls] Final graded spans: T={graded_type_counts['T']}, "
              f"B={graded_type_counts['B']}, E={graded_type_counts['E']}")
        
        return graded_spans
    
    def _grade_spans_with_batching(self, llm_record: LLMRecord, span_list: List[tuple[str, Any]],
                                   aspect_judges: Dict[str, BaseJudge], error_type: str,
                                   graded_spans: GradedSpans, ensemble_config, error_config,
                                   generation_seed: Optional[int] = None) -> None:
        """Grade spans using batching (optimized for vLLM)"""
        import random as rng
        
        print(f"[DEBUG Judge Calls] Using BATCHING for {error_type} spans (vLLM optimization)")
        
        # Process each judge separately (each judge batches all spans)
        for judge_idx, (judge_name, judge) in enumerate(aspect_judges.items()):
            try:
                # Prepare span records for batching: (LLMRecord, SpanTag) tuples
                span_records = [(llm_record, span) for _, span in span_list]
                
                # Derive unique seeds per judge AND per span item
                # Formula: base_seed + (judge_idx * 10000) + (span_idx * 100)
                # This ensures:
                # - Different seeds across judges (judge_idx offset)
                # - Different seeds across batch items (span_idx offset)
                # - Controlled/reproducible when generation_seed is provided
                if generation_seed is not None:
                    # Derive unique seed for each span in the batch (reproducible)
                    span_seeds = [
                        generation_seed + (judge_idx * 10000) + (span_idx * 100)
                        for span_idx in range(len(span_list))
                    ]
                    print(f"[DEBUG Seeds] Judge {judge_idx}, generation_seed={generation_seed}, span_seeds={span_seeds}")
                    # Batch analyze all spans for this judge with unique seeds per span
                    analyses = judge.batch_analyze_spans(span_records, seeds=span_seeds)
                else:
                    # Backward compatibility: use single random seed per judge (old behavior)
                    # This preserves the behavior for sensitivity/specificity analyses
                    judge_seed = rng.randint(1, 2**31 - 1)  # Random seed per judge
                    analyses = judge.batch_analyze_spans(span_records, seed=judge_seed)
                
                # Assign analyses to corresponding spans
                for (span_id, span), analysis in zip(span_list, analyses):
                    # Get or create graded span
                    if span_id not in graded_spans.spans:
                        graded_span = GradedSpan(
                            start=span.start,
                            end=span.end,
                            type=span.type,
                            subtype=span.subtype,
                            explanation=span.explanation
                        )
                        graded_spans.add_graded_span(span_id, graded_span)
                    else:
                        graded_span = graded_spans.spans[span_id]
                    
                    graded_span.add_judge_analysis(judge_name, analysis)
                    print(f"[DEBUG Judge Calls] ✓ Judge '{judge_name}' batched analysis for span {span_id}")
                    
            except Exception as e:
                print(f"[ERROR Judge Calls] ✗ Judge '{judge_name}' batch analysis FAILED: {str(e)}")
                import traceback
                print(f"[ERROR Judge Calls]   Traceback: {traceback.format_exc()}")
                if error_config.fail_fast:
                    raise
                continue
        
        # Validate and filter spans based on ensemble requirements
        self._validate_and_filter_spans(span_list, graded_spans, ensemble_config, error_config)
    
    def _grade_spans_with_parallelization(self, llm_record: LLMRecord, span_list: List[tuple[str, Any]],
                                          aspect_judges: Dict[str, BaseJudge], error_type: str,
                                          graded_spans: GradedSpans, ensemble_config, error_config,
                                          performance_config) -> None:
        """Grade spans using parallelization (optimized for non-vLLM providers)"""
        import random as rng
        
        print(f"[DEBUG Judge Calls] Using PARALLELIZATION for {error_type} spans")
        
        max_workers = min(performance_config.max_concurrent_judges, len(aspect_judges))
        
        # Process each span
        for span_id, span in span_list:
            graded_span = GradedSpan(
                start=span.start,
                end=span.end,
                type=span.type,
                subtype=span.subtype,
                explanation=span.explanation
            )
            
            # Parallelize judge calls for this span
            def call_judge(judge_name_judge_pair):
                judge_name, judge = judge_name_judge_pair
                try:
                    judge_seed = rng.randint(1, 2**31 - 1)
                    analysis = judge.analyze_span(llm_record, span, seed=judge_seed)
                    return (judge_name, analysis, None)
                except Exception as e:
                    return (judge_name, None, e)
            
            # Use ThreadPoolExecutor for parallel judge calls
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(call_judge, (name, judge)): name 
                          for name, judge in aspect_judges.items()}
                
                judge_failures = 0
                successful_analyses = 0
                
                for future in as_completed(futures):
                    judge_name, analysis, error = future.result()
                    if error:
                        judge_failures += 1
                        print(f"[ERROR Judge Calls] ✗ Judge '{judge_name}' FAILED for span {span_id}: {str(error)}")
                        if error_config.fail_fast and judge_failures > error_config.max_judge_failures:
                            if not error_config.continue_on_span_errors:
                                raise RuntimeError(f"Too many judge failures ({judge_failures})")
                    else:
                        graded_span.add_judge_analysis(judge_name, analysis)
                        successful_analyses += 1
                        print(f"[DEBUG Judge Calls] ✓ Judge '{judge_name}' succeeded for span {span_id}")
            
            # Validate and add span if it meets requirements
            if successful_analyses >= ensemble_config.min_judges_required:
                if ensemble_config.require_consensus:
                    if self._check_consensus(graded_span, ensemble_config.consensus_threshold):
                        graded_spans.add_graded_span(span_id, graded_span)
                    else:
                        print(f"[DEBUG Judge Calls] Span {span_id} NOT added (consensus failed)")
                else:
                    graded_spans.add_graded_span(span_id, graded_span)
            elif not error_config.continue_on_span_errors:
                raise RuntimeError(f"Insufficient judge analyses: {successful_analyses} < {ensemble_config.min_judges_required}")
            else:
                print(f"[WARNING Judge Calls] Span {span_id} NOT added: insufficient analyses "
                      f"({successful_analyses} < {ensemble_config.min_judges_required})")
    
    def _grade_spans_sequential(self, llm_record: LLMRecord, span_list: List[tuple[str, Any]],
                                aspect_judges: Dict[str, BaseJudge], error_type: str,
                                graded_spans: GradedSpans, ensemble_config, error_config) -> None:
        """Grade spans sequentially (fallback when parallelization is disabled)"""
        import random as rng
        
        print(f"[DEBUG Judge Calls] Using SEQUENTIAL processing for {error_type} spans")
        
        for span_id, span in span_list:
            graded_span = GradedSpan(
                start=span.start,
                end=span.end,
                type=span.type,
                subtype=span.subtype,
                explanation=span.explanation
            )
            
            judge_failures = 0
            successful_analyses = 0
            
            # Get analyses from all judges for this aspect (sequential)
            for judge_name, judge in aspect_judges.items():
                print(f"[DEBUG Judge Calls] Calling judge '{judge_name}' for span {span_id} (type={error_type})")
                try:
                    judge_seed = rng.randint(1, 2**31 - 1)
                    analysis = judge.analyze_span(llm_record, span, seed=judge_seed)
                    
                    print(f"[DEBUG Judge Calls] ✓ Judge '{judge_name}' succeeded for span {span_id}: "
                          f"severity_score={analysis.severity_score:.3f}, "
                          f"confidence={analysis.confidence:.3f}, "
                          f"severity_bucket={analysis.severity_bucket.value}")
                    
                    graded_span.add_judge_analysis(judge_name, analysis)
                    successful_analyses += 1
                except Exception as e:
                    judge_failures += 1
                    print(f"[ERROR Judge Calls] ✗ Judge '{judge_name}' FAILED for span {span_id}: {str(e)}")
                    import traceback
                    print(f"[ERROR Judge Calls]   Traceback: {traceback.format_exc()}")
                    
                    if error_config.fail_fast and judge_failures > error_config.max_judge_failures:
                        if error_config.continue_on_span_errors:
                            break
                        else:
                            raise RuntimeError(f"Too many judge failures ({judge_failures})")
                    continue
            
            # Validate and add span if it meets requirements
            if successful_analyses >= ensemble_config.min_judges_required:
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
    
    def _validate_and_filter_spans(self, span_list: List[tuple[str, Any]], graded_spans: GradedSpans,
                                   ensemble_config, error_config) -> None:
        """Validate and filter spans based on ensemble requirements"""
        for span_id, span in span_list:
            if span_id not in graded_spans.spans:
                continue
            
            graded_span = graded_spans.spans[span_id]
            successful_analyses = len(graded_span.analysis)
            
            if successful_analyses >= ensemble_config.min_judges_required:
                if ensemble_config.require_consensus:
                    if self._check_consensus(graded_span, ensemble_config.consensus_threshold):
                        print(f"[DEBUG Judge Calls] Span {span_id} added to graded_spans (consensus passed)")
                    else:
                        graded_spans.spans.pop(span_id)
                        print(f"[DEBUG Judge Calls] Span {span_id} NOT added (consensus failed)")
                else:
                    print(f"[DEBUG Judge Calls] Span {span_id} added to graded_spans")
            elif not error_config.continue_on_span_errors:
                graded_spans.spans.pop(span_id)
                raise RuntimeError(f"Insufficient judge analyses: {successful_analyses} < {ensemble_config.min_judges_required}")
            else:
                graded_spans.spans.pop(span_id)
                print(f"[WARNING Judge Calls] Span {span_id} NOT added: insufficient analyses "
                      f"({successful_analyses} < {ensemble_config.min_judges_required})")
    
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
