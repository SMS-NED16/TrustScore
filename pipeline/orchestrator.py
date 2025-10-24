"""
TrustScore Pipeline - Main Orchestrator

This module coordinates the entire TrustScore pipeline, orchestrating
the span tagger, judges, and aggregator components.
"""

import asyncio
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
               generated_on: Optional[datetime] = None) -> AggregatedOutput:
        """
        Process a single LLM prompt/response pair through the pipeline.
        
        Args:
            prompt: Input prompt to the LLM
            response: LLM's response
            model: Model name/identifier
            generated_on: When the response was generated
            
        Returns:
            AggregatedOutput: Final TrustScore analysis
        """
        # Step 1: Create LLMRecord
        llm_record: LLMRecord = self._create_llm_record(prompt, response, model, generated_on)
        
        # Step 2: Tag spans
        spans_tags: SpansLevelTags = self.span_tagger.tag_spans(llm_record)
        
        # Step 3: Grade spans with judges
        graded_spans: GradedSpans = self._grade_spans(llm_record, spans_tags)
        
        # Step 4: Aggregate scores
        aggregated_output: AggregatedOutput = self.aggregator.aggregate(llm_record, graded_spans)
        
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
        
        for input_data in inputs:
            try:
                result: AggregatedOutput = self.process(
                    prompt=input_data['prompt'],
                    response=input_data['response'],
                    model=input_data.get('model', 'unknown'),
                    generated_on=input_data.get('generated_on')
                )
                results.append(result)
            except Exception as e:
                # Create empty result for failed processing
                print(f"Error processing input: {str(e)}")
                # You might want to create a default/empty AggregatedOutput here
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
            x=prompt,
            y=response,
            M=metadata
        )
    
    def _grade_spans(self, llm_record: LLMRecord, spans_tags: SpansLevelTags) -> GradedSpans:
        """
        Grade spans using ensemble of judges for each aspect.
        
        Args:
            llm_record: The original LLM input/output pair
            spans_tags: Collection of identified error spans
            
        Returns:
            GradedSpans: Collection of graded spans with ensemble analyses
        """
        graded_spans: GradedSpans = GradedSpans()
        
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
            
            # Get analyses from all judges for this aspect
            for judge_name, judge in aspect_judges.items():
                try:
                    analysis = judge.analyze_span(llm_record, span)
                    graded_span.add_judge_analysis(judge_name, analysis)
                except Exception as e:
                    # Log error but continue with other judges
                    print(f"Error from judge {judge_name}: {str(e)}")
                    continue
            
            # Only add span if we got at least one analysis
            if graded_span.analysis:
                graded_spans.add_graded_span(span_id, graded_span)
        
        return graded_spans
    
    
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
                "confidence_level": self.config.confidence_level
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
