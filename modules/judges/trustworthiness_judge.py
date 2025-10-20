"""
TrustScore Pipeline - Trustworthiness Judge

This module implements a specialized judge for evaluating
trustworthiness-related errors in LLM responses.
"""

from typing import Dict, Any, Optional
from models.llm_record import LLMRecord, SpanTag, JudgeAnalysis, ErrorType
from modules.judges.base_judge import BaseJudge
from config.settings import JudgeConfig, TrustScoreConfig
from prompts.system_prompts import TRUSTWORTHINESS_JUDGE_PROMPT


class TrustworthinessJudge(BaseJudge):
    """
    Specialized judge for evaluating trustworthiness errors.
    
    This judge focuses on factual accuracy, consistency, and reliability
    of information in LLM responses.
    """
    
    def __init__(self, config: JudgeConfig, trust_score_config: TrustScoreConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config, trust_score_config, api_key)
        
        # Override system prompt for trustworthiness focus
        self.system_prompt: str = TRUSTWORTHINESS_JUDGE_PROMPT

    def analyze_span(self, llm_record: LLMRecord, span: SpanTag) -> JudgeAnalysis:
        """
        Analyze a trustworthiness error span.
        
        Args:
            llm_record: The original LLM input/output pair
            span: The trustworthiness error span to analyze
            
        Returns:
            JudgeAnalysis: Detailed analysis from this judge
        """
        if span.type != ErrorType.TRUSTWORTHINESS:
            raise ValueError("TrustworthinessJudge can only analyze trustworthiness errors")
        
        user_prompt: str = f"""Prompt: {llm_record.task_prompt}

Response: {llm_record.llm_response}

Error Span: "{llm_record.llm_response[span.start:span.end]}"
Error Type: {span.subtype}
Error Explanation: {span.explanation}

Please analyze this trustworthiness error and provide detailed severity scoring. Consider the specific subtype and its implications for factual accuracy and reliability."""

        analysis_data: Dict[str, Any] = self._call_llm(user_prompt)
        return self._create_judge_analysis(analysis_data)
    
    def _get_subtype_severity_multiplier(self, subtype: str) -> float:
        """Get severity multiplier based on error subtype."""
        subtype_weights: Dict[str, float] = {
            "spelling": 0.3,
            "factual_error": 1.0,
            "hallucination": 1.5,
            "inconsistency": 0.8
        }
        return subtype_weights.get(subtype, 1.0)


