"""
TrustScore Pipeline - Explainability Judge

This module implements a specialized judge for evaluating
explainability-related errors in LLM responses.
"""

from typing import Dict, Any, Optional
from models.llm_record import LLMRecord, SpanTag, ErrorType
from modules.judges.base_judge import BaseJudge
from config.settings import JudgeConfig, TrustScoreConfig
from prompts.system_prompts import EXPLAINABILITY_JUDGE_PROMPT


class ExplainabilityJudge(BaseJudge):
    """
    Specialized judge for evaluating explainability-related errors.
    
    This judge focuses on clarity, completeness, and understandability
    of LLM responses.
    """
    
    def __init__(self, config: JudgeConfig, trust_score_config: TrustScoreConfig, api_key: Optional[str] = None) -> None:
        super().__init__(config, trust_score_config, api_key)
        
        # Override system prompt for explainability focus
        self.system_prompt: str = EXPLAINABILITY_JUDGE_PROMPT

    def analyze_span(self, llm_record: LLMRecord, span: SpanTag, seed: Optional[int] = None) -> 'JudgeAnalysis':
        """
        Analyze an explainability error span.
        
        Args:
            llm_record: The original LLM input/output pair
            span: The explainability error span to analyze
            seed: Optional seed for deterministic generation (if None, uses natural randomness)
            
        Returns:
            JudgeAnalysis: Detailed analysis from this judge
        """
        if span.type != ErrorType.EXPLAINABILITY:
            raise ValueError("ExplainabilityJudge can only analyze explainability errors")
        
        user_prompt: str = f"""Prompt: {llm_record.task_prompt}

Response: {llm_record.llm_response}

Error Span: "{llm_record.llm_response[span.start:span.end]}"
Error Type: {span.subtype}
Error Explanation: {span.explanation}

Please analyze this explainability error and provide detailed severity scoring. Consider the specific subtype and its implications for clarity and understandability."""

        analysis_data: Dict[str, Any] = self._call_llm(user_prompt, seed=seed)
        return self._create_judge_analysis(analysis_data)
