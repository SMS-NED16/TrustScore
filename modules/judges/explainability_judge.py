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


class MockExplainabilityJudge(ExplainabilityJudge):
    """Mock implementation of ExplainabilityJudge for testing."""
    
    def __init__(self, config: JudgeConfig, trust_score_config: TrustScoreConfig, api_key: Optional[str] = None) -> None:
        # Don't call super().__init__ to avoid initializing LLM provider
        self.config: JudgeConfig = config
        self.trust_score_config: TrustScoreConfig = trust_score_config
        self.system_prompt: str = EXPLAINABILITY_JUDGE_PROMPT
    
    def analyze_span(self, llm_record: LLMRecord, span: SpanTag, seed: Optional[int] = None) -> 'JudgeAnalysis':
        """Return mock analysis without calling LLM."""
        from models.llm_record import JudgeAnalysis, JudgeIndicators, JudgeWeights, SeverityBucket
        
        # Generate varied mock scores based on judge name for ensemble testing
        judge_num = self._extract_judge_number(self.config.name)
        base_score = 0.8 + (judge_num * 0.1)  # Explainability errors typically less severe
        base_confidence = 0.70 + (judge_num * 0.05)
        
        # Adjust based on subtype
        subtype_multiplier = self._get_subtype_severity_multiplier(span.subtype)
        severity_score = base_score * subtype_multiplier
        
        # Determine severity bucket
        if severity_score <= 0.5:
            bucket = SeverityBucket.MINOR
        elif severity_score <= 1.5:
            bucket = SeverityBucket.MAJOR
        else:
            bucket = SeverityBucket.CRITICAL
        
        return JudgeAnalysis(
            indicators=JudgeIndicators(
                centrality=0.5 + (judge_num * 0.05),
                domain_sensitivity=0.4 + (judge_num * 0.03),
                harm_potential=0.3 + (judge_num * 0.02),  # Lower harm for explainability
                instruction_criticality=0.7 + (judge_num * 0.04)  # Higher instruction criticality
            ),
            weights=JudgeWeights(
                centrality=1.0,
                domain_sensitivity=1.0,
                harm_potential=1.0,
                instruction_criticality=1.0
            ),
            confidence=base_confidence,
            severity_score=severity_score,
            severity_bucket=bucket
        )
    
    def _extract_judge_number(self, judge_name: str) -> int:
        """Extract judge number from name."""
        import re
        match = re.search(r'(\d+)$', judge_name)
        return int(match.group(1)) if match else 1
    
    def _get_subtype_severity_multiplier(self, subtype: str) -> float:
        """Get severity multiplier based on error subtype."""
        subtype_weights: Dict[str, float] = {
            "unclear_explanation": 0.8,
            "missing_context": 1.0,
            "overly_complex": 0.6,
            "assumption_not_stated": 0.9
        }
        return subtype_weights.get(subtype, 1.0)