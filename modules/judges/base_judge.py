"""
TrustScore Pipeline - Base Judge Module

This module provides the base interface and common functionality
for LLM judges that score error severity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from models.llm_record import LLMRecord, SpanTag, JudgeAnalysis, JudgeIndicators, JudgeWeights, SeverityBucket
from config.settings import JudgeConfig, TrustScoreConfig
from prompts.system_prompts import BASE_JUDGE_PROMPT
from modules.llm_providers.factory import LLMProviderFactory


class BaseJudge(ABC):
    """
    Abstract base class for LLM judges that score error severity.
    
    Each judge analyzes individual spans and provides:
    - Indicator scores (centrality, domain_sensitivity, harm_potential, instruction_criticality)
    - Confidence level
    - Severity score and bucket
    """
    
    def __init__(self, config: JudgeConfig, trust_score_config: TrustScoreConfig, api_key: Optional[str] = None) -> None:
        self.config: JudgeConfig = config
        self.trust_score_config: TrustScoreConfig = trust_score_config
        if api_key:
            self.config.api_key = api_key
        
        # Create LLM provider
        self.llm_provider = LLMProviderFactory.create_judge_provider(config)
        
        # Base system prompt for judges
        self.system_prompt: str = BASE_JUDGE_PROMPT

    @abstractmethod
    def analyze_span(self, llm_record: LLMRecord, span: SpanTag) -> JudgeAnalysis:
        """
        Analyze a specific span and return judge analysis.
        
        Args:
            llm_record: The original LLM input/output pair
            span: The error span to analyze
            
        Returns:
            JudgeAnalysis: Detailed analysis from this judge
        """
        pass
    
    def _call_llm(self, user_prompt: str) -> Dict[str, Any]:
        """Make API call to the LLM using configured provider."""
        if not self.llm_provider.is_available():
            raise ValueError(f"LLM provider {self.config.provider} not available")
        
        try:
            messages = self.llm_provider.format_messages(self.system_prompt, user_prompt)
            content: str = self.llm_provider.generate(messages)
            return self._parse_response(content)
            
        except Exception as e:
            raise RuntimeError(f"Error in judge analysis: {str(e)}")
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response to extract analysis data."""
        import json
        import re
        
        try:
            # Try to extract JSON from the response
            json_match: Optional[re.Match[str]] = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire content
                return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse judge response as JSON: {str(e)}")
    
    def _create_judge_analysis(self, analysis_data: Dict[str, Any]) -> JudgeAnalysis:
        """Create JudgeAnalysis object from parsed data."""
        try:
            indicators: JudgeIndicators = JudgeIndicators(**analysis_data["indicators"])
            weights: JudgeWeights = JudgeWeights(**analysis_data["weights"])
            confidence: float = float(analysis_data["confidence"])
            severity_score: float = float(analysis_data["severity_score"])
            severity_bucket: SeverityBucket = SeverityBucket(analysis_data["severity_bucket"])
            
            return JudgeAnalysis(
                indicators=indicators,
                weights=weights,
                confidence=confidence,
                severity_score=severity_score,
                severity_bucket=severity_bucket
            )
            
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid judge analysis data: {str(e)}")
    
    def _get_severity_bucket(self, score: float) -> SeverityBucket:
        """Determine severity bucket based on score and thresholds."""
        bucket_name: str = self.trust_score_config.get_severity_bucket(score)
        return SeverityBucket(bucket_name)
