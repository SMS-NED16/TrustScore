"""
TrustScore Pipeline - Base Judge Module

This module provides the base interface and common functionality
for LLM judges that score error severity.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
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
    
    def batch_analyze_spans(self, span_records: List[tuple[LLMRecord, SpanTag]]) -> List[JudgeAnalysis]:
        """
        Analyze multiple spans using batch processing for efficiency.
        
        Args:
            span_records: List of (LLMRecord, SpanTag) tuples to analyze
            
        Returns:
            List of JudgeAnalysis for each span
        """
        if not self.llm_provider.is_available():
            raise ValueError(f"LLM provider {self.config.provider} not available")
        
        # Create user prompts for all spans
        user_prompts = []
        for llm_record, span in span_records:
            user_prompt = self._create_user_prompt(llm_record, span)
            user_prompts.append(user_prompt)
        
        # Batch generate using the provider
        messages_list = [
            self.llm_provider.format_messages(self.system_prompt, prompt)
            for prompt in user_prompts
        ]
        
        try:
            # Use batch_generate if available
            if hasattr(self.llm_provider, 'batch_generate'):
                contents = self.llm_provider.batch_generate(messages_list)
            else:
                # Fallback to individual calls
                contents = []
                for messages in messages_list:
                    content = self.llm_provider.generate(messages)
                    contents.append(content)
            
            # Parse all responses
            analyses = []
            for content in contents:
                analysis_data = self._parse_response(content)
                analysis = self._create_judge_analysis(analysis_data)
                analyses.append(analysis)
            
            return analyses
            
        except Exception as e:
            raise RuntimeError(f"Error in batch judge analysis: {str(e)}")
    
    def _create_user_prompt(self, llm_record: LLMRecord, span: SpanTag) -> str:
        """Create user prompt for a specific span analysis"""
        return f"""Prompt: {llm_record.task_prompt}

Response: {llm_record.llm_response}

Error Span:
- Position: {span.start}-{span.end}
- Type: {span.type.value}
- Subtype: {span.subtype}
- Explanation: {span.explanation}

Please analyze this error span for severity."""
    
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
        
        # Strip whitespace and check if empty
        content = content.strip()
        if not content:
            raise ValueError("Empty response from LLM")
        
        try:
            # First, try to find JSON between markdown code blocks
            json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_block:
                return json.loads(json_block.group(1))
            
            # Try to extract JSON from the response (improved regex for nested objects)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Try a more greedy match
                    json_match_greedy = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match_greedy:
                        return json.loads(json_match_greedy.group())
            
            # Fallback: try to parse the entire content
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            # Log the actual content for debugging
            content_preview = content[:500] if len(content) > 500 else content
            raise ValueError(
                f"Failed to parse judge response as JSON: {str(e)}\n"
                f"Response content (first 500 chars): {content_preview}"
            )
    
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
