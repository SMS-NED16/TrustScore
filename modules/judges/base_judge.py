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
        
        content = content.strip()
        if not content:
            raise ValueError("Empty response from LLM")
        
        # Try markdown code block first (most reliable)
        json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # Fallback: Parse structured markdown format
        try:
            return self._parse_markdown_response(content)
        except Exception:
            pass
        
        # If all parsing fails, raise error
        content_preview = content[:500] if len(content) > 500 else content
        raise ValueError(
            f"Failed to parse judge response as JSON\n"
            f"Response content (first 500 chars): {content_preview}"
        )
    
    def _parse_markdown_response(self, content: str) -> Dict[str, Any]:
        """Parse markdown-formatted judge response into JSON structure."""
        import re
        
        result = {
            "indicators": {},
            "weights": {},
            "confidence": 0.5,
            "severity_score": 0.0,
            "severity_bucket": "major"
        }
        
        # Pattern 1: **centrality:** 0.2 (description) or **centrality:** 0.2
        pattern1 = r'\*\*(centrality|domain_sensitivity|harm_potential|instruction_criticality)\*\*:\s*([0-9.]+)(?:\s*\([^)]*\))?'
        for match in re.finditer(pattern1, content, re.IGNORECASE):
            key = match.group(1).lower().replace(' ', '_')
            value = float(match.group(2))
            result["indicators"][key] = value
        
        # Pattern 2: * **centrality:** 0.2 or - centrality: 0.2 (description)
        pattern2 = r'[-*]\s*(?:\*\*)?(centrality|domain_sensitivity|harm_potential|instruction_criticality)(?:\*\*)?:\s*([0-9.]+)(?:\s*\([^)]*\))?'
        for match in re.finditer(pattern2, content, re.IGNORECASE):
            key = match.group(1).lower().replace(' ', '_')
            value = float(match.group(2))
            if key not in result["indicators"]:
                result["indicators"][key] = value
        
        # Pattern 3: centrality: 0.2 (without bullet, at start of line)
        pattern3 = r'^(?:[-*]\s*)?(centrality|domain_sensitivity|harm_potential|instruction_criticality)[:\s]+([0-9.]+)'
        for match in re.finditer(pattern3, content, re.IGNORECASE | re.MULTILINE):
            key = match.group(1).lower().replace(' ', '_')
            value = float(match.group(2))
            if key not in result["indicators"]:
                result["indicators"][key] = value
        
        # Extract confidence (with or without markdown)
        confidence_patterns = [
            r'\*\*confidence\*\*:\s*([0-9.]+)',
            r'[-*]\s*confidence[:\s]+([0-9.]+)',
            r'confidence[:\s]+([0-9.]+)'
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result["confidence"] = float(match.group(1))
                break
        
        # Extract severity_score
        severity_patterns = [
            r'\*\*severity\s*score\*\*:\s*([0-9.]+)',
            r'[-*]\s*severity\s*score[:\s]+([0-9.]+)',
            r'severity\s*score[:\s]+([0-9.]+)'
        ]
        for pattern in severity_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                result["severity_score"] = float(match.group(1))
                break
        
        # Extract severity_bucket
        bucket_patterns = [
            r'\*\*severity\s*bucket\*\*:\s*["\']?(\w+)["\']?',
            r'[-*]\s*severity\s*bucket[:\s]+["\']?(\w+)["\']?',
            r'severity\s*bucket[:\s]+["\']?(\w+)["\']?',
            r'severity\s*classification[:\s]+["\']?(\w+)["\']?'
        ]
        for pattern in bucket_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                bucket = match.group(1).lower()
                result["severity_bucket"] = bucket
                break
        
        # Set default weights (all 1.0 if not specified)
        if not result["weights"]:
            result["weights"] = {
                "centrality": 1.0,
                "domain_sensitivity": 1.0,
                "harm_potential": 1.0,
                "instruction_criticality": 1.0
            }
        
        # Ensure all required indicators are present (set defaults if missing)
        required_indicators = ["centrality", "domain_sensitivity", "harm_potential", "instruction_criticality"]
        for indicator in required_indicators:
            if indicator not in result["indicators"]:
                result["indicators"][indicator] = 0.5  # Default value
        
        return result
    
    def _create_judge_analysis(self, analysis_data: Dict[str, Any]) -> JudgeAnalysis:
        """Create JudgeAnalysis object from parsed data."""
        try:
            indicators: JudgeIndicators = JudgeIndicators(**analysis_data["indicators"])
            weights: JudgeWeights = JudgeWeights(**analysis_data["weights"])
            confidence: float = float(analysis_data["confidence"])
            severity_score: float = float(analysis_data["severity_score"])
            
            # Normalize severity bucket - map invalid values to valid ones
            severity_bucket_str = str(analysis_data.get("severity_bucket", "")).lower().strip()
            severity_mapping = {
                "moderate": "major",
                "medium": "major",
                "low": "minor",
                "high": "major",
                "severe": "critical"
            }
            severity_bucket_str = severity_mapping.get(severity_bucket_str, severity_bucket_str)
            
            try:
                severity_bucket: SeverityBucket = SeverityBucket(severity_bucket_str)
            except ValueError:
                # Default to MAJOR if invalid
                print(f"[WARNING] Invalid severity bucket '{analysis_data.get('severity_bucket')}', defaulting to MAJOR")
                severity_bucket: SeverityBucket = SeverityBucket.MAJOR
            
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
