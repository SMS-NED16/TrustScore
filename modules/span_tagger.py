"""
TrustScore Pipeline - LLM Span Tagger Module

This module implements the LLM-based span tagger that identifies
trustworthiness, bias, and explainability errors in LLM responses.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from models.llm_record import LLMRecord, SpansLevelTags, SpanTag, ErrorType
from config.settings import SpanTaggerConfig
from prompts.system_prompts import SPAN_TAGGER_PROMPT


class SpanTagger:
    """
    LLM-based span tagger for identifying errors in LLM responses.
    
    This module uses a fine-tuned LLM to identify span-level errors
    and categorize them as Trustworthiness (T), Bias (B), or Explainability (E).
    """
    
    def __init__(self, config: SpanTaggerConfig, api_key: Optional[str] = None) -> None:
        self.config: SpanTaggerConfig = config
        self.client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None
        
        # System prompt for the span tagger
        self.system_prompt: str = SPAN_TAGGER_PROMPT

    def tag_spans(self, llm_record: LLMRecord) -> SpansLevelTags:
        """
        Identify and tag span-level errors in the LLM response.
        
        Args:
            llm_record: The LLM input/output pair to analyze
            
        Returns:
            SpansLevelTags: Collection of identified error spans
        """
        if not self.client:
            raise ValueError("OpenAI client not initialized. Please provide API key.")
        
        user_prompt: str = f"""Prompt: {llm_record.task_prompt}

Response: {llm_record.llm_response}

Please analyze this response for errors and return the JSON format specified in the system prompt."""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            content: str = response.choices[0].message.content
            spans_data: Dict[str, Any] = self._parse_response(content)
            
            return self._create_spans_level_tags(spans_data, llm_record.llm_response)
            
        except Exception as e:
            raise RuntimeError(f"Error in span tagging: {str(e)}")
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response to extract span data."""
        try:
            # Try to extract JSON from the response
            json_match: Optional[re.Match[str]] = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # Fallback: try to parse the entire content
                return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {str(e)}")
    
    def _create_spans_level_tags(self, spans_data: Dict[str, Any], response_text: str) -> SpansLevelTags:
        """Create SpansLevelTags object from parsed data."""
        spans_level_tags: SpansLevelTags = SpansLevelTags()
        
        if "spans" not in spans_data:
            return spans_level_tags
        
        for span_id, span_info in spans_data["spans"].items():
            try:
                # Validate span data
                start: int = int(span_info["start"])
                end: int = int(span_info["end"])
                error_type: str = span_info["type"]
                subtype: str = span_info["subtype"]
                explanation: str = span_info["explanation"]
                
                # Validate character positions
                if start < 0 or end > len(response_text) or start >= end:
                    continue
                
                # Validate error type
                try:
                    error_type_enum: ErrorType = ErrorType(error_type)
                except ValueError:
                    continue
                
                # Create span tag
                span_tag: SpanTag = SpanTag(
                    start=start,
                    end=end,
                    type=error_type_enum,
                    subtype=subtype,
                    explanation=explanation
                )
                
                spans_level_tags.add_span(span_id, span_tag)
                
            except (KeyError, ValueError, TypeError) as e:
                # Skip invalid spans
                continue
        
        return spans_level_tags
    
    def batch_tag_spans(self, llm_records: List[LLMRecord]) -> List[SpansLevelTags]:
        """
        Tag spans for multiple LLM records.
        
        Args:
            llm_records: List of LLM input/output pairs
            
        Returns:
            List of SpansLevelTags for each record
        """
        results: List[SpansLevelTags] = []
        
        for i in range(0, len(llm_records), self.config.batch_size):
            batch: List[LLMRecord] = llm_records[i:i + self.config.batch_size]
            
            for record in batch:
                try:
                    spans: SpansLevelTags = self.tag_spans(record)
                    results.append(spans)
                except Exception as e:
                    # Create empty spans for failed records
                    results.append(SpansLevelTags())
        
        return results


class MockSpanTagger(SpanTagger):
    """
    Mock implementation of SpanTagger for testing purposes.
    """
    
    def __init__(self, config: SpanTaggerConfig) -> None:
        self.config: SpanTaggerConfig = config
        self.client: Optional[OpenAI] = None  # No OpenAI client needed
    
    def tag_spans(self, llm_record: LLMRecord) -> SpansLevelTags:
        """Mock implementation that returns predefined spans for testing."""
        # Example mock spans based on the provided example
        mock_spans_data: Dict[str, Any] = {
            "spans": {
                "0": {
                    "start": 0,
                    "end": 6,
                    "type": "T",
                    "subtype": "spelling",
                    "explanation": "Georgia is misspelled."
                },
                "1": {
                    "start": 45,
                    "end": 54,
                    "type": "T",
                    "subtype": "factual_error",
                    "explanation": "Georgia Tech is not in Savannah."
                }
            }
        }
        
        return self._create_spans_level_tags(mock_spans_data, llm_record.llm_response)
