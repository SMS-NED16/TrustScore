"""
TrustScore Pipeline - LLM Span Tagger Module

This module implements the LLM-based span tagger that identifies
trustworthiness, bias, and explainability errors in LLM responses.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from models.llm_record import LLMRecord, SpansLevelTags, SpanTag, ErrorType
from config.settings import SpanTaggerConfig
from prompts.system_prompts import SPAN_TAGGER_PROMPT
from modules.llm_providers.factory import LLMProviderFactory


class SpanTagger:
    """
    LLM-based span tagger for identifying errors in LLM responses.
    
    This module uses a fine-tuned LLM to identify span-level errors
    and categorize them as Trustworthiness (T), Bias (B), or Explainability (E).
    """
    
    def __init__(self, config: SpanTaggerConfig, api_key: Optional[str] = None) -> None:
        self.config: SpanTaggerConfig = config
        if api_key:
            self.config.api_key = api_key
        
        # Create LLM provider
        self.llm_provider = LLMProviderFactory.create_span_tagger_provider(config)
        
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
        if not self.llm_provider.is_available():
            raise ValueError(f"LLM provider {self.config.provider} not available")
        
        user_prompt: str = f"""Prompt: {llm_record.task_prompt}

Response: {llm_record.llm_response}

Please analyze this response for MEANINGFUL errors that impact the response's quality, accuracy, fairness, or clarity in the context of the given instruction.

Key considerations:
- Focus on errors that MATTER: factual errors, hallucinations, inconsistencies, bias, missing context, unclear explanations
- DO NOT flag minor formatting issues (e.g., lowercase proper nouns in lowercased text) unless they obscure meaning
- Consider the instruction: Does the error prevent the response from fulfilling its purpose?
- For summarization tasks: Prioritize factual accuracy, completeness, and clarity over formatting

Return the JSON format specified in the system prompt.

Remember: Each error span MUST include a clear, detailed explanation explaining what's wrong, why it's problematic, and how it impacts the response quality relative to the instruction."""

        try:
            messages = self.llm_provider.format_messages(self.system_prompt, user_prompt)
            content: str = self.llm_provider.generate(messages)
            
            # DEBUG: Log the raw LLM response
            print(f"[DEBUG Span Tagger] Raw LLM response (first 1000 chars):\n{content[:1000]}")
            print(f"[DEBUG Span Tagger] Full response length: {len(content)} chars")
            
            spans_data: Dict[str, Any] = self._parse_response(content)
            
            # DEBUG: Log the parsed spans data
            print(f"[DEBUG Span Tagger] Parsed spans_data: {spans_data}")
            if "spans" in spans_data:
                print(f"[DEBUG Span Tagger] Number of spans in parsed data: {len(spans_data['spans'])}")
                if len(spans_data['spans']) > 0:
                    print(f"[DEBUG Span Tagger] First span keys: {list(spans_data['spans'].keys())[:3]}")
            else:
                print(f"[DEBUG Span Tagger] WARNING: 'spans' key not found in parsed data!")
            
            result = self._create_spans_level_tags(spans_data, llm_record.llm_response)
            
            # DEBUG: Log final result
            print(f"[DEBUG Span Tagger] Final spans created: {len(result.spans)}")
            if len(result.spans) > 0:
                for span_id, span in list(result.spans.items())[:3]:
                    print(f"[DEBUG Span Tagger] Span {span_id}: type={span.type.value}, start={span.start}, end={span.end}")
            
            return result
            
        except Exception as e:
            print(f"[DEBUG Span Tagger] Error in span tagging: {str(e)}")
            print(f"[DEBUG Span Tagger] Exception type: {type(e).__name__}")
            raise RuntimeError(f"Error in span tagging: {str(e)}")
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse the LLM response to extract span data."""
        # Strip whitespace and check if empty
        content = content.strip()
        if not content:
            raise ValueError("Empty response from span tagger LLM")
        
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
                f"Failed to parse LLM response as JSON: {str(e)}\n"
                f"Response content (first 500 chars): {content_preview}"
            )
    
    def _create_spans_level_tags(self, spans_data: Dict[str, Any], response_text: str) -> SpansLevelTags:
        """Create SpansLevelTags object from parsed data with configurable validation."""
        spans_level_tags: SpansLevelTags = SpansLevelTags()
        
        # Handle both formats: {"spans": {...}} or just {...} (the spans dict directly)
        if "spans" in spans_data:
            # Format: {"spans": {"0": {...}, "1": {...}}}
            spans_dict = spans_data["spans"]
        elif isinstance(spans_data, dict) and all(isinstance(k, str) for k in spans_data.keys()):
            # Format: {"0": {...}, "1": {...}} - already the spans dict
            # Check if it looks like a spans dict (has span-like structure with start/end/type)
            first_value = next(iter(spans_data.values())) if spans_data else {}
            if isinstance(first_value, dict) and "start" in first_value and "end" in first_value:
                spans_dict = spans_data
            else:
                return spans_level_tags
        else:
            return spans_level_tags
        
        # Get span processing configuration
        span_config = self.config.span_processing if hasattr(self.config, 'span_processing') else None
        
        span_count = 0
        for span_id, span_info in spans_dict.items():
            try:
                # Check max spans limit
                if span_config and span_count >= span_config.max_spans_per_response:
                    break
                
                # Validate span data
                start: int = int(span_info["start"])
                end: int = int(span_info["end"])
                error_type: str = span_info["type"]
                subtype: str = span_info["subtype"]
                
                # Validate explanation exists and is meaningful
                if "explanation" not in span_info:
                    raise ValueError(f"Span {span_id} is missing required 'explanation' field")
                
                explanation: str = span_info["explanation"]
                
                # Validate explanation is not empty or whitespace-only
                if not explanation or not explanation.strip():
                    raise ValueError(f"Span {span_id} has empty or whitespace-only explanation")
                
                # Get minimum explanation length (default 5 characters - reduced for debugging)
                min_explanation_length = 5
                if span_config and hasattr(span_config, 'min_explanation_length'):
                    min_explanation_length = span_config.min_explanation_length
                
                # Validate explanation meets minimum length requirement
                if len(explanation.strip()) < min_explanation_length:
                    raise ValueError(
                        f"Span {span_id} explanation is too short (minimum {min_explanation_length} characters, "
                        f"got {len(explanation.strip())}). Please provide a more detailed explanation."
                    )
                
                # Apply configurable span validation
                if span_config:
                    # Check span length constraints
                    span_length = end - start
                    if span_length < span_config.min_span_length or span_length > span_config.max_span_length:
                        continue
                    
                    # Check character position bounds
                    if start < 0 or end > len(response_text) or start >= end:
                        if span_config.span_validation_strict:
                            continue
                        else:
                            # Adjust bounds if not strict
                            start = max(0, start)
                            end = min(len(response_text), end)
                            if start >= end:
                                continue
                else:
                    # Default validation
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
                span_count += 1
                
            except (KeyError, ValueError, TypeError) as e:
                # Skip invalid spans
                continue
        
        # Apply span merging if configured
        if span_config and span_config.merge_adjacent_spans:
            spans_level_tags = self._merge_adjacent_spans(spans_level_tags, response_text)
        
        return spans_level_tags
    
    def _merge_adjacent_spans(self, spans_level_tags: SpansLevelTags, response_text: str) -> SpansLevelTags:
        """Merge adjacent spans if they are close enough."""
        if not hasattr(self.config, 'span_processing'):
            return spans_level_tags
        
        span_config = self.config.span_processing
        merged_spans = SpansLevelTags()
        spans_list = list(spans_level_tags.spans.items())
        
        if not spans_list:
            return merged_spans
        
        # Sort spans by start position
        spans_list.sort(key=lambda x: x[1].start)
        
        current_span_id, current_span = spans_list[0]
        merged_spans.add_span(current_span_id, current_span)
        
        for span_id, span in spans_list[1:]:
            # Check if spans are close enough to merge
            gap = span.start - current_span.end
            if gap <= span_config.min_span_gap and span.type == current_span.type:
                # Merge spans
                current_span = SpanTag(
                    start=current_span.start,
                    end=span.end,
                    type=current_span.type,
                    subtype=current_span.subtype,
                    explanation=f"{current_span.explanation}; {span.explanation}"
                )
                # Update the last added span
                merged_spans.spans[current_span_id] = current_span
            else:
                # Add as separate span
                merged_spans.add_span(span_id, span)
                current_span_id = span_id
                current_span = span
        
        return merged_spans
    
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
        self.client: Optional[Any] = None  # No OpenAI client needed
    
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
