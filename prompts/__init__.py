"""
TrustScore Pipeline - Prompts Package

This package contains all system prompts used by the TrustScore pipeline.
"""

from .system_prompts import (
    SPAN_TAGGER_PROMPT,
    BASE_JUDGE_PROMPT,
    TRUSTWORTHINESS_JUDGE_PROMPT,
    BIAS_JUDGE_PROMPT,
    EXPLAINABILITY_JUDGE_PROMPT
)

__all__ = [
    "SPAN_TAGGER_PROMPT",
    "BASE_JUDGE_PROMPT", 
    "TRUSTWORTHINESS_JUDGE_PROMPT",
    "BIAS_JUDGE_PROMPT",
    "EXPLAINABILITY_JUDGE_PROMPT"
]
