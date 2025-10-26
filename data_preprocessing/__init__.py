"""
TrustScore Pipeline - Data Preprocessing Module

This module provides tools for preprocessing datasets for fine-tuning
the span tagger and other LLM components.
"""

from .data_formats import (
    TrainingExample, 
    SpanAnnotation, 
    DatasetFormat,
    convert_to_training_format
)
from .preprocessor import DatasetPreprocessor
from .synthetic_generator import SyntheticDataGenerator
from .fine_tuning_utils import FineTuningDataManager

__all__ = [
    "TrainingExample",
    "SpanAnnotation", 
    "DatasetFormat",
    "convert_to_training_format",
    "DatasetPreprocessor",
    "SyntheticDataGenerator",
    "FineTuningDataManager"
]

