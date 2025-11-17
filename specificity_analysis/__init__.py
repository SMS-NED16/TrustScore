"""
Specificity Analysis Module

This module contains scripts for evaluating TrustScore's specificity
by injecting different error types (Trustworthiness, Bias, Explainability)
and measuring the corresponding score drops.
"""

from .load_dataset import load_and_sample_dataset, save_samples
from .error_injector import ErrorInjector
from .baseline_inference import run_baseline_inference
from .perturbed_inference import run_perturbed_inference
from .score_comparison import compare_scores, generate_report, load_results

__all__ = [
    'load_and_sample_dataset',
    'save_samples',
    'ErrorInjector',
    'run_baseline_inference',
    'run_perturbed_inference',
    'compare_scores',
    'generate_report',
    'load_results'
]

# Step-by-step runner is available via import
# Commented out to prevent automatic execution of module-level code
# try:
#     from . import run_step_by_step
# except ImportError:
#     pass
