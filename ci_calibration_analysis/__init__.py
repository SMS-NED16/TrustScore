"""
CI Calibration Analysis Module

This module contains tools and scripts for analyzing the calibration
of confidence intervals across the TrustScore pipeline.
"""

# Lazy imports to avoid circular dependencies
# These can be imported directly when needed:
# from ci_calibration_analysis.run_ci_calibration import run_ci_calibration_analysis
# from ci_calibration_analysis.generate_ci_calibration_report import generate_ci_calibration_report
# from ci_calibration_analysis.ci_calibration_utils import load_calibration_results, analyze_ci_level, make_json_serializable

__all__ = [
    "run_ci_calibration_analysis",
    "generate_ci_calibration_report",
    "load_calibration_results",
    "analyze_ci_level",
    "make_json_serializable",
]

