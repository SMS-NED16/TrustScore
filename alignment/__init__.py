"""
Alignment module: learn TrustScore configurations that maximize
correlation with human quality scores on task-specific datasets.
"""

from alignment.tasks.base import AlignmentTask
from alignment.run_alignment import run

__all__ = ["AlignmentTask", "run"]
