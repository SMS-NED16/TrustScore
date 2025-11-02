"""
Dual Logger Utility

Outputs all console output to both console and a log file for debugging.
"""

import sys
from typing import TextIO
from datetime import datetime
import os


class DualLogger:
    """
    Logger that writes to both console (stdout) and a log file.
    """
    
    def __init__(self, log_file_path: str, mode: str = 'w'):
        """
        Initialize dual logger.
        
        Args:
            log_file_path: Path to log file
            mode: File mode ('w' for overwrite, 'a' for append)
        """
        self.log_file_path = log_file_path
        self.original_stdout = sys.stdout
        self.log_file: TextIO = None
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        
        # Open log file
        self.log_file = open(log_file_path, mode, encoding='utf-8')
        
        # Write header
        self.log_file.write(f"\n{'='*80}\n")
        self.log_file.write(f"TrustScore Specificity Analysis - Log File\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write(f"{'='*80}\n\n")
        self.log_file.flush()
    
    def write(self, message: str):
        """Write to both console and file."""
        if message:  # Only write non-empty messages
            # Write to console
            self.original_stdout.write(message)
            self.original_stdout.flush()
            
            # Write to file
            self.log_file.write(message)
            self.log_file.flush()
    
    def flush(self):
        """Flush both outputs."""
        self.original_stdout.flush()
        if self.log_file:
            self.log_file.flush()
    
    def close(self):
        """Close log file and restore stdout."""
        if self.log_file:
            self.log_file.write(f"\n{'='*80}\n")
            self.log_file.write(f"Ended: {datetime.now().isoformat()}\n")
            self.log_file.write(f"{'='*80}\n")
            self.log_file.close()
            self.log_file = None
        
        # Restore original stdout
        sys.stdout = self.original_stdout
    
    def __enter__(self):
        """Context manager entry."""
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def initialize_logging(output_dir: str, log_filename: str = "execution.log") -> DualLogger:
    """
    Initialize dual logging for the specificity analysis.
    
    Args:
        output_dir: Directory where log file should be saved
        log_filename: Name of log file
        
    Returns:
        DualLogger instance
    """
    log_path = os.path.join(output_dir, log_filename)
    logger = DualLogger(log_path)
    sys.stdout = logger
    return logger


def cleanup_logging(logger: DualLogger):
    """
    Clean up logging and restore stdout.
    
    Args:
        logger: DualLogger instance to clean up
    """
    if logger:
        logger.close()

