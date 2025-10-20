"""
TrustScore Pipeline - Error Handling and Validation

This module provides comprehensive error handling, validation,
and logging functionality for the TrustScore pipeline.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum

from models.llm_record import LLMRecord, ErrorType
from config.settings import TrustScoreConfig


class ErrorSeverity(str, Enum):
    """Error severity levels for logging"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TrustScoreError(Exception):
    """Base exception for TrustScore pipeline errors"""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 component: Optional[str] = None, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message: str = message
        self.severity: ErrorSeverity = severity
        self.component: Optional[str] = component
        self.details: Dict[str, Any] = details or {}
        self.timestamp: datetime = datetime.now()


class ValidationError(TrustScoreError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs) -> None:
        super().__init__(message, ErrorSeverity.MEDIUM, "validation", **kwargs)
        self.field: Optional[str] = field


class APIError(TrustScoreError):
    """Raised when API calls fail"""
    
    def __init__(self, message: str, api_name: Optional[str] = None, **kwargs) -> None:
        super().__init__(message, ErrorSeverity.HIGH, "api", **kwargs)
        self.api_name: Optional[str] = api_name


class ProcessingError(TrustScoreError):
    """Raised when processing fails"""
    
    def __init__(self, message: str, stage: Optional[str] = None, **kwargs) -> None:
        super().__init__(message, ErrorSeverity.HIGH, "processing", **kwargs)
        self.stage: Optional[str] = stage


class TrustScoreValidator:
    """
    Comprehensive validator for TrustScore pipeline inputs and outputs.
    """
    
    def __init__(self, config: TrustScoreConfig) -> None:
        self.config: TrustScoreConfig = config
        self.logger: logging.Logger = logging.getLogger(__name__)
    
    def validate_llm_record(self, llm_record: LLMRecord) -> List[ValidationError]:
        """
        Validate an LLMRecord object.
        
        Args:
            llm_record: The record to validate
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        # Validate prompt
        if not llm_record.x or not isinstance(llm_record.x, str):
            errors.append(ValidationError("Prompt must be a non-empty string", "x"))
        elif len(llm_record.x.strip()) == 0:
            errors.append(ValidationError("Prompt cannot be empty or whitespace only", "x"))
        
        # Validate response
        if not llm_record.y or not isinstance(llm_record.y, str):
            errors.append(ValidationError("Response must be a non-empty string", "y"))
        elif len(llm_record.y.strip()) == 0:
            errors.append(ValidationError("Response cannot be empty or whitespace only", "y"))
        
        # Validate metadata
        if not llm_record.M or not isinstance(llm_record.M, dict):
            errors.append(ValidationError("Metadata must be provided", "M"))
        else:
            if not llm_record.M.get("model"):
                errors.append(ValidationError("Model name must be provided in metadata", "M.model"))
            
            if not llm_record.M.get("generated_on"):
                errors.append(ValidationError("Generation timestamp must be provided", "M.generated_on"))
        
        return errors
    
    def validate_span_positions(self, text: str, start: int, end: int) -> List[ValidationError]:
        """
        Validate span positions within text.
        
        Args:
            text: The text containing the span
            start: Start position
            end: End position
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        if not isinstance(start, int) or start < 0:
            errors.append(ValidationError("Start position must be a non-negative integer"))
        
        if not isinstance(end, int) or end < 0:
            errors.append(ValidationError("End position must be a non-negative integer"))
        
        if isinstance(start, int) and isinstance(end, int):
            if start >= end:
                errors.append(ValidationError("Start position must be less than end position"))
            
            if end > len(text):
                errors.append(ValidationError("End position exceeds text length"))
        
        return errors
    
    def validate_error_type(self, error_type: str) -> List[ValidationError]:
        """
        Validate error type.
        
        Args:
            error_type: The error type to validate
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        try:
            ErrorType(error_type)
        except ValueError:
            errors.append(ValidationError(f"Invalid error type: {error_type}. Must be T, B, or E"))
        
        return errors
    
    def validate_config(self, config: TrustScoreConfig) -> List[ValidationError]:
        """
        Validate TrustScore configuration.
        
        Args:
            config: The configuration to validate
            
        Returns:
            List of validation errors
        """
        errors: List[ValidationError] = []
        
        # Validate aggregation weights
        weights = config.aggregation_weights
        total_weight: float = weights.trustworthiness + weights.explainability + weights.bias
        if abs(total_weight - 1.0) > 0.01:
            errors.append(ValidationError(f"Aggregation weights must sum to 1.0, got {total_weight}"))
        
        # Validate confidence level
        if not 0.5 <= config.confidence_level <= 0.99:
            errors.append(ValidationError("Confidence level must be between 0.5 and 0.99"))
        
        # Validate severity thresholds
        thresholds = config.severity_thresholds
        if "minor" not in thresholds or "major" not in thresholds or "critical" not in thresholds:
            errors.append(ValidationError("All severity thresholds (minor, major, critical) must be defined"))
        
        if thresholds.get("minor", 0) >= thresholds.get("major", 1):
            errors.append(ValidationError("Minor threshold must be less than major threshold"))
        
        if thresholds.get("major", 1) >= thresholds.get("critical", 2):
            errors.append(ValidationError("Major threshold must be less than critical threshold"))
        
        return errors


class TrustScoreLogger:
    """
    Centralized logging for TrustScore pipeline.
    """
    
    def __init__(self, name: str = "trustscore", level: int = logging.INFO) -> None:
        self.logger: logging.Logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatter
        formatter: logging.Formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Create console handler if not already present
        if not self.logger.handlers:
            console_handler: logging.StreamHandler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
    
    def log_error(self, error: TrustScoreError, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a TrustScore error with context.
        
        Args:
            error: The error to log
            context: Additional context information
        """
        context = context or {}
        
        log_message: str = f"Error in {error.component or 'unknown'}: {error.message}"
        if error.details:
            log_message += f" Details: {error.details}"
        if context:
            log_message += f" Context: {context}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def log_processing_stage(self, stage: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log processing stage information.
        
        Args:
            stage: The processing stage
            details: Additional details
        """
        message: str = f"Processing stage: {stage}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def log_performance(self, operation: str, duration: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: The operation performed
            duration: Duration in seconds
            details: Additional details
        """
        message: str = f"Performance - {operation}: {duration:.3f}s"
        if details:
            message += f" - {details}"
        self.logger.info(message)


class ErrorHandler:
    """
    Centralized error handling for TrustScore pipeline.
    """
    
    def __init__(self, logger: Optional[TrustScoreLogger] = None) -> None:
        self.logger: TrustScoreLogger = logger or TrustScoreLogger()
    
    def handle_api_error(self, error: Exception, api_name: str, context: Optional[Dict[str, Any]] = None) -> APIError:
        """
        Handle API-related errors.
        
        Args:
            error: The original exception
            api_name: Name of the API that failed
            context: Additional context
            
        Returns:
            APIError: Wrapped error
        """
        api_error: APIError = APIError(
            message=f"API call failed for {api_name}: {str(error)}",
            api_name=api_name,
            details={
                "original_error": str(error),
                "error_type": type(error).__name__,
                "context": context or {}
            }
        )
        
        self.logger.log_error(api_error, context)
        return api_error
    
    def handle_processing_error(self, error: Exception, stage: str, context: Optional[Dict[str, Any]] = None) -> ProcessingError:
        """
        Handle processing-related errors.
        
        Args:
            error: The original exception
            stage: The processing stage that failed
            context: Additional context
            
        Returns:
            ProcessingError: Wrapped error
        """
        processing_error: ProcessingError = ProcessingError(
            message=f"Processing failed at stage '{stage}': {str(error)}",
            stage=stage,
            details={
                "original_error": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "context": context or {}
            }
        )
        
        self.logger.log_error(processing_error, context)
        return processing_error
    
    def handle_validation_error(self, error: Exception, field: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> ValidationError:
        """
        Handle validation-related errors.
        
        Args:
            error: The original exception
            field: The field that failed validation
            context: Additional context
            
        Returns:
            ValidationError: Wrapped error
        """
        validation_error: ValidationError = ValidationError(
            message=f"Validation failed{f' for field {field}' if field else ''}: {str(error)}",
            field=field,
            details={
                "original_error": str(error),
                "error_type": type(error).__name__,
                "context": context or {}
            }
        )
        
        self.logger.log_error(validation_error, context)
        return validation_error


# Global error handler instance
error_handler: ErrorHandler = ErrorHandler()
