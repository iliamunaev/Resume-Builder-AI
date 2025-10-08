"""Custom exceptions for the Resume Builder AI application."""

from typing import Optional, Dict, Any


class ResumeBuilderError(Exception):
    """Base exception class for Resume Builder AI."""

    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """Initialize base exception.

        Args:
            message: Error message.
            error_code: Optional error code for categorization.
            details: Optional additional error details.
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details
        }


class ConfigurationError(ResumeBuilderError):
    """Exception raised for configuration-related errors."""
    pass


class DataError(ResumeBuilderError):
    """Exception raised for data processing errors."""
    pass


class ModelError(ResumeBuilderError):
    """Exception raised for model-related errors."""
    pass


class ValidationError(ResumeBuilderError):
    """Exception raised for input validation errors."""
    pass


class APIError(ResumeBuilderError):
    """Exception raised for API-related errors."""
    pass


class FileNotFoundError(DataError):
    """Exception raised when required files are not found."""

    def __init__(self, file_path: str, message: Optional[str] = None):
        """Initialize file not found error.

        Args:
            file_path: Path to the missing file.
            message: Custom error message.
        """
        if message is None:
            message = f"Required file not found: {file_path}"
        super().__init__(message, error_code="FILE_NOT_FOUND", details={"file_path": file_path})


class ModelLoadError(ModelError):
    """Exception raised when model loading fails."""

    def __init__(self, model_name: str, message: Optional[str] = None):
        """Initialize model load error.

        Args:
            model_name: Name of the model that failed to load.
            message: Custom error message.
        """
        if message is None:
            message = f"Failed to load model: {model_name}"
        super().__init__(message, error_code="MODEL_LOAD_FAILED", details={"model_name": model_name})


class EmbeddingError(ModelError):
    """Exception raised for embedding-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize embedding error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message, error_code="EMBEDDING_ERROR", details=details)


class GenerationError(ModelError):
    """Exception raised for text generation errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize generation error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message, error_code="GENERATION_ERROR", details=details)


class SearchError(ResumeBuilderError):
    """Exception raised for search-related errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize search error.

        Args:
            message: Error message.
            details: Additional error details.
        """
        super().__init__(message, error_code="SEARCH_ERROR", details=details)


def handle_error(error: Exception) -> ResumeBuilderError:
    """Convert generic exceptions to ResumeBuilderError instances.

    Args:
        error: The original exception.

    Returns:
        Appropriate ResumeBuilderError instance.
    """
    if isinstance(error, ResumeBuilderError):
        return error

    # Map common exception types to our custom exceptions
    if isinstance(error, FileNotFoundError):
        return FileNotFoundError(str(error))
    elif isinstance(error, ValueError):
        return ValidationError(str(error))
    elif isinstance(error, (OSError, IOError)):
        return DataError(str(error))
    elif isinstance(error, (ImportError, ModuleNotFoundError)):
        return ModelError(str(error))
    else:
        return ResumeBuilderError(f"Unexpected error: {str(error)}")


def create_error_response(error: Exception) -> Dict[str, Any]:
    """Create standardized error response for API endpoints.

    Args:
        error: The exception that occurred.

    Returns:
        Dictionary containing error information for API response.
    """
    resume_error = handle_error(error)
    return resume_error.to_dict()
