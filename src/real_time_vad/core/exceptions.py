"""
Custom exceptions for the VAD library.
"""

from typing import Optional


class VADError(Exception):
    """Base exception class for VAD-related errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ModelNotFoundError(VADError):
    """Raised when a Silero model file cannot be found or loaded."""
    
    def __init__(self, model_path: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Silero model not found at path: {model_path}"
        super().__init__(message, "MODEL_NOT_FOUND")
        self.model_path = model_path


class ConfigurationError(VADError):
    """Raised when there's an error in VAD configuration."""
    
    def __init__(self, parameter: str, value: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Invalid configuration for parameter '{parameter}': {value}"
        super().__init__(message, "CONFIGURATION_ERROR")
        self.parameter = parameter
        self.value = value


class AudioProcessingError(VADError):
    """Raised when there's an error processing audio data."""
    
    def __init__(self, message: str, audio_data_info: Optional[str] = None) -> None:
        super().__init__(message, "AUDIO_PROCESSING_ERROR")
        self.audio_data_info = audio_data_info


class ModelInitializationError(VADError):
    """Raised when there's an error initializing the Silero model."""
    
    def __init__(self, model_version: str, message: Optional[str] = None) -> None:
        if message is None:
            message = f"Failed to initialize Silero model version: {model_version}"
        super().__init__(message, "MODEL_INITIALIZATION_ERROR")
        self.model_version = model_version


class CallbackError(VADError):
    """Raised when there's an error in callback execution."""
    
    def __init__(self, callback_name: str, original_error: Exception) -> None:
        message = f"Error in callback '{callback_name}': {str(original_error)}"
        super().__init__(message, "CALLBACK_ERROR")
        self.callback_name = callback_name
        self.original_error = original_error
