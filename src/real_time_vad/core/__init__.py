"""
Core modules for Real-Time VAD library.
"""

from .config import VADConfig, SampleRate, SileroModelVersion
from .vad_wrapper import VADWrapper
from .async_vad_wrapper import AsyncVADWrapper
from .exceptions import (
    VADError,
    ModelNotFoundError,
    ConfigurationError,
    AudioProcessingError,
    ModelInitializationError,
    CallbackError
)

__all__ = [
    # Configuration
    "VADConfig",
    "SampleRate",
    "SileroModelVersion",
    
    # Main classes
    "VADWrapper",
    "AsyncVADWrapper",
    
    # Exceptions
    "VADError",
    "ModelNotFoundError",
    "ConfigurationError",
    "AudioProcessingError",
    "ModelInitializationError",
    "CallbackError",
]
