"""
Real-Time Voice Activity Detection (VAD) Library
================================================

A comprehensive Python implementation of real-time Voice Activity Detection
using Silero models. This library provides efficient voice detection capabilities
with configurable parameters and easy integration into any Python project.

Basic Usage:
    >>> from real_time_vad import VADWrapper, SampleRate, SileroModelVersion
    >>> import numpy as np
    >>> 
    >>> # Create VAD instance
    >>> vad = VADWrapper()
    >>> vad.set_sample_rate(SampleRate.SAMPLERATE_16)
    >>> vad.set_silero_model(SileroModelVersion.V5)
    >>> 
    >>> # Set up callbacks
    >>> def on_voice_start():
    ...     print("Voice started!")
    >>> 
    >>> def on_voice_end(wav_data: bytes):
    ...     print(f"Voice ended! Got {len(wav_data)} bytes")
    >>> 
    >>> vad.set_callbacks(
    ...     voice_start_callback=on_voice_start,
    ...     voice_end_callback=on_voice_end
    ... )
    >>> 
    >>> # Process audio data
    >>> audio_data = np.random.randn(1024).astype(np.float32)
    >>> vad.process_audio_data(audio_data)
"""

from .core.config import VADConfig, SampleRate, SileroModelVersion
from .core.vad_wrapper import VADWrapper
from .core.async_vad_wrapper import AsyncVADWrapper
from .core.exceptions import VADError, ModelNotFoundError, ConfigurationError
from .utils.audio import AudioUtils
from .utils.wav_writer import WAVWriter

__version__ = "1.0.0"
__author__ = "VAD Library Team"
__email__ = "team@vadlibrary.com"

__all__ = [
    # Core classes
    "VADWrapper",
    "AsyncVADWrapper",
    "VADConfig",
    
    # Enums
    "SampleRate", 
    "SileroModelVersion",
    
    # Exceptions
    "VADError",
    "ModelNotFoundError", 
    "ConfigurationError",
    
    # Utilities
    "AudioUtils",
    "WAVWriter",
    
    # Version info
    "__version__",
    "__author__",
    "__email__",
]

# Package metadata
__package_name__ = "real-time-vad-silero"
__description__ = "Real-time Voice Activity Detection using Silero models"
__url__ = "https://github.com/picurit/cutter-vad"
__license__ = "MIT"
