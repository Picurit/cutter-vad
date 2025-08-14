"""
Main VAD wrapper class that provides the public API.
"""

import threading
import time
from typing import Optional, Callable, Union, Any, Dict, List, ClassVar
from contextlib import contextmanager
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict, ValidationError

from .config import VADConfig, SampleRate, SileroModelVersion
from .silero_model import VADProcessor, ProcessingResult, ProcessingStatistics
from .exceptions import VADError, ConfigurationError, CallbackError, AudioProcessingError
from ..utils.audio import AudioUtils


# Type aliases for callbacks
VoiceStartCallback = Callable[[], None]
VoiceEndCallback = Callable[[bytes], None]
VoiceContinueCallback = Callable[[bytes], None]


class VADWrapperState(BaseModel):
    """
    Pydantic model for VAD wrapper internal state.
    
    Provides type validation and state management for the VAD wrapper's
    internal state, ensuring consistency and thread-safety.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
        frozen=False  # Allow state updates
    )
    
    # Initialization state
    is_initialized: bool = Field(
        default=False,
        description="Whether the VAD processor is initialized"
    )
    
    # Processing statistics
    total_frames_processed: int = Field(
        default=0,
        ge=0,
        description="Total number of audio frames processed"
    )
    
    total_processing_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Total time spent processing audio (seconds)"
    )
    
    # Error state
    last_error: Optional[str] = Field(
        default=None,
        description="String representation of the last error that occurred"
    )
    
    @property
    def average_processing_time_per_frame(self) -> float:
        """Calculate average processing time per frame."""
        if self.total_frames_processed == 0:
            return 0.0
        return self.total_processing_time / self.total_frames_processed
    
    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self.total_frames_processed = 0
        self.total_processing_time = 0.0
    
    def record_error(self, error: Exception) -> None:
        """Record an error in the state."""
        self.last_error = str(error)
    
    def clear_error(self) -> None:
        """Clear the last recorded error."""
        self.last_error = None


class CallbackConfiguration(BaseModel):
    """
    Pydantic model for callback configuration.
    
    Provides type validation and management of callback functions
    with proper error handling and validation.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    voice_start_callback: Optional[VoiceStartCallback] = Field(
        default=None,
        description="Callback function called when voice activity starts"
    )
    
    voice_end_callback: Optional[VoiceEndCallback] = Field(
        default=None,
        description="Callback function called when voice activity ends"
    )
    
    voice_continue_callback: Optional[VoiceContinueCallback] = Field(
        default=None,
        description="Callback function called during continuous voice activity"
    )
    
    @field_validator('voice_start_callback', 'voice_end_callback', 'voice_continue_callback')
    @classmethod
    def validate_callback(cls, v: Optional[Callable]) -> Optional[Callable]:
        """Validate that the callback is callable if provided."""
        if v is not None and not callable(v):
            raise ValueError("Callback must be a callable function")
        return v
    
    def has_any_callback(self) -> bool:
        """Check if any callback is configured."""
        return any([
            self.voice_start_callback is not None,
            self.voice_end_callback is not None,
            self.voice_continue_callback is not None
        ])


class ThresholdConfiguration(BaseModel):
    """
    Pydantic model for VAD threshold configuration.
    
    Provides comprehensive validation of VAD detection thresholds
    with proper range checking and logical validation.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    vad_start_probability: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability threshold for starting VAD detection"
    )
    
    vad_end_probability: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Probability threshold for ending VAD detection"
    )
    
    voice_start_ratio: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="True positive ratio for voice start detection"
    )
    
    voice_end_ratio: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="False positive ratio for voice end detection"
    )
    
    voice_start_frame_count: int = Field(
        default=10,
        ge=1,
        description="Frame count required to confirm voice start"
    )
    
    voice_end_frame_count: int = Field(
        default=57,
        ge=1,
        description="Frame count required to confirm voice end"
    )
    
    @model_validator(mode='after')
    def validate_threshold_logic(self) -> 'ThresholdConfiguration':
        """Validate logical consistency of threshold values."""
        # Ensure start and end probabilities are reasonable
        if self.vad_start_probability < 0.1:
            raise ValueError("Start probability should be at least 0.1 for reliable detection")
        
        if self.vad_end_probability < 0.1:
            raise ValueError("End probability should be at least 0.1 for reliable detection")
        
        # Ensure frame counts are reasonable
        if self.voice_start_frame_count > 100:
            raise ValueError("Voice start frame count should not exceed 100 for responsive detection")
        
        if self.voice_end_frame_count > 200:
            raise ValueError("Voice end frame count should not exceed 200 for responsive detection")
        
        return self


class VADWrapper:
    """
    Main Voice Activity Detection wrapper class with Pydantic-based validation.
    
    This class provides a high-level interface for real-time voice activity detection
    using Silero models. It features robust type validation, state management, and 
    comprehensive error handling using Pydantic models.
    
    The class follows SOLID principles:
    - Single Responsibility: Manages VAD operations and state
    - Open/Closed: Extensible through configuration and callbacks
    - Liskov Substitution: Can be used wherever VAD interface is expected
    - Interface Segregation: Clean, focused public API
    - Dependency Inversion: Depends on abstractions (VADProcessor, VADConfig)
    
    Example:
        >>> config = VADConfig(sample_rate=SampleRate.SAMPLERATE_16)
        >>> vad = VADWrapper(config=config)
        >>> 
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
        >>> # Process audio data with automatic validation
        >>> audio_data = np.random.randn(1024).astype(np.float32)
        >>> vad.process_audio_data(audio_data)
    """
    
    # Class-level configuration
    _DEFAULT_FRAME_OVERLAP_RATIO: ClassVar[float] = 0.5
    _MAX_PROCESSING_TIME_WARNING: ClassVar[float] = 1.0  # seconds
    
    def __init__(self, config: Optional[VADConfig] = None) -> None:
        """
        Initialize VAD wrapper with robust validation and state management.
        
        Args:
            config: VAD configuration. If None, uses default configuration.
            
        Raises:
            VADError: If initialization fails
            ValidationError: If configuration is invalid
        """
        try:
            # Initialize and validate configuration
            self._config = config if config is not None else VADConfig()
            self._validate_initial_config()
            
            # Initialize Pydantic models for state management
            self._state = VADWrapperState()
            self._callbacks = CallbackConfiguration()
            
            # Thread safety
            self._lock = threading.Lock()
            
            # Core components
            self._processor: Optional[VADProcessor] = None
            
            # Initialize processor
            self._initialize_processor()
            
        except ValidationError as e:
            raise VADError(f"Invalid configuration provided: {e}")
        except Exception as e:
            raise VADError(f"Failed to initialize VAD wrapper: {e}")
    
    def _validate_initial_config(self) -> None:
        """Validate the initial configuration using Pydantic."""
        try:
            # Re-validate the config to ensure it's properly formed
            if not isinstance(self._config, VADConfig):
                raise ValueError("Config must be a VADConfig instance")
            
            # Additional business logic validation
            if self._config.buffer_size <= 0:
                raise ValueError("Buffer size must be positive")
                
        except ValidationError as e:
            raise ConfigurationError("config", str(self._config), str(e))
    
    def _initialize_processor(self) -> None:
        """Initialize the VAD processor with proper error handling."""
        try:
            self._processor = VADProcessor(self._config)
            self._state.is_initialized = True
            self._state.clear_error()
            
        except Exception as e:
            self._state.record_error(e)
            self._state.is_initialized = False
            raise VADError(f"Failed to initialize VAD processor: {e}")
    
    # ========================= Configuration Management =========================
    
    def set_sample_rate(self, sample_rate: SampleRate) -> None:
        """
        Set the audio sample rate with validation and processor reinitialization.
        
        Args:
            sample_rate: Target sample rate (must be a valid SampleRate enum)
            
        Raises:
            ConfigurationError: If sample rate is invalid
            VADError: If processor reinitialization fails
        """
        with self._lock:
            try:
                # Validate input using Pydantic
                if not isinstance(sample_rate, SampleRate):
                    raise ValueError(f"Invalid sample rate type: {type(sample_rate)}")
                
                old_sample_rate = self._config.sample_rate
                
                # Update configuration
                self._config.sample_rate = sample_rate
                
                # Reinitialize processor if sample rate changed and we're initialized
                if old_sample_rate != sample_rate and self._state.is_initialized:
                    self._initialize_processor()
                    
            except ValidationError as e:
                raise ConfigurationError("sample_rate", str(sample_rate), str(e))
            except Exception as e:
                self._state.record_error(e)
                raise ConfigurationError("sample_rate", str(sample_rate), str(e))
    
    def set_silero_model(self, model_version: SileroModelVersion) -> None:
        """
        Set the Silero model version with validation.
        
        Args:
            model_version: Model version to use (must be a valid SileroModelVersion enum)
            
        Raises:
            ConfigurationError: If model version is invalid
            VADError: If processor reinitialization fails
        """
        with self._lock:
            try:
                # Validate input
                if not isinstance(model_version, SileroModelVersion):
                    raise ValueError(f"Invalid model version type: {type(model_version)}")
                
                old_version = self._config.model_version
                
                # Update configuration
                self._config.model_version = model_version
                
                # Reinitialize processor if model version changed and we're initialized
                if old_version != model_version and self._state.is_initialized:
                    self._initialize_processor()
                    
            except ValidationError as e:
                raise ConfigurationError("model_version", str(model_version), str(e))
            except Exception as e:
                self._state.record_error(e)
                raise ConfigurationError("model_version", str(model_version), str(e))
    
    def set_thresholds(
        self,
        vad_start_probability: float = 0.7,
        vad_end_probability: float = 0.7,
        voice_start_ratio: float = 0.8,
        voice_end_ratio: float = 0.95,
        voice_start_frame_count: int = 10,
        voice_end_frame_count: int = 57
    ) -> None:
        """
        Configure VAD detection thresholds with comprehensive validation.
        
        Args:
            vad_start_probability: Probability threshold for starting VAD detection (0.0-1.0)
            vad_end_probability: Probability threshold for ending VAD detection (0.0-1.0)
            voice_start_ratio: True positive ratio for voice start detection (0.0-1.0)
            voice_end_ratio: False positive ratio for voice end detection (0.0-1.0)
            voice_start_frame_count: Frame count required to confirm voice start (≥1)
            voice_end_frame_count: Frame count required to confirm voice end (≥1)
            
        Raises:
            ConfigurationError: If any threshold value is invalid
            ValidationError: If threshold configuration is logically inconsistent
        """
        with self._lock:
            try:
                # Create and validate threshold configuration using Pydantic
                threshold_config = ThresholdConfiguration(
                    vad_start_probability=vad_start_probability,
                    vad_end_probability=vad_end_probability,
                    voice_start_ratio=voice_start_ratio,
                    voice_end_ratio=voice_end_ratio,
                    voice_start_frame_count=voice_start_frame_count,
                    voice_end_frame_count=voice_end_frame_count
                )
                
                # Update main configuration
                self._config.vad_start_probability = threshold_config.vad_start_probability
                self._config.vad_end_probability = threshold_config.vad_end_probability
                self._config.voice_start_ratio = threshold_config.voice_start_ratio
                self._config.voice_end_ratio = threshold_config.voice_end_ratio
                self._config.voice_start_frame_count = threshold_config.voice_start_frame_count
                self._config.voice_end_frame_count = threshold_config.voice_end_frame_count
                
                # Reset processor state to apply new thresholds
                if self._processor:
                    self._processor.reset()
                    
            except ValidationError as e:
                raise ConfigurationError("thresholds", "multiple", str(e))
            except Exception as e:
                self._state.record_error(e)
                raise ConfigurationError("thresholds", "multiple", str(e))
    
    # ========================= Callback Management =========================
    
    def set_callbacks(
        self,
        voice_start_callback: Optional[VoiceStartCallback] = None,
        voice_end_callback: Optional[VoiceEndCallback] = None,
        voice_continue_callback: Optional[VoiceContinueCallback] = None
    ) -> None:
        """
        Set callback functions for voice activity events with validation.
        
        Args:
            voice_start_callback: Called when voice activity starts
            voice_end_callback: Called when voice activity ends with WAV data
            voice_continue_callback: Called continuously during voice activity with PCM data
            
        Raises:
            ValidationError: If any callback is not callable
        """
        try:
            # Create and validate callback configuration using Pydantic
            self._callbacks = CallbackConfiguration(
                voice_start_callback=voice_start_callback,
                voice_end_callback=voice_end_callback,
                voice_continue_callback=voice_continue_callback
            )
            
        except ValidationError as e:
            raise VADError(f"Invalid callback configuration: {e}")
    
    def _execute_callback_safely(
        self,
        callback: Optional[Callable],
        callback_name: str,
        *args, **kwargs
    ) -> None:
        """
        Execute a callback with proper error handling and logging.
        
        Args:
            callback: The callback function to execute
            callback_name: Name of the callback for error reporting
            *args: Positional arguments to pass to callback
            **kwargs: Keyword arguments to pass to callback
            
        Raises:
            CallbackError: If callback execution fails
        """
        if callback is None:
            return
            
        try:
            callback(*args, **kwargs)
        except Exception as e:
            self._state.record_error(e)
            raise CallbackError(callback_name, e)
    
    def _handle_callbacks(self, result: ProcessingResult) -> None:
        """
        Handle callback execution with Pydantic model validation and error handling.
        
        Args:
            result: Processing result from VAD processor
            
        Raises:
            CallbackError: If any callback execution fails
        """
        try:
            # Validate the processing result using Pydantic
            if not isinstance(result, ProcessingResult):
                raise ValueError("Invalid processing result type")
            
            # Debug: Log processing result details
            import logging
            logging.debug(f"VAD Processing Result: voice_started={result.voice_started}, voice_ended={result.voice_ended}, voice_continuing={result.voice_continuing}, voice_probability={getattr(result, 'voice_probability', 'N/A')}")
            
            # Execute callbacks based on voice activity state
            if result.voice_started:
                logging.info(f"VAD: Voice started detected!")
                self._execute_callback_safely(
                    self._callbacks.voice_start_callback,
                    "voice_start"
                )
            
            if result.voice_ended and result.wav_data:
                logging.info(f"VAD: Voice ended detected!")
                self._execute_callback_safely(
                    self._callbacks.voice_end_callback,
                    "voice_end",
                    result.wav_data
                )
            
            if result.voice_continuing and result.pcm_data:
                logging.debug(f"VAD: Voice continuing detected!")
                self._execute_callback_safely(
                    self._callbacks.voice_continue_callback,
                    "voice_continue",
                    result.pcm_data
                )
                
        except ValidationError as e:
            raise CallbackError("result_validation", e)
    
    # ========================= Audio Processing =========================
    
    @contextmanager
    def _processing_context(self):
        """Context manager for audio processing with timing and error handling."""
        if not self._state.is_initialized or self._processor is None:
            raise VADError("VAD processor not initialized")
        
        start_time = time.time()
        try:
            yield
        finally:
            processing_time = time.time() - start_time
            self._state.total_processing_time += processing_time
            
            # Warn about long processing times
            if processing_time > self._MAX_PROCESSING_TIME_WARNING:
                import warnings
                warnings.warn(
                    f"Audio processing took {processing_time:.3f}s, "
                    f"which may indicate performance issues"
                )
    
    def process_audio_data(self, audio_data: Union[np.ndarray, List[float]]) -> None:
        """
        Process audio data for voice activity detection with comprehensive validation.
        
        Args:
            audio_data: Audio data as numpy array or list of float values (mono only)
                       Values should be in the range [-1.0, 1.0]
                       
        Raises:
            VADError: If VAD processor is not initialized
            AudioProcessingError: If audio processing fails
            ValidationError: If audio data is invalid
        """
        with self._lock:
            with self._processing_context():
                try:
                    # Validate and convert input to numpy array
                    validated_audio = self._validate_and_prepare_audio(audio_data)
                    
                    # Process audio in frames
                    self._process_audio_frames(validated_audio)
                    
                except ValidationError as e:
                    self._state.record_error(e)
                    raise AudioProcessingError(f"Audio validation failed: {e}")
                except Exception as e:
                    self._state.record_error(e)
                    raise AudioProcessingError(f"Audio processing failed: {e}")
    
    def _validate_and_prepare_audio(
        self,
        audio_data: Union[np.ndarray, List[float]]
    ) -> np.ndarray:
        """
        Validate and prepare audio data for processing.
        
        Args:
            audio_data: Raw audio data
            
        Returns:
            Validated and prepared numpy array
            
        Raises:
            AudioProcessingError: If audio data is invalid
        """
        # Convert input to numpy array if needed
        if isinstance(audio_data, list):
            if not audio_data:
                raise AudioProcessingError("Audio data cannot be empty")
            validated_audio = np.array(audio_data, dtype=np.float32)
        elif isinstance(audio_data, np.ndarray):
            validated_audio = audio_data.astype(np.float32)
        else:
            raise AudioProcessingError(f"Unsupported audio data type: {type(audio_data)}")
        
        # Validate audio data using AudioUtils
        AudioUtils.validate_audio_data(validated_audio)
        
        # Convert to mono if needed
        validated_audio = AudioUtils.convert_to_mono(validated_audio)
        
        return validated_audio
    
    def _process_audio_frames(self, audio_data: np.ndarray) -> None:
        """
        Process audio data in frames with proper frame management.
        
        Args:
            audio_data: Validated audio data
            
        Raises:
            AudioProcessingError: If frame processing fails
        """
        try:
            # Resample if needed (placeholder for future implementation)
            if self._config.auto_convert_sample_rate:
                # In a real implementation, you'd need to track the input sample rate
                pass
            
            # Process audio in frames
            frame_size = self._config.buffer_size
            hop_size = int(frame_size * self._DEFAULT_FRAME_OVERLAP_RATIO)
            
            frames = AudioUtils.split_into_frames(audio_data, frame_size, hop_size)
            
            for frame in frames:
                # Pad frame if necessary
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)))
                
                # Process frame
                result = self._processor.process_frame(frame)
                
                # Handle callbacks
                self._handle_callbacks(result)
                
                # Update statistics
                self._state.total_frames_processed += 1
                
        except Exception as e:
            raise AudioProcessingError(f"Frame processing failed: {e}")
    
    def process_audio_data_with_buffer(
        self,
        audio_buffer: np.ndarray,
        count: int
    ) -> None:
        """
        Process audio data from a buffer with specified count.
        
        This method provides compatibility with the original C-style interface
        while maintaining validation and error handling.
        
        Args:
            audio_buffer: Audio buffer (float32 array)
            count: Number of samples to process
            
        Raises:
            AudioProcessingError: If buffer parameters are invalid
            ValidationError: If count exceeds buffer size
        """
        try:
            # Validate buffer parameters
            if not isinstance(audio_buffer, np.ndarray):
                raise AudioProcessingError("Audio buffer must be a numpy array")
            
            if count < 0:
                raise AudioProcessingError("Count must be non-negative")
            
            if count > len(audio_buffer):
                raise AudioProcessingError(
                    f"Count {count} exceeds buffer size {len(audio_buffer)}"
                )
            
            # Process only the specified number of samples
            audio_data = audio_buffer[:count]
            self.process_audio_data(audio_data)
            
        except Exception as e:
            if not isinstance(e, (AudioProcessingError, ValidationError)):
                raise AudioProcessingError(f"Buffer processing failed: {e}")
            raise
    
    # ========================= Backward Compatibility =========================
    
    @property
    def processor(self) -> Optional[VADProcessor]:
        """
        Access to the internal VAD processor for backward compatibility.
        
        Note: Direct access to the processor is discouraged. Use the wrapper's
        methods instead for better error handling and validation.
        
        Returns:
            The internal VAD processor instance
        """
        return self._processor
    
    @property
    def config(self) -> VADConfig:
        """
        Access to the configuration for backward compatibility.
        
        Returns:
            The current VAD configuration
        """
        return self._config
    
    @config.setter
    def config(self, value: VADConfig) -> None:
        """
        Set the configuration for backward compatibility.
        
        Args:
            value: New VAD configuration
        """
        self.update_config(value)
    
    # ========================= State Management =========================
    
    def reset(self) -> None:
        """
        Reset VAD state and statistics.
        
        Clears all processing statistics and resets the processor state
        while maintaining the current configuration.
        """
        with self._lock:
            try:
                if self._processor:
                    self._processor.reset()
                
                # Reset state using Pydantic model methods
                self._state.reset_statistics()
                self._state.clear_error()
                
            except Exception as e:
                self._state.record_error(e)
                raise VADError(f"Failed to reset VAD state: {e}")
    
    def cleanup(self) -> None:
        """
        Clean up resources and reset initialization state.
        
        Releases all resources and marks the wrapper as uninitialized.
        After calling this method, the wrapper must be reinitialized to be used again.
        """
        with self._lock:
            try:
                self._processor = None
                self._state.is_initialized = False
                self._state.clear_error()
                
            except Exception as e:
                # Log error but don't raise during cleanup
                self._state.record_error(e)
    
    # ========================= Information and Statistics =========================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive processing statistics using Pydantic models.
        
        Returns:
            Dictionary with processing statistics including state and processor info
            
        Example:
            >>> stats = vad.get_statistics()
            >>> print(f"Frames processed: {stats['total_frames_processed']}")
            >>> print(f"Average time per frame: {stats['average_processing_time_per_frame']}")
        """
        with self._lock:
            try:
                # Base statistics from our state model
                stats = {
                    'total_frames_processed': self._state.total_frames_processed,
                    'total_processing_time': self._state.total_processing_time,
                    'average_processing_time_per_frame': self._state.average_processing_time_per_frame,
                    'is_initialized': self._state.is_initialized,
                    'last_error': self._state.last_error,
                    'has_callbacks': self._callbacks.has_any_callback(),
                    'config': self._serialize_config_for_json()
                }
                
                # Add processor statistics if available
                if self._processor:
                    processor_stats = self._processor.get_statistics()
                    # Convert Pydantic model to dictionary
                    if isinstance(processor_stats, ProcessingStatistics):
                        stats.update(processor_stats.model_dump())
                    else:
                        stats.update(processor_stats)
                
                return stats
                
            except Exception as e:
                self._state.record_error(e)
                # Return minimal stats if error occurs
                return {
                    'error': str(e),
                    'is_initialized': self._state.is_initialized,
                    'total_frames_processed': self._state.total_frames_processed
                }
    
    def _serialize_config_for_json(self) -> Dict[str, Any]:
        """
        Serialize configuration for JSON compatibility.
        
        Returns:
            JSON-serializable dictionary representation of the configuration
        """
        try:
            config_dict = self._config.model_dump()
            
            # Convert enums to their values for JSON serialization
            if 'sample_rate' in config_dict:
                config_dict['sample_rate'] = config_dict['sample_rate']
            
            if 'model_version' in config_dict:
                if hasattr(config_dict['model_version'], 'value'):
                    config_dict['model_version'] = config_dict['model_version'].value
                else:
                    config_dict['model_version'] = str(config_dict['model_version'])
            
            # Convert Path objects to strings
            if 'model_path' in config_dict and config_dict['model_path'] is not None:
                config_dict['model_path'] = str(config_dict['model_path'])
            
            return config_dict
            
        except Exception as e:
            # Fallback to basic representation
            return {
                'sample_rate': self._config.sample_rate.value if hasattr(self._config.sample_rate, 'value') else int(self._config.sample_rate),
                'model_version': self._config.model_version.value if hasattr(self._config.model_version, 'value') else str(self._config.model_version),
                'error': f"Serialization error: {e}"
            }
    
    def get_config(self) -> VADConfig:
        """
        Get current configuration.
        
        Returns:
            Current VAD configuration as a Pydantic model
        """
        return self._config
    
    def update_config(self, config: VADConfig) -> None:
        """
        Update configuration and reinitialize if needed.
        
        Args:
            config: New configuration (must be a valid VADConfig instance)
            
        Raises:
            VADError: If configuration update fails
            ValidationError: If configuration is invalid
        """
        with self._lock:
            try:
                # Validate configuration using Pydantic
                if not isinstance(config, VADConfig):
                    raise ValueError("Config must be a VADConfig instance")
                
                # Validate the configuration
                config.model_validate(config.model_dump())
                
                old_config = self._config
                
                if self._processor:
                    # Use the processor's update_config method for proper handling
                    self._processor.update_config(config)
                else:
                    # If no processor, just update config and initialize
                    self._config = config
                    self._initialize_processor()
                
                # Update wrapper's config reference
                self._config = config
                self._state.clear_error()
                
            except ValidationError as e:
                self._state.record_error(e)
                raise VADError(f"Invalid configuration: {e}")
            except Exception as e:
                # Rollback to old configuration if possible
                if 'old_config' in locals():
                    try:
                        self._config = old_config
                    except:
                        pass
                
                self._state.record_error(e)
                raise VADError(f"Failed to update configuration: {e}")
    
    def is_voice_active(self) -> bool:
        """
        Check if voice is currently active.
        
        Returns:
            True if voice is currently active, False otherwise
        """
        try:
            if self._processor:
                return self._processor.is_voice_active
            return False
        except Exception as e:
            self._state.record_error(e)
            return False
    
    def get_last_error(self) -> Optional[str]:
        """
        Get the last error that occurred as a string.
        
        Returns:
            String representation of the last error, or None if no error
        """
        return self._state.last_error
    
    def get_last_error_details(self) -> Dict[str, Any]:
        """
        Get detailed information about the last error.
        
        Returns:
            Dictionary with error details including timestamp and context
        """
        return {
            'last_error': self._state.last_error,
            'is_initialized': self._state.is_initialized,
            'total_frames_processed': self._state.total_frames_processed,
            'has_processor': self._processor is not None
        }
    
    # ========================= Context Management =========================
    
    def __enter__(self) -> 'VADWrapper':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with proper cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Ignore errors during cleanup in context manager
            pass
    
    def __del__(self) -> None:
        """Destructor with safe cleanup."""
        try:
            self.cleanup()
        except:
            # Ignore all errors during destruction
            pass
    
    # ========================= String Representation =========================
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"VADWrapper("
            f"initialized={self._state.is_initialized}, "
            f"sample_rate={self._config.sample_rate}, "
            f"model_version={self._config.model_version}, "
            f"frames_processed={self._state.total_frames_processed}"
            f")"
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        status = "Initialized" if self._state.is_initialized else "Not Initialized"
        return (
            f"VAD Wrapper - {status}\n"
            f"Sample Rate: {self._config.sample_rate.value} Hz\n"
            f"Model Version: {self._config.model_version.value}\n"
            f"Frames Processed: {self._state.total_frames_processed}\n"
            f"Has Callbacks: {self._callbacks.has_any_callback()}"
        )
