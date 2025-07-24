"""
Main VAD wrapper class that provides the public API.
"""

import threading
import time
from typing import Optional, Callable, Union, Any, Dict, List
import numpy as np
from pathlib import Path

from .config import VADConfig, SampleRate, SileroModelVersion
from .silero_model import VADProcessor
from .exceptions import VADError, ConfigurationError, CallbackError, AudioProcessingError
from ..utils.audio import AudioUtils


# Type aliases for callbacks
VoiceStartCallback = Callable[[], None]
VoiceEndCallback = Callable[[bytes], None]
VoiceContinueCallback = Callable[[bytes], None]


class VADWrapper:
    """
    Main Voice Activity Detection wrapper class.
    
    This class provides a high-level interface for real-time voice activity detection
    using Silero models. It handles audio processing, state management, and callbacks.
    
    Example:
        >>> vad = VADWrapper()
        >>> vad.set_sample_rate(SampleRate.SAMPLERATE_16)
        >>> vad.set_silero_model(SileroModelVersion.V5)
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
        >>> # Process audio data
        >>> audio_data = np.random.randn(1024).astype(np.float32)
        >>> vad.process_audio_data(audio_data)
    """
    
    def __init__(self, config: Optional[VADConfig] = None) -> None:
        """
        Initialize VAD wrapper.
        
        Args:
            config: VAD configuration. If None, uses default configuration.
        """
        self.config = config if config is not None else VADConfig()
        self.processor: Optional[VADProcessor] = None
        self._lock = threading.Lock()
        self._is_initialized = False
        
        # Callbacks
        self._voice_start_callback: Optional[VoiceStartCallback] = None
        self._voice_end_callback: Optional[VoiceEndCallback] = None
        self._voice_continue_callback: Optional[VoiceContinueCallback] = None
        
        # Processing state
        self._total_frames_processed = 0
        self._total_processing_time = 0.0
        self._last_error: Optional[Exception] = None
        
        # Initialize processor
        self._initialize_processor()
    
    def _initialize_processor(self) -> None:
        """Initialize the VAD processor."""
        try:
            self.processor = VADProcessor(self.config)
            self._is_initialized = True
        except Exception as e:
            self._last_error = e
            raise VADError(f"Failed to initialize VAD processor: {str(e)}")
    
    def set_sample_rate(self, sample_rate: SampleRate) -> None:
        """
        Set the audio sample rate.
        
        Args:
            sample_rate: Target sample rate
        """
        with self._lock:
            try:
                old_sample_rate = self.config.sample_rate
                self.config.sample_rate = sample_rate
                
                # Reinitialize processor if sample rate changed
                if old_sample_rate != sample_rate and self._is_initialized:
                    self._initialize_processor()
                    
            except Exception as e:
                raise ConfigurationError("sample_rate", str(sample_rate), str(e))
    
    def set_silero_model(self, model_version: SileroModelVersion) -> None:
        """
        Set the Silero model version.
        
        Args:
            model_version: Model version to use
        """
        with self._lock:
            try:
                old_version = self.config.model_version
                self.config.model_version = model_version
                
                # Reinitialize processor if model version changed
                if old_version != model_version and self._is_initialized:
                    self._initialize_processor()
                    
            except Exception as e:
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
        Configure VAD detection thresholds.
        
        Args:
            vad_start_probability: Probability threshold for starting VAD detection (0.0-1.0)
            vad_end_probability: Probability threshold for ending VAD detection (0.0-1.0)
            voice_start_ratio: True positive ratio for voice start detection (0.0-1.0)
            voice_end_ratio: False positive ratio for voice end detection (0.0-1.0)
            voice_start_frame_count: Frame count required to confirm voice start
            voice_end_frame_count: Frame count required to confirm voice end
        """
        with self._lock:
            try:
                # Validate parameters
                if not (0.0 <= vad_start_probability <= 1.0):
                    raise ValueError("vad_start_probability must be between 0.0 and 1.0")
                if not (0.0 <= vad_end_probability <= 1.0):
                    raise ValueError("vad_end_probability must be between 0.0 and 1.0")
                if not (0.0 <= voice_start_ratio <= 1.0):
                    raise ValueError("voice_start_ratio must be between 0.0 and 1.0")
                if not (0.0 <= voice_end_ratio <= 1.0):
                    raise ValueError("voice_end_ratio must be between 0.0 and 1.0")
                if voice_start_frame_count < 1:
                    raise ValueError("voice_start_frame_count must be >= 1")
                if voice_end_frame_count < 1:
                    raise ValueError("voice_end_frame_count must be >= 1")
                
                # Update configuration
                self.config.vad_start_probability = vad_start_probability
                self.config.vad_end_probability = vad_end_probability
                self.config.voice_start_ratio = voice_start_ratio
                self.config.voice_end_ratio = voice_end_ratio
                self.config.voice_start_frame_count = voice_start_frame_count
                self.config.voice_end_frame_count = voice_end_frame_count
                
                # Reset processor state to apply new thresholds
                if self.processor:
                    self.processor.reset()
                    
            except Exception as e:
                raise ConfigurationError("thresholds", "multiple", str(e))
    
    def set_callbacks(
        self,
        voice_start_callback: Optional[VoiceStartCallback] = None,
        voice_end_callback: Optional[VoiceEndCallback] = None,
        voice_continue_callback: Optional[VoiceContinueCallback] = None
    ) -> None:
        """
        Set callback functions for voice activity events.
        
        Args:
            voice_start_callback: Called when voice activity starts
            voice_end_callback: Called when voice activity ends with WAV data
            voice_continue_callback: Called continuously during voice activity with PCM data
        """
        self._voice_start_callback = voice_start_callback
        self._voice_end_callback = voice_end_callback
        self._voice_continue_callback = voice_continue_callback
    
    def process_audio_data(self, audio_data: Union[np.ndarray, List[float]]) -> None:
        """
        Process audio data for voice activity detection.
        
        Args:
            audio_data: Audio data as numpy array or list of float values (mono only)
                       Values should be in the range [-1.0, 1.0]
        """
        if not self._is_initialized or self.processor is None:
            raise VADError("VAD processor not initialized")
        
        start_time = time.time()
        
        try:
            with self._lock:
                # Convert input to numpy array if needed
                if isinstance(audio_data, list):
                    audio_data = np.array(audio_data, dtype=np.float32)
                elif isinstance(audio_data, np.ndarray):
                    audio_data = audio_data.astype(np.float32)
                else:
                    raise AudioProcessingError(f"Unsupported audio data type: {type(audio_data)}")
                
                # Validate audio data
                AudioUtils.validate_audio_data(audio_data)
                
                # Convert to mono if needed
                audio_data = AudioUtils.convert_to_mono(audio_data)
                
                # Resample if needed
                if self.config.auto_convert_sample_rate:
                    # Assume input is at target sample rate for now
                    # In a real implementation, you'd need to track the input sample rate
                    pass
                
                # Process audio in frames
                frame_size = self.config.buffer_size
                hop_size = frame_size // 2  # 50% overlap
                
                frames = AudioUtils.split_into_frames(audio_data, frame_size, hop_size)
                
                for frame in frames:
                    # Pad frame if necessary
                    if len(frame) < frame_size:
                        frame = np.pad(frame, (0, frame_size - len(frame)))
                    
                    # Process frame
                    result = self.processor.process_frame(frame)
                    
                    # Handle callbacks
                    self._handle_callbacks(result)
                    
                    self._total_frames_processed += 1
                
        except Exception as e:
            self._last_error = e
            raise AudioProcessingError(f"Audio processing failed: {str(e)}")
        
        finally:
            self._total_processing_time += time.time() - start_time
    
    def process_audio_data_with_buffer(self, audio_buffer: np.ndarray, count: int) -> None:
        """
        Process audio data from a buffer with specified count.
        
        This method provides compatibility with the original C-style interface.
        
        Args:
            audio_buffer: Audio buffer (float32 array)
            count: Number of samples to process
        """
        if count > len(audio_buffer):
            raise AudioProcessingError(f"Count {count} exceeds buffer size {len(audio_buffer)}")
        
        # Process only the specified number of samples
        audio_data = audio_buffer[:count]
        self.process_audio_data(audio_data)
    
    def _handle_callbacks(self, result: Dict[str, Any]) -> None:
        """Handle callback execution."""
        try:
            if result.get('voice_started', False) and self._voice_start_callback:
                self._voice_start_callback()
            
            if result.get('voice_ended', False) and self._voice_end_callback:
                wav_data = result.get('wav_data')
                if wav_data:
                    self._voice_end_callback(wav_data)
            
            if result.get('voice_continuing', False) and self._voice_continue_callback:
                pcm_data = result.get('pcm_data')
                if pcm_data:
                    self._voice_continue_callback(pcm_data)
                    
        except Exception as e:
            callback_name = "unknown"
            if result.get('voice_started'):
                callback_name = "voice_start"
            elif result.get('voice_ended'):
                callback_name = "voice_end"
            elif result.get('voice_continuing'):
                callback_name = "voice_continue"
            
            raise CallbackError(callback_name, e)
    
    def reset(self) -> None:
        """Reset VAD state."""
        with self._lock:
            if self.processor:
                self.processor.reset()
            self._total_frames_processed = 0
            self._total_processing_time = 0.0
            self._last_error = None
    
    def cleanup(self) -> None:
        """Clean up resources."""
        with self._lock:
            self.processor = None
            self._is_initialized = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        with self._lock:
            stats = {
                'total_frames_processed': self._total_frames_processed,
                'total_processing_time': self._total_processing_time,
                'average_processing_time_per_frame': (
                    self._total_processing_time / self._total_frames_processed
                    if self._total_frames_processed > 0 else 0.0
                ),
                'is_initialized': self._is_initialized,
                'last_error': str(self._last_error) if self._last_error else None,
                'config': self.config.to_dict()
            }
            
            # Add processor statistics if available
            if self.processor:
                processor_stats = self.processor.get_statistics()
                stats.update(processor_stats)
            
            return stats
    
    def get_config(self) -> VADConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, config: VADConfig) -> None:
        """
        Update configuration and reinitialize if needed.
        
        Args:
            config: New configuration
        """
        with self._lock:
            old_config = self.config
            self.config = config
            
            # Check if reinitializion is needed
            needs_reinit = (
                old_config.sample_rate != config.sample_rate or
                old_config.model_version != config.model_version or
                old_config.model_path != config.model_path
            )
            
            if needs_reinit and self._is_initialized:
                self._initialize_processor()
    
    def is_voice_active(self) -> bool:
        """
        Check if voice is currently active.
        
        Returns:
            True if voice is currently active
        """
        if self.processor:
            return self.processor.is_voice_active
        return False
    
    def get_last_error(self) -> Optional[Exception]:
        """Get the last error that occurred."""
        return self._last_error
    
    def __enter__(self) -> 'VADWrapper':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self) -> None:
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
