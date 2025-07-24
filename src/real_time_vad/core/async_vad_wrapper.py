"""
Asynchronous VAD wrapper for non-blocking operations.
"""

import asyncio
import threading
from typing import Optional, Callable, Awaitable, Union, Any, Dict, List
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .config import VADConfig, SampleRate, SileroModelVersion
from .vad_wrapper import VADWrapper
from .exceptions import VADError, CallbackError


# Type aliases for async callbacks
AsyncVoiceStartCallback = Callable[[], Awaitable[None]]
AsyncVoiceEndCallback = Callable[[bytes], Awaitable[None]]
AsyncVoiceContinueCallback = Callable[[bytes], Awaitable[None]]


class AsyncVADWrapper:
    """
    Asynchronous wrapper for Voice Activity Detection.
    
    This class provides an async interface for VAD operations, allowing
    non-blocking audio processing and callback execution.
    
    Example:
        >>> import asyncio
        >>> 
        >>> async def main():
        ...     vad = AsyncVADWrapper()
        ...     
        ...     async def on_voice_start():
        ...         print("Voice started!")
        ...     
        ...     async def on_voice_end(wav_data: bytes):
        ...         print(f"Voice ended! Got {len(wav_data)} bytes")
        ...     
        ...     vad.set_async_callbacks(
        ...         voice_start_callback=on_voice_start,
        ...         voice_end_callback=on_voice_end
        ...     )
        ...     
        ...     # Process audio asynchronously
        ...     audio_data = np.random.randn(1024).astype(np.float32)
        ...     await vad.process_audio_data_async(audio_data)
        ... 
        >>> asyncio.run(main())
    """
    
    def __init__(
        self,
        config: Optional[VADConfig] = None,
        max_workers: int = 2
    ) -> None:
        """
        Initialize async VAD wrapper.
        
        Args:
            config: VAD configuration. If None, uses default configuration.
            max_workers: Maximum number of worker threads for processing
        """
        self.config = config if config is not None else VADConfig()
        self.vad_wrapper = VADWrapper(self.config)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Async callbacks
        self._async_voice_start_callback: Optional[AsyncVoiceStartCallback] = None
        self._async_voice_end_callback: Optional[AsyncVoiceEndCallback] = None
        self._async_voice_continue_callback: Optional[AsyncVoiceContinueCallback] = None
        
        # Event loop for callback execution
        self._callback_loop: Optional[asyncio.AbstractEventLoop] = None
        self._setup_sync_callbacks()
    
    def _setup_sync_callbacks(self) -> None:
        """Set up synchronous callbacks that dispatch to async callbacks."""
        
        def sync_voice_start_callback():
            if self._async_voice_start_callback:
                asyncio.run_coroutine_threadsafe(
                    self._handle_async_callback(
                        self._async_voice_start_callback, "voice_start"
                    ),
                    self._get_event_loop()
                )
        
        def sync_voice_end_callback(wav_data: bytes):
            if self._async_voice_end_callback:
                asyncio.run_coroutine_threadsafe(
                    self._handle_async_callback(
                        lambda: self._async_voice_end_callback(wav_data), "voice_end"
                    ),
                    self._get_event_loop()
                )
        
        def sync_voice_continue_callback(pcm_data: bytes):
            if self._async_voice_continue_callback:
                asyncio.run_coroutine_threadsafe(
                    self._handle_async_callback(
                        lambda: self._async_voice_continue_callback(pcm_data), "voice_continue"
                    ),
                    self._get_event_loop()
                )
        
        self.vad_wrapper.set_callbacks(
            voice_start_callback=sync_voice_start_callback,
            voice_end_callback=sync_voice_end_callback,
            voice_continue_callback=sync_voice_continue_callback
        )
    
    def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop for callback execution."""
        if self._callback_loop is None or self._callback_loop.is_closed():
            try:
                self._callback_loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, create a new one
                self._callback_loop = asyncio.new_event_loop()
        
        return self._callback_loop
    
    async def _handle_async_callback(
        self,
        callback: Callable[[], Awaitable[None]],
        callback_name: str
    ) -> None:
        """Handle async callback execution with error handling."""
        try:
            await callback()
        except Exception as e:
            raise CallbackError(callback_name, e)
    
    def set_async_callbacks(
        self,
        voice_start_callback: Optional[AsyncVoiceStartCallback] = None,
        voice_end_callback: Optional[AsyncVoiceEndCallback] = None,
        voice_continue_callback: Optional[AsyncVoiceContinueCallback] = None
    ) -> None:
        """
        Set asynchronous callback functions for voice activity events.
        
        Args:
            voice_start_callback: Async callback when voice activity starts
            voice_end_callback: Async callback when voice activity ends with WAV data
            voice_continue_callback: Async callback during voice activity with PCM data
        """
        self._async_voice_start_callback = voice_start_callback
        self._async_voice_end_callback = voice_end_callback
        self._async_voice_continue_callback = voice_continue_callback
    
    async def set_sample_rate_async(self, sample_rate: SampleRate) -> None:
        """
        Asynchronously set the audio sample rate.
        
        Args:
            sample_rate: Target sample rate
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.set_sample_rate,
            sample_rate
        )
    
    async def set_silero_model_async(self, model_version: SileroModelVersion) -> None:
        """
        Asynchronously set the Silero model version.
        
        Args:
            model_version: Model version to use
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.set_silero_model,
            model_version
        )
    
    async def set_thresholds_async(
        self,
        vad_start_probability: float = 0.7,
        vad_end_probability: float = 0.7,
        voice_start_ratio: float = 0.8,
        voice_end_ratio: float = 0.95,
        voice_start_frame_count: int = 10,
        voice_end_frame_count: int = 57
    ) -> None:
        """
        Asynchronously configure VAD detection thresholds.
        
        Args:
            vad_start_probability: Probability threshold for starting VAD detection
            vad_end_probability: Probability threshold for ending VAD detection
            voice_start_ratio: True positive ratio for voice start detection
            voice_end_ratio: False positive ratio for voice end detection
            voice_start_frame_count: Frame count required to confirm voice start
            voice_end_frame_count: Frame count required to confirm voice end
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.set_thresholds,
            vad_start_probability,
            vad_end_probability,
            voice_start_ratio,
            voice_end_ratio,
            voice_start_frame_count,
            voice_end_frame_count
        )
    
    async def process_audio_data_async(
        self,
        audio_data: Union[np.ndarray, List[float]]
    ) -> None:
        """
        Asynchronously process audio data for voice activity detection.
        
        Args:
            audio_data: Audio data as numpy array or list of float values
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.process_audio_data,
            audio_data
        )
    
    async def process_audio_data_with_buffer_async(
        self,
        audio_buffer: np.ndarray,
        count: int
    ) -> None:
        """
        Asynchronously process audio data from a buffer.
        
        Args:
            audio_buffer: Audio buffer (float32 array)
            count: Number of samples to process
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.process_audio_data_with_buffer,
            audio_buffer,
            count
        )
    
    async def reset_async(self) -> None:
        """Asynchronously reset VAD state."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.executor, self.vad_wrapper.reset)
    
    async def get_statistics_async(self) -> Dict[str, Any]:
        """
        Asynchronously get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.get_statistics
        )
    
    async def is_voice_active_async(self) -> bool:
        """
        Asynchronously check if voice is currently active.
        
        Returns:
            True if voice is currently active
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.is_voice_active
        )
    
    async def update_config_async(self, config: VADConfig) -> None:
        """
        Asynchronously update configuration.
        
        Args:
            config: New configuration
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor,
            self.vad_wrapper.update_config,
            config
        )
    
    # Synchronous methods for compatibility
    def set_sample_rate(self, sample_rate: SampleRate) -> None:
        """Set the audio sample rate (synchronous)."""
        self.vad_wrapper.set_sample_rate(sample_rate)
    
    def set_silero_model(self, model_version: SileroModelVersion) -> None:
        """Set the Silero model version (synchronous)."""
        self.vad_wrapper.set_silero_model(model_version)
    
    def set_thresholds(
        self,
        vad_start_probability: float = 0.7,
        vad_end_probability: float = 0.7,
        voice_start_ratio: float = 0.8,
        voice_end_ratio: float = 0.95,
        voice_start_frame_count: int = 10,
        voice_end_frame_count: int = 57
    ) -> None:
        """Configure VAD detection thresholds (synchronous)."""
        self.vad_wrapper.set_thresholds(
            vad_start_probability,
            vad_end_probability,
            voice_start_ratio,
            voice_end_ratio,
            voice_start_frame_count,
            voice_end_frame_count
        )
    
    def process_audio_data(self, audio_data: Union[np.ndarray, List[float]]) -> None:
        """Process audio data for voice activity detection (synchronous)."""
        self.vad_wrapper.process_audio_data(audio_data)
    
    def reset(self) -> None:
        """Reset VAD state (synchronous)."""
        self.vad_wrapper.reset()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics (synchronous)."""
        return self.vad_wrapper.get_statistics()
    
    def is_voice_active(self) -> bool:
        """Check if voice is currently active (synchronous)."""
        return self.vad_wrapper.is_voice_active()
    
    def get_config(self) -> VADConfig:
        """Get current configuration."""
        return self.vad_wrapper.get_config()
    
    def update_config(self, config: VADConfig) -> None:
        """Update configuration (synchronous)."""
        self.vad_wrapper.update_config(config)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.vad_wrapper.cleanup()
        self.executor.shutdown(wait=True)
    
    async def acleanup(self) -> None:
        """Asynchronously clean up resources."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.cleanup)
    
    def __enter__(self) -> 'AsyncVADWrapper':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.cleanup()
    
    async def __aenter__(self) -> 'AsyncVADWrapper':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.acleanup()
    
    def __del__(self) -> None:
        """Destructor."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
