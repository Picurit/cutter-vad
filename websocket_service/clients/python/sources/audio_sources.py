"""
Audio Sources for Python WebSocket VAD Client

This module provides audio source implementations for capturing or reading
audio data and emitting PCM frames for WebSocket transmission.
"""

import abc
import asyncio
import logging
import time
import wave
from typing import AsyncGenerator, Optional, Dict, Any
from pathlib import Path

import numpy as np
import soundfile as sf


class AudioSource(abc.ABC):
    """Abstract base class for audio sources."""
    
    def __init__(self, sample_rate: int, channels: int, frame_duration_ms: int):
        """
        Initialize audio source.
        
        Args:
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels
            frame_duration_ms: Frame duration in milliseconds
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_duration_ms = frame_duration_ms
        
        # Calculate frame size in samples
        self.frame_size_samples = int(sample_rate * (frame_duration_ms / 1000))
        
        # Setup logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abc.abstractmethod
    async def generate_frames(self) -> AsyncGenerator[bytes, None]:
        """
        Generate PCM audio frames as bytes.
        
        Yields:
            PCM frame data as bytes (16-bit signed int, little-endian)
        """
        pass
    
    def samples_to_pcm_bytes(self, samples: np.ndarray) -> bytes:
        """
        Convert audio samples to PCM bytes.
        
        Args:
            samples: Audio samples as float32 array
            
        Returns:
            PCM bytes (16-bit signed int, little-endian)
        """
        # Ensure samples are in range [-1, 1]
        samples = np.clip(samples, -1.0, 1.0)
        
        # Convert to 16-bit signed integers
        pcm_int16 = (samples * 32767).astype(np.int16)
        
        # Convert to bytes (little-endian)
        return pcm_int16.tobytes()


class StoredAudioSource(AudioSource):
    """Audio source that reads from a stored audio file."""
    
    def __init__(self, 
                 file_path: str, 
                 sample_rate: int = 16000, 
                 channels: int = 1, 
                 frame_duration_ms: int = 30,
                 loop: bool = False):
        """
        Initialize stored audio source.
        
        Args:
            file_path: Path to audio file (WAV, MP3, etc.)
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels  
            frame_duration_ms: Frame duration in milliseconds
            loop: Whether to loop the audio file
        """
        super().__init__(sample_rate, channels, frame_duration_ms)
        
        self.file_path = Path(file_path)
        self.loop = loop
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Load and validate audio file
        self._load_audio()
    
    def _load_audio(self):
        """Load audio file and convert to target format."""
        try:
            # Read audio file
            audio_data, original_sample_rate = sf.read(self.file_path, always_2d=True)
            
            self.logger.info(f"Loaded audio: {self.file_path}")
            self.logger.info(f"Original format: {original_sample_rate} Hz, {audio_data.shape[1]} channels, {len(audio_data)} samples")
            
            # Debug: Check original audio levels
            self.logger.debug(f"Original audio: min={audio_data.min():.6f}, max={audio_data.max():.6f}, rms={np.sqrt(np.mean(audio_data**2)):.6f}")
            
            # Convert to mono if needed
            if audio_data.shape[1] > 1 and self.channels == 1:
                audio_data = np.mean(audio_data, axis=1, keepdims=True)
                self.logger.info("Converted stereo to mono")
                # Debug: Check mono audio levels
                self.logger.debug(f"Mono audio: min={audio_data.min():.6f}, max={audio_data.max():.6f}, rms={np.sqrt(np.mean(audio_data**2)):.6f}")
            elif audio_data.shape[1] == 1 and self.channels == 1:
                self.logger.info("Audio is already mono")
            
            # Resample if needed
            if original_sample_rate != self.sample_rate:
                import librosa
                # Properly extract mono data for resampling
                if audio_data.shape[1] == 1:
                    audio_data_mono = audio_data[:, 0]
                else:
                    audio_data_mono = audio_data.flatten()
                    
                resampled = librosa.resample(audio_data_mono, 
                                           orig_sr=original_sample_rate, 
                                           target_sr=self.sample_rate)
                audio_data = resampled.reshape(-1, 1)
                self.logger.info(f"Resampled from {original_sample_rate} Hz to {self.sample_rate} Hz")
                # Debug: Check resampled audio levels
                self.logger.debug(f"Resampled audio: min={audio_data.min():.6f}, max={audio_data.max():.6f}, rms={np.sqrt(np.mean(audio_data**2)):.6f}")
            
            # Store as float32
            self.audio_data = audio_data.astype(np.float32).flatten()
            self.total_samples = len(self.audio_data)
            self.duration_seconds = self.total_samples / self.sample_rate
            
            # Debug: Check final audio levels
            self.logger.debug(f"Final audio: min={self.audio_data.min():.6f}, max={self.audio_data.max():.6f}, rms={np.sqrt(np.mean(self.audio_data**2)):.6f}")
            
            self.logger.info(f"Final audio: {self.sample_rate} Hz, {self.channels} channels, {self.total_samples} samples ({self.duration_seconds:.2f}s)")
            
        except Exception as e:
            self.logger.error(f"Failed to load audio file {self.file_path}: {e}")
            raise
    
    async def generate_frames(self) -> AsyncGenerator[bytes, None]:
        """
        Generate PCM frames from stored audio.
        
        Yields:
            PCM frame data as bytes
        """
        self.logger.info(f"Starting audio stream: {self.frame_size_samples} samples per frame")
        
        current_position = 0
        frame_count = 0
        
        while True:
            # Check if we have enough samples for a complete frame
            remaining_samples = self.total_samples - current_position
            
            if remaining_samples >= self.frame_size_samples:
                # Extract frame
                frame_samples = self.audio_data[current_position:current_position + self.frame_size_samples]
                current_position += self.frame_size_samples
                
            elif remaining_samples > 0:
                # Pad last frame with zeros
                frame_samples = np.zeros(self.frame_size_samples, dtype=np.float32)
                frame_samples[:remaining_samples] = self.audio_data[current_position:]
                current_position = self.total_samples
                
            else:
                # End of file
                if self.loop:
                    self.logger.info("Looping audio file")
                    current_position = 0
                    continue
                else:
                    self.logger.info("Reached end of audio file")
                    break
            
            # Convert to PCM bytes
            pcm_bytes = self.samples_to_pcm_bytes(frame_samples)
            
            # Debug: Log frame audio levels periodically
            if frame_count % 50 == 0:
                frame_rms = np.sqrt(np.mean(frame_samples**2))
                self.logger.debug(f"Frame {frame_count}: min={frame_samples.min():.6f}, max={frame_samples.max():.6f}, rms={frame_rms:.6f}")
            
            frame_count += 1
            if frame_count % 100 == 0:
                progress = (current_position / self.total_samples) * 100
                self.logger.debug(f"Sent {frame_count} frames ({progress:.1f}% complete)")
            
            yield pcm_bytes
            
            # Simulate real-time playback
            await asyncio.sleep(self.frame_duration_ms / 1000)


class RealTimeMicSource(AudioSource):
    """Audio source that captures from microphone in real-time."""
    
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1, 
                 frame_duration_ms: int = 30,
                 device_index: Optional[int] = None):
        """
        Initialize real-time microphone source.
        
        Args:
            sample_rate: Target sample rate in Hz
            channels: Number of audio channels
            frame_duration_ms: Frame duration in milliseconds
            device_index: Audio device index (None for default)
        """
        super().__init__(sample_rate, channels, frame_duration_ms)
        
        self.device_index = device_index
        self.audio_stream = None
        
        # Initialize PyAudio
        try:
            import pyaudio
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()
            
        except ImportError:
            raise RuntimeError("PyAudio is required for real-time microphone capture")
        
        # Auto-detect device if not specified
        if self.device_index is None:
            self.device_index = self._auto_detect_microphone()
        
        self.logger.info(f"Using microphone device: {self.device_index}")
    
    def _auto_detect_microphone(self) -> int:
        """Auto-detect the best available microphone."""
        try:
            # Try to get default input device
            default_device = self.audio.get_default_input_device_info()
            return default_device['index']
        except Exception:
            # Fallback: find any input device
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    return i
            raise RuntimeError("No input audio devices found")
    
    async def generate_frames(self) -> AsyncGenerator[bytes, None]:
        """
        Generate PCM frames from microphone.
        
        Yields:
            PCM frame data as bytes
        """
        self.logger.info("Starting microphone capture")
        
        try:
            # Open audio stream
            self.audio_stream = self.audio.open(
                format=self.pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.frame_size_samples
            )
            
            frame_count = 0
            
            while True:
                # Read audio data
                audio_data = self.audio_stream.read(
                    self.frame_size_samples,
                    exception_on_overflow=False
                )
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Convert to PCM bytes
                pcm_bytes = self.samples_to_pcm_bytes(audio_array)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    self.logger.debug(f"Captured {frame_count} frames from microphone")
                
                yield pcm_bytes
        
        except Exception as e:
            self.logger.error(f"Microphone capture error: {e}")
            raise
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up audio resources."""
        if self.audio_stream:
            self.audio_stream.stop_stream()
            self.audio_stream.close()
            self.audio_stream = None
        
        if hasattr(self, 'audio'):
            self.audio.terminate()
        
        self.logger.info("Microphone resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self._cleanup()


# Example usage and testing
async def test_stored_audio_source():
    """Test function for StoredAudioSource."""
    logging.basicConfig(level=logging.INFO)
    
    # Test with the sample voice file
    audio_file = "../../examples/audios/SampleVoice.wav"
    
    try:
        source = StoredAudioSource(
            file_path=audio_file,
            sample_rate=16000,
            channels=1,
            frame_duration_ms=30
        )
        
        frame_count = 0
        async for frame in source.generate_frames():
            frame_count += 1
            print(f"Frame {frame_count}: {len(frame)} bytes")
            
            # Stop after 10 frames for testing
            if frame_count >= 10:
                break
                
        print(f"Successfully generated {frame_count} frames")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_stored_audio_source())
