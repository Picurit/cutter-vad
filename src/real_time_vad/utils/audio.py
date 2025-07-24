"""
Audio processing utilities for the VAD library.
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import signal
from scipy.io import wavfile
import io

from ..core.config import SampleRate
from ..core.exceptions import AudioProcessingError


class AudioUtils:
    """Utility class for audio processing operations."""
    
    @staticmethod
    def resample_audio(
        audio_data: np.ndarray,
        original_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio data to target sample rate.
        
        Args:
            audio_data: Input audio data as numpy array
            original_rate: Original sample rate in Hz
            target_rate: Target sample rate in Hz
            
        Returns:
            Resampled audio data
            
        Raises:
            AudioProcessingError: If resampling fails
        """
        try:
            if original_rate == target_rate:
                return audio_data
            
            # Calculate resampling ratio
            ratio = target_rate / original_rate
            
            # Use scipy's resample function for high-quality resampling
            resampled_length = int(len(audio_data) * ratio)
            resampled_audio = signal.resample(audio_data, resampled_length)
            
            return resampled_audio.astype(np.float32)
            
        except Exception as e:
            raise AudioProcessingError(
                f"Failed to resample audio from {original_rate}Hz to {target_rate}Hz: {str(e)}",
                f"Input shape: {audio_data.shape}, dtype: {audio_data.dtype}"
            )
    
    @staticmethod
    def normalize_audio(audio_data: np.ndarray, target_level: float = 0.9) -> np.ndarray:
        """
        Normalize audio data to target level.
        
        Args:
            audio_data: Input audio data
            target_level: Target normalization level (0.0 to 1.0)
            
        Returns:
            Normalized audio data
        """
        try:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                return audio_data * (target_level / max_val)
            return audio_data
        except Exception as e:
            raise AudioProcessingError(f"Failed to normalize audio: {str(e)}")
    
    @staticmethod
    def apply_window(audio_data: np.ndarray, window_type: str = 'hanning') -> np.ndarray:
        """
        Apply windowing function to audio data.
        
        Args:
            audio_data: Input audio data
            window_type: Type of window ('hanning', 'hamming', 'blackman')
            
        Returns:
            Windowed audio data
        """
        try:
            if window_type == 'hanning':
                window = np.hanning(len(audio_data))
            elif window_type == 'hamming':
                window = np.hamming(len(audio_data))
            elif window_type == 'blackman':
                window = np.blackman(len(audio_data))
            else:
                raise ValueError(f"Unsupported window type: {window_type}")
            
            return audio_data * window
        except Exception as e:
            raise AudioProcessingError(f"Failed to apply window: {str(e)}")
    
    @staticmethod
    def denoise_audio(audio_data: np.ndarray, noise_threshold: float = 0.01) -> np.ndarray:
        """
        Simple denoising by thresholding low-amplitude samples.
        
        Args:
            audio_data: Input audio data
            noise_threshold: Threshold below which samples are considered noise
            
        Returns:
            Denoised audio data
        """
        try:
            # Simple thresholding-based denoising
            mask = np.abs(audio_data) > noise_threshold
            denoised = np.where(mask, audio_data, 0.0)
            return denoised
        except Exception as e:
            raise AudioProcessingError(f"Failed to denoise audio: {str(e)}")
    
    @staticmethod
    def detect_clipping(audio_data: np.ndarray, threshold: float = 0.95) -> bool:
        """
        Detect if audio data contains clipping.
        
        Args:
            audio_data: Input audio data
            threshold: Clipping detection threshold
            
        Returns:
            True if clipping is detected
        """
        return np.any(np.abs(audio_data) >= threshold)
    
    @staticmethod
    def calculate_rms(audio_data: np.ndarray) -> float:
        """
        Calculate Root Mean Square (RMS) of audio data.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            RMS value
        """
        return np.sqrt(np.mean(audio_data ** 2))
    
    @staticmethod
    def calculate_energy(audio_data: np.ndarray) -> float:
        """
        Calculate energy of audio data.
        
        Args:
            audio_data: Input audio data
            
        Returns:
            Energy value
        """
        return np.sum(audio_data ** 2)
    
    @staticmethod
    def split_into_frames(
        audio_data: np.ndarray,
        frame_size: int,
        hop_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Split audio data into overlapping frames.
        
        Args:
            audio_data: Input audio data
            frame_size: Size of each frame
            hop_size: Hop size between frames (default: frame_size // 2)
            
        Returns:
            Array of frames with shape (num_frames, frame_size)
        """
        if hop_size is None:
            hop_size = frame_size // 2
        
        num_frames = (len(audio_data) - frame_size) // hop_size + 1
        frames = np.zeros((num_frames, frame_size), dtype=audio_data.dtype)
        
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = audio_data[start:start + frame_size]
        
        return frames
    
    @staticmethod
    def convert_to_mono(audio_data: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono by averaging channels.
        
        Args:
            audio_data: Input audio data (can be mono or stereo)
            
        Returns:
            Mono audio data
        """
        if audio_data.ndim == 1:
            return audio_data
        elif audio_data.ndim == 2:
            return np.mean(audio_data, axis=1)
        else:
            raise AudioProcessingError(f"Unsupported audio shape: {audio_data.shape}")
    
    @staticmethod
    def validate_audio_data(audio_data: np.ndarray) -> None:
        """
        Validate audio data format and content.
        
        Args:
            audio_data: Audio data to validate
            
        Raises:
            AudioProcessingError: If audio data is invalid
        """
        if not isinstance(audio_data, np.ndarray):
            raise AudioProcessingError("Audio data must be a numpy array")
        
        if audio_data.size == 0:
            raise AudioProcessingError("Audio data is empty")
        
        if not np.isfinite(audio_data).all():
            raise AudioProcessingError("Audio data contains infinite or NaN values")
        
        if audio_data.ndim > 2:
            raise AudioProcessingError(f"Audio data has too many dimensions: {audio_data.ndim}")
    
    @staticmethod
    def load_audio_file(file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return data with sample rate.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            sample_rate, audio_data = wavfile.read(file_path)
            
            # Convert to float32 and normalize if needed
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Convert to mono if stereo
            audio_data = AudioUtils.convert_to_mono(audio_data)
            
            return audio_data, sample_rate
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio file {file_path}: {str(e)}")
    
    @staticmethod
    def save_audio_file(
        file_path: str,
        audio_data: np.ndarray,
        sample_rate: int,
        bit_depth: int = 16
    ) -> None:
        """
        Save audio data to file.
        
        Args:
            file_path: Output file path
            audio_data: Audio data to save
            sample_rate: Sample rate
            bit_depth: Bit depth (16 or 32)
        """
        try:
            # Convert float32 to appropriate integer format
            if bit_depth == 16:
                audio_int = (audio_data * 32767).astype(np.int16)
            elif bit_depth == 32:
                audio_int = (audio_data * 2147483647).astype(np.int32)
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
            
            wavfile.write(file_path, sample_rate, audio_int)
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to save audio file {file_path}: {str(e)}")
    
    @staticmethod
    def pcm_to_float32(pcm_data: bytes, bit_depth: int = 16) -> np.ndarray:
        """
        Convert PCM bytes to float32 numpy array.
        
        Args:
            pcm_data: Raw PCM data as bytes
            bit_depth: Bit depth of PCM data (16 or 32)
            
        Returns:
            Float32 audio array
        """
        try:
            if bit_depth == 16:
                audio_int = np.frombuffer(pcm_data, dtype=np.int16)
                return audio_int.astype(np.float32) / 32768.0
            elif bit_depth == 32:
                audio_int = np.frombuffer(pcm_data, dtype=np.int32)
                return audio_int.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
                
        except Exception as e:
            raise AudioProcessingError(f"Failed to convert PCM to float32: {str(e)}")
    
    @staticmethod
    def float32_to_pcm(audio_data: np.ndarray, bit_depth: int = 16) -> bytes:
        """
        Convert float32 numpy array to PCM bytes.
        
        Args:
            audio_data: Float32 audio array
            bit_depth: Target bit depth (16 or 32)
            
        Returns:
            PCM data as bytes
        """
        try:
            if bit_depth == 16:
                audio_int = (audio_data * 32767).astype(np.int16)
            elif bit_depth == 32:
                audio_int = (audio_data * 2147483647).astype(np.int32)
            else:
                raise ValueError(f"Unsupported bit depth: {bit_depth}")
            
            return audio_int.tobytes()
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to convert float32 to PCM: {str(e)}")
