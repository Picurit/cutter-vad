"""
WAV file writer utility for the VAD library.
"""

import struct
import io
from typing import Optional, Union
import numpy as np

from ..core.exceptions import AudioProcessingError


class WAVWriter:
    """Utility class for creating WAV files from audio data."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        bit_depth: int = 16,
        channels: int = 1
    ) -> None:
        """
        Initialize WAV writer.
        
        Args:
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth (16 or 32)
            channels: Number of channels (1 for mono, 2 for stereo)
        """
        self.sample_rate = sample_rate
        self.bit_depth = bit_depth
        self.channels = channels
        
        if bit_depth not in [16, 32]:
            raise ValueError(f"Unsupported bit depth: {bit_depth}. Must be 16 or 32.")
        
        if channels not in [1, 2]:
            raise ValueError(f"Unsupported channel count: {channels}. Must be 1 or 2.")
    
    def write_wav_data(self, audio_data: np.ndarray) -> bytes:
        """
        Convert audio data to WAV format bytes.
        
        Args:
            audio_data: Audio data as numpy array (float32, values in [-1, 1])
            
        Returns:
            WAV file data as bytes
            
        Raises:
            AudioProcessingError: If conversion fails
        """
        try:
            # Validate input
            if not isinstance(audio_data, np.ndarray):
                raise ValueError("Audio data must be a numpy array")
            
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Ensure mono if configured for mono
            if self.channels == 1 and audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Convert to appropriate integer format
            if self.bit_depth == 16:
                # Convert to 16-bit signed integers
                audio_int = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
                bytes_per_sample = 2
                format_code = 1  # PCM
            else:  # 32-bit
                # Convert to 32-bit signed integers
                audio_int = np.clip(audio_data * 2147483647, -2147483648, 2147483647).astype(np.int32)
                bytes_per_sample = 4
                format_code = 1  # PCM
            
            # Calculate sizes
            data_size = len(audio_int) * bytes_per_sample
            file_size = 36 + data_size
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            # Write WAV header
            self._write_wav_header(
                wav_buffer,
                file_size,
                data_size,
                format_code,
                bytes_per_sample
            )
            
            # Write audio data
            wav_buffer.write(audio_int.tobytes())
            
            return wav_buffer.getvalue()
            
        except Exception as e:
            raise AudioProcessingError(f"Failed to create WAV data: {str(e)}")
    
    def _write_wav_header(
        self,
        buffer: io.BytesIO,
        file_size: int,
        data_size: int,
        format_code: int,
        bytes_per_sample: int
    ) -> None:
        """Write WAV file header to buffer."""
        
        # RIFF header
        buffer.write(b'RIFF')
        buffer.write(struct.pack('<I', file_size))  # File size - 8
        buffer.write(b'WAVE')
        
        # Format chunk
        buffer.write(b'fmt ')
        buffer.write(struct.pack('<I', 16))  # Format chunk size
        buffer.write(struct.pack('<H', format_code))  # Audio format (1 = PCM)
        buffer.write(struct.pack('<H', self.channels))  # Number of channels
        buffer.write(struct.pack('<I', self.sample_rate))  # Sample rate
        
        # Byte rate (sample_rate * channels * bytes_per_sample)
        byte_rate = self.sample_rate * self.channels * bytes_per_sample
        buffer.write(struct.pack('<I', byte_rate))
        
        # Block align (channels * bytes_per_sample)
        block_align = self.channels * bytes_per_sample
        buffer.write(struct.pack('<H', block_align))
        
        # Bits per sample
        buffer.write(struct.pack('<H', self.bit_depth))
        
        # Data chunk
        buffer.write(b'data')
        buffer.write(struct.pack('<I', data_size))
    
    def write_wav_file(self, filename: str, audio_data: np.ndarray) -> None:
        """
        Write audio data to WAV file.
        
        Args:
            filename: Output filename
            audio_data: Audio data as numpy array
        """
        try:
            wav_data = self.write_wav_data(audio_data)
            with open(filename, 'wb') as f:
                f.write(wav_data)
        except Exception as e:
            raise AudioProcessingError(f"Failed to write WAV file {filename}: {str(e)}")
    
    @staticmethod
    def create_wav_header(
        sample_rate: int,
        bit_depth: int,
        channels: int,
        data_size: int
    ) -> bytes:
        """
        Create WAV header bytes.
        
        Args:
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth (16 or 32)
            channels: Number of channels
            data_size: Size of audio data in bytes
            
        Returns:
            WAV header as bytes
        """
        bytes_per_sample = bit_depth // 8
        file_size = 36 + data_size
        
        header = io.BytesIO()
        
        # RIFF header
        header.write(b'RIFF')
        header.write(struct.pack('<I', file_size))
        header.write(b'WAVE')
        
        # Format chunk
        header.write(b'fmt ')
        header.write(struct.pack('<I', 16))  # Format chunk size
        header.write(struct.pack('<H', 1))   # Audio format (PCM)
        header.write(struct.pack('<H', channels))
        header.write(struct.pack('<I', sample_rate))
        
        # Byte rate
        byte_rate = sample_rate * channels * bytes_per_sample
        header.write(struct.pack('<I', byte_rate))
        
        # Block align
        block_align = channels * bytes_per_sample
        header.write(struct.pack('<H', block_align))
        
        # Bits per sample
        header.write(struct.pack('<H', bit_depth))
        
        # Data chunk header
        header.write(b'data')
        header.write(struct.pack('<I', data_size))
        
        return header.getvalue()
    
    @staticmethod
    def validate_wav_parameters(
        sample_rate: int,
        bit_depth: int,
        channels: int
    ) -> None:
        """
        Validate WAV parameters.
        
        Args:
            sample_rate: Sample rate in Hz
            bit_depth: Bit depth
            channels: Number of channels
            
        Raises:
            ValueError: If parameters are invalid
        """
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        
        if bit_depth not in [16, 32]:
            raise ValueError(f"Unsupported bit depth: {bit_depth}")
        
        if channels not in [1, 2]:
            raise ValueError(f"Unsupported channel count: {channels}")
    
    def get_format_info(self) -> dict:
        """
        Get format information.
        
        Returns:
            Dictionary with format information
        """
        return {
            'sample_rate': self.sample_rate,
            'bit_depth': self.bit_depth,
            'channels': self.channels,
            'bytes_per_sample': self.bit_depth // 8,
            'format': 'PCM'
        }
