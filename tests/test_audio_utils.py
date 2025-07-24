"""
Unit tests for audio utilities.
"""

import pytest
import numpy as np
import tempfile
import os

from real_time_vad.utils.audio import AudioUtils
from real_time_vad.core.exceptions import AudioProcessingError


class TestAudioUtils:
    """Test audio utility functions."""
    
    def test_validate_audio_data_valid(self):
        """Test validation of valid audio data."""
        # Valid audio data should not raise
        audio_data = np.array([0.1, -0.2, 0.5, -0.8], dtype=np.float32)
        AudioUtils.validate_audio_data(audio_data)  # Should not raise
    
    def test_validate_audio_data_invalid(self):
        """Test validation of invalid audio data."""
        # Empty array
        with pytest.raises(AudioProcessingError):
            AudioUtils.validate_audio_data(np.array([]))
        
        # Non-numpy array
        with pytest.raises(AudioProcessingError):
            AudioUtils.validate_audio_data([0.1, 0.2, 0.3])
        
        # Array with NaN values
        with pytest.raises(AudioProcessingError):
            AudioUtils.validate_audio_data(np.array([0.1, np.nan, 0.3]))
        
        # Array with infinite values
        with pytest.raises(AudioProcessingError):
            AudioUtils.validate_audio_data(np.array([0.1, np.inf, 0.3]))
    
    def test_convert_to_mono_already_mono(self):
        """Test mono conversion with already mono audio."""
        mono_audio = np.array([0.1, 0.2, 0.3, 0.4])
        result = AudioUtils.convert_to_mono(mono_audio)
        np.testing.assert_array_equal(result, mono_audio)
    
    def test_convert_to_mono_stereo(self):
        """Test mono conversion with stereo audio."""
        stereo_audio = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # Shape: (3, 2)
        result = AudioUtils.convert_to_mono(stereo_audio)
        
        expected = np.array([0.15, 0.35, 0.55])  # Average of channels
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_convert_to_mono_invalid_shape(self):
        """Test mono conversion with invalid audio shape."""
        invalid_audio = np.array([[[0.1, 0.2], [0.3, 0.4]]])  # 3D array
        with pytest.raises(AudioProcessingError):
            AudioUtils.convert_to_mono(invalid_audio)
    
    def test_resample_audio_same_rate(self):
        """Test resampling with same source and target rates."""
        audio_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        result = AudioUtils.resample_audio(audio_data, 16000, 16000)
        np.testing.assert_array_equal(result, audio_data)
    
    def test_resample_audio_different_rates(self):
        """Test resampling with different rates."""
        # Create a simple sine wave
        sample_rate = 1000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 50 * t).astype(np.float32)  # 50 Hz sine wave
        
        # Resample to half the rate
        target_rate = 500
        result = AudioUtils.resample_audio(audio_data, sample_rate, target_rate)
        
        # Result should have half the length
        expected_length = len(audio_data) // 2
        assert abs(len(result) - expected_length) <= 1  # Allow for rounding
        assert result.dtype == np.float32
    
    def test_normalize_audio(self):
        """Test audio normalization."""
        # Test with audio that needs normalization
        audio_data = np.array([0.1, -0.2, 0.05, -0.15])
        result = AudioUtils.normalize_audio(audio_data, target_level=0.8)
        
        # Maximum absolute value should be close to target level
        max_val = np.max(np.abs(result))
        assert abs(max_val - 0.8) < 0.01
    
    def test_normalize_audio_zero_amplitude(self):
        """Test normalization with zero amplitude audio."""
        audio_data = np.array([0.0, 0.0, 0.0, 0.0])
        result = AudioUtils.normalize_audio(audio_data)
        np.testing.assert_array_equal(result, audio_data)
    
    def test_denoise_audio(self):
        """Test audio denoising."""
        # Create audio with signal and noise
        signal = np.array([0.5, -0.4, 0.6, -0.3])
        noise = np.array([0.005, -0.008, 0.003, -0.007])
        audio_data = signal + noise
        
        result = AudioUtils.denoise_audio(audio_data, noise_threshold=0.01)
        
        # Signal should be preserved, noise should be removed
        np.testing.assert_array_almost_equal(result, signal, decimal=3)
    
    def test_detect_clipping(self):
        """Test clipping detection."""
        # Audio without clipping
        normal_audio = np.array([0.1, -0.2, 0.5, -0.8])
        assert not AudioUtils.detect_clipping(normal_audio)
        
        # Audio with clipping
        clipped_audio = np.array([0.1, -0.2, 0.98, -0.8])
        assert AudioUtils.detect_clipping(clipped_audio, threshold=0.95)
    
    def test_calculate_rms(self):
        """Test RMS calculation."""
        audio_data = np.array([0.1, -0.2, 0.3, -0.4])
        rms = AudioUtils.calculate_rms(audio_data)
        
        expected_rms = np.sqrt(np.mean(audio_data ** 2))
        assert abs(rms - expected_rms) < 1e-6
    
    def test_calculate_energy(self):
        """Test energy calculation."""
        audio_data = np.array([0.1, -0.2, 0.3, -0.4])
        energy = AudioUtils.calculate_energy(audio_data)
        
        expected_energy = np.sum(audio_data ** 2)
        assert abs(energy - expected_energy) < 1e-6
    
    def test_split_into_frames(self):
        """Test frame splitting."""
        audio_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        frame_size = 4
        hop_size = 2
        
        frames = AudioUtils.split_into_frames(audio_data, frame_size, hop_size)
        
        expected_frames = np.array([
            [1, 2, 3, 4],
            [3, 4, 5, 6],
            [5, 6, 7, 8],
            [7, 8, 9, 10]
        ])
        
        np.testing.assert_array_equal(frames, expected_frames)
    
    def test_split_into_frames_default_hop(self):
        """Test frame splitting with default hop size."""
        audio_data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        frame_size = 4
        
        frames = AudioUtils.split_into_frames(audio_data, frame_size)
        
        expected_frames = np.array([
            [1, 2, 3, 4],
            [3, 4, 5, 6],
            [5, 6, 7, 8]
        ])
        
        np.testing.assert_array_equal(frames, expected_frames)
    
    def test_apply_window(self):
        """Test windowing functions."""
        audio_data = np.ones(8, dtype=np.float32)
        
        # Test Hanning window
        windowed = AudioUtils.apply_window(audio_data, "hanning")
        assert len(windowed) == len(audio_data)
        assert windowed[0] < audio_data[0]  # Window should reduce amplitude at edges
        assert windowed[-1] < audio_data[-1]
        
        # Test Hamming window
        windowed_hamming = AudioUtils.apply_window(audio_data, "hamming")
        assert len(windowed_hamming) == len(audio_data)
        
        # Test Blackman window
        windowed_blackman = AudioUtils.apply_window(audio_data, "blackman")
        assert len(windowed_blackman) == len(audio_data)
        
        # Test invalid window type
        with pytest.raises(AudioProcessingError):
            AudioUtils.apply_window(audio_data, "invalid_window")
    
    def test_pcm_to_float32_16bit(self):
        """Test PCM to float32 conversion for 16-bit data."""
        # Create 16-bit PCM data
        pcm_data = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        pcm_bytes = pcm_data.tobytes()
        
        result = AudioUtils.pcm_to_float32(pcm_bytes, bit_depth=16)
        
        expected = pcm_data.astype(np.float32) / 32768.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_pcm_to_float32_32bit(self):
        """Test PCM to float32 conversion for 32-bit data."""
        # Create 32-bit PCM data
        pcm_data = np.array([0, 1073741824, -1073741824], dtype=np.int32)
        pcm_bytes = pcm_data.tobytes()
        
        result = AudioUtils.pcm_to_float32(pcm_bytes, bit_depth=32)
        
        expected = pcm_data.astype(np.float32) / 2147483648.0
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_float32_to_pcm_16bit(self):
        """Test float32 to PCM conversion for 16-bit output."""
        audio_data = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        pcm_bytes = AudioUtils.float32_to_pcm(audio_data, bit_depth=16)
        
        # Convert back to verify
        result = AudioUtils.pcm_to_float32(pcm_bytes, bit_depth=16)
        
        # Should be close to original (within quantization error)
        np.testing.assert_array_almost_equal(result, audio_data, decimal=4)
    
    def test_float32_to_pcm_32bit(self):
        """Test float32 to PCM conversion for 32-bit output."""
        audio_data = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        
        pcm_bytes = AudioUtils.float32_to_pcm(audio_data, bit_depth=32)
        
        # Convert back to verify
        result = AudioUtils.pcm_to_float32(pcm_bytes, bit_depth=32)
        
        # Should be very close to original
        np.testing.assert_array_almost_equal(result, audio_data, decimal=6)
    
    def test_save_and_load_audio_file(self):
        """Test audio file save and load operations."""
        # Create test audio data
        sample_rate = 16000
        duration = 0.1  # 100ms
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)  # 440 Hz tone
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            # Save audio file
            AudioUtils.save_audio_file(temp_path, audio_data, sample_rate, bit_depth=16)
            assert os.path.exists(temp_path)
            
            # Load audio file
            loaded_audio, loaded_rate = AudioUtils.load_audio_file(temp_path)
            
            # Check sample rate
            assert loaded_rate == sample_rate
            
            # Check audio data (within quantization error for 16-bit)
            assert len(loaded_audio) == len(audio_data)
            np.testing.assert_array_almost_equal(loaded_audio, audio_data, decimal=3)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
