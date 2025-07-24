"""
Unit tests for VAD configuration.
"""

import pytest
import tempfile
import os
from pathlib import Path

from real_time_vad.core.config import VADConfig, SampleRate, SileroModelVersion
from real_time_vad.core.exceptions import ConfigurationError


class TestVADConfig:
    """Test VAD configuration functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = VADConfig()
        
        assert config.sample_rate == SampleRate.SAMPLERATE_16
        assert config.model_version == SileroModelVersion.V5
        assert config.vad_start_probability == 0.7
        assert config.vad_end_probability == 0.7
        assert config.voice_start_ratio == 0.8
        assert config.voice_end_ratio == 0.95
        assert config.voice_start_frame_count == 10
        assert config.voice_end_frame_count == 57
        assert config.enable_denoising is True
        assert config.auto_convert_sample_rate is True
        assert config.buffer_size == 512
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_48,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.8,
            vad_end_probability=0.6,
            voice_start_frame_count=5,
            voice_end_frame_count=30
        )
        
        assert config.sample_rate == SampleRate.SAMPLERATE_48
        assert config.model_version == SileroModelVersion.V4
        assert config.vad_start_probability == 0.8
        assert config.vad_end_probability == 0.6
        assert config.voice_start_frame_count == 5
        assert config.voice_end_frame_count == 30
    
    def test_validation_probability_range(self):
        """Test probability validation."""
        # Valid probabilities
        config = VADConfig(vad_start_probability=0.0)
        assert config.vad_start_probability == 0.0
        
        config = VADConfig(vad_start_probability=1.0)
        assert config.vad_start_probability == 1.0
        
        # Invalid probabilities
        with pytest.raises(ValueError):
            VADConfig(vad_start_probability=-0.1)
        
        with pytest.raises(ValueError):
            VADConfig(vad_start_probability=1.1)
        
        with pytest.raises(ValueError):
            VADConfig(vad_end_probability=-0.1)
        
        with pytest.raises(ValueError):
            VADConfig(vad_end_probability=1.1)
    
    def test_validation_frame_counts(self):
        """Test frame count validation."""
        # Valid frame counts
        config = VADConfig(voice_start_frame_count=1)
        assert config.voice_start_frame_count == 1
        
        config = VADConfig(voice_end_frame_count=1)
        assert config.voice_end_frame_count == 1
        
        # Invalid frame counts
        with pytest.raises(ValueError):
            VADConfig(voice_start_frame_count=0)
        
        with pytest.raises(ValueError):
            VADConfig(voice_end_frame_count=0)
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "sample_rate": 16000,
            "model_version": "v4",
            "vad_start_probability": 0.8,
            "voice_start_frame_count": 15,
            "enable_denoising": False
        }
        
        config = VADConfig.from_dict(config_dict)
        
        assert config.sample_rate == SampleRate.SAMPLERATE_16
        assert config.model_version == SileroModelVersion.V4
        assert config.vad_start_probability == 0.8
        assert config.voice_start_frame_count == 15
        assert config.enable_denoising is False
    
    def test_from_dict_string_sample_rate(self):
        """Test sample rate conversion from string."""
        config_dict = {"sample_rate": "16"}
        config = VADConfig.from_dict(config_dict)
        assert config.sample_rate == SampleRate.SAMPLERATE_16
    
    def test_to_dict(self):
        """Test configuration conversion to dictionary."""
        config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_24,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.9
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["sample_rate"] == SampleRate.SAMPLERATE_24
        assert config_dict["model_version"] == SileroModelVersion.V4
        assert config_dict["vad_start_probability"] == 0.9
    
    def test_yaml_roundtrip(self):
        """Test YAML save and load."""
        original_config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_48,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.6,
            voice_start_frame_count=20
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Save to YAML
            original_config.to_yaml(yaml_path)
            assert os.path.exists(yaml_path)
            
            # Load from YAML
            loaded_config = VADConfig.from_yaml(yaml_path)
            
            # Compare
            assert loaded_config.sample_rate == original_config.sample_rate
            assert loaded_config.model_version == original_config.model_version
            assert loaded_config.vad_start_probability == original_config.vad_start_probability
            assert loaded_config.voice_start_frame_count == original_config.voice_start_frame_count
            
        finally:
            if os.path.exists(yaml_path):
                os.unlink(yaml_path)
    
    def test_from_env(self):
        """Test configuration from environment variables."""
        # Set environment variables
        env_vars = {
            "VAD_SAMPLE_RATE": "24000",
            "VAD_MODEL_VERSION": "v4",
            "VAD_START_PROBABILITY": "0.8",
            "VAD_VOICE_START_FRAME_COUNT": "15",
            "VAD_ENABLE_DENOISING": "false"
        }
        
        # Temporarily set environment variables
        original_env = {}
        for key, value in env_vars.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value
        
        try:
            config = VADConfig.from_env()
            
            assert config.sample_rate == SampleRate.SAMPLERATE_24
            assert config.model_version == SileroModelVersion.V4
            assert config.vad_start_probability == 0.8
            assert config.voice_start_frame_count == 15
            assert config.enable_denoising is False
            
        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    del os.environ[key]
                else:
                    os.environ[key] = original_value
    
    def test_get_model_filename(self):
        """Test model filename generation."""
        config_v4 = VADConfig(model_version=SileroModelVersion.V4)
        assert config_v4.get_model_filename() == "silero_vad.onnx"
        
        config_v5 = VADConfig(model_version=SileroModelVersion.V5)
        assert config_v5.get_model_filename() == "silero_vad_v5.onnx"
    
    def test_get_frame_duration_ms(self):
        """Test frame duration calculation."""
        config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_16,
            buffer_size=512
        )
        
        expected_duration = (512 / 16000) * 1000  # 32ms
        assert abs(config.get_frame_duration_ms() - expected_duration) < 0.01
        
        config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_48,
            buffer_size=1024
        )
        
        expected_duration = (1024 / 48000) * 1000  # ~21.33ms
        assert abs(config.get_frame_duration_ms() - expected_duration) < 0.01
    
    def test_str_representation(self):
        """Test string representation."""
        config = VADConfig()
        str_repr = str(config)
        
        assert "VADConfig" in str_repr
        assert "16000Hz" in str_repr
        assert "v5" in str_repr
        assert "0.7" in str_repr


class TestSampleRate:
    """Test SampleRate enum."""
    
    def test_sample_rate_values(self):
        """Test sample rate enum values."""
        assert SampleRate.SAMPLERATE_8 == 8000
        assert SampleRate.SAMPLERATE_16 == 16000
        assert SampleRate.SAMPLERATE_24 == 24000
        assert SampleRate.SAMPLERATE_48 == 48000
    
    def test_sample_rate_from_int(self):
        """Test creating sample rate from integer."""
        assert SampleRate(8000) == SampleRate.SAMPLERATE_8
        assert SampleRate(16000) == SampleRate.SAMPLERATE_16
        assert SampleRate(24000) == SampleRate.SAMPLERATE_24
        assert SampleRate(48000) == SampleRate.SAMPLERATE_48


class TestSileroModelVersion:
    """Test SileroModelVersion enum."""
    
    def test_model_version_values(self):
        """Test model version enum values."""
        assert SileroModelVersion.V4.value == "v4"
        assert SileroModelVersion.V5.value == "v5"
    
    def test_model_version_from_string(self):
        """Test creating model version from string."""
        assert SileroModelVersion("v4") == SileroModelVersion.V4
        assert SileroModelVersion("v5") == SileroModelVersion.V5
