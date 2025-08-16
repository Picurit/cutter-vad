"""
Unit tests for VAD configuration.

This module contains comprehensive unit tests for the VAD configuration system,
including the VADConfig class and related enums.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from typing import Dict, Any
import yaml

from real_time_vad.core.config import VADConfig, SampleRate, SileroModelVersion
from real_time_vad.core.exceptions import ConfigurationError


class TestVADConfig:
    """Test VAD configuration functionality."""

    def test_default_config(self):
        """Test that default configuration values are correctly set."""
        config = VADConfig()
        
        # Model configuration
        assert config.sample_rate == SampleRate.SAMPLERATE_16
        assert config.model_version == SileroModelVersion.V5
        assert config.model_path is None
        
        # VAD thresholds
        assert config.vad_start_probability == 0.7
        assert config.vad_end_probability == 0.7
        
        # Voice detection ratios
        assert config.voice_start_ratio == 0.8
        assert config.voice_end_ratio == 0.95
        
        # Frame counts
        assert config.voice_start_frame_count == 10
        assert config.voice_end_frame_count == 50
        
        # Processing options
        assert config.enable_denoising is True
        assert config.auto_convert_sample_rate is True
        assert config.buffer_size == 512
        
        # Output options
        assert config.output_wav_sample_rate == 16000
        assert config.output_wav_bit_depth == 16

    @pytest.mark.parametrize("sample_rate,model_version,start_prob,end_prob", [
        (SampleRate.SAMPLERATE_8, SileroModelVersion.V4, 0.5, 0.6),
        (SampleRate.SAMPLERATE_24, SileroModelVersion.V5, 0.8, 0.9),
        (SampleRate.SAMPLERATE_48, SileroModelVersion.V4, 0.3, 0.2),
    ])
    def test_custom_config_values(self, sample_rate, model_version, start_prob, end_prob):
        """Test custom configuration values with parametrized inputs."""
        config = VADConfig(
            sample_rate=sample_rate,
            model_version=model_version,
            vad_start_probability=start_prob,
            vad_end_probability=end_prob,
            voice_start_frame_count=15,
            voice_end_frame_count=30
        )
        
        assert config.sample_rate == sample_rate
        assert config.model_version == model_version
        assert config.vad_start_probability == start_prob
        assert config.vad_end_probability == end_prob
        assert config.voice_start_frame_count == 15
        assert config.voice_end_frame_count == 30

    @pytest.mark.parametrize("probability,field_name", [
        (-0.1, "vad_start_probability"),
        (1.1, "vad_start_probability"),
        (-0.01, "vad_end_probability"),
        (1.01, "vad_end_probability"),
        (-0.5, "voice_start_ratio"),
        (1.5, "voice_start_ratio"),
        (-0.1, "voice_end_ratio"),
        (1.1, "voice_end_ratio"),
    ])
    def test_invalid_probability_values(self, probability, field_name):
        """Test that invalid probability values raise ValueError."""
        kwargs = {field_name: probability}
        with pytest.raises(ValueError):
            VADConfig(**kwargs)

    @pytest.mark.parametrize("probability,field_name", [
        (0.0, "vad_start_probability"),
        (1.0, "vad_start_probability"),
        (0.0, "vad_end_probability"),
        (1.0, "vad_end_probability"),
        (0.5, "voice_start_ratio"),
        (0.95, "voice_end_ratio"),
    ])
    def test_valid_probability_boundary_values(self, probability, field_name):
        """Test that boundary probability values are accepted."""
        kwargs = {field_name: probability}
        config = VADConfig(**kwargs)
        assert getattr(config, field_name) == probability

    @pytest.mark.parametrize("frame_count,field_name", [
        (0, "voice_start_frame_count"),
        (-1, "voice_start_frame_count"),
        (0, "voice_end_frame_count"),
        (-5, "voice_end_frame_count"),
    ])
    def test_invalid_frame_count_values(self, frame_count, field_name):
        """Test that invalid frame count values raise ValueError."""
        kwargs = {field_name: frame_count}
        with pytest.raises(ValueError):
            VADConfig(**kwargs)

    @pytest.mark.parametrize("buffer_size", [255, 2049, 100])
    def test_invalid_buffer_size_values(self, buffer_size):
        """Test that invalid buffer size values raise ValueError."""
        with pytest.raises(ValueError):
            VADConfig(buffer_size=buffer_size)

    @pytest.mark.parametrize("buffer_size", [256, 512, 1024, 2048])
    def test_valid_buffer_size_values(self, buffer_size):
        """Test that valid buffer size values are accepted."""
        config = VADConfig(buffer_size=buffer_size)
        assert config.buffer_size == buffer_size

    def test_model_path_validation_none(self):
        """Test that None model path is valid."""
        config = VADConfig(model_path=None)
        assert config.model_path is None

    def test_model_path_validation_nonexistent_path(self):
        """Test that nonexistent model path raises ValueError."""
        fake_path = Path("/nonexistent/path")
        with pytest.raises(ValueError, match="Model path does not exist"):
            VADConfig(model_path=fake_path)

    def test_model_path_validation_file_instead_of_directory(self):
        """Test that file path instead of directory raises ValueError."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            file_path = Path(tmp_file.name)
            with pytest.raises(ValueError, match="Model path must be a directory"):
                VADConfig(model_path=file_path)

    def test_model_path_validation_valid_directory(self):
        """Test that valid directory path is accepted."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir_path = Path(tmp_dir)
            config = VADConfig(model_path=dir_path)
            assert config.model_path == dir_path

    def test_pydantic_config_settings(self):
        """Test Pydantic configuration settings."""
        # Test that extra fields are forbidden
        with pytest.raises(ValueError):
            VADConfig(invalid_field="value")

    @pytest.mark.parametrize("config_dict,expected_sample_rate,expected_model", [
        ({"sample_rate": 16000}, SampleRate.SAMPLERATE_16, SileroModelVersion.V5),
        ({"sample_rate": "8"}, SampleRate.SAMPLERATE_8, SileroModelVersion.V5),
        ({"model_version": "v4"}, SampleRate.SAMPLERATE_16, SileroModelVersion.V4),
        ({"model_version": "v5"}, SampleRate.SAMPLERATE_16, SileroModelVersion.V5),
    ])
    def test_from_dict_enum_conversion(self, config_dict, expected_sample_rate, expected_model):
        """Test enum conversion in from_dict method."""
        config = VADConfig.from_dict(config_dict)
        assert config.sample_rate == expected_sample_rate
        assert config.model_version == expected_model

    def test_from_dict_comprehensive(self):
        """Test comprehensive dictionary conversion."""
        config_dict = {
            "sample_rate": 24000,
            "model_version": "v4",
            "model_path": None,
            "vad_start_probability": 0.8,
            "vad_end_probability": 0.6,
            "voice_start_ratio": 0.75,
            "voice_end_ratio": 0.85,
            "voice_start_frame_count": 15,
            "voice_end_frame_count": 40,
            "enable_denoising": False,
            "auto_convert_sample_rate": False,
            "buffer_size": 1024,
            "output_wav_sample_rate": 22050,
            "output_wav_bit_depth": 24
        }
        
        config = VADConfig.from_dict(config_dict)
        
        assert config.sample_rate == SampleRate.SAMPLERATE_24
        assert config.model_version == SileroModelVersion.V4
        assert config.model_path is None
        assert config.vad_start_probability == 0.8
        assert config.vad_end_probability == 0.6
        assert config.voice_start_ratio == 0.75
        assert config.voice_end_ratio == 0.85
        assert config.voice_start_frame_count == 15
        assert config.voice_end_frame_count == 40
        assert config.enable_denoising is False
        assert config.auto_convert_sample_rate is False
        assert config.buffer_size == 1024
        assert config.output_wav_sample_rate == 22050
        assert config.output_wav_bit_depth == 24

    def test_from_dict_with_model_path(self):
        """Test from_dict with model path conversion."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_dict = {"model_path": tmp_dir}
            config = VADConfig.from_dict(config_dict)
            assert config.model_path == Path(tmp_dir)

    def test_to_dict_conversion(self):
        """Test configuration conversion to dictionary."""
        config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_48,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.9,
            buffer_size=1024
        )
        
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["sample_rate"] == SampleRate.SAMPLERATE_48
        assert config_dict["model_version"] == SileroModelVersion.V4
        assert config_dict["vad_start_probability"] == 0.9
        assert config_dict["buffer_size"] == 1024

    def test_yaml_save_and_load_roundtrip(self):
        """Test complete YAML save and load roundtrip."""
        original_config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_48,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.6,
            vad_end_probability=0.5,
            voice_start_frame_count=20,
            voice_end_frame_count=35,
            enable_denoising=False,
            buffer_size=1024
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml_path = f.name
        
        try:
            # Save to YAML
            original_config.to_yaml(yaml_path)
            assert Path(yaml_path).exists()
            
            # Verify YAML content
            with open(yaml_path, 'r') as f:
                yaml_content = yaml.safe_load(f)
            assert isinstance(yaml_content, dict)
            
            # Load from YAML
            loaded_config = VADConfig.from_yaml(yaml_path)
            
            # Verify all fields match
            assert loaded_config.sample_rate == original_config.sample_rate
            assert loaded_config.model_version == original_config.model_version
            assert loaded_config.vad_start_probability == original_config.vad_start_probability
            assert loaded_config.vad_end_probability == original_config.vad_end_probability
            assert loaded_config.voice_start_frame_count == original_config.voice_start_frame_count
            assert loaded_config.voice_end_frame_count == original_config.voice_end_frame_count
            assert loaded_config.enable_denoising == original_config.enable_denoising
            assert loaded_config.buffer_size == original_config.buffer_size
            
        finally:
            if Path(yaml_path).exists():
                Path(yaml_path).unlink()

    def test_yaml_load_nonexistent_file(self):
        """Test loading YAML from nonexistent file raises FileNotFoundError."""
        fake_path = Path("/nonexistent/config.yaml")
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            VADConfig.from_yaml(fake_path)

    def test_yaml_save_creates_directory(self):
        """Test that YAML save creates parent directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "subdir" / "config.yaml"
            config = VADConfig()
            
            config.to_yaml(yaml_path)
            
            assert yaml_path.exists()
            assert yaml_path.parent.exists()

    @pytest.fixture
    def env_cleanup(self):
        """Fixture to clean up environment variables after test."""
        original_env = {}
        yield original_env
        # Cleanup after test
        for key in list(os.environ.keys()):
            if key.startswith("VAD_") or key.startswith("CUSTOM_"):
                if key in original_env:
                    if original_env[key] is not None:
                        os.environ[key] = original_env[key]
                    else:
                        os.environ.pop(key, None)
                else:
                    os.environ.pop(key, None)

    def test_from_env_comprehensive(self, env_cleanup):
        """Test comprehensive environment variable loading."""
        env_vars = {
            "VAD_SAMPLE_RATE": "24000",
            "VAD_MODEL_VERSION": "v4",
            "VAD_MODEL_PATH": "/test/path",
            "VAD_START_PROBABILITY": "0.8",
            "VAD_END_PROBABILITY": "0.6",
            "VAD_VOICE_START_RATIO": "0.75",
            "VAD_VOICE_END_RATIO": "0.85",
            "VAD_VOICE_START_FRAME_COUNT": "15",
            "VAD_VOICE_END_FRAME_COUNT": "40",
            "VAD_ENABLE_DENOISING": "false",
            "VAD_AUTO_CONVERT_SAMPLE_RATE": "no",
            "VAD_BUFFER_SIZE": "1024"
        }
        
        # Set environment variables
        for key, value in env_vars.items():
            env_cleanup[key] = os.environ.get(key)
            os.environ[key] = value
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.is_dir', return_value=True):
            config = VADConfig.from_env()
        
        assert config.sample_rate == SampleRate.SAMPLERATE_24
        assert config.model_version == SileroModelVersion.V4
        assert config.model_path == Path("/test/path")
        assert config.vad_start_probability == 0.8
        assert config.vad_end_probability == 0.6
        assert config.voice_start_ratio == 0.75
        assert config.voice_end_ratio == 0.85
        assert config.voice_start_frame_count == 15
        assert config.voice_end_frame_count == 40
        assert config.enable_denoising is False
        assert config.auto_convert_sample_rate is False
        assert config.buffer_size == 1024

    @pytest.mark.parametrize("bool_value,expected", [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
        ("0", False),
        ("no", False),
        ("off", False),
        ("", False),
        ("random", False),
    ])
    def test_from_env_boolean_conversion(self, bool_value, expected, env_cleanup):
        """Test boolean conversion from environment variables."""
        env_cleanup["VAD_ENABLE_DENOISING"] = os.environ.get("VAD_ENABLE_DENOISING")
        os.environ["VAD_ENABLE_DENOISING"] = bool_value
        
        config = VADConfig.from_env()
        assert config.enable_denoising == expected

    def test_from_env_custom_prefix(self, env_cleanup):
        """Test environment variable loading with custom prefix."""
        prefix = "CUSTOM_"
        env_vars = {
            "CUSTOM_SAMPLE_RATE": "48000",
            "CUSTOM_MODEL_VERSION": "v5",
            "CUSTOM_START_PROBABILITY": "0.9"
        }
        
        for key, value in env_vars.items():
            env_cleanup[key] = os.environ.get(key)
            os.environ[key] = value
        
        config = VADConfig.from_env(prefix=prefix)
        
        assert config.sample_rate == SampleRate.SAMPLERATE_48
        assert config.model_version == SileroModelVersion.V5
        assert config.vad_start_probability == 0.9

    def test_from_env_no_variables_returns_default(self, env_cleanup):
        """Test that from_env returns default config when no variables are set."""
        # Ensure no VAD_ variables are set
        for key in list(os.environ.keys()):
            if key.startswith("VAD_"):
                env_cleanup[key] = os.environ[key]
                del os.environ[key]
        
        config = VADConfig.from_env()
        default_config = VADConfig()
        
        assert config.sample_rate == default_config.sample_rate
        assert config.model_version == default_config.model_version
        assert config.vad_start_probability == default_config.vad_start_probability

    @pytest.mark.parametrize("model_version,expected_filename", [
        (SileroModelVersion.V4, "silero_vad.onnx"),
        (SileroModelVersion.V5, "silero_vad_v5.onnx"),
    ])
    def test_get_model_filename(self, model_version, expected_filename):
        """Test model filename generation for different versions."""
        config = VADConfig(model_version=model_version)
        assert config.get_model_filename() == expected_filename

    @pytest.mark.parametrize("sample_rate,buffer_size,expected_duration", [
        (SampleRate.SAMPLERATE_16, 512, 32.0),
        (SampleRate.SAMPLERATE_16, 1024, 64.0),
        (SampleRate.SAMPLERATE_48, 512, 512000/48000),  # 10.666... ms
        (SampleRate.SAMPLERATE_8, 256, 32.0),
    ])
    def test_get_frame_duration_ms(self, sample_rate, buffer_size, expected_duration):
        """Test frame duration calculation for different configurations."""
        config = VADConfig(sample_rate=sample_rate, buffer_size=buffer_size)
        calculated_duration = config.get_frame_duration_ms()
        assert abs(calculated_duration - expected_duration) < 0.01

    def test_str_representation(self):
        """Test string representation contains key information."""
        config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_24,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.8,
            vad_end_probability=0.6
        )
        
        str_repr = str(config)
        
        assert "VADConfig" in str_repr
        assert "24000Hz" in str_repr
        assert "v4" in str_repr
        assert "0.8" in str_repr
        assert "0.6" in str_repr

    def test_repr_representation(self):
        """Test that repr returns the same as str."""
        config = VADConfig()
        assert repr(config) == str(config)

    def test_config_immutability_after_creation(self):
        """Test that configuration validation works on assignment."""
        config = VADConfig()
        
        # This should work - valid assignment
        config.vad_start_probability = 0.8
        assert config.vad_start_probability == 0.8
        
        # This should fail - invalid assignment
        with pytest.raises(ValueError):
            config.vad_start_probability = 1.5


class TestSampleRate:
    """Test SampleRate enum functionality."""

    def test_sample_rate_enum_values(self):
        """Test that SampleRate enum has correct values."""
        assert SampleRate.SAMPLERATE_8 == 8000
        assert SampleRate.SAMPLERATE_16 == 16000
        assert SampleRate.SAMPLERATE_24 == 24000
        assert SampleRate.SAMPLERATE_48 == 48000

    @pytest.mark.parametrize("int_value,expected_enum", [
        (8000, SampleRate.SAMPLERATE_8),
        (16000, SampleRate.SAMPLERATE_16),
        (24000, SampleRate.SAMPLERATE_24),
        (48000, SampleRate.SAMPLERATE_48),
    ])
    def test_sample_rate_from_int(self, int_value, expected_enum):
        """Test creating SampleRate from integer value."""
        assert SampleRate(int_value) == expected_enum

    def test_sample_rate_invalid_value(self):
        """Test that invalid sample rate raises ValueError."""
        with pytest.raises(ValueError):
            SampleRate(44100)  # Not supported sample rate

    def test_sample_rate_iteration(self):
        """Test that SampleRate enum can be iterated."""
        sample_rates = list(SampleRate)
        assert len(sample_rates) == 4
        assert SampleRate.SAMPLERATE_8 in sample_rates
        assert SampleRate.SAMPLERATE_16 in sample_rates
        assert SampleRate.SAMPLERATE_24 in sample_rates
        assert SampleRate.SAMPLERATE_48 in sample_rates

    def test_sample_rate_comparison(self):
        """Test SampleRate comparison operations."""
        assert SampleRate.SAMPLERATE_8 < SampleRate.SAMPLERATE_16
        assert SampleRate.SAMPLERATE_48 > SampleRate.SAMPLERATE_24
        assert SampleRate.SAMPLERATE_16 == SampleRate.SAMPLERATE_16


class TestSileroModelVersion:
    """Test SileroModelVersion enum functionality."""

    def test_model_version_enum_values(self):
        """Test that SileroModelVersion enum has correct values."""
        assert SileroModelVersion.V4.value == "v4"
        assert SileroModelVersion.V5.value == "v5"

    @pytest.mark.parametrize("str_value,expected_enum", [
        ("v4", SileroModelVersion.V4),
        ("v5", SileroModelVersion.V5),
    ])
    def test_model_version_from_string(self, str_value, expected_enum):
        """Test creating SileroModelVersion from string value."""
        assert SileroModelVersion(str_value) == expected_enum

    def test_model_version_invalid_value(self):
        """Test that invalid model version raises ValueError."""
        with pytest.raises(ValueError):
            SileroModelVersion("v6")  # Not supported version

    def test_model_version_case_sensitivity(self):
        """Test that model version is case sensitive."""
        assert SileroModelVersion("v4") == SileroModelVersion.V4
        
        # These should fail due to case sensitivity
        with pytest.raises(ValueError):
            SileroModelVersion("V4")
        with pytest.raises(ValueError):
            SileroModelVersion("V5")

    def test_model_version_iteration(self):
        """Test that SileroModelVersion enum can be iterated."""
        versions = list(SileroModelVersion)
        assert len(versions) == 2
        assert SileroModelVersion.V4 in versions
        assert SileroModelVersion.V5 in versions

    def test_model_version_string_representation(self):
        """Test string representation of model versions."""
        assert str(SileroModelVersion.V4) == "SileroModelVersion.V4"
        assert str(SileroModelVersion.V5) == "SileroModelVersion.V5"


class TestConfigIntegration:
    """Integration tests for configuration components."""

    def test_config_with_all_enums(self):
        """Test configuration with all enum combinations."""
        for sample_rate in SampleRate:
            for model_version in SileroModelVersion:
                config = VADConfig(
                    sample_rate=sample_rate,
                    model_version=model_version
                )
                assert config.sample_rate == sample_rate
                assert config.model_version == model_version

    def test_config_serialization_deserialization_cycle(self):
        """Test complete serialization/deserialization cycle."""
        original_config = VADConfig(
            sample_rate=SampleRate.SAMPLERATE_48,
            model_version=SileroModelVersion.V4,
            vad_start_probability=0.85,
            voice_start_frame_count=25
        )
        
        # Convert to dict and back
        config_dict = original_config.to_dict()
        restored_config = VADConfig.from_dict(config_dict)
        
        assert restored_config.sample_rate == original_config.sample_rate
        assert restored_config.model_version == original_config.model_version
        assert restored_config.vad_start_probability == original_config.vad_start_probability
        assert restored_config.voice_start_frame_count == original_config.voice_start_frame_count

    def test_config_edge_cases(self):
        """Test configuration edge cases and boundary conditions."""
        # Minimum valid configuration
        min_config = VADConfig(
            vad_start_probability=0.0,
            vad_end_probability=0.0,
            voice_start_ratio=0.0,
            voice_end_ratio=0.0,
            voice_start_frame_count=1,
            voice_end_frame_count=1,
            buffer_size=256
        )
        
        assert min_config.vad_start_probability == 0.0
        assert min_config.voice_start_frame_count == 1
        assert min_config.buffer_size == 256
        
        # Maximum valid configuration
        max_config = VADConfig(
            vad_start_probability=1.0,
            vad_end_probability=1.0,
            voice_start_ratio=1.0,
            voice_end_ratio=1.0,
            buffer_size=2048
        )
        
        assert max_config.vad_start_probability == 1.0
        assert max_config.voice_start_ratio == 1.0
        assert max_config.buffer_size == 2048
