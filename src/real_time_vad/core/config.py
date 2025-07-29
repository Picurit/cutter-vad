"""
Configuration classes and enums for the VAD library.
"""

from __future__ import annotations

import os
import yaml
from enum import Enum, IntEnum
from typing import Optional, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class SampleRate(IntEnum):
    """Supported sample rates for audio processing."""
    SAMPLERATE_8 = 8000   # 8 kHz sample rate
    SAMPLERATE_16 = 16000 # 16 kHz sample rate
    SAMPLERATE_24 = 24000 # 24 kHz sample rate  
    SAMPLERATE_48 = 48000 # 48 kHz sample rate


class SileroModelVersion(Enum):
    """Silero model versions."""
    V4 = "v4"  # Silero Model Version 4
    V5 = "v5"  # Silero Model Version 5


class VADConfig(BaseModel):
    """
    Configuration class for Voice Activity Detection parameters.
    
    This class encapsulates all the configurable parameters for the VAD system,
    including model settings, thresholds, and frame counts.
    """
    
    # Model configuration
    sample_rate: SampleRate = Field(
        default=SampleRate.SAMPLERATE_16,
        description="Audio sample rate for processing"
    )
    
    model_version: SileroModelVersion = Field(
        default=SileroModelVersion.V5,
        description="Silero model version to use"
    )
    
    model_path: Optional[Path] = Field(
        default=None,
        description="Custom path to model files (optional)"
    )
    
    # VAD thresholds
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
    
    # Voice detection ratios
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
    
    # Frame counts
    voice_start_frame_count: int = Field(
        default=10,
        ge=1,
        description="Frame count required to confirm voice start, with defaults: buffer size 512, sample rate 16kHz; then -> (10 frames * 512 buffer size) / 16000 sample rate = 0.32 seconds = 320 ms"
    )
    
    voice_end_frame_count: int = Field(
        default=50,
        ge=1,
        description="Frame count required to confirm voice end, with defaults: buffer size 512, sample rate 16kHz; then -> (50 frames * 512 buffer size) / 16000 sample rate = 1.6 seconds = 1600 ms"
    )
    
    # Processing options
    enable_denoising: bool = Field(
        default=True,
        description="Enable audio denoising during processing"
    )
    
    auto_convert_sample_rate: bool = Field(
        default=True,
        description="Automatically convert input audio to target sample rate"
    )
    
    buffer_size: int = Field(
        default=512,
        ge=256,
        le=2048,
        description="Audio buffer size for processing"
    )
    
    # Output options
    output_wav_sample_rate: int = Field(
        default=16000,
        description="Sample rate for output WAV data"
    )
    
    output_wav_bit_depth: int = Field(
        default=16,
        description="Bit depth for output WAV data"
    )
    
    model_config = ConfigDict(
        use_enum_values=False,
        validate_assignment=True,
        extra="forbid"
    )
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: Optional[Path]) -> Optional[Path]:
        """Validate model path exists if provided."""
        if v is not None:
            if not v.exists():
                raise ValueError(f"Model path does not exist: {v}")
            if not v.is_dir():
                raise ValueError(f"Model path must be a directory: {v}")
        return v
    

    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> VADConfig:
        """Create configuration from dictionary."""
        # Convert string enums to proper enum values
        if 'sample_rate' in config_dict:
            sr = config_dict['sample_rate']
            if isinstance(sr, str):
                config_dict['sample_rate'] = getattr(SampleRate, f'SAMPLERATE_{sr}')
            elif isinstance(sr, int):
                config_dict['sample_rate'] = SampleRate(sr)
        
        if 'model_version' in config_dict:
            mv = config_dict['model_version']
            if isinstance(mv, str):
                config_dict['model_version'] = SileroModelVersion(mv.lower())
        
        if 'model_path' in config_dict and config_dict['model_path']:
            config_dict['model_path'] = Path(config_dict['model_path'])
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> VADConfig:
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_env(cls, prefix: str = "VAD_") -> VADConfig:
        """Load configuration from environment variables."""
        config_dict = {}
        
        # Map environment variable names to config fields
        env_mappings = {
            f"{prefix}SAMPLE_RATE": "sample_rate",
            f"{prefix}MODEL_VERSION": "model_version", 
            f"{prefix}MODEL_PATH": "model_path",
            f"{prefix}START_PROBABILITY": "vad_start_probability",
            f"{prefix}END_PROBABILITY": "vad_end_probability",
            f"{prefix}VOICE_START_RATIO": "voice_start_ratio",
            f"{prefix}VOICE_END_RATIO": "voice_end_ratio",
            f"{prefix}VOICE_START_FRAME_COUNT": "voice_start_frame_count",
            f"{prefix}VOICE_END_FRAME_COUNT": "voice_end_frame_count",
            f"{prefix}ENABLE_DENOISING": "enable_denoising",
            f"{prefix}AUTO_CONVERT_SAMPLE_RATE": "auto_convert_sample_rate",
            f"{prefix}BUFFER_SIZE": "buffer_size",
        }
        
        for env_key, config_key in env_mappings.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                # Type conversion based on field
                if config_key in ['sample_rate', 'voice_start_frame_count', 
                                 'voice_end_frame_count', 'buffer_size', 'output_wav_sample_rate', 
                                 'output_wav_bit_depth']:
                    config_dict[config_key] = int(env_value)
                elif config_key in ['vad_start_probability', 'vad_end_probability',
                                   'voice_start_ratio', 'voice_end_ratio']:
                    config_dict[config_key] = float(env_value)
                elif config_key in ['enable_denoising', 'auto_convert_sample_rate']:
                    config_dict[config_key] = env_value.lower() in ('true', '1', 'yes', 'on')
                else:
                    config_dict[config_key] = env_value
        
        return cls.from_dict(config_dict) if config_dict else cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        # Return dictionary with enum objects, not their values
        return self.model_dump()
    
    def _to_serializable_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary with serializable values."""
        data = self.model_dump()
        # Convert enums to their values for serialization
        if isinstance(data.get('sample_rate'), SampleRate):
            data['sample_rate'] = data['sample_rate'].value
        if isinstance(data.get('model_version'), SileroModelVersion):
            data['model_version'] = data['model_version'].value
        # Convert Path to string
        if data.get('model_path') is not None:
            data['model_path'] = str(data['model_path'])
        return data

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(self._to_serializable_dict(), f, default_flow_style=False)
    
    def get_model_filename(self) -> str:
        """Get the expected model filename for the current configuration."""
        if self.model_version == SileroModelVersion.V4:
            return "silero_vad.onnx"
        elif self.model_version == SileroModelVersion.V5:
            return "silero_vad_v5.onnx"
        else:
            return "silero_vad.onnx"  # default fallback
    
    def get_frame_duration_ms(self) -> float:
        """Get the duration of each VAD frame in milliseconds."""
        # Each frame processes buffer_size samples at the given sample rate
        return (self.buffer_size / self.sample_rate) * 1000 # Defaults: buffer size 512, sample rate 16kHz; then -> (512 samples / 16000 samples/sec) * 1000 = 32 ms
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return (
            f"VADConfig("
            f"\tsample_rate={self.sample_rate}Hz, "
            f"\tmodel={self.model_version.value}, "
            f"\tstart_prob={self.vad_start_probability}, "
            f"\tend_prob={self.vad_end_probability}"
            f")"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return self.__str__()
