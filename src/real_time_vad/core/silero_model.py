"""
Core Silero model wrapper for Voice Activity Detection.

This module provides Pydantic-based models for robust type validation,
state management, and error handling in voice activity detection.
"""

import os
import logging
import numpy as np
from typing import Optional, Callable, Any, Dict, Union, List
from pathlib import Path
import onnxruntime as ort
from collections import deque

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from pydantic.dataclasses import dataclass

from .config import VADConfig, SampleRate, SileroModelVersion
from .exceptions import (
    VADError, 
    ModelNotFoundError, 
    ModelInitializationError,
    AudioProcessingError,
    CallbackError
)
from ..utils.audio import AudioUtils
from ..utils.wav_writer import WAVWriter


# ========================= Pydantic Models =========================

class ModelState(BaseModel):
    """
    Pydantic model for LSTM model states.
    
    Provides type validation and state management for Silero VAD model
    hidden states, ensuring consistency across different model versions.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # For V5 model - combined state
    state: Optional[np.ndarray] = Field(
        default=None,
        description="Combined LSTM state for V5 model (2, batch, 128)"
    )
    
    # For V4 and earlier - separate states  
    hidden_state: Optional[np.ndarray] = Field(
        default=None,
        description="Hidden state for LSTM (2, batch, 64)"
    )
    
    cell_state: Optional[np.ndarray] = Field(
        default=None,
        description="Cell state for LSTM (2, batch, 64)"
    )
    
    @field_validator('state', 'hidden_state', 'cell_state')
    @classmethod
    def validate_numpy_arrays(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate numpy arrays for state management."""
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("State must be a numpy array")
            if v.dtype != np.float32:
                raise ValueError("State arrays must be float32")
        return v
    
    @model_validator(mode='after')
    def validate_state_consistency(self) -> 'ModelState':
        """Ensure state consistency based on model version requirements."""
        has_combined = self.state is not None
        has_separate = self.hidden_state is not None or self.cell_state is not None
        
        if has_combined and has_separate:
            raise ValueError("Cannot have both combined state and separate states")
            
        return self


class ProcessingResult(BaseModel):
    """
    Pydantic model for audio frame processing results.
    
    Provides structured output from VAD processing with type validation
    and clear semantics for voice activity detection results.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Voice activity events
    voice_started: bool = Field(
        default=False,
        description="True if voice activity just started"
    )
    
    voice_ended: bool = Field(
        default=False,
        description="True if voice activity just ended"
    )
    
    voice_continuing: bool = Field(
        default=False,
        description="True if voice activity is continuing"
    )
    
    # Probability and audio data
    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Voice activity probability (0.0 to 1.0)"
    )
    
    wav_data: Optional[bytes] = Field(
        default=None,
        description="WAV formatted audio data when voice ends"
    )
    
    pcm_data: Optional[bytes] = Field(
        default=None,
        description="PCM audio data during voice activity"
    )
    
    @field_validator('wav_data', 'pcm_data')
    @classmethod
    def validate_audio_data(cls, v: Optional[bytes]) -> Optional[bytes]:
        """Validate audio data is proper bytes."""
        if v is not None and not isinstance(v, bytes):
            raise ValueError("Audio data must be bytes")
        return v


class ProcessingStatistics(BaseModel):
    """
    Pydantic model for VAD processing statistics.
    
    Provides comprehensive statistics about the current state and
    performance of the VAD processor.
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Current state
    is_voice_active: bool = Field(
        description="Whether voice is currently active"
    )
    
    voice_start_frame_count: int = Field(
        ge=0,
        description="Number of consecutive frames above start threshold"
    )
    
    voice_end_frame_count: int = Field(
        ge=0,
        description="Number of consecutive frames below end threshold"
    )
    
    # Probability statistics
    recent_probabilities: List[float] = Field(
        default_factory=list,
        description="Recent voice probability values"
    )
    
    average_probability: float = Field(
        ge=0.0,
        le=1.0,
        description="Average of recent probabilities"
    )
    
    # Buffer information
    voice_buffer_size: int = Field(
        ge=0,
        description="Current size of voice buffer"
    )
    
    current_voice_length: int = Field(
        ge=0,
        description="Length of current voice data in samples"
    )
    
    @field_validator('recent_probabilities')
    @classmethod
    def validate_probabilities(cls, v: List[float]) -> List[float]:
        """Validate all probabilities are in valid range."""
        for prob in v:
            if not (0.0 <= prob <= 1.0):
                raise ValueError(f"Probability {prob} must be between 0.0 and 1.0")
        return v


class ModelConfiguration(BaseModel):
    """
    Pydantic model for Silero model configuration.
    
    Validates model initialization parameters and provides
    type-safe configuration management.
    """
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    model_path: str = Field(
        description="Path to the ONNX model file"
    )
    
    model_version: SileroModelVersion = Field(
        description="Silero model version (V4 or V5)"
    )
    
    @field_validator('model_path')
    @classmethod
    def validate_model_path(cls, v: str) -> str:
        """Validate model path exists and is readable."""
        if not os.path.exists(v):
            raise ModelNotFoundError(v)
        if not os.path.isfile(v):
            raise ValueError(f"Model path must be a file: {v}")
        if not v.endswith('.onnx'):
            raise ValueError(f"Model file must have .onnx extension: {v}")
        return v


# ========================= Core Classes =========================


class SileroVADModel(BaseModel):
    """
    Pydantic-based Silero VAD model wrapper using ONNX Runtime.
    
    This class provides type-safe model management with automatic validation
    of inputs, outputs, and internal states. Follows SOLID principles with
    clear separation of concerns and dependency injection.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Configuration
    config: ModelConfiguration = Field(
        description="Model configuration with validated parameters"
    )
    
    # Runtime state
    session: Optional[ort.InferenceSession] = Field(
        default=None,
        description="ONNX Runtime inference session"
    )
    
    model_state: ModelState = Field(
        default_factory=ModelState,
        description="Current model LSTM states"
    )
    
    # Performance tracking
    prediction_count: int = Field(
        default=0,
        ge=0,
        description="Number of predictions made"
    )
    
    def __init__(self, model_path: str, model_version: SileroModelVersion, **data: Any) -> None:
        """
        Initialize Silero VAD model with validation.
        
        Args:
            model_path: Path to the ONNX model file
            model_version: Model version (V4 or V5)
            **data: Additional data for Pydantic model
        
        Raises:
            ModelNotFoundError: If model file doesn't exist
            ModelInitializationError: If model loading fails
            ValidationError: If parameters are invalid
        """
        # Create configuration and validate
        config = ModelConfiguration(
            model_path=model_path,
            model_version=model_version
        )
        
        # Initialize parent with configuration
        super().__init__(config=config, **data)
        
        # Load model and initialize states
        self._load_model()
        self._reset_states()
    
    def _load_model(self) -> None:
        """
        Load the ONNX model with optimized settings.
        
        Raises:
            ModelInitializationError: If model loading fails
        """
        try:
            # Determine optimal execution providers
            providers = self._get_execution_providers()
            
            # Configure session options for optimal performance
            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Create inference session
            self.session = ort.InferenceSession(
                self.config.model_path,
                sess_options=session_options,
                providers=providers
            )
            
            # Validate model inputs/outputs
            self._validate_model_signature()
                        
        except Exception as e:
            raise ModelInitializationError(
                self.config.model_version.value,
                f"Failed to load model from {self.config.model_path}: {str(e)}"
            )
    
    def _get_execution_providers(self) -> List[str]:
        """
        Get optimal execution providers for the current system.
        
        Returns:
            List of execution providers in priority order
        """
        providers = ['CPUExecutionProvider']
        
        available_providers = ort.get_available_providers()
        if available_providers and 'CUDAExecutionProvider' in available_providers:
            providers.insert(0, 'CUDAExecutionProvider')
            
        return providers
    
    def _validate_model_signature(self) -> None:
        """
        Validate that the loaded model has the expected input/output signature.
        
        Raises:
            ModelInitializationError: If model signature is invalid
        """
        if self.session is None:
            raise ModelInitializationError(
                self.config.model_version.value,
                "Session not initialized"
            )
        
        try:
            inputs = self.session.get_inputs()
            outputs = self.session.get_outputs()
            
            # Expected inputs: input, state/h+c, sr
            expected_input_count = 3 if self.config.model_version == SileroModelVersion.V5 else 4
            if len(inputs) != expected_input_count:
                raise ValueError(f"Expected {expected_input_count} inputs, got {len(inputs)}")
            
            # Expected outputs: probability, state/h+c
            expected_output_count = 2 if self.config.model_version == SileroModelVersion.V5 else 3
            if len(outputs) != expected_output_count:
                raise ValueError(f"Expected {expected_output_count} outputs, got {len(outputs)}")
                
        except Exception as e:
            raise ModelInitializationError(
                self.config.model_version.value,
                f"Model signature validation failed: {str(e)}"
            )
    
    def _reset_states(self) -> None:
        """
        Reset LSTM hidden states based on model version.
        
        Initializes appropriate state tensors with correct dimensions
        for the specific Silero model version.
        """
        if self.config.model_version == SileroModelVersion.V5:
            # V5 uses combined state (2, batch, 128)
            self.model_state = ModelState(
                state=np.zeros((2, 1, 128), dtype=np.float32)
            )
        else:
            # V4 and earlier use separate h and c states (2, batch, 64)
            self.model_state = ModelState(
                hidden_state=np.zeros((2, 1, 64), dtype=np.float32),
                cell_state=np.zeros((2, 1, 64), dtype=np.float32)
            )
    
    def predict(self, audio_chunk: np.ndarray, sample_rate: int) -> float:
        """
        Predict voice activity probability for audio chunk.
        
        Args:
            audio_chunk: Audio chunk (exactly 512 samples for 16kHz)
            sample_rate: Sample rate of the audio
            
        Returns:
            Voice activity probability (0.0 to 1.0)
            
        Raises:
            ModelInitializationError: If model not loaded
            AudioProcessingError: If prediction fails
            ValidationError: If inputs are invalid
        """
        try:
            if self.session is None:
                raise ModelInitializationError(
                    self.config.model_version.value,
                    "Model not loaded"
                )
            
            # Validate and prepare audio input
            processed_audio = self._prepare_audio_input(audio_chunk)
            
            # Prepare model inputs based on version
            inputs = self._prepare_model_inputs(processed_audio, sample_rate)
            
            # Run inference
            outputs = self.session.run(None, inputs)
            
            # Extract and validate results
            probability = self._extract_probability(outputs)
            self._update_states(outputs)
            
            # Update prediction counter
            self.prediction_count += 1
            
            return probability
            
        except (ModelInitializationError, AudioProcessingError):
            raise
        except Exception as e:
            raise AudioProcessingError(f"Model prediction failed: {str(e)}")
    
    def _prepare_audio_input(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Prepare audio input for model inference.
        
        Args:
            audio_chunk: Raw audio chunk
            
        Returns:
            Processed audio chunk ready for model input
            
        Raises:
            AudioProcessingError: If audio preparation fails
        """
        try:
            # Ensure correct length (512 samples)
            if len(audio_chunk) != 512:
                if len(audio_chunk) < 512:
                    audio_chunk = np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:512]
            
            # Ensure correct type and shape for model input
            return audio_chunk.reshape(1, -1).astype(np.float32)
            
        except Exception as e:
            raise AudioProcessingError(f"Audio input preparation failed: {str(e)}")
    
    def _prepare_model_inputs(self, audio_input: np.ndarray, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        Prepare all model inputs based on version.
        
        Args:
            audio_input: Processed audio input
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary of model inputs
        """
        if self.config.model_version == SileroModelVersion.V5:
            return {
                'input': audio_input,
                'state': self.model_state.state,
                'sr': np.array([sample_rate], dtype=np.int64)
            }
        else:
            return {
                'input': audio_input,
                'h': self.model_state.hidden_state,
                'c': self.model_state.cell_state,
                'sr': np.array([sample_rate], dtype=np.int64)
            }
    
    def _extract_probability(self, outputs: List[np.ndarray]) -> float:
        """
        Extract voice probability from model outputs.
        
        Args:
            outputs: Raw model outputs
            
        Returns:
            Voice activity probability
            
        Raises:
            AudioProcessingError: If probability extraction fails
        """
        try:
            probability = float(outputs[0][0][0])
            
            # Validate probability range
            if not (0.0 <= probability <= 1.0):
                raise ValueError(f"Invalid probability value: {probability}")
                
            return probability
            
        except Exception as e:
            raise AudioProcessingError(f"Probability extraction failed: {str(e)}")
    
    def _update_states(self, outputs: List[np.ndarray]) -> None:
        """
        Update internal LSTM states from model outputs.
        
        Args:
            outputs: Model outputs containing updated states
        """
        if self.config.model_version == SileroModelVersion.V5:
            self.model_state.state = outputs[1]
        else:
            self.model_state.hidden_state = outputs[1]
            self.model_state.cell_state = outputs[2]
    
    def reset(self) -> None:
        """
        Reset model states to initial values.
        
        This method should be called when starting a new audio stream
        or when the audio context changes significantly.
        """
        self._reset_states()
        
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and statistics
        """
        return {
            'model_path': self.config.model_path,
            'model_version': self.config.model_version.value,
            'prediction_count': self.prediction_count,
            'session_providers': self.session.get_providers() if self.session else None,
            'has_cuda': 'CUDAExecutionProvider' in (self.session.get_providers() if self.session else []),
            'state_shape': {
                'state': self.model_state.state.shape if self.model_state.state is not None else None,
                'hidden_state': self.model_state.hidden_state.shape if self.model_state.hidden_state is not None else None,
                'cell_state': self.model_state.cell_state.shape if self.model_state.cell_state is not None else None
            }
        }


class VADProcessor(BaseModel):
    """
    Pydantic-based Voice Activity Detection processor with state management.
    
    This class manages the complete VAD pipeline including model inference,
    state machine logic, and audio buffering. Uses Pydantic for robust
    type validation and state management.
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # Configuration
    config: VADConfig = Field(
        description="VAD configuration parameters"
    )
    
    # Core model
    model: Optional[SileroVADModel] = Field(
        default=None,
        description="Silero VAD model instance"
    )
    
    # State management
    is_voice_active: bool = Field(
        default=False,
        description="Current voice activity state"
    )
    
    voice_start_frame_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive frames above start threshold"
    )
    
    voice_end_frame_count: int = Field(
        default=0,
        ge=0,
        description="Consecutive frames below end threshold"
    )
    
    # Probability tracking
    voice_probabilities: deque = Field(
        default_factory=lambda: deque(maxlen=100),
        description="Recent voice probabilities"
    )
    
    # Audio buffer management
    voice_buffer: deque = Field(
        default_factory=deque,
        description="Buffer for potential voice data"
    )
    
    current_voice_data: Optional[np.ndarray] = Field(
        default=None,
        description="Current accumulated voice data"
    )
    
    # Output handling
    wav_writer: WAVWriter = Field(
        description="WAV file writer instance"
    )
    
    def __init__(self, config: VADConfig, **data: Any) -> None:
        """
        Initialize VAD processor with configuration validation.
        
        Args:
            config: VAD configuration
            **data: Additional data for Pydantic model
            
        Raises:
            ValidationError: If configuration is invalid
            ModelInitializationError: If model loading fails
        """
        # Initialize WAV writer based on config
        wav_writer = WAVWriter(
            sample_rate=config.output_wav_sample_rate,
            bit_depth=config.output_wav_bit_depth,
            channels=1
        )
        
        # Initialize parent with all required fields
        super().__init__(
            config=config,
            wav_writer=wav_writer,
            **data
        )
        
        # Load the Silero model
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load the Silero VAD model with proper error handling.
        
        Raises:
            ModelInitializationError: If model loading fails
            ModelNotFoundError: If model file not found
        """
        try:
            # Determine model path
            model_dir = self._get_model_directory()
            model_filename = self.config.get_model_filename()
            model_path = model_dir / model_filename
            
            if not model_path.exists():
                raise ModelNotFoundError(str(model_path))
            
            # Create model instance with validation
            self.model = SileroVADModel(
                str(model_path), 
                self.config.model_version
            )
            
        except (ModelNotFoundError, ModelInitializationError):
            raise
        except Exception as e:
            raise ModelInitializationError(
                self.config.model_version.value,
                f"Failed to initialize VAD processor: {str(e)}"
            )
    
    def _get_model_directory(self) -> Path:
        """
        Get the directory containing model files.
        
        Returns:
            Path to model directory
        """
        if self.config.model_path:
            model_dir = Path(self.config.model_path)
            logging.info(f"Using configured model path: {model_dir}")
            return model_dir
        else:
            # Use package resources
            model_dir = Path(__file__).parent.parent / "models"
            logging.info(f"Using package model path: {model_dir}")
            return model_dir
    
    def process_frame(self, audio_frame: np.ndarray) -> ProcessingResult:
        """
        Process a single audio frame with comprehensive validation.
        
        Args:
            audio_frame: Audio frame data (float32)
            
        Returns:
            ProcessingResult with structured output
            
        Raises:
            ModelInitializationError: If model not loaded
            AudioProcessingError: If processing fails
            ValidationError: If inputs are invalid
        """
        try:
            if self.model is None:
                raise ModelInitializationError(
                    self.config.model_version.value, 
                    "Model not loaded"
                )
            
            # Validate and preprocess audio
            processed_frame = self._preprocess_audio_frame(audio_frame)
            
            # Get voice probability from model
            probability = self.model.predict(processed_frame, self.config.sample_rate)
            self.voice_probabilities.append(probability)
            
            # Process voice activity state machine
            result_data = self._process_voice_state(probability, processed_frame)
            result_data['probability'] = probability
            
            # Create and validate result
            return ProcessingResult(**result_data)
            
        except (ModelInitializationError, AudioProcessingError):
            raise
        except Exception as e:
            raise AudioProcessingError(f"Frame processing failed: {str(e)}")
    
    def _preprocess_audio_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """
        Preprocess audio frame with validation and optional denoising.
        
        Args:
            audio_frame: Raw audio frame
            
        Returns:
            Processed audio frame
            
        Raises:
            AudioProcessingError: If preprocessing fails
        """
        try:
            # Validate audio data
            AudioUtils.validate_audio_data(audio_frame)
            
            # Apply denoising if enabled
            if self.config.enable_denoising:
                audio_frame = AudioUtils.denoise_audio(audio_frame)
            
            return audio_frame
            
        except Exception as e:
            raise AudioProcessingError(f"Audio preprocessing failed: {str(e)}")
    
    def _process_voice_state(self, probability: float, audio_frame: np.ndarray) -> Dict[str, Any]:
        """
        Process voice activity state machine with robust logic.
        
        Args:
            probability: Voice activity probability
            audio_frame: Current audio frame
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'voice_started': False,
            'voice_ended': False,
            'voice_continuing': False,
            'wav_data': None,
            'pcm_data': None
        }
        
        if not self.is_voice_active:
            # Check for voice start
            result.update(self._handle_voice_start_detection(probability, audio_frame))
        else:
            # Handle ongoing voice activity
            result.update(self._handle_ongoing_voice_activity(probability, audio_frame))
        
        return result
    
    def _handle_voice_start_detection(self, probability: float, audio_frame: np.ndarray) -> Dict[str, Any]:
        """
        Handle voice start detection logic.
        
        Args:
            probability: Voice activity probability
            audio_frame: Current audio frame
            
        Returns:
            Dictionary with voice start results
        """
        result = {}
        
        if probability >= self.config.vad_start_probability:
            self.voice_start_frame_count += 1
            
            # Add frame to potential voice buffer
            self.voice_buffer.append(audio_frame.copy())
            
            # Confirm voice start
            if self.voice_start_frame_count >= self.config.voice_start_frame_count:
                self._confirm_voice_start()
                result['voice_started'] = True
        else:
            # Reset if probability drops
            self._reset_voice_start_detection()
        
        return result
    
    def _confirm_voice_start(self) -> None:
        """Confirm voice activity has started."""
        self.is_voice_active = True
        self.voice_start_frame_count = 0
        self.voice_end_frame_count = 0
        
        # Initialize voice recording with buffered data
        if self.voice_buffer:
            self.current_voice_data = np.concatenate(list(self.voice_buffer))
        self.voice_buffer.clear()
    
    def _reset_voice_start_detection(self) -> None:
        """Reset voice start detection state."""
        self.voice_start_frame_count = 0
        self.voice_buffer.clear()
    
    def _handle_ongoing_voice_activity(self, probability: float, audio_frame: np.ndarray) -> Dict[str, Any]:
        """
        Handle ongoing voice activity processing.
        
        Args:
            probability: Voice activity probability
            audio_frame: Current audio frame
            
        Returns:
            Dictionary with voice activity results
        """
        result = {}
        
        # Add frame to voice recording
        self._accumulate_voice_data(audio_frame)
        
        # Provide continuous PCM data
        result['voice_continuing'] = True
        result['pcm_data'] = audio_frame.tobytes()
        
        # Check for voice end
        if probability < self.config.vad_end_probability:
            self.voice_end_frame_count += 1
            
            # Confirm voice end
            if self.voice_end_frame_count >= self.config.voice_end_frame_count:
                wav_data = self._finalize_voice_segment()
                result['voice_ended'] = True
                result['wav_data'] = wav_data
        else:
            # Reset end count if probability rises
            self.voice_end_frame_count = 0
        
        return result
    
    def _accumulate_voice_data(self, audio_frame: np.ndarray) -> None:
        """Accumulate audio frame into current voice data."""
        if self.current_voice_data is not None:
            self.current_voice_data = np.concatenate([self.current_voice_data, audio_frame])
        else:
            self.current_voice_data = audio_frame.copy()
    
    def _finalize_voice_segment(self) -> Optional[bytes]:
        """
        Finalize voice segment and generate WAV data.
        
        Returns:
            WAV data bytes or None if no voice data
        """
        wav_data = None
        
        if self.current_voice_data is not None:
            wav_data = self.wav_writer.write_wav_data(self.current_voice_data)
        
        # Reset state
        self.is_voice_active = False
        self.voice_end_frame_count = 0
        self.current_voice_data = None
        
        return wav_data
    
    def reset(self) -> None:
        """
        Reset VAD processor to initial state.
        
        This method should be called when starting a new audio stream
        or when the audio context changes significantly.
        """
        if self.model:
            self.model.reset()
        
        self.is_voice_active = False
        self.voice_start_frame_count = 0
        self.voice_end_frame_count = 0
        self.voice_probabilities.clear()
        self.voice_buffer.clear()
        self.current_voice_data = None
    
    def get_statistics(self) -> ProcessingStatistics:
        """
        Get comprehensive processing statistics.
        
        Returns:
            ProcessingStatistics with validated data
        """
        return ProcessingStatistics(
            is_voice_active=self.is_voice_active,
            voice_start_frame_count=self.voice_start_frame_count,
            voice_end_frame_count=self.voice_end_frame_count,
            recent_probabilities=list(self.voice_probabilities),
            average_probability=np.mean(self.voice_probabilities) if self.voice_probabilities else 0.0,
            voice_buffer_size=len(self.voice_buffer),
            current_voice_length=len(self.current_voice_data) if self.current_voice_data is not None else 0
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and diagnostics.
        
        Returns:
            Dictionary with model information
        """
        if self.model:
            return self.model.get_model_info()
        return {
            'model_path': None,
            'model_version': self.config.model_version.value,
            'model_loaded': False
        }
    
    def update_config(self, new_config: VADConfig) -> None:
        """
        Update configuration and reinitialize if necessary.
        
        Args:
            new_config: New VAD configuration
            
        Raises:
            ModelInitializationError: If model reloading fails
        """
        # Check if model-related settings changed
        model_changed = (
            new_config.model_version != self.config.model_version or
            new_config.model_path != self.config.model_path
        )
        
        # Update configuration
        self.config = new_config
        
        # Update WAV writer if output settings changed
        self.wav_writer = WAVWriter(
            sample_rate=new_config.output_wav_sample_rate,
            bit_depth=new_config.output_wav_bit_depth,
            channels=1
        )
        
        # Reload model if necessary
        if model_changed:
            self._load_model()
            
        # Reset processor state
        self.reset()
