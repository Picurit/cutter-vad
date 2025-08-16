"""
Unit tests for Silero VAD model implementation.

This module contains comprehensive unit tests for the SileroVADModel and VADProcessor
classes, including Pydantic model validation, ONNX model interactions, and audio processing.

Test Coverage:
- ModelState: Pydantic model validation for LSTM states (V4/V5 compatibility)
- ProcessingResult: Validation of VAD processing output structure
- ProcessingStatistics: Validation of statistical data collection
- ModelConfiguration: Validation of model configuration parameters
- SileroVADModel: Core model functionality including initialization, prediction, and state management
- VADProcessor: Complete VAD pipeline including voice detection, audio buffering, and state machine logic
- Integration scenarios: End-to-end testing of realistic voice detection cycles

The tests follow pytest best practices including:
- Comprehensive fixtures for test data and mocking
- Parametrized tests where appropriate
- Proper mocking of external dependencies (ONNX Runtime, file system)
- Clear test organization with descriptive names and docstrings
- Validation of both successful and error scenarios
- Integration tests for realistic usage patterns
"""

import os
import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from collections import deque
from typing import List, Dict, Any, Optional

import onnxruntime as ort
from pydantic import ValidationError

from real_time_vad.core.silero_model import (
    ModelState,
    ProcessingResult,
    ProcessingStatistics,
    ModelConfiguration,
    SileroVADModel,
    VADProcessor
)
from real_time_vad.core.config import VADConfig, SampleRate, SileroModelVersion
from real_time_vad.core.exceptions import (
    VADError,
    ModelNotFoundError,
    ModelInitializationError,
    AudioProcessingError,
    CallbackError
)
from real_time_vad.utils.wav_writer import WAVWriter


class TestModelState:
    """Test ModelState Pydantic model."""

    def test_create_v5_model_state(self):
        """Test creating V5 model state with combined state."""
        state_array = np.zeros((2, 1, 128), dtype=np.float32)
        model_state = ModelState(state=state_array)
        
        assert model_state.state is not None
        assert model_state.state.shape == (2, 1, 128)
        assert model_state.state.dtype == np.float32
        assert model_state.hidden_state is None
        assert model_state.cell_state is None

    def test_create_v4_model_state(self):
        """Test creating V4 model state with separate states."""
        hidden_state = np.zeros((2, 1, 64), dtype=np.float32)
        cell_state = np.zeros((2, 1, 64), dtype=np.float32)
        
        model_state = ModelState(
            hidden_state=hidden_state,
            cell_state=cell_state
        )
        
        assert model_state.state is None
        assert model_state.hidden_state is not None
        assert model_state.cell_state is not None
        assert model_state.hidden_state.shape == (2, 1, 64)
        assert model_state.cell_state.shape == (2, 1, 64)

    def test_empty_model_state(self):
        """Test creating empty model state."""
        model_state = ModelState()
        
        assert model_state.state is None
        assert model_state.hidden_state is None
        assert model_state.cell_state is None

    def test_invalid_state_type(self):
        """Test validation fails for invalid state types."""
        with pytest.raises(ValidationError, match="Input should be an instance of ndarray"):
            ModelState(state=[1, 2, 3])

    def test_invalid_state_dtype(self):
        """Test validation fails for invalid state dtype."""
        invalid_state = np.zeros((2, 1, 128), dtype=np.int32)
        with pytest.raises(ValidationError, match="State arrays must be float32"):
            ModelState(state=invalid_state)

    def test_conflicting_states(self):
        """Test validation fails when both combined and separate states are provided."""
        state_array = np.zeros((2, 1, 128), dtype=np.float32)
        hidden_state = np.zeros((2, 1, 64), dtype=np.float32)
        
        with pytest.raises(ValidationError, match="Cannot have both combined state and separate states"):
            ModelState(state=state_array, hidden_state=hidden_state)

    def test_state_assignment_validation(self):
        """Test that assignment validation works."""
        model_state = ModelState()
        
        # Valid assignment
        model_state.state = np.zeros((2, 1, 128), dtype=np.float32)
        assert model_state.state is not None
        
        # Invalid assignment should raise error
        with pytest.raises(ValidationError):
            model_state.state = np.zeros((2, 1, 128), dtype=np.int32)


class TestProcessingResult:
    """Test ProcessingResult Pydantic model."""

    def test_create_processing_result(self):
        """Test creating a valid processing result."""
        wav_data = b"mock_wav_data"
        pcm_data = b"mock_pcm_data"
        
        result = ProcessingResult(
            voice_started=True,
            voice_ended=False,
            voice_continuing=True,
            probability=0.85,
            wav_data=wav_data,
            pcm_data=pcm_data
        )
        
        assert result.voice_started is True
        assert result.voice_ended is False
        assert result.voice_continuing is True
        assert result.probability == 0.85
        assert result.wav_data == wav_data
        assert result.pcm_data == pcm_data

    def test_default_processing_result(self):
        """Test creating processing result with defaults."""
        result = ProcessingResult(probability=0.5)
        
        assert result.voice_started is False
        assert result.voice_ended is False
        assert result.voice_continuing is False
        assert result.probability == 0.5
        assert result.wav_data is None
        assert result.pcm_data is None

    def test_invalid_probability_range(self):
        """Test validation fails for invalid probability values."""
        with pytest.raises(ValidationError):
            ProcessingResult(probability=-0.1)
        
        with pytest.raises(ValidationError):
            ProcessingResult(probability=1.1)

    def test_invalid_audio_data_type(self):
        """Test validation fails for invalid audio data types."""
        with pytest.raises(ValidationError, match="Input should be a valid bytes"):
            ProcessingResult(probability=0.5, wav_data=123)
        
        with pytest.raises(ValidationError, match="Input should be a valid bytes"):
            ProcessingResult(probability=0.5, pcm_data=[1, 2, 3])


class TestProcessingStatistics:
    """Test ProcessingStatistics Pydantic model."""

    def test_create_processing_statistics(self):
        """Test creating valid processing statistics."""
        probabilities = [0.1, 0.5, 0.8, 0.9, 0.7]
        
        stats = ProcessingStatistics(
            is_voice_active=True,
            voice_start_frame_count=5,
            voice_end_frame_count=2,
            recent_probabilities=probabilities,
            average_probability=0.6,
            voice_buffer_size=10,
            current_voice_length=1024
        )
        
        assert stats.is_voice_active is True
        assert stats.voice_start_frame_count == 5
        assert stats.voice_end_frame_count == 2
        assert stats.recent_probabilities == probabilities
        assert stats.average_probability == 0.6
        assert stats.voice_buffer_size == 10
        assert stats.current_voice_length == 1024

    def test_invalid_frame_counts(self):
        """Test validation fails for negative frame counts."""
        with pytest.raises(ValidationError):
            ProcessingStatistics(
                is_voice_active=False,
                voice_start_frame_count=-1,
                voice_end_frame_count=0,
                average_probability=0.5,
                voice_buffer_size=0,
                current_voice_length=0
            )

    def test_invalid_probabilities_in_list(self):
        """Test validation fails for invalid probabilities in list."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 1.0"):
            ProcessingStatistics(
                is_voice_active=False,
                voice_start_frame_count=0,
                voice_end_frame_count=0,
                recent_probabilities=[0.5, 1.5, 0.8],  # 1.5 is invalid
                average_probability=0.5,
                voice_buffer_size=0,
                current_voice_length=0
            )


class TestModelConfiguration:
    """Test ModelConfiguration Pydantic model."""

    def test_create_valid_configuration(self, temp_directory):
        """Test creating valid model configuration."""
        # Create a mock ONNX file
        model_path = temp_directory / "test_model.onnx"
        model_path.write_bytes(b"mock_onnx_content")
        
        config = ModelConfiguration(
            model_path=str(model_path),
            model_version=SileroModelVersion.V5
        )
        
        assert config.model_path == str(model_path)
        assert config.model_version == SileroModelVersion.V5

    def test_nonexistent_model_path(self):
        """Test validation fails for nonexistent model path."""
        with pytest.raises(ModelNotFoundError):
            ModelConfiguration(
                model_path="/nonexistent/path/model.onnx",
                model_version=SileroModelVersion.V5
            )

    def test_directory_instead_of_file(self, temp_directory):
        """Test validation fails when model path is a directory."""
        with pytest.raises(ValidationError, match="Model path must be a file"):
            ModelConfiguration(
                model_path=str(temp_directory),
                model_version=SileroModelVersion.V5
            )

    def test_wrong_file_extension(self, temp_directory):
        """Test validation fails for wrong file extension."""
        wrong_file = temp_directory / "model.txt"
        wrong_file.write_text("not an onnx file")
        
        with pytest.raises(ValidationError, match="Model file must have .onnx extension"):
            ModelConfiguration(
                model_path=str(wrong_file),
                model_version=SileroModelVersion.V5
            )


class TestSileroVADModel:
    """Test SileroVADModel class."""

    @pytest.fixture
    def mock_onnx_session(self):
        """Create mock ONNX runtime session."""
        session = Mock(spec=ort.InferenceSession)
        session.get_providers.return_value = ['CPUExecutionProvider']
        
        # Mock inputs and outputs for validation
        mock_input = Mock()
        mock_input.name = 'input'
        mock_output = Mock()
        mock_output.name = 'output'
        
        session.get_inputs.return_value = [mock_input, mock_input, mock_input]  # V5: 3 inputs
        session.get_outputs.return_value = [mock_output, mock_output]  # V5: 2 outputs
        
        return session

    @pytest.fixture
    def mock_model_file(self, temp_directory):
        """Create mock model file."""
        model_path = temp_directory / "silero_vad_v5.onnx"
        model_path.write_bytes(b"mock_onnx_model_data")
        return str(model_path)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_model_initialization_v5(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test successful V5 model initialization."""
        mock_session_class.return_value = mock_onnx_session
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        
        assert model.config.model_path == mock_model_file
        assert model.config.model_version == SileroModelVersion.V5
        assert model.session == mock_onnx_session
        assert model.model_state.state is not None
        assert model.model_state.state.shape == (2, 1, 128)
        assert model.prediction_count == 0

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_model_initialization_v4(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test successful V4 model initialization."""
        # Adjust mock for V4 model (4 inputs, 3 outputs)
        mock_onnx_session.get_inputs.return_value = [Mock(), Mock(), Mock(), Mock()]
        mock_onnx_session.get_outputs.return_value = [Mock(), Mock(), Mock()]
        
        mock_session_class.return_value = mock_onnx_session
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V4)
        
        assert model.config.model_version == SileroModelVersion.V4
        assert model.model_state.hidden_state is not None
        assert model.model_state.cell_state is not None
        assert model.model_state.hidden_state.shape == (2, 1, 64)
        assert model.model_state.cell_state.shape == (2, 1, 64)

    def test_model_initialization_file_not_found(self):
        """Test model initialization fails for missing file."""
        with pytest.raises(ModelNotFoundError):
            SileroVADModel("/nonexistent/model.onnx", SileroModelVersion.V5)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_model_initialization_session_error(self, mock_session_class, mock_model_file):
        """Test model initialization fails when ONNX session creation fails."""
        mock_session_class.side_effect = Exception("ONNX Runtime error")
        
        with pytest.raises(ModelInitializationError, match="Failed to load model"):
            SileroVADModel(mock_model_file, SileroModelVersion.V5)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_invalid_model_signature(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test validation fails for invalid model signature."""
        # Mock incorrect number of inputs/outputs
        mock_onnx_session.get_inputs.return_value = [Mock(), Mock()]  # Wrong count for V5
        mock_onnx_session.get_outputs.return_value = [Mock()]  # Wrong count for V5
        
        mock_session_class.return_value = mock_onnx_session
        
        with pytest.raises(ModelInitializationError, match="Model signature validation failed"):
            SileroVADModel(mock_model_file, SileroModelVersion.V5)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_predict_success_v5(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test successful prediction with V5 model."""
        # Setup mock session
        mock_session_class.return_value = mock_onnx_session
        mock_onnx_session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),  # probability
            np.zeros((2, 1, 128), dtype=np.float32)  # new state
        ]
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        
        # Create test audio
        audio_chunk = np.random.randn(512).astype(np.float32)
        
        probability = model.predict(audio_chunk, 16000)
        
        assert probability == pytest.approx(0.8, abs=1e-6)
        assert model.prediction_count == 1
        
        # Verify session.run was called with correct inputs
        mock_onnx_session.run.assert_called_once()
        call_args = mock_onnx_session.run.call_args[0][1]  # Second argument (inputs dict)
        assert 'input' in call_args
        assert 'state' in call_args
        assert 'sr' in call_args

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_predict_invalid_audio_length(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test prediction with invalid audio length."""
        mock_session_class.return_value = mock_onnx_session
        mock_onnx_session.run.return_value = [
            np.array([[0.8]], dtype=np.float32),
            np.zeros((2, 1, 128), dtype=np.float32)
        ]
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        
        # Test with wrong length (will be padded/truncated)
        audio_chunk = np.random.randn(256).astype(np.float32)  # Too short
        probability = model.predict(audio_chunk, 16000)
        
        assert isinstance(probability, float)
        
        # Test with too long (will be truncated)
        audio_chunk = np.random.randn(1024).astype(np.float32)  # Too long
        probability = model.predict(audio_chunk, 16000)
        
        assert isinstance(probability, float)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_predict_session_error(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test prediction fails when session.run fails."""
        mock_session_class.return_value = mock_onnx_session
        mock_onnx_session.run.side_effect = Exception("Runtime error")
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        audio_chunk = np.random.randn(512).astype(np.float32)
        
        with pytest.raises(AudioProcessingError, match="Model prediction failed"):
            model.predict(audio_chunk, 16000)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_predict_invalid_probability_output(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test prediction fails when model returns invalid probability."""
        mock_session_class.return_value = mock_onnx_session
        mock_onnx_session.run.return_value = [
            np.array([[1.5]], dtype=np.float32),  # Invalid probability > 1.0
            np.zeros((2, 1, 128), dtype=np.float32)
        ]
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        audio_chunk = np.random.randn(512).astype(np.float32)
        
        with pytest.raises(AudioProcessingError, match="Probability extraction failed"):
            model.predict(audio_chunk, 16000)

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_reset_model_state(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test resetting model state."""
        mock_session_class.return_value = mock_onnx_session
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        
        # Modify state
        model.model_state.state = np.ones((2, 1, 128), dtype=np.float32)
        model.prediction_count = 10
        
        # Reset
        model.reset()
        
        # Verify state is reset but prediction count is preserved
        assert np.allclose(model.model_state.state, np.zeros((2, 1, 128), dtype=np.float32))
        assert model.prediction_count == 10  # This should not be reset

    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_get_model_info(self, mock_session_class, mock_model_file, mock_onnx_session):
        """Test getting model information."""
        mock_session_class.return_value = mock_onnx_session
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        model.prediction_count = 5
        
        info = model.get_model_info()
        
        assert info['model_path'] == mock_model_file
        assert info['model_version'] == 'v5'
        assert info['prediction_count'] == 5
        assert info['session_providers'] == ['CPUExecutionProvider']
        assert info['has_cuda'] is False
        assert 'state_shape' in info

    @patch('real_time_vad.core.silero_model.ort.get_available_providers')
    @patch('real_time_vad.core.silero_model.ort.InferenceSession')
    def test_cuda_provider_selection(self, mock_session_class, mock_providers, mock_model_file, mock_onnx_session):
        """Test CUDA provider is selected when available."""
        mock_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        mock_session_class.return_value = mock_onnx_session
        
        model = SileroVADModel(mock_model_file, SileroModelVersion.V5)
        
        # Verify CUDA provider was prioritized in session creation
        mock_session_class.assert_called_once()
        call_kwargs = mock_session_class.call_args[1]
        assert 'CUDAExecutionProvider' in call_kwargs['providers']
        assert call_kwargs['providers'].index('CUDAExecutionProvider') < call_kwargs['providers'].index('CPUExecutionProvider')


class TestVADProcessor:
    """Test VADProcessor class."""

    @pytest.fixture
    def mock_vad_config(self):
        """Create mock VAD configuration."""
        return VADConfig(
            model_version=SileroModelVersion.V5,
            vad_start_probability=0.7,
            vad_end_probability=0.3,
            voice_start_frame_count=3,
            voice_end_frame_count=5
        )

    @pytest.fixture
    def mock_model_directory(self, temp_directory):
        """Create mock model directory with model files."""
        model_dir = temp_directory / "models"
        model_dir.mkdir()
        
        # Create mock model files
        (model_dir / "silero_vad_v5.onnx").write_bytes(b"mock_v5_model")
        (model_dir / "silero_vad.onnx").write_bytes(b"mock_v4_model")
        
        return model_dir

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    @patch('real_time_vad.utils.audio.AudioUtils.validate_audio_data')
    @patch('real_time_vad.utils.audio.AudioUtils.denoise_audio')
    def test_vad_processor_initialization(self, mock_denoise, mock_validate, mock_model_class, 
                                        mock_vad_config, mock_model_directory):
        """Test VAD processor initialization."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_class.return_value = mock_model_instance
        mock_validate.return_value = None
        mock_denoise.side_effect = lambda x: x
        
        # Patch the model directory path
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                assert processor.config == mock_vad_config
                assert processor.model == mock_model_instance
                assert processor.is_voice_active is False
                assert processor.voice_start_frame_count == 0
                assert processor.voice_end_frame_count == 0
                assert len(processor.voice_probabilities) == 0
                assert isinstance(processor.wav_writer, WAVWriter)

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    @patch('real_time_vad.utils.audio.AudioUtils.validate_audio_data')
    @patch('real_time_vad.utils.audio.AudioUtils.denoise_audio')
    def test_process_frame_voice_start(self, mock_denoise, mock_validate, mock_model_class, 
                                     mock_vad_config, mock_model_directory):
        """Test processing frame that starts voice activity."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_instance.predict.return_value = 0.8  # Above start threshold
        mock_model_class.return_value = mock_model_instance
        mock_validate.return_value = None
        mock_denoise.side_effect = lambda x: x
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                # Process frames to trigger voice start
                audio_frame = np.random.randn(512).astype(np.float32)
                
                # First frame - should not start yet
                result1 = processor.process_frame(audio_frame)
                assert result1.voice_started is False
                assert result1.probability == 0.8
                
                # Process enough frames to trigger start
                for _ in range(mock_vad_config.voice_start_frame_count - 1):
                    result = processor.process_frame(audio_frame)
                
                # Should have started voice activity
                assert result.voice_started is True
                assert processor.is_voice_active is True

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    @patch('real_time_vad.utils.audio.AudioUtils.validate_audio_data')
    @patch('real_time_vad.utils.audio.AudioUtils.denoise_audio')
    def test_process_frame_voice_end(self, mock_denoise, mock_validate, mock_model_class, 
                                   mock_vad_config, mock_model_directory):
        """Test processing frame that ends voice activity."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_class.return_value = mock_model_instance
        mock_validate.return_value = None
        mock_denoise.side_effect = lambda x: x
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                # Manually set voice active state
                processor.is_voice_active = True
                processor.current_voice_data = np.random.randn(1024).astype(np.float32)
                
                # Mock low probability (below end threshold)
                mock_model_instance.predict.return_value = 0.2
                
                # Mock WAV writer
                mock_wav_data = b"mock_wav_data"
                processor.wav_writer.write_wav_data = Mock(return_value=mock_wav_data)
                
                audio_frame = np.random.randn(512).astype(np.float32)
                
                # Process frames to trigger voice end
                for _ in range(mock_vad_config.voice_end_frame_count):
                    result = processor.process_frame(audio_frame)
                
                # Should have ended voice activity
                assert result.voice_ended is True
                assert result.wav_data == mock_wav_data
                assert processor.is_voice_active is False

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    @patch('real_time_vad.utils.audio.AudioUtils.validate_audio_data')
    def test_process_frame_voice_continuing(self, mock_validate, mock_model_class, 
                                          mock_vad_config, mock_model_directory):
        """Test processing frame during ongoing voice activity."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_instance.predict.return_value = 0.8  # High probability
        mock_model_class.return_value = mock_model_instance
        mock_validate.return_value = None
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                # Manually set voice active state
                processor.is_voice_active = True
                processor.current_voice_data = np.random.randn(512).astype(np.float32)
                
                audio_frame = np.random.randn(512).astype(np.float32)
                result = processor.process_frame(audio_frame)
                
                assert result.voice_continuing is True
                assert result.voice_started is False
                assert result.voice_ended is False
                assert result.pcm_data is not None
                assert len(processor.current_voice_data) == 1024  # Original + new frame

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    def test_process_frame_model_error(self, mock_model_class, mock_vad_config, mock_model_directory):
        """Test processing frame when model prediction fails."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_instance.predict.side_effect = Exception("Model error")
        mock_model_class.return_value = mock_model_instance
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                audio_frame = np.random.randn(512).astype(np.float32)
                
                with pytest.raises(AudioProcessingError, match="Frame processing failed"):
                    processor.process_frame(audio_frame)

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    def test_reset_processor(self, mock_model_class, mock_vad_config, mock_model_directory):
        """Test resetting VAD processor."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_class.return_value = mock_model_instance
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                # Set some state
                processor.is_voice_active = True
                processor.voice_start_frame_count = 5
                processor.voice_end_frame_count = 3
                processor.voice_probabilities.extend([0.1, 0.5, 0.8])
                processor.voice_buffer.extend([np.array([1, 2, 3])])
                processor.current_voice_data = np.array([1, 2, 3, 4, 5])
                
                # Reset
                processor.reset()
                
                # Verify all state is reset
                assert processor.is_voice_active is False
                assert processor.voice_start_frame_count == 0
                assert processor.voice_end_frame_count == 0
                assert len(processor.voice_probabilities) == 0
                assert len(processor.voice_buffer) == 0
                assert processor.current_voice_data is None
                
                # Verify model reset was called
                mock_model_instance.reset.assert_called_once()

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    def test_get_statistics(self, mock_model_class, mock_vad_config, mock_model_directory):
        """Test getting processing statistics."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_class.return_value = mock_model_instance
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                # Set some state
                processor.is_voice_active = True
                processor.voice_start_frame_count = 3
                processor.voice_end_frame_count = 2
                processor.voice_probabilities.extend([0.2, 0.6, 0.8, 0.9])
                processor.voice_buffer.extend([np.array([1]), np.array([2])])
                processor.current_voice_data = np.array([1, 2, 3, 4, 5])
                
                stats = processor.get_statistics()
                
                assert isinstance(stats, ProcessingStatistics)
                assert stats.is_voice_active is True
                assert stats.voice_start_frame_count == 3
                assert stats.voice_end_frame_count == 2
                assert stats.recent_probabilities == [0.2, 0.6, 0.8, 0.9]
                assert stats.average_probability == 0.625
                assert stats.voice_buffer_size == 2
                assert stats.current_voice_length == 5

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    def test_get_model_info(self, mock_model_class, mock_vad_config, mock_model_directory):
        """Test getting model information."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_info = {
            'model_path': '/path/to/model.onnx',
            'model_version': 'V5',
            'prediction_count': 10
        }
        mock_model_instance.get_model_info.return_value = mock_info
        mock_model_class.return_value = mock_model_instance
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                info = processor.get_model_info()
                
                assert info == mock_info
                mock_model_instance.get_model_info.assert_called_once()

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    def test_update_config(self, mock_model_class, mock_vad_config, mock_model_directory):
        """Test updating processor configuration."""
        # Setup mocks
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_class.return_value = mock_model_instance
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                processor = VADProcessor(mock_vad_config)
                
                # Create new config with different model version
                new_config = VADConfig(
                    model_version=SileroModelVersion.V4,
                    vad_start_probability=0.8,
                    output_wav_sample_rate=22050
                )
                
                # Reset call count
                mock_model_class.reset_mock()
                
                processor.update_config(new_config)
                
                # Should have reloaded model due to version change
                assert mock_model_class.call_count == 1
                assert processor.config == new_config
                assert processor.wav_writer.sample_rate == 22050

    def test_model_not_found_error(self, temp_directory):
        """Test initialization fails when model file is not found."""
        # Create a config with valid directory but missing model file
        config = VADConfig(
            model_version=SileroModelVersion.V5,
            model_path=temp_directory  # Valid directory but no model files
        )
        
        with pytest.raises(ModelNotFoundError):
            VADProcessor(config)

    @patch('real_time_vad.core.silero_model.SileroVADModel')
    def test_processing_with_denoising_enabled(self, mock_model_class, mock_model_directory):
        """Test frame processing with denoising enabled."""
        # Setup config with denoising
        config = VADConfig(
            model_version=SileroModelVersion.V5,
            enable_denoising=True
        )
        
        mock_model_instance = Mock(spec=SileroVADModel)
        mock_model_instance.predict.return_value = 0.5
        mock_model_class.return_value = mock_model_instance
        
        with patch.object(Path, 'exists', return_value=True):
            with patch('real_time_vad.core.silero_model.Path') as mock_path:
                mock_path.return_value = mock_model_directory
                mock_path.parent.parent = mock_model_directory.parent
                
                with patch('real_time_vad.utils.audio.AudioUtils.validate_audio_data') as mock_validate:
                    with patch('real_time_vad.utils.audio.AudioUtils.denoise_audio') as mock_denoise:
                        mock_validate.return_value = None
                        mock_denoise.return_value = np.random.randn(512).astype(np.float32)
                        
                        processor = VADProcessor(config)
                        
                        audio_frame = np.random.randn(512).astype(np.float32)
                        result = processor.process_frame(audio_frame)
                        
                        # Verify denoising was called
                        mock_denoise.assert_called_once()
                        assert result.probability == 0.5


class TestIntegrationScenarios:
    """Integration tests for realistic VAD scenarios."""

    @pytest.fixture
    def mock_realistic_processor(self, temp_directory):
        """Create a realistic VAD processor for integration testing."""
        # Create mock model directory
        mock_model_directory = temp_directory / "models"
        mock_model_directory.mkdir()
        (mock_model_directory / "silero_vad_v5.onnx").write_bytes(b"mock_v5_model")
        
        config = VADConfig(
            model_version=SileroModelVersion.V5,
            vad_start_probability=0.6,
            vad_end_probability=0.4,
            voice_start_frame_count=3,
            voice_end_frame_count=5
        )
        
        with patch('real_time_vad.core.silero_model.SileroVADModel') as mock_model_class:
            mock_model_instance = Mock(spec=SileroVADModel)
            mock_model_class.return_value = mock_model_instance
            
            with patch.object(Path, 'exists', return_value=True):
                with patch('real_time_vad.core.silero_model.Path') as mock_path:
                    mock_path.return_value = mock_model_directory
                    mock_path.parent.parent = mock_model_directory.parent
                    
                    with patch('real_time_vad.utils.audio.AudioUtils.validate_audio_data'):
                        processor = VADProcessor(config)
                        processor.model = mock_model_instance
                        
                        return processor, mock_model_instance

    def test_complete_voice_detection_cycle(self, mock_realistic_processor):
        """Test complete voice detection cycle: silence -> voice -> silence."""
        processor, mock_model = mock_realistic_processor
        
        # Mock WAV writer
        mock_wav_data = b"mock_wav_file_data"
        processor.wav_writer.write_wav_data = Mock(return_value=mock_wav_data)
        
        audio_frame = np.random.randn(512).astype(np.float32)
        results = []
        
        # Phase 1: Silence (low probabilities)
        mock_model.predict.return_value = 0.1
        for _ in range(5):
            result = processor.process_frame(audio_frame)
            results.append(result)
            assert result.voice_started is False
            assert result.voice_ended is False
        
        # Phase 2: Voice start (high probabilities)
        mock_model.predict.return_value = 0.8
        for i in range(5):
            result = processor.process_frame(audio_frame)
            results.append(result)
            if i >= 2:  # After voice_start_frame_count frames
                assert processor.is_voice_active is True
                if i == 2:
                    assert result.voice_started is True
                else:
                    assert result.voice_continuing is True
        
        # Phase 3: Ongoing voice activity
        mock_model.predict.return_value = 0.7
        for _ in range(10):
            result = processor.process_frame(audio_frame)
            results.append(result)
            assert result.voice_continuing is True
            assert result.pcm_data is not None
        
        # Phase 4: Voice end (low probabilities)
        mock_model.predict.return_value = 0.2
        voice_ended = False
        for i in range(8):
            result = processor.process_frame(audio_frame)
            results.append(result)
            if i >= 4:  # After voice_end_frame_count frames
                assert result.voice_ended is True
                assert result.wav_data == mock_wav_data
                voice_ended = True
                break
        
        assert voice_ended
        assert processor.is_voice_active is False

    def test_multiple_voice_segments(self, mock_realistic_processor):
        """Test detection of multiple voice segments."""
        processor, mock_model = mock_realistic_processor
        
        # Mock WAV writer
        mock_wav_data = b"mock_wav_segment"
        processor.wav_writer.write_wav_data = Mock(return_value=mock_wav_data)
        
        audio_frame = np.random.randn(512).astype(np.float32)
        voice_segments_detected = 0
        
        for cycle in range(3):  # Three voice segments
            # Silence
            mock_model.predict.return_value = 0.1
            for _ in range(10):
                processor.process_frame(audio_frame)
            
            # Voice
            mock_model.predict.return_value = 0.8
            for _ in range(15):
                processor.process_frame(audio_frame)
            
            # End voice
            mock_model.predict.return_value = 0.2
            for i in range(8):
                result = processor.process_frame(audio_frame)
                if result.voice_ended:
                    voice_segments_detected += 1
                    break
        
        assert voice_segments_detected == 3

    def test_voice_activity_statistics_tracking(self, mock_realistic_processor):
        """Test that statistics are properly tracked during voice activity."""
        processor, mock_model = mock_realistic_processor
        
        audio_frame = np.random.randn(512).astype(np.float32)
        
        # Process frames with varying probabilities
        probabilities = [0.1, 0.3, 0.7, 0.9, 0.8, 0.5, 0.2, 0.1]
        
        for prob in probabilities:
            mock_model.predict.return_value = prob
            processor.process_frame(audio_frame)
        
        stats = processor.get_statistics()
        
        assert len(stats.recent_probabilities) == len(probabilities)
        assert stats.recent_probabilities == probabilities
        assert abs(stats.average_probability - np.mean(probabilities)) < 0.001
        
        # Check that statistics are valid ProcessingStatistics
        assert isinstance(stats, ProcessingStatistics)
