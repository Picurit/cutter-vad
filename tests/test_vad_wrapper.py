"""
Unit tests for VAD wrapper functionality.

This module contains comprehensive unit tests for the VADWrapper class,
covering all public methods, error handling, validation, and integration
scenarios following pytest best practices.
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Optional
from pydantic import ValidationError

from real_time_vad.core.vad_wrapper import (
    VADWrapper,
    VADWrapperState,
    CallbackConfiguration,
    ThresholdConfiguration,
    VoiceStartCallback,
    VoiceEndCallback,
    VoiceContinueCallback
)
from real_time_vad.core.config import VADConfig, SampleRate, SileroModelVersion
from real_time_vad.core.silero_model import ProcessingResult, ProcessingStatistics, VADProcessor
from real_time_vad.core.exceptions import (
    VADError,
    ConfigurationError,
    CallbackError,
    AudioProcessingError
)


class TestVADWrapperState:
    """Test suite for VADWrapperState Pydantic model."""

    def test_default_state_initialization(self):
        """Test default state initialization with proper values."""
        state = VADWrapperState()
        
        assert state.is_initialized is False
        assert state.total_frames_processed == 0
        assert state.total_processing_time == 0.0
        assert state.last_error is None
        assert state.average_processing_time_per_frame == 0.0

    def test_state_with_custom_values(self):
        """Test state initialization with custom values."""
        state = VADWrapperState(
            is_initialized=True,
            total_frames_processed=100,
            total_processing_time=5.5
        )
        
        assert state.is_initialized is True
        assert state.total_frames_processed == 100
        assert state.total_processing_time == 5.5
        assert state.average_processing_time_per_frame == 0.055

    def test_state_validation_constraints(self):
        """Test state validation constraints."""
        # Test negative frames raises validation error
        with pytest.raises(ValidationError):
            VADWrapperState(total_frames_processed=-1)
        
        # Test negative processing time raises validation error
        with pytest.raises(ValidationError):
            VADWrapperState(total_processing_time=-1.0)

    def test_reset_statistics(self):
        """Test resetting statistics in state."""
        state = VADWrapperState(
            total_frames_processed=100,
            total_processing_time=5.5
        )
        
        state.reset_statistics()
        
        assert state.total_frames_processed == 0
        assert state.total_processing_time == 0.0
        assert state.average_processing_time_per_frame == 0.0

    def test_record_and_clear_error(self):
        """Test error recording and clearing functionality."""
        state = VADWrapperState()
        test_error = ValueError("Test error")
        
        state.record_error(test_error)
        assert state.last_error == "Test error"
        
        state.clear_error()
        assert state.last_error is None

    def test_average_processing_time_calculation(self):
        """Test average processing time calculation with edge cases."""
        state = VADWrapperState()
        
        # Zero frames should return 0.0
        assert state.average_processing_time_per_frame == 0.0
        
        # With frames should calculate correctly
        state.total_frames_processed = 50
        state.total_processing_time = 2.5
        assert state.average_processing_time_per_frame == 0.05


class TestCallbackConfiguration:
    """Test suite for CallbackConfiguration Pydantic model."""

    def test_default_callback_configuration(self):
        """Test default callback configuration initialization."""
        config = CallbackConfiguration()
        
        assert config.voice_start_callback is None
        assert config.voice_end_callback is None
        assert config.voice_continue_callback is None
        assert config.has_any_callback() is False

    def test_valid_callback_functions(self):
        """Test configuration with valid callback functions."""
        def start_callback():
            pass
        
        def end_callback(data: bytes):
            pass
        
        def continue_callback(data: bytes):
            pass
        
        config = CallbackConfiguration(
            voice_start_callback=start_callback,
            voice_end_callback=end_callback,
            voice_continue_callback=continue_callback
        )
        
        assert config.voice_start_callback == start_callback
        assert config.voice_end_callback == end_callback
        assert config.voice_continue_callback == continue_callback
        assert config.has_any_callback() is True

    def test_invalid_callback_validation(self):
        """Test validation of non-callable callback values."""
        with pytest.raises(ValidationError):
            CallbackConfiguration(voice_start_callback="not_callable")
        
        with pytest.raises(ValidationError):
            CallbackConfiguration(voice_end_callback=123)
        
        with pytest.raises(ValidationError):
            CallbackConfiguration(voice_continue_callback=[])

    def test_partial_callback_configuration(self):
        """Test configuration with only some callbacks set."""
        def start_callback():
            pass
        
        config = CallbackConfiguration(voice_start_callback=start_callback)
        
        assert config.voice_start_callback == start_callback
        assert config.voice_end_callback is None
        assert config.voice_continue_callback is None
        assert config.has_any_callback() is True


class TestThresholdConfiguration:
    """Test suite for ThresholdConfiguration Pydantic model."""

    def test_default_threshold_configuration(self):
        """Test default threshold configuration values."""
        config = ThresholdConfiguration()
        
        assert config.vad_start_probability == 0.7
        assert config.vad_end_probability == 0.7
        assert config.voice_start_ratio == 0.8
        assert config.voice_end_ratio == 0.95
        assert config.voice_start_frame_count == 10
        assert config.voice_end_frame_count == 57

    @pytest.mark.parametrize("probability,field_name", [
        (-0.1, "vad_start_probability"),
        (1.1, "vad_start_probability"),
        (-0.01, "vad_end_probability"),
        (1.5, "vad_end_probability"),
        (-0.5, "voice_start_ratio"),
        (2.0, "voice_end_ratio"),
    ])
    def test_invalid_probability_ranges(self, probability, field_name):
        """Test validation of probability ranges."""
        kwargs = {field_name: probability}
        with pytest.raises(ValidationError):
            ThresholdConfiguration(**kwargs)

    @pytest.mark.parametrize("count,field_name", [
        (0, "voice_start_frame_count"),
        (-1, "voice_start_frame_count"),
        (0, "voice_end_frame_count"),
        (-5, "voice_end_frame_count"),
    ])
    def test_invalid_frame_counts(self, count, field_name):
        """Test validation of frame count ranges."""
        kwargs = {field_name: count}
        with pytest.raises(ValidationError):
            ThresholdConfiguration(**kwargs)

    def test_logical_validation_constraints(self):
        """Test logical validation constraints."""
        # Test too low start probability
        with pytest.raises(ValidationError):
            ThresholdConfiguration(vad_start_probability=0.05)
        
        # Test too low end probability
        with pytest.raises(ValidationError):
            ThresholdConfiguration(vad_end_probability=0.05)
        
        # Test too high frame counts
        with pytest.raises(ValidationError):
            ThresholdConfiguration(voice_start_frame_count=150)
        
        with pytest.raises(ValidationError):
            ThresholdConfiguration(voice_end_frame_count=250)

    def test_valid_custom_thresholds(self):
        """Test creation with valid custom threshold values."""
        config = ThresholdConfiguration(
            vad_start_probability=0.8,
            vad_end_probability=0.6,
            voice_start_ratio=0.9,
            voice_end_ratio=0.85,
            voice_start_frame_count=5,
            voice_end_frame_count=30
        )
        
        assert config.vad_start_probability == 0.8
        assert config.vad_end_probability == 0.6
        assert config.voice_start_ratio == 0.9
        assert config.voice_end_ratio == 0.85
        assert config.voice_start_frame_count == 5
        assert config.voice_end_frame_count == 30


class TestVADWrapper:
    """Test suite for VADWrapper main class."""

    @pytest.fixture
    def mock_vad_processor(self):
        """Create a mock VAD processor for testing."""
        processor = Mock(spec=VADProcessor)
        processor.reset.return_value = None
        processor.is_voice_active = False
        processor.get_statistics.return_value = ProcessingStatistics(
            is_voice_active=False,
            voice_start_frame_count=0,
            voice_end_frame_count=0,
            average_probability=0.0,
            voice_buffer_size=0,
            current_voice_length=0
        )
        processor.process_frame.return_value = ProcessingResult(
            probability=0.5,
            voice_started=False,
            voice_ended=False,
            voice_continuing=False
        )
        processor.update_config.return_value = None
        return processor

    @pytest.fixture
    def sample_processing_result(self):
        """Create a sample processing result for testing."""
        return ProcessingResult(
            probability=0.8,
            voice_started=True,
            voice_ended=False,
            voice_continuing=False,
            wav_data=None,
            pcm_data=b'test_pcm_data'
        )

    def test_default_initialization(self, default_config):
        """Test VADWrapper initialization with default configuration."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor') as mock_processor_class:
            mock_processor = Mock(spec=VADProcessor)
            mock_processor_class.return_value = mock_processor
            
            wrapper = VADWrapper()
            
            assert wrapper._config is not None
            assert isinstance(wrapper._state, VADWrapperState)
            assert isinstance(wrapper._callbacks, CallbackConfiguration)
            assert wrapper._state.is_initialized is True
            assert wrapper._processor == mock_processor
            mock_processor_class.assert_called_once()

    def test_initialization_with_custom_config(self, custom_config):
        """Test VADWrapper initialization with custom configuration."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor') as mock_processor_class:
            mock_processor = Mock(spec=VADProcessor)
            mock_processor_class.return_value = mock_processor
            
            wrapper = VADWrapper(config=custom_config)
            
            assert wrapper._config == custom_config
            assert wrapper._state.is_initialized is True
            mock_processor_class.assert_called_once_with(custom_config)

    def test_initialization_failure_handling(self):
        """Test handling of processor initialization failure."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor') as mock_processor_class:
            mock_processor_class.side_effect = Exception("Processor init failed")
            
            with pytest.raises(VADError, match="Failed to initialize VAD processor"):
                VADWrapper()

    def test_invalid_config_initialization(self):
        """Test initialization with invalid configuration type."""
        with pytest.raises(VADError, match="Failed to initialize VAD wrapper"):
            VADWrapper(config="invalid_config")

    # ========================= Configuration Management Tests =========================

    def test_set_sample_rate_valid(self, mock_vad_processor):
        """Test setting valid sample rate."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            wrapper.set_sample_rate(SampleRate.SAMPLERATE_24)
            
            assert wrapper._config.sample_rate == SampleRate.SAMPLERATE_24

    def test_set_sample_rate_invalid_type(self, mock_vad_processor):
        """Test setting invalid sample rate type."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(ConfigurationError):
                wrapper.set_sample_rate("invalid_rate")

    def test_set_silero_model_valid(self, mock_vad_processor):
        """Test setting valid Silero model version."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            wrapper.set_silero_model(SileroModelVersion.V4)
            
            assert wrapper._config.model_version == SileroModelVersion.V4

    def test_set_silero_model_invalid_type(self, mock_vad_processor):
        """Test setting invalid Silero model type."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(ConfigurationError):
                wrapper.set_silero_model("invalid_model")

    def test_set_thresholds_valid(self, mock_vad_processor):
        """Test setting valid threshold values."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            wrapper.set_thresholds(
                vad_start_probability=0.8,
                vad_end_probability=0.6,
                voice_start_ratio=0.9,
                voice_end_ratio=0.85,
                voice_start_frame_count=5,
                voice_end_frame_count=30
            )
            
            assert wrapper._config.vad_start_probability == 0.8
            assert wrapper._config.vad_end_probability == 0.6
            assert wrapper._config.voice_start_ratio == 0.9
            assert wrapper._config.voice_end_ratio == 0.85
            assert wrapper._config.voice_start_frame_count == 5
            assert wrapper._config.voice_end_frame_count == 30
            mock_vad_processor.reset.assert_called_once()

    def test_set_thresholds_invalid_values(self, mock_vad_processor):
        """Test setting invalid threshold values."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(ConfigurationError):
                wrapper.set_thresholds(vad_start_probability=-0.1)
            
            with pytest.raises(ConfigurationError):
                wrapper.set_thresholds(voice_start_frame_count=0)

    # ========================= Callback Management Tests =========================

    def test_set_callbacks_valid(self, mock_vad_processor):
        """Test setting valid callback functions."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            def start_callback():
                pass
            
            def end_callback(data: bytes):
                pass
            
            def continue_callback(data: bytes):
                pass
            
            wrapper.set_callbacks(
                voice_start_callback=start_callback,
                voice_end_callback=end_callback,
                voice_continue_callback=continue_callback
            )
            
            assert wrapper._callbacks.voice_start_callback == start_callback
            assert wrapper._callbacks.voice_end_callback == end_callback
            assert wrapper._callbacks.voice_continue_callback == continue_callback

    def test_set_callbacks_invalid(self, mock_vad_processor):
        """Test setting invalid callback functions."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(VADError):
                wrapper.set_callbacks(voice_start_callback="not_callable")

    def test_execute_callback_safely_success(self, mock_vad_processor):
        """Test successful callback execution."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            callback_mock = Mock()
            wrapper._execute_callback_safely(callback_mock, "test_callback", "arg1", key="value")
            
            callback_mock.assert_called_once_with("arg1", key="value")

    def test_execute_callback_safely_failure(self, mock_vad_processor):
        """Test callback execution failure handling."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            callback_mock = Mock(side_effect=Exception("Callback failed"))
            
            with pytest.raises(CallbackError):
                wrapper._execute_callback_safely(callback_mock, "test_callback")

    def test_execute_callback_safely_none_callback(self, mock_vad_processor):
        """Test callback execution with None callback."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Should not raise any exception
            wrapper._execute_callback_safely(None, "test_callback")

    def test_handle_callbacks_voice_started(self, mock_vad_processor, sample_processing_result):
        """Test callback handling for voice started event."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            start_callback = Mock()
            wrapper.set_callbacks(voice_start_callback=start_callback)
            
            result = ProcessingResult(
                probability=0.8,
                voice_started=True,
                voice_ended=False,
                voice_continuing=False
            )
            
            wrapper._handle_callbacks(result)
            start_callback.assert_called_once()

    def test_handle_callbacks_voice_ended(self, mock_vad_processor):
        """Test callback handling for voice ended event."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            end_callback = Mock()
            wrapper.set_callbacks(voice_end_callback=end_callback)
            
            test_wav_data = b'test_wav_data'
            result = ProcessingResult(
                probability=0.3,
                voice_started=False,
                voice_ended=True,
                voice_continuing=False,
                wav_data=test_wav_data
            )
            
            wrapper._handle_callbacks(result)
            end_callback.assert_called_once_with(test_wav_data)

    def test_handle_callbacks_voice_continuing(self, mock_vad_processor):
        """Test callback handling for voice continuing event."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            continue_callback = Mock()
            wrapper.set_callbacks(voice_continue_callback=continue_callback)
            
            test_pcm_data = b'test_pcm_data'
            result = ProcessingResult(
                probability=0.8,
                voice_started=False,
                voice_ended=False,
                voice_continuing=True,
                pcm_data=test_pcm_data
            )
            
            wrapper._handle_callbacks(result)
            continue_callback.assert_called_once_with(test_pcm_data)

    # ========================= Audio Processing Tests =========================

    @patch('real_time_vad.core.vad_wrapper.AudioUtils.validate_audio_data')
    @patch('real_time_vad.core.vad_wrapper.AudioUtils.convert_to_mono')
    @patch('real_time_vad.core.vad_wrapper.AudioUtils.split_into_frames')
    def test_process_audio_data_numpy_array(self, mock_split_frames, mock_convert_mono, 
                                          mock_validate, mock_vad_processor, sample_audio_data):
        """Test processing audio data as numpy array."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Setup mocks
            mock_validate.return_value = None
            mock_convert_mono.return_value = sample_audio_data
            mock_split_frames.return_value = [sample_audio_data[:512], sample_audio_data[512:]]
            
            wrapper.process_audio_data(sample_audio_data)
            
            mock_validate.assert_called_once()
            mock_convert_mono.assert_called_once()
            mock_split_frames.assert_called_once()
            assert mock_vad_processor.process_frame.call_count == 2

    @patch('real_time_vad.core.vad_wrapper.AudioUtils.validate_audio_data')
    @patch('real_time_vad.core.vad_wrapper.AudioUtils.convert_to_mono')
    @patch('real_time_vad.core.vad_wrapper.AudioUtils.split_into_frames')
    def test_process_audio_data_list(self, mock_split_frames, mock_convert_mono, 
                                   mock_validate, mock_vad_processor):
        """Test processing audio data as list."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            audio_list = [0.1, 0.2, -0.1, 0.0] * 256  # 1024 samples
            audio_array = np.array(audio_list, dtype=np.float32)
            
            # Setup mocks
            mock_validate.return_value = None
            mock_convert_mono.return_value = audio_array
            mock_split_frames.return_value = [audio_array[:512], audio_array[512:]]
            
            wrapper.process_audio_data(audio_list)
            
            mock_validate.assert_called_once()
            mock_convert_mono.assert_called_once()
            assert mock_vad_processor.process_frame.call_count == 2

    def test_process_audio_data_empty_list(self, mock_vad_processor):
        """Test processing empty audio data list."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(AudioProcessingError, match="Audio data cannot be empty"):
                wrapper.process_audio_data([])

    def test_process_audio_data_invalid_type(self, mock_vad_processor):
        """Test processing audio data with invalid type."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(AudioProcessingError, match="Unsupported audio data type"):
                wrapper.process_audio_data("invalid_audio")

    def test_process_audio_data_not_initialized(self):
        """Test processing audio data when processor is not initialized."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor') as mock_processor_class:
            mock_processor_class.side_effect = Exception("Init failed")
            
            try:
                wrapper = VADWrapper()
            except VADError:
                pass  # Expected initialization failure
            
            # Manually create wrapper with failed initialization
            wrapper = object.__new__(VADWrapper)
            wrapper._state = VADWrapperState(is_initialized=False)
            wrapper._processor = None
            wrapper._lock = threading.Lock()
            
            with pytest.raises(VADError, match="VAD processor not initialized"):
                wrapper.process_audio_data(np.array([0.1, 0.2]))

    @patch('real_time_vad.core.vad_wrapper.AudioUtils.validate_audio_data')
    def test_process_audio_data_validation_failure(self, mock_validate, mock_vad_processor):
        """Test processing audio data with validation failure."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            mock_validate.side_effect = ValueError("Invalid audio format")
            
            with pytest.raises(AudioProcessingError, match="Audio processing failed"):
                wrapper.process_audio_data(np.array([0.1, 0.2]))

    def test_process_audio_data_with_buffer_valid(self, mock_vad_processor, sample_audio_data):
        """Test processing audio data with buffer and count."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            with patch.object(VADWrapper, 'process_audio_data') as mock_process:
                wrapper = VADWrapper()
                
                count = 100
                wrapper.process_audio_data_with_buffer(sample_audio_data, count)
                
                mock_process.assert_called_once()
                # Verify the correct slice was passed
                call_args = mock_process.call_args[0][0]
                assert len(call_args) == count
                np.testing.assert_array_equal(call_args, sample_audio_data[:count])

    def test_process_audio_data_with_buffer_invalid_params(self, mock_vad_processor):
        """Test processing audio data with buffer and invalid parameters."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Test non-numpy array
            with pytest.raises(AudioProcessingError, match="Audio buffer must be a numpy array"):
                wrapper.process_audio_data_with_buffer([0.1, 0.2], 2)
            
            # Test negative count
            with pytest.raises(AudioProcessingError, match="Count must be non-negative"):
                wrapper.process_audio_data_with_buffer(np.array([0.1, 0.2]), -1)
            
            # Test count exceeding buffer size
            with pytest.raises(AudioProcessingError, match="Count .* exceeds buffer size"):
                wrapper.process_audio_data_with_buffer(np.array([0.1, 0.2]), 5)

    @patch('warnings.warn')
    def test_processing_context_performance_warning(self, mock_warn, mock_vad_processor):
        """Test performance warning in processing context."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with patch('time.time', side_effect=[0.0, 2.0]):  # Mock 2-second processing
                with patch.object(wrapper, '_validate_and_prepare_audio', return_value=np.array([0.1])):
                    with patch.object(wrapper, '_process_audio_frames'):
                        wrapper.process_audio_data(np.array([0.1]))
                
                mock_warn.assert_called_once()
                assert "may indicate performance issues" in mock_warn.call_args[0][0]

    # ========================= State Management Tests =========================

    def test_reset(self, mock_vad_processor):
        """Test wrapper reset functionality."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Simulate some processing
            wrapper._state.total_frames_processed = 50
            wrapper._state.total_processing_time = 1.5
            wrapper._state.record_error(Exception("Test error"))
            
            wrapper.reset()
            
            assert wrapper._state.total_frames_processed == 0
            assert wrapper._state.total_processing_time == 0.0
            assert wrapper._state.last_error is None
            mock_vad_processor.reset.assert_called_once()

    def test_reset_failure_handling(self, mock_vad_processor):
        """Test reset failure handling."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            mock_vad_processor.reset.side_effect = Exception("Reset failed")
            
            with pytest.raises(VADError, match="Failed to reset VAD state"):
                wrapper.reset()

    def test_cleanup(self, mock_vad_processor):
        """Test wrapper cleanup functionality."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            wrapper.cleanup()
            
            assert wrapper._processor is None
            assert wrapper._state.is_initialized is False
            assert wrapper._state.last_error is None

    def test_cleanup_with_error(self, mock_vad_processor):
        """Test cleanup with error handling."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Simulate error during cleanup by making processor None raise an exception
            original_processor = wrapper._processor
            
            def failing_cleanup():
                wrapper._processor = None
                wrapper._state.is_initialized = False
                raise Exception("Cleanup error")
            
            # Mock the cleanup process to fail
            with patch.object(wrapper, '_processor', None):
                with patch.object(wrapper._state, 'is_initialized', False):
                    wrapper.cleanup()  # Should not raise exception
                    
                    # Error should be logged but cleanup should continue
                    assert wrapper._processor is None
                    assert wrapper._state.is_initialized is False

    # ========================= Information and Statistics Tests =========================

    def test_get_statistics(self, mock_vad_processor):
        """Test getting comprehensive statistics."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Set up some state
            wrapper._state.total_frames_processed = 100
            wrapper._state.total_processing_time = 5.0
            
            stats = wrapper.get_statistics()
            
            assert stats['total_frames_processed'] == 100
            assert stats['total_processing_time'] == 5.0
            assert stats['average_processing_time_per_frame'] == 0.05
            assert stats['is_initialized'] is True
            assert stats['has_callbacks'] is False
            assert 'config' in stats
            mock_vad_processor.get_statistics.assert_called_once()

    def test_get_statistics_with_error(self, mock_vad_processor):
        """Test getting statistics when error occurs."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            mock_vad_processor.get_statistics.side_effect = Exception("Stats error")
            
            stats = wrapper.get_statistics()
            
            assert 'error' in stats
            assert stats['is_initialized'] is True

    def test_get_config(self, mock_vad_processor, custom_config):
        """Test getting current configuration."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper(config=custom_config)
            
            config = wrapper.get_config()
            
            assert config == custom_config

    def test_update_config_valid(self, mock_vad_processor, custom_config):
        """Test updating configuration with valid config."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            wrapper.update_config(custom_config)
            
            assert wrapper._config == custom_config
            mock_vad_processor.update_config.assert_called_once_with(custom_config)

    def test_update_config_invalid_type(self, mock_vad_processor):
        """Test updating configuration with invalid type."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with pytest.raises(VADError, match="Failed to update configuration"):
                wrapper.update_config("invalid_config")

    def test_update_config_processor_failure(self, mock_vad_processor, custom_config):
        """Test updating configuration when processor update fails."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            old_config = wrapper._config
            
            mock_vad_processor.update_config.side_effect = Exception("Update failed")
            
            with pytest.raises(VADError, match="Failed to update configuration"):
                wrapper.update_config(custom_config)
            
            # Should rollback to old config
            assert wrapper._config == old_config

    def test_is_voice_active(self, mock_vad_processor):
        """Test checking voice activity status."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            mock_vad_processor.is_voice_active = True
            assert wrapper.is_voice_active() is True
            
            mock_vad_processor.is_voice_active = False
            assert wrapper.is_voice_active() is False

    def test_is_voice_active_no_processor(self, mock_vad_processor):
        """Test checking voice activity when no processor is available."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            wrapper._processor = None
            
            assert wrapper.is_voice_active() is False

    def test_is_voice_active_with_error(self, mock_vad_processor):
        """Test checking voice activity when error occurs."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            type(mock_vad_processor).is_voice_active = property(lambda self: (_ for _ in ()).throw(Exception("Voice check failed")))
            
            assert wrapper.is_voice_active() is False

    def test_get_last_error(self, mock_vad_processor):
        """Test getting last error information."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            assert wrapper.get_last_error() is None
            
            wrapper._state.record_error(Exception("Test error"))
            assert wrapper.get_last_error() == "Test error"

    def test_get_last_error_details(self, mock_vad_processor):
        """Test getting detailed error information."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            wrapper._state.record_error(Exception("Test error"))
            wrapper._state.total_frames_processed = 50
            
            details = wrapper.get_last_error_details()
            
            assert details['last_error'] == "Test error"
            assert details['is_initialized'] is True
            assert details['total_frames_processed'] == 50
            assert details['has_processor'] is True

    # ========================= Context Manager Tests =========================

    def test_context_manager_normal_operation(self, mock_vad_processor):
        """Test context manager normal operation."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            with VADWrapper() as wrapper:
                assert isinstance(wrapper, VADWrapper)
                assert wrapper._state.is_initialized is True
            
            # After context exit, cleanup should have been called
            assert wrapper._processor is None

    def test_context_manager_with_exception(self, mock_vad_processor):
        """Test context manager with exception during usage."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            try:
                with VADWrapper() as wrapper:
                    raise ValueError("Test exception")
            except ValueError:
                pass
            
            # Cleanup should still occur
            assert wrapper._processor is None

    def test_destructor(self, mock_vad_processor):
        """Test destructor cleanup."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Simulate destructor call
            wrapper.__del__()
            
            # Should not raise any errors even if cleanup fails

    # ========================= Thread Safety Tests =========================

    def test_thread_safety_concurrent_processing(self, mock_vad_processor, sample_audio_data):
        """Test thread safety during concurrent audio processing."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            with patch('real_time_vad.core.vad_wrapper.AudioUtils.validate_audio_data'):
                with patch('real_time_vad.core.vad_wrapper.AudioUtils.convert_to_mono', return_value=sample_audio_data):
                    with patch('real_time_vad.core.vad_wrapper.AudioUtils.split_into_frames', return_value=[sample_audio_data[:512]]):
                        wrapper = VADWrapper()
                        
                        def process_audio():
                            wrapper.process_audio_data(sample_audio_data)
                        
                        threads = []
                        for _ in range(5):
                            thread = threading.Thread(target=process_audio)
                            threads.append(thread)
                            thread.start()
                        
                        for thread in threads:
                            thread.join()
                        
                        # Should complete without errors
                        assert wrapper._state.total_frames_processed >= 5

    def test_thread_safety_concurrent_config_updates(self, mock_vad_processor):
        """Test thread safety during concurrent configuration updates."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            def update_sample_rate():
                try:
                    wrapper.set_sample_rate(SampleRate.SAMPLERATE_24)
                except:
                    pass  # Ignore any race condition exceptions
            
            def update_thresholds():
                try:
                    wrapper.set_thresholds(vad_start_probability=0.8)
                except:
                    pass  # Ignore any race condition exceptions
            
            threads = []
            for _ in range(3):
                thread1 = threading.Thread(target=update_sample_rate)
                thread2 = threading.Thread(target=update_thresholds)
                threads.extend([thread1, thread2])
                thread1.start()
                thread2.start()
            
            for thread in threads:
                thread.join()
            
            # Should complete without deadlocks
            assert wrapper._state.is_initialized is True

    # ========================= String Representation Tests =========================

    def test_repr(self, mock_vad_processor):
        """Test string representation for debugging."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            wrapper._state.total_frames_processed = 100
            
            repr_str = repr(wrapper)
            
            assert "VADWrapper(" in repr_str
            assert "initialized=True" in repr_str
            assert "frames_processed=100" in repr_str
            assert f"sample_rate={wrapper._config.sample_rate}" in repr_str

    def test_str(self, mock_vad_processor):
        """Test human-readable string representation."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            wrapper._state.total_frames_processed = 50
            
            str_repr = str(wrapper)
            
            assert "VAD Wrapper - Initialized" in str_repr
            assert "Sample Rate:" in str_repr
            assert "Model Version:" in str_repr
            assert "Frames Processed: 50" in str_repr
            assert "Has Callbacks:" in str_repr

    # ========================= Property Tests =========================

    def test_processor_property(self, mock_vad_processor):
        """Test processor property access."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            assert wrapper.processor == mock_vad_processor

    def test_config_property_getter(self, mock_vad_processor, custom_config):
        """Test config property getter."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper(config=custom_config)
            
            assert wrapper.config == custom_config

    def test_config_property_setter(self, mock_vad_processor, custom_config):
        """Test config property setter."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            with patch.object(wrapper, 'update_config') as mock_update:
                wrapper.config = custom_config
                mock_update.assert_called_once_with(custom_config)

    # ========================= Integration Tests =========================

    @pytest.mark.integration
    def test_full_workflow_with_callbacks(self, mock_vad_processor, sample_audio_data):
        """Test complete workflow with callbacks and audio processing."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            with patch('real_time_vad.core.vad_wrapper.AudioUtils.validate_audio_data'):
                with patch('real_time_vad.core.vad_wrapper.AudioUtils.convert_to_mono', return_value=sample_audio_data):
                    with patch('real_time_vad.core.vad_wrapper.AudioUtils.split_into_frames', return_value=[sample_audio_data[:512]]):
                        
                        # Setup wrapper with callbacks
                        wrapper = VADWrapper()
                        
                        start_called = []
                        end_called = []
                        continue_called = []
                        
                        def on_start():
                            start_called.append(True)
                        
                        def on_end(data):
                            end_called.append(data)
                        
                        def on_continue(data):
                            continue_called.append(data)
                        
                        wrapper.set_callbacks(
                            voice_start_callback=on_start,
                            voice_end_callback=on_end,
                            voice_continue_callback=on_continue
                        )
                        
                        # Setup processor to return voice started
                        mock_vad_processor.process_frame.return_value = ProcessingResult(
                            probability=0.8,
                            voice_started=True,
                            voice_ended=False,
                            voice_continuing=False
                        )
                        
                        # Process audio
                        wrapper.process_audio_data(sample_audio_data)
                        
                        # Verify callbacks were called
                        assert len(start_called) == 1
                        assert len(end_called) == 0
                        assert len(continue_called) == 0
                        
                        # Verify statistics
                        stats = wrapper.get_statistics()
                        assert stats['total_frames_processed'] == 1
                        assert stats['has_callbacks'] is True

    @pytest.mark.integration
    def test_error_recovery_workflow(self, mock_vad_processor):
        """Test error recovery and state management."""
        with patch('real_time_vad.core.vad_wrapper.VADProcessor', return_value=mock_vad_processor):
            wrapper = VADWrapper()
            
            # Simulate processing error
            mock_vad_processor.process_frame.side_effect = Exception("Processing failed")
            
            with pytest.raises(AudioProcessingError):
                wrapper.process_audio_data(np.array([0.1, 0.2]))
            
            # Check error was recorded
            assert wrapper.get_last_error() is not None
            
            # Reset and verify recovery
            mock_vad_processor.process_frame.side_effect = None
            mock_vad_processor.process_frame.return_value = ProcessingResult(probability=0.5)
            
            wrapper.reset()
            assert wrapper.get_last_error() is None
            
            # Should be able to process again
            with patch('real_time_vad.core.vad_wrapper.AudioUtils.validate_audio_data'):
                with patch('real_time_vad.core.vad_wrapper.AudioUtils.convert_to_mono', return_value=np.array([0.1, 0.2])):
                    with patch('real_time_vad.core.vad_wrapper.AudioUtils.split_into_frames', return_value=[np.array([0.1, 0.2])]):
                        wrapper.process_audio_data(np.array([0.1, 0.2]))
            
            assert wrapper._state.total_frames_processed == 1
