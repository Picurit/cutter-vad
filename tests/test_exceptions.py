"""
Unit tests for VAD custom exceptions.

This module contains comprehensive unit tests for all custom exceptions
defined in the real_time_vad.core.exceptions module, following pytest
best practices and ensuring complete coverage of exception behavior.
"""

import pytest
from typing import Optional

from real_time_vad.core.exceptions import (
    VADError,
    ModelNotFoundError,
    ConfigurationError,
    AudioProcessingError,
    ModelInitializationError,
    CallbackError,
)


class TestVADError:
    """Test suite for the base VADError exception class."""

    def test_vad_error_without_error_code(self):
        """Test VADError initialization without error code."""
        message = "This is a test error message"
        error = VADError(message)
        
        assert error.message == message
        assert error.error_code is None
        assert str(error) == message
        assert isinstance(error, Exception)

    def test_vad_error_with_error_code(self):
        """Test VADError initialization with error code."""
        message = "This is a test error message"
        error_code = "TEST_ERROR"
        error = VADError(message, error_code)
        
        assert error.message == message
        assert error.error_code == error_code
        assert str(error) == f"[{error_code}] {message}"

    def test_vad_error_str_formatting(self):
        """Test VADError string representation formatting."""
        # Test without error code
        error_without_code = VADError("Simple message")
        assert str(error_without_code) == "Simple message"
        
        # Test with error code
        error_with_code = VADError("Detailed message", "CODE_123")
        assert str(error_with_code) == "[CODE_123] Detailed message"

    def test_vad_error_inheritance(self):
        """Test that VADError properly inherits from Exception."""
        error = VADError("Test message")
        assert isinstance(error, Exception)
        assert isinstance(error, VADError)

    @pytest.mark.parametrize("message,error_code,expected_str", [
        ("Simple error", None, "Simple error"),
        ("Error with code", "ERR_001", "[ERR_001] Error with code"),
        ("", "EMPTY_MSG", "[EMPTY_MSG] "),
        ("No code here", "", "No code here"),  # Empty string is falsy, so no brackets
    ])
    def test_vad_error_parametrized_formatting(self, message: str, error_code: Optional[str], expected_str: str):
        """Test VADError formatting with various parameter combinations."""
        error = VADError(message, error_code)
        assert str(error) == expected_str


class TestModelNotFoundError:
    """Test suite for ModelNotFoundError exception."""

    def test_model_not_found_error_with_default_message(self):
        """Test ModelNotFoundError with auto-generated message."""
        model_path = "/path/to/model.onnx"
        error = ModelNotFoundError(model_path)
        
        assert error.model_path == model_path
        assert error.error_code == "MODEL_NOT_FOUND"
        assert error.message == f"Silero model not found at path: {model_path}"
        assert str(error) == f"[MODEL_NOT_FOUND] Silero model not found at path: {model_path}"

    def test_model_not_found_error_with_custom_message(self):
        """Test ModelNotFoundError with custom message."""
        model_path = "/custom/path/model.onnx"
        custom_message = "Custom error message for model loading"
        error = ModelNotFoundError(model_path, custom_message)
        
        assert error.model_path == model_path
        assert error.error_code == "MODEL_NOT_FOUND"
        assert error.message == custom_message
        assert str(error) == f"[MODEL_NOT_FOUND] {custom_message}"

    def test_model_not_found_error_inheritance(self):
        """Test ModelNotFoundError inheritance."""
        error = ModelNotFoundError("/path/to/model")
        assert isinstance(error, VADError)
        assert isinstance(error, ModelNotFoundError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize("model_path", [
        "/absolute/path/to/model.onnx",
        "relative/path/model.onnx",
        "model.onnx",
        "/very/long/path/to/some/deep/directory/structure/model.onnx",
        "",
    ])
    def test_model_not_found_error_various_paths(self, model_path: str):
        """Test ModelNotFoundError with various path formats."""
        error = ModelNotFoundError(model_path)
        assert error.model_path == model_path
        assert model_path in error.message


class TestConfigurationError:
    """Test suite for ConfigurationError exception."""

    def test_configuration_error_with_default_message(self):
        """Test ConfigurationError with auto-generated message."""
        parameter = "threshold"
        value = "invalid_value"
        error = ConfigurationError(parameter, value)
        
        assert error.parameter == parameter
        assert error.value == value
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.message == f"Invalid configuration for parameter '{parameter}': {value}"
        assert str(error) == f"[CONFIGURATION_ERROR] Invalid configuration for parameter '{parameter}': {value}"

    def test_configuration_error_with_custom_message(self):
        """Test ConfigurationError with custom message."""
        parameter = "sample_rate"
        value = "48000"
        custom_message = "Sample rate must be 16000 or 8000"
        error = ConfigurationError(parameter, value, custom_message)
        
        assert error.parameter == parameter
        assert error.value == value
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.message == custom_message
        assert str(error) == f"[CONFIGURATION_ERROR] {custom_message}"

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inheritance."""
        error = ConfigurationError("param", "value")
        assert isinstance(error, VADError)
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize("parameter,value", [
        ("threshold", "1.5"),
        ("model_version", "v6"),
        ("sample_rate", "44100"),
        ("", ""),
        ("very_long_parameter_name", "very_long_value_string"),
    ])
    def test_configuration_error_various_parameters(self, parameter: str, value: str):
        """Test ConfigurationError with various parameter combinations."""
        error = ConfigurationError(parameter, value)
        assert error.parameter == parameter
        assert error.value == value
        assert parameter in error.message
        assert value in error.message


class TestAudioProcessingError:
    """Test suite for AudioProcessingError exception."""

    def test_audio_processing_error_basic(self):
        """Test AudioProcessingError basic initialization."""
        message = "Failed to process audio data"
        error = AudioProcessingError(message)
        
        assert error.message == message
        assert error.error_code == "AUDIO_PROCESSING_ERROR"
        assert error.audio_data_info is None
        assert str(error) == f"[AUDIO_PROCESSING_ERROR] {message}"

    def test_audio_processing_error_with_audio_info(self):
        """Test AudioProcessingError with audio data information."""
        message = "Invalid audio format"
        audio_info = "16-bit PCM, 44.1kHz, mono"
        error = AudioProcessingError(message, audio_info)
        
        assert error.message == message
        assert error.error_code == "AUDIO_PROCESSING_ERROR"
        assert error.audio_data_info == audio_info
        assert str(error) == f"[AUDIO_PROCESSING_ERROR] {message}"

    def test_audio_processing_error_inheritance(self):
        """Test AudioProcessingError inheritance."""
        error = AudioProcessingError("Test message")
        assert isinstance(error, VADError)
        assert isinstance(error, AudioProcessingError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize("message,audio_info", [
        ("Format error", "WAV 16kHz"),
        ("Buffer overflow", None),
        ("Decoding failed", "MP3 compressed"),
        ("", ""),
    ])
    def test_audio_processing_error_parametrized(self, message: str, audio_info: Optional[str]):
        """Test AudioProcessingError with various parameter combinations."""
        error = AudioProcessingError(message, audio_info)
        assert error.message == message
        assert error.audio_data_info == audio_info


class TestModelInitializationError:
    """Test suite for ModelInitializationError exception."""

    def test_model_initialization_error_with_default_message(self):
        """Test ModelInitializationError with auto-generated message."""
        model_version = "v5"
        error = ModelInitializationError(model_version)
        
        assert error.model_version == model_version
        assert error.error_code == "MODEL_INITIALIZATION_ERROR"
        assert error.message == f"Failed to initialize Silero model version: {model_version}"
        assert str(error) == f"[MODEL_INITIALIZATION_ERROR] Failed to initialize Silero model version: {model_version}"

    def test_model_initialization_error_with_custom_message(self):
        """Test ModelInitializationError with custom message."""
        model_version = "v4"
        custom_message = "Model initialization failed due to memory constraints"
        error = ModelInitializationError(model_version, custom_message)
        
        assert error.model_version == model_version
        assert error.error_code == "MODEL_INITIALIZATION_ERROR"
        assert error.message == custom_message
        assert str(error) == f"[MODEL_INITIALIZATION_ERROR] {custom_message}"

    def test_model_initialization_error_inheritance(self):
        """Test ModelInitializationError inheritance."""
        error = ModelInitializationError("v5")
        assert isinstance(error, VADError)
        assert isinstance(error, ModelInitializationError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize("model_version", [
        "v3",
        "v4",
        "v5",
        "latest",
        "",
        "custom-1.0.0",
    ])
    def test_model_initialization_error_various_versions(self, model_version: str):
        """Test ModelInitializationError with various model versions."""
        error = ModelInitializationError(model_version)
        assert error.model_version == model_version
        assert model_version in error.message


class TestCallbackError:
    """Test suite for CallbackError exception."""

    def test_callback_error_basic(self):
        """Test CallbackError basic initialization."""
        callback_name = "on_voice_detected"
        original_error = ValueError("Invalid callback parameter")
        error = CallbackError(callback_name, original_error)
        
        assert error.callback_name == callback_name
        assert error.original_error == original_error
        assert error.error_code == "CALLBACK_ERROR"
        assert error.message == f"Error in callback '{callback_name}': {str(original_error)}"
        assert str(error) == f"[CALLBACK_ERROR] Error in callback '{callback_name}': {str(original_error)}"

    def test_callback_error_inheritance(self):
        """Test CallbackError inheritance."""
        error = CallbackError("test_callback", RuntimeError("Test"))
        assert isinstance(error, VADError)
        assert isinstance(error, CallbackError)
        assert isinstance(error, Exception)

    @pytest.mark.parametrize("callback_name,original_error", [
        ("on_voice_start", ValueError("Invalid parameter")),
        ("on_voice_end", RuntimeError("Runtime issue")),
        ("custom_callback", TypeError("Type mismatch")),
        ("", Exception("Generic error")),
    ])
    def test_callback_error_various_scenarios(self, callback_name: str, original_error: Exception):
        """Test CallbackError with various callback names and error types."""
        error = CallbackError(callback_name, original_error)
        assert error.callback_name == callback_name
        assert error.original_error == original_error
        assert callback_name in error.message
        assert str(original_error) in error.message


class TestExceptionInteractions:
    """Test suite for exception interactions and edge cases."""

    def test_all_exceptions_inherit_from_vad_error(self):
        """Test that all custom exceptions inherit from VADError."""
        exceptions = [
            ModelNotFoundError("/path"),
            ConfigurationError("param", "value"),
            AudioProcessingError("message"),
            ModelInitializationError("v5"),
            CallbackError("callback", ValueError("error")),
        ]
        
        for exception in exceptions:
            assert isinstance(exception, VADError)
            assert isinstance(exception, Exception)

    def test_all_exceptions_have_error_codes(self):
        """Test that all exceptions have appropriate error codes."""
        expected_codes = {
            ModelNotFoundError("/path"): "MODEL_NOT_FOUND",
            ConfigurationError("param", "value"): "CONFIGURATION_ERROR",
            AudioProcessingError("message"): "AUDIO_PROCESSING_ERROR",
            ModelInitializationError("v5"): "MODEL_INITIALIZATION_ERROR",
            CallbackError("callback", ValueError("error")): "CALLBACK_ERROR",
        }
        
        for exception, expected_code in expected_codes.items():
            assert exception.error_code == expected_code

    def test_exception_string_formatting_consistency(self):
        """Test that all exceptions follow consistent string formatting."""
        exceptions = [
            ModelNotFoundError("/path"),
            ConfigurationError("param", "value"),
            AudioProcessingError("message"),
            ModelInitializationError("v5"),
            CallbackError("callback", ValueError("error")),
        ]
        
        for exception in exceptions:
            error_str = str(exception)
            assert error_str.startswith(f"[{exception.error_code}]")
            assert exception.message in error_str

    def test_vad_error_can_be_raised_and_caught(self):
        """Test that VADError can be properly raised and caught."""
        with pytest.raises(VADError) as exc_info:
            raise VADError("Test error", "TEST_CODE")
        
        assert exc_info.value.message == "Test error"
        assert exc_info.value.error_code == "TEST_CODE"

    def test_specific_exceptions_can_be_caught_as_vad_error(self):
        """Test that specific exceptions can be caught as VADError."""
        with pytest.raises(VADError):
            raise ModelNotFoundError("/nonexistent/path")
        
        with pytest.raises(VADError):
            raise ConfigurationError("invalid", "param")
        
        with pytest.raises(VADError):
            raise AudioProcessingError("processing failed")

    def test_exception_chaining_preservation(self):
        """Test that CallbackError preserves the original exception."""
        original = ValueError("Original error message")
        callback_error = CallbackError("test_callback", original)
        
        assert callback_error.original_error is original
        assert isinstance(callback_error.original_error, ValueError)
        assert str(original) in str(callback_error)


# Fixtures for common test data
@pytest.fixture
def sample_model_path():
    """Fixture providing a sample model path."""
    return "/path/to/silero_model.onnx"


@pytest.fixture
def sample_configuration_params():
    """Fixture providing sample configuration parameters."""
    return {
        "parameter": "threshold",
        "value": "invalid_threshold_value"
    }


@pytest.fixture
def sample_audio_info():
    """Fixture providing sample audio information."""
    return "16-bit PCM, 16kHz, mono, 1024 samples"


@pytest.fixture
def sample_callback_error():
    """Fixture providing a sample callback error scenario."""
    return {
        "callback_name": "on_speech_detected",
        "original_error": RuntimeError("Callback execution failed")
    }


class TestExceptionFixtures:
    """Test suite using fixtures for more realistic scenarios."""

    def test_model_not_found_with_fixture(self, sample_model_path):
        """Test ModelNotFoundError using fixture data."""
        error = ModelNotFoundError(sample_model_path)
        assert sample_model_path in str(error)
        assert error.model_path == sample_model_path

    def test_configuration_error_with_fixture(self, sample_configuration_params):
        """Test ConfigurationError using fixture data."""
        error = ConfigurationError(
            sample_configuration_params["parameter"],
            sample_configuration_params["value"]
        )
        assert sample_configuration_params["parameter"] in str(error)
        assert sample_configuration_params["value"] in str(error)

    def test_audio_processing_error_with_fixture(self, sample_audio_info):
        """Test AudioProcessingError using fixture data."""
        error = AudioProcessingError("Processing failed", sample_audio_info)
        assert error.audio_data_info == sample_audio_info

    def test_callback_error_with_fixture(self, sample_callback_error):
        """Test CallbackError using fixture data."""
        error = CallbackError(
            sample_callback_error["callback_name"],
            sample_callback_error["original_error"]
        )
        assert error.callback_name == sample_callback_error["callback_name"]
        assert error.original_error == sample_callback_error["original_error"]
