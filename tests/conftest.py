"""
Test configuration and fixtures.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from real_time_vad.core.config import VADConfig, SampleRate, SileroModelVersion


@pytest.fixture
def default_config():
    """Provide a default VAD configuration for testing."""
    return VADConfig()


@pytest.fixture
def custom_config():
    """Provide a custom VAD configuration for testing."""
    return VADConfig(
        sample_rate=SampleRate.SAMPLERATE_16,
        model_version=SileroModelVersion.V5,
        vad_start_probability=0.8,
        vad_end_probability=0.6,
        voice_start_frame_count=5,
        voice_end_frame_count=30,
        enable_denoising=True,
        buffer_size=1024
    )


@pytest.fixture
def sample_audio_data():
    """Provide sample audio data for testing."""
    # Generate a simple sine wave
    sample_rate = 16000
    duration = 0.1  # 100ms
    frequency = 440  # A4 note
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    return audio_data


@pytest.fixture
def sample_audio_frames():
    """Provide sample audio frames for testing."""
    frames = []
    for i in range(10):
        # Alternating between voice-like and silence-like patterns
        if i % 2 == 0:
            # Voice-like: higher amplitude sine wave
            frame = np.sin(2 * np.pi * 440 * np.arange(512) / 16000).astype(np.float32) * 0.5
        else:
            # Silence-like: low amplitude noise
            frame = np.random.randn(512).astype(np.float32) * 0.01
        frames.append(frame)
    
    return frames


@pytest.fixture
def temp_directory():
    """Provide a temporary directory for testing."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_yaml_file(temp_directory):
    """Provide a temporary YAML file for testing."""
    yaml_file = temp_directory / "test_config.yaml"
    yield yaml_file
    
    # Cleanup handled by temp_directory fixture


@pytest.fixture
def sample_wav_data():
    """Provide sample WAV data for testing."""
    # Create a simple WAV file data
    sample_rate = 16000
    duration = 0.5  # 500ms
    frequency = 440
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32) * 0.8
    
    return audio_data, sample_rate


@pytest.fixture
def mock_callbacks():
    """Provide mock callback functions for testing."""
    class MockCallbacks:
        def __init__(self):
            self.voice_started_called = False
            self.voice_ended_called = False
            self.voice_continued_called = False
            self.voice_ended_data = None
            self.voice_continued_data = None
            self.call_count = {'start': 0, 'end': 0, 'continue': 0}
        
        def voice_start_callback(self):
            self.voice_started_called = True
            self.call_count['start'] += 1
        
        def voice_end_callback(self, wav_data: bytes):
            self.voice_ended_called = True
            self.voice_ended_data = wav_data
            self.call_count['end'] += 1
        
        def voice_continue_callback(self, pcm_data: bytes):
            self.voice_continued_called = True
            self.voice_continued_data = pcm_data
            self.call_count['continue'] += 1
        
        def reset(self):
            self.voice_started_called = False
            self.voice_ended_called = False
            self.voice_continued_called = False
            self.voice_ended_data = None
            self.voice_continued_data = None
            self.call_count = {'start': 0, 'end': 0, 'continue': 0}
    
    return MockCallbacks()


@pytest.fixture
def environment_variables():
    """Provide and manage environment variables for testing."""
    original_env = {}
    
    def set_env_vars(env_dict):
        nonlocal original_env
        for key, value in env_dict.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = str(value)
    
    def restore_env_vars():
        for key, original_value in original_env.items():
            if original_value is None:
                if key in os.environ:
                    del os.environ[key]
            else:
                os.environ[key] = original_value
        original_env.clear()
    
    yield set_env_vars
    restore_env_vars()


# Test markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Add unit marker to tests in test_* files (except test_integration_*)
        if item.fspath.basename.startswith('test_') and not item.fspath.basename.startswith('test_integration_'):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to integration tests
        if item.fspath.basename.startswith('test_integration_'):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to tests that might be slow
        if any(keyword in item.name.lower() for keyword in ['file', 'audio', 'model', 'processing']):
            item.add_marker(pytest.mark.slow)
