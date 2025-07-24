# Real-Time Voice Activity Detection (VAD) - Python Implementation

A comprehensive Python implementation of real-time Voice Activity Detection using Silero models. This library provides efficient voice detection capabilities with configurable parameters and easy integration into any Python project.

## 🚀 Features

- **Real-time Voice Activity Detection** using Silero models (v4 and v5)
- **Multiple sample rate support** (8kHz, 16kHz, 24kHz, 48kHz)
- **Automatic sample rate conversion** to 16kHz for processing
- **Configurable detection thresholds** and parameters
- **Callback-based architecture** for voice start/end/continue events
- **Clean, modular design** for easy integration
- **Type hints and comprehensive documentation**
- **Async support** for non-blocking operations
- **Virtual environment ready** with `pyproject.toml`

## 📦 Installation

### Using pip (recommended)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install real-time-vad-silero
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/picurit/cutter-vad.git
cd cutter-vad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,audio,examples]"
```

## 🔧 Quick Start

### Basic Usage

```python
from real_time_vad import VADWrapper, SampleRate, SileroModelVersion
import numpy as np

# Create VAD instance
vad = VADWrapper()

# Configure sample rate and model
vad.set_sample_rate(SampleRate.SAMPLERATE_16)
vad.set_silero_model(SileroModelVersion.V5)

# Set up callbacks
def on_voice_start():
    print("Voice started!")

def on_voice_end(wav_data: bytes):
    print(f"Voice ended! Got {len(wav_data)} bytes of WAV data")

def on_voice_continue(pcm_data: bytes):
    print(f"Voice continuing... {len(pcm_data)} bytes of PCM data")

vad.set_callbacks(
    voice_start_callback=on_voice_start,
    voice_end_callback=on_voice_end,
    voice_continue_callback=on_voice_continue
)

# Process audio data (float32 array)
audio_data = np.random.randn(1024).astype(np.float32)
vad.process_audio_data(audio_data)
```

### Advanced Configuration

```python
from real_time_vad import VADWrapper, VADConfig

# Create configuration
config = VADConfig(
    sample_rate=SampleRate.SAMPLERATE_16,
    model_version=SileroModelVersion.V5,
    vad_start_probability=0.7,
    vad_end_probability=0.7,
    voice_start_ratio=0.8,
    voice_end_ratio=0.95,
    voice_start_frame_count=10,
    voice_end_frame_count=57
)

# Create VAD with configuration
vad = VADWrapper(config=config)

# Use context manager for automatic cleanup
with vad:
    # Process audio data
    vad.process_audio_data(audio_data)
```

### Async Usage

```python
import asyncio
from real_time_vad import AsyncVADWrapper

async def main():
    vad = AsyncVADWrapper()
    
    async def on_voice_start():
        print("Voice started!")
    
    async def on_voice_end(wav_data: bytes):
        print(f"Voice ended! Got {len(wav_data)} bytes")
    
    vad.set_async_callbacks(
        voice_start_callback=on_voice_start,
        voice_end_callback=on_voice_end
    )
    
    # Process audio asynchronously
    await vad.process_audio_data_async(audio_data)

asyncio.run(main())
```

## 🔧 Configuration

### Environment Variables

You can configure default settings using environment variables:

```bash
export VAD_DEFAULT_SAMPLE_RATE=16000
export VAD_DEFAULT_MODEL_VERSION=v5
export VAD_MODEL_PATH=/path/to/models
export VAD_START_PROBABILITY=0.7
export VAD_END_PROBABILITY=0.7
```

### Configuration File

Create a `vad_config.yaml` file:

```yaml
sample_rate: 16000
model_version: "v5"
vad_start_probability: 0.7
vad_end_probability: 0.7
voice_start_ratio: 0.8
voice_end_ratio: 0.95
voice_start_frame_count: 10
voice_end_frame_count: 57
```

Load configuration:

```python
from real_time_vad import VADConfig

config = VADConfig.from_yaml("vad_config.yaml")
vad = VADWrapper(config=config)
```

## 📚 API Reference

### VADWrapper

Main class for voice activity detection.

#### Methods

- `__init__(config: Optional[VADConfig] = None)` - Initialize VAD wrapper
- `set_sample_rate(sample_rate: SampleRate)` - Set audio sample rate
- `set_silero_model(model_version: SileroModelVersion)` - Set Silero model version
- `set_thresholds(...)` - Configure detection thresholds
- `set_callbacks(...)` - Set callback functions
- `process_audio_data(audio_data: np.ndarray)` - Process audio data
- `cleanup()` - Clean up resources

### VADConfig

Configuration class for VAD parameters.

#### Parameters

- `sample_rate: SampleRate` - Audio sample rate
- `model_version: SileroModelVersion` - Silero model version
- `vad_start_probability: float` - Probability threshold for voice start (default: 0.7)
- `vad_end_probability: float` - Probability threshold for voice end (default: 0.7)
- `voice_start_ratio: float` - True positive ratio for voice start (default: 0.8)
- `voice_end_ratio: float` - False positive ratio for voice end (default: 0.95)
- `voice_start_frame_count: int` - Frames to confirm voice start (default: 10)
- `voice_end_frame_count: int` - Frames to confirm voice end (default: 57)

## 🧪 Testing

Run tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "not slow"
```

## 🛠️ Development

### Setting up development environment

```bash
# Clone repository
git clone https://github.com/picurit/cutter-vad.git
cd cutter-vad

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code formatting

```bash
# Format code
black src tests
isort src tests

# Check types
mypy src

# Lint code
flake8 src tests
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔗 Related Projects

- [Original iOS/macOS Library](https://github.com/picurit/cutter-vad)
- [Silero VAD Models](https://github.com/snakers4/silero-vad)

## 📞 Support

- Create an [Issue](https://github.com/picurit/cutter-vad/issues)
- Check [Documentation](https://github.com/picurit/cutter-vad/blob/main/README.md)
