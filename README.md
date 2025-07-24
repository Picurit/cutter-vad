# Real-Time Voice Activity Detection (VAD) - Python Implementation

A comprehensive Python implementation of real-time Voice Activity Detection using Silero models. This library provides efficient voice detection capabilities with configurable parameters and easy integration into any Python project.

## 🚀 Features

- **Real-time Voice Activity Detection** using Silero models (v4 and v5)
- **512-sample frame processing** at 16kHz for optimal performance
- **Multiple sample rate support** (8kHz, 16kHz, 24kHz, 48kHz)
- **Automatic sample rate conversion** to 16kHz for processing
- **LSTM state management** for continuous audio processing
- **Configurable detection thresholds** and parameters
- **Callback-based architecture** for voice start/end/continue events
- **Clean, modular design** for easy integration
- **Type hints and comprehensive documentation**
- **Async support** for non-blocking operations
- **Production-ready** with comprehensive error handling
- **High performance** (~0.0004s per frame processing)
- **Virtual environment ready** with `pyproject.toml`

## ⚡ Performance

The library is optimized for real-time performance:

- **Processing Speed**: ~0.0004s per frame (extremely fast!)
- **Memory Efficiency**: Minimal state management overhead
- **Model Loading**: Quick ONNX model initialization
- **Real-time Capable**: Suitable for live audio processing
- **Frame Processing**: 512 samples at 16kHz

### Detection Accuracy

Probability outputs demonstrate excellent detection across different scenarios:

- **Silence** (0.005 amplitude): 0.046-0.187 (correctly low)
- **Soft Voice** (0.3 amplitude): 0.928-0.999 (correctly high)  
- **Loud Voice** (0.7 amplitude): 0.992-0.996 (correctly very high)

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

## 📁 Project Structure

The project follows a clean, modular architecture with clear separation of concerns:

```
cutter-vad/
├── src/real_time_vad/           # Main package source code
│   ├── core/                    # Core VAD functionality
│   │   ├── config.py            # Configuration system
│   │   ├── exceptions.py        # Custom exceptions
│   │   ├── vad_wrapper.py       # Main VAD interface
│   │   ├── async_vad_wrapper.py # Async support
│   │   └── silero_model.py      # ONNX model handling
│   ├── utils/                   # Utility functions
│   │   ├── audio.py             # Audio processing
│   │   └── wav_writer.py        # WAV file creation
│   └── models/                  # Silero ONNX model files
│       ├── silero_vad.onnx      # V4 model
│       └── silero_vad_v5.onnx   # V5 model
├── examples/                    # User-facing examples and demonstrations
│   ├── basic_usage.py          # Simple usage example
│   ├── advanced_usage.py       # Advanced configuration example
│   ├── simple_demo.py          # Interactive basic demo
│   ├── enhanced_demo.py        # Comprehensive demo
│   └── probability_demo.py     # Probability analysis demo
├── tools/                       # Development and debugging utilities
│   ├── inspect_models.py       # ONNX model inspector
│   └── test_imports.py         # Import validation tool
├── tests/                       # Unit and integration tests
└── pyproject.toml              # Package configuration
```

### Architecture Benefits

- **📋 Clear Separation**: User examples vs developer tools are well organized
- **🎯 Easy Discovery**: Intuitive directory structure for quick navigation
- **🔧 Maintainability**: Logical grouping makes updates and extensions easier
- **👥 Better UX**: New users can easily find relevant examples and documentation

## 🎯 Examples and Demos

The `examples/` directory provides comprehensive demonstrations for users at all levels:

### 📚 Learning Path

#### **Basic Examples** - Start Here
```bash
# Basic usage - minimal setup and simple callbacks
python examples/basic_usage.py

# Advanced usage - comprehensive configuration options
python examples/advanced_usage.py
```

#### **Interactive Demos** - See It in Action
```bash
# Simple demo - audio simulation with basic VAD functionality
python examples/simple_demo.py

# Enhanced demo - detailed output and advanced features
python examples/enhanced_demo.py

# Probability demo - understand VAD sensitivity and thresholds
python examples/probability_demo.py
```

Each example is self-contained and includes:
- **Clear documentation** explaining the concepts
- **Practical code** you can modify for your needs
- **Output examples** showing expected results

### 🛠️ Development Tools

For developers and maintainers working with the library:

```bash
# Inspect ONNX model specifications and debug model loading
python tools/inspect_models.py

# Validate package installation and import functionality
python tools/test_imports.py
```

**Tool Benefits:**
- **🔍 Model Inspection**: Analyze ONNX model inputs, outputs, and specifications
- **✅ Installation Validation**: Verify all components are working correctly
- **🐛 Debugging Support**: Essential tools for troubleshooting and development

## 🧪 Demo Results

The `probability_demo.py` demonstrates the VAD accuracy in action:

```
🎙️  VAD Probability Demo
==============================
✓ VAD ready with sensitive settings

📊 Testing: Silence (amplitude: 0.005)
   Frame 1: 0.187
   Frame 2: 0.105
   Frame 3: 0.046

📊 Testing: Soft voice (amplitude: 0.3)
   Frame 1: 0.928
   Frame 2: 0.996
   Frame 3: 0.998
   Frame 4: 0.999

📊 Testing: Loud voice (amplitude: 0.7)
   Frame 1: 0.994
   Frame 2: 0.995
   Frame 3: 0.996
   Frame 4: 0.994
   Frame 5: 0.992

📈 Final Statistics:
   Frames processed: 14
   Processing time: 0.008s
✅ Demo completed!
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

## 🚀 Production Ready

This implementation is **production-ready** and provides:

1. **🔧 Robust Architecture**: Clean, modular design with proper separation of concerns
2. **📦 Easy Installation**: Standard Python packaging with pip support
3. **🔒 Type Safety**: Full type hints for better development experience
4. **⚡ High Performance**: Optimized for real-time processing with minimal overhead
5. **🧪 Test Coverage**: Comprehensive test suite ensuring reliability
6. **📚 Documentation**: Clear API documentation and practical examples
7. **🔄 Async Support**: Built-in async wrapper for concurrent applications
8. **🛡️ Error Handling**: Robust exception handling throughout the codebase

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

- [Silero VAD Models](https://github.com/snakers4/silero-vad)

## 📞 Support

- Create an [Issue](https://github.com/picurit/cutter-vad/issues)
- Check [Documentation](https://github.com/picurit/cutter-vad/blob/main/README.md)
