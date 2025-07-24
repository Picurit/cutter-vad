# Examples Directory

This directory contains practical examples and demonstrations of the Real-Time VAD library.

## 📁 Files Overview

### User Examples
- **`basic_usage.py`** - Simple, minimal example showing essential VAD usage
- **`advanced_usage.py`** - Advanced configuration and features demonstration

### Interactive Demos
- **`simple_demo.py`** - Basic interactive demo with audio pattern simulation
- **`enhanced_demo.py`** - Comprehensive demo with detailed output and statistics
- **`probability_demo.py`** - Focused demo showing VAD probability detection

## 🚀 Running Examples

### Prerequisites
Ensure you have the package installed and virtual environment activated:

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or ensure package is installed
pip install -e .
```

### Basic Usage Example
```bash
python examples/basic_usage.py
```

### Interactive Demos
```bash
# Simple demo - basic VAD functionality
python examples/simple_demo.py

# Enhanced demo - comprehensive features
python examples/enhanced_demo.py

# Probability demo - detailed probability analysis
python examples/probability_demo.py
```

## 📖 What Each Example Teaches

### `basic_usage.py`
- Essential VAD setup and configuration
- Basic callback implementation
- Simple audio processing workflow

### `advanced_usage.py`
- Advanced configuration options
- Environment variable usage
- Context manager patterns
- Async/await usage

### `simple_demo.py`
- Interactive demonstration
- Audio pattern simulation
- Real-time processing concepts
- Basic statistics and monitoring

### `enhanced_demo.py`
- Comprehensive feature showcase
- Advanced audio pattern generation
- Detailed probability reporting
- Error handling and recovery

### `probability_demo.py`
- Direct probability access
- Frame-level processing
- Voice detection accuracy validation
- Performance measurement

## 💡 Usage Tips

1. **Start with `basic_usage.py`** to understand the fundamentals
2. **Use `simple_demo.py`** to see VAD in action with simulated audio
3. **Explore `probability_demo.py`** to understand detection sensitivity
4. **Reference `advanced_usage.py`** for production configuration patterns

## 🔧 Customization

All examples are designed to be easily modified:
- Adjust audio patterns in demo files
- Modify configuration parameters
- Add custom callback logic
- Experiment with different thresholds

Feel free to copy and modify these examples for your specific use case!
