# Tools Directory

This directory contains development and debugging utilities for the Real-Time VAD library.

## 📁 Files Overview

### Development Tools
- **`inspect_models.py`** - ONNX model inspection and analysis utility
- **`test_imports.py`** - Package import validation and smoke testing

## 🛠️ Tool Usage

### Model Inspector
Analyze ONNX model files to understand their input/output specifications:

```bash
python tools/inspect_models.py
```

**Output Example:**
```
🔍 Inspecting: silero_vad_v5.onnx
==================================================
📥 Model Inputs:
  1. Name: 'input'
     Shape: [None, None]
     Type: tensor(float)
  2. Name: 'state'
     Shape: [2, None, 128]
     Type: tensor(float)
  3. Name: 'sr'
     Shape: []
     Type: tensor(int64)

📤 Model Outputs:
  1. Name: 'output'
     Shape: [None, 1]
     Type: tensor(float)
  2. Name: 'stateN'
     Shape: [None, None, None]
     Type: tensor(float)
```

### Import Validator
Verify that the package is correctly installed and all modules can be imported:

```bash
python tools/test_imports.py
```

**Expected Output:**
```
==================================================
Real-Time VAD Library - Import Test
==================================================
1. Testing core config import...
   ✓ Core config imported successfully!
2. Testing core exceptions import...
   ✓ Core exceptions imported successfully!
3. Testing utility imports...
   ✓ Utilities imported successfully!
4. Testing main package import...
   ✓ Main package imported successfully!
5. Creating a test configuration...
   ✓ Configuration created: <VADConfig...>

==================================================
🎉 All imports successful!
✓ Package is working correctly!
==================================================
```

## 🎯 When to Use These Tools

### `inspect_models.py`
- **Model debugging** - When ONNX models aren't loading correctly
- **Interface validation** - Verifying model input/output specifications
- **Model comparison** - Comparing different Silero model versions
- **Development** - Understanding model requirements for new features

### `test_imports.py`
- **Installation verification** - After package installation
- **Environment validation** - Confirming virtual environment setup
- **CI/CD testing** - Automated build verification
- **Troubleshooting** - Diagnosing import-related issues

## 🔧 Extending the Tools

### Adding New Model Analysis
To extend `inspect_models.py` for additional analysis:

```python
# Add to inspect_model function
def inspect_model(model_path: str):
    # ... existing code ...
    
    # Add custom analysis
    print("\n🔍 Custom Analysis:")
    # Your analysis code here
```

### Adding Import Tests
To extend `test_imports.py` with new modules:

```python
# Add to test_imports function
def test_imports():
    # ... existing tests ...
    
    print("6. Testing new module...")
    from real_time_vad.new_module import NewClass
    print("   ✓ New module imported successfully!")
```

## 📝 Development Workflow

1. **Before making changes** - Run `test_imports.py` to ensure baseline functionality
2. **After model updates** - Use `inspect_models.py` to verify model compatibility
3. **During debugging** - Use these tools to isolate and identify issues
4. **Before commits** - Validate that all imports still work correctly

These tools are essential for maintaining the reliability and debuggability of the Real-Time VAD library.
