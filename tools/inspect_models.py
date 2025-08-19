#!/usr/bin/env python3
"""
Inspect Silero ONNX model inputs and outputs.
"""

import onnxruntime as ort
import os

def inspect_model(model_path: str):
    """Inspect ONNX model inputs and outputs."""
    print(f"\n🔍 Inspecting: {os.path.basename(model_path)}")
    print("=" * 50)
    
    try:
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        print("📥 Model Inputs:")
        for i, input_info in enumerate(session.get_inputs()):
            print(f"  {i+1}. Name: '{input_info.name}'")
            print(f"     Shape: {input_info.shape}")
            print(f"     Type: {input_info.type}")
        
        print("\n📤 Model Outputs:")
        for i, output_info in enumerate(session.get_outputs()):
            print(f"  {i+1}. Name: '{output_info.name}'")
            print(f"     Shape: {output_info.shape}")
            print(f"     Type: {output_info.type}")
            
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")

def get_models_directory():
    """Get the models directory path using the same logic as the VAD system."""
    # Try environment variable first
    model_path = os.environ.get('VAD_MODEL_PATH')
    if model_path and os.path.exists(model_path):
        return model_path
    
    # Try package installation path
    try:
        import real_time_vad
        package_path = os.path.dirname(real_time_vad.__file__)
        models_path = os.path.join(package_path, "models")
        if os.path.exists(models_path):
            return models_path
    except ImportError:
        pass
    
    # Fallback to relative path
    return "src/real_time_vad/models"

def main():
    """Main inspection function."""
    models_dir = get_models_directory()
    print(f"🔍 Using models directory: {models_dir}")
    
    models = [
        "silero_vad.onnx",
        "silero_vad_v5.onnx"
    ]
    
    for model_file in models:
        model_path = os.path.join(models_dir, model_file)
        if os.path.exists(model_path):
            inspect_model(model_path)
        else:
            print(f"❌ Model not found: {model_path}")

if __name__ == "__main__":
    main()
