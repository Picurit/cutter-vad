#!/usr/bin/env python3
"""
Script to verify model files are available in Docker container
"""
import os
import sys

def check_model_files():
    """Check if model files exist and are accessible."""
    
    model_paths = [
        "/app/src/real_time_vad/models/silero_vad_v5.onnx",
        "/app/src/real_time_vad/models/silero_vad.onnx"
    ]
    
    print("🔍 Checking model files in Docker container...")
    print("=" * 50)
    
    all_good = True
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"✅ {model_path} - Size: {size:,} bytes")
        else:
            print(f"❌ {model_path} - NOT FOUND")
            all_good = False
    
    print("\n📁 Directory structure:")
    print("=" * 50)
    
    # Check if directories exist
    base_dir = "/app"
    for root, dirs, files in os.walk(base_dir):
        level = root.replace(base_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.onnx'):
                file_path = os.path.join(root, file)
                size = os.path.getsize(file_path)
                print(f"{subindent}📄 {file} ({size:,} bytes)")
        
        # Limit depth to avoid too much output
        if level > 4:
            break
    
    print("\n🔍 Environment variables:")
    print("=" * 50)
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"PWD: {os.getcwd()}")
    
    if all_good:
        print("\n🎉 All model files found successfully!")
        return 0
    else:
        print("\n❌ Some model files are missing!")
        return 1

if __name__ == "__main__":
    sys.exit(check_model_files())
