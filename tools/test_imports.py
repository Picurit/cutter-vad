#!/usr/bin/env python3
"""
Simple test script to check package imports.
"""

import sys
import traceback

def test_imports():
    """Test importing various modules."""
    
    print("=" * 50)
    print("Real-Time VAD Library - Import Test")
    print("=" * 50)
    
    try:
        print("1. Testing core config import...")
        from real_time_vad.core.config import VADConfig, SampleRate, SileroModelVersion
        print("   ✓ Core config imported successfully!")
        
        print("2. Testing core exceptions import...")
        from real_time_vad.core.exceptions import VADError
        print("   ✓ Core exceptions imported successfully!")
        
        print("3. Testing utility imports...")
        from real_time_vad.utils.audio import AudioUtils
        from real_time_vad.utils.wav_writer import WAVWriter
        print("   ✓ Utilities imported successfully!")
        
        print("4. Testing main package import...")
        import real_time_vad
        print("   ✓ Main package imported successfully!")
        
        print("5. Creating a test configuration...")
        config = VADConfig()
        print(f"   ✓ Configuration created: {config}")
        
        print("\n" + "=" * 50)
        print("🎉 All imports successful!")
        print("✓ Package is working correctly!")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        print("\nTraceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
