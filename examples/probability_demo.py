#!/usr/bin/env python3
"""
Simple probability demo showing VAD detection.
"""

import numpy as np
import time
from real_time_vad import VADWrapper, SampleRate, SileroModelVersion
from real_time_vad.core.config import VADConfig

def demo():
    """Run a VAD probability demonstration."""
    
    print("🎙️  VAD Probability Demo")
    print("=" * 30)
    
    # Create VAD wrapper with sensitive settings
    config = VADConfig(
        vad_start_probability=0.3,
        vad_end_probability=0.2,
        voice_start_frame_count=5,
        voice_end_frame_count=10
    )
    
    vad = VADWrapper(config)
    vad.set_sample_rate(SampleRate.SAMPLERATE_16)
    vad.set_silero_model(SileroModelVersion.V5)
    
    # Set up callbacks
    def on_voice_start():
        print("🟢 VOICE DETECTED!")
    
    def on_voice_end(wav_data: bytes):
        print(f"🔴 VOICE ENDED! ({len(wav_data)} bytes)")
    
    vad.set_callbacks(
        voice_start_callback=on_voice_start,
        voice_end_callback=on_voice_end
    )
    
    print("✓ VAD ready with sensitive settings")
    print()
    
    # Generate test patterns
    patterns = [
        ("Silence", 0.005, 3),
        ("Soft voice", 0.3, 4),
        ("Loud voice", 0.7, 5),
        ("Background", 0.02, 2),
    ]
    
    try:
        for name, amplitude, chunks in patterns:
            print(f"📊 Testing: {name} (amplitude: {amplitude})")
            
            for i in range(chunks):
                # Create test audio
                if "voice" in name.lower():
                    # Voice-like signal with harmonics
                    t = np.arange(512) / 16000.0
                    signal = (
                        np.sin(2 * np.pi * 150 * t) * 0.4 +
                        np.sin(2 * np.pi * 300 * t) * 0.3 +
                        np.sin(2 * np.pi * 600 * t) * 0.2
                    )
                    noise = np.random.randn(512) * 0.1
                    audio = (signal + noise) * amplitude
                else:
                    # Noise/silence
                    audio = np.random.randn(512) * amplitude
                
                audio = audio.astype(np.float32)
                
                # Process manually to get probability
                if hasattr(vad, 'processor') and vad.processor:
                    try:
                        result = vad.processor.process_frame(audio)
                        prob = result.get('probability', 0.0)
                        print(f"   Frame {i+1}: {prob:.3f}")
                        
                        # Also trigger callbacks by processing through wrapper
                        vad.process_audio_data(audio)
                        
                    except Exception as e:
                        print(f"   Frame {i+1}: Error - {e}")
                
                time.sleep(0.1)
            
            print()
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        stats = vad.get_statistics()
        print("📈 Final Statistics:")
        print(f"   Frames processed: {stats['total_frames_processed']}")
        print(f"   Processing time: {stats['total_processing_time']:.3f}s")
        
        vad.cleanup()
        print("✅ Demo completed!")

if __name__ == "__main__":
    demo()
