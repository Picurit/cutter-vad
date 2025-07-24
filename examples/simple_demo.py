#!/usr/bin/env python3
"""
Simple demo of the Real-Time VAD library.
"""

import numpy as np
import time
from real_time_vad import VADWrapper, SampleRate, SileroModelVersion

def demo():
    """Run a simple VAD demonstration."""
    
    print("🎙️  Real-Time VAD Library Demo")
    print("=" * 40)
    
    # Create VAD wrapper
    vad = VADWrapper()
    
    # Configure VAD
    print("Configuring VAD...")
    vad.set_sample_rate(SampleRate.SAMPLERATE_16)
    vad.set_silero_model(SileroModelVersion.V5)
    
    # Set up callbacks
    def on_voice_start():
        print("🟢 Voice activity STARTED")
    
    def on_voice_end(wav_data: bytes):
        print(f"🔴 Voice activity ENDED (got {len(wav_data)} bytes of WAV data)")
    
    def on_voice_continue(pcm_data: bytes):
        print(f"🔊 Voice continuing... ({len(pcm_data)} bytes)")
    
    vad.set_callbacks(
        voice_start_callback=on_voice_start,
        voice_end_callback=on_voice_end,
        voice_continue_callback=on_voice_continue
    )
    
    print("✓ VAD configured successfully!")
    print("\nSimulating audio processing...")
    print("(In a real application, you would feed actual microphone data)")
    print()
    
    # Simulate audio data patterns
    patterns = [
        ("background noise", 0.01, 5),    # Quiet background
        ("voice activity", 0.3, 8),       # Louder voice-like signal
        ("brief pause", 0.02, 3),         # Short pause
        ("more voice", 0.4, 6),           # More voice activity
        ("silence", 0.005, 4),            # Final silence
    ]
    
    try:
        for pattern_name, amplitude, chunks in patterns:
            print(f"📊 Processing: {pattern_name} (amplitude: {amplitude})")
            
            for i in range(chunks):
                # Generate test audio (1024 samples ≈ 64ms at 16kHz)
                if "voice" in pattern_name:
                    # Generate voice-like signal (sine wave + noise)
                    t = np.arange(1024) / 16000.0
                    signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
                    noise = np.random.randn(1024) * 0.1
                    audio_data = (signal + noise) * amplitude
                else:
                    # Generate noise/silence
                    audio_data = np.random.randn(1024) * amplitude
                
                audio_data = audio_data.astype(np.float32)
                
                # Process the audio chunk
                vad.process_audio_data(audio_data)
                
                # Simulate real-time processing delay
                time.sleep(0.1)
            
            print()
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Show final statistics
        stats = vad.get_statistics()
        print("\n📈 Final Statistics:")
        print(f"   • Frames processed: {stats['total_frames_processed']}")
        print(f"   • Processing time: {stats['total_processing_time']:.3f}s")
        print(f"   • Avg time per frame: {stats['average_processing_time_per_frame']:.6f}s")
        print(f"   • Voice currently active: {stats['is_voice_active']}")
        
        # Clean up
        vad.cleanup()
        print("\n✓ Demo completed successfully!")

if __name__ == "__main__":
    demo()
