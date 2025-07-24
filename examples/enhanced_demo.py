#!/usr/bin/env python3
"""
Enhanced demo with voice detection and probability display.
"""

import numpy as np
import time
from real_time_vad import VADWrapper, SampleRate, SileroModelVersion

def demo():
    """Run an enhanced VAD demonstration."""
    
    print("🎙️  Real-Time VAD Library Enhanced Demo")
    print("=" * 45)
    
    # Create VAD wrapper
    vad = VADWrapper()
    
    # Configure VAD with lower thresholds for demo
    print("Configuring VAD...")
    vad.set_sample_rate(SampleRate.SAMPLERATE_16)
    vad.set_silero_model(SileroModelVersion.V5)
    
    # Load a more sensitive configuration
    from real_time_vad.core.config import VADConfig
    config = VADConfig(
        threshold_start=0.3,        # Lower start threshold
        threshold_end=0.2,          # Lower end threshold
        min_voice_duration_ms=100,  # Shorter minimum duration
        min_silence_duration_ms=200 # Shorter silence duration
    )
    vad.update_configuration(config)
    
    # Set up callbacks with more details
    def on_voice_start():
        print("🟢 VOICE STARTED!")
    
    def on_voice_end(wav_data: bytes):
        print(f"🔴 VOICE ENDED! (WAV: {len(wav_data)} bytes)")
    
    def on_voice_continue(pcm_data: bytes):
        print(f"🔊 Voice continuing... ({len(pcm_data)} bytes)")
    
    vad.set_callbacks(
        voice_start_callback=on_voice_start,
        voice_end_callback=on_voice_end,
        voice_continue_callback=on_voice_continue
    )
    
    print("✓ VAD configured with sensitive settings!")
    print(f"  • Start threshold: {config.threshold_start}")
    print(f"  • End threshold: {config.threshold_end}")
    print("\nProcessing audio patterns...")
    print()
    
    # Enhanced audio patterns
    patterns = [
        ("background noise", 0.01, 3, "noise"),
        ("speaking starts", 0.5, 5, "voice"),    # Stronger voice signal
        ("brief pause", 0.05, 2, "pause"),
        ("loud speech", 0.8, 6, "voice"),        # Very strong voice signal
        ("conversation ends", 0.02, 3, "silence"),
    ]
    
    total_probabilities = []
    
    try:
        for pattern_name, amplitude, chunks, pattern_type in patterns:
            print(f"📊 {pattern_name.title()} (amp: {amplitude})")
            
            chunk_probs = []
            for i in range(chunks):
                # Generate different types of test audio
                if pattern_type == "voice":
                    # Generate realistic voice-like signal
                    t = np.arange(1024) / 16000.0
                    
                    # Mix multiple frequencies for voice-like spectrum
                    signal = (
                        np.sin(2 * np.pi * 200 * t) * 0.3 +    # Fundamental
                        np.sin(2 * np.pi * 400 * t) * 0.2 +    # First harmonic
                        np.sin(2 * np.pi * 800 * t) * 0.1      # Second harmonic
                    )
                    
                    # Add some random variation and noise
                    noise = np.random.randn(1024) * 0.1
                    envelope = np.exp(-np.abs(t - 0.032) * 10)  # Voice-like envelope
                    audio_data = (signal * envelope + noise) * amplitude
                    
                elif pattern_type == "pause":
                    # Brief pause with very low amplitude
                    audio_data = np.random.randn(1024) * amplitude * 0.5
                    
                else:
                    # Background noise or silence
                    audio_data = np.random.randn(1024) * amplitude
                
                audio_data = audio_data.astype(np.float32)
                
                # Process the audio chunk and get probability
                result = vad.process_audio_data(audio_data)
                prob = getattr(result, 'probability', 0.0) if result else 0.0
                chunk_probs.append(prob)
                total_probabilities.append(prob)
                
                # Show probability for interesting chunks
                if prob > 0.1 or pattern_type == "voice":
                    print(f"   📈 Chunk {i+1}: probability = {prob:.3f}")
                
                # Simulate real-time processing
                time.sleep(0.05)
            
            avg_prob = np.mean(chunk_probs) if chunk_probs else 0.0
            print(f"   📊 Average probability: {avg_prob:.3f}")
            print()
    
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Show comprehensive statistics
        stats = vad.get_statistics()
        print("\n📈 Final Statistics:")
        print(f"   • Frames processed: {stats['total_frames_processed']}")
        print(f"   • Processing time: {stats['total_processing_time']:.3f}s")
        if stats['total_frames_processed'] > 0:
            print(f"   • Avg time per frame: {stats['average_processing_time_per_frame']:.6f}s")
        print(f"   • Voice currently active: {stats['is_voice_active']}")
        
        if total_probabilities:
            print(f"   • Average probability: {np.mean(total_probabilities):.3f}")
            print(f"   • Max probability: {max(total_probabilities):.3f}")
            print(f"   • Min probability: {min(total_probabilities):.3f}")
        
        # Clean up
        vad.cleanup()
        print("\n✅ Enhanced demo completed successfully!")

if __name__ == "__main__":
    demo()
