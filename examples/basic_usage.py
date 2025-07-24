"""
Basic usage example for Real-Time VAD library.
"""

import numpy as np
import time
from real_time_vad import VADWrapper, SampleRate, SileroModelVersion


def main():
    """Basic VAD usage example."""
    print("Real-Time VAD Library - Basic Example")
    print("=" * 40)
    
    # Create VAD instance
    vad = VADWrapper()
    
    # Configure VAD
    vad.set_sample_rate(SampleRate.SAMPLERATE_16)
    vad.set_silero_model(SileroModelVersion.V5)
    
    # Set up callbacks
    def on_voice_start():
        print("🎙️  Voice started!")
    
    def on_voice_end(wav_data: bytes):
        print(f"🔇 Voice ended! Got {len(wav_data)} bytes of WAV data")
        
        # Save WAV data to file (optional)
        with open("voice_output.wav", "wb") as f:
            f.write(wav_data)
        print("   Saved voice segment to 'voice_output.wav'")
    
    def on_voice_continue(pcm_data: bytes):
        print(f"🔊 Voice continuing... {len(pcm_data)} bytes of PCM data")
    
    vad.set_callbacks(
        voice_start_callback=on_voice_start,
        voice_end_callback=on_voice_end,
        voice_continue_callback=on_voice_continue
    )
    
    print("VAD initialized successfully!")
    print("Processing simulated audio data...")
    print()
    
    # Simulate audio processing
    # In a real application, you would get audio data from a microphone or file
    try:
        for i in range(10):
            # Generate some test audio data
            # This simulates 1024 samples (about 64ms at 16kHz)
            if i in [3, 4, 5, 6]:  # Simulate voice activity in the middle
                # Generate audio with higher amplitude to trigger voice detection
                audio_data = np.random.randn(1024).astype(np.float32) * 0.5 + 0.3
            else:
                # Generate quiet background noise
                audio_data = np.random.randn(1024).astype(np.float32) * 0.01
            
            # Process the audio
            vad.process_audio_data(audio_data)
            
            # Wait a bit to simulate real-time processing
            time.sleep(0.1)
            
            print(f"Processed chunk {i+1}/10")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    except Exception as e:
        print(f"Error during processing: {e}")
    
    finally:
        # Get statistics
        stats = vad.get_statistics()
        print("\nProcessing Statistics:")
        print(f"- Total frames processed: {stats['total_frames_processed']}")
        print(f"- Total processing time: {stats['total_processing_time']:.3f}s")
        print(f"- Average time per frame: {stats['average_processing_time_per_frame']:.6f}s")
        print(f"- Voice currently active: {stats['is_voice_active']}")
        
        # Clean up
        vad.cleanup()
        print("\nVAD cleaned up successfully!")


if __name__ == "__main__":
    main()
