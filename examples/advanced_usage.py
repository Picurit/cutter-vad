"""
Advanced configuration example for Real-Time VAD library.
"""

import numpy as np
import asyncio
from pathlib import Path
from real_time_vad import (
    VADWrapper, 
    AsyncVADWrapper, 
    VADConfig, 
    SampleRate, 
    SileroModelVersion
)


def synchronous_advanced_example():
    """Advanced synchronous VAD example with custom configuration."""
    print("Real-Time VAD Library - Advanced Synchronous Example")
    print("=" * 50)
    
    # Create custom configuration
    config = VADConfig(
        sample_rate=SampleRate.SAMPLERATE_16,
        model_version=SileroModelVersion.V5,
        vad_start_probability=0.6,  # Lower threshold for more sensitive detection
        vad_end_probability=0.8,    # Higher threshold for more stable ending
        voice_start_ratio=0.7,
        voice_end_ratio=0.9,
        voice_start_frame_count=5,  # Faster voice start detection
        voice_end_frame_count=30,   # Shorter silence before ending
        enable_denoising=True,
        auto_convert_sample_rate=True,
        buffer_size=512,
        output_wav_sample_rate=16000,
        output_wav_bit_depth=16
    )
    
    print("Configuration:")
    print(config)
    print()
    
    # Create VAD with custom configuration
    with VADWrapper(config=config) as vad:
        # Set up callbacks with more detailed logging
        def on_voice_start():
            stats = vad.get_statistics()
            print(f"🎙️  Voice STARTED at frame {stats['total_frames_processed']}")
        
        def on_voice_end(wav_data: bytes):
            stats = vad.get_statistics()
            print(f"🔇 Voice ENDED at frame {stats['total_frames_processed']}")
            print(f"   WAV data size: {len(wav_data)} bytes")
            
            # Save with timestamp
            timestamp = int(time.time() * 1000)
            filename = f"voice_segment_{timestamp}.wav"
            with open(filename, "wb") as f:
                f.write(wav_data)
            print(f"   Saved to: {filename}")
        
        def on_voice_continue(pcm_data: bytes):
            # Only log every 10th continuation to avoid spam
            stats = vad.get_statistics()
            if stats['total_frames_processed'] % 10 == 0:
                print(f"🔊 Voice continuing... (frame {stats['total_frames_processed']})")
        
        vad.set_callbacks(
            voice_start_callback=on_voice_start,
            voice_end_callback=on_voice_end,
            voice_continue_callback=on_voice_continue
        )
        
        # Simulate more realistic audio patterns
        import time
        patterns = [
            ("silence", 0.01),     # Background noise
            ("voice", 0.3),        # Voice activity  
            ("silence", 0.01),     # Brief pause
            ("voice", 0.4),        # More voice
            ("silence", 0.01),     # Final silence
        ]
        
        chunk_size = 1024
        for pattern_name, amplitude in patterns:
            # Process several chunks for each pattern
            chunks_per_pattern = 20
            print(f"\nProcessing {pattern_name} pattern (amplitude: {amplitude})")
            
            for chunk_idx in range(chunks_per_pattern):
                if pattern_name == "voice":
                    # Generate voice-like signal with some variation
                    base_signal = np.sin(2 * np.pi * 440 * np.arange(chunk_size) / 16000)
                    noise = np.random.randn(chunk_size) * 0.1
                    audio_data = (base_signal + noise) * amplitude
                else:
                    # Generate background noise
                    audio_data = np.random.randn(chunk_size) * amplitude
                
                audio_data = audio_data.astype(np.float32)
                vad.process_audio_data(audio_data)
                
                time.sleep(0.05)  # Simulate real-time processing
        
        # Final statistics
        final_stats = vad.get_statistics()
        print("\nFinal Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                print(f"- {key}: {value:.6f}")
            elif isinstance(value, dict):
                print(f"- {key}: {value}")
            else:
                print(f"- {key}: {value}")


async def asynchronous_example():
    """Asynchronous VAD example."""
    print("\nReal-Time VAD Library - Asynchronous Example")
    print("=" * 45)
    
    # Create async VAD wrapper
    async with AsyncVADWrapper() as vad:
        # Set up async callbacks
        async def on_voice_start():
            print("🎙️  [ASYNC] Voice started!")
        
        async def on_voice_end(wav_data: bytes):
            print(f"🔇 [ASYNC] Voice ended! Got {len(wav_data)} bytes")
            # Could perform async I/O here, like uploading to cloud storage
            await asyncio.sleep(0.01)  # Simulate async operation
        
        async def on_voice_continue(pcm_data: bytes):
            # Simulate async processing of continuous audio
            if len(pcm_data) > 1000:
                print(f"🔊 [ASYNC] Processing large chunk: {len(pcm_data)} bytes")
        
        vad.set_async_callbacks(
            voice_start_callback=on_voice_start,
            voice_end_callback=on_voice_end,
            voice_continue_callback=on_voice_continue
        )
        
        # Configure asynchronously
        await vad.set_sample_rate_async(SampleRate.SAMPLERATE_16)
        await vad.set_silero_model_async(SileroModelVersion.V5)
        await vad.set_thresholds_async(
            vad_start_probability=0.7,
            vad_end_probability=0.7,
            voice_start_frame_count=8,
            voice_end_frame_count=40
        )
        
        # Process audio asynchronously
        print("Processing audio asynchronously...")
        
        # Simulate concurrent audio processing
        audio_tasks = []
        for i in range(5):
            # Create different audio patterns
            if i % 2 == 0:
                audio_data = np.random.randn(2048).astype(np.float32) * 0.4  # Voice-like
            else:
                audio_data = np.random.randn(2048).astype(np.float32) * 0.05  # Silence
            
            # Process audio chunks concurrently
            task = vad.process_audio_data_async(audio_data)
            audio_tasks.append(task)
        
        # Wait for all processing to complete
        await asyncio.gather(*audio_tasks)
        
        # Get final statistics asynchronously
        stats = await vad.get_statistics_async()
        print(f"\n[ASYNC] Processed {stats['total_frames_processed']} frames")
        print(f"[ASYNC] Voice active: {await vad.is_voice_active_async()}")


def configuration_management_example():
    """Example of configuration management."""
    print("\nConfiguration Management Example")
    print("=" * 35)
    
    # Create configuration from environment variables
    print("1. Loading configuration from environment...")
    config_from_env = VADConfig.from_env()
    print(f"Environment config: {config_from_env}")
    
    # Create configuration from dictionary
    print("\n2. Creating configuration from dictionary...")
    config_dict = {
        "sample_rate": 16000,
        "model_version": "v5",
        "vad_start_probability": 0.8,
        "vad_end_probability": 0.6,
        "voice_start_frame_count": 15,
        "voice_end_frame_count": 50,
        "enable_denoising": True
    }
    config_from_dict = VADConfig.from_dict(config_dict)
    print(f"Dictionary config: {config_from_dict}")
    
    # Save configuration to YAML
    print("\n3. Saving configuration to YAML...")
    yaml_path = Path("vad_config_example.yaml")
    config_from_dict.to_yaml(yaml_path)
    print(f"Saved configuration to: {yaml_path}")
    
    # Load configuration from YAML
    print("\n4. Loading configuration from YAML...")
    config_from_yaml = VADConfig.from_yaml(yaml_path)
    print(f"YAML config: {config_from_yaml}")
    
    # Compare configurations
    print("\n5. Configuration comparison:")
    print(f"From dict == From YAML: {config_from_dict.to_dict() == config_from_yaml.to_dict()}")
    
    # Get model filename and frame duration
    print(f"\nModel filename: {config_from_yaml.get_model_filename()}")
    print(f"Frame duration: {config_from_yaml.get_frame_duration_ms():.2f}ms")


def main():
    """Run all examples."""
    try:
        # Run synchronous example
        synchronous_advanced_example()
        
        # Run asynchronous example
        asyncio.run(asynchronous_example())
        
        # Run configuration example
        configuration_management_example()
        
    except Exception as e:
        print(f"Error in examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import time
    main()
