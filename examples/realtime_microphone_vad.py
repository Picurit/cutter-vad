#!/usr/bin/env python3
"""
Real-Time Microphone Voice Activity Detection and Audio Segmentation

This script captures audio from the microphone in real-time, detects voice activity,
and automatically saves each detected voice segment as a separate audio file.

Features:
- Real-time microphone capture with PyAudio
- Configurable VAD sensitivity and thresholds
- Automatic voice segment file naming with timestamps
- Live audio level monitoring
- Configurable output directory and file formats
- Interactive controls during recording
- Multiple VAD profile presets (sensitive, balanced, robust)
- Audio quality settings (sample rate, bit depth)
- Silence detection and filtering
"""

import os
import sys
import time
import threading
import signal
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import numpy as np
import pyaudio

# Add the src directory to the path to import our VAD library
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_time_vad import (
    VADWrapper, 
    VADConfig, 
    SampleRate, 
    SileroModelVersion
)
from real_time_vad.utils.audio import AudioUtils


@dataclass
class MicrophoneConfig:
    """Configuration for microphone capture settings."""
    device_index: Optional[int] = None  # Auto-detect if None
    sample_rate: int = 16000
    channels: int = 1  # Mono
    chunk_size: int = 1024  # Frame size
    format: int = pyaudio.paFloat32
    
    
@dataclass 
class RecordingConfig:
    """Configuration for recording and output settings."""
    output_dir: str = "voice_segments"
    file_prefix: str = "voice_segment"
    file_format: str = "wav"
    min_segment_duration: float = 0.5  # Minimum duration in seconds
    max_segment_duration: float = 30.0  # Maximum duration in seconds
    auto_cleanup: bool = True  # Remove very short segments
    
    
@dataclass
class VADProfileConfig:
    """Predefined VAD configuration profiles for different scenarios."""
    name: str
    vad_start_probability: float
    vad_end_probability: float
    voice_start_frame_count: int
    voice_end_frame_count: int
    description: str


class VADProfiles:
    """Collection of predefined VAD profiles."""
    
    SENSITIVE = VADProfileConfig(
        name="sensitive",
        vad_start_probability=0.3,
        vad_end_probability=0.2,
        voice_start_frame_count=3,
        voice_end_frame_count=8,
        description="High sensitivity - detects quiet speech and whispers"
    )
    
    BALANCED = VADProfileConfig(
        name="balanced", 
        vad_start_probability=0.5,
        vad_end_probability=0.4,
        voice_start_frame_count=5,
        voice_end_frame_count=15,
        description="Balanced detection - good for normal conversation"
    )
    
    ROBUST = VADProfileConfig(
        name="robust",
        vad_start_probability=0.7,
        vad_end_probability=0.6,
        voice_start_frame_count=8,
        voice_end_frame_count=25,
        description="Low sensitivity - only detects clear speech, filters noise"
    )
    
    VERY_SENSITIVE = VADProfileConfig(
        name="very_sensitive",
        vad_start_probability=0.2,
        vad_end_probability=0.15,
        voice_start_frame_count=2,
        voice_end_frame_count=5,
        description="Ultra-high sensitivity - may pick up background sounds"
    )
    
    VERY_ROBUST = VADProfileConfig(
        name="very_robust",
        vad_start_probability=0.8,
        vad_end_probability=0.7,
        voice_start_frame_count=12,
        voice_end_frame_count=40,
        description="Ultra-low sensitivity - only very clear, loud speech"
    )
    
    @classmethod
    def get_all_profiles(cls) -> Dict[str, VADProfileConfig]:
        """Get all available VAD profiles."""
        return {
            cls.SENSITIVE.name: cls.SENSITIVE,
            cls.BALANCED.name: cls.BALANCED,
            cls.ROBUST.name: cls.ROBUST,
            cls.VERY_SENSITIVE.name: cls.VERY_SENSITIVE,
            cls.VERY_ROBUST.name: cls.VERY_ROBUST,
        }
    
    @classmethod
    def get_profile(cls, name: str) -> VADProfileConfig:
        """Get a specific VAD profile by name."""
        profiles = cls.get_all_profiles()
        if name not in profiles:
            available = list(profiles.keys())
            raise ValueError(f"Unknown profile '{name}'. Available: {available}")
        return profiles[name]


class RealTimeMicrophoneVAD:
    """Real-time microphone VAD processor with automatic voice segment saving."""
    
    def __init__(
        self,
        vad_profile: str = "balanced",
        microphone_config: Optional[MicrophoneConfig] = None,
        recording_config: Optional[RecordingConfig] = None,
        custom_vad_config: Optional[VADConfig] = None
    ):
        """
        Initialize real-time microphone VAD processor.
        
        Args:
            vad_profile: Name of VAD profile to use ('sensitive', 'balanced', 'robust', etc.)
            microphone_config: Microphone capture configuration
            recording_config: Recording and output configuration
            custom_vad_config: Custom VAD configuration (overrides profile)
        """
        self.mic_config = microphone_config or MicrophoneConfig()
        self.rec_config = recording_config or RecordingConfig()
        
        # Set up VAD configuration
        if custom_vad_config:
            self.vad_config = custom_vad_config
            self.vad_profile_name = "custom"
        else:
            profile = VADProfiles.get_profile(vad_profile)
            self.vad_profile_name = profile.name
            self.vad_config = VADConfig(
                sample_rate=SampleRate(self.mic_config.sample_rate),
                model_version=SileroModelVersion.V5,
                vad_start_probability=profile.vad_start_probability,
                vad_end_probability=profile.vad_end_probability,
                voice_start_frame_count=profile.voice_start_frame_count,
                voice_end_frame_count=profile.voice_end_frame_count,
                enable_denoising=True,
                auto_convert_sample_rate=True,
                buffer_size=self.mic_config.chunk_size
            )
        
        # Initialize components
        self.vad = VADWrapper(config=self.vad_config)
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # State tracking
        self.is_recording = False
        self.segment_counter = 0
        self.current_segment_start_time: Optional[datetime] = None
        self.total_segments_saved = 0
        self.total_recording_time = 0.0
        self.audio_level_history: List[float] = []
        
        # Threading
        self.recording_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Create output directory
        self.output_path = Path(self.rec_config.output_dir)
        self.output_path.mkdir(exist_ok=True)
        
        # Set up VAD callbacks
        self._setup_vad_callbacks()
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_vad_callbacks(self) -> None:
        """Set up VAD event callbacks."""
        
        def on_voice_start():
            """Called when voice activity starts."""
            self.current_segment_start_time = datetime.now()
            self.segment_counter += 1
            
            timestamp = self.current_segment_start_time.strftime("%H:%M:%S")
            print(f"\\n🎙️  VOICE STARTED - Segment #{self.segment_counter} at {timestamp}")
            
        def on_voice_end(wav_data: bytes):
            """Called when voice activity ends - save the segment."""
            if not self.current_segment_start_time:
                return
                
            # Calculate segment duration
            segment_duration = (datetime.now() - self.current_segment_start_time).total_seconds()
            
            # Check minimum duration filter
            if segment_duration < self.rec_config.min_segment_duration:
                print(f"   ⏭️  Segment too short ({segment_duration:.2f}s < {self.rec_config.min_segment_duration}s) - skipping")
                return
            
            # Check maximum duration filter
            if segment_duration > self.rec_config.max_segment_duration:
                print(f"   ⚠️  Segment too long ({segment_duration:.2f}s > {self.rec_config.max_segment_duration}s) - truncated")
            
            # Generate filename with timestamp
            timestamp = self.current_segment_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.rec_config.file_prefix}_{timestamp}_{self.segment_counter:04d}.{self.rec_config.file_format}"
            filepath = self.output_path / filename
            
            # Save the audio file
            try:
                with open(filepath, 'wb') as f:
                    f.write(wav_data)
                
                file_size_kb = len(wav_data) / 1024
                self.total_segments_saved += 1
                
                print(f"🔴 VOICE ENDED - Saved: {filename}")
                print(f"   📊 Duration: {segment_duration:.2f}s, Size: {file_size_kb:.1f}KB")
                print(f"   📁 Location: {filepath}")
                
            except Exception as e:
                print(f"   ❌ Error saving segment: {e}")
        
        def on_voice_continue(pcm_data: bytes):
            """Called continuously during voice activity."""
            # Update audio level monitoring
            try:
                audio_array = AudioUtils.pcm_to_float32(pcm_data, 16)
                rms_level = AudioUtils.calculate_rms(audio_array)
                self.audio_level_history.append(rms_level)
                
                # Keep only recent history
                if len(self.audio_level_history) > 50:
                    self.audio_level_history.pop(0)
                    
            except Exception:
                pass  # Ignore level calculation errors
        
        # Set the callbacks
        self.vad.set_callbacks(
            voice_start_callback=on_voice_start,
            voice_end_callback=on_voice_end,
            voice_continue_callback=on_voice_continue
        )
    
    def list_audio_devices(self) -> None:
        """List all available audio input devices."""
        print("\\n🎤 Available Audio Input Devices:")
        print("=" * 50)
        
        device_count = self.audio.get_device_count()
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Input device
                print(f"  {i}: {device_info['name']}")
                print(f"      Max Input Channels: {device_info['maxInputChannels']}")
                print(f"      Default Sample Rate: {device_info['defaultSampleRate']:.0f} Hz")
                print()
    
    def _auto_detect_microphone(self) -> int:
        """Auto-detect the best available microphone."""
        try:
            # Try to get default input device
            default_device = self.audio.get_default_input_device_info()
            return default_device['index']
        except Exception:
            # Fallback: find any input device
            device_count = self.audio.get_device_count()
            for i in range(device_count):
                device_info = self.audio.get_device_info_by_index(i)
                if device_info['maxInputChannels'] > 0:
                    return i
            raise RuntimeError("No input audio devices found")
    
    def start_recording(self) -> None:
        """Start real-time recording and VAD processing."""
        if self.is_recording:
            print("⚠️  Recording is already active!")
            return
        
        # Auto-detect microphone if not specified
        if self.mic_config.device_index is None:
            self.mic_config.device_index = self._auto_detect_microphone()
        
        print(f"\\n🎙️  Starting Real-Time Microphone VAD")
        print("=" * 50)
        print(f"📊 VAD Profile: {self.vad_profile_name}")
        print(f"🎤 Microphone Device: {self.mic_config.device_index}")
        print(f"📡 Sample Rate: {self.mic_config.sample_rate} Hz")
        print(f"📦 Chunk Size: {self.mic_config.chunk_size} samples")
        print(f"📁 Output Directory: {self.output_path}")
        print(f"⚙️  VAD Thresholds: Start={self.vad_config.vad_start_probability:.2f}, End={self.vad_config.vad_end_probability:.2f}")
        print("\\nPress Ctrl+C to stop recording...")
        print("-" * 50)
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.mic_config.format,
                channels=self.mic_config.channels,
                rate=self.mic_config.sample_rate,
                input=True,
                input_device_index=self.mic_config.device_index,
                frames_per_buffer=self.mic_config.chunk_size,
                stream_callback=None  # We'll use blocking mode
            )
            
            self.is_recording = True
            self.stop_event.clear()
            
            # Start recording in a separate thread
            self.recording_thread = threading.Thread(target=self._recording_loop, daemon=True)
            self.recording_thread.start()
            
            # Start monitoring thread
            monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            monitor_thread.start()
            
            print("🟢 Recording started! Listening for voice...")
            
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            self.stop_recording()
            raise
    
    def _recording_loop(self) -> None:
        """Main recording loop - runs in separate thread."""
        start_time = time.time()
        
        try:
            while self.is_recording and not self.stop_event.is_set():
                # Read audio data from microphone
                audio_data = self.stream.read(
                    self.mic_config.chunk_size,
                    exception_on_overflow=False
                )
                
                # Convert bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Process with VAD
                self.vad.process_audio_data(audio_array)
                
                # Update total recording time
                self.total_recording_time = time.time() - start_time
                
        except Exception as e:
            print(f"\\n❌ Recording error: {e}")
        finally:
            print("\\n🔴 Recording loop ended")
    
    def _monitoring_loop(self) -> None:
        """Audio level monitoring loop - runs in separate thread."""
        last_update = time.time()
        
        while self.is_recording and not self.stop_event.is_set():
            try:
                current_time = time.time()
                
                # Update every 2 seconds
                if current_time - last_update >= 2.0:
                    self._print_status_update()
                    last_update = current_time
                
                time.sleep(0.5)
                
            except Exception:
                pass  # Ignore monitoring errors
    
    def _print_status_update(self) -> None:
        """Print periodic status updates."""
        # Get current audio level
        avg_level = 0.0
        if self.audio_level_history:
            avg_level = np.mean(self.audio_level_history[-10:])  # Average of last 10 samples
        
        # Get VAD statistics
        vad_stats = self.vad.get_statistics()
        
        # Create audio level visualization
        level_bars = int(avg_level * 20)  # Scale to 20 bars
        level_display = "█" * level_bars + "░" * (20 - level_bars)
        
        print(f"\\r📊 Level: [{level_display}] {avg_level:.3f} | "
              f"Segments: {self.total_segments_saved} | "
              f"Time: {self.total_recording_time:.1f}s | "
              f"Voice: {'🟢' if vad_stats.get('is_voice_active', False) else '⚪'}", end="", flush=True)
    
    def stop_recording(self) -> None:
        """Stop recording and clean up resources."""
        if not self.is_recording:
            return
        
        print("\\n\\n🛑 Stopping recording...")
        self.is_recording = False
        self.stop_event.set()
        
        # Close audio stream
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        # Wait for recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=2.0)
        
        # Clean up VAD
        self.vad.cleanup()
        
        # Print final statistics
        self._print_final_statistics()
        
        print("✅ Recording stopped successfully!")
    
    def _print_final_statistics(self) -> None:
        """Print final recording statistics."""
        print("\\n📈 Final Recording Statistics:")
        print("=" * 40)
        print(f"📊 Total Recording Time: {self.total_recording_time:.1f} seconds")
        print(f"🎙️  Total Voice Segments: {self.total_segments_saved}")
        print(f"📁 Output Directory: {self.output_path}")
        print(f"⚙️  VAD Profile Used: {self.vad_profile_name}")
        
        if self.total_segments_saved > 0:
            avg_segment_interval = self.total_recording_time / self.total_segments_saved
            print(f"⏱️  Average Segment Interval: {avg_segment_interval:.1f} seconds")
        
        # Get VAD processing statistics
        vad_stats = self.vad.get_statistics()
        print(f"🔢 Total Frames Processed: {vad_stats.get('total_frames_processed', 0)}")
        
        if vad_stats.get('total_processing_time', 0) > 0:
            avg_processing_time = vad_stats['average_processing_time_per_frame']
            print(f"⚡ Avg Processing Time/Frame: {avg_processing_time:.6f}s")
        
        print()
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle interrupt signals for graceful shutdown."""
        print("\\n\\n⚠️  Interrupt signal received...")
        self.stop_recording()
        sys.exit(0)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.stop_recording()
        if self.audio:
            self.audio.terminate()


def create_custom_vad_config(
    start_prob: float,
    end_prob: float,
    start_frames: int,
    end_frames: int,
    sample_rate: int = 16000
) -> VADConfig:
    """Create a custom VAD configuration with specified parameters."""
    return VADConfig(
        sample_rate=SampleRate(sample_rate),
        model_version=SileroModelVersion.V5,
        vad_start_probability=start_prob,
        vad_end_probability=end_prob,
        voice_start_frame_count=start_frames,
        voice_end_frame_count=end_frames,
        enable_denoising=True,
        auto_convert_sample_rate=True,
        buffer_size=1024
    )


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Real-Time Microphone Voice Activity Detection and Segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VAD Profiles:
  sensitive      - High sensitivity, detects quiet speech and whispers
  balanced       - Balanced detection, good for normal conversation  
  robust         - Low sensitivity, only detects clear speech
  very_sensitive - Ultra-high sensitivity, may pick up background sounds
  very_robust    - Ultra-low sensitivity, only very clear, loud speech

Examples:
  python realtime_microphone_vad.py --profile balanced
  python realtime_microphone_vad.py --profile sensitive --output voice_recordings
  python realtime_microphone_vad.py --list-devices
  python realtime_microphone_vad.py --custom 0.4 0.3 6 12
        """
    )
    
    # Main options
    parser.add_argument(
        "--profile", "-p",
        choices=list(VADProfiles.get_all_profiles().keys()),
        default="balanced",
        help="VAD sensitivity profile (default: balanced)"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="voice_segments",
        help="Output directory for voice segments (default: voice_segments)"
    )
    
    parser.add_argument(
        "--device", "-d",
        type=int,
        help="Audio input device index (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--sample-rate", "-sr",
        type=int,
        choices=[8000, 16000, 24000, 48000],
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    
    parser.add_argument(
        "--chunk-size", "-cs",
        type=int,
        default=1024,
        help="Audio chunk size in samples (default: 1024)"
    )
    
    # Custom VAD configuration
    parser.add_argument(
        "--custom",
        nargs=4,
        metavar=('START_PROB', 'END_PROB', 'START_FRAMES', 'END_FRAMES'),
        type=float,
        help="Custom VAD parameters: start_probability end_probability start_frame_count end_frame_count"
    )
    
    # Recording options
    parser.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Minimum segment duration in seconds (default: 0.5)"
    )
    
    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum segment duration in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--prefix",
        default="voice_segment",
        help="Filename prefix for saved segments (default: voice_segment)"
    )
    
    # Utility options
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio input devices and exit"
    )
    
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available VAD profiles and exit"
    )
    
    args = parser.parse_args()
    
    # Handle utility options
    if args.list_devices:
        temp_audio = pyaudio.PyAudio()
        temp_vad = RealTimeMicrophoneVAD()
        temp_vad.list_audio_devices()
        temp_audio.terminate()
        return
    
    if args.list_profiles:
        print("\\n🎛️  Available VAD Profiles:")
        print("=" * 50)
        for name, profile in VADProfiles.get_all_profiles().items():
            print(f"  {name:<15} - {profile.description}")
            print(f"    {'':15}   Start: {profile.vad_start_probability:.2f}, End: {profile.vad_end_probability:.2f}")
            print(f"    {'':15}   Frames: {profile.voice_start_frame_count}-{profile.voice_end_frame_count}")
            print()
        return
    
    # Create configurations
    mic_config = MicrophoneConfig(
        device_index=args.device,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size
    )
    
    rec_config = RecordingConfig(
        output_dir=args.output,
        file_prefix=args.prefix,
        min_segment_duration=args.min_duration,
        max_segment_duration=args.max_duration
    )
    
    # Handle custom VAD configuration
    custom_vad_config = None
    if args.custom:
        start_prob, end_prob, start_frames, end_frames = args.custom
        # Convert last two to integers
        start_frames = int(start_frames)
        end_frames = int(end_frames)
        
        print(f"\\n⚙️  Using Custom VAD Configuration:")
        print(f"   Start Probability: {start_prob:.2f}")
        print(f"   End Probability: {end_prob:.2f}")
        print(f"   Start Frame Count: {start_frames}")
        print(f"   End Frame Count: {end_frames}")
        
        custom_vad_config = create_custom_vad_config(
            start_prob, end_prob, start_frames, end_frames, args.sample_rate
        )
    
    # Create and run VAD processor
    try:
        with RealTimeMicrophoneVAD(
            vad_profile=args.profile,
            microphone_config=mic_config,
            recording_config=rec_config,
            custom_vad_config=custom_vad_config
        ) as vad_processor:
            
            vad_processor.start_recording()
            
            # Keep main thread alive
            try:
                while vad_processor.is_recording:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
    
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
