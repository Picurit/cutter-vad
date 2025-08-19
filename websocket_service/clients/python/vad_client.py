"""
Python WebSocket VAD Client

Main client implementation that combines audio sources and WebSocket gateway
for real-time Voice Activity Detection via WebSocket server.
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Import local modules
from sources.audio_sources import StoredAudioSource, RealTimeMicSource
from gateway.websocket_gateway import WebSocketGateway, VADEventHandler


class VADClient:
    """Main VAD client that orchestrates audio streaming and VAD processing."""
    
    def __init__(self, 
                 server_url: str = "ws://localhost:8000/vad",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize VAD client.
        
        Args:
            server_url: WebSocket server URL
            config: Client configuration parameters
        """
        self.server_url = server_url
        self.config = config or self._default_config()
        
        # Create event handler
        self.event_handler = VADEventHandler()
        
        # Create gateway
        self.gateway = WebSocketGateway(
            server_url=server_url,
            config=config,
            event_handler=self.event_handler
        )
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default client configuration."""
        return {
            "mode": "pcm",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "frame_duration_ms": 30,
            "start_probability": 0.4,
            "end_probability": 0.3,
            "start_frame_count": 6,
            "end_frame_count": 12,
            "start_ratio": 0.8,    # voice_start_ratio - explicitly set
            "end_ratio": 0.95,     # voice_end_ratio - explicitly set
            "timeout": 30.0
        }
    
    async def process_stored_audio(self, 
                                  audio_file: str, 
                                  duration_seconds: Optional[float] = None,
                                  loop: bool = False) -> Dict[str, Any]:
        """
        Process audio from a stored file.
        
        Args:
            audio_file: Path to audio file
            duration_seconds: Maximum processing duration
            loop: Whether to loop the audio file
            
        Returns:
            Processing results and statistics
        """
        self.logger.info(f"Processing stored audio: {audio_file}")
        
        # Create audio source
        audio_source = StoredAudioSource(
            file_path=audio_file,
            sample_rate=self.config["sample_rate"],
            channels=self.config["channels"],
            frame_duration_ms=self.config["frame_duration_ms"],
            loop=loop
        )
        
        # Process audio
        start_time = time.time()
        
        try:
            await self.gateway.run_with_reconnect(audio_source, duration_seconds)
            
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            raise
        
        finally:
            # Wait a bit for final events
            await asyncio.sleep(1.0)
        
        # Get results
        processing_time = time.time() - start_time
        statistics = self.event_handler.get_statistics()
        segments = self.event_handler.get_voice_segments()
        
        results = {
            "processing_time_seconds": processing_time,
            "statistics": statistics,
            "voice_segments": segments,
            "config": self.config
        }
        
        self.logger.info(f"Processing completed in {processing_time:.2f}s")
        self.logger.info(f"Detected {len(segments)} voice segments")
        
        return results
    
    async def process_microphone(self, 
                               duration_seconds: Optional[float] = None,
                               device_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Process audio from microphone.
        
        Args:
            duration_seconds: Maximum processing duration
            device_index: Audio device index
            
        Returns:
            Processing results and statistics
        """
        self.logger.info("Processing microphone audio")
        
        # Create audio source
        audio_source = RealTimeMicSource(
            sample_rate=self.config["sample_rate"],
            channels=self.config["channels"],
            frame_duration_ms=self.config["frame_duration_ms"],
            device_index=device_index
        )
        
        # Process audio
        start_time = time.time()
        
        try:
            await self.gateway.run_with_reconnect(audio_source, duration_seconds)
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error processing audio: {e}")
            raise
        
        finally:
            # Wait longer for final events to ensure all segments are captured
            await asyncio.sleep(3.0)
        
        # Get results
        processing_time = time.time() - start_time
        statistics = self.event_handler.get_statistics()
        segments = self.event_handler.get_voice_segments()
        
        results = {
            "processing_time_seconds": processing_time,
            "statistics": statistics,
            "voice_segments": segments,
            "config": self.config
        }
        
        self.logger.info(f"Processing completed in {processing_time:.2f}s")
        self.logger.info(f"Detected {len(segments)} voice segments")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print processing results in a formatted way."""
        print("\n" + "="*60)
        print("VAD PROCESSING RESULTS")
        print("="*60)
        
        # Configuration
        print(f"Configuration:")
        for key, value in results["config"].items():
            print(f"  {key}: {value}")
        
        # Statistics
        stats = results["statistics"]
        print(f"\nStatistics:")
        print(f"  Processing Time: {results['processing_time_seconds']:.2f}s")
        print(f"  Total Events: {stats['total_events']}")
        print(f"  Voice Segments: {stats['voice_segments']}")
        print(f"  Total Voice Duration: {stats['total_voice_duration_ms']}ms")
        
        print(f"  Events by Type:")
        for event_type, count in stats["events_by_type"].items():
            if count > 0:
                print(f"    {event_type}: {count}")
        
        # Voice segments
        segments = results["voice_segments"]
        if segments:
            print(f"\nVoice Segments ({len(segments)} detected):")
            for i, segment in enumerate(segments):
                start_s = segment["start_ms"] / 1000
                end_s = segment["end_ms"] / 1000
                duration_s = segment["duration_ms"] / 1000
                print(f"  Segment {i+1}: {start_s:.2f}s - {end_s:.2f}s (duration: {duration_s:.2f}s)")
        else:
            print("\nNo voice segments detected")
        
        print("="*60)


def validate_test_results(results: Dict[str, Any], expected_segments: int = 4) -> bool:
    """
    Validate test results for automated testing.
    
    Args:
        results: Processing results
        expected_segments: Expected number of voice segments
        
    Returns:
        True if validation passes, False otherwise
    """
    segments = results["voice_segments"]
    detected_segments = len(segments)
    
    print(f"\nTest Validation:")
    print(f"Expected segments: {expected_segments}")
    print(f"Detected segments: {detected_segments}")
    
    if detected_segments == expected_segments:
        print("✅ Test PASSED - Correct number of segments detected")
        return True
    else:
        print(f"❌ Test FAILED - Expected {expected_segments} segments, got {detected_segments}")
        return False


async def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Python WebSocket VAD Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process stored audio file
  python vad_client.py --file ../../../examples/audios/SampleVoice.wav
  
  # Process microphone for 30 seconds
  python vad_client.py --microphone --duration 30
  
  # Custom server and configuration
  python vad_client.py --file audio.wav --server ws://192.168.1.100:8000/vad --timeout 10
  
  # Test mode (validates expected 4 segments)
  python vad_client.py --file ../../../examples/audios/SampleVoice.wav --test
        """
    )
    
    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--file", "-f",
        help="Audio file to process"
    )
    source_group.add_argument(
        "--microphone", "-m",
        action="store_true",
        help="Process microphone input"
    )
    
    # Server options
    parser.add_argument(
        "--server", "-s",
        default="ws://localhost:8000/vad",
        help="WebSocket server URL (default: ws://localhost:8000/vad)"
    )
    
    # Processing options
    parser.add_argument(
        "--duration", "-d",
        type=float,
        help="Maximum processing duration in seconds"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        help="Audio device index for microphone"
    )
    
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop audio file"
    )
    
    # VAD configuration
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        choices=[8000, 16000, 24000, 48000],
        help="Audio sample rate (default: 16000)"
    )
    
    parser.add_argument(
        "--start-prob",
        type=float,
        default=0.4,
        help="VAD start probability threshold (default: 0.4)"
    )
    
    parser.add_argument(
        "--end-prob", 
        type=float,
        default=0.3,
        help="VAD end probability threshold (default: 0.3)"
    )
    
    parser.add_argument(
        "--start-frames",
        type=int,
        default=6,
        help="VAD start frame count (default: 6)"
    )
    
    parser.add_argument(
        "--end-frames",
        type=int,
        default=12,
        help="VAD end frame count (default: 12)"
    )
    
    parser.add_argument(
        "--start-ratio",
        type=float,
        default=0.8,
        help="VAD start ratio threshold (default: 0.8)"
    )
    
    parser.add_argument(
        "--end-ratio",
        type=float,
        default=0.95,
        help="VAD end ratio threshold (default: 0.95)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Voice timeout in seconds (default: 30.0)"
    )
    
    # Test mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode - validate that exactly 4 segments are detected"
    )
    
    # Logging
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Build configuration
    config = {
        "mode": "pcm",
        "sample_rate": args.sample_rate,
        "channels": 1,
        "sample_width": 2,
        "frame_duration_ms": 30,
        "start_probability": args.start_prob,
        "end_probability": args.end_prob,
        "start_frame_count": args.start_frames,
        "end_frame_count": args.end_frames,
        "start_ratio": args.start_ratio,   # voice_start_ratio - explicitly set
        "end_ratio": args.end_ratio,       # voice_end_ratio - explicitly set
        "timeout": args.timeout
    }
    
    # Create client
    client = VADClient(
        server_url=args.server,
        config=config
    )
    
    try:
        # Process audio
        if args.file:
            # Validate file exists
            if not Path(args.file).exists():
                print(f"Error: Audio file not found: {args.file}")
                return 1
            
            results = await client.process_stored_audio(
                audio_file=args.file,
                duration_seconds=args.duration,
                loop=args.loop
            )
        else:
            results = await client.process_microphone(
                duration_seconds=args.duration,
                device_index=args.device
            )
        
        # Print results
        client.print_results(results)
        
        # Test validation
        if args.test:
            if validate_test_results(results, expected_segments=4):
                return 0  # Success
            else:
                return 1  # Failure
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
