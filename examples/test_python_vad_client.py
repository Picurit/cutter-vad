#!/usr/bin/env python3
"""
Python WebSocket VAD Client Test

This script tests the Python client with SampleVoiceMono.wav to verify
that exactly 4 voice segments are detected as required.

Run from the examples directory to ensure proper imports.
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the required paths for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
websocket_service_dir = project_root / "websocket_service"
python_client_dir = websocket_service_dir / "clients" / "python"

# Add paths to sys.path
sys.path.insert(0, str(python_client_dir))
sys.path.insert(0, str(project_root / "src"))

# Now import the client modules
try:
    from sources.audio_sources import StoredAudioSource
    from gateway.websocket_gateway import WebSocketGateway, VADEventHandler
    print("✅ Successfully imported client modules")
except ImportError as e:
    print(f"❌ Failed to import client modules: {e}")
    sys.exit(1)


class TestVADEventHandler(VADEventHandler):
    """Enhanced event handler for testing that tracks segments."""
    
    def __init__(self):
        super().__init__()
        self.voice_segments = []
        self.current_segment = None
    
    async def handle_event(self, event_data):
        """Handle VAD events and track voice segments."""
        await super().handle_event(event_data)
        
        event_type = event_data.get("event")
        
        if event_type == "VOICE_START":
            segment_index = event_data.get("segment_index", 0)
            timestamp_ms = event_data.get("timestamp_ms", 0)
            
            self.current_segment = {
                "segment_index": segment_index,
                "start_ms": timestamp_ms,
                "end_ms": None,
                "duration_ms": None
            }
            
        elif event_type == "VOICE_END":
            if self.current_segment is not None:
                segment_index = event_data.get("segment_index", 0)
                timestamp_ms = event_data.get("timestamp_ms", 0)
                duration_ms = event_data.get("duration_ms", 0)
                
                self.current_segment.update({
                    "end_ms": timestamp_ms,
                    "duration_ms": duration_ms
                })
                
                self.voice_segments.append(self.current_segment)
                self.current_segment = None


async def test_python_client():
    """Test the Python client with SampleVoiceMono.wav."""
    
    # Configure logging - enable debug for troubleshooting
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🐍 Testing Python WebSocket VAD Client")
    print("=" * 60)
    
    # Test configuration - as specified in requirements
    config = {
        "mode": "pcm",
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "frame_duration_ms": 30,
        "start_probability": 0.4,
        "end_probability": 0.3,
        "start_frame_count": 6,
        "end_frame_count": 12,
        "timeout": 30.0  # Required: tests must use timeout > 0
    }
    
    # Audio file path
    audio_file = current_dir / "audios" / "SampleVoiceMono.wav"
    
    if not audio_file.exists():
        print(f"❌ Test audio file not found: {audio_file}")
        print("Please ensure the SampleVoiceMono.wav file exists in examples/audios/")
        return False
    
    print(f"📁 Audio file: {audio_file}")
    print(f"🔧 Configuration: {config}")
    print()
    
    # Create event handler
    event_handler = TestVADEventHandler()
    
    # Create WebSocket gateway
    gateway = WebSocketGateway(
        server_url="ws://localhost:8000/vad",
        config=config,
        event_handler=event_handler
    )
    
    # Create audio source
    try:
        audio_source = StoredAudioSource(
            file_path=str(audio_file),
            sample_rate=config["sample_rate"],
            channels=config["channels"],
            frame_duration_ms=config["frame_duration_ms"],
            loop=False
        )
        print(f"✅ Created audio source: {audio_source.duration_seconds:.2f}s duration")
    except Exception as e:
        print(f"❌ Failed to create audio source: {e}")
        return False
    
    # Test the connection and processing
    try:
        print("\n🚀 Starting VAD processing...")
        
        # Connect to server
        if not await gateway.connect():
            print("❌ Failed to connect to WebSocket server")
            return False
        
        print("✅ Connected to WebSocket server")
        
        # Start event listener in background
        listen_task = asyncio.create_task(gateway.listen_for_events())
        
        # Stream audio and process
        await gateway.stream_audio(audio_source, duration_seconds=None)
        
        # Wait a bit for final events
        await asyncio.sleep(1.0)
        
        # Cancel listening task
        listen_task.cancel()
        try:
            await listen_task
        except asyncio.CancelledError:
            pass
        
        # Disconnect
        await gateway.disconnect()
        
        print("\n📊 Processing Results:")
        print("=" * 40)
        
        # Print all events
        print(f"Total events received: {len(event_handler.events)}")
        for i, event in enumerate(event_handler.events):
            event_type = event.get("event", "UNKNOWN")
            timestamp_ms = event.get("timestamp_ms", 0)
            timestamp_s = timestamp_ms / 1000
            
            if event_type == "VOICE_START":
                segment_idx = event.get("segment_index", "?")
                print(f"  {i+1:2d}. {event_type:12s} - Segment #{segment_idx} at {timestamp_s:.3f}s")
            elif event_type == "VOICE_END":
                segment_idx = event.get("segment_index", "?")
                duration_ms = event.get("duration_ms", 0)
                print(f"  {i+1:2d}. {event_type:12s} - Segment #{segment_idx} at {timestamp_s:.3f}s (duration: {duration_ms}ms)")
            else:
                print(f"  {i+1:2d}. {event_type:12s} - at {timestamp_s:.3f}s")
        
        # Print voice segments
        print(f"\nDetected voice segments: {len(event_handler.voice_segments)}")
        for i, segment in enumerate(event_handler.voice_segments):
            start_s = segment["start_ms"] / 1000
            end_s = segment["end_ms"] / 1000
            duration_s = segment["duration_ms"] / 1000
            print(f"  Segment {i+1}: {start_s:.3f}s - {end_s:.3f}s (duration: {duration_s:.3f}s)")
        
        # Validation
        print(f"\n🔍 Validation:")
        print(f"Expected voice segments: 4")
        print(f"Actual voice segments: {len(event_handler.voice_segments)}")
        
        if len(event_handler.voice_segments) == 4:
            print("✅ Segment count validation: PASSED")
            
            # Additional validation - check durations are reasonable
            valid_durations = all(
                segment["duration_ms"] > 0 
                for segment in event_handler.voice_segments
            )
            
            if valid_durations:
                print("✅ Segment duration validation: PASSED")
                print("\n🎉 Python client test PASSED!")
                return True
            else:
                print("❌ Segment duration validation: FAILED")
                return False
        else:
            print("❌ Segment count validation: FAILED")
            print(f"Expected 4 segments but got {len(event_handler.voice_segments)}")
            return False
    
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("WebSocket VAD Client - Python Test")
    print("Make sure the VAD server is running:")
    print("  uvicorn websocket_service.server.vad_websocket_server:app --host 0.0.0.0 --port 8000")
    print()
    
    success = await test_python_client()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
