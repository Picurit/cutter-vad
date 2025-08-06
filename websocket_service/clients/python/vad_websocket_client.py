#!/usr/bin/env python3
"""
Python WebSocket Client for Real-Time Voice Activity Detection

This client connects to the VAD WebSocket service and streams audio from the microphone
in real-time. It demonstrates how to:
- Connect to the VAD WebSocket service
- Configure VAD parameters
- Stream audio data from microphone
- Receive and handle VAD events

Usage:
    python vad_websocket_client.py --server ws://localhost:8765/ws
"""

import asyncio
import json
import logging
import argparse
import signal
import sys
import os
from datetime import datetime
from typing import Optional, Dict, Any
import threading
import time

import websockets
import numpy as np
import pyaudio

# Add the src directory to the path to import VAD utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from real_time_vad.utils.audio import AudioUtils


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VADWebSocketClient:
    """Python client for VAD WebSocket service."""
    
    def __init__(
        self,
        server_url: str,
        sample_rate: int = 16000,
        chunk_size: int = 1024,
        channels: int = 1,
        device_index: Optional[int] = None
    ):
        """
        Initialize the VAD WebSocket client.
        
        Args:
            server_url: WebSocket server URL (e.g., ws://localhost:8765/ws)
            sample_rate: Audio sample rate in Hz
            chunk_size: Audio chunk size in samples
            channels: Number of audio channels (should be 1 for mono)
            device_index: Audio device index (None for default)
        """
        self.server_url = server_url
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.device_index = device_index
        
        # WebSocket and audio components
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.audio = pyaudio.PyAudio()
        self.stream: Optional[pyaudio.Stream] = None
        
        # State management
        self.is_connected = False
        self.is_recording = False
        self.client_id: Optional[str] = None
        self.stop_event = asyncio.Event()
        
        # Statistics
        self.connection_start_time: Optional[datetime] = None
        self.total_audio_sent = 0
        self.vad_events_received = 0
        self.last_voice_event: Optional[str] = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals for graceful shutdown."""
        logger.info("Interrupt signal received, shutting down...")
        asyncio.create_task(self.stop())
    
    async def connect(self) -> bool:
        """Connect to the WebSocket server."""
        try:
            logger.info(f"Connecting to {self.server_url}...")
            
            self.websocket = await websockets.connect(self.server_url)
            self.is_connected = True
            self.connection_start_time = datetime.now()
            
            logger.info("Connected to VAD WebSocket service")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WebSocket server: {e}")
            return False
    
    async def configure_vad(
        self,
        vad_start_probability: float = 0.5,
        vad_end_probability: float = 0.7,
        voice_start_frame_count: int = 3,
        voice_end_frame_count: int = 10
    ):
        """Configure VAD parameters on the server."""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected to WebSocket server")
            return
        
        config_message = {
            "type": "configure_vad",
            "vad_start_probability": vad_start_probability,
            "vad_end_probability": vad_end_probability,
            "voice_start_frame_count": voice_start_frame_count,
            "voice_end_frame_count": voice_end_frame_count,
            "sample_rate": self.sample_rate,
            "chunk_size": self.chunk_size
        }
        
        try:
            await self.websocket.send(json.dumps(config_message))
            logger.info(f"VAD configuration sent: {config_message}")
        except Exception as e:
            logger.error(f"Failed to send VAD configuration: {e}")
    
    def list_audio_devices(self):
        """List available audio input devices."""
        logger.info("Available audio input devices:")
        logger.info("=" * 50)
        
        device_count = self.audio.get_device_count()
        for i in range(device_count):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:  # Input device
                logger.info(f"  {i}: {device_info['name']}")
                logger.info(f"      Max Input Channels: {device_info['maxInputChannels']}")
                logger.info(f"      Default Sample Rate: {device_info['defaultSampleRate']:.0f} Hz")
                logger.info("")
    
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
    
    async def start_audio_streaming(self):
        """Start streaming audio from microphone to WebSocket server."""
        if not self.is_connected or not self.websocket:
            logger.error("Not connected to WebSocket server")
            return
        
        if self.is_recording:
            logger.warning("Audio streaming is already active")
            return
        
        # Auto-detect microphone if not specified
        if self.device_index is None:
            self.device_index = self._auto_detect_microphone()
        
        logger.info("Starting audio streaming...")
        logger.info(f"Sample Rate: {self.sample_rate} Hz")
        logger.info(f"Chunk Size: {self.chunk_size} samples")
        logger.info(f"Device Index: {self.device_index}")
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            self.is_recording = True
            
            # Start audio capture loop
            await self._audio_capture_loop()
            
        except Exception as e:
            logger.error(f"Failed to start audio streaming: {e}")
            await self.stop_audio_streaming()
    
    async def _audio_capture_loop(self):
        """Main audio capture loop."""
        logger.info("🎙️  Audio streaming started - speak into microphone")
        
        try:
            while self.is_recording and not self.stop_event.is_set():
                try:
                    # Read audio data from microphone
                    audio_data = self.stream.read(
                        self.chunk_size,
                        exception_on_overflow=False
                    )
                    
                    # Convert bytes to numpy array
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                    
                    # Send audio data to server as JSON
                    audio_message = {
                        "type": "audio_data",
                        "audio_data": audio_array.tolist()  # Convert to list for JSON
                    }
                    
                    await self.websocket.send(json.dumps(audio_message))
                    
                    # Update statistics
                    self.total_audio_sent += len(audio_array)
                    
                    # Small delay to prevent overwhelming the server
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Error in audio capture loop: {e}")
                    break
        
        except Exception as e:
            logger.error(f"Audio capture loop error: {e}")
        finally:
            logger.info("Audio capture loop stopped")
    
    async def stop_audio_streaming(self):
        """Stop audio streaming."""
        if not self.is_recording:
            return
        
        logger.info("Stopping audio streaming...")
        self.is_recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        logger.info("Audio streaming stopped")
    
    async def handle_server_messages(self):
        """Handle messages from the WebSocket server."""
        if not self.websocket:
            return
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_server_message(data)
                
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from server: {e}")
                except Exception as e:
                    logger.error(f"Error processing server message: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by server")
        except Exception as e:
            logger.error(f"Error handling server messages: {e}")
    
    async def _process_server_message(self, data: Dict[str, Any]):
        """Process a message from the server."""
        message_type = data.get("type")
        
        if message_type == "connection_established":
            self.client_id = data.get("client_id")
            logger.info(f"Connection established - Client ID: {self.client_id}")
            logger.info(f"Default VAD params: {data.get('default_vad_params')}")
        
        elif message_type == "vad_configured":
            logger.info("VAD configuration updated successfully")
            logger.info(f"Current VAD params: {data.get('vad_params')}")
        
        elif message_type == "vad_event":
            await self._handle_vad_event(data)
        
        elif message_type == "error":
            logger.error(f"Server error: {data.get('message')}")
        
        elif message_type == "pong":
            logger.debug(f"Pong received at {data.get('timestamp')}")
        
        elif message_type == "client_status":
            await self._handle_status_update(data)
        
        else:
            logger.debug(f"Unknown message type: {message_type}")
    
    async def _handle_vad_event(self, data: Dict[str, Any]):
        """Handle VAD events from the server."""
        event_type = data.get("event_type")
        timestamp = data.get("timestamp")
        audio_length = data.get("audio_length")
        
        self.vad_events_received += 1
        self.last_voice_event = event_type
        
        if event_type == "voice_start":
            logger.info(f"🎙️  VOICE STARTED at {timestamp}")
        
        elif event_type == "voice_end":
            wav_data_b64 = data.get("wav_data_base64")
            logger.info(f"🔴 VOICE ENDED at {timestamp}")
            if audio_length:
                logger.info(f"   📊 Audio length: {audio_length} bytes")
            if wav_data_b64:
                logger.info(f"   📁 WAV data received (base64 length: {len(wav_data_b64)})")
        
        elif event_type == "voice_continue":
            # Don't log every continue event to avoid spam
            logger.debug(f"🟢 Voice continuing... ({audio_length} bytes)")
    
    async def _handle_status_update(self, data: Dict[str, Any]):
        """Handle status update from server."""
        logger.info("📊 Client Status Update:")
        logger.info(f"   Client ID: {data.get('client_id')}")
        logger.info(f"   Connected: {data.get('connected_at')}")
        logger.info(f"   Audio processed: {data.get('total_audio_processed')} samples")
        logger.info(f"   Voice events: {data.get('voice_events_count')}")
        logger.info(f"   Voice active: {data.get('is_voice_active')}")
    
    async def send_ping(self):
        """Send a ping to the server."""
        if not self.websocket:
            return
        
        ping_message = {
            "type": "ping",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            await self.websocket.send(json.dumps(ping_message))
        except Exception as e:
            logger.error(f"Failed to send ping: {e}")
    
    async def request_status(self):
        """Request status from the server."""
        if not self.websocket:
            return
        
        status_request = {"type": "get_status"}
        
        try:
            await self.websocket.send(json.dumps(status_request))
        except Exception as e:
            logger.error(f"Failed to request status: {e}")
    
    async def run(
        self,
        vad_start_probability: float = 0.7,
        vad_end_probability: float = 0.7,
        voice_start_frame_count: int = 10,
        voice_end_frame_count: int = 50
    ):
        """Run the client with specified VAD configuration."""
        
        # Connect to server
        if not await self.connect():
            return
        
        try:
            # Start message handling
            message_task = asyncio.create_task(self.handle_server_messages())
            
            # Wait a bit for connection to stabilize
            await asyncio.sleep(1)
            
            # Configure VAD parameters
            await self.configure_vad(
                vad_start_probability=vad_start_probability,
                vad_end_probability=vad_end_probability,
                voice_start_frame_count=voice_start_frame_count,
                voice_end_frame_count=voice_end_frame_count
            )
            
            # Wait a bit for configuration
            await asyncio.sleep(1)
            
            # Start audio streaming
            audio_task = asyncio.create_task(self.start_audio_streaming())
            
            # Create periodic status task
            async def periodic_status():
                while not self.stop_event.is_set():
                    await asyncio.sleep(30)  # Every 30 seconds
                    await self.request_status()
            
            status_task = asyncio.create_task(periodic_status())
            
            # Wait for stop event or task completion
            await asyncio.gather(
                message_task,
                audio_task,
                status_task,
                return_exceptions=True
            )
        
        except Exception as e:
            logger.error(f"Error in client run: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the client and clean up resources."""
        logger.info("Stopping VAD WebSocket client...")
        
        # Set stop event
        self.stop_event.set()
        
        # Stop audio streaming
        await self.stop_audio_streaming()
        
        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.is_connected = False
        
        # Clean up audio
        if self.audio:
            self.audio.terminate()
        
        # Print final statistics
        self._print_final_statistics()
        
        logger.info("Client stopped successfully")
    
    def _print_final_statistics(self):
        """Print final client statistics."""
        if not self.connection_start_time:
            return
        
        uptime = (datetime.now() - self.connection_start_time).total_seconds()
        
        logger.info("\n📈 Final Client Statistics:")
        logger.info("=" * 40)
        logger.info(f"📊 Client ID: {self.client_id}")
        logger.info(f"⏱️  Connection time: {uptime:.1f} seconds")
        logger.info(f"🎙️  Total audio sent: {self.total_audio_sent} samples")
        logger.info(f"📢 VAD events received: {self.vad_events_received}")
        logger.info(f"🎯 Last voice event: {self.last_voice_event or 'None'}")
        
        if self.total_audio_sent > 0 and uptime > 0:
            avg_rate = self.total_audio_sent / uptime
            logger.info(f"📈 Average audio rate: {avg_rate:.1f} samples/sec")


async def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Python WebSocket Client for VAD Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python vad_websocket_client.py --server ws://localhost:8765/ws
  python vad_websocket_client.py --server ws://localhost:8765/ws --list-devices
  python vad_websocket_client.py --server ws://localhost:8765/ws --vad-start 0.5 --vad-end 0.4
        """
    )
    
    parser.add_argument(
        "--server",
        default="ws://localhost:8765/ws",
        help="WebSocket server URL (default: ws://localhost:8765/ws)"
    )
    
    parser.add_argument(
        "--device",
        type=int,
        help="Audio input device index (auto-detect if not specified)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        choices=[8000, 16000, 24000, 48000],
        default=16000,
        help="Audio sample rate in Hz (default: 16000)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Audio chunk size in samples (default: 1024)"
    )
    
    # VAD configuration
    parser.add_argument(
        "--vad-start",
        type=float,
        default=0.7,
        help="VAD start probability threshold (default: 0.7)"
    )
    
    parser.add_argument(
        "--vad-end",
        type=float,
        default=0.7,
        help="VAD end probability threshold (default: 0.7)"
    )
    
    parser.add_argument(
        "--voice-start-frames",
        type=int,
        default=10,
        help="Voice start frame count (default: 10)"
    )
    
    parser.add_argument(
        "--voice-end-frames",
        type=int,
        default=50,
        help="Voice end frame count (default: 50)"
    )
    
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create client
    client = VADWebSocketClient(
        server_url=args.server,
        sample_rate=args.sample_rate,
        chunk_size=args.chunk_size,
        device_index=args.device
    )
    
    # Handle list devices
    if args.list_devices:
        client.list_audio_devices()
        client.audio.terminate()
        return
    
    # Run client
    try:
        await client.run(
            vad_start_probability=args.vad_start,
            vad_end_probability=args.vad_end,
            voice_start_frame_count=args.voice_start_frames,
            voice_end_frame_count=args.voice_end_frames
        )
    except KeyboardInterrupt:
        logger.info("Client stopped by user")
    except Exception as e:
        logger.error(f"Client error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
