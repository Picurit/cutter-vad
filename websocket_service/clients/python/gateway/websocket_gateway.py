"""
WebSocket Gateway for Python VAD Client

This module provides the WebSocket gateway that manages connections to the VAD server,
handles configuration, and streams audio data while receiving VAD events.
"""

import asyncio
import json
import logging
import time
from typing import Optional, Dict, Any, Callable, List, TYPE_CHECKING
from urllib.parse import urlencode

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

if TYPE_CHECKING:
    from sources.audio_sources import AudioSource


class VADEventHandler:
    """Handler for VAD events received from server."""
    
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
        self.segment_count = 0
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    async def handle_event(self, event_data: Dict[str, Any]):
        """
        Handle a VAD event from the server.
        
        Args:
            event_data: Event data as dictionary
        """
        event_type = event_data.get("event")
        timestamp = event_data.get("timestamp_ms", 0)
        
        # Store event
        self.events.append(event_data)
        
        # Log event
        if event_type == "VOICE_START":
            segment_index = event_data.get("segment_index", 0)
            self.logger.info(f"🎙️  VOICE START - Segment #{segment_index} at {self._format_timestamp(timestamp)}")
            
        elif event_type == "VOICE_END":
            segment_index = event_data.get("segment_index", 0)
            duration = event_data.get("duration_ms", 0)
            self.segment_count = max(self.segment_count, segment_index + 1)
            self.logger.info(f"🔴 VOICE END - Segment #{segment_index}, Duration: {duration}ms")
            
        elif event_type == "VOICE_CONTINUE":
            segment_index = event_data.get("segment_index", 0)
            self.logger.debug(f"🔄 VOICE CONTINUE - Segment #{segment_index}")
            
        elif event_type == "TIMEOUT":
            message = event_data.get("message", "")
            self.logger.warning(f"⏰ TIMEOUT - {message}")
            
        elif event_type == "ERROR":
            message = event_data.get("message", "")
            self.logger.error(f"❌ ERROR - {message}")
            
        elif event_type == "INFO":
            message = event_data.get("message", "")
            self.logger.info(f"ℹ️  INFO - {message}")
        
        else:
            self.logger.debug(f"Unknown event type: {event_type}")
    
    def _format_timestamp(self, timestamp_ms: int) -> str:
        """Format timestamp for display."""
        timestamp_s = timestamp_ms / 1000
        return time.strftime("%H:%M:%S", time.localtime(timestamp_s))
    
    def get_voice_segments(self) -> List[Dict[str, Any]]:
        """Get list of detected voice segments."""
        segments = []
        
        # Group START/END events by segment_index
        start_events = {e["segment_index"]: e for e in self.events if e.get("event") == "VOICE_START"}
        end_events = {e["segment_index"]: e for e in self.events if e.get("event") == "VOICE_END"}
        
        for segment_index in sorted(set(start_events.keys()) | set(end_events.keys())):
            start_event = start_events.get(segment_index)
            end_event = end_events.get(segment_index)
            
            if start_event and end_event:
                segments.append({
                    "segment_index": segment_index,
                    "start_ms": start_event["timestamp_ms"],
                    "end_ms": end_event["timestamp_ms"],
                    "duration_ms": end_event.get("duration_ms", 0)
                })
        
        return segments
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        segments = self.get_voice_segments()
        
        return {
            "total_events": len(self.events),
            "voice_segments": len(segments),
            "total_voice_duration_ms": sum(s["duration_ms"] for s in segments),
            "events_by_type": {
                event_type: len([e for e in self.events if e.get("event") == event_type])
                for event_type in ["VOICE_START", "VOICE_END", "VOICE_CONTINUE", "TIMEOUT", "ERROR", "INFO"]
            }
        }


class WebSocketGateway:
    """WebSocket gateway for VAD server communication."""
    
    def __init__(self, 
                 server_url: str = "ws://localhost:8000/vad",
                 config: Optional[Dict[str, Any]] = None,
                 event_handler: Optional[VADEventHandler] = None):
        """
        Initialize WebSocket gateway.
        
        Args:
            server_url: WebSocket server URL (without query params)
            config: Initial configuration parameters
            event_handler: Event handler for VAD events
        """
        self.server_url = server_url
        self.config = config or {}
        self.event_handler = event_handler or VADEventHandler()
        
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
    
    def _build_connection_url(self) -> str:
        """Build WebSocket connection URL with query parameters."""
        if not self.config:
            return self.server_url
        
        # Build query string from config
        query_params = {}
        for key, value in self.config.items():
            if value is not None:
                query_params[key] = str(value)
        
        if query_params:
            query_string = urlencode(query_params)
            return f"{self.server_url}?{query_string}"
        else:
            return self.server_url
    
    async def connect(self) -> bool:
        """
        Connect to WebSocket server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            connection_url = self._build_connection_url()
            self.logger.info(f"Connecting to {connection_url}")
            
            self.websocket = await websockets.connect(
                connection_url,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10
            )
            
            self.connected = True
            self.reconnect_attempts = 0
            self.logger.info("WebSocket connection established")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        self.connected = False
        self.logger.info("WebSocket connection closed")
    
    async def send_config(self, config_update: Dict[str, Any]):
        """
        Send configuration update to server.
        
        Args:
            config_update: Configuration parameters to update
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server")
        
        config_message = {
            "type": "CONFIG",
            **config_update
        }
        
        try:
            await self.websocket.send(json.dumps(config_message))
            self.logger.info(f"Sent configuration update: {config_update}")
            
        except Exception as e:
            self.logger.error(f"Failed to send config: {e}")
            raise
    
    async def send_audio_frame(self, frame_data: bytes):
        """
        Send audio frame to server.
        
        Args:
            frame_data: PCM audio frame as bytes
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server")
        
        try:
            await self.websocket.send(frame_data)
            
        except Exception as e:
            self.logger.error(f"Failed to send audio frame: {e}")
            raise
    
    async def send_heartbeat(self):
        """Send heartbeat message to server."""
        if not self.connected or not self.websocket:
            return
        
        heartbeat_message = {"type": "HEARTBEAT"}
        
        try:
            await self.websocket.send(json.dumps(heartbeat_message))
            self.logger.debug("Sent heartbeat")
            
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat: {e}")
    
    async def listen_for_events(self):
        """
        Listen for events from server in a loop.
        Should be run as a background task.
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server")
        
        try:
            async for message in self.websocket:
                try:
                    # Parse JSON message
                    event_data = json.loads(message)
                    
                    # Handle event
                    await self.event_handler.handle_event(event_data)
                    
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse server message: {e}")
                except Exception as e:
                    self.logger.error(f"Error handling event: {e}")
        
        except ConnectionClosed:
            self.logger.warning("Server connection closed")
            self.connected = False
        except WebSocketException as e:
            self.logger.error(f"WebSocket error: {e}")
            self.connected = False
        except Exception as e:
            self.logger.error(f"Unexpected error in event listener: {e}")
            self.connected = False
    
    async def stream_audio(self, audio_source: "AudioSource", duration_seconds: Optional[float] = None):
        """
        Stream audio from source to server.
        
        Args:
            audio_source: Audio source to stream from
            duration_seconds: Maximum duration to stream (None for unlimited)
        """
        if not self.connected or not self.websocket:
            raise RuntimeError("Not connected to server")
        
        self.logger.info("Starting audio stream")
        
        start_time = time.time()
        frame_count = 0
        
        try:
            async for frame_data in audio_source.generate_frames():
                # Check duration limit
                if duration_seconds and (time.time() - start_time) >= duration_seconds:
                    self.logger.info(f"Reached duration limit: {duration_seconds}s")
                    break
                
                # Send frame
                await self.send_audio_frame(frame_data)
                frame_count += 1
                
                # Log progress occasionally
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    self.logger.debug(f"Streamed {frame_count} frames in {elapsed:.1f}s")
        
        except Exception as e:
            self.logger.error(f"Error during audio streaming: {e}")
            raise
        
        finally:
            elapsed = time.time() - start_time
            self.logger.info(f"Audio streaming completed: {frame_count} frames in {elapsed:.1f}s")
    
    async def run_with_reconnect(self, 
                                audio_source: "AudioSource", 
                                duration_seconds: Optional[float] = None):
        """
        Run audio streaming with automatic reconnection.
        
        Args:
            audio_source: Audio source to stream from
            duration_seconds: Maximum duration to stream
        """
        while self.reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Connect if not connected
                if not self.connected:
                    if not await self.connect():
                        self.reconnect_attempts += 1
                        await asyncio.sleep(2 ** self.reconnect_attempts)  # Exponential backoff
                        continue
                
                # Start event listener
                event_task = asyncio.create_task(self.listen_for_events())
                
                # Start audio streaming
                stream_task = asyncio.create_task(
                    self.stream_audio(audio_source, duration_seconds)
                )
                
                # Wait for either task to complete
                done, pending = await asyncio.wait(
                    [event_task, stream_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Cancel remaining tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check for exceptions
                for task in done:
                    if task.exception():
                        raise task.exception()
                
                # If we get here, streaming completed successfully
                break
                
            except (ConnectionClosed, WebSocketException) as e:
                self.logger.warning(f"Connection lost: {e}")
                self.connected = False
                self.reconnect_attempts += 1
                
                if self.reconnect_attempts < self.max_reconnect_attempts:
                    backoff_time = 2 ** self.reconnect_attempts
                    self.logger.info(f"Reconnecting in {backoff_time}s (attempt {self.reconnect_attempts})")
                    await asyncio.sleep(backoff_time)
                else:
                    self.logger.error("Max reconnection attempts reached")
                    raise
            
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                raise
        
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            raise RuntimeError("Failed to establish stable connection after maximum attempts")


# Example usage and testing
async def test_websocket_gateway():
    """Test function for WebSocketGateway."""
    logging.basicConfig(level=logging.INFO)
    
    # Test configuration
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
        "timeout": 30.0
    }
    
    # Create gateway
    gateway = WebSocketGateway(
        server_url="ws://localhost:8000/vad",
        config=config
    )
    
    try:
        # Connect
        if await gateway.connect():
            print("Connected successfully")
            
            # Send heartbeat
            await gateway.send_heartbeat()
            
            # Listen for a few seconds
            event_task = asyncio.create_task(gateway.listen_for_events())
            await asyncio.sleep(5)
            event_task.cancel()
            
            # Print statistics
            stats = gateway.event_handler.get_statistics()
            print(f"Statistics: {stats}")
            
        else:
            print("Failed to connect")
    
    except Exception as e:
        print(f"Test failed: {e}")
    
    finally:
        await gateway.disconnect()


if __name__ == "__main__":
    asyncio.run(test_websocket_gateway())
