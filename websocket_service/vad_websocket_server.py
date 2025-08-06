#!/usr/bin/env python3
"""
Real-Time Voice Activity Detection WebSocket Service

This service provides a WebSocket interface for real-time voice activity detection
using the Silero VAD models. It supports multiple concurrent clients and allows
configurable VAD parameters per connection.

Features:
- Multiple concurrent client connections
- Configurable VAD parameters per client
- Real-time audio processing with byte array support
- Voice activity event notifications (start, continue, end)
- Client session management with unique IDs
- Comprehensive error handling and logging
- Health check endpoints
- Metrics and statistics tracking

WebSocket Message Format:
- JSON messages for configuration and control
- Binary messages for audio data (float32 PCM)

Usage:
    python vad_websocket_server.py --host 0.0.0.0 --port 8765
"""

import asyncio
import json
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Set, Optional, Any, List
from dataclasses import dataclass, asdict
import struct
import traceback

import websockets
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import uvicorn

# Add the src directory to the path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from real_time_vad import (
    AsyncVADWrapper,
    VADConfig,
    SampleRate,
    SileroModelVersion
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class VADParameters:
    """VAD configuration parameters for a client session."""
    vad_start_probability: float = 0.7
    vad_end_probability: float = 0.7
    voice_start_frame_count: int = 10
    voice_end_frame_count: int = 50
    sample_rate: int = 16000
    chunk_size: int = 1024


@dataclass
class ClientSession:
    """Client session information."""
    client_id: str
    websocket: WebSocket
    vad_wrapper: AsyncVADWrapper
    vad_params: VADParameters
    connected_at: datetime
    last_activity: datetime
    total_audio_processed: int = 0
    voice_events_count: int = 0
    is_voice_active: bool = False


@dataclass
class VADEvent:
    """Voice activity detection event."""
    event_type: str  # 'voice_start', 'voice_continue', 'voice_end'
    client_id: str
    timestamp: datetime
    data: Optional[bytes] = None
    duration: Optional[float] = None
    audio_length: Optional[int] = None


class VADWebSocketService:
    """WebSocket service for real-time Voice Activity Detection."""
    
    def __init__(self):
        """Initialize the VAD WebSocket service."""
        self.clients: Dict[str, ClientSession] = {}
        self.active_connections: Set[WebSocket] = set()
        self.app = FastAPI(title="VAD WebSocket Service", version="1.0.0")
        self.setup_routes()
        
        # Service statistics
        self.service_start_time = datetime.now()
        self.total_connections = 0
        self.total_audio_processed = 0
        self.total_events_sent = 0
        
    def setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/")
        async def root():
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>VAD WebSocket Service</title>
            </head>
            <body>
                <h1>Real-Time Voice Activity Detection Service</h1>
                <p>WebSocket endpoint: <code>ws://localhost:8765/ws</code></p>
                <p>Health check: <a href="/health">/health</a></p>
                <p>Status: <a href="/status">/status</a></p>
                <p>Active connections: <span id="connections">0</span></p>
                <script>
                    setInterval(() => {
                        fetch('/status')
                            .then(r => r.json())
                            .then(data => {
                                document.getElementById('connections').textContent = data.active_connections;
                            });
                    }, 1000);
                </script>
            </body>
            </html>
            """)
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "service": "VAD WebSocket Service",
                "timestamp": datetime.now().isoformat(),
                "active_connections": len(self.active_connections)
            }
        
        @self.app.get("/status")
        async def status():
            """Service status endpoint."""
            uptime = (datetime.now() - self.service_start_time).total_seconds()
            
            return {
                "service": "VAD WebSocket Service",
                "status": "running",
                "uptime_seconds": uptime,
                "active_connections": len(self.active_connections),
                "total_connections": self.total_connections,
                "total_audio_processed": self.total_audio_processed,
                "total_events_sent": self.total_events_sent,
                "clients": [
                    {
                        "client_id": client.client_id,
                        "connected_at": client.connected_at.isoformat(),
                        "last_activity": client.last_activity.isoformat(),
                        "vad_params": asdict(client.vad_params),
                        "total_audio_processed": client.total_audio_processed,
                        "voice_events_count": client.voice_events_count,
                        "is_voice_active": client.is_voice_active
                    }
                    for client in self.clients.values()
                ]
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket_connection(websocket)
    
    async def handle_websocket_connection(self, websocket: WebSocket):
        """Handle a new WebSocket connection."""
        client_id = str(uuid.uuid4())
        
        try:
            await websocket.accept()
            self.active_connections.add(websocket)
            self.total_connections += 1
            
            logger.info(f"New client connected: {client_id}")
            
            # Create client session with default parameters
            session = await self.create_client_session(client_id, websocket)
            self.clients[client_id] = session
            
            # Send welcome message
            await self.send_json_message(websocket, {
                "type": "connection_established",
                "client_id": client_id,
                "default_vad_params": asdict(session.vad_params),
                "message": "Connected to VAD WebSocket Service"
            })
            
            # Handle messages from this client
            await self.handle_client_messages(session)
            
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
            logger.error(traceback.format_exc())
        finally:
            await self.cleanup_client(client_id, websocket)
    
    async def create_client_session(self, client_id: str, websocket: WebSocket) -> ClientSession:
        """Create a new client session with VAD wrapper."""
        vad_params = VADParameters()
        
        # Create VAD configuration
        vad_config = VADConfig(
            sample_rate=SampleRate(vad_params.sample_rate),
            model_version=SileroModelVersion.V5,
            vad_start_probability=vad_params.vad_start_probability,
            vad_end_probability=vad_params.vad_end_probability,
            voice_start_frame_count=vad_params.voice_start_frame_count,
            voice_end_frame_count=vad_params.voice_end_frame_count,
            enable_denoising=True,
            auto_convert_sample_rate=True,
            buffer_size=vad_params.chunk_size
        )
        
        # Create async VAD wrapper
        vad_wrapper = AsyncVADWrapper(config=vad_config)
        
        # Setup callbacks for this client
        await self.setup_vad_callbacks(client_id, vad_wrapper)
        
        return ClientSession(
            client_id=client_id,
            websocket=websocket,
            vad_wrapper=vad_wrapper,
            vad_params=vad_params,
            connected_at=datetime.now(),
            last_activity=datetime.now()
        )
    
    async def setup_vad_callbacks(self, client_id: str, vad_wrapper: AsyncVADWrapper):
        """Setup VAD callbacks for a specific client."""
        
        async def on_voice_start():
            """Called when voice activity starts."""
            if client_id in self.clients:
                self.clients[client_id].is_voice_active = True
                self.clients[client_id].voice_events_count += 1
                
                event = VADEvent(
                    event_type="voice_start",
                    client_id=client_id,
                    timestamp=datetime.now()
                )
                
                await self.send_vad_event(client_id, event)
                logger.debug(f"Voice started for client {client_id}")
        
        async def on_voice_end(wav_data: bytes):
            """Called when voice activity ends."""
            if client_id in self.clients:
                self.clients[client_id].is_voice_active = False
                
                event = VADEvent(
                    event_type="voice_end",
                    client_id=client_id,
                    timestamp=datetime.now(),
                    data=wav_data,
                    audio_length=len(wav_data)
                )
                
                await self.send_vad_event(client_id, event)
                logger.debug(f"Voice ended for client {client_id}, WAV data: {len(wav_data)} bytes")
        
        async def on_voice_continue(pcm_data: bytes):
            """Called continuously during voice activity."""
            if client_id in self.clients:
                event = VADEvent(
                    event_type="voice_continue",
                    client_id=client_id,
                    timestamp=datetime.now(),
                    data=pcm_data,
                    audio_length=len(pcm_data)
                )
                
                await self.send_vad_event(client_id, event)
        
        # Set the async callbacks
        vad_wrapper.set_async_callbacks(
            voice_start_callback=on_voice_start,
            voice_end_callback=on_voice_end,
            voice_continue_callback=on_voice_continue
        )
    
    async def handle_client_messages(self, session: ClientSession):
        """Handle messages from a client."""
        client_id = session.client_id
        websocket = session.websocket
        
        try:
            async for message in websocket.iter_text():
                try:
                    # Update last activity
                    session.last_activity = datetime.now()
                    
                    # Parse JSON message
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    if message_type == "configure_vad":
                        await self.handle_vad_configuration(session, data)
                    
                    elif message_type == "audio_data":
                        await self.handle_audio_data_json(session, data)
                    
                    elif message_type == "ping":
                        await self.send_json_message(websocket, {
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    elif message_type == "get_status":
                        await self.send_client_status(session)
                    
                    else:
                        await self.send_json_message(websocket, {
                            "type": "error",
                            "message": f"Unknown message type: {message_type}"
                        })
                
                except json.JSONDecodeError as e:
                    await self.send_json_message(websocket, {
                        "type": "error",
                        "message": f"Invalid JSON: {e}"
                    })
                except Exception as e:
                    logger.error(f"Error processing message from {client_id}: {e}")
                    await self.send_json_message(websocket, {
                        "type": "error",
                        "message": f"Processing error: {e}"
                    })
        
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during message handling")
        except Exception as e:
            logger.error(f"Error in message handling for {client_id}: {e}")
    
    async def handle_audio_data_binary(self, session: ClientSession, audio_bytes: bytes):
        """Handle binary audio data from client."""
        try:
            # Convert bytes to float32 array
            # Assuming audio_bytes contains float32 PCM data
            audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
            
            # Update session statistics
            session.total_audio_processed += len(audio_array)
            session.last_activity = datetime.now()
            self.total_audio_processed += len(audio_array)
            
            # Process audio with VAD
            await session.vad_wrapper.process_audio_data_async(audio_array)
            
        except Exception as e:
            logger.error(f"Error processing audio data for {session.client_id}: {e}")
            await self.send_json_message(session.websocket, {
                "type": "error",
                "message": f"Audio processing error: {e}"
            })
    
    async def handle_audio_data_json(self, session: ClientSession, data: Dict[str, Any]):
        """Handle audio data received via JSON message."""
        try:
            audio_data = data.get("audio_data")
            if not audio_data:
                raise ValueError("No audio_data field in message")
            
            # Convert list to numpy array
            if isinstance(audio_data, list):
                audio_array = np.array(audio_data, dtype=np.float32)
            else:
                raise ValueError("audio_data must be a list of float values")
            
            # Update session statistics
            session.total_audio_processed += len(audio_array)
            session.last_activity = datetime.now()
            self.total_audio_processed += len(audio_array)
            
            # Process audio with VAD
            await session.vad_wrapper.process_audio_data_async(audio_array)
            
        except Exception as e:
            logger.error(f"Error processing JSON audio data for {session.client_id}: {e}")
            await self.send_json_message(session.websocket, {
                "type": "error",
                "message": f"Audio processing error: {e}"
            })
    
    async def handle_vad_configuration(self, session: ClientSession, data: Dict[str, Any]):
        """Handle VAD configuration update from client."""
        try:
            # Extract VAD parameters
            vad_start_probability = data.get("vad_start_probability", session.vad_params.vad_start_probability)
            vad_end_probability = data.get("vad_end_probability", session.vad_params.vad_end_probability)
            voice_start_frame_count = data.get("voice_start_frame_count", session.vad_params.voice_start_frame_count)
            voice_end_frame_count = data.get("voice_end_frame_count", session.vad_params.voice_end_frame_count)
            sample_rate = data.get("sample_rate", session.vad_params.sample_rate)
            chunk_size = data.get("chunk_size", session.vad_params.chunk_size)
            
            # Validate parameters
            if not (0.0 <= vad_start_probability <= 1.0):
                raise ValueError("vad_start_probability must be between 0.0 and 1.0")
            if not (0.0 <= vad_end_probability <= 1.0):
                raise ValueError("vad_end_probability must be between 0.0 and 1.0")
            if voice_start_frame_count < 1:
                raise ValueError("voice_start_frame_count must be >= 1")
            if voice_end_frame_count < 1:
                raise ValueError("voice_end_frame_count must be >= 1")
            if sample_rate not in [8000, 16000, 24000, 48000]:
                raise ValueError("sample_rate must be 8000, 16000, 24000, or 48000")
            
            # Update VAD configuration
            await session.vad_wrapper.set_thresholds_async(
                vad_start_probability=vad_start_probability,
                vad_end_probability=vad_end_probability,
                voice_start_frame_count=voice_start_frame_count,
                voice_end_frame_count=voice_end_frame_count
            )
            
            # Update session parameters
            session.vad_params.vad_start_probability = vad_start_probability
            session.vad_params.vad_end_probability = vad_end_probability
            session.vad_params.voice_start_frame_count = voice_start_frame_count
            session.vad_params.voice_end_frame_count = voice_end_frame_count
            session.vad_params.sample_rate = sample_rate
            session.vad_params.chunk_size = chunk_size
            
            # Send confirmation
            await self.send_json_message(session.websocket, {
                "type": "vad_configured",
                "vad_params": asdict(session.vad_params),
                "message": "VAD parameters updated successfully"
            })
            
            logger.info(f"VAD configuration updated for client {session.client_id}")
            
        except Exception as e:
            logger.error(f"Error configuring VAD for {session.client_id}: {e}")
            await self.send_json_message(session.websocket, {
                "type": "error",
                "message": f"VAD configuration error: {e}"
            })
    
    async def send_vad_event(self, client_id: str, event: VADEvent):
        """Send a VAD event to the client."""
        if client_id not in self.clients:
            return
        
        session = self.clients[client_id]
        
        try:
            message = {
                "type": "vad_event",
                "event_type": event.event_type,
                "client_id": event.client_id,
                "timestamp": event.timestamp.isoformat(),
            }
            
            # Add optional fields
            if event.audio_length is not None:
                message["audio_length"] = event.audio_length
            
            if event.duration is not None:
                message["duration"] = event.duration
            
            # For voice_end events, we can include WAV data as base64
            if event.event_type == "voice_end" and event.data:
                import base64
                message["wav_data_base64"] = base64.b64encode(event.data).decode('utf-8')
            
            await self.send_json_message(session.websocket, message)
            self.total_events_sent += 1
            
        except Exception as e:
            logger.error(f"Error sending VAD event to {client_id}: {e}")
    
    async def send_client_status(self, session: ClientSession):
        """Send current status to client."""
        try:
            vad_stats = await session.vad_wrapper.get_statistics_async()
            
            status = {
                "type": "client_status",
                "client_id": session.client_id,
                "connected_at": session.connected_at.isoformat(),
                "last_activity": session.last_activity.isoformat(),
                "vad_params": asdict(session.vad_params),
                "total_audio_processed": session.total_audio_processed,
                "voice_events_count": session.voice_events_count,
                "is_voice_active": session.is_voice_active,
                "vad_statistics": vad_stats
            }
            
            await self.send_json_message(session.websocket, status)
            
        except Exception as e:
            logger.error(f"Error sending status to {session.client_id}: {e}")
    
    async def send_json_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send a JSON message to a WebSocket client."""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {e}")
    
    async def cleanup_client(self, client_id: str, websocket: WebSocket):
        """Clean up client session and resources."""
        try:
            # Remove from active connections
            self.active_connections.discard(websocket)
            
            # Clean up client session
            if client_id in self.clients:
                session = self.clients[client_id]
                
                # Clean up VAD wrapper
                if session.vad_wrapper:
                    await session.vad_wrapper.acleanup()
                
                # Remove from clients dict
                del self.clients[client_id]
                
                logger.info(f"Cleaned up client {client_id}")
        
        except Exception as e:
            logger.error(f"Error cleaning up client {client_id}: {e}")
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the WebSocket server."""
        logger.info(f"Starting VAD WebSocket Service on {host}:{port}")
        
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()


def main():
    """Main function to start the WebSocket service."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Real-Time Voice Activity Detection WebSocket Service"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind the server to (default: 8765)"
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
    
    # Create and start service
    service = VADWebSocketService()
    
    try:
        asyncio.run(service.start_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
        raise


if __name__ == "__main__":
    main()
