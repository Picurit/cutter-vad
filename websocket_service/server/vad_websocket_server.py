"""
WebSocket VAD Server

FastAPI-based WebSocket server for real-time Voice Activity Detection.
Supports multiple concurrent clients with isolated VAD state and configurable
audio streaming modes (PCM, Opus, AAC).

Usage:
    uvicorn websocket_service.server.vad_websocket_server:app --host 0.0.0.0 --port 8000

Connection URL example:
    ws://localhost:8000/vad?mode=pcm&sample_rate=16000&channels=1&frame_duration_ms=30&start_probability=0.4&end_probability=0.3&start_frame_count=6&end_frame_count=12&timeout=2.0
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Optional, Any, Union
from urllib.parse import parse_qs

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from pydantic import BaseModel, ValidationError
import uvicorn
import json

# Import VAD components
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from real_time_vad import VADWrapper, VADConfig, SampleRate, SileroModelVersion


# =============================================================================
# Pydantic Models for Configuration and Events
# =============================================================================

class AudioMode(BaseModel):
    """Audio streaming mode configuration."""
    mode: str = "pcm"  # "pcm", "opus", "aac"
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # bytes per sample
    frame_duration_ms: int = 30


class VADParameters(BaseModel):
    """VAD detection parameters."""
    start_probability: float = 0.4
    end_probability: float = 0.3
    start_frame_count: int = 6
    end_frame_count: int = 12


class ClientConfig(BaseModel):
    """Complete client configuration."""
    audio: AudioMode = AudioMode()
    vad: VADParameters = VADParameters()
    timeout: float = 0.0  # 0 means no timeout


class ConfigMessage(BaseModel):
    """Configuration message from client."""
    type: str = "CONFIG"
    mode: Optional[str] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    sample_width: Optional[int] = None
    frame_duration_ms: Optional[int] = None
    start_probability: Optional[float] = None
    end_probability: Optional[float] = None
    start_frame_count: Optional[int] = None
    end_frame_count: Optional[int] = None
    timeout: Optional[float] = None


class VADEvent(BaseModel):
    """Base VAD event."""
    event: str
    timestamp_ms: int
    segment_index: Optional[int] = None


class VoiceStartEvent(VADEvent):
    """Voice start event."""
    event: str = "VOICE_START"


class VoiceContinueEvent(VADEvent):
    """Voice continue event."""
    event: str = "VOICE_CONTINUE"


class VoiceEndEvent(VADEvent):
    """Voice end event."""
    event: str = "VOICE_END"
    segment_start_ms: int
    segment_end_ms: int
    duration_ms: int


class TimeoutEvent(VADEvent):
    """Timeout event."""
    event: str = "TIMEOUT"
    message: str = "no voice detected in configured timeout"


class ErrorEvent(VADEvent):
    """Error event."""
    event: str = "ERROR"
    message: str


class InfoEvent(VADEvent):
    """Info event."""
    event: str = "INFO"
    message: str


# =============================================================================
# Audio Decoder (for Opus/AAC support)
# =============================================================================

class AudioDecoder:
    """Audio decoder for Opus and AAC formats using PyAV."""
    
    def __init__(self, codec: str, sample_rate: int, channels: int):
        """
        Initialize audio decoder.
        
        Args:
            codec: "opus" or "aac"
            sample_rate: Target sample rate
            channels: Number of channels
        """
        self.codec = codec
        self.sample_rate = sample_rate
        self.channels = channels
        
        try:
            import av
            self.av = av
            self._decoder = None
            self._initialize_decoder()
        except ImportError:
            raise RuntimeError(f"PyAV is required for {codec} decoding but not installed")
    
    def _initialize_decoder(self):
        """Initialize the decoder."""
        if self.codec == "opus":
            self._decoder = self.av.CodecContext.create("libopus", "r")
        elif self.codec == "aac":
            self._decoder = self.av.CodecContext.create("aac", "r")
        else:
            raise ValueError(f"Unsupported codec: {self.codec}")
        
        self._decoder.sample_rate = self.sample_rate
        self._decoder.channels = self.channels
    
    def decode_packet(self, packet_data: bytes) -> Optional[np.ndarray]:
        """
        Decode a single packet to PCM data.
        
        Args:
            packet_data: Encoded packet bytes
            
        Returns:
            PCM data as float32 numpy array, or None if decoding failed
        """
        try:
            # Create packet from bytes
            packet = self.av.Packet(packet_data)
            
            # Decode packet
            frames = self._decoder.decode(packet)
            
            if frames:
                # Convert to numpy array
                frame = frames[0]  # Get first frame
                audio_array = frame.to_ndarray()
                
                # Convert to float32 and normalize
                if audio_array.dtype != np.float32:
                    audio_array = audio_array.astype(np.float32)
                    
                # Normalize if needed
                if audio_array.max() > 1.0 or audio_array.min() < -1.0:
                    audio_array = audio_array / np.max(np.abs(audio_array))
                
                return audio_array.flatten() if len(audio_array.shape) > 1 else audio_array
            
            return None
            
        except Exception as e:
            logging.error(f"Failed to decode {self.codec} packet: {e}")
            return None


# =============================================================================
# Client State Management
# =============================================================================

class ClientState:
    """Manages state for a single WebSocket client."""
    
    def __init__(self, client_id: str, websocket: WebSocket, config: ClientConfig):
        self.client_id = client_id
        self.websocket = websocket
        self.config = config
        self.start_time = time.time()
        
        # VAD state
        self.vad_wrapper: Optional[VADWrapper] = None
        self.segment_index = 0
        self.voice_start_time: Optional[float] = None
        self.last_voice_time: Optional[float] = None
        
        # Audio decoder for Opus/AAC
        self.audio_decoder: Optional[AudioDecoder] = None
        
        # Frame tracking
        self.expected_frame_bytes = self._calculate_frame_bytes()
        
        # Timeout handling
        self.timeout_task: Optional[asyncio.Task] = None
        
        # Initialize components
        self._initialize_vad()
        self._initialize_decoder()
    
    def _calculate_frame_bytes(self) -> int:
        """Calculate expected bytes per frame for PCM mode."""
        if self.config.audio.mode == "pcm":
            return int(
                self.config.audio.sample_rate *
                (self.config.audio.frame_duration_ms / 1000) *
                self.config.audio.channels *
                self.config.audio.sample_width
            )
        return 0  # Variable size for Opus/AAC
    
    def _initialize_vad(self):
        """Initialize VAD wrapper with client configuration."""
        try:
            # Create VAD configuration
            frame_samples = int(self.config.audio.sample_rate * (self.config.audio.frame_duration_ms / 1000))
            
            # Get model path from environment variable if set
            model_path = os.environ.get('VAD_MODEL_PATH')
            if model_path:
                logging.info(f"Client {self.client_id}: Using model path from VAD_MODEL_PATH: {model_path}")
            else:
                logging.info(f"Client {self.client_id}: Using default model path resolution")
            
            vad_config = VADConfig(
                sample_rate=SampleRate(self.config.audio.sample_rate),
                model_version=SileroModelVersion.V5,
                model_path=model_path,  # Use environment variable for model path
                vad_start_probability=self.config.vad.start_probability,
                vad_end_probability=self.config.vad.end_probability,
                voice_start_frame_count=self.config.vad.start_frame_count,
                voice_end_frame_count=self.config.vad.end_frame_count,
                enable_denoising=True,
                auto_convert_sample_rate=True,
                buffer_size=frame_samples  # Match frame size instead of fixed 512
            )
            
            # Create VAD wrapper
            self.vad_wrapper = VADWrapper(config=vad_config)
            
            # Set up callbacks
            self.vad_wrapper.set_callbacks(
                voice_start_callback=self._on_voice_start,
                voice_end_callback=self._on_voice_end,
                voice_continue_callback=self._on_voice_continue
            )
            
            logging.info(f"Client {self.client_id}: VAD initialized successfully")
            
        except Exception as e:
            logging.error(f"Client {self.client_id}: Failed to initialize VAD: {e}")
            raise
    
    def _initialize_decoder(self):
        """Initialize audio decoder if needed."""
        if self.config.audio.mode in ["opus", "aac"]:
            try:
                self.audio_decoder = AudioDecoder(
                    codec=self.config.audio.mode,
                    sample_rate=self.config.audio.sample_rate,
                    channels=self.config.audio.channels
                )
                logging.info(f"Client {self.client_id}: {self.config.audio.mode.upper()} decoder initialized")
                
            except Exception as e:
                logging.error(f"Client {self.client_id}: Failed to initialize {self.config.audio.mode} decoder: {e}")
                raise
    
    def update_config(self, new_config: ClientConfig):
        """Update client configuration."""
        old_config = self.config
        self.config = new_config
        
        # Recalculate frame bytes if audio config changed
        if old_config.audio != new_config.audio:
            self.expected_frame_bytes = self._calculate_frame_bytes()
            
            # Reinitialize decoder if mode changed
            if old_config.audio.mode != new_config.audio.mode:
                self._initialize_decoder()
        
        # Update VAD if parameters changed
        if old_config.vad != new_config.vad or old_config.audio.sample_rate != new_config.audio.sample_rate:
            self._initialize_vad()
        
        logging.info(f"Client {self.client_id}: Configuration updated")
    
    async def process_audio_frame(self, frame_data: bytes):
        """Process a single audio frame."""
        try:
            # Add debugging
            logging.info(f"Client {self.client_id}: Received audio frame of {len(frame_data)} bytes")
            
            # Validate frame size for PCM mode
            if self.config.audio.mode == "pcm":
                if len(frame_data) != self.expected_frame_bytes:
                    await self._send_error(f"Invalid frame size: expected {self.expected_frame_bytes}, got {len(frame_data)}")
                    return
                
                # Convert PCM bytes to float32 array
                if self.config.audio.sample_width == 2:
                    # 16-bit signed int, little-endian
                    audio_array = np.frombuffer(frame_data, dtype=np.int16).astype(np.float32) / 32767.0
                elif self.config.audio.sample_width == 4:
                    # 32-bit float
                    audio_array = np.frombuffer(frame_data, dtype=np.float32)
                else:
                    await self._send_error(f"Unsupported sample width: {self.config.audio.sample_width}")
                    return
                    
            else:
                # Decode Opus/AAC packet
                if not self.audio_decoder:
                    await self._send_error(f"No decoder available for {self.config.audio.mode}")
                    return
                
                audio_array = self.audio_decoder.decode_packet(frame_data)
                if audio_array is None:
                    await self._send_error(f"Failed to decode {self.config.audio.mode} packet")
                    return
            
            # Process with VAD
            if self.vad_wrapper:
                # Process with VAD (reduced logging)
                self.frame_count = getattr(self, 'frame_count', 0) + 1
                
                if self.frame_count % 100 == 0:  # Log every 100 frames instead of every frame
                    rms = np.sqrt(np.mean(audio_array**2))
                    logging.debug(f"Client {self.client_id}: Frame {self.frame_count}, RMS: {rms:.6f}, Range: [{audio_array.min():.6f}, {audio_array.max():.6f}]")
                
                self.vad_wrapper.process_audio_data(audio_array)
                
                # Update last voice time for timeout tracking
                self.last_voice_time = time.time()
                
                # Start timeout task if needed
                if self.config.timeout > 0 and not self.timeout_task:
                    self.timeout_task = asyncio.create_task(self._timeout_monitor())
            else:
                logging.warning(f"Client {self.client_id}: VAD wrapper not initialized")
        
        except Exception as e:
            logging.error(f"Client {self.client_id}: Error processing audio frame: {e}")
            await self._send_error(f"Audio processing error: {e}")
    
    async def _timeout_monitor(self):
        """Monitor for voice timeout."""
        try:
            while True:
                await asyncio.sleep(1.0)  # Check every second
                
                if self.last_voice_time and self.config.timeout > 0:
                    time_since_voice = time.time() - self.last_voice_time
                    if time_since_voice >= self.config.timeout:
                        await self._send_timeout()
                        break
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logging.error(f"Client {self.client_id}: Timeout monitor error: {e}")
    
    def _on_voice_start(self):
        """Called when voice activity starts."""
        logging.info(f"Client {self.client_id}: VAD detected VOICE_START")
        self.voice_start_time = time.time()
        asyncio.create_task(self._send_voice_start())
    
    def _on_voice_end(self, wav_data: bytes):
        """Called when voice activity ends."""
        logging.info(f"Client {self.client_id}: VAD detected VOICE_END")
        asyncio.create_task(self._send_voice_end())
        self.segment_index += 1
        
        # Reset timeout
        if self.timeout_task:
            self.timeout_task.cancel()
            self.timeout_task = None
        
        if self.config.timeout > 0:
            self.timeout_task = asyncio.create_task(self._timeout_monitor())
    
    def _on_voice_continue(self, pcm_data: bytes):
        """Called during ongoing voice activity."""
        # Reduce logging for continue events
        if not hasattr(self, '_continue_count'):
            self._continue_count = 0
        self._continue_count += 1
        
        if self._continue_count % 50 == 0:  # Log every 50th continue event
            logging.debug(f"Client {self.client_id}: VAD continue event #{self._continue_count}")
        
        asyncio.create_task(self._send_voice_continue())
    
    async def _send_voice_start(self):
        """Send voice start event."""
        event = VoiceStartEvent(
            timestamp_ms=int(time.time() * 1000),
            segment_index=self.segment_index
        )
        await self._send_event(event)
    
    async def _send_voice_continue(self):
        """Send voice continue event."""
        event = VoiceContinueEvent(
            timestamp_ms=int(time.time() * 1000),
            segment_index=self.segment_index
        )
        await self._send_event(event)
    
    async def _send_voice_end(self):
        """Send voice end event."""
        current_time = time.time()
        start_ms = int(self.voice_start_time * 1000) if self.voice_start_time else int(current_time * 1000)
        end_ms = int(current_time * 1000)
        
        event = VoiceEndEvent(
            timestamp_ms=end_ms,
            segment_index=self.segment_index,
            segment_start_ms=start_ms,
            segment_end_ms=end_ms,
            duration_ms=end_ms - start_ms
        )
        await self._send_event(event)
    
    async def _send_timeout(self):
        """Send timeout event."""
        event = TimeoutEvent(
            timestamp_ms=int(time.time() * 1000)
        )
        await self._send_event(event)
    
    async def _send_error(self, message: str):
        """Send error event."""
        event = ErrorEvent(
            timestamp_ms=int(time.time() * 1000),
            message=message
        )
        await self._send_event(event)
    
    async def _send_info(self, message: str):
        """Send info event."""
        event = InfoEvent(
            timestamp_ms=int(time.time() * 1000),
            message=message
        )
        await self._send_event(event)
    
    async def _send_event(self, event: VADEvent):
        """Send event to client."""
        try:
            message = event.model_dump_json()
            await self.websocket.send_text(message)
            
        except Exception as e:
            logging.error(f"Client {self.client_id}: Failed to send event: {e}")
    
    def cleanup(self):
        """Clean up client resources."""
        if self.timeout_task:
            self.timeout_task.cancel()
        
        if self.vad_wrapper:
            self.vad_wrapper.cleanup()
        
        logging.info(f"Client {self.client_id}: Resources cleaned up")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="VAD WebSocket Server",
    description="Real-time Voice Activity Detection WebSocket Server",
    version="1.0.0"
)

# Global client state storage
clients: Dict[str, ClientState] = {}

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_query_params(query_string: str) -> Dict[str, Any]:
    """Parse query parameters from connection URL."""
    params = {}
    if query_string:
        parsed = parse_qs(query_string)
        for key, values in parsed.items():
            if values:
                value = values[0]  # Take first value
                
                # Convert to appropriate types
                if key in ["sample_rate", "channels", "sample_width", "frame_duration_ms", 
                          "start_frame_count", "end_frame_count"]:
                    params[key] = int(value)
                elif key in ["start_probability", "end_probability", "timeout"]:
                    params[key] = float(value)
                else:
                    params[key] = value
    
    return params


def create_client_config(query_params: Dict[str, Any], config_message: Optional[ConfigMessage] = None) -> ClientConfig:
    """Create client configuration from query params and/or config message."""
    
    # Start with defaults
    config_data = {
        "audio": {
            "mode": "pcm",
            "sample_rate": 16000,
            "channels": 1,
            "sample_width": 2,
            "frame_duration_ms": 30
        },
        "vad": {
            "start_probability": 0.4,
            "end_probability": 0.3,
            "start_frame_count": 6,
            "end_frame_count": 12
        },
        "timeout": 0.0
    }
    
    # Apply query parameters
    for key, value in query_params.items():
        if key in ["mode"]:
            config_data["audio"][key] = value
        elif key in ["sample_rate", "channels", "sample_width", "frame_duration_ms"]:
            config_data["audio"][key] = value
        elif key in ["start_probability", "end_probability", "start_frame_count", "end_frame_count"]:
            config_data["vad"][key] = value
        elif key == "timeout":
            config_data["timeout"] = value
    
    # Apply config message (takes priority over query params)
    if config_message:
        if config_message.mode is not None:
            config_data["audio"]["mode"] = config_message.mode
        if config_message.sample_rate is not None:
            config_data["audio"]["sample_rate"] = config_message.sample_rate
        if config_message.channels is not None:
            config_data["audio"]["channels"] = config_message.channels
        if config_message.sample_width is not None:
            config_data["audio"]["sample_width"] = config_message.sample_width
        if config_message.frame_duration_ms is not None:
            config_data["audio"]["frame_duration_ms"] = config_message.frame_duration_ms
        if config_message.start_probability is not None:
            config_data["vad"]["start_probability"] = config_message.start_probability
        if config_message.end_probability is not None:
            config_data["vad"]["end_probability"] = config_message.end_probability
        if config_message.start_frame_count is not None:
            config_data["vad"]["start_frame_count"] = config_message.start_frame_count
        if config_message.end_frame_count is not None:
            config_data["vad"]["end_frame_count"] = config_message.end_frame_count
        if config_message.timeout is not None:
            config_data["timeout"] = config_message.timeout
    
    return ClientConfig(**config_data)


@app.websocket("/vad")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for VAD processing."""
    
    client_id = str(uuid.uuid4())
    client_state = None
    
    try:
        await websocket.accept()
        logger.info(f"Client {client_id}: WebSocket connection accepted")
        
        # Parse query parameters
        query_params = parse_query_params(str(websocket.query_params))
        logger.info(f"Client {client_id}: Query params: {query_params}")
        
        # Create initial configuration
        config = create_client_config(query_params)
        
        # Validate configuration
        try:
            # Validate audio mode support
            if config.audio.mode not in ["pcm", "opus", "aac"]:
                await websocket.send_text(json.dumps({
                    "event": "ERROR",
                    "message": f"Unsupported audio mode: {config.audio.mode}",
                    "timestamp_ms": int(time.time() * 1000)
                }))
                await websocket.close()
                return
            
            # Check decoder availability for Opus/AAC
            if config.audio.mode in ["opus", "aac"]:
                try:
                    import av
                except ImportError:
                    await websocket.send_text(json.dumps({
                        "event": "ERROR",
                        "message": f"PyAV is required for {config.audio.mode} decoding but not installed",
                        "timestamp_ms": int(time.time() * 1000)
                    }))
                    await websocket.close()
                    return
            
            # Validate frame duration produces integer bytes for PCM
            if config.audio.mode == "pcm":
                frame_bytes = (config.audio.sample_rate * 
                             (config.audio.frame_duration_ms / 1000) * 
                             config.audio.channels * 
                             config.audio.sample_width)
                if frame_bytes != int(frame_bytes):
                    await websocket.send_text(json.dumps({
                        "event": "ERROR", 
                        "message": f"Frame duration {config.audio.frame_duration_ms}ms produces non-integer bytes ({frame_bytes})",
                        "timestamp_ms": int(time.time() * 1000)
                    }))
                    await websocket.close()
                    return
                    
        except ValidationError as e:
            await websocket.send_text(json.dumps({
                "event": "ERROR",
                "message": f"Invalid configuration: {e}",
                "timestamp_ms": int(time.time() * 1000)
            }))
            await websocket.close()
            return
        
        # Create client state
        client_state = ClientState(client_id, websocket, config)
        clients[client_id] = client_state
        
        logger.info(f"Client {client_id}: Initialized with config: {config}")
        
        # Send welcome message
        await client_state._send_info("VAD WebSocket server ready")
        
        # Main message loop
        while True:
            try:
                # Receive message
                message = await websocket.receive()
                
                if "bytes" in message:
                    # Binary message - audio data
                    frame_data = message["bytes"]
                    logger.debug(f"Client {client_id}: Received binary frame of {len(frame_data)} bytes")
                    await client_state.process_audio_frame(frame_data)
                    
                elif "text" in message:
                    # Text message - control message
                    try:
                        data = json.loads(message["text"])
                        
                        if data.get("type") == "CONFIG":
                            # Configuration update
                            config_msg = ConfigMessage(**data)
                            new_config = create_client_config(query_params, config_msg)
                            client_state.update_config(new_config)
                            await client_state._send_info("Configuration updated")
                            
                        elif data.get("type") == "HEARTBEAT":
                            # Heartbeat - just acknowledge
                            await client_state._send_info("Heartbeat received")
                            
                        else:
                            await client_state._send_error(f"Unknown message type: {data.get('type')}")
                            
                    except json.JSONDecodeError as e:
                        await client_state._send_error(f"Invalid JSON: {e}")
                    except ValidationError as e:
                        await client_state._send_error(f"Invalid message format: {e}")
                        
            except WebSocketDisconnect:
                logger.info(f"Client {client_id}: WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Client {client_id}: Error in message loop: {e}")
                await client_state._send_error(f"Server error: {e}")
                break
    
    except Exception as e:
        logger.error(f"Client {client_id}: Connection error: {e}")
        
    finally:
        # Cleanup
        if client_state:
            client_state.cleanup()
        
        if client_id in clients:
            del clients[client_id]
            
        logger.info(f"Client {client_id}: Connection closed and cleaned up")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "VAD WebSocket Server",
        "status": "running",
        "connected_clients": len(clients),
        "timestamp": int(time.time() * 1000)
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "connected_clients": len(clients),
        "timestamp": int(time.time() * 1000)
    }


@app.get("/clients")
async def list_clients():
    """List connected clients (for debugging)."""
    client_info = {}
    for client_id, client_state in clients.items():
        client_info[client_id] = {
            "connected_at": client_state.start_time,
            "config": client_state.config.model_dump(),
            "segment_index": client_state.segment_index
        }
    
    return {
        "connected_clients": len(clients),
        "clients": client_info,
        "timestamp": int(time.time() * 1000)
    }


if __name__ == "__main__":
    uvicorn.run(
        "vad_websocket_server:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
