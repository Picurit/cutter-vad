# WebSocket VAD Service

A real-time Voice Activity Detection (VAD) service using WebSocket connections. This service provides a server that processes audio streams and detects voice segments, along with Python and Java clients for easy integration.

## Features

- **Real-time VAD processing** using Silero VAD models
- **Multiple audio formats**: PCM, Opus, AAC support on server-side
- **WebSocket-based communication** for low-latency streaming
- **Multiple client support** with isolated processing states
- **Python and Java clients** with different audio source implementations
- **Configurable VAD parameters** per client connection

## Architecture

```
┌─────────────────┐    WebSocket    ┌─────────────────┐
│                 │ ◄────────────► │                 │
│  Python Client  │                │  VAD Server     │
│                 │                │   (FastAPI)     │
├─────────────────┤                │                 │
│  Audio Sources  │                │                 │
│  - Files        │                │  ┌─────────────┐│
│  - Microphone   │                │  │ Silero VAD  ││
└─────────────────┘                │  │  Processing ││
                                   │  └─────────────┘│
┌─────────────────┐                │                 │
│                 │                │                 │
│   Java Client   │ ◄──────────────┤                 │
│                 │                │                 │
├─────────────────┤                └─────────────────┘
│  Audio Sources  │
│  - Files (WAV)  │
│  - PCM only     │
└─────────────────┘
```

## Quick Start

### 1. Install Dependencies

Make sure you have Python 3.8+ installed. Install the required dependencies:

```bash
# Install the main package with WebSocket dependencies
pip install -e .[websocket]

# Or install dependencies manually
pip install fastapi uvicorn websockets av pydantic
```

### 2. Start the VAD Server

```bash
# Start the server on default port 8000
uvicorn websocket_service.server.vad_websocket_server:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn websocket_service.server.vad_websocket_server:app --host 0.0.0.0 --port 8000 --reload
```

The server will be available at `ws://localhost:8000/vad`

### 3. Test with Python Client

```bash
# Test with sample audio file
cd websocket_service
python examples/test_python_client.py

# Or run the client directly
cd websocket_service/clients/python
python vad_client.py --server-url ws://localhost:8000/vad --mode pcm --audio-file ../../../examples/audios/SampleVoice.wav
```

### 4. Test with Java Client

```bash
# Compile Java client (from websocket_service directory)
javac -cp clients/java examples/TestJavaClient.java clients/java/**/*.java

# Run Java client test
java -cp .:clients/java examples.TestJavaClient
```

## Server Configuration

The VAD server accepts the following WebSocket message format for configuration:

```json
{
  "type": "config",
  "mode": "pcm",
  "sample_rate": 16000,
  "channels": 1,
  "sample_width": 2,
  "frame_duration_ms": 30,
  "start_probability": 0.4,
  "end_probability": 0.3,
  "start_frame_count": 6,
  "end_frame_count": 12
}
```

### Configuration Parameters

- **mode**: Audio encoding format (`pcm`, `opus`, `aac`)
- **sample_rate**: Audio sample rate in Hz (default: 16000)
- **channels**: Number of audio channels (default: 1)
- **sample_width**: Bytes per sample (default: 2 for 16-bit)
- **frame_duration_ms**: Frame duration in milliseconds (default: 30)
- **start_probability**: Probability threshold for voice start (default: 0.4)
- **end_probability**: Probability threshold for voice end (default: 0.3)
- **start_frame_count**: Consecutive frames needed to start voice (default: 6)
- **end_frame_count**: Consecutive frames needed to end voice (default: 12)

## Client Usage

### Python Client

The Python client supports multiple audio sources and provides both stored audio and real-time microphone processing.

#### Basic Usage

```python
import asyncio
from vad_client import VADClient

async def main():
    config = {
        "mode": "pcm",
        "sample_rate": 16000,
        "channels": 1,
        "sample_width": 2,
        "frame_duration_ms": 30
    }
    
    client = VADClient("ws://localhost:8000/vad", config)
    
    # Process audio file
    results = await client.process_stored_audio("audio.wav")
    
    # Print results
    client.print_results(results)

asyncio.run(main())
```

#### Real-time Microphone

```python
# Process microphone input for 30 seconds
results = await client.process_realtime_microphone(duration_seconds=30)
```

#### Command Line Interface

```bash
# Process audio file
python vad_client.py --audio-file sample.wav

# Process microphone input
python vad_client.py --microphone --duration 30

# Custom server and configuration
python vad_client.py --server-url ws://192.168.1.100:8000/vad --mode pcm --sample-rate 8000 --audio-file sample.wav
```

### Java Client

The Java client supports PCM-only mode and uses only standard Java libraries.

#### Basic Usage

```java
// Create event handler
VADEventHandler handler = new VADEventHandler() {
    @Override
    public void onVoiceStart(double timestamp) {
        System.out.println("Voice started at: " + timestamp + "s");
    }
    
    @Override
    public void onVoiceEnd(double timestamp) {
        System.out.println("Voice ended at: " + timestamp + "s");
    }
    
    @Override
    public void onError(String error) {
        System.err.println("Error: " + error);
    }
    
    @Override
    public void onConnectionClosed() {
        System.out.println("Connection closed");
    }
};

// Create and use client
VADClient client = new VADClient("ws://localhost:8000/vad", handler);
boolean success = client.processStoredAudio("sample.wav", null);
client.close();
```

#### Command Line Interface

```bash
# Compile all Java files
javac -cp clients/java clients/java/**/*.java

# Process audio file
java -cp .:clients/java VADClient --server-url ws://localhost:8000/vad --audio-file sample.wav

# Show help
java -cp .:clients/java VADClient --help
```

## WebSocket Protocol

### Client to Server Messages

1. **Configuration Message**
```json
{
  "type": "config",
  "mode": "pcm",
  "sample_rate": 16000,
  "channels": 1,
  "sample_width": 2,
  "frame_duration_ms": 30,
  "start_probability": 0.4,
  "end_probability": 0.3,
  "start_frame_count": 6,
  "end_frame_count": 12
}
```

2. **Audio Data (Binary)**
   - Raw audio bytes in the specified format
   - PCM: Raw 16-bit samples
   - Opus: Opus-encoded frames
   - AAC: AAC-encoded frames

3. **End Processing**
```json
{
  "type": "end"
}
```

### Server to Client Messages

1. **Voice Start Event**
```json
{
  "type": "voice_start",
  "timestamp": 1.234
}
```

2. **Voice End Event**
```json
{
  "type": "voice_end",
  "timestamp": 2.567
}
```

3. **Error Message**
```json
{
  "type": "error",
  "message": "Error description"
}
```

4. **Processing Complete**
```json
{
  "type": "complete"
}
```

## Testing and Validation

The service includes comprehensive testing to ensure correct VAD behavior:

### Expected Test Results

Using the provided `SampleVoice.wav` file, the VAD should detect exactly **4 voice segments**:

1. Segment 1: ~0.5s - ~1.2s
2. Segment 2: ~2.0s - ~3.1s  
3. Segment 3: ~4.2s - ~5.0s
4. Segment 4: ~6.1s - ~7.3s

### Running Tests

```bash
# Test Python client
python websocket_service/examples/test_python_client.py

# Test Java client
java -cp .:websocket_service/clients/java websocket_service.examples.TestJavaClient
```

## Audio Format Support

### Server Side (All Formats)

- **PCM**: Raw 16-bit audio samples
- **Opus**: Opus-encoded audio (requires PyAV)
- **AAC**: AAC-encoded audio (requires PyAV)

### Python Client

- **Input formats**: WAV, MP3, M4A, etc. (using PyAV)
- **Output to server**: PCM, Opus, or AAC

### Java Client (PCM Only)

- **Input formats**: WAV files only
- **Output to server**: PCM only
- **Constraint**: Uses only standard Java libraries

## Performance Considerations

### Frame Duration

- **30ms frames (default)**: Good balance of latency and accuracy
- **10ms frames**: Lower latency, more CPU usage
- **60ms frames**: Higher latency, less CPU usage

### VAD Sensitivity

- **Lower start_probability**: More sensitive, may catch more speech but also noise
- **Higher end_probability**: Voice segments end more quickly
- **Frame counts**: Smooth out brief interruptions

### Audio Quality

- **16 kHz sample rate**: Recommended for speech recognition
- **8 kHz sample rate**: Acceptable for basic VAD, lower bandwidth
- **16-bit samples**: Standard quality, good balance

## Troubleshooting

### Common Issues

1. **"Connection refused"**
   - Ensure the VAD server is running
   - Check the server URL and port

2. **"Audio format not supported"**
   - For Java client, ensure WAV files are 16-bit PCM
   - For Python client, install PyAV: `pip install av`

3. **"No voice detected"**
   - Check audio file has actual speech
   - Adjust VAD sensitivity parameters
   - Verify audio format and sample rate

4. **High CPU usage**
   - Increase frame_duration_ms
   - Reduce concurrent connections
   - Check for audio encoding overhead

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check server logs for connection and processing information.

## Development

### Project Structure

```
websocket_service/
├── server/
│   └── vad_websocket_server.py     # FastAPI WebSocket server
├── clients/
│   ├── python/
│   │   ├── sources/
│   │   │   └── audio_sources.py    # Audio source implementations
│   │   ├── gateway/
│   │   │   └── websocket_gateway.py # WebSocket communication
│   │   └── vad_client.py           # Main Python client
│   └── java/
│       ├── sources/
│       │   ├── AudioSource.java    # Base audio source
│       │   └── StoredAudioSource.java # File audio source
│       ├── gateway/
│       │   ├── VADEventHandler.java # Event handler interface
│       │   └── WebSocketGateway.java # WebSocket client
│       └── VADClient.java          # Main Java client
└── examples/
    ├── test_python_client.py       # Python test example
    └── TestJavaClient.java         # Java test example
```

### Adding New Features

1. **New audio source**: Implement `AudioSource` interface
2. **New audio format**: Add decoder to server
3. **New client language**: Implement WebSocket protocol

## License

This project follows the same license as the parent real-time VAD project.
