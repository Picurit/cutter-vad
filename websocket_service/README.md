# Real-Time Voice Activity Detection WebSocket Service

This project provides a WebSocket service for real-time voice activity detection using Silero VAD models. It supports multiple concurrent clients and allows configurable VAD parameters per connection.

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Build and run the service:**
```bash
docker-compose up --build
```

2. **Access the service:**
- WebSocket endpoint: `ws://localhost:8765/ws`
- Health check: `http://localhost:8765/health`
- Status dashboard: `http://localhost:8765/`

### Manual Installation

1. **Install dependencies:**
```bash
pip install -e .
pip install websockets fastapi uvicorn python-multipart
```

2. **Run the service:**
```bash
python websocket_service/vad_websocket_server.py
```

## 📋 Features

- **Multiple concurrent clients**: Supports many simultaneous WebSocket connections
- **Configurable VAD parameters**: Each client can set custom sensitivity settings
- **Real-time audio processing**: Processes audio streams with minimal latency
- **Event notifications**: Sends voice start, continue, and end events
- **Session management**: Tracks client sessions with unique IDs
- **Health monitoring**: Built-in health checks and status endpoints
- **Docker support**: Easy deployment with containerization

## 🔌 Client Examples

### Python Client

```bash
cd websocket_service/clients/python
python vad_websocket_client.py --server ws://localhost:8765/ws
```

### Java Client

```bash
cd websocket_service/clients/java
mvn clean package
java -jar target/vad-websocket-client-1.0.0-jar-with-dependencies.jar ws://localhost:8765/ws
```

## 📡 WebSocket API

### Connection
Connect to: `ws://your-server:8765/ws`

### Message Format
All messages are JSON format:

#### Configure VAD Parameters
```json
{
  "type": "configure_vad",
  "vad_start_probability": 0.7,
  "vad_end_probability": 0.7,
  "voice_start_frame_count": 10,
  "voice_end_frame_count": 50,
  "sample_rate": 16000,
  "chunk_size": 1024
}
```

#### Send Audio Data
```json
{
  "type": "audio_data",
  "audio_data": [0.1, 0.2, -0.1, 0.3, ...]
}
```

#### Ping/Pong
```json
{
  "type": "ping",
  "timestamp": "2025-01-01T12:00:00"
}
```

#### Request Status
```json
{
  "type": "get_status"
}
```

### Server Responses

#### Connection Established
```json
{
  "type": "connection_established",
  "client_id": "uuid-string",
  "default_vad_params": {...},
  "message": "Connected to VAD WebSocket Service"
}
```

#### VAD Events
```json
{
  "type": "vad_event",
  "event_type": "voice_start|voice_continue|voice_end",
  "client_id": "uuid-string",
  "timestamp": "2025-01-01T12:00:00",
  "audio_length": 1024,
  "wav_data_base64": "..." // Only for voice_end events
}
```

#### Configuration Confirmation
```json
{
  "type": "vad_configured",
  "vad_params": {...},
  "message": "VAD parameters updated successfully"
}
```

#### Error Messages
```json
{
  "type": "error",
  "message": "Error description"
}
```

## ⚙️ Configuration

### VAD Parameters

- **vad_start_probability** (0.0-1.0): Threshold for detecting voice start
- **vad_end_probability** (0.0-1.0): Threshold for detecting voice end
- **voice_start_frame_count** (≥1): Frames needed to confirm voice start
- **voice_end_frame_count** (≥1): Frames needed to confirm voice end
- **sample_rate**: Audio sample rate (8000, 16000, 24000, 48000 Hz)
- **chunk_size**: Audio chunk size in samples

### Server Configuration

Edit `websocket_service/config/service_config.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8765
  log_level: "INFO"

vad_defaults:
  sample_rate: 16000
  vad_start_probability: 0.7
  vad_end_probability: 0.7
  # ... other defaults
```

## 🐳 Docker Deployment

### Basic Deployment
```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f vad-websocket-service

# Stop service
docker-compose down
```

### With SSL Proxy
```bash
# Run with nginx SSL proxy
docker-compose --profile with-proxy up --build
```

### Environment Variables
```bash
# Set in docker-compose.yml or .env file
LOG_LEVEL=DEBUG
VAD_MODEL_VERSION=v5
MAX_CONNECTIONS=50
```

## 🔍 Monitoring

### Health Check
```bash
curl http://localhost:8765/health
```

### Service Status
```bash
curl http://localhost:8765/status
```

### Logs
```bash
# Docker logs
docker-compose logs -f vad-websocket-service

# Direct logs (manual installation)
python websocket_service/vad_websocket_server.py --log-level DEBUG
```

## 🧪 Testing

### Test WebSocket Connection
```python
import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://localhost:8765/ws"
    async with websockets.connect(uri) as websocket:
        # Send ping
        await websocket.send(json.dumps({"type": "ping"}))
        
        # Receive response
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_connection())
```

### Load Testing
```bash
# Install dependencies
pip install websockets asyncio

# Run load test script
python tests/load_test.py --clients 10 --duration 60
```

## 🔧 Troubleshooting

### Common Issues

1. **Connection refused**
   - Check if service is running: `curl http://localhost:8765/health`
   - Verify port availability: `netstat -an | grep 8765`

2. **Audio not processing**
   - Check VAD model files in `src/real_time_vad/models/`
   - Verify audio format (float32, mono, correct sample rate)

3. **High CPU usage**
   - Reduce client connection count
   - Adjust chunk size and processing intervals
   - Check for memory leaks in long-running sessions

4. **WebSocket disconnections**
   - Implement client-side reconnection logic
   - Check network stability
   - Monitor server logs for errors

### Debug Mode
```bash
# Enable debug logging
python websocket_service/vad_websocket_server.py --log-level DEBUG

# Or with Docker
docker-compose up --build -e LOG_LEVEL=DEBUG
```

## 📊 Performance

### Typical Performance Metrics
- **Processing latency**: < 50ms per chunk
- **Memory usage**: ~100MB base + ~10MB per client
- **CPU usage**: ~5-10% per active client
- **Concurrent clients**: Up to 100+ (depends on hardware)

### Optimization Tips
- Use smaller chunk sizes for lower latency
- Increase buffer sizes for better throughput
- Deploy on multi-core systems for better concurrency
- Use Redis or database for session persistence in production

## 🛡️ Security Considerations

- Enable CORS restrictions in production
- Implement authentication/authorization
- Use WSS (WebSocket Secure) for encrypted connections
- Set rate limiting to prevent abuse
- Validate all client input
- Monitor for suspicious activity

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
