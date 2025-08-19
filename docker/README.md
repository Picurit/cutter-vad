# VAD WebSocket Server - Docker Deployment

This directory contains Docker deployment files for the VAD WebSocket Server, providing a production-ready containerized environment.

## Quick Start

### Prerequisites

- Docker Engine 20.10+ 
- Docker Compose 2.0+

### Basic Deployment

1. **Build and start the server:**
   ```bash
   # Linux/macOS
   ./deploy.sh build
   ./deploy.sh up
   
   # Windows
   deploy.bat build
   deploy.bat up
   ```

2. **Verify the deployment:**
   ```bash
   # Check server health
   curl http://localhost:8000/health
   
   # Run integration test
   ./deploy.sh test  # Linux/macOS
   deploy.bat test   # Windows
   ```

3. **Test WebSocket connection:**
   ```
   WebSocket URL: ws://localhost:8000/vad
   Health Check: http://localhost:8000/health
   API Docs: http://localhost:8000/docs
   ```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key configuration options:

- `VAD_PORT`: Server port (default: 8000)
- `LOG_LEVEL`: Logging level (debug, info, warning, error)
- `VAD_DEFAULT_*`: Default VAD parameters
- Performance tuning: `TORCH_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`

### WebSocket Configuration

The server accepts configuration via URL parameters or WebSocket messages:

```
ws://localhost:8000/vad?mode=pcm&sample_rate=16000&channels=1&frame_duration_ms=30&start_probability=0.4&end_probability=0.3&start_frame_count=6&end_frame_count=12&timeout=2.0
```

## Deployment Profiles

### Default Profile
Runs only the VAD WebSocket server:
```bash
docker-compose up -d
```

### Production Profile
Includes Nginx reverse proxy with rate limiting and security headers:
```bash
docker-compose --profile production up -d
```

## Deployment Scripts

### Linux/macOS: `deploy.sh`
```bash
./deploy.sh {build|up|down|restart|logs|test|clean} [profile]
```

### Windows: `deploy.bat`
```batch
deploy.bat {build|up|down|restart|logs|test|clean} [profile]
```

### Available Commands

- **build**: Build Docker images
- **up**: Start the VAD WebSocket Server
- **down**: Stop the VAD WebSocket Server  
- **restart**: Restart the VAD WebSocket Server
- **logs**: Show server logs
- **test**: Run integration test with example audio
- **clean**: Clean up Docker resources

### Examples
```bash
# Basic deployment
./deploy.sh build
./deploy.sh up

# Production deployment with Nginx
./deploy.sh up production

# View logs
./deploy.sh logs

# Run integration test
./deploy.sh test

# Clean up
./deploy.sh clean
```

## Production Considerations

### Resource Limits
The Docker Compose file includes resource limits:
- CPU: 2.0 cores limit, 0.5 cores reserved
- Memory: 2GB limit, 512MB reserved

### Security Features
- Non-root user execution
- Security headers via Nginx
- Rate limiting (10 requests/second with burst of 20)
- Network isolation

### Monitoring
- Health check endpoint: `/health`
- Prometheus metrics (if enabled)
- Container health checks

### Persistence
- Model files: `./models` volume (read-only)
- Logs: `./logs` volume

## Testing

### Integration Test
The deployment includes an integration test using the example audio file:

```bash
./deploy.sh test
```

This test:
1. Verifies server is running
2. Connects to WebSocket endpoint
3. Streams `examples/audios/SampleVoiceMono.wav`
4. Validates that exactly 4 voice segments are detected
5. Reports success/failure

### Manual Testing
```bash
# Check server status
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs

# Test WebSocket connection (using websocat)
echo "Hello" | websocat ws://localhost:8000/vad?mode=pcm&sample_rate=16000
```

## Troubleshooting

### Common Issues

1. **Port already in use:**
   ```bash
   # Change port in .env file
   VAD_PORT=8001
   ```

2. **Permission denied:**
   ```bash
   # Make scripts executable
   chmod +x deploy.sh
   ```

3. **Docker build fails:**
   ```bash
   # Clean and rebuild
   ./deploy.sh clean
   ./deploy.sh build
   ```

4. **Integration test fails:**
   ```bash
   # Check logs
   ./deploy.sh logs
   
   # Verify audio file exists
   ls -la ../examples/audios/SampleVoiceMono.wav
   ```

### Logs and Debugging

```bash
# View real-time logs
./deploy.sh logs

# Debug mode deployment
LOG_LEVEL=debug ./deploy.sh up

# Container shell access
docker exec -it vad-websocket-server /bin/sh
```

## File Structure

```
docker/
├── Dockerfile              # Multi-stage production Docker image
├── docker-compose.yml      # Docker Compose configuration
├── .env.example           # Environment variables template
├── requirements.txt       # Python dependencies
├── deploy.sh             # Linux/macOS deployment script
├── deploy.bat            # Windows deployment script
├── README.md             # This file
└── nginx/
    ├── nginx.conf        # Nginx reverse proxy configuration
    └── ssl/              # SSL certificates (production)
```

## Performance Optimization

- Multi-stage Docker build for smaller images
- Alpine Linux base for minimal footprint  
- Thread pool configuration for CPU-intensive operations
- Resource limits to prevent resource exhaustion
- Connection pooling and rate limiting

## Security

- Non-root container execution
- Read-only model files
- Network isolation
- Security headers
- Rate limiting
- No secrets in environment variables
