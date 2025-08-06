# Use Python 3.9 slim image as base
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-dev \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Ensure model files are present and have correct permissions
RUN ls -la src/real_time_vad/models/ || echo "Models directory not found"
RUN test -f src/real_time_vad/models/silero_vad_v5.onnx || echo "silero_vad_v5.onnx not found"

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Install additional WebSocket dependencies
RUN pip install --no-cache-dir \
    websockets>=11.0.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    python-multipart>=0.0.6

# Verify model files are present
RUN python websocket_service/check_models.py

# Create non-root user for security
RUN useradd -m -u 1001 vaduser && \
    chown -R vaduser:vaduser /app
USER vaduser

# Expose the WebSocket service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the WebSocket service
CMD ["python", "websocket_service/vad_websocket_server.py", "--host", "0.0.0.0", "--port", "8000"]
