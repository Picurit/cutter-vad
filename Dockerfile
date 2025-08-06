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
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .

# Install additional WebSocket dependencies
RUN pip install --no-cache-dir \
    websockets>=11.0.0 \
    fastapi>=0.100.0 \
    uvicorn>=0.23.0 \
    python-multipart>=0.0.6

# Create non-root user for security
RUN useradd -m -u 1001 vaduser && \
    chown -R vaduser:vaduser /app
USER vaduser

# Expose the WebSocket service port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Default command to run the WebSocket service
CMD ["python", "websocket_service/vad_websocket_server.py", "--host", "0.0.0.0", "--port", "8765"]
