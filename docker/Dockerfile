# Multi-stage build using PyTorch base image for efficiency
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS builder

# Install minimal build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /build

# Create virtual environment (PyTorch already available in base)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install dependencies (excluding PyTorch which is pre-installed)
COPY docker/requirements.txt .
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Install dependencies excluding PyTorch (already available in base image)
RUN pip install --no-cache-dir -r requirements.txt

# Copy source and install the package
COPY src/ ./src/
RUN pip install --no-cache-dir --no-deps .

# Production stage - use same PyTorch base for consistency
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime AS runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY websocket_service/ ./websocket_service/

# Change ownership to non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV UVICORN_HOST=0.0.0.0
ENV UVICORN_PORT=8000
ENV UVICORN_LOG_LEVEL=info

# Default command
CMD ["uvicorn", "websocket_service.server.vad_websocket_server:app", "--host", "0.0.0.0", "--port", "8000"]
