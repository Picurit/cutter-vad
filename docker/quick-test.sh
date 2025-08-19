#!/bin/bash
# Quick build and test script for Docker setup

echo "🚀 Building optimized Docker image..."
time docker compose -f docker/docker-compose.yml build --no-cache

echo "🔄 Starting services..."
docker compose -f docker/docker-compose.yml up -d

echo "⏳ Waiting for service to be ready..."
sleep 10

echo "🩺 Checking health..."
docker compose -f docker/docker-compose.yml ps

echo "📋 Checking logs..."
docker compose -f docker/docker-compose.yml logs --tail=20

echo "✅ Ready to test!"
echo "Run: cd examples && python test_python_vad_client.py"
