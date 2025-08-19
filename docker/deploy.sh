#!/bin/bash

# Docker deployment script for VAD WebSocket Server
# This script builds and deploys the VAD WebSocket server using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$DOCKER_DIR")"
ENV_FILE="$DOCKER_DIR/.env"

echo -e "${BLUE}🐳 VAD WebSocket Server - Docker Deployment${NC}"
echo "=================================================="

# Check if Docker and Docker Compose are installed
echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed or not in PATH${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed or not in PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose are available${NC}"

# Check if .env file exists, create from example if not
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}📄 Creating .env file from example...${NC}"
    cp "$DOCKER_DIR/.env.example" "$ENV_FILE"
    echo -e "${GREEN}✅ .env file created. Edit it if needed before deployment.${NC}"
fi

# Parse command line arguments
ACTION="${1:-build}"
PROFILE="${2:-default}"

case "$ACTION" in
    build)
        echo -e "${YELLOW}🔨 Building Docker images...${NC}"
        cd "$DOCKER_DIR"
        
        if [ "$PROFILE" == "production" ]; then
            docker-compose --profile production build --no-cache
        else
            docker-compose build --no-cache
        fi
        
        echo -e "${GREEN}✅ Docker images built successfully${NC}"
        ;;
        
    up)
        echo -e "${YELLOW}🚀 Starting VAD WebSocket Server...${NC}"
        cd "$DOCKER_DIR"
        
        if [ "$PROFILE" == "production" ]; then
            docker-compose --profile production up -d
        else
            docker-compose up -d
        fi
        
        echo -e "${GREEN}✅ VAD WebSocket Server is running${NC}"
        echo
        echo "📋 Service Information:"
        echo "  - WebSocket URL: ws://localhost:8000/vad"
        echo "  - Health Check: http://localhost:8000/health"
        echo "  - API Docs: http://localhost:8000/docs"
        
        if [ "$PROFILE" == "production" ]; then
            echo "  - Nginx Proxy: http://localhost:80"
        fi
        ;;
        
    down)
        echo -e "${YELLOW}🛑 Stopping VAD WebSocket Server...${NC}"
        cd "$DOCKER_DIR"
        
        if [ "$PROFILE" == "production" ]; then
            docker-compose --profile production down
        else
            docker-compose down
        fi
        
        echo -e "${GREEN}✅ VAD WebSocket Server stopped${NC}"
        ;;
        
    restart)
        echo -e "${YELLOW}🔄 Restarting VAD WebSocket Server...${NC}"
        cd "$DOCKER_DIR"
        
        if [ "$PROFILE" == "production" ]; then
            docker-compose --profile production down
            docker-compose --profile production up -d
        else
            docker-compose down
            docker-compose up -d
        fi
        
        echo -e "${GREEN}✅ VAD WebSocket Server restarted${NC}"
        ;;
        
    logs)
        echo -e "${YELLOW}📋 Showing logs...${NC}"
        cd "$DOCKER_DIR"
        docker-compose logs -f vad-websocket-server
        ;;
        
    test)
        echo -e "${YELLOW}🧪 Running integration test...${NC}"
        
        # Check if server is running
        if ! curl -f http://localhost:8000/health > /dev/null 2>&1; then
            echo -e "${RED}❌ VAD WebSocket Server is not running. Start it first with: $0 up${NC}"
            exit 1
        fi
        
        # Run the Python test
        cd "$PROJECT_ROOT"
        python examples/test_python_vad_client.py
        ;;
        
    clean)
        echo -e "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
        cd "$DOCKER_DIR"
        
        docker-compose down -v --remove-orphans
        docker system prune -f
        
        echo -e "${GREEN}✅ Docker resources cleaned up${NC}"
        ;;
        
    *)
        echo "Usage: $0 {build|up|down|restart|logs|test|clean} [profile]"
        echo
        echo "Commands:"
        echo "  build      - Build Docker images"
        echo "  up         - Start the VAD WebSocket Server"
        echo "  down       - Stop the VAD WebSocket Server"
        echo "  restart    - Restart the VAD WebSocket Server"
        echo "  logs       - Show server logs"
        echo "  test       - Run integration test"
        echo "  clean      - Clean up Docker resources"
        echo
        echo "Profiles:"
        echo "  default    - Run only VAD server (default)"
        echo "  production - Run VAD server + Nginx reverse proxy"
        echo
        echo "Examples:"
        echo "  $0 build                    # Build images"
        echo "  $0 up                       # Start with default profile"
        echo "  $0 up production            # Start with production profile"
        echo "  $0 test                     # Run integration test"
        exit 1
        ;;
esac
