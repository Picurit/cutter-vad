@echo off
REM Docker deployment script for VAD WebSocket Server (Windows batch version)

setlocal enabledelayedexpansion

echo 🐳 VAD WebSocket Server - Docker Deployment
echo ==================================================

REM Check if Docker and Docker Compose are installed
echo 🔍 Checking prerequisites...

docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not in PATH
    exit /b 1
)

docker-compose --version >nul 2>&1 || docker compose version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed or not in PATH
    exit /b 1
)

echo ✅ Docker and Docker Compose are available

REM Get script directory and project root
set DOCKER_DIR=%~dp0
set PROJECT_ROOT=%DOCKER_DIR%..
set ENV_FILE=%DOCKER_DIR%.env

REM Check if .env file exists
if not exist "%ENV_FILE%" (
    echo 📄 Creating .env file from example...
    copy "%DOCKER_DIR%.env.example" "%ENV_FILE%" >nul
    echo ✅ .env file created. Edit it if needed before deployment.
)

REM Parse command line arguments
set ACTION=%1
set PROFILE=%2

if "%ACTION%"=="" set ACTION=build
if "%PROFILE%"=="" set PROFILE=default

cd /d "%DOCKER_DIR%"

if "%ACTION%"=="build" (
    echo 🔨 Building Docker images...
    
    if "%PROFILE%"=="production" (
        docker-compose --profile production build --no-cache
    ) else (
        docker-compose build --no-cache
    )
    
    echo ✅ Docker images built successfully
    
) else if "%ACTION%"=="up" (
    echo 🚀 Starting VAD WebSocket Server...
    
    if "%PROFILE%"=="production" (
        docker-compose --profile production up -d
    ) else (
        docker-compose up -d
    )
    
    echo ✅ VAD WebSocket Server is running
    echo.
    echo 📋 Service Information:
    echo   - WebSocket URL: ws://localhost:8000/vad
    echo   - Health Check: http://localhost:8000/health
    echo   - API Docs: http://localhost:8000/docs
    
    if "%PROFILE%"=="production" (
        echo   - Nginx Proxy: http://localhost:80
    )
    
) else if "%ACTION%"=="down" (
    echo 🛑 Stopping VAD WebSocket Server...
    
    if "%PROFILE%"=="production" (
        docker-compose --profile production down
    ) else (
        docker-compose down
    )
    
    echo ✅ VAD WebSocket Server stopped
    
) else if "%ACTION%"=="restart" (
    echo 🔄 Restarting VAD WebSocket Server...
    
    if "%PROFILE%"=="production" (
        docker-compose --profile production down
        docker-compose --profile production up -d
    ) else (
        docker-compose down
        docker-compose up -d
    )
    
    echo ✅ VAD WebSocket Server restarted
    
) else if "%ACTION%"=="logs" (
    echo 📋 Showing logs...
    docker-compose logs -f vad-websocket-server
    
) else if "%ACTION%"=="test" (
    echo 🧪 Running integration test...
    
    REM Check if server is running
    curl -f http://localhost:8000/health >nul 2>&1
    if errorlevel 1 (
        echo ❌ VAD WebSocket Server is not running. Start it first with: %0 up
        exit /b 1
    )
    
    REM Run the Python test
    cd /d "%PROJECT_ROOT%"
    python examples\test_python_vad_client.py
    
) else if "%ACTION%"=="clean" (
    echo 🧹 Cleaning up Docker resources...
    
    docker-compose down -v --remove-orphans
    docker system prune -f
    
    echo ✅ Docker resources cleaned up
    
) else (
    echo Usage: %0 {build^|up^|down^|restart^|logs^|test^|clean} [profile]
    echo.
    echo Commands:
    echo   build      - Build Docker images
    echo   up         - Start the VAD WebSocket Server
    echo   down       - Stop the VAD WebSocket Server
    echo   restart    - Restart the VAD WebSocket Server
    echo   logs       - Show server logs
    echo   test       - Run integration test
    echo   clean      - Clean up Docker resources
    echo.
    echo Profiles:
    echo   default    - Run only VAD server ^(default^)
    echo   production - Run VAD server + Nginx reverse proxy
    echo.
    echo Examples:
    echo   %0 build                    # Build images
    echo   %0 up                       # Start with default profile
    echo   %0 up production            # Start with production profile
    echo   %0 test                     # Run integration test
)
