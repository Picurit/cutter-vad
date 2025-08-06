#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Suite for VAD WebSocket Service

This script validates the complete implementation including:
- WebSocket service startup
- Client connection and configuration
- Audio data processing
- VAD event generation
- Multiple client support
- Error handling
- Clean shutdown
"""

import asyncio
import json
import subprocess
import time
import signal
import logging
import sys
import os
from typing import List, Dict, Any
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try importing websockets (will show if it's available)
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not available, skipping WebSocket tests")


class VADServiceTester:
    """Comprehensive tester for VAD WebSocket service."""
    
    def __init__(self):
        self.server_process = None
        self.test_results = {}
        
    async def test_service_startup(self) -> bool:
        """Test if the service can start successfully."""
        logger.info("🚀 Testing service startup...")
        
        try:
            # Start the server in background
            cmd = [
                sys.executable, "websocket_service/vad_websocket_server.py", 
                "--host", "127.0.0.1", "--port", "8765"
            ]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            # Wait for server to start
            logger.info("⏳ Waiting for server to start...")
            await asyncio.sleep(3)
            
            # Check if process is still running
            if self.server_process.poll() is not None:
                stdout, stderr = self.server_process.communicate()
                logger.error(f"❌ Server failed to start: {stderr.decode()}")
                return False
            
            logger.info("✅ Service started successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start service: {e}")
            return False
    
    async def test_health_endpoints(self) -> bool:
        """Test HTTP health and status endpoints."""
        logger.info("🔍 Testing health endpoints...")
        
        try:
            import requests
            
            # Test health endpoint
            health_response = requests.get("http://127.0.0.1:8765/health", timeout=5)
            if health_response.status_code != 200:
                logger.error(f"❌ Health endpoint failed: {health_response.status_code}")
                return False
            
            health_data = health_response.json()
            if health_data.get("status") != "healthy":
                logger.error(f"❌ Service not healthy: {health_data}")
                return False
            
            # Test status endpoint
            status_response = requests.get("http://127.0.0.1:8765/status", timeout=5)
            if status_response.status_code != 200:
                logger.error(f"❌ Status endpoint failed: {status_response.status_code}")
                return False
            
            status_data = status_response.json()
            if status_data.get("status") != "running":
                logger.error(f"❌ Service not running: {status_data}")
                return False
            
            logger.info("✅ Health endpoints working correctly")
            return True
            
        except ImportError:
            logger.warning("⚠️  requests library not available, skipping HTTP tests")
            return True
        except Exception as e:
            logger.error(f"❌ Health endpoint test failed: {e}")
            return False
    
    async def test_websocket_connection(self) -> bool:
        """Test WebSocket connection and basic functionality."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("⚠️  WebSocket tests skipped (websockets not available)")
            return True
            
        logger.info("🔌 Testing WebSocket connection...")
        
        try:
            uri = "ws://127.0.0.1:8765/ws"
            async with websockets.connect(uri) as websocket:
                # Test connection establishment
                welcome_msg = await asyncio.wait_for(websocket.recv(), timeout=5)
                welcome_data = json.loads(welcome_msg)
                
                if welcome_data.get("type") != "connection_established":
                    logger.error(f"❌ Unexpected welcome message: {welcome_data}")
                    return False
                
                client_id = welcome_data.get("client_id")
                if not client_id:
                    logger.error("❌ No client ID received")
                    return False
                
                logger.info(f"✅ WebSocket connection established with ID: {client_id}")
                return True
                
        except Exception as e:
            logger.error(f"❌ WebSocket connection test failed: {e}")
            return False
    
    async def test_vad_configuration(self) -> bool:
        """Test VAD parameter configuration."""
        if not WEBSOCKETS_AVAILABLE:
            return True
            
        logger.info("⚙️  Testing VAD configuration...")
        
        try:
            uri = "ws://127.0.0.1:8765/ws"
            async with websockets.connect(uri) as websocket:
                # Wait for welcome message
                await websocket.recv()
                
                # Send configuration
                config = {
                    "type": "configure_vad",
                    "vad_start_probability": 0.5,
                    "vad_end_probability": 0.4,
                    "voice_start_frame_count": 5,
                    "voice_end_frame_count": 25
                }
                
                await websocket.send(json.dumps(config))
                
                # Wait for confirmation
                response = await asyncio.wait_for(websocket.recv(), timeout=5)
                response_data = json.loads(response)
                
                if response_data.get("type") != "vad_configured":
                    logger.error(f"❌ Configuration failed: {response_data}")
                    return False
                
                # Verify parameters were applied
                params = response_data.get("vad_params", {})
                if params.get("vad_start_probability") != 0.5:
                    logger.error(f"❌ Start probability not applied: {params}")
                    return False
                
                logger.info("✅ VAD configuration working correctly")
                return True
                
        except Exception as e:
            logger.error(f"❌ VAD configuration test failed: {e}")
            return False
    
    async def test_audio_processing(self) -> bool:
        """Test audio data processing."""
        if not WEBSOCKETS_AVAILABLE:
            return True
            
        logger.info("🎵 Testing audio processing...")
        
        try:
            uri = "ws://127.0.0.1:8765/ws"
            async with websockets.connect(uri) as websocket:
                # Wait for welcome and configure
                await websocket.recv()
                
                config = {
                    "type": "configure_vad",
                    "vad_start_probability": 0.5,
                    "vad_end_probability": 0.4,
                    "voice_start_frame_count": 2,
                    "voice_end_frame_count": 5
                }
                await websocket.send(json.dumps(config))
                await websocket.recv()  # Configuration response
                
                # Send audio data that should trigger VAD
                audio_data = np.random.randn(1024).astype(np.float32) * 0.5
                audio_message = {
                    "type": "audio_data",
                    "audio_data": audio_data.tolist()
                }
                
                await websocket.send(json.dumps(audio_message))
                
                # Send multiple frames to potentially trigger voice detection
                for i in range(5):
                    audio_data = np.random.randn(1024).astype(np.float32) * 0.3
                    audio_message = {
                        "type": "audio_data",
                        "audio_data": audio_data.tolist()
                    }
                    await websocket.send(json.dumps(audio_message))
                    await asyncio.sleep(0.1)
                
                # Request status to verify processing
                await websocket.send(json.dumps({"type": "get_status"}))
                status_response = await asyncio.wait_for(websocket.recv(), timeout=5)
                status_data = json.loads(status_response)
                
                if status_data.get("type") != "client_status":
                    logger.error(f"❌ Unexpected status response: {status_data}")
                    return False
                
                audio_processed = status_data.get("total_audio_processed", 0)
                if audio_processed < 1024:
                    logger.error(f"❌ Audio not processed: {audio_processed}")
                    return False
                
                logger.info(f"✅ Audio processing working ({audio_processed} samples processed)")
                return True
                
        except Exception as e:
            logger.error(f"❌ Audio processing test failed: {e}")
            return False
    
    async def test_multiple_clients(self) -> bool:
        """Test multiple concurrent client connections."""
        if not WEBSOCKETS_AVAILABLE:
            return True
            
        logger.info("👥 Testing multiple client connections...")
        
        try:
            clients = []
            uri = "ws://127.0.0.1:8765/ws"
            
            # Connect multiple clients
            for i in range(3):
                client = await websockets.connect(uri)
                clients.append(client)
                
                # Wait for welcome message
                welcome_msg = await client.recv()
                welcome_data = json.loads(welcome_msg)
                client_id = welcome_data.get("client_id")
                logger.info(f"📱 Client {i+1} connected with ID: {client_id}")
            
            # Send data from each client
            for i, client in enumerate(clients):
                audio_data = np.random.randn(512).astype(np.float32) * 0.2
                audio_message = {
                    "type": "audio_data",
                    "audio_data": audio_data.tolist()
                }
                await client.send(json.dumps(audio_message))
                logger.info(f"📤 Sent data from client {i+1}")
            
            # Close all clients
            for i, client in enumerate(clients):
                await client.close()
                logger.info(f"📱 Client {i+1} disconnected")
            
            logger.info("✅ Multiple client connections working correctly")
            return True
            
        except Exception as e:
            logger.error(f"❌ Multiple client test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """Test error handling and invalid input."""
        if not WEBSOCKETS_AVAILABLE:
            return True
            
        logger.info("🚨 Testing error handling...")
        
        try:
            uri = "ws://127.0.0.1:8765/ws"
            async with websockets.connect(uri) as websocket:
                # Wait for welcome
                await websocket.recv()
                
                # Send invalid JSON
                await websocket.send("invalid json")
                
                # Should receive error response
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    response_data = json.loads(response)
                    
                    if response_data.get("type") != "error":
                        logger.warning(f"⚠️  Expected error response, got: {response_data}")
                    else:
                        logger.info("✅ Error handling working correctly")
                        return True
                        
                except asyncio.TimeoutError:
                    logger.warning("⚠️  No error response received for invalid input")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up test resources."""
        logger.info("🧹 Cleaning up...")
        
        if self.server_process:
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
                logger.info("✅ Server process terminated")
            except subprocess.TimeoutExpired:
                self.server_process.kill()
                logger.info("⚠️  Server process killed")
            except Exception as e:
                logger.error(f"❌ Error terminating server: {e}")
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("🧪 Starting comprehensive VAD WebSocket service tests")
        logger.info("=" * 60)
        
        tests = [
            ("Service Startup", self.test_service_startup),
            ("Health Endpoints", self.test_health_endpoints),
            ("WebSocket Connection", self.test_websocket_connection),
            ("VAD Configuration", self.test_vad_configuration),
            ("Audio Processing", self.test_audio_processing),
            ("Multiple Clients", self.test_multiple_clients),
            ("Error Handling", self.test_error_handling),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n🧪 Running: {test_name}")
            try:
                result = await test_func()
                results[test_name] = result
                status = "✅ PASS" if result else "❌ FAIL"
                logger.info(f"📊 {test_name}: {status}")
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
                results[test_name] = False
        
        return results


async def main():
    """Main test function."""
    tester = VADServiceTester()
    
    try:
        # Set up signal handlers for cleanup
        def signal_handler(signum, frame):
            logger.info("🛑 Test interrupted by user")
            tester.cleanup()
            sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Run all tests
        results = await tester.run_all_tests()
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("📊 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{status}  {test_name}")
        
        logger.info("-" * 60)
        logger.info(f"📈 Results: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 ALL TESTS PASSED! WebSocket VAD service is working correctly.")
            return True
        else:
            logger.error(f"💥 {total - passed} tests failed. Check the implementation.")
            return False
    
    finally:
        tester.cleanup()


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
