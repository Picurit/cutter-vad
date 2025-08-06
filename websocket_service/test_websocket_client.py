#!/usr/bin/env python3
"""
Simple WebSocket Client Test for VAD Service

This script tests the WebSocket connection and basic functionality
without requiring microphone access.
"""

import asyncio
import json
import websockets
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_websocket_connection():
    """Test basic WebSocket connection and functionality."""
    uri = "ws://127.0.0.1:8765/ws"
    
    try:
        logger.info(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            logger.info("✅ Connected successfully!")
            
            # Test 1: Wait for welcome message
            welcome_msg = await websocket.recv()
            welcome_data = json.loads(welcome_msg)
            logger.info(f"📨 Welcome message: {welcome_data}")
            
            client_id = welcome_data.get("client_id")
            logger.info(f"🆔 Client ID: {client_id}")
            
            # Test 2: Configure VAD parameters
            vad_config = {
                "type": "configure_vad",
                "vad_start_probability": 0.5,
                "vad_end_probability": 0.4,
                "voice_start_frame_count": 5,
                "voice_end_frame_count": 25,
                "sample_rate": 16000,
                "chunk_size": 1024
            }
            
            logger.info("⚙️  Configuring VAD parameters...")
            await websocket.send(json.dumps(vad_config))
            
            # Wait for configuration response
            config_response = await websocket.recv()
            config_data = json.loads(config_response)
            logger.info(f"📨 VAD configured: {config_data}")
            
            # Test 3: Send test audio data (simulated voice)
            logger.info("🎵 Sending test audio data...")
            
            # Generate test audio that should trigger voice detection
            for i in range(10):
                # Generate audio samples that simulate voice activity
                if i >= 3 and i <= 7:  # Voice activity in the middle
                    audio_data = np.random.randn(1024).astype(np.float32) * 0.3 + 0.2
                else:
                    audio_data = np.random.randn(1024).astype(np.float32) * 0.01
                
                audio_message = {
                    "type": "audio_data",
                    "audio_data": audio_data.tolist()
                }
                
                await websocket.send(json.dumps(audio_message))
                logger.info(f"📊 Sent audio chunk {i+1}/10")
                
                # Check for any VAD events
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    event_data = json.loads(response)
                    if event_data.get("type") == "vad_event":
                        event_type = event_data.get("event_type")
                        logger.info(f"🎙️  VAD Event: {event_type}")
                except asyncio.TimeoutError:
                    pass  # No immediate response
                
                await asyncio.sleep(0.1)
            
            # Test 4: Send ping
            logger.info("🏓 Sending ping...")
            ping_message = {
                "type": "ping",
                "timestamp": "2025-01-01T12:00:00"
            }
            await websocket.send(json.dumps(ping_message))
            
            # Wait for pong
            pong_response = await websocket.recv()
            pong_data = json.loads(pong_response)
            logger.info(f"📨 Pong received: {pong_data}")
            
            # Test 5: Request status
            logger.info("📊 Requesting status...")
            status_request = {"type": "get_status"}
            await websocket.send(json.dumps(status_request))
            
            # Wait for status response
            status_response = await websocket.recv()
            status_data = json.loads(status_response)
            logger.info(f"📨 Status: {status_data}")
            
            logger.info("✅ All tests completed successfully!")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
    
    return True


async def main():
    """Main test function."""
    logger.info("🧪 Starting WebSocket VAD Service Test")
    logger.info("=" * 50)
    
    success = await test_websocket_connection()
    
    if success:
        logger.info("🎉 All tests passed! WebSocket VAD service is working correctly.")
    else:
        logger.error("💥 Tests failed! Check the server and try again.")
    
    return success


if __name__ == "__main__":
    result = asyncio.run(main())
    exit(0 if result else 1)
