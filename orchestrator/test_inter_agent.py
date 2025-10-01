#!/usr/bin/env python3
"""
Test inter-agent communication initialization
"""
import os
import asyncio
import sys
from pathlib import Path

# Add the orchestrator directory to Python path
sys.path.insert(0, '/root/automatos-ai/orchestrator')

from dotenv import load_dotenv
load_dotenv()

async def test_inter_agent_communication():
    """Test inter-agent communication initialization"""
    try:
        print("🔄 Testing inter-agent communication initialization...")
        
        # Import the communication components
        from services.inter_agent_communication import (
            AgentCommunicationProtocol,
            SharedContextManager,
            MessageType
        )
        print("✅ Import successful")
        
        # Initialize communication protocol
        communication = AgentCommunicationProtocol()
        print("✅ AgentCommunicationProtocol created")
        
        # Test Redis connection
        await communication.connect()
        print("✅ Redis connection established")
        
        # Test sending a message
        from services.inter_agent_communication import MessageResult
        result = await communication.send_message(
            from_agent=1,
            to_agent=2,
            message_type=MessageType.KNOWLEDGE_SHARE,
            content="Test message",
            priority=5,
            require_ack=False
        )
        print(f"✅ Message sent successfully: {result}")
        
        # Clean up
        if communication.redis_client:
            await communication.redis_client.aclose()
        
        print("🎉 Inter-agent communication is working!")
        return True
        
    except Exception as e:
        print(f"❌ Inter-agent communication test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_inter_agent_communication())
    if success:
        print("✅ Inter-agent communication is ready!")
    else:
        print("💥 Inter-agent communication failed")

