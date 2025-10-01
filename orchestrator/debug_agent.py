#!/usr/bin/env python3
"""
Debug agent creation with tools
"""
import sys
import os
sys.path.insert(0, '/root/automatos-ai/orchestrator')

from database.database import get_db
from database.models import AgentCreate, AgentType
from api.agents import create_agent
from sqlalchemy.orm import Session

async def debug_agent_creation():
    """Debug agent creation with tools"""
    print("🔄 Testing agent creation with tools...")
    
    # Get database session
    db_gen = get_db()
    db = next(db_gen)
    
    try:
        # Create agent data
        agent_data = AgentCreate(
            name="Debug Agent Test",
            description="Debug agent creation",
            agent_type=AgentType.CUSTOM,
            tool_ids=[2]
        )
        
        print(f"📝 Agent data: {agent_data}")
        print(f"🔧 Tool IDs: {agent_data.tool_ids}")
        
        # Call the create_agent function
        result = await create_agent(agent_data, db)
        
        print(f"✅ Agent created: {result}")
        print(f"🛠️ Tools in response: {result.tools}")
        
        # Check database
        from database.models import AgentToolAssignment
        assignments = db.query(AgentToolAssignment).filter(
            AgentToolAssignment.agent_id == result.id
        ).all()
        
        print(f"📊 Tool assignments in DB: {len(assignments)}")
        for assignment in assignments:
            print(f"   - Agent {assignment.agent_id} -> Tool {assignment.tool_id}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(debug_agent_creation())

