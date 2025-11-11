#!/usr/bin/env python3
"""
Fix Agent Context Limits - Apply PRD-22 context window fixes directly to database
"""
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import SessionLocal
from database.models import Agent
from sqlalchemy import text

def fix_agent_context_limits():
    """Update agents with larger context windows for PRD-22 skill support"""
    
    print("🔧 Applying agent context limit fixes...")
    print()
    
    session = SessionLocal()
    
    try:
        # New LLM config with gpt-4o
        new_llm_config = {
            "provider": "openai",
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 4000,
            "context_window": 128000
        }
        
        agents_updated = []
        
        # Get agents to update
        agents_to_fix = session.query(Agent).filter(
            Agent.id.in_([96, 102])
        ).all()
        
        if not agents_to_fix:
            print("⚠️  Agents 96 and 102 not found in database")
            print("   This may be expected if your database has different IDs")
            print()
            
            # Try to find by name instead
            agents_to_fix = session.query(Agent).filter(
                Agent.name.in_(['Technical Writer Pro', 'Document Generation'])
            ).all()
            
            if agents_to_fix:
                print(f"✅ Found agents by name: {[a.name for a in agents_to_fix]}")
            else:
                print("❌ Could not find Technical Writer Pro or Document Generation agents")
                return
        
        for agent in agents_to_fix:
            # Get current configuration
            current_config = agent.configuration or {}
            current_llm = current_config.get('llm_config', {})
            
            print(f"📝 Updating {agent.name} (ID: {agent.id})")
            print(f"   Current model: {current_llm.get('model', 'not set')}")
            print(f"   Current context: {current_llm.get('context_window', 'not set')}")
            
            # Update llm_config
            current_config['llm_config'] = new_llm_config
            agent.configuration = current_config
            
            print(f"   ✅ Updated to: gpt-4o (128K context)")
            print()
            
            agents_updated.append(agent.name)
        
        # Also update any other agents still using old gpt-4
        print("🔍 Checking for other agents using old gpt-4...")
        all_agents = session.query(Agent).all()
        
        for agent in all_agents:
            if agent.id in [a.id for a in agents_to_fix]:
                continue  # Already updated
            
            config = agent.configuration or {}
            llm_config = config.get('llm_config', {})
            
            if (llm_config.get('model') == 'gpt-4' and 
                llm_config.get('context_window') == 8192):
                
                print(f"📝 Updating {agent.name} (ID: {agent.id})")
                
                # Update to gpt-4o
                llm_config['model'] = 'gpt-4o'
                llm_config['context_window'] = 128000
                llm_config['max_tokens'] = 4000
                config['llm_config'] = llm_config
                agent.configuration = config
                
                print(f"   ✅ Updated to: gpt-4o (128K context)")
                agents_updated.append(agent.name)
        
        # Commit changes
        session.commit()
        
        print()
        print(f"✅ Successfully updated {len(agents_updated)} agent(s):")
        for name in agents_updated:
            print(f"   - {name}")
        
        print()
        print("🎯 Changes summary:")
        print("   - DEFAULT_LLM_CONFIG: gpt-4 → gpt-4o (in code)")
        print("   - Intelligent skill selection: Enabled (in code)")
        print(f"   - Database agents updated: {len(agents_updated)}")
        print()
        print("📋 Next steps:")
        print("   1. Restart the orchestrator if it's running")
        print("   2. Test your report writing workflow")
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    fix_agent_context_limits()

