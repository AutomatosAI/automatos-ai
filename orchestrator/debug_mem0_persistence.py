
import asyncio
import sys
import os

# Add orchestrator to path
sys.path.append(os.path.join(os.getcwd(), 'orchestrator'))

from modules.memory.storage.mem0_system import Mem0MemorySystem

async def check_mem0():
    print("Initializing Mem0 System...")
    memory_system = Mem0MemorySystem()

    workspace_id = "ae8320bc-95e1-4de1-bbe9-396bef19cbf8"
    agent_id = 19

    user_id = memory_system._get_scoped_user_id(workspace_id=workspace_id, agent_id=agent_id)
    print(f"\nChecking memories for User ID: {user_id}")
    
    print("\n--- Retrieving All Memories ---")
    try:
        # Search with a generic query to get recent items
        memories = memory_system.client.search(query="Gerard", user_id=user_id, limit=10)
        
        if not memories:
            print("❌ No memories found for this user!")
            
            # Try searching just by workspace default
            user_id_default = memory_system._get_scoped_user_id(workspace_id=workspace_id, agent_id=None)
            print(f"\nChecking fallback User ID: {user_id_default}")
            memories_default = memory_system.client.search(query="Gerard", user_id=user_id_default, limit=10)
            if memories_default:
                print(f"⚠️ Found {len(memories_default)} memories in default scope!")
                for mem in memories_default:
                    print(f"- {mem}")
            else:
                print("❌ No memories found in fallback scope either.")
                
        else:
            print(f"✅ Found {len(memories)} memories:")
            for mem in memories:
                print(f"\nID: {mem.get('id')}")
                print(f"Memory: {mem.get('memory')}")
                print(f"Score: {mem.get('score')}")
                print(f"Metadata: {mem.get('metadata')}")
                print(f"Created: {mem.get('created_at')}")
                
    except Exception as e:
        print(f"❌ Error during search: {e}")

    # Test Add
    print("\n--- Testing Memory Addition ---")
    try:
        # We won't add unless we find nothing, to avoid polluting if it's just a retrieval issue
        pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(check_mem0())
