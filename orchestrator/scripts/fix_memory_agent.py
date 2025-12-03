#!/usr/bin/env python3
"""Fix: Create agent_id=1 for chat memory storage"""
import sys
sys.path.insert(0, '/root/automatos-ai/orchestrator')

from database.database import get_db_session
from sqlalchemy import text

with get_db_session() as db:
    # Check if agent 1 exists
    result = db.execute(text("SELECT id FROM agents WHERE id = 1")).fetchone()
    if result:
        print("Agent 1 already exists")
    else:
        # Create agent 1 for chat memory (using actual column names)
        db.execute(text("""
            INSERT INTO agents (id, name, agent_type, status, configuration, created_at, updated_at)
            VALUES (1, 'ChatMemory', 'system', 'active', '{}', NOW(), NOW())
            ON CONFLICT (id) DO NOTHING
        """))
        db.commit()
        print("✅ Created agent_id=1 for chat memory")
    
    # Verify
    count = db.execute(text("SELECT COUNT(*) FROM agents")).fetchone()[0]
    print(f"Total agents: {count}")

