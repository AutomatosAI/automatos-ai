#!/usr/bin/env python3
"""
Update Tool Metadata
===================
Updates existing MCP tools in the database with credential_type from seed file.
This ensures all tools have proper credential_type mappings.
"""

import json
import sys
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def update_tool_metadata():
    """Update tool metadata with credential_type from seed file"""
    
    print("🔄 UPDATING TOOL METADATA")
    print("=" * 60)
    
    try:
        # Get database config - support both DATABASE_URL and individual vars
        database_url = os.getenv('DATABASE_URL')
        
        if database_url:
            # Parse DATABASE_URL: postgresql://user:password@host:port/dbname
            from urllib.parse import urlparse
            parsed = urlparse(database_url)
            db_name = parsed.path.lstrip('/')
            db_user = parsed.username
            db_password = parsed.password
            db_host = parsed.hostname
            db_port = parsed.port or '5432'
        else:
            db_name = os.getenv('POSTGRES_DB', 'orchestrator_db')
            db_user = os.getenv('POSTGRES_USER', 'postgres')
            db_password = os.getenv('POSTGRES_PASSWORD', '')
            db_host = os.getenv('POSTGRES_HOST', 'localhost')
            db_port = os.getenv('POSTGRES_PORT', '5432')
        
        print(f"📍 Database: {db_name}")
        print(f"🖥️  Host: {db_host}:{db_port}\n")
        
        # Connect
        conn = psycopg2.connect(
            dbname=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port
        )
        conn.autocommit = False
        cursor = conn.cursor()
        
        print("✅ Connected successfully!\n")
        
        # Load seed file
        db_path = Path(__file__).parent
        tools_file = db_path / "mcp_tools_seed.json"
        
        if not tools_file.exists():
            print(f"❌ Seed file not found: {tools_file}")
            return False
        
        print("📂 Loading seed file...")
        with open(tools_file, 'r') as f:
            mcp_tools = json.load(f)
        
        # Create mapping of tool name -> metadata
        seed_metadata = {}
        for tool in mcp_tools:
            name = tool.get('name')
            metadata = tool.get('metadata', {})
            if name and metadata:
                seed_metadata[name] = metadata
        
        print(f"✅ Loaded {len(seed_metadata)} tools from seed file\n")
        
        # Get all tools from database
        cursor.execute("SELECT id, name, metadata FROM mcp_tools")
        db_tools = cursor.fetchall()
        
        print(f"📊 Found {len(db_tools)} tools in database\n")
        
        # Update tools
        updated = 0
        skipped = 0
        not_found = 0
        
        for tool_id, tool_name, db_metadata in db_tools:
            if tool_name not in seed_metadata:
                not_found += 1
                continue
            
            seed_meta = seed_metadata[tool_name]
            # Handle both JSON string and dict (PostgreSQL JSONB returns dict)
            if isinstance(db_metadata, str):
                db_meta = json.loads(db_metadata) if db_metadata else {}
            elif isinstance(db_metadata, dict):
                db_meta = db_metadata
            else:
                db_meta = {}
            
            # Check if credential_type needs updating
            seed_cred_type = seed_meta.get('credential_type')
            db_cred_type = db_meta.get('credential_type')
            
            if seed_cred_type and seed_cred_type != db_cred_type:
                # Merge metadata, prioritizing seed values
                updated_metadata = {**db_meta, **seed_meta}
                
                cursor.execute("""
                    UPDATE mcp_tools 
                    SET metadata = %s
                    WHERE id = %s
                """, (json.dumps(updated_metadata), tool_id))
                
                updated += 1
                print(f"  ✅ Updated {tool_name}: credential_type = {seed_cred_type}")
            else:
                skipped += 1
        
        conn.commit()
        
        print("\n" + "=" * 60)
        print(f"✅ UPDATE COMPLETE!")
        print(f"   • Updated: {updated} tools")
        print(f"   • Skipped: {skipped} tools (already up to date)")
        print(f"   • Not in seed: {not_found} tools")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"\n❌ Error updating tool metadata: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = update_tool_metadata()
    sys.exit(0 if success else 1)
