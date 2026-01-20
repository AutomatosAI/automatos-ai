#!/usr/bin/env python3
"""
Seed Data Loader
================

Loads essential seed data (credential types and MCP tools) into the database.
Run after init_database.py to populate with 416 credential types and 416 MCP tools.

Usage:
    python load_seed_data.py                    # Load all seed data
    python load_seed_data.py --credentials-only # Load only credential types
    python load_seed_data.py --tools-only       # Load only MCP tools
"""

import json
import sys
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_seed_data(load_credentials=True, load_tools=True):
    """Load seed data from JSON files"""
    
    print("🌱 SEED DATA LOADER")
    print("=" * 60)
    
    try:
        # Get database config from environment variables (works in Docker and locally)
        db_name = os.getenv('POSTGRES_DB', 'orchestrator_db')
        db_user = os.getenv('POSTGRES_USER', 'postgres')
        db_password = os.getenv('POSTGRES_PASSWORD', '')
        db_host = os.getenv('POSTGRES_HOST', 'localhost')
        db_port = os.getenv('POSTGRES_PORT', '5432')
        
        print(f"📍 Database: {db_name}")
        print(f"🖥️  Host: {db_host}:{db_port}")
        print()
        
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
        
        # Get database directory (where this script and JSON files are located)
        db_path = Path(__file__).parent
        
        # Load Credential Types
        if load_credentials:
            cred_file = db_path / "credential_types_seed.json"
            if cred_file.exists():
                print("📂 Loading credential types...")
                with open(cred_file, 'r') as f:
                    credential_types = json.load(f)
                
                inserted = 0
                skipped = 0
                
                for cred in credential_types:
                    try:
                        cursor.execute("""
                            INSERT INTO credential_types 
                            (id, name, display_name, category, icon, description, 
                             schema_definition, test_endpoint, documentation_url, is_system, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (name) DO NOTHING
                        """, (
                            cred['id'],
                            cred['name'],
                            cred['display_name'],
                            cred['category'],
                            cred.get('icon'),
                            cred.get('description'),
                            json.dumps(cred['schema_definition']),
                            json.dumps(cred.get('test_endpoint')) if cred.get('test_endpoint') else None,
                            cred.get('documentation_url'),
                            cred.get('is_system', True),
                            cred.get('is_active', True)
                        ))
                        if cursor.rowcount > 0:
                            inserted += 1
                        else:
                            skipped += 1
                    except Exception as e:
                        print(f"  ⚠️  Error inserting {cred['name']}: {str(e)[:100]}")
                        skipped += 1
                
                conn.commit()
                print(f"  ✅ Inserted: {inserted} credential types")
                print(f"  ⏭️  Skipped: {skipped} (already exist)")
            else:
                print(f"  ⚠️  Credential types file not found: {cred_file}")
        
        # Load MCP Tools
        if load_tools:
            tools_file = db_path / "mcp_tools_seed.json"
            if tools_file.exists():
                print("\n📂 Loading MCP tools...")
                with open(tools_file, 'r') as f:
                    mcp_tools = json.load(f)
                
                inserted = 0
                updated = 0
                skipped = 0
                
                for tool in mcp_tools:
                    try:
                        # Check if tool exists first
                        cursor.execute("SELECT id FROM mcp_tools WHERE name = %s", (tool['name'],))
                        existing = cursor.fetchone()
                        is_update = existing is not None
                        
                        cursor.execute("""
                            INSERT INTO mcp_tools 
                            (id, name, description, mcp_server_url, capabilities, credentials_schema,
                             status, provider, version, icon, category, tags, metadata, created_by)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (name) DO UPDATE SET
                                description = EXCLUDED.description,
                                mcp_server_url = EXCLUDED.mcp_server_url,
                                capabilities = EXCLUDED.capabilities,
                                credentials_schema = EXCLUDED.credentials_schema,
                                status = EXCLUDED.status,
                                provider = EXCLUDED.provider,
                                version = EXCLUDED.version,
                                icon = EXCLUDED.icon,
                                category = EXCLUDED.category,
                                tags = EXCLUDED.tags,
                                metadata = EXCLUDED.metadata,
                                updated_at = CURRENT_TIMESTAMP
                        """, (
                            tool['id'],
                            tool['name'],
                            tool.get('description'),
                            tool.get('mcp_server_url'),
                            json.dumps(tool.get('capabilities')) if tool.get('capabilities') else None,
                            json.dumps(tool.get('credentials_schema')) if tool.get('credentials_schema') else None,
                            tool.get('status', 'active'),
                            tool.get('provider'),
                            tool.get('version'),
                            tool.get('icon'),
                            tool.get('category'),
                            tool.get('tags'),  # PostgreSQL array
                            json.dumps(tool.get('metadata')) if tool.get('metadata') else None,
                            tool.get('created_by', 'system')
                        ))
                        
                        if is_update:
                            updated += 1
                        else:
                            inserted += 1
                    except Exception as e:
                        print(f"  ⚠️  Error inserting {tool['name']}: {str(e)[:100]}")
                        skipped += 1
                
                conn.commit()
                print(f"  ✅ Inserted: {inserted} MCP tools")
                print(f"  🔄 Updated: {updated} MCP tools")
                print(f"  ⏭️  Skipped: {skipped} (no changes)")
            else:
                print(f"  ⚠️  MCP tools file not found: {tools_file}")
        
        # Verify counts
        print("\n📊 Database Verification:")
        cursor.execute("SELECT COUNT(*) FROM credential_types")
        cred_count = cursor.fetchone()[0]
        print(f"  • Credential Types: {cred_count}")
        
        cursor.execute("SELECT COUNT(*) FROM mcp_tools")
        tools_count = cursor.fetchone()[0]
        print(f"  • MCP Tools: {tools_count}")
        
        cursor.close()
        conn.close()
        
        # Load System Settings (PRD-25)
        print("\n📂 Loading system settings...")
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from seeds.seed_system_settings import seed_system_settings
            from database.database import get_db_session
            
            with get_db_session() as db:
                created, updated = seed_system_settings(db)
                print(f"  ✅ System settings: {created} created, {updated} updated")
        except Exception as e:
            print(f"  ⚠️  Error loading system settings: {e}")
            # Don't fail the entire seed process
        
        # Load LLM Models
        print("\n📂 Loading LLM models...")
        try:
            from seeds.seed_models import seed_models
            seed_models()
            print("  ✅ LLM models seeded")
        except Exception as e:
            print(f"  ⚠️  Error loading LLM models: {e}")
        
        # Load Skills and Patterns
        print("\n📂 Loading skills and patterns...")
        try:
            from seeds.seed_skills import seed_skills, seed_patterns
            seed_skills()
            seed_patterns()
            print("  ✅ Skills and patterns seeded")
        except Exception as e:
            print(f"  ⚠️  Error loading skills/patterns: {e}")
        
        print("\n" + "=" * 60)
        print("✅ SEED DATA LOADED SUCCESSFULLY!")
        print(f"   {cred_count} credential types + {tools_count} MCP tools + system settings + models + skills")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading seed data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load seed data into database')
    parser.add_argument('--credentials-only', action='store_true', help='Load only credential types')
    parser.add_argument('--tools-only', action='store_true', help='Load only MCP tools')
    
    args = parser.parse_args()
    
    load_creds = not args.tools_only
    load_tools = not args.credentials_only
    
    success = load_seed_data(load_credentials=load_creds, load_tools=load_tools)
    sys.exit(0 if success else 1)

