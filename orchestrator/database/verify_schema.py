#!/usr/bin/env python3
"""
Schema Verification Script
===========================

Quick verification that the schema file is valid and complete.
Does NOT require database connection - just parses the SQL file.
"""

import re
from pathlib import Path

def verify_schema():
    """Verify the schema file structure"""
    
    print("🔍 Schema File Verification")
    print("=" * 60)
    
    schema_path = Path(__file__).parent / "init_complete_schema.sql"
    
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        return False
    
    print(f"📄 File: {schema_path.name}")
    print(f"📍 Path: {schema_path}")
    print()
    
    with open(schema_path, 'r') as f:
        content = f.read()
    
    # Count different types of SQL statements
    extensions = len(re.findall(r'CREATE EXTENSION', content, re.IGNORECASE))
    tables = len(re.findall(r'CREATE TABLE', content, re.IGNORECASE))
    indexes = len(re.findall(r'CREATE INDEX', content, re.IGNORECASE))
    inserts = len(re.findall(r'INSERT INTO', content, re.IGNORECASE))
    comments = len(re.findall(r'COMMENT ON', content, re.IGNORECASE))
    
    # Count lines
    lines = len(content.split('\n'))
    
    print(f"📊 Schema Statistics:")
    print(f"  • Total lines: {lines:,}")
    print(f"  • Extensions: {extensions}")
    print(f"  • Tables: {tables}")
    print(f"  • Indexes: {indexes}")
    print(f"  • Inserts (seed data): {inserts}")
    print(f"  • Comments: {comments}")
    print()
    
    # Extract table names
    table_matches = re.findall(r'CREATE TABLE IF NOT EXISTS\s+(\w+)', content, re.IGNORECASE)
    
    # Expected critical tables by category
    critical_tables = {
        "Core": ['agents', 'users', 'workflows', 'skills', 'patterns'],
        "Documents": ['documents', 'document_chunks', 'document_usage'],
        "Memory": ['memory_items', 'knowledge_nodes', 'knowledge_edges'],
        "Tools": ['mcp_tools', 'agent_tool_assignments', 'tool_credentials', 'tool_usage_logs'],
        "Context": ['context_policies', 'context_templates', 'context_examples'],
        "Analytics": ['system_metrics', 'analytics_snapshots', 'dashboard_configs'],
        "Collaboration": ['agent_messages', 'shared_contexts', 'collaboration_sessions'],
        "Learning": ['learning_outcomes', 'agent_performance_tracking']
    }
    
    print("🔍 Critical Tables Verification:")
    all_found = True
    for category, expected_tables in critical_tables.items():
        found = [t for t in expected_tables if t in table_matches]
        missing = [t for t in expected_tables if t not in table_matches]
        
        if len(found) == len(expected_tables):
            print(f"  ✅ {category}: {len(found)}/{len(expected_tables)}")
        else:
            print(f"  ❌ {category}: {len(found)}/{len(expected_tables)}")
            print(f"     Missing: {', '.join(missing)}")
            all_found = False
    
    print()
    
    # Check for required extensions
    required_extensions = ['uuid-ossp', 'vector', 'pg_trgm']
    print("🔌 Required Extensions:")
    for ext in required_extensions:
        if ext in content:
            print(f"  ✅ {ext}")
        else:
            print(f"  ❌ {ext} - NOT FOUND!")
            all_found = False
    
    print()
    
    # Check for seed data
    print("🌱 Seed Data:")
    seed_tables = ['system_configurations', 'rag_configurations', 'llm_models', 'schema_versions']
    for table in seed_tables:
        if f'INSERT INTO {table}' in content.upper():
            print(f"  ✅ {table}")
        else:
            print(f"  ⚠️  {table} - No seed data")
    
    print()
    print("=" * 60)
    
    if all_found:
        print("✅ Schema file verification PASSED!")
        print(f"   {tables} tables, {indexes} indexes, {extensions} extensions")
        return True
    else:
        print("❌ Schema file verification FAILED!")
        print("   Some critical tables or extensions are missing")
        return False

if __name__ == "__main__":
    import sys
    success = verify_schema()
    sys.exit(0 if success else 1)

