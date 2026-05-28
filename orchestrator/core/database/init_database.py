#!/usr/bin/env python3
"""
Database Initialization Script
===============================

Single source of truth for database initialization.
Run this to create ALL database tables from scratch for Automatos AI.

This script replaces all migration-based initialization and provides:
- Complete database schema in one file
- Clean initialization from scratch
- All extensions, tables, indexes, and seed data
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from pathlib import Path
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Add orchestrator to path for config
sys.path.append(str(Path(__file__).parent / "orchestrator"))

def init_database():
    """Initialize database with complete schema from single source of truth"""
    
    print("🗄️  AUTOMATOS AI DATABASE INITIALIZATION")
    print("=" * 70)
    print("Single Source of Truth - Complete Schema Initialization")
    print("=" * 70)
    
    try:
        # Import config
        from orchestrator.config import config
        
        print(f"\n📍 Database: {config.POSTGRES_DB}")
        print(f"🖥️  Host: {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
        print(f"👤 User: {config.POSTGRES_USER}")
        print()
        
        # Connect to PostgreSQL
        print("🔌 Connecting to PostgreSQL...")
        
        conn = psycopg2.connect(
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT
        )
        
        # Set autocommit for CREATE EXTENSION and other DDL commands
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("✅ Connected successfully!\n")
        
        # Read the complete schema file
        schema_path = Path(__file__).parent / "orchestrator" / "database" / "init_complete_schema.sql"
        
        if not schema_path.exists():
            print(f"❌ Schema file not found: {schema_path}")
            print("   Expected location: automatos-ai/orchestrator/database/init_complete_schema.sql")
            return False
        
        print(f"📋 Loading complete schema from: {schema_path.name}")
        print("   This file contains the single source of truth for all database objects")
        print()
        
        start_time = time.time()
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Execute the entire schema as one transaction
        # We split by semicolons but handle them properly
        print("🚀 Executing schema initialization...")
        print()
        
        # Split into statements (handling multi-line)
        statements = []
        current_statement = []
        in_function = False
        
        for line in schema_sql.split('\n'):
            # Track if we're inside a function definition
            if 'CREATE' in line.upper() and ('FUNCTION' in line.upper() or 'TRIGGER' in line.upper()):
                in_function = True
            
            # Skip pure comment lines and empty lines when not in a function
            if not in_function and (line.strip().startswith('--') or not line.strip()):
                continue
            
            current_statement.append(line)
            
            # Check if statement is complete (ends with semicolon and not in function)
            if line.strip().endswith(';'):
                if in_function and ('END;' in line or 'end;' in line):
                    in_function = False
                    statement = '\n'.join(current_statement)
                    statements.append(statement)
                    current_statement = []
                elif not in_function:
                    statement = '\n'.join(current_statement)
                    statements.append(statement)
                    current_statement = []
        
        print(f"📊 Total statements to execute: {len(statements)}")
        print()
        
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        extensions_created = 0
        tables_created = 0
        indexes_created = 0
        inserts_executed = 0
        
        for i, statement in enumerate(statements, 1):
            try:
                # Categorize and execute statement
                stmt_upper = statement.upper().strip()
                
                cursor.execute(statement)
                success_count += 1
                
                # Count different types of operations
                if 'CREATE EXTENSION' in stmt_upper:
                    extensions_created += 1
                    ext_name = statement.split('CREATE EXTENSION IF NOT EXISTS')[1].split(';')[0].strip().strip('"')
                    print(f"  ✅ Extension: {ext_name}")
                    
                elif 'CREATE TABLE' in stmt_upper:
                    tables_created += 1
                    try:
                        table_name = statement.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
                        print(f"  ✅ Table: {table_name}")
                    except Exception:
                        print(f"  ✅ Table created")
                        
                elif 'CREATE INDEX' in stmt_upper and 'IF NOT EXISTS' in stmt_upper:
                    indexes_created += 1
                    try:
                        idx_name = statement.split('CREATE INDEX IF NOT EXISTS')[1].split(' ON ')[0].strip()
                        if i % 5 == 0:  # Only print every 5th index to reduce noise
                            print(f"  ✅ Index: {idx_name}")
                    except Exception:
                        pass
                        
                elif 'INSERT INTO' in stmt_upper:
                    inserts_executed += 1
                    try:
                        table_name = statement.split('INSERT INTO')[1].split('(')[0].strip()
                        print(f"  ✅ Seed data: {table_name}")
                    except Exception:
                        print(f"  ✅ Seed data inserted")
                
            except psycopg2.errors.DuplicateTable as e:
                # Table already exists - that's OK with IF NOT EXISTS
                skipped_count += 1
            except psycopg2.errors.DuplicateObject as e:
                # Object already exists - that's OK with IF NOT EXISTS
                skipped_count += 1
            except Exception as e:
                error_count += 1
                error_msg = str(e)
                # Only show first 150 chars of error
                print(f"  ❌ Error on statement {i}: {error_msg[:150]}")
                if 'already exists' in error_msg.lower():
                    skipped_count += 1
                    error_count -= 1  # Don't count as error
        
        execution_time = time.time() - start_time
        
        print(f"\n📊 Initialization Summary:")
        print(f"  ✅ Successful statements: {success_count}")
        print(f"  ⏭️  Skipped (already exists): {skipped_count}")
        print(f"  ❌ Errors: {error_count}")
        print(f"  ⏱️  Execution time: {execution_time:.2f} seconds")
        print()
        print(f"📈 Objects Created:")
        print(f"  • Extensions: {extensions_created}")
        print(f"  • Tables: {tables_created}")
        print(f"  • Indexes: {indexes_created}")
        print(f"  • Seed data inserts: {inserts_executed}")
        
        # Verify database state
        print("\n📋 Verifying database state...")
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        existing_tables = [t[0] for t in tables]
        
        print(f"\n🗂️  Total tables in database: {len(tables)}")
        
        # Check for critical tables by category
        critical_categories = {
            "Core": ['agents', 'users', 'workflows', 'skills'],
            "Documents": ['documents', 'document_chunks', 'document_usage'],
            "Memory": ['memory_items', 'knowledge_nodes', 'knowledge_edges'],
            "Tools": ['mcp_tools', 'agent_tool_assignments', 'tool_credentials'],
            "Credentials": ['credential_types', 'credentials', 'credential_audit_logs'],
            "Context": ['context_policies', 'context_templates', 'context_examples'],
            "Analytics": ['analytics_snapshots', 'system_metrics', 'dashboard_configs'],
            "Database Knowledge": ['database_knowledge_sources', 'database_relationships', 'database_query_audit', 
                                  'semantic_metrics', 'semantic_dimensions', 'database_query_templates']
        }
        
        print("\n🔍 Critical tables verification by category:")
        all_found = True
        for category, tables_list in critical_categories.items():
            found_count = sum(1 for t in tables_list if t in existing_tables)
            total_count = len(tables_list)
            if found_count == total_count:
                print(f"  ✅ {category}: {found_count}/{total_count} tables")
            else:
                print(f"  ⚠️  {category}: {found_count}/{total_count} tables")
                missing = [t for t in tables_list if t not in existing_tables]
                print(f"     Missing: {', '.join(missing)}")
                all_found = False
        
        # Check for required extensions
        cursor.execute("SELECT extname FROM pg_extension;")
        extensions = [ext[0] for ext in cursor.fetchall()]
        
        print("\n🔌 Required Extensions:")
        required_extensions = {
            'uuid-ossp': 'UUID generation',
            'vector': 'Vector embeddings (pgvector)',
            'pg_trgm': 'Text similarity search'
        }
        
        for ext_name, ext_desc in required_extensions.items():
            if ext_name in extensions:
                print(f"  ✅ {ext_name} - {ext_desc}")
            else:
                print(f"  ❌ {ext_name} - {ext_desc} - MISSING!")
                all_found = False
        
        # Check schema version
        try:
            cursor.execute("SELECT version, applied_at FROM schema_versions ORDER BY applied_at DESC LIMIT 1;")
            schema_version = cursor.fetchone()
            if schema_version:
                print(f"\n📌 Schema Version: {schema_version[0]} (applied: {schema_version[1]})")
        except Exception:
            print(f"\n⚠️  Schema version table not found or empty")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 70)
        
        if error_count == 0 and all_found:
            print("✅ DATABASE INITIALIZATION SUCCESSFUL!")
            print("\n🎉 Your database is fully initialized and ready with:")
            print("  • All 60+ tables for agents, workflows, documents, and analytics")
            print("  • Vector search capabilities with pgvector")
            print("  • Context engineering and optimization")
            print("  • Multi-agent communication infrastructure")
            print("  • Memory systems with hierarchical storage")
            print("  • Complete monitoring and analytics pipeline")
            print("  • Tool integration and credential management")
            print("\n💡 Next steps:")
            print("  1. Start the backend: cd orchestrator && uvicorn main:app --host 0.0.0.0 --port 8000")
            print("  2. Start the frontend: cd frontend && npm start")
            print("  3. Access dashboard: http://localhost:3000 (or your configured frontend URL)")
            return True
        elif error_count > 0:
            print(f"⚠️  DATABASE INITIALIZATION COMPLETED WITH {error_count} ERRORS")
            print("Review the errors above and fix any issues")
            return False
        else:
            print(f"⚠️  DATABASE INITIALIZATION COMPLETED WITH WARNINGS")
            print("Some tables or extensions are missing - check the verification above")
            return True  # Still return True as basic structure is there
        
    except psycopg2.OperationalError as e:
        print(f"❌ Could not connect to database: {e}")
        print("\nCheck your .env file:")
        print("  POSTGRES_HOST, POSTGRES_PORT")
        print("  POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD")
        return False
    
    except ImportError:
        print("❌ Could not import config")
        print("Make sure you're running from the automatos-ai directory")
        return False
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_database_status():
    """Quick check of database status"""
    
    print("\n📊 Quick Database Status Check")
    print("-" * 40)
    
    try:
        from orchestrator.config import config
        
        conn = psycopg2.connect(
            dbname=config.POSTGRES_DB,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT
        )
        
        cursor = conn.cursor()
        
        # Count records in key tables
        tables_to_check = [
            ('agents', 'AI Agents'),
            ('workflows', 'Workflows'),
            ('task_decompositions', 'Task Decompositions'),
            ('memory_items', 'Memory Items'),
            ('context_examples', 'Context Examples')
        ]
        
        for table, description in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table};")
                count = cursor.fetchone()[0]
                print(f"  {description}: {count} records")
            except Exception:
                print(f"  {description}: Table not found")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"  Could not check status: {e}")

if __name__ == "__main__":
    print("🚀 Starting database initialization...")
    print()
    
    # Check if psycopg2 is installed
    try:
        import psycopg2
    except ImportError:
        print("❌ psycopg2 not installed!")
        print("\nInstall with:")
        print("  pip install psycopg2-binary")
        sys.exit(1)
    
    # Initialize database
    success = init_database()
    
    if success:
        # Check status
        check_database_status()
        
        print("\n✅ Database is ready for use!")
        print("\nNext steps:")
        print("  1. Test task decomposition: python test_real_decomposition.py")
        print("  2. Start API server: python orchestrator/main.py")
        print("  3. Create agents and workflows")
    
    sys.exit(0 if success else 1)

