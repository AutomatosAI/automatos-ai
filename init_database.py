#!/usr/bin/env python3
"""
Database Initialization Script
===============================

Run this to create/update all database tables for Automatos AI
This ensures the database matches our complete schema
"""

import os
import sys
import psycopg2
from psycopg2 import sql
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add orchestrator to path for config
sys.path.append(str(Path(__file__).parent / "orchestrator"))

def init_database():
    """Initialize database with complete schema"""
    
    print("🗄️  AUTOMATOS AI DATABASE INITIALIZATION")
    print("=" * 60)
    
    try:
        # Import config
        from orchestrator.config import config
        
        print(f"📍 Database: {config.POSTGRES_DB}")
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
        
        # Set autocommit for CREATE EXTENSION commands
        conn.autocommit = True
        cursor = conn.cursor()
        
        print("✅ Connected successfully!")
        
        # Read the schema file
        schema_path = Path(__file__).parent / "orchestrator" / "database" / "init_complete_schema.sql"
        
        if not schema_path.exists():
            print(f"❌ Schema file not found: {schema_path}")
            return False
        
        print(f"\n📋 Loading schema from: {schema_path.name}")
        
        with open(schema_path, 'r') as f:
            schema_sql = f.read()
        
        # Split into individual statements (crude but works for our schema)
        statements = []
        current_statement = []
        
        for line in schema_sql.split('\n'):
            # Skip comments and empty lines
            if line.strip().startswith('--') or not line.strip():
                continue
            
            current_statement.append(line)
            
            # Check if statement is complete
            if line.strip().endswith(';'):
                statement = '\n'.join(current_statement)
                statements.append(statement)
                current_statement = []
        
        # Execute each statement
        print(f"\n🚀 Executing {len(statements)} SQL statements...")
        
        success_count = 0
        error_count = 0
        
        for i, statement in enumerate(statements, 1):
            try:
                # Get first few words for logging
                stmt_preview = ' '.join(statement.split()[:3])
                
                # Execute statement
                cursor.execute(statement)
                success_count += 1
                
                # Log progress for important statements
                if 'CREATE TABLE' in statement:
                    table_name = statement.split('CREATE TABLE IF NOT EXISTS')[1].split('(')[0].strip()
                    print(f"  ✅ Created table: {table_name}")
                elif 'CREATE EXTENSION' in statement:
                    print(f"  ✅ Enabled extension")
                elif 'CREATE INDEX' in statement:
                    index_name = statement.split('CREATE INDEX IF NOT EXISTS')[1].split(' ON ')[0].strip()
                    print(f"  ✅ Created index: {index_name}")
                
            except psycopg2.errors.DuplicateTable:
                # Table already exists - that's fine
                success_count += 1
            except psycopg2.errors.DuplicateObject:
                # Object already exists - that's fine
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"  ❌ Error on statement {i}: {str(e)[:100]}")
        
        print(f"\n📊 Results:")
        print(f"  ✅ Successful: {success_count}")
        print(f"  ❌ Errors: {error_count}")
        
        # Check what tables were created
        print("\n📋 Verifying tables...")
        
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        
        print(f"\n🗂️  Total tables in database: {len(tables)}")
        
        # Check for critical tables
        critical_tables = [
            'agents', 'workflows', 'task_decompositions', 'task_assignments',
            'agent_runtimes', 'context_templates', 'memory_items', 'knowledge_nodes'
        ]
        
        existing_tables = [t[0] for t in tables]
        
        print("\n🔍 Critical tables check:")
        for table in critical_tables:
            if table in existing_tables:
                print(f"  ✅ {table}")
            else:
                print(f"  ❌ {table} - MISSING!")
        
        # Check for extensions
        cursor.execute("SELECT extname FROM pg_extension;")
        extensions = [ext[0] for ext in cursor.fetchall()]
        
        print("\n🔌 Extensions:")
        if 'uuid-ossp' in extensions:
            print("  ✅ uuid-ossp (UUID generation)")
        else:
            print("  ❌ uuid-ossp - MISSING!")
        
        if 'vector' in extensions:
            print("  ✅ pgvector (Vector embeddings)")
        else:
            print("  ⚠️  pgvector - Not installed (optional)")
        
        # Close connection
        cursor.close()
        conn.close()
        
        print("\n" + "=" * 60)
        
        if error_count == 0:
            print("✅ DATABASE INITIALIZATION SUCCESSFUL!")
            print("\nYour database is ready for:")
            print("  • Task decomposition with orchestration tables")
            print("  • Agent management with runtime configuration")
            print("  • Context engineering with templates and examples")
            print("  • Memory systems with hierarchical storage")
            print("  • Multi-agent communication")
            print("  • Analytics and monitoring")
            return True
        else:
            print(f"⚠️  DATABASE INITIALIZATION COMPLETED WITH {error_count} ERRORS")
            print("Review the errors above and fix any issues")
            return False
        
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
            except:
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
