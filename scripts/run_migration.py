#!/usr/bin/env python3
"""
Database Migration Runner
========================

Safely run SQL migration files against the orchestrator database.
"""

import os
import sys
import argparse
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.database.connection import get_db_connection
from sqlalchemy import text

def run_migration(migration_file: str):
    """
    Run a SQL migration file against the database.
    
    Args:
        migration_file: Path to the migration SQL file
    """
    migration_path = Path(migration_file)
    
    if not migration_path.exists():
        print(f"❌ Migration file not found: {migration_file}")
        sys.exit(1)
    
    print(f"📁 Reading migration file: {migration_path.name}")
    
    with open(migration_path, 'r') as f:
        migration_sql = f.read()
    
    print(f"🔌 Connecting to database...")
    
    try:
        # Get database connection
        db = next(get_db_connection())
        
        print(f"🚀 Executing migration: {migration_path.name}")
        print("=" * 60)
        
        # Execute migration
        result = db.execute(text(migration_sql))
        db.commit()
        
        print("=" * 60)
        print(f"✅ Migration completed successfully!")
        print(f"📊 Affected rows: {result.rowcount if hasattr(result, 'rowcount') else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Migration failed:")
        print(f"   {str(e)}")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()

def list_migrations():
    """List all available migrations"""
    migrations_dir = Path(__file__).parent.parent / "database" / "migrations"
    
    if not migrations_dir.exists():
        print("❌ Migrations directory not found")
        return
    
    print("📋 Available migrations:")
    print("=" * 60)
    
    for migration_file in sorted(migrations_dir.glob("*.sql")):
        print(f"  • {migration_file.name}")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description="Run database migrations for Automatos AI Platform"
    )
    parser.add_argument(
        "--file",
        "-f",
        help="Path to migration SQL file (relative to orchestrator/database/)"
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available migrations"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_migrations()
        return
    
    if not args.file:
        print("❌ Please specify a migration file with --file")
        print("   Use --list to see available migrations")
        sys.exit(1)
    
    # Construct full path
    base_dir = Path(__file__).parent.parent / "database"
    migration_path = base_dir / args.file
    
    run_migration(str(migration_path))

if __name__ == "__main__":
    main()
