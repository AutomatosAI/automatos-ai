#!/usr/bin/env python3
"""
Seed Data Loader
================

Loads essential seed data (credential types and system settings) into the database.
Run after init_database.py to populate credential types and core platform defaults.

Usage:
    python load_seed_data.py                    # Load all seed data
    python load_seed_data.py --credentials-only # Load only credential types
"""

import json
import sys
import os
import psycopg2
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from config import config

def load_seed_data(load_credentials: bool = True, load_platform_defaults: bool = True) -> bool:
    """Load seed data from JSON files"""
    
    print("🌱 SEED DATA LOADER")
    print("=" * 60)
    
    try:
        # Get database config from environment variables (works in Docker and locally)
        db_name = config.POSTGRES_DB or 'orchestrator_db'
        db_user = config.POSTGRES_USER or 'postgres'
        db_password = config.POSTGRES_PASSWORD or ''
        db_host = config.POSTGRES_HOST or 'localhost'
        db_port = config.POSTGRES_PORT or '5432'
        
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
                            ON CONFLICT (name) DO UPDATE SET
                                display_name = EXCLUDED.display_name,
                                category = EXCLUDED.category,
                                icon = EXCLUDED.icon,
                                description = EXCLUDED.description,
                                schema_definition = EXCLUDED.schema_definition,
                                test_endpoint = EXCLUDED.test_endpoint,
                                documentation_url = EXCLUDED.documentation_url
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
        
        # Verify counts
        print("\n📊 Database Verification:")
        cursor.execute("SELECT COUNT(*) FROM credential_types")
        cred_count = cursor.fetchone()[0]
        print(f"  • Credential Types: {cred_count}")
        
        cursor.close()
        conn.close()
        
        if load_platform_defaults:
            # Load System Settings (PRD-25)
            print("\n📂 Loading system settings...")
            try:
                from core.seeds.seed_system_settings import seed_system_settings
                from core.database.database import get_db_session
                
                with get_db_session() as db:
                    created, updated = seed_system_settings(db)
                    print(f"  ✅ System settings: {created} created, {updated} updated")
            except Exception as e:
                print(f"  ⚠️  Error loading system settings: {e}")
                # Don't fail the entire seed process
            
            # Load LLM Models
            print("\n📂 Loading LLM models...")
            try:
                from core.seeds.seed_models import seed_models
                seed_models()
                print("  ✅ LLM models seeded")
            except Exception as e:
                print(f"  ⚠️  Error loading LLM models: {e}")
            
            # Load Skills and Patterns
            print("\n📂 Loading skills and patterns...")
            try:
                from core.seeds.seed_skills import seed_skills, seed_patterns
                seed_skills()
                seed_patterns()
                print("  ✅ Skills and patterns seeded")
            except Exception as e:
                print(f"  ⚠️  Error loading skills/patterns: {e}")

            # Load Personas
            print("\n📂 Loading personas...")
            try:
                from core.seeds.seed_personas import seed_personas
                from core.database.database import get_db_session as _get_db_session

                with _get_db_session() as db:
                    created, updated = seed_personas(db)
                    print(f"  ✅ Personas: {created} created, {updated} updated")
            except Exception as e:
                print(f"  ⚠️  Error loading personas: {e}")

            # Marketplace catalog (PRD-209 local first-run; PRD-233 S3 owns the
            # curated refresh). All idempotent: v2 agents check by name, Shopify
            # agents + packages upsert by slug. Starter agents DELETE+reinsert the
            # 'Automatos Team' items (would churn ids that marketplace_installs
            # reference) — so they run only into an EMPTY catalog.
            print("\n📂 Loading marketplace catalog...")
            try:
                from core.database.database import get_db_session as _mk_session
                from sqlalchemy import text as _sql
                with _mk_session() as db:
                    catalog_rows = db.execute(_sql("SELECT count(*) FROM marketplace_items")).scalar() or 0
                if catalog_rows == 0:
                    from scripts.seed_starter_agents import seed_starter_agents
                    seed_starter_agents()
                    print("  ✅ Starter agents seeded (empty catalog)")
                from scripts.seed_marketplace_agents_v2 import seed_marketplace_agents_v2
                seed_marketplace_agents_v2()
                print("  ✅ Marketplace agents v2 seeded")
            except Exception as e:
                print(f"  ⚠️  Error loading marketplace agents: {e}")
            try:
                from core.seeds.seed_shopify_agents import seed_shopify_agents
                seed_shopify_agents()
                print("  ✅ Shopify agents seeded")
            except Exception as e:
                print(f"  ⚠️  Error loading Shopify agents: {e}")
            try:
                from core.seeds.seed_packages import seed_packages
                created, updated = seed_packages()
                print(f"  ✅ Packages: {created} created, {updated} updated")
            except Exception as e:
                print(f"  ⚠️  Error loading packages: {e}")

            # Load Plugin Categories
            print("\n📂 Loading plugin categories...")
            try:
                from core.seeds.seed_plugin_categories import seed_plugin_categories
                from core.database.database import get_db_session as __get_db_session

                with __get_db_session() as db:
                    created, updated = seed_plugin_categories(db)
                    print(f"  ✅ Plugin categories: {created} created, {updated} updated")
            except Exception as e:
                print(f"  ⚠️  Error loading plugin categories: {e}")

            # PRD-233 S3: local-edition first-run content — the workspace +
            # operator rows, Auto, a starter roster, one demo Playbook and a
            # welcome Deliverable. Gated INSIDE the seed on AUTH_EDITION=local
            # + DEFAULT_WORKSPACE_ID (saas ⇒ no-op); idempotent-refresh, never
            # overwrites edits. Package-qualified imports so the seed shares the
            # app's ORM session/model objects (module mode: python -m ...).
            print("\n📂 Loading local-edition first-run content...")
            try:
                from core.seeds.seed_local_first_run import seed_local_first_run
                from core.database.database import get_db_session as _local_session

                with _local_session() as db:
                    outcome = seed_local_first_run(db)
                    print(f"  ✅ Local first-run: {outcome}")
            except Exception as e:
                print(f"  ⚠️  Error loading local-edition first-run content: {e}")

        print("\n" + "=" * 60)
        print("✅ SEED DATA LOADED SUCCESSFULLY!")
        print(f"   {cred_count} credential types + system settings + models + skills")
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
    
    args = parser.parse_args()
    
    success = load_seed_data(load_credentials=True, load_platform_defaults=not args.credentials_only)
    sys.exit(0 if success else 1)

