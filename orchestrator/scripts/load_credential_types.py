#!/usr/bin/env python3
"""
Load Credential Types into Database
====================================

Loads all credential type definitions (cloned from n8n) into the database.
This populates the credential_types table with 15+ curated types.

Usage:
    python scripts/load_credential_types.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from database.database import SessionLocal, init_database
from models.credentials import CredentialType
from credential_types.all_credential_types import ALL_CREDENTIAL_TYPES
import json


def load_credential_types():
    """Load all credential type definitions into database"""
    print("🔐 Loading Credential Types from n8n Definitions")
    print("=" * 70)
    print()

    # Initialize database
    print("📊 Initializing database...")
    init_database()
    print("✅ Database initialized")
    print()

    db = SessionLocal()
    
    loaded_count = 0
    updated_count = 0
    skipped_count = 0

    print(f"📦 Loading {len(ALL_CREDENTIAL_TYPES)} credential types...")
    print()

    for cred_type_data in ALL_CREDENTIAL_TYPES:
        try:
            # Check if type already exists
            existing = db.query(CredentialType).filter(
                CredentialType.name == cred_type_data['name']
            ).first()

            if existing:
                # Update existing
                existing.display_name = cred_type_data['display_name']
                existing.category = cred_type_data.get('category')
                existing.icon = cred_type_data.get('icon')
                existing.description = cred_type_data.get('description')
                existing.schema_definition = cred_type_data['schema_definition']
                existing.test_endpoint = cred_type_data.get('test_endpoint')
                existing.documentation_url = cred_type_data.get('documentation_url')
                existing.is_system = cred_type_data.get('is_system', False)
                existing.is_active = True
                
                print(f"🔄 Updated: {cred_type_data['display_name']}")
                updated_count += 1
            else:
                # Create new
                new_type = CredentialType(
                    name=cred_type_data['name'],
                    display_name=cred_type_data['display_name'],
                    category=cred_type_data.get('category'),
                    icon=cred_type_data.get('icon'),
                    description=cred_type_data.get('description'),
                    schema_definition=cred_type_data['schema_definition'],
                    test_endpoint=cred_type_data.get('test_endpoint'),
                    documentation_url=cred_type_data.get('documentation_url'),
                    is_system=cred_type_data.get('is_system', False),
                    is_active=True
                )
                db.add(new_type)
                
                print(f"✅ Loaded: {cred_type_data['display_name']}")
                loaded_count += 1

            db.commit()

        except Exception as e:
            print(f"❌ Failed to load {cred_type_data.get('name', 'unknown')}: {e}")
            db.rollback()
            skipped_count += 1

    db.close()

    print()
    print("=" * 70)
    print("📊 Loading Summary:")
    print(f"   ✅ Loaded: {loaded_count}")
    print(f"   🔄 Updated: {updated_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print(f"   📦 Total: {loaded_count + updated_count + skipped_count}")
    print()
    print("🎉 Credential types loaded successfully!")
    print()
    print("📋 Next Steps:")
    print("   1. View credential types: http://localhost:8000/docs#/🔐%20Credential%20Management/list_credential_types_api_credentials_types_get")
    print("   2. Create credentials in Settings > Credentials tab")
    print("   3. Or run: python scripts/seed_credentials_from_env.py")
    print()


if __name__ == "__main__":
    load_credential_types()

