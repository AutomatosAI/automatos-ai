#!/usr/bin/env python3
"""
Seed Credentials from .env File
================================

One-time migration script to populate credential system from existing .env file.
This allows transition from environment variables to encrypted database storage.

Usage:
    python scripts/seed_credentials_from_env.py [--dry-run] [--force]
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from core.database.database import SessionLocal, init_database
from core.credentials.service import CredentialStore
from core.models.credentials import CredentialCreate

# Load .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


def seed_credentials(dry_run: bool = False, force: bool = False):
    """
    Seed credentials from .env file into credential system.
    
    Args:
        dry_run: If True, only show what would be created
        force: If True, update existing credentials
    """
    print("🔐 PRD-18: Credential Migration from .env to Database")
    print("=" * 70)
    print()

    if dry_run:
        print("⚠️  DRY RUN MODE - No changes will be made")
        print()

    # Initialize database
    if not dry_run:
        print("📊 Initializing database...")
        init_database()
        print("✅ Database initialized")
        print()

    db = SessionLocal()
    store = CredentialStore(db)

    credentials_to_create = [
        # PostgreSQL Credential
        {
            "name": "postgres_main",
            "credential_type_name": "postgres_credentials",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "description": "Main PostgreSQL database connection (migrated from .env)",
            "tags": ["database", "primary", "migrated"],
            "credential_data": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DB", "orchestrator_db"),
                "user": os.getenv("POSTGRES_USER", "postgres"),
                "password": os.getenv("POSTGRES_PASSWORD", ""),
                "ssl_mode": "prefer"
            }
        },
        
        # Redis Credential
        {
            "name": "redis_main",
            "credential_type_name": "redis_credentials",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "description": "Main Redis cache connection (migrated from .env)",
            "tags": ["cache", "primary", "migrated"],
            "credential_data": {
                "host": os.getenv("REDIS_HOST", "localhost"),
                "port": int(os.getenv("REDIS_PORT", "6379")),
                "password": os.getenv("REDIS_PASSWORD", ""),
                "database": 0
            }
        },
        
        # OpenAI Credential
        {
            "name": "openai_main",
            "credential_type_name": "openai_api",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "description": "Main OpenAI API credentials (migrated from .env)",
            "tags": ["ai", "llm", "primary", "migrated"],
            "credential_data": {
                "api_key": os.getenv("OPENAI_API_KEY", ""),
                "base_url": "https://api.openai.com/v1"
            }
        },
        
        # Anthropic Credential
        {
            "name": "anthropic_main",
            "credential_type_name": "anthropic_api",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "description": "Main Anthropic Claude API credentials (migrated from .env)",
            "tags": ["ai", "llm", "primary", "migrated"],
            "credential_data": {
                "api_key": os.getenv("ANTHROPIC_API_KEY", ""),
                "base_url": "https://api.anthropic.com"
            }
        },
        
        # GitHub Credential
        {
            "name": "github_main",
            "credential_type_name": "github_api",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "description": "Main GitHub API token (migrated from .env)",
            "tags": ["code", "git", "primary", "migrated"],
            "credential_data": {
                "access_token": os.getenv("GITHUB_TOKEN", ""),
                "token_type": "personal"
            }
        },
        
        # SSH Credential (if configured)
        {
            "name": "ssh_deployment",
            "credential_type_name": "ssh_credentials",
            "environment": os.getenv("ENVIRONMENT", "production"),
            "description": "SSH deployment credentials (migrated from .env)",
            "tags": ["infrastructure", "deployment", "migrated"],
            "credential_data": {
                "host": os.getenv("DEPLOY_HOST", ""),
                "port": int(os.getenv("DEPLOY_PORT", "22")),
                "username": os.getenv("DEPLOY_USER", "root"),
                "auth_method": "key",
                "private_key": "",  # Would need to read from DEPLOY_KEY_PATH
                "passphrase": ""
            }
        }
    ]

    created_count = 0
    skipped_count = 0
    updated_count = 0

    for cred_config in credentials_to_create:
        credential_type_name = cred_config.pop("credential_type_name")
        
        # Get credential type ID
        cred_type = store.get_credential_type_by_name(credential_type_name)
        if not cred_type:
            print(f"❌ Credential type '{credential_type_name}' not found - skipping {cred_config['name']}")
            skipped_count += 1
            continue

        # Check if any required credential data is empty
        has_required_data = False
        for field in cred_type.schema_definition:
            field_name = field.get('name')
            if field.get('required') and cred_config['credential_data'].get(field_name):
                has_required_data = True
                break
        
        if not has_required_data:
            print(f"⏭️  Skipping {cred_config['name']} - no data in .env file")
            skipped_count += 1
            continue

        # Check if credential already exists
        existing = store.get_credential_by_name(cred_config['name'], cred_config['environment'])
        
        if existing and not force:
            print(f"⏭️  Skipping {cred_config['name']} - already exists (use --force to update)")
            skipped_count += 1
            continue

        if dry_run:
            print(f"📝 Would create: {cred_config['name']} ({credential_type_name})")
            print(f"   Environment: {cred_config['environment']}")
            print(f"   Fields: {', '.join(cred_config['credential_data'].keys())}")
            print()
            created_count += 1
            continue

        try:
            if existing and force:
                # Update existing credential
                from core.models.credentials import CredentialUpdate
                store.update_credential(
                    credential_id=existing.id,
                    update_data=CredentialUpdate(
                        credential_data=cred_config['credential_data'],
                        description=cred_config['description'],
                        tags=cred_config['tags']
                    ),
                    user_id="migration_script"
                )
                print(f"✅ Updated: {cred_config['name']}")
                updated_count += 1
            else:
                # Create new credential
                store.create_credential(
                    credential_data=CredentialCreate(
                        name=cred_config['name'],
                        credential_type_id=cred_type.id,
                        credential_data=cred_config['credential_data'],
                        environment=cred_config['environment'],
                        description=cred_config['description'],
                        tags=cred_config['tags']
                    ),
                    user_id="migration_script"
                )
                print(f"✅ Created: {cred_config['name']}")
                created_count += 1

        except Exception as e:
            print(f"❌ Failed to create {cred_config['name']}: {e}")
            skipped_count += 1

    db.close()

    print()
    print("=" * 70)
    print("📊 Migration Summary:")
    print(f"   ✅ Created: {created_count}")
    print(f"   🔄 Updated: {updated_count}")
    print(f"   ⏭️  Skipped: {skipped_count}")
    print()
    
    if not dry_run:
        print("🎉 Credential migration complete!")
        print()
        print("📋 Next Steps:")
        print("   1. Verify credentials in Settings > Credentials tab")
        print("   2. Test each credential using the 'Test' button")
        print("   3. Once verified, remove credentials from .env file")
        print("   4. Restart services to use new credential system")
        print()
        print("⚠️  CRITICAL: Backup your encryption key!")
        print(f"   Key location: {Path(__file__).parent.parent / '.credential_key'}")
        print("   Command: cp .credential_key .credential_key.backup")
    else:
        print("💡 Run without --dry-run to perform migration")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed credentials from .env file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be created without making changes")
    parser.add_argument("--force", action="store_true", help="Update existing credentials")
    
    args = parser.parse_args()
    
    seed_credentials(dry_run=args.dry_run, force=args.force)

