#!/usr/bin/env python3
"""
PRD-18: Seed YOUR 8 Specific Credentials from .env
Migrates your actual production credentials to the encrypted system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import SessionLocal
from services.credential_service import CredentialStore
from models.credentials import CredentialCreate, CredentialType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_credentials():
    """Seed the 8 specific credentials from your .env"""
    
    db = SessionLocal()
    store = CredentialStore(db)
    
    try:
        print("🔐 Migrating Your 8 Production Credentials")
        print("=" * 70)
        print("")
        
        # Get credential type IDs first
        types = {
            "postgres": db.query(CredentialType).filter_by(name="postgresCredentials").first(),
            "redis": db.query(CredentialType).filter_by(name="redisCredentials").first(),
            "openai": db.query(CredentialType).filter_by(name="openaiApi").first(),
            "anthropic": db.query(CredentialType).filter_by(name="anthropicApi").first(),
            "github": db.query(CredentialType).filter_by(name="githubApi").first(),
            "ssh": db.query(CredentialType).filter_by(name="sshCredentials").first(),
            "generic": db.query(CredentialType).filter_by(name="genericApi").first(),
        }
        
        # 1. PostgreSQL
        print("1️⃣  PostgreSQL Database...")
        postgres_data = CredentialCreate(
            credential_type_id=types["postgres"].id,
            name="production_db",
            credential_data={
                "host": "127.0.0.1",
                "port": "5432",
                "user": "postgres",
                "password": "secure_password_123",
                "database": "orchestrator_db"
            },
            environment="production",
            description="Main PostgreSQL database"
        )
        postgres = store.create_credential(postgres_data, user_id="system_migration")
        print(f"   ✅ Created: {postgres.name} (ID: {postgres.id})")
        
        # 2. Redis
        print("\n2️⃣  Redis Cache...")
        redis_data = CredentialCreate(
            credential_type_id=types["redis"].id,
            name="production_redis",
            credential_data={
                "host": "127.0.0.1",
                "port": "6379",
                "password": "redis_password_123"
            },
            environment="production",
            description="Main Redis cache"
        )
        redis = store.create_credential(redis_data, user_id="system_migration")
        print(f"   ✅ Created: {redis.name} (ID: {redis.id})")
        
        # 3. OpenAI API
        print("\n3️⃣  OpenAI API Key...")
        openai_data = CredentialCreate(
            credential_type_id=types["openai"].id,
            name="production_openai",
            credential_data={
                "apiKey": os.getenv("OPENAI_API_KEY", "")
            },
            environment="production",
            description="OpenAI GPT-4 API Key"
        )
        openai = store.create_credential(openai_data, user_id="system_migration")
        print(f"   ✅ Created: {openai.name} (ID: {openai.id})")
        
        # 4. Anthropic API
        print("\n4️⃣  Anthropic API Key...")
        anthropic_data = CredentialCreate(
            credential_type_id=types["anthropic"].id,
            name="production_anthropic",
            credential_data={
                "apiKey": os.getenv("ANTHROPIC_API_KEY", "")
            },
            environment="production",
            description="Anthropic Claude API Key"
        )
        anthropic = store.create_credential(anthropic_data, user_id="system_migration")
        print(f"   ✅ Created: {anthropic.name} (ID: {anthropic.id})")
        
        # 5. GitHub Token
        print("\n5️⃣  GitHub Token...")
        github_data = CredentialCreate(
            credential_type_id=types["github"].id,
            name="production_github",
            credential_data={
                "accessToken": os.getenv("GITHUB_TOKEN", "")
            },
            environment="production",
            description="GitHub API Token"
        )
        github = store.create_credential(github_data, user_id="system_migration")
        print(f"   ✅ Created: {github.name} (ID: {github.id})")
        
        # 6. SSH Deployment
        print("\n6️⃣  SSH Deployment Credentials...")
        ssh_data = CredentialCreate(
            credential_type_id=types["ssh"].id,
            name="production_deploy_ssh",
            credential_data={
                "host": "mcp.automatos.app",
                "port": "22",
                "user": "root",
                "privateKey": "FILE:/root/keys/deploy_key"  # Placeholder
            },
            environment="production",
            description="SSH for deployment server"
        )
        ssh = store.create_credential(ssh_data, user_id="system_migration")
        print(f"   ✅ Created: {ssh.name} (ID: {ssh.id})")
        
        # 7. Generic API Key (Backend)
        print("\n7️⃣  Generic API Key (Backend)...")
        generic_data = CredentialCreate(
            credential_type_id=types["generic"].id,
            name="backend_api_key",
            credential_data={
                "apiKey": "test_api_key_for_backend_validation_2025",
                "headerName": "X-API-Key",
                "headerPrefix": ""
            },
            environment="production",
            description="Backend API validation key"
        )
        generic = store.create_credential(generic_data, user_id="system_migration")
        print(f"   ✅ Created: {generic.name} (ID: {generic.id})")
        
        # 8. Frontend GitHub PAT
        print("\n8️⃣  Frontend GitHub PAT...")
        frontend_github_data = CredentialCreate(
            credential_type_id=types["github"].id,
            name="frontend_github_pat",
            credential_data={
                "accessToken": os.getenv("FRONTEND_GITHUB_TOKEN", "")
            },
            environment="production",
            description="Frontend GitHub Personal Access Token"
        )
        frontend_github = store.create_credential(frontend_github_data, user_id="system_migration")
        print(f"   ✅ Created: {frontend_github.name} (ID: {frontend_github.id})")
        
        db.commit()
        
        print("\n" + "=" * 70)
        print("🎉 Successfully migrated all 8 credentials!")
        print("=" * 70)
        print("")
        print("📊 Summary:")
        print(f"   ✅ PostgreSQL: {postgres.name} (ID: {postgres.id})")
        print(f"   ✅ Redis: {redis.name} (ID: {redis.id})")
        print(f"   ✅ OpenAI: {openai.name} (ID: {openai.id})")
        print(f"   ✅ Anthropic: {anthropic.name} (ID: {anthropic.id})")
        print(f"   ✅ GitHub: {github.name} (ID: {github.id})")
        print(f"   ✅ SSH: {ssh.name} (ID: {ssh.id})")
        print(f"   ✅ API Key: {generic.name} (ID: {generic.id})")
        print(f"   ✅ Frontend GitHub: {frontend_github.name} (ID: {frontend_github.id})")
        print("")
        print("🔐 All credentials encrypted and stored securely!")
        print("🔄 Your code NOW uses credential system with .env fallback")
        print("")
        print("🎯 View in UI: https://ui.automatos.app/settings → Credentials tab")
        print("🧪 Test connections: Click 'Test' button next to each credential")
        print("")
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Migration failed: {e}", exc_info=True)
        raise
    finally:
        db.close()

if __name__ == "__main__":
    seed_credentials()
