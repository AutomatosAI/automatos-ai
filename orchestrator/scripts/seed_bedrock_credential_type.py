#!/usr/bin/env python3
"""
Seed AWS Bedrock Credential Type
=================================

Adds AWS Bedrock API credential type to the database.
Supports both new Bedrock API Keys and traditional IAM credentials.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from database.database import SessionLocal
from models.credentials import CredentialType


def seed_bedrock_credential_type():
    """Seed AWS Bedrock credential type into database"""
    db = SessionLocal()
    
    try:
        print("🚀 Seeding AWS Bedrock Credential Type")
        print("=" * 60)
        
        # Load credential type definition (simplified version)
        cred_file = Path(__file__).parent.parent / "database" / "aws_bedrock_credential_type_simplified.json"
        
        if not cred_file.exists():
            print(f"❌ Credential type file not found: {cred_file}")
            return False
        
        with open(cred_file, 'r') as f:
            cred_types = json.load(f)
        
        for cred_data in cred_types:
            # Check if already exists
            existing = db.query(CredentialType).filter(
                CredentialType.name == cred_data['name']
            ).first()
            
            if existing:
                # Update existing
                print(f"🔄 Updating existing credential type: {cred_data['display_name']}")
                existing.display_name = cred_data['display_name']
                existing.category = cred_data.get('category')
                existing.icon = cred_data.get('icon')
                existing.description = cred_data.get('description')
                existing.schema_definition = cred_data['schema_definition']
                existing.test_endpoint = cred_data.get('test_endpoint')
                existing.documentation_url = cred_data.get('documentation_url')
                existing.is_system = cred_data.get('is_system', True)
                existing.is_active = True
                
                print(f"  ✅ Updated successfully")
            else:
                # Create new
                print(f"✨ Creating new credential type: {cred_data['display_name']}")
                new_type = CredentialType(
                    name=cred_data['name'],
                    display_name=cred_data['display_name'],
                    category=cred_data.get('category'),
                    icon=cred_data.get('icon'),
                    description=cred_data.get('description'),
                    schema_definition=cred_data['schema_definition'],
                    test_endpoint=cred_data.get('test_endpoint'),
                    documentation_url=cred_data.get('documentation_url'),
                    is_system=cred_data.get('is_system', True),
                    is_active=True
                )
                db.add(new_type)
                print(f"  ✅ Created successfully")
            
            db.commit()
        
        print()
        print("=" * 60)
        print("✅ AWS Bedrock credential type seeded successfully!")
        print()
        print("📝 Next Steps:")
        print("  1. Restart your backend: systemctl restart automatos-backend")
        print("  2. Go to Settings → Credentials → Add Credential")
        print("  3. Select 'AWS Bedrock API' as credential type")
        print("  4. Add your Bedrock API Key or IAM credentials")
        print("  5. Create agents using Bedrock models")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error seeding credential type: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        return False
        
    finally:
        db.close()


if __name__ == "__main__":
    success = seed_bedrock_credential_type()
    sys.exit(0 if success else 1)

