#!/usr/bin/env python3
"""
Fix credential field names to match schema expectations
"""

import psycopg2
from services.encryption_service import get_encryption_service
import json

# Database connection
conn = psycopg2.connect(
    host="127.0.0.1",
    port=5432,
    database="orchestrator_db",
    user="postgres",
    password="secure_password_123"
)

# Get encryption service
encryption_service = get_encryption_service()

# Field name mappings (camelCase -> snake_case)
field_mappings = {
    'apiKey': 'api_key',
    'accessToken': 'access_token',
    'privateKey': 'private_key',
    'headerName': 'header_name',
    'headerPrefix': 'header_prefix'
}

def fix_credential_fields():
    """Fix field names in credential data"""
    
    cursor = conn.cursor()
    
    # Get all credentials
    cursor.execute("SELECT id, name, encrypted_data FROM credentials")
    credentials = cursor.fetchall()
    
    for cred_id, name, encrypted_data in credentials:
        print(f"🔧 Fixing {name}...")
        
        try:
            # Decrypt data
            decrypted = encryption_service.decrypt_dict(encrypted_data)
            
            # Fix field names
            fixed_data = {}
            changed = False
            
            for old_key, value in decrypted.items():
                new_key = field_mappings.get(old_key, old_key)
                fixed_data[new_key] = value
                if old_key != new_key:
                    changed = True
                    print(f"  📝 {old_key} -> {new_key}")
            
            if changed:
                # Re-encrypt with fixed field names
                new_encrypted = encryption_service.encrypt_dict(fixed_data)
                
                # Update database
                cursor.execute(
                    "UPDATE credentials SET encrypted_data = %s WHERE id = %s",
                    (new_encrypted, cred_id)
                )
                print(f"  ✅ Updated {name}")
            else:
                print(f"  ⏭️  No changes needed for {name}")
                
        except Exception as e:
            print(f"  ❌ Error fixing {name}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("\n🎉 Field name fixes complete!")

if __name__ == "__main__":
    fix_credential_fields()
