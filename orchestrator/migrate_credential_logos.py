"""
Migration script to add logo field to credential_types table and populate it
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from core.database.database import SessionLocal

# Brand mapping
BRAND_MAPPING = {
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'huggingface': 'HuggingFace',
    'aws': 'AWS',
    'azure': 'Azure',
    'google': 'GoogleCloud',
    'github': 'GitHub',
    'gitlab': 'GitLab',
    'slack': 'Slack',
    'discord': 'Discord',
    'telegram': 'Telegram',
    'postgresql': 'PostgreSQL',
    'mysql': 'MySQL',
    'mongodb': 'MongoDB',
    'redis': 'Redis',
    'elasticsearch': 'Elasticsearch',
    'stripe': 'Stripe',
    'twilio': 'Twilio',
    'sendgrid': 'SendGrid',
    'mailgun': 'Mailgun',
    'datadog': 'Datadog',
    'sentry': 'Sentry',
    'shopify': 'Shopify',
    'salesforce': 'Salesforce',
    'hubspot': 'HubSpot',
    'zendesk': 'Zendesk',
    'intercom': 'Intercom',
    'bitbucket': 'Bitbucket',
    'dropbox': 'Dropbox',
    'box': 'Box',
}

def get_brand_from_name(name):
    """Extract brand name from credential name"""
    if not name:
        return None
    
    name_lower = name.lower()
    
    # Check for known brands
    for key, brand in BRAND_MAPPING.items():
        if key in name_lower:
            return brand
    
    # Default: first word title case
    import re
    first_word = re.split(r'[\s_-]', name)[0]
    return first_word.capitalize()

def main():
    db = SessionLocal()
    
    try:
        # 1. Add logo column if it doesn't exist
        print("Adding logo column to credential_types table...")
        db.execute(text("""
            ALTER TABLE credential_types 
            ADD COLUMN IF NOT EXISTS logo VARCHAR(255)
        """))
        db.commit()
        print("✅ Logo column added")
        
        # 2. Get all credential types
        result = db.execute(text("SELECT id, name, display_name FROM credential_types"))
        creds = result.fetchall()
        
        print(f"\n📦 Found {len(creds)} credential types")
        print("Updating logo paths...\n")
        
        # 3. Update each credential type's logo
        updated = 0
        for cred_id, name, display_name in creds:
            # Try display_name first, then name
            brand = get_brand_from_name(display_name) or get_brand_from_name(name)
            if brand:
                logo_path = f"/logos/{brand}.png"
                db.execute(
                    text("UPDATE credential_types SET logo = :logo WHERE id = :id"),
                    {"logo": logo_path, "id": cred_id}
                )
                print(f"✅ {display_name or name} → {logo_path}")
                updated += 1
        
        db.commit()
        print(f"\n✨ Updated {updated} credential types with logo paths!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
