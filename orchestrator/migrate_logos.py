"""
Migration script to add logo field to tools table and populate it
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from core.database.database import SessionLocal

# Brand mapping - extract brand from tool name
BRAND_MAPPING = {
    'discord': 'Discord',
    'github': 'GitHub',
    'gitlab': 'GitLab',
    'aws': 'AWS',
    'azure': 'Azure',
    'google': 'GoogleCloud',
    'slack': 'Slack',
    'openai': 'OpenAI',
    'anthropic': 'Anthropic',
    'mongodb': 'MongoDB',
    'postgresql': 'PostgreSQL',
    'mysql': 'MySQL',
    'redis': 'Redis',
    'stripe': 'Stripe',
    'twilio': 'Twilio',
    'sendgrid': 'SendGrid',
    'mailgun': 'Mailgun',
    'datadog': 'Datadog',
    'sentry': 'Sentry',
    'atlassian': 'Atlassian',
    'asana': 'Asana',
    'trello': 'Trello',
    'notion': 'Notion',
    'linear': 'Linear',
    'shopify': 'Shopify',
    'salesforce': 'Salesforce',
    'hubspot': 'HubSpot',
    'zendesk': 'Zendesk',
    'intercom': 'Intercom',
    'amplitude': 'Amplitude',
    'mixpanel': 'Mixpanel',
    'segment': 'Segment',
    'docker': 'Docker',
    'kubernetes': 'Kubernetes',
    'terraform': 'Terraform',
    'jenkins': 'Jenkins',
    'circleci': 'CircleCI',
    'bitbucket': 'Bitbucket',
    'dropbox': 'Dropbox',
    'box': 'Box',
    'zoom': 'Zoom',
    'telegram': 'Telegram',
    'paypal': 'PayPal',
    'square': 'Square',
    'cloudflare': 'Cloudflare',
    'vercel': 'Vercel',
    'netlify': 'Netlify',
    'heroku': 'Heroku',
    'digitalocean': 'DigitalOcean',
}

def get_brand_from_name(tool_name: str) -> str:
    """Extract brand name from tool name"""
    if not tool_name:
        return None
    
    # Normalize
    name_lower = tool_name.lower()
    
    # Check for known brands
    for brand_key, brand_name in BRAND_MAPPING.items():
        if brand_key in name_lower:
            return brand_name
    
    # Default: use first word, title case
    first_word = tool_name.split()[0]
    return first_word.capitalize()

def main():
    db = SessionLocal()
    
    try:
        # 1. Add logo column if it doesn't exist
        print("Adding logo column to mcp_tools table...")
        db.execute(text("""
            ALTER TABLE mcp_tools 
            ADD COLUMN IF NOT EXISTS logo VARCHAR(255)
        """))
        db.commit()
        print("✅ Logo column added")
        
        # 2. Get all tools
        result = db.execute(text("SELECT id, name FROM mcp_tools"))
        tools = result.fetchall()
        
        print(f"\n📦 Found {len(tools)} tools")
        print("Updating logo paths...\n")
        
        # 3. Update each tool's logo
        updated = 0
        for tool_id, tool_name in tools:
            brand = get_brand_from_name(tool_name)
            if brand:
                logo_path = f"/logos/{brand}.png"
                db.execute(
                    text("UPDATE mcp_tools SET logo = :logo WHERE id = :id"),
                    {"logo": logo_path, "id": tool_id}
                )
                print(f"✅ {tool_name} → {logo_path}")
                updated += 1
        
        db.commit()
        print(f"\n✨ Updated {updated} tools with logo paths!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()
