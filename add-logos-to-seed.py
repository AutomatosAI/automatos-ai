"""
Script to add logo field to seed data files
"""
import json
import re

# Brand mapping function
def get_brand_from_name(name):
    """Extract brand name from tool/credential name"""
    if not name:
        return None
    
    # Multi-word brands
    multi_word = {
        'Google Analytics': 'GoogleAnalytics',
        'Google Ads': 'GoogleAds',
        'Google Books': 'GoogleBooks',
        'Google Business': 'GoogleBusiness',
        'Google Calendar': 'GoogleCalendar',
        'Google Drive': 'GoogleDrive',
        'Google Sheets': 'GoogleSheets',
        'Google Docs': 'GoogleDocs',
        'Google Cloud': 'GoogleCloud',
        'Microsoft Teams': 'MicrosoftTeams',
        'Microsoft Security': 'MicrosoftSecurity',
        'Azure Storage': 'Azure',
        'New Relic': 'NewRelic',
        'Digital Ocean': 'DigitalOcean',
        'Send Grid': 'SendGrid',
        'Hub Spot': 'HubSpot',
        'Hugging Face': 'HuggingFace',
    }
    
    for pattern, brand in multi_word.items():
        if pattern.lower() in name.lower():
            return brand
    
    # Extract first word
    first_word = re.split(r'[\s_-]', name)[0]
    
    # Proper capitalization
    brand_caps = {
        'github': 'GitHub',
        'gitlab': 'GitLab',
        'mongodb': 'MongoDB',
        'mysql': 'MySQL',
        'postgresql': 'PostgreSQL',
        'redis': 'Redis',
        'elasticsearch': 'Elasticsearch',
        'kubernetes': 'Kubernetes',
        'aws': 'AWS',
        'azure': 'Azure',
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'huggingface': 'HuggingFace',
        'discord': 'Discord',
        'slack': 'Slack',
        'telegram': 'Telegram',
        'twilio': 'Twilio',
        'sendgrid': 'SendGrid',
    }
    
    lower_first = first_word.lower()
    if lower_first in brand_caps:
        return brand_caps[lower_first]
    
    # Already uppercase
    if first_word.isupper():
        return first_word
    
    # Title case
    return first_word.capitalize()

# Update mcp_tools_seed.json
print("Updating mcp_tools_seed.json...")
with open('orchestrator/core/database/mcp_tools_seed.json', 'r') as f:
    tools = json.load(f)

updated = 0
for tool in tools:
    brand = get_brand_from_name(tool.get('name', ''))
    if brand:
        tool['logo'] = f"/logos/{brand}.png"
        updated += 1

with open('orchestrator/core/database/mcp_tools_seed.json', 'w') as f:
    json.dump(tools, f, indent=2)

print(f"✅ Updated {updated} tools in mcp_tools_seed.json")

# Update credentials types.py
print("\nUpdating credentials/types.py...")
with open('orchestrator/core/credentials/types.py', 'r') as f:
    content = f.read()

# Find ALL_CREDENTIAL_TYPES list
import ast
tree = ast.parse(content)

# Count credentials
cred_count = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'ALL_CREDENTIAL_TYPES':
                if isinstance(node.value, ast.List):
                    cred_count = len(node.value.elts)

print(f"Found {cred_count} credential types")

# Add logo field to each credential type
# This is complex because we need to parse Python dict literals
# For now, let's just add a note that it needs manual update

print("""
⚠️  Manual update needed for credentials/types.py:

Add 'logo' field to each credential type dict, for example:

{
    "name": "openai_api",
    "display_name": "OpenAI API",
    "category": "ai",
    "icon": "brain",
    "logo": "/logos/OpenAI.png",  # <-- ADD THIS
    "domain": "openai.com",
    ...
}

The logo should be: "/logos/{BrandName}.png"
Where BrandName is extracted from display_name or name.
""")

print(f"\n✨ Done! Updated {updated} tools")



