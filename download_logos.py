"""
Automated Logo Downloader
Downloads brand logos from multiple sources
"""
import requests
import time
from pathlib import Path
import json

# Brand to domain mapping (common domains)
BRAND_DOMAINS = {
    # Cloud/Infrastructure
    "AWS": "aws.amazon.com",
    "Azure": "azure.microsoft.com",
    "GoogleCloud": "cloud.google.com",
    "Cloudflare": "cloudflare.com",
    "Netlify": "netlify.com",
    "DigitalOcean": "digitalocean.com",
    
    # Development
    "GitHub": "github.com",
    "GitLab": "gitlab.com",
    "Docker": "docker.com",
    "Kubernetes": "kubernetes.io",
    "CircleCI": "circleci.com",
    "Jenkins": "jenkins.io",
    "Git": "git-scm.com",
    
    # Communication
    "Slack": "slack.com",
    "Discord": "discord.com",
    "Telegram": "telegram.org",
    "Zoom": "zoom.us",
    "Microsoft": "microsoft.com",
    "Mattermost": "mattermost.com",
    "Matrix": "matrix.org",
    "Rocket": "rocket.chat",
    
    # Databases
    "MongoDB": "mongodb.com",
    "PostgreSQL": "postgresql.org",
    "MySQL": "mysql.com",
    "Redis": "redis.io",
    "Elasticsearch": "elastic.co",
    "Postgres": "postgresql.org",
    "Snowflake": "snowflake.com",
    "Supabase": "supabase.com",
    
    # AI/ML
    "OpenAI": "openai.com",
    "Anthropic": "anthropic.com",
    "Huggingface": "huggingface.co",
    "Perplexity": "perplexity.ai",
    
    # CRM/Sales
    "Salesforce": "salesforce.com",
    "HubSpot": "hubspot.com",
    "Zendesk": "zendesk.com",
    "Intercom": "intercom.com",
    "Pipedrive": "pipedrive.com",
    "Freshdesk": "freshdesk.com",
    "Freshworks": "freshworks.com",
    "Zoho": "zoho.com",
    
    # Payment
    "Stripe": "stripe.com",
    "PayPal": "paypal.com",
    "Shopify": "shopify.com",
    
    # Project Management
    "Jira": "atlassian.com",
    "Asana": "asana.com",
    "Monday.com": "monday.com",
    "Trello": "trello.com",
    "Linear": "linear.app",
    "Notion": "notion.so",
    "Airtable": "airtable.com",
    "Coda": "coda.io",
    "Baserow": "baserow.io",
    "Clickup": "clickup.com",
    
    # Marketing
    "Mailchimp": "mailchimp.com",
    "SendGrid": "sendgrid.com",
    "Twilio": "twilio.com",
    "Brevo": "brevo.com",
    "Mailgun": "mailgun.com",
    "Mailjet": "mailjet.com",
    "Postmark": "postmarkapp.com",
    
    # Analytics
    "Datadog": "datadoghq.com",
    "Sentry": "sentry.io",
    "Grafana": "grafana.com",
    "Splunk": "splunk.com",
    "Mixpanel": "mixpanel.com",
    
    # Design/Creative
    "Figma": "figma.com",
    "Miro": "miro.com",
    "Canva": "canva.com",
    
    # Productivity
    "Calendly": "calendly.com",
    "Cal": "cal.com",
    "Todoist": "todoist.com",
    "Toggl": "toggl.com",
    "Clockify": "clockify.me",
    
    # Development Tools
    "Bitbucket": "bitbucket.org",
    "Contentful": "contentful.com",
    "Strapi": "strapi.io",
    "Ghost": "ghost.org",
    "Wordpress": "wordpress.org",
    "Webflow": "webflow.com",
    
    # Business
    "Quickbooks": "quickbooks.intuit.com",
    "Xero": "xero.com",
    "Oracle": "oracle.com",
    "Odoo": "odoo.com",
    
    # Social
    "Facebook": "facebook.com",
    "Linkedin": "linkedin.com",
    "Reddit": "reddit.com",
    "Twitter": "twitter.com",
    "X": "x.com",
    "Youtube": "youtube.com",
    
    # Security
    "Okta": "okta.com",
    "Auth0": "auth0.com",
    "Crowdstrike": "crowdstrike.com",
    "Fortinet": "fortinet.com",
    "Cisco": "cisco.com",
    
    # Other Popular
    "Dropbox": "dropbox.com",
    "Box": "box.com",
    "Spotify": "spotify.com",
    "Typeform": "typeform.com",
    "Surveymonkey": "surveymonkey.com",
    "Eventbrite": "eventbrite.com",
    "Gong": "gong.io",
    "Harvest": "getharvest.com",
    "Woocommerce": "woocommerce.com",
    "Magento": "magento.com",
    "Bubble": "bubble.io",
    "Zapier": "zapier.com",
    "Nextcloud": "nextcloud.com",
    "Bitwarden": "bitwarden.com",
    "Kafka": "kafka.apache.org",
    "Rabbitmq": "rabbitmq.com",
    "Nginx": "nginx.com",
    "Apache": "apache.org",
}

CLEARBIT_URL = "https://logo.clearbit.com/{domain}"

def download_logo(brand, domain, output_dir):
    """Download logo from Clearbit"""
    try:
        url = CLEARBIT_URL.format(domain=domain)
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Save logo
            output_path = output_dir / f"{brand}.png"
            output_path.write_bytes(response.content)
            return True, "Success"
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    # Output directory
    output_dir = Path("frontend/public/logos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting automated logo download...")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Attempting to download {len(BRAND_DOMAINS)} logos\n")
    
    success = []
    failed = []
    
    for i, (brand, domain) in enumerate(BRAND_DOMAINS.items(), 1):
        print(f"[{i}/{len(BRAND_DOMAINS)}] {brand:30s} ", end="")
        
        # Check if already exists
        logo_path = output_dir / f"{brand}.png"
        if logo_path.exists():
            print("⏭️  Already exists")
            success.append(brand)
            continue
        
        # Download
        ok, msg = download_logo(brand, domain, output_dir)
        
        if ok:
            print(f"✅ {msg}")
            success.append(brand)
        else:
            print(f"❌ {msg}")
            failed.append({"brand": brand, "domain": domain, "error": msg})
        
        # Rate limiting
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Success: {len(success)}/{len(BRAND_DOMAINS)} ({len(success)/len(BRAND_DOMAINS)*100:.1f}%)")
    print(f"❌ Failed:  {len(failed)}/{len(BRAND_DOMAINS)} ({len(failed)/len(BRAND_DOMAINS)*100:.1f}%)")
    
    if failed:
        print(f"\n❌ Failed brands:")
        for item in failed:
            print(f"  - {item['brand']:30s} ({item['domain']}): {item['error']}")
        
        # Save failed list
        with open('failed_logos.json', 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"\n💾 Failed logos saved to failed_logos.json")
    
    print(f"\n✨ Done! Downloaded {len(success)} logos to {output_dir}")

if __name__ == "__main__":
    main()
