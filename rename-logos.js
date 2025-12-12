#!/usr/bin/env node
/**
 * Rename logo files from domain format to brand format
 * discord_com.png → Discord.png
 * aws_amazon_com.png → AWS.png
 */

const fs = require('fs');
const path = require('path');

const LOGO_DIR = path.join(__dirname, 'frontend', 'public', 'logos');

// Mapping from domain filenames to brand names
const BRAND_MAPPINGS = {
    'openai_com.png': 'OpenAI.png',
    'anthropic_com.png': 'Anthropic.png',
    'huggingface_co.png': 'HuggingFace.png',
    'cohere_com.png': 'Cohere.png',
    'replicate_com.png': 'Replicate.png',

    'postgresql_org.png': 'PostgreSQL.png',
    'mysql_com.png': 'MySQL.png',
    'mongodb_com.png': 'MongoDB.png',
    'redis_io.png': 'Redis.png',
    'elastic_co.png': 'Elasticsearch.png',

    'aws_amazon_com.png': 'AWS.png',
    'azure_microsoft_com.png': 'Azure.png',
    'cloud_google_com.png': 'GoogleCloud.png',
    'digitalocean_com.png': 'DigitalOcean.png',
    'heroku_com.png': 'Heroku.png',
    'vercel_com.png': 'Vercel.png',
    'netlify_com.png': 'Netlify.png',

    'slack_com.png': 'Slack.png',
    'discord_com.png': 'Discord.png',
    'telegram_org.png': 'Telegram.png',
    'twilio_com.png': 'Twilio.png',
    'sendgrid_com.png': 'SendGrid.png',
    'mailgun_com.png': 'Mailgun.png',
    'zoom_us.png': 'Zoom.png',
    'teams_microsoft_com.png': 'MicrosoftTeams.png',

    'github_com.png': 'GitHub.png',
    'gitlab_com.png': 'GitLab.png',
    'bitbucket_org.png': 'Bitbucket.png',
    'jenkins_io.png': 'Jenkins.png',
    'circleci_com.png': 'CircleCI.png',

    'atlassian_com.png': 'Atlassian.png',
    'asana_com.png': 'Asana.png',
    'trello_com.png': 'Trello.png',
    'notion_so.png': 'Notion.png',
    'linear_app.png': 'Linear.png',
    'monday_com.png': 'Monday.png',

    'datadoghq_com.png': 'Datadog.png',
    'newrelic_com.png': 'NewRelic.png',
    'sentry_io.png': 'Sentry.png',
    'mixpanel_com.png': 'Mixpanel.png',
    'amplitude_com.png': 'Amplitude.png',
    'analytics_google_com.png': 'GoogleAnalytics.png',

    'stripe_com.png': 'Stripe.png',
    'paypal_com.png': 'PayPal.png',
    'squareup_com.png': 'Square.png',
    'shopify_com.png': 'Shopify.png',

    'cloudflare_com.png': 'Cloudflare.png',
    'dropbox_com.png': 'Dropbox.png',
    'box_com.png': 'Box.png',
    'salesforce_com.png': 'Salesforce.png',
    'hubspot_com.png': 'HubSpot.png',
    'mailchimp_com.png': 'Mailchimp.png',
    'intercom_com.png': 'Intercom.png',

    'docker_com.png': 'Docker.png',
    'kubernetes_io.png': 'Kubernetes.png',
    'terraform_io.png': 'Terraform.png',

    // Google services
    'ads_google_com.png': 'GoogleAds.png',
    'books_google_com.png': 'GoogleBooks.png',
    'business_google_com.png': 'GoogleBusiness.png',
    'calendar_google_com.png': 'GoogleCalendar.png',
    'drive_google_com.png': 'GoogleDrive.png',
    'gmail_com.png': 'Gmail.png',
    'sheets_google_com.png': 'GoogleSheets.png',
    'docs_google_com.png': 'GoogleDocs.png',

    // Microsoft services
    'onedrive_live_com.png': 'OneDrive.png',
    'dynamics_microsoft_com.png': 'Dynamics365.png',
    'office_com.png': 'Office365.png',
    'security_microsoft_com.png': 'MicrosoftSecurity.png',
    'outlook_com.png': 'Outlook.png',
};

console.log(`\n🔄 Renaming ${Object.keys(BRAND_MAPPINGS).length} logo files...\n`);

let renamed = 0;
let skipped = 0;

for (const [oldName, newName] of Object.entries(BRAND_MAPPINGS)) {
    const oldPath = path.join(LOGO_DIR, oldName);
    const newPath = path.join(LOGO_DIR, newName);

    if (fs.existsSync(oldPath)) {
        fs.renameSync(oldPath, newPath);
        console.log(`✅ ${oldName} → ${newName}`);
        renamed++;
    } else {
        console.log(`⏭️  ${oldName} (not found)`);
        skipped++;
    }
}

console.log(`\n✨ Done!`);
console.log(`✅ Renamed: ${renamed}`);
console.log(`⏭️  Skipped: ${skipped}\n`);
