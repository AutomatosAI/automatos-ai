#!/usr/bin/env node
/**
 * Bulk Logo Downloader
 * Downloads all brand logos for tools and stores them locally
 */

const https = require('https');
const fs = require('fs');
const path = require('path');

// All domains from TOOL_DOMAINS
const DOMAINS = [
    'openai.com', 'anthropic.com', 'huggingface.co', 'cohere.com', 'replicate.com',
    'postgresql.org', 'mysql.com', 'mongodb.com', 'redis.io', 'elastic.co',
    'aws.amazon.com', 'azure.microsoft.com', 'cloud.google.com', 'digitalocean.com',
    'heroku.com', 'vercel.com', 'netlify.com',
    'slack.com', 'discord.com', 'telegram.org', 'twilio.com', 'sendgrid.com',
    'mailgun.com', 'zoom.us', 'teams.microsoft.com',
    'github.com', 'gitlab.com', 'bitbucket.org', 'jenkins.io', 'circleci.com',
    'atlassian.com', 'asana.com', 'trello.com', 'notion.so', 'linear.app', 'monday.com',
    'datadoghq.com', 'newrelic.com', 'sentry.io', 'mixpanel.com', 'amplitude.com',
    'analytics.google.com', 'stripe.com', 'paypal.com', 'squareup.com', 'shopify.com',
    'cloudflare.com', 'dropbox.com', 'box.com', 'salesforce.com', 'hubspot.com',
    'mailchimp.com', 'intercom.com', 'docker.com', 'kubernetes.io', 'terraform.io',
    'ads.google.com', 'books.google.com', 'business.google.com', 'calendar.google.com',
    'drive.google.com', 'gmail.com', 'sheets.google.com', 'docs.google.com',
    'gong.io', 'onedrive.live.com', 'dynamics.microsoft.com', 'entra.microsoft.com',
    'office.com', 'security.microsoft.com', 'outlook.com', 'sharepoint.com'
];

const OUTPUT_DIR = path.join(__dirname, 'frontend', 'public', 'logos');

// Create output directory
if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
}

// Logo sources to try (in order)
const LOGO_SOURCES = [
    (domain) => `https://img.brandfetch.io/${domain}`,
    (domain) => `https://icons.duckduckgo.com/ip3/${domain}.ico`,
    (domain) => `https://www.google.com/s2/favicons?domain=${domain}&sz=128`
];

function downloadLogo(domain, sourceIndex = 0) {
    return new Promise((resolve, reject) => {
        if (sourceIndex >= LOGO_SOURCES.length) {
            console.log(`❌ ${domain} - All sources failed`);
            resolve(false);
            return;
        }

        const url = LOGO_SOURCES[sourceIndex](domain);
        const filename = domain.replace(/\./g, '_') + '.png';
        const filepath = path.join(OUTPUT_DIR, filename);

        // Skip if already exists
        if (fs.existsSync(filepath)) {
            console.log(`⏭️  ${domain} - Already exists`);
            resolve(true);
            return;
        }

        console.log(`⬇️  ${domain} - Trying source ${sourceIndex + 1}...`);

        https.get(url, (response) => {
            if (response.statusCode === 200) {
                const fileStream = fs.createWriteStream(filepath);
                response.pipe(fileStream);

                fileStream.on('finish', () => {
                    fileStream.close();
                    console.log(`✅ ${domain} - Downloaded successfully`);
                    resolve(true);
                });

                fileStream.on('error', (err) => {
                    fs.unlink(filepath, () => { });
                    console.log(`⚠️  ${domain} - Write error, trying next source...`);
                    downloadLogo(domain, sourceIndex + 1).then(resolve).catch(reject);
                });
            } else {
                console.log(`⚠️  ${domain} - HTTP ${response.statusCode}, trying next source...`);
                downloadLogo(domain, sourceIndex + 1).then(resolve).catch(reject);
            }
        }).on('error', (err) => {
            console.log(`⚠️  ${domain} - Network error, trying next source...`);
            downloadLogo(domain, sourceIndex + 1).then(resolve).catch(reject);
        });
    });
}

async function downloadAll() {
    console.log(`\n🚀 Starting bulk logo download for ${DOMAINS.length} domains...\n`);

    let successful = 0;
    let failed = 0;

    for (const domain of DOMAINS) {
        const result = await downloadLogo(domain);
        if (result) {
            successful++;
        } else {
            failed++;
        }
        // Small delay to avoid rate limiting
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    console.log(`\n✨ Download complete!`);
    console.log(`✅ Successful: ${successful}`);
    console.log(`❌ Failed: ${failed}`);
    console.log(`📁 Logos saved to: ${OUTPUT_DIR}\n`);
}

downloadAll().catch(console.error);
