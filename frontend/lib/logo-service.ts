/**
 * Logo Service - Get Real Company Logos for Tools & Credentials
 * ==============================================================
 * 
 * This utility provides real brand logos for tools and credentials.
 * It uses a fallback chain for maximum reliability:
 * 
 * 1. Logo.dev API (if API key configured) - Best quality
 * 2. Google Favicon Service - Free, no API key required
 * 3. DuckDuckGo Icons - Alternative free service
 * 4. Fallback to Lucide icon name
 */

/**
 * Comprehensive domain mapping for tools and services
 * Add more mappings as you discover new tools
 */
const TOOL_DOMAINS: Record<string, string> = {
    // AI & ML Services
    'openai': 'openai.com',
    'openai_api': 'openai.com',
    'anthropic': 'anthropic.com',
    'anthropic_api': 'anthropic.com',
    'huggingface': 'huggingface.co',
    'huggingface_api': 'huggingface.co',
    'cohere': 'cohere.com',
    'replicate': 'replicate.com',

    // Databases
    'postgres': 'postgresql.org',
    'postgresql': 'postgresql.org',
    'postgres_credentials': 'postgresql.org',
    'mysql': 'mysql.com',
    'mysql_credentials': 'mysql.com',
    'mongodb': 'mongodb.com',
    'mongodb_credentials': 'mongodb.com',
    'redis': 'redis.io',
    'redis_credentials': 'redis.io',
    'elasticsearch': 'elastic.co',
    'elasticsearch_credentials': 'elastic.co',
    'cassandra': 'cassandra.apache.org',
    'dynamodb': 'aws.amazon.com',

    // Cloud Providers
    'aws': 'aws.amazon.com',
    'aws_credentials': 'aws.amazon.com',
    'amazon': 'aws.amazon.com',
    'azure': 'azure.microsoft.com',
    'azure_credentials': 'azure.microsoft.com',
    'microsoft_azure': 'azure.microsoft.com',
    'gcp': 'cloud.google.com',
    'digitalocean': 'digitalocean.com',
    'heroku': 'heroku.com',
    'vercel': 'vercel.com',
    'netlify': 'netlify.com',

    // Communication & Collaboration
    'slack': 'slack.com',
    'slack_api': 'slack.com',
    'discord': 'discord.com',
    'discord_webhook': 'discord.com',
    'telegram': 'telegram.org',
    'telegram_api': 'telegram.org',
    'twilio': 'twilio.com',
    'twilio_api': 'twilio.com',
    'sendgrid': 'sendgrid.com',
    'sendgrid_api': 'sendgrid.com',
    'mailgun': 'mailgun.com',
    'zoom': 'zoom.us',
    'teams': 'teams.microsoft.com',
    'microsoft_teams': 'teams.microsoft.com',

    // Version Control & CI/CD
    'github': 'github.com',
    'github_api': 'github.com',
    'gitlab': 'gitlab.com',
    'gitlab_api': 'gitlab.com',
    'bitbucket': 'bitbucket.org',
    'jenkins': 'jenkins.io',
    'circleci': 'circleci.com',
    'travis': 'travis-ci.com',
    'travis_ci': 'travis-ci.com',

    // Project Management
    'jira': 'atlassian.com',
    'asana': 'asana.com',
    'trello': 'trello.com',
    'notion': 'notion.so',
    'linear': 'linear.app',
    'monday': 'monday.com',

    // Analytics & Monitoring
    'datadog': 'datadoghq.com',
    'newrelic': 'newrelic.com',
    'new_relic': 'newrelic.com',
    'sentry': 'sentry.io',
    'mixpanel': 'mixpanel.com',
    'amplitude': 'amplitude.com',
    'google_analytics': 'analytics.google.com',

    // Payment & E-commerce
    'stripe': 'stripe.com',
    'paypal': 'paypal.com',
    'square': 'squareup.com',
    'shopify': 'shopify.com',

    // Storage & CDN
    's3': 'aws.amazon.com',
    'cloudflare': 'cloudflare.com',
    'dropbox': 'dropbox.com',
    'box': 'box.com',

    // CRM & Marketing
    'salesforce': 'salesforce.com',
    'hubspot': 'hubspot.com',
    'mailchimp': 'mailchimp.com',
    'intercom': 'intercom.com',

    // Development Tools
    'docker': 'docker.com',
    'kubernetes': 'kubernetes.io',
    'terraform': 'terraform.io',
    'ansible': 'ansible.com',
    'npm': 'npmjs.com',
    'yarn': 'yarnpkg.com',
    'pip': 'pypi.org',

    // ========================================================================
    // GOOGLE SERVICES - Service-specific domains for better logo matching
    // ========================================================================
    'google_ads': 'ads.google.com',
    'google_ads_oauth2_mcp': 'ads.google.com',
    'google_analytics': 'analytics.google.com',
    'google_analytics_oauth2_mcp': 'analytics.google.com',
    'google_bigquery': 'cloud.google.com',
    'google_bigquery_oauth2_mcp': 'cloud.google.com',
    'google_books': 'books.google.com',
    'google_books_oauth2_mcp': 'books.google.com',
    'google_business_profile': 'business.google.com',
    'google_business_profile_oauth2_mcp': 'business.google.com',
    'google_calendar': 'calendar.google.com',
    'google_calendar_oauth2_mcp': 'calendar.google.com',
    'google_cloud': 'cloud.google.com',
    'google_cloud_natural_language_oauth2_mcp': 'cloud.google.com',
    'google_drive': 'drive.google.com',
    'google_drive_oauth2_mcp': 'drive.google.com',
    'google_gmail': 'gmail.com',
    'google_gmail_oauth2_mcp': 'gmail.com',
    'google_sheets': 'sheets.google.com',
    'google_sheets_oauth2_mcp': 'sheets.google.com',
    'google_docs': 'docs.google.com',
    'google_docs_oauth2_mcp': 'docs.google.com',
    'gong_oauth2_mcp': 'gong.io',

    // ========================================================================
    // MICROSOFT SERVICES - Service-specific domains for better logo matching
    // ========================================================================
    'microsoft_azure_monitor_oauth2_mcp': 'azure.microsoft.com',
    'microsoft_drive_oauth2_mcp': 'onedrive.live.com',
    'microsoft_dynamics_oauth2_mcp': 'dynamics.microsoft.com',
    'microsoft_entra_id_oauth2_mcp': 'entra.microsoft.com',
    'microsoft_excel_oauth2_mcp': 'office.com',
    'microsoft_graph_security_oauth2_mcp': 'security.microsoft.com',
    'microsoft_oauth2_mcp': 'microsoft.com',
    'microsoft_outlook_oauth2_mcp': 'outlook.com',
    'microsoft_sharepoint_oauth2_mcp': 'sharepoint.com',
    'microsoft_teams_oauth2_mcp': 'teams.microsoft.com',
    'microsoft_onedrive': 'onedrive.live.com',
    'microsoft_onedrive_oauth2_mcp': 'onedrive.live.com',
}

/**
 * Logo service configuration
 */
interface LogoConfig {
    size?: number
    format?: 'png' | 'svg' | 'webp'
    fallbackToIcon?: boolean
}

/**
 * Extract brand name from tool name
 * Examples:
 * - "Discord Bot MCP" → "Discord"
 * - "AWS Services" → "AWS"
 * - "GitHub OAuth2 MCP" → "GitHub"
 * - "Google Analytics OAuth2 MCP" → "GoogleAnalytics"
 */
export function getBrandFromToolName(toolName: string): string | null {
    if (!toolName) return null;

    const normalized = toolName.trim();

    // Special cases for multi-word brands
    const multiWordBrands: Record<string, string> = {
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
        'Mail Gun': 'Mailgun',
        'Mail Chimp': 'Mailchimp',
        'Hub Spot': 'HubSpot',
        'Dynamics 365': 'Dynamics365',
        'Office 365': 'Office365',
        'Circle CI': 'CircleCI',
        'Hugging Face': 'HuggingFace',
    };

    // Check for multi-word brand matches
    for (const [pattern, brand] of Object.entries(multiWordBrands)) {
        if (normalized.toLowerCase().includes(pattern.toLowerCase())) {
            return brand;
        }
    }

    // Extract first word as brand name
    const firstWord = normalized.split(/[\s_-]/)[0];

    // Special case for proper capitalization
    const brandCapitalization: Record<string, string> = {
        'github': 'GitHub',
        'gitlab': 'GitLab',
        'mongodb': 'MongoDB',
        'mysql': 'MySQL',
        'postgresql': 'PostgreSQL',
        'redis': 'Redis',
        'elasticsearch': 'Elasticsearch',
        'kubernetes': 'Kubernetes',
        'terraform': 'Terraform',
        'jenkins': 'Jenkins',
        'docker': 'Docker',
        'aws': 'AWS',
        'gcp': 'GoogleCloud',
        'azure': 'Azure',
    };

    const lowerFirst = firstWord.toLowerCase();
    if (brandCapitalization[lowerFirst]) {
        return brandCapitalization[lowerFirst];
    }

    // Already uppercase (AWS, API, etc.)
    if (firstWord.toUpperCase() === firstWord) {
        return firstWord;
    }

    // Title case
    return firstWord.charAt(0).toUpperCase() + firstWord.slice(1).toLowerCase();
}

/**
 * Get logo URL for a tool
 * Simply maps tool name to /logos/BrandName.png
 */
export function getLogoUrl(
    toolName: string,
    provider?: string,
    config: LogoConfig = {}
): string {
    const brand = getBrandFromToolName(provider || toolName);

    if (!brand) {
        return '';
    }

    return `/logos/${brand}.png`;
}

/**
 * Get domain for a tool (kept for backwards compatibility)
 */
export function getDomainForTool(name: string, provider?: string): string | null {
    const brand = getBrandFromToolName(provider || name);
    return brand ? `${brand.toLowerCase()}.com` : null;
}

/**
 * React hook for logo URLs with error handling
 */
export function useLogoUrl(toolName: string, provider?: string, config: LogoConfig = {}) {
    const logoUrl = getLogoUrl(toolName, provider, config)
    const fallbackIcon = toolName // Use tool name as fallback

    return {
        logoUrl,
        fallbackIcon,
        hasDomain: !!getDomainForTool(toolName, provider)
    }
}

/**
 * Preload logos for better performance
 */
export function preloadLogos(tools: Array<{ name: string; provider?: string }>) {
    tools.forEach(tool => {
        const url = getLogoUrl(tool.name, tool.provider)
        if (url) {
            const img = new Image()
            img.src = url
        }
    })
}

/**
 * Add a custom domain mapping at runtime
 */
export function addDomainMapping(toolName: string, domain: string) {
    TOOL_DOMAINS[toolName.toLowerCase()] = domain
}

/**
 * Get all available domain mappings
 */
export function getAllDomainMappings(): Record<string, string> {
    return { ...TOOL_DOMAINS }
}
