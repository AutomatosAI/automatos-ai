// Test getBrandFromToolName function
function getBrandFromToolName(toolName) {
    if (!toolName) return null;

    const normalized = toolName.trim();

    // Special cases for multi-word brands
    const multiWordBrands = {
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

    // Capitalize properly
    if (firstWord.toUpperCase() === firstWord) {
        return firstWord;
    }

    return firstWord.charAt(0).toUpperCase() + firstWord.slice(1).toLowerCase();
}

// Test cases
console.log('Discord Bot MCP:', getBrandFromToolName('Discord Bot MCP'));
console.log('Azure Storage OAuth2 MCP:', getBrandFromToolName('Azure Storage OAuth2 MCP'));
console.log('Microsoft Teams:', getBrandFromToolName('Microsoft Teams'));
console.log('AWS Services:', getBrandFromToolName('AWS Services'));
console.log('GitHub OAuth2 MCP:', getBrandFromToolName('GitHub OAuth2 MCP'));
console.log('Google Analytics OAuth2 MCP:', getBrandFromToolName('Google Analytics OAuth2 MCP'));
console.log('Infrastructure:', getBrandFromToolName('Infrastructure'));
