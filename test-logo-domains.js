// Test logo domain extraction
const TOOL_DOMAINS = {
    'aws': 'aws.amazon.com',
    'azure': 'azure.microsoft.com',
    'discord': 'discord.com',
    'github': 'github.com',
    'gitlab': 'gitlab.com',
    'google': 'google.com',
};

function getDomainForTool(name, provider) {
    const originalKey = (provider || name).toLowerCase().replace(/[\s-]/g, '_');

    console.log(`Testing: "${name}" (provider: "${provider}")`);
    console.log(`  originalKey: "${originalKey}"`);

    // Direct match
    if (TOOL_DOMAINS[originalKey]) {
        console.log(`  ✅ Direct match: ${TOOL_DOMAINS[originalKey]}`);
        return TOOL_DOMAINS[originalKey];
    }

    // Try without underscores
    const cleanKey = originalKey.replace(/_/g, '');
    if (TOOL_DOMAINS[cleanKey]) {
        console.log(`  ✅ Clean match: ${TOOL_DOMAINS[cleanKey]}`);
        return TOOL_DOMAINS[cleanKey];
    }

    // Strip suffixes
    const suffixes = ['_mcp', '_oauth2_mcp', '_oauth2', '_api', '_credentials', '_webhook', '_bot', '_services'];
    for (const suffix of suffixes) {
        const withoutSuffix = originalKey.replace(new RegExp(`_?${suffix}$`, 'i'), '');
        if (withoutSuffix && TOOL_DOMAINS[withoutSuffix]) {
            console.log(`  ✅ Suffix stripped (${suffix}): ${TOOL_DOMAINS[withoutSuffix]}`);
            return TOOL_DOMAINS[withoutSuffix];
        }
    }

    // First word
    const firstWord = originalKey.split('_')[0];
    if (firstWord && TOOL_DOMAINS[firstWord]) {
        console.log(`  ✅ First word match: ${TOOL_DOMAINS[firstWord]}`);
        return TOOL_DOMAINS[firstWord];
    }

    console.log(`  ❌ No match found`);
    return null;
}

// Test cases from user's screenshots
console.log('\n=== Testing Tool Names ===\n');
getDomainForTool('Discord Bot MCP', null);
getDomainForTool('Discord OAuth2 MCP', null);
getDomainForTool('Discord Webhook MCP', null);
getDomainForTool('AWS MCP', null);
getDomainForTool('AWS Services', null);
getDomainForTool('Azure Services', null);
getDomainForTool('GitHub MCP', null);
getDomainForTool('GitHub OAuth2 MCP', null);
getDomainForTool('GitHub Integration', null);
