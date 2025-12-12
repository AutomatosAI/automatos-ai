// Quick test of logo URL generation
const getDomainForTool = (name, provider) => {
    const TOOL_DOMAINS = {
        'github': 'github.com',
        'aws': 'aws.amazon.com',
        'discord': 'discord.com',
        'azure': 'azure.microsoft.com'
    };

    const originalKey = (provider || name).toLowerCase().replace(/[\s-]/g, '_');

    // Direct match
    if (TOOL_DOMAINS[originalKey]) return TOOL_DOMAINS[originalKey];

    // Strip suffixes
    const suffixes = ['_mcp', '_oauth2_mcp', '_oauth2', '_services', '_bot', '_webhook'];
    for (const suffix of suffixes) {
        const withoutSuffix = originalKey.replace(new RegExp(`_?${suffix}$`, 'i'), '');
        if (withoutSuffix && TOOL_DOMAINS[withoutSuffix]) return TOOL_DOMAINS[withoutSuffix];
    }

    // First word
    const firstWord = originalKey.split('_')[0];
    if (firstWord && TOOL_DOMAINS[firstWord]) return TOOL_DOMAINS[firstWord];

    return null;
};

const getLogoUrl = (toolName) => {
    const domain = getDomainForTool(toolName);
    if (!domain) return '';
    const filename = domain.replace(/\./g, '_') + '.png';
    return `/logos/${filename}`;
};

// Test cases
console.log('GitHub Integration:', getLogoUrl('GitHub Integration'));
console.log('GitHub MCP:', getLogoUrl('GitHub MCP'));
console.log('AWS Services:', getLogoUrl('AWS Services'));
console.log('AWS MCP:', getLogoUrl('AWS MCP'));
console.log('Discord Bot MCP:', getLogoUrl('Discord Bot MCP'));
console.log('Azure Services:', getLogoUrl('Azure Services'));
