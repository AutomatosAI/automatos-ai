#!/bin/bash
# Apply agent context limit fixes

set -e

echo "🔧 Applying agent context limit fixes..."

# Check if docker compose is running
if ! docker compose ps | grep -q "orchestrator.*Up"; then
    echo "❌ Error: Orchestrator is not running. Please start with 'docker compose up -d'"
    exit 1
fi

# Apply the migration
echo "📝 Running migration SQL..."
docker compose exec -T db psql -U postgres -d automatos < 001_fix_agent_context_limits.sql

if [ $? -eq 0 ]; then
    echo "✅ Migration applied successfully!"
    echo ""
    echo "Summary of changes:"
    echo "  - Updated DEFAULT_LLM_CONFIG to gpt-4o (128K context)"
    echo "  - Technical Writer Pro: gpt-4 → gpt-4o"
    echo "  - Document Generation: gpt-4 → gpt-4o"
    echo "  - Implemented intelligent skill selection (max 2 skills per task)"
    echo ""
    echo "🎯 Restart the orchestrator for changes to take effect:"
    echo "   docker compose restart orchestrator"
else
    echo "❌ Migration failed. Check the error messages above."
    exit 1
fi

