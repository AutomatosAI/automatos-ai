#!/bin/bash
# Generate OpenAPI JSON from running FastAPI server

API_URL="${API_URL:-http://localhost:8000}"  # Default to local, override with: API_URL=https://your-api-url.com ./generate_openapi.sh
OUTPUT_FILE="${OUTPUT_FILE:-openapi.json}"

echo "Fetching OpenAPI spec from $API_URL/openapi.json..."

curl -s "$API_URL/openapi.json" -o "$OUTPUT_FILE"

if [ -f "$OUTPUT_FILE" ] && [ -s "$OUTPUT_FILE" ]; then
    echo "✅ OpenAPI spec generated: $OUTPUT_FILE"
    
    # Display summary
    python3 << PYTHON
import json
with open('$OUTPUT_FILE') as f:
    spec = json.load(f)
    print(f"   Title: {spec['info']['title']}")
    print(f"   Version: {spec['info']['version']}")
    print(f"   Endpoints: {len(spec['paths'])}")
    print(f"   Tags: {len(spec.get('tags', []))}")
PYTHON
else
    echo "❌ Failed to fetch OpenAPI spec"
    echo "   Make sure the API server is running at: $API_URL"
    exit 1
fi
