#!/bin/bash

API_URL="http://206.81.0.227:8000"
API_KEY="test_api_key_for_backend_validation_2025"

echo "=== Testing Automatos API Endpoints ==="
echo ""

# Test each endpoint
endpoints=(
    "/api/agents/"
    "/api/agents/available"
    "/api/documents/"
    "/api/workflows"
    "/api/workflows/active"
    "/api/workflows/stats/dashboard"
    "/api/system/health"
    "/api/system/metrics"
    "/api/system/config"
)

for endpoint in "${endpoints[@]}"; do
    echo "Testing $endpoint..."
    response=$(curl -s -o /dev/null -w "%{http_code}" -H "X-API-Key: $API_KEY" "$API_URL$endpoint")
    
    if [ "$response" = "200" ]; then
        data=$(curl -s -H "X-API-Key: $API_KEY" "$API_URL$endpoint")
        count=$(echo "$data" | python3 -c "import sys, json; d=json.load(sys.stdin); print(len(d) if isinstance(d, list) else 'object')" 2>/dev/null || echo "parse error")
        echo "  ✅ Status: $response | Data: $count items/type"
    else
        echo "  ❌ Status: $response"
    fi
done
