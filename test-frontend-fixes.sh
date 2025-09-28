#!/bin/bash

# Colors
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
RESET="\033[0m"

echo -e "🔍 Testing Frontend Fixes"
echo "---------------------------------"

# Test 1: Check for browser-compatible logger
echo -n "Testing browser-compatible logger... "
if grep -q "logToConsole" /root/automatos-ai/frontend/lib/logger.ts; then
  echo -e "PASS"
else
  echo -e "FAIL - Browser-compatible logger not found"
fi

# Test 2: Check for fixed document API hooks
echo -n "Testing fixed document API hooks... "
if grep -q "useDocuments()" /root/automatos-ai/frontend/hooks/use-document-api.ts &&    grep -q "useMutation" /root/automatos-ai/frontend/hooks/use-document-api.ts; then
  echo -e "PASS"
else
  echo -e "FAIL - Document API hooks not properly fixed"
fi

# Test 3: Check for proper environment variables
echo -n "Testing environment variables... "
if grep -q "NEXT_PUBLIC_WS_URL" /root/automatos-ai/frontend/.env.local &&    grep -q "LOG_LEVEL" /root/automatos-ai/frontend/.env.local; then
  echo -e "PASS"
else
  echo -e "FAIL - Missing required environment variables"
fi

# Test 4: Check for debug-enabled next.config.js
echo -n "Testing Next.js config for debugging... "
if grep -q "logging" /root/automatos-ai/frontend/next.config.js ||    grep -q "experimental" /root/automatos-ai/frontend/next.config.js; then
  echo -e "PASS"
else
  echo -e "WARNING - Next.js config may not have debugging enabled"
fi

# Summary
echo "---------------------------------"
echo -e "Frontend Fix Testing Complete"
echo ""
echo "To restart the frontend with debugging enabled:"
echo "  ./restart-frontend-with-debug.sh"
echo ""
echo "To monitor logs:"
echo "  tail -f /root/automatos-ai/frontend.log"
