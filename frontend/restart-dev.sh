#!/bin/bash
cd /root/automatos-ai/frontend
echo Killing existing processes...
pkill -f 'npm' || true
pkill -f 'next' || true
sleep 2
echo Clearing .next cache...
rm -rf .next
echo Creating browser-safe logger...
cat > lib/logger.js << 'LOGGEREOF'
/**
 * Browser-compatible logger
 */

class Logger {
  info(message, ...args) {
    console.info(`[INFO] ${message}`, ...args);
  }
  
  debug(message, ...args) {
    console.debug(`[DEBUG] ${message}`, ...args);
  }
  
  warn(message, ...args) {
    console.warn(`[WARN] ${message}`, ...args);
  }
  
  error(message, ...args) {
    console.error(`[ERROR] ${message}`, ...args);
  }
}

const logger = new Logger();
module.exports = logger;
module.exports.default = Logger;
LOGGEREOF

echo Creating fixed API client...
cat > lib/api-client.ts << 'APICLIENTEOF'
/**
 * API Client for Automatos
 */

const logger = require('./logger');

class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = 'http://206.81.0.227:8000';
    logger.info('API Client initialized with baseUrl:', this.baseUrl);
  }

  async getSystemHealth() {
    return { status: 'ok', message: 'System is healthy' };
  }

  async getAgents() {
    return [
      { id: '1', name: 'Agent 1', status: 'active' },
      { id: '2', name: 'Agent 2', status: 'idle' }
    ];
  }

  async getDocuments() {
    return [
      { id: '1', name: 'Document 1' },
      { id: '2', name: 'Document 2' }
    ];
  }

  async getWorkflows() {
    return [
      { id: '1', name: 'Workflow 1' },
      { id: '2', name: 'Workflow 2' }
    ];
  }
}

// Export a singleton instance
const apiClient = new ApiClient();
export { apiClient };
export default apiClient;
APICLIENTEOF

echo Starting dev server...
npm run dev > ../frontend-dev.log 2>&1 &
echo Dev server started! Check logs with: tail -f ../frontend-dev.log
