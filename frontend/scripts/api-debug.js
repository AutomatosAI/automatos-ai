/**
 * API Debugging Utility
 * 
 * This script helps diagnose issues with API connectivity
 * between the frontend and backend.
 */

const http = require('http');
const https = require('https');
const url = require('url');
const logger = require('./logger');

// Configuration
const API_ENDPOINTS = [
  {
    name: 'Backend Health',
    url: 'http://localhost:8000/health',
    expected: { status: true }
  },
  {
    name: 'System Health',
    url: 'http://localhost:8000/api/system/health',
    expected: { status: 'ok' }
  },
  {
    name: 'Agents List',
    url: 'http://localhost:8000/api/agents/',
    expectedStatus: 200
  },
  {
    name: 'Documents List',
    url: 'http://localhost:8000/api/documents/',
    expectedStatus: 200
  }
];

// Function to make an HTTP request
function makeRequest(endpoint) {
  return new Promise((resolve, reject) => {
    logger.info(`Testing endpoint: ${endpoint.name} (${endpoint.url})`);
    
    const parsedUrl = url.parse(endpoint.url);
    const options = {
      hostname: parsedUrl.hostname,
      port: parsedUrl.port,
      path: parsedUrl.path,
      method: 'GET',
      timeout: 5000,
      headers: {
        'Accept': 'application/json',
        'x-api-key': process.env.API_KEY || ''
      }
    };

    const client = parsedUrl.protocol === 'https:' ? https : http;
    
    const req = client.request(options, (res) => {
      let data = '';
      
      // Log status code
      logger.info(`${endpoint.name} status: ${res.statusCode}`);
      
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        try {
          const jsonData = data ? JSON.parse(data) : {};
          resolve({
            name: endpoint.name,
            url: endpoint.url,
            statusCode: res.statusCode,
            data: jsonData,
            headers: res.headers,
            success: endpoint.expectedStatus ? 
              res.statusCode === endpoint.expectedStatus : 
              true
          });
        } catch (e) {
          logger.error(`Error parsing response: ${e.message}`);
          resolve({
            name: endpoint.name,
            url: endpoint.url,
            statusCode: res.statusCode,
            data: data,
            headers: res.headers,
            error: e.message,
            success: false
          });
        }
      });
    });
    
    req.on('error', (error) => {
      logger.error(`Error connecting to ${endpoint.url}: ${error.message}`);
      resolve({
        name: endpoint.name,
        url: endpoint.url,
        error: error.message,
        success: false
      });
    });
    
    req.on('timeout', () => {
      logger.error(`Timeout connecting to ${endpoint.url}`);
      req.abort();
      resolve({
        name: endpoint.name,
        url: endpoint.url,
        error: 'Request timeout',
        success: false
      });
    });
    
    req.end();
  });
}

// Function to check network connectivity
async function checkNetworkConnectivity() {
  logger.info('Checking network connectivity...');
  
  try {
    // Check if we can connect to the backend
    const results = await Promise.all(API_ENDPOINTS.map(makeRequest));
    
    // Print summary
    logger.info('=== API Connectivity Test Results ===');
    results.forEach(result => {
      if (result.success) {
        logger.info(`✅ ${result.name} - SUCCESS`);
      } else {
        logger.error(`❌ ${result.name} - FAILED: ${result.error || `Status: ${result.statusCode}`}`);
      }
      
      // Log detailed response
      logger.debug(`Details for ${result.name}:`, result);
    });
    
    // Overall status
    const allSuccessful = results.every(r => r.success);
    if (allSuccessful) {
      logger.info('✅ All API connectivity tests PASSED');
    } else {
      logger.error('❌ Some API connectivity tests FAILED');
    }
    
    // Check for specific issues
    const connectionErrors = results.filter(r => r.error && r.error.includes('ECONNREFUSED'));
    if (connectionErrors.length > 0) {
      logger.error('🔥 CONNECTION REFUSED errors detected. The backend server may be down or unreachable.');
      logger.info('🔍 Suggestion: Check if the backend is running:');
      logger.info('   - Run: `ps aux | grep uvicorn` to check process');
      logger.info('   - Check backend logs: `tail -f /root/automatos-ai/backend.log`');
      logger.info('   - Restart backend if needed');
    }
    
    return {
      success: allSuccessful,
      results
    };
  } catch (error) {
    logger.error('Error during connectivity test:', error);
    return {
      success: false,
      error: error.message
    };
  }
}

// Run the network connectivity check
checkNetworkConnectivity().then(result => {
  if (!result.success) {
    process.exit(1);
  }
});
