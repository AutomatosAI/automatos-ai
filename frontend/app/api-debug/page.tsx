'use client';

import { useState, useEffect } from 'react';

export default function ApiDebugPage() {
  const [apiUrl, setApiUrl] = useState('');
  const [testResults, setTestResults] = useState([]);
  
  useEffect(() => {
    // Get the current API URL from environment
    setApiUrl(process.env.NEXT_PUBLIC_API_URL || 'Not set');
  }, []);
  
  const runTests = async () => {
    try {
      // Test direct health check
      const healthResponse = await fetch();
      const healthData = await healthResponse.json();
      
      alert();
    } catch (error) {
      console.error('Error testing API:', error);
      alert();
    }
  };
  
  return (
    <div style={{ padding: '20px' }}>
      <h1 style={{ marginBottom: '20px' }}>API Debug Tool</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <h2>Current Configuration</h2>
        <div style={{ marginBottom: '10px' }}>
          <label>API URL: {apiUrl}</label>
        </div>
        
        <button
          onClick={runTests}
          style={{ 
            padding: '8px 16px',
            backgroundColor: 'blue',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer'
          }}
        >
          Test API Connection
        </button>
      </div>
      
      <div>
        <h2>Backend URLs to Test Directly:</h2>
        <ul>
          <li>Health check: {apiUrl}/health</li>
          <li>System health: {apiUrl}/api/system/health</li>
        </ul>
      </div>
    </div>
  );
}
