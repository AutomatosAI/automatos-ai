'use client';

import { useEffect, useState } from 'react';

export default function TestApiPage() {
  const [apiStatus, setApiStatus] = useState('Loading...');
  const [error, setError] = useState<string | null>(null);
  
  useEffect(() => {
    const testApiConnection = async () => {
      try {
        console.log('Testing API connection...');
        const response = await fetch('http://localhost:8000/health');
        const data = await response.json();
        console.log('API response:', data);
        setApiStatus(data.status === 'healthy' ? 'Connected ✅' : 'Unhealthy ❌');
      } catch (err) {
        console.error('API connection error:', err);
        setApiStatus('Connection Failed ❌');
        setError((err as Error).toString());
      }
    };
    
    testApiConnection();
  }, []);
  
  return (
    <div style={{ padding: '20px' }}>
      <h1>API Connection Test</h1>
      <p>Status: {apiStatus}</p>
      {error && (
        <div style={{ color: 'red', marginTop: '20px' }}>
          <h3>Error Details:</h3>
          <pre>{error}</pre>
        </div>
      )}
    </div>
  );
}
