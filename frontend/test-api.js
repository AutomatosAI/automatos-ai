const fetch = require('node-fetch');

async function test() {
  console.log('Testing Automatos API...');
  
  // Test backend directly
  try {
    const res = await fetch('http://206.81.0.227:8000/api/agents/', {
      headers: {'X-API-Key': 'test_api_key_for_backend_validation_2025'}
    });
    const data = await res.json();
    console.log('✅ Backend API: Working -', data.length, 'agents found');
  } catch(e) {
    console.log('❌ Backend API: Failed -', e.message);
  }
  
  // Test frontend pages
  try {
    const res = await fetch('http://206.81.0.227:3000/agents');
    console.log('✅ Frontend /agents page: Status', res.status);
  } catch(e) {
    console.log('❌ Frontend /agents page: Failed -', e.message);
  }
}

test();
