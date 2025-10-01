// Global Mock System Test Script
// This tests mock controls across ALL pages in your application

console.log('🌍 Testing GLOBAL Mock System Across All Pages...')

// Test 1: Check current global status
console.log('1. Current global mock status:', window.automatos.mocks.getStatus())

// Test 2: Enable mocks globally (affects ALL pages)
console.log('2. Enabling mocks globally...')
window.automatos.mocks.enable()
console.log('   Status after enabling:', window.automatos.mocks.getStatus())

// Test 3: Test specific endpoints across different pages
const testEndpoints = [
  '/api/system/health',        // Dashboard page
  '/api/agents',              // Agents page  
  '/api/metrics/all',         // Analytics page
  '/api/tools',               // Tools page
  '/api/context/stats',       // Context page
  '/api/workflows/active',    // Workflows page
]

console.log('3. Testing individual endpoints:')
testEndpoints.forEach(endpoint => {
  window.automatos.mocks.toggle(endpoint)
  console.log(`   Toggled ${endpoint}`)
})

console.log('   Final status:', window.automatos.mocks.getStatus())

// Test 4: Disable mocks globally
console.log('4. Disabling mocks globally...')
window.automatos.mocks.disable()
console.log('   Status after disabling:', window.automatos.mocks.getStatus())

// Test 5: Toggle back on globally
console.log('5. Toggling mocks back on globally...')
window.automatos.mocks.toggle()
console.log('   Final global status:', window.automatos.mocks.getStatus())

console.log('✅ Global mock system test complete!')
console.log('💡 Now navigate to different pages and refresh to see mock data everywhere!')
console.log('📋 Pages to test: /dashboard, /agents, /analytics, /tools, /context, /workflows')

// Helper function to test a specific page
window.testPageMocks = function(pageName) {
  console.log(`🧪 Testing ${pageName} page with current mock settings...`)
  console.log('Current mock status:', window.automatos.mocks.getStatus())
  console.log('Navigate to the page and refresh to see the effect!')
}

console.log('💡 Use testPageMocks("Dashboard") to test specific pages')





