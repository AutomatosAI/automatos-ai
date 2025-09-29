// Test script for mock data controls
// Run this in browser console to test mock functionality

console.log('🧪 Testing Mock Data Controls...')

// Test 1: Check initial status
console.log('1. Initial status:', window.automatos.mocks.getStatus())

// Test 2: Enable mocks
window.automatos.mocks.enable()
console.log('2. After enabling:', window.automatos.mocks.getStatus())

// Test 3: Test specific endpoint
window.automatos.mocks.toggle('/api/context/stats')
console.log('3. After toggling context stats:', window.automatos.mocks.getStatus())

// Test 4: Disable mocks
window.automatos.mocks.disable()
console.log('4. After disabling:', window.automatos.mocks.getStatus())

// Test 5: Toggle back on
window.automatos.mocks.toggle()
console.log('5. After toggle:', window.automatos.mocks.getStatus())

console.log('✅ Mock controls test complete!')
console.log('💡 Now refresh the page to see the effect')

