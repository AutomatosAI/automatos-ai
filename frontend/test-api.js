// Simple API test
const testApi = async () => {
  try {
    console.log('Testing API...')
    const response = await fetch('https://api.automatos.app/api/system/health')
    const data = await response.json()
    console.log('API Response:', data)
  } catch (error) {
    console.error('API Error:', error)
  }
}

testApi()
