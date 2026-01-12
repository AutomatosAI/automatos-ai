/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { 
    ignoreBuildErrors: true 
  },
  // Proxy API requests to backend (works like Nginx proxy)
  // This allows frontend to use relative URLs like /api/agents
  async rewrites() {
    // Use NEXT_PUBLIC_API_URL (set in Railway at build time)
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    
    // Remove trailing slash if present
    const cleanUrl = backendUrl.replace(/\/$/, '')
    
    console.log('🔀 Next.js rewrites configured:', {
      backendUrl: cleanUrl,
      source: '/api/:path*',
      destination: `${cleanUrl}/api/:path*`
    })
    
    return [
      {
        source: '/api/:path*',
        destination: `${cleanUrl}/api/:path*`,
      },
    ]
  }
}

module.exports = nextConfig
