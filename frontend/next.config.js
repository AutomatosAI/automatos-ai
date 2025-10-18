/** @type {import('next').NextConfig} */
const nextConfig = {
  typescript: { 
    ignoreBuildErrors: true 
  },
  async rewrites() {
    return [
      {
        source: '/chatbot-api/:path*',
        destination: '/api/chatbot/:path*',
      },
    ]
  }
}

module.exports = nextConfig
